"""The inversion a siting run rests on: root at what is FIXED, look up what MOVES.

Pricing many candidate substation positions against fixed turbines is done by
turning the search around. Instead of one search per candidate, one search per
turbine settles the cost to every cell, and each candidate is then a lookup.
That is only legitimate because the raster graph is undirected, so

    d(turbine -> candidate) == d(candidate -> turbine)

and the field rooted at the turbine answers a question posed in the other
direction. These tests pin that equality, and pin that the lookup really does
replace a route computation rather than approximate one.

Measured the same way against the real CIRED case study (53.3 M-cell window,
r2, 7 turbine fields): agreement with the independently computed pairwise
routes 5.5e-09 relative, and symmetry exactly 0.0.
"""
import itertools

import numpy as np
import pytest
import rasterio

import pyorps


CELL = 2.0
SHAPE = (60, 80)


@pytest.fixture
def terrain(tmp_path):
    """A raster with structure, so routes bend rather than run straight."""
    rng = np.random.default_rng(5)
    data = rng.integers(90, 500, size=SHAPE).astype(np.uint16)
    data[15:45, 30:33] = 65535            # a barrier with a gap at the top
    data[15:20, 30:33] = 150
    path = tmp_path / "terrain.tif"
    transform = rasterio.transform.from_origin(400000.0, 5600000.0, CELL, CELL)
    with rasterio.open(
        path, "w", driver="GTiff", height=SHAPE[0], width=SHAPE[1], count=1,
        dtype="uint16", crs="EPSG:25832", transform=transform,
    ) as dst:
        dst.write(data, 1)
    with rasterio.open(path) as src:
        tr = src.transform
    pts = [tuple(map(float, rasterio.transform.xy(tr, r, c)))
           for r, c in [(5, 5), (8, 70), (50, 10), (52, 66)]]
    return path, pts


def _finder(path, pts):
    """One finder carrying every terminal, so all fields share a window."""
    return pyorps.PathFinder(
        dataset_source=str(path),
        source_coords=list(pts), target_coords=list(pts),
        search_space_buffer_m=pyorps.full_window_buffer_m(str(path)),
        neighborhood_str="r2", graph_api="cython", ignore_max_cost=True,
    )


class TestSymmetry:
    """d(i->j) read from i's field must equal d(j->i) read from j's."""

    def test_exact_under_dijkstra(self, terrain):
        path, pts = terrain
        finder = _finder(path, pts)
        fields = [finder.cost_field(p, algorithm="dijkstra") for p in pts]
        try:
            for (i, a), (j, b) in itertools.combinations(enumerate(pts), 2):
                ab = fields[i].cost_to(b)
                ba = fields[j].cost_to(a)
                assert np.isfinite(ab)
                # float64 labels on an undirected graph: not "close", equal.
                assert ab == ba, f"{a}->{b} {ab} vs {ba}"
        finally:
            for f in fields:
                f.close()

    def test_within_float32_under_delta_stepping(self, terrain):
        """The parallel kernel keeps float32 labels, so equality is to eps."""
        path, pts = terrain
        finder = _finder(path, pts)
        fields = [finder.cost_field(p, algorithm="delta-stepping")
                  for p in pts]
        try:
            for (i, a), (j, b) in itertools.combinations(enumerate(pts), 2):
                ab, ba = fields[i].cost_to(b), fields[j].cost_to(a)
                assert ab == pytest.approx(ba, rel=1e-4)
        finally:
            for f in fields:
                f.close()


class TestLookupReplacesRouting:
    """A field lookup must equal the route computed the ordinary way."""

    def test_cost_to_matches_find_route(self, terrain):
        path, pts = terrain
        origin, others = pts[0], pts[1:]
        finder = _finder(path, pts)
        with finder.cost_field(origin, algorithm="dijkstra") as field:
            looked_up = field.costs_to(others)
        routed = []
        for target in others:
            f = _finder(path, [origin, target])
            routed.append(float(f.find_route(origin, target).total_cost))
            f.release_device_resources()
        # The two are the same quantity reached by different summation
        # orders -- the field accumulates cell labels and scales once, the
        # metric kernel sums priced steps -- so they agree to float64
        # accumulation noise and not to the last bit. 1e-7 of a 30 kEUR
        # route is a third of a cent.
        np.testing.assert_allclose(looked_up, routed, rtol=1e-7)

    def test_path_to_reprices_to_the_lookup(self, terrain):
        """The route, not just the number, is the optimal one."""
        path, pts = terrain
        origin, others = pts[0], pts[1:]
        finder = _finder(path, pts)
        with finder.cost_field(origin, algorithm="dijkstra") as field:
            for target in others:
                lookup = field.cost_to(target)
                route = field.path_to(target, calculate_metrics=True)
                assert route.total_cost == pytest.approx(lookup, rel=1e-7)

    def test_one_field_answers_every_candidate(self, terrain):
        """The whole point: N candidates cost ONE search, not N."""
        path, pts = terrain
        origin = pts[0]
        finder = _finder(path, pts)
        with rasterio.open(path) as src:
            tr, h, w = src.transform, src.height, src.width
        rng = np.random.default_rng(2)
        rows, cols = rng.integers(0, h, 300), rng.integers(0, w, 300)
        xs, ys = rasterio.transform.xy(tr, rows.tolist(), cols.tolist())
        candidates = list(zip(map(float, xs), map(float, ys)))

        with finder.cost_field(origin, algorithm="dijkstra") as field:
            costs = field.costs_to(candidates)
            after = field.expansion_count
            more = field.costs_to(candidates)          # asking again is free
            assert field.expansion_count == after
        assert costs.shape == (300,)
        assert np.isfinite(costs).sum() > 200
        np.testing.assert_array_equal(costs, more)
        # One full settle bounds the work, however many candidates are priced.
        assert after <= SHAPE[0] * SHAPE[1]
