"""The siting entry point: one field per fixed terminal, candidates by lookup.

`cost_fields` is the natural shape of the problem -- the turbines and grid
connection points stay put, the substation moves -- so these tests pin that
it gives the same answers as doing it by hand, in both the in-memory and the
spilled-to-disk modes, and that the two modes agree with each other.
"""
import numpy as np
import pytest
import rasterio

import pyorps


CELL = 2.0
SHAPE = (70, 90)


@pytest.fixture
def site(tmp_path):
    rng = np.random.default_rng(17)
    data = rng.integers(90, 450, size=SHAPE).astype(np.uint16)
    data[25:55, 40:43] = 65535          # a barrier with a gap at the bottom
    data[50:55, 40:43] = 180
    path = tmp_path / "site.tif"
    tr = rasterio.transform.from_origin(400000.0, 5600000.0, CELL, CELL)
    with rasterio.open(
        path, "w", driver="GTiff", height=SHAPE[0], width=SHAPE[1], count=1,
        dtype="uint16", crs="EPSG:25832", transform=tr,
    ) as dst:
        dst.write(data, 1)
    with rasterio.open(path) as src:
        tr = src.transform
    terminals = [tuple(map(float, rasterio.transform.xy(tr, r, c)))
                 for r, c in [(6, 6), (10, 80), (60, 8), (62, 78)]]
    labels = ["WT0", "WT1", "PCC0", "PCC1"]
    rng2 = np.random.default_rng(9)
    rows, cols = rng2.integers(0, SHAPE[0], 120), rng2.integers(0, SHAPE[1], 120)
    xs, ys = rasterio.transform.xy(tr, rows.tolist(), cols.tolist())
    candidates = np.column_stack([np.asarray(xs, float), np.asarray(ys, float)])
    return path, terminals, labels, candidates


def _finder(path, terminals):
    return pyorps.PathFinder(
        dataset_source=str(path),
        source_coords=list(terminals), target_coords=list(terminals),
        search_space_buffer_m=pyorps.full_window_buffer_m(str(path)),
        neighborhood_str="r2", graph_api="cython", ignore_max_cost=True,
    )


class TestMatchesDoingItByHand:
    def test_costs_match_individual_cost_fields(self, site):
        path, terminals, labels, candidates = site
        finder = _finder(path, terminals)
        with finder.cost_fields(terminals, labels=labels) as fields:
            got = fields.costs_to(candidates)
        assert got.shape == (len(terminals), len(candidates))

        finder2 = _finder(path, terminals)
        for i, origin in enumerate(terminals):
            with finder2.cost_field(origin, algorithm="auto") as one:
                np.testing.assert_allclose(got[i], one.costs_to(candidates),
                                           rtol=1e-6)

    def test_default_algorithm_is_delta_stepping(self, site):
        path, terminals, labels, _ = site
        finder = _finder(path, terminals)
        with finder.cost_fields(terminals, labels=labels) as fields:
            assert fields.algorithm == "delta-stepping"

    def test_cost_field_now_defaults_to_auto_too(self, site):
        path, terminals, _, _ = site
        finder = _finder(path, terminals)
        with finder.cost_field(terminals[0]) as field:
            assert field.algorithm == "delta-stepping"


class TestSpilling:
    """Held live when they fit, paged off disk when they do not."""

    def test_spilled_and_live_agree_exactly(self, site):
        path, terminals, labels, candidates = site
        finder = _finder(path, terminals)
        with finder.cost_fields(terminals, labels=labels,
                                spill=False) as live:
            assert live.spilled is False
            a = live.costs_to(candidates)
            leg_a = live.path_to("WT1", tuple(candidates[3]))

        finder2 = _finder(path, terminals)
        with finder2.cost_fields(terminals, labels=labels,
                                 spill=True) as spilled:
            assert spilled.spilled is True
            b = spilled.costs_to(candidates)
            leg_b = spilled.path_to("WT1", tuple(candidates[3]))

        np.testing.assert_allclose(b, a, rtol=1e-6)
        assert leg_b.length_m == pytest.approx(leg_a.length_m, rel=1e-9)
        assert leg_b.cost == pytest.approx(leg_a.cost, rel=1e-5)

    def test_a_spilled_set_holds_nothing_in_memory(self, site):
        path, terminals, labels, _ = site
        finder = _finder(path, terminals)
        with finder.cost_fields(terminals, labels=labels,
                                spill=True) as fields:
            assert fields.memory_bytes == 0

    def test_the_temporary_cache_is_cleaned_up(self, site):
        path, terminals, labels, _ = site
        finder = _finder(path, terminals)
        fields = finder.cost_fields(terminals, labels=labels, spill=True)
        cache = fields._cache
        assert any(cache.iterdir())
        fields.close()
        assert not cache.exists()

    def test_an_explicit_cache_dir_is_kept(self, site, tmp_path):
        path, terminals, labels, _ = site
        keep = tmp_path / "keep"
        finder = _finder(path, terminals)
        with finder.cost_fields(terminals, labels=labels, spill=True,
                                cache_dir=keep) as fields:
            pass
        assert sorted(p.name for p in keep.glob("*.npz")) == [
            f"field_{lab}.npz" for lab in sorted(labels)]


class TestQueries:
    def test_nearest_picks_the_cheapest_terminal(self, site):
        path, terminals, labels, candidates = site
        finder = _finder(path, terminals)
        with finder.cost_fields(terminals, labels=labels) as fields:
            costs = fields.costs_to(candidates)
            value, which = fields.nearest(candidates)
        reachable = np.isfinite(value)
        assert reachable.any()
        np.testing.assert_allclose(value[reachable],
                                   costs.min(axis=0)[reachable])
        assert set(np.unique(which[reachable])) <= set(range(len(labels)))

    def test_nearest_over_a_subset_of_terminals(self, site):
        path, terminals, labels, candidates = site
        finder = _finder(path, terminals)
        with finder.cost_fields(terminals, labels=labels) as fields:
            value, which = fields.nearest(candidates, labels=["PCC0", "PCC1"])
            direct = fields.costs_to(candidates, labels=["PCC0", "PCC1"])
        ok = np.isfinite(value)
        np.testing.assert_allclose(value[ok], direct.min(axis=0)[ok])
        assert set(np.unique(which[ok])) <= {0, 1}

    def test_path_to_agrees_with_the_lookup(self, site):
        path, terminals, labels, candidates = site
        finder = _finder(path, terminals)
        with finder.cost_fields(terminals, labels=labels) as fields:
            for lab in labels:
                target = tuple(candidates[7])
                if not np.isfinite(fields.cost_to(lab, target)):
                    continue
                leg = fields.path_to(lab, target)
                assert leg.coords.shape[0] > 1
                assert leg.length_m > 0
                assert leg.cost == pytest.approx(
                    fields.cost_to(lab, target), rel=1e-5)

    def test_unknown_label_is_rejected(self, site):
        path, terminals, labels, _ = site
        finder = _finder(path, terminals)
        with finder.cost_fields(terminals, labels=labels) as fields:
            with pytest.raises(KeyError, match="no terminal"):
                fields.cost_to("WT9", terminals[0])

    def test_duplicate_labels_are_rejected(self, site):
        path, terminals, _, _ = site
        finder = _finder(path, terminals)
        with pytest.raises(ValueError, match="unique"):
            finder.cost_fields(terminals, labels=["a", "a", "b", "c"])


class TestLifetime:
    def test_closed_set_refuses_queries(self, site):
        path, terminals, labels, candidates = site
        finder = _finder(path, terminals)
        fields = finder.cost_fields(terminals, labels=labels)
        fields.close()
        with pytest.raises(RuntimeError, match="closed"):
            fields.costs_to(candidates)

    def test_release_device_resources_closes_the_set(self, site):
        path, terminals, labels, _ = site
        finder = _finder(path, terminals)
        fields = finder.cost_fields(terminals, labels=labels)
        finder.release_device_resources()
        assert fields._closed

    def test_exported_at_top_level(self):
        assert "CostFieldSet" in pyorps.__all__
        assert "Leg" in pyorps.__all__
        assert pyorps.Leg._fields == ("cells", "coords", "length_m", "cost")
