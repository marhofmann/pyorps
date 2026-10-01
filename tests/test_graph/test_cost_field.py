"""CostField: rooted-at-a-fixed-terminal least-cost field.

Golden answers come from ``PathFinder.find_route`` (source=candidate,
target=origin) -- the raster graph is undirected with symmetric edge
weights, so a field rooted at the fixed terminal answers every query in
the OTHER direction too. That symmetry is the thing under test.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from rasterio.transform import from_origin

from pyorps.core.types import IMPASSABLE_CELL_COST
from pyorps.graph.path_finder import PathFinder
from pyorps.graph.search_session import _delta_dummy_target
from pyorps.utils.neighborhood import get_neighborhood_steps

try:
    import cupy as cp
    try:
        cp.cuda.runtime.getDeviceCount()
        GPU = True
    except Exception:
        GPU = False
except ImportError:
    GPU = False


def _make_finder(raster, src, tgt, *, graph_api="cython", buffer_m=500,
                 neighborhood="r2", **kw):
    rows = raster.shape[0]
    return PathFinder(
        dataset_source=raster,
        crs="EPSG:32632",
        transform=from_origin(0.0, float(rows), 1.0, 1.0),
        source_coords=src,
        target_coords=tgt,
        search_space_buffer_m=buffer_m,
        graph_api=graph_api,
        neighborhood_str=neighborhood,
        **kw,
    )


def _xy(col, row, rows):
    """Pixel-centre coordinates for ``from_origin(0, rows, 1, 1)``."""
    return (col + 0.5, rows - row - 0.5)


def _uniform(n=60):
    return np.ones((n, n), dtype=np.uint16)


def _walled(n=80):
    raster = np.ones((n, n), dtype=np.uint16)
    raster[10:70, 40] = IMPASSABLE_CELL_COST
    raster[40, 40] = 1  # gap
    return raster


def _solid_wall(n=80):
    """Full-height wall, no gap and no bypass around the ends."""
    raster = np.ones((n, n), dtype=np.uint16)
    raster[:, 40] = IMPASSABLE_CELL_COST
    return raster


def _unique(n=60):
    """Strictly patterned costs so competing shortest paths are rare."""
    rows = np.arange(n, dtype=np.int32)[:, None]
    cols = np.arange(n, dtype=np.int32)[None, :]
    return (1 + (rows * 31 + cols * 17) % 250).astype(np.uint16)


def _rand(n=60, seed=7):
    return np.random.default_rng(seed).integers(1, 250, (n, n)).astype(np.uint16)


def _corner_blocked(n=80):
    """Window corner impassable -- the shape that silently emptied the
    delta workspace (_delta_stepping.pyx:1963-1965 returns before
    take_dist_pred at :1982)."""
    raster = np.ones((n, n), dtype=np.uint16)
    raster[0, 0] = IMPASSABLE_CELL_COST
    raster[0, 1] = IMPASSABLE_CELL_COST
    return raster


BACKENDS = [("cython", "dijkstra", {}),
            ("cython", "delta-stepping", {"delta": 5.0})]


# ---------------------------------------------------------------------------
# Core equivalence / symmetry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("api,algo,kw", BACKENDS,
                         ids=[b[1] for b in BACKENDS])
class TestCostFieldCore:
    def test_cost_to_matches_find_route_total_cost(self, api, algo, kw):
        raster = _rand()
        origin = _xy(5, 30, 60)
        site = _xy(45, 12, 60)
        finder = _make_finder(raster, origin, site, graph_api=api)
        golden = finder.find_route(source=site, target=origin, algorithm=algo,
                                   calculate_metrics=True, **kw)
        with finder.cost_field(origin, algorithm=algo, **kw) as field:
            cost = field.cost_to(site)
        assert cost == pytest.approx(golden.total_cost, rel=1e-6)

    def test_field_is_direction_free(self, api, algo, kw):
        raster = _rand(seed=11)
        origin = _xy(50, 50, 60)
        rng = np.random.default_rng(4)
        cands = [_xy(int(c), int(r), 60) for c, r in
                 zip(rng.integers(2, 58, 10), rng.integers(2, 58, 10))]
        finder = _make_finder(raster, origin, cands[0], graph_api=api)
        golden = [finder.find_route(source=c, target=origin, algorithm=algo,
                                    calculate_metrics=True, **kw).total_cost
                 for c in cands]
        with finder.cost_field(origin, algorithm=algo, **kw) as field:
            costs = field.costs_to(cands)
        for got, want in zip(costs, golden):
            assert got == pytest.approx(want, rel=1e-6)

    def test_unreachable_returns_inf(self, api, algo, kw):
        raster = _solid_wall(80)
        origin = _xy(5, 40, 80)
        far_side = _xy(70, 40, 80)
        near_side = _xy(20, 40, 80)
        finder = _make_finder(raster, origin, far_side, graph_api=api,
                              ignore_max_cost=True)
        with finder.cost_field(origin, algorithm=algo, **kw) as field:
            costs = field.costs_to([far_side, near_side])
        assert math.isinf(costs[0])
        assert math.isfinite(costs[1])

    def test_point_outside_window_raises(self, api, algo, kw):
        raster = _uniform(60)
        origin = _xy(5, 30, 60)
        tgt = _xy(20, 30, 60)
        finder = _make_finder(raster, origin, tgt, graph_api=api, buffer_m=15)
        with finder.cost_field(origin, algorithm=algo, **kw) as field:
            with pytest.raises(ValueError):
                field.costs_to([_xy(58, 58, 60)])


# ---------------------------------------------------------------------------
# Settle / gather bookkeeping (dijkstra-specific: only backend with partial
# settle)
# ---------------------------------------------------------------------------

class TestSettleOnDemand:
    """Dijkstra's early stop. Named explicitly because it is the ONE thing
    the default ``algorithm="auto"`` does not do: the parallel kernels are
    full-field by construction, so there is nothing to settle on demand."""

    def test_settle_on_demand_never_exceeds_one_full_search(self):
        raster = _rand(n=70, seed=2)
        origin = _xy(5, 35, 70)
        rng = np.random.default_rng(9)
        clustered = [_xy(int(c), int(r), 70) for c, r in
                    zip(rng.integers(2, 15, 15), rng.integers(2, 15, 15))]

        finder_a = _make_finder(raster, origin, clustered[0])
        with finder_a.cost_field(origin, algorithm="dijkstra") as field_a:
            counts = []
            for pt in clustered:
                field_a.cost_to(pt)
                counts.append(field_a.expansion_count)

        finder_b = _make_finder(raster, origin, clustered[0])
        with finder_b.cost_field(origin, algorithm="dijkstra") as field_b:
            field_b.settle_all()

        assert counts == sorted(counts)
        assert counts[-1] <= field_b.expansion_count
        assert counts[-1] < field_b.expansion_count

    def test_settle_all_then_query_matches_incremental(self):
        raster = _rand(n=50, seed=3)
        origin = _xy(5, 25, 50)
        pts = [_xy(c, r, 50) for c, r in [(10, 10), (30, 40), (45, 5)]]
        finder = _make_finder(raster, origin, pts[0])
        with finder.cost_field(origin) as field:
            each = field.costs_to(pts, settle="each")
        finder2 = _make_finder(raster, origin, pts[0])
        with finder2.cost_field(origin) as field2:
            allc = field2.costs_to(pts, settle="all")
        np.testing.assert_allclose(each, allc, rtol=1e-9)

    def test_bulk_query_is_vectorised(self):
        raster = _rand(n=60, seed=5)
        origin = _xy(5, 30, 60)
        rng = np.random.default_rng(1)
        pts = [_xy(int(c), int(r), 60) for c, r in
              zip(rng.integers(0, 60, 30), rng.integers(0, 60, 30))]
        finder = _make_finder(raster, origin, pts[0])
        with finder.cost_field(origin) as field:
            bulk = field.costs_to(pts)
        finder2 = _make_finder(raster, origin, pts[0])
        with finder2.cost_field(origin) as field2:
            singles = [field2.cost_to(p) for p in pts]
        np.testing.assert_allclose(bulk, singles, rtol=1e-9)


class TestPathToAndFieldArray:
    def test_path_to_matches_costs_to(self):
        raster = _rand(n=50, seed=6)
        origin = _xy(5, 25, 50)
        tgt = _xy(40, 10, 50)
        finder = _make_finder(raster, origin, tgt)
        with finder.cost_field(origin) as field:
            cost = field.costs_to([tgt])[0]
            path = field.path_to(tgt, calculate_metrics=True)
            reverse_path = field.path_to(tgt, reverse=True)
        assert path.total_cost == pytest.approx(cost, rel=1e-6)
        assert np.array_equal(np.asarray(reverse_path.path_indices),
                              np.asarray(path.path_indices)[::-1])

    def test_field_array_shape_and_units(self):
        raster = _rand(n=40, seed=8)
        origin = _xy(5, 20, 40)
        tgt = _xy(25, 8, 40)
        finder = _make_finder(raster, origin, tgt, buffer_m=500)
        with finder.cost_field(origin) as field:
            cost = field.costs_to([tgt])[0]
            arr = field.field_array()
            rows, cols = finder.raster_handler.data.shape[-2:]
            assert arr.shape == (rows, cols)
            origin_row, origin_col = finder.raster_handler.coords_to_indices(
                [origin])[0]
            assert arr[origin_row, origin_col] == pytest.approx(0.0, abs=1e-3)
            tgt_row, tgt_col = finder.raster_handler.coords_to_indices(
                [tgt])[0]
            assert arr[tgt_row, tgt_col] == pytest.approx(cost, rel=1e-3)


class TestLifetime:
    def test_close_frees_and_deregisters(self):
        raster = _uniform(30)
        origin = _xy(5, 15, 30)
        tgt = _xy(20, 15, 30)
        finder = _make_finder(raster, origin, tgt)
        field = finder.cost_field(origin)
        assert field.memory_bytes > 0
        field.close()
        assert field.memory_bytes == 0
        assert field not in finder._search_sessions
        field.close()  # no-op
        with pytest.raises(RuntimeError):
            field.cost_to(tgt)

    def test_context_manager_closes(self):
        raster = _uniform(30)
        origin = _xy(5, 15, 30)
        tgt = _xy(20, 15, 30)
        finder = _make_finder(raster, origin, tgt)
        with finder.cost_field(origin) as field:
            pass
        assert field.memory_bytes == 0

    def test_release_device_resources_closes_field(self):
        raster = _uniform(30)
        origin = _xy(5, 15, 30)
        tgt = _xy(20, 15, 30)
        finder = _make_finder(raster, origin, tgt)
        field = finder.cost_field(origin)
        finder.release_device_resources()
        assert field.memory_bytes == 0

    def test_neighborhood_change_invalidates(self):
        raster = _uniform(30)
        origin = _xy(5, 15, 30)
        tgt = _xy(20, 15, 30)
        finder = _make_finder(raster, origin, tgt, neighborhood="r1")
        with finder.cost_field(origin) as field:
            finder.neighborhood_str = "r2"
            with pytest.raises(ValueError, match="invalidat"):
                field.costs_to([tgt])

    def test_reopen_after_close(self):
        raster = _uniform(30)
        origin = _xy(5, 15, 30)
        tgt = _xy(20, 15, 30)
        finder = _make_finder(raster, origin, tgt)
        field1 = finder.cost_field(origin)
        cost1 = field1.cost_to(tgt)
        field1.close()
        field2 = finder.cost_field(origin)
        cost2 = field2.cost_to(tgt)
        field2.close()
        assert cost1 == pytest.approx(cost2, rel=1e-9)


class TestPreconditionGuards:
    def test_half_step_set_is_rejected(self):
        raster = _uniform(30)
        origin = _xy(5, 15, 30)
        tgt = _xy(20, 15, 30)
        finder = _make_finder(raster, origin, tgt)
        finder.steps = get_neighborhood_steps("r1", directed=False)
        with pytest.raises(ValueError, match="closed under negation"):
            finder.cost_field(origin)

    def test_asymmetric_backend_is_rejected(self, monkeypatch):
        raster = _uniform(30)
        origin = _xy(5, 15, 30)
        tgt = _xy(20, 15, 30)
        finder = _make_finder(raster, origin, tgt)
        finder.create_raster_handler()
        api = finder.graph_api
        monkeypatch.setattr(type(api), "symmetric_edge_weights", False,
                            raising=False)
        with pytest.raises(ValueError):
            finder.cost_field(origin)


# ---------------------------------------------------------------------------
# Delta-stepping corner-cell regression (see plan §2.9 / §2.10)
# ---------------------------------------------------------------------------

class TestDeltaCornerCellRegression:
    def test_search_session_route_survives_blocked_corner(self):
        raster = _corner_blocked(80)
        src, tgt = _xy(5, 40, 80), _xy(70, 40, 80)
        finder = _make_finder(raster, src, tgt, buffer_m=500)
        session = finder.search_session(algorithm="delta-stepping", delta=5.0)
        path = session.route([src, tgt])
        golden = finder.find_route(source=src, target=tgt,
                                   algorithm="delta-stepping",
                                   calculate_metrics=False, delta=5.0)
        assert [int(i) for i in path.path_indices] == \
            [int(i) for i in golden.path_indices]

    def test_cost_field_survives_blocked_corner(self):
        raster = _corner_blocked(80)
        origin, tgt = _xy(5, 40, 80), _xy(70, 40, 80)
        finder = _make_finder(raster, origin, tgt, buffer_m=500)
        with finder.cost_field(origin, algorithm="delta-stepping",
                               delta=5.0) as field:
            cost = field.cost_to(tgt)
        golden = finder.find_route(source=tgt, target=origin,
                                   algorithm="delta-stepping",
                                   calculate_metrics=True, delta=5.0)
        assert cost == pytest.approx(golden.total_cost, rel=1e-6)

    def test_zero_cost_dummy_does_not_truncate_field(self):
        n = 60
        raster = np.zeros((n, n), dtype=np.uint16)
        raster[:, 55:] = 3  # one positive band, far from origin
        origin = _xy(2, 2, n)
        farthest = _xy(58, 58, n)
        finder = _make_finder(raster, origin, farthest, buffer_m=500)
        with finder.cost_field(origin, algorithm="delta-stepping",
                               delta=5.0) as field:
            cost = field.cost_to(farthest)
        assert math.isfinite(cost)

    def test_delta_dummy_target_helper(self):
        raster = _corner_blocked(80)
        flat = raster.reshape(-1)
        dummy = _delta_dummy_target(flat, root_idx=0, max_value=IMPASSABLE_CELL_COST)
        assert flat[dummy] != IMPASSABLE_CELL_COST
        assert flat[dummy] > 0
        assert dummy != 0

        allzero = np.zeros(100, dtype=np.uint16)
        dummy2 = _delta_dummy_target(allzero, root_idx=0,
                                     max_value=IMPASSABLE_CELL_COST)
        assert dummy2 != 0

    def test_delta_unreachable_label_is_inf(self):
        raster = _solid_wall(80)
        origin = _xy(5, 40, 80)
        far_side = _xy(70, 40, 80)
        finder = _make_finder(raster, origin, far_side, buffer_m=500,
                              ignore_max_cost=True)
        with finder.cost_field(origin, algorithm="delta-stepping",
                               delta=5.0) as field:
            cost = field.cost_to(far_side)
        assert math.isinf(cost)


# ---------------------------------------------------------------------------
# GPU / FIM (skipped without CUDA)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not GPU, reason="CUDA GPU not available")
class TestGpuCostField:
    def test_cost_to_matches_find_route(self):
        raster = _unique(60)
        origin = _xy(5, 30, 60)
        site = _xy(45, 12, 60)
        finder = _make_finder(raster, origin, site, graph_api="raster_gpu")
        golden = finder.find_route(source=site, target=origin,
                                   calculate_metrics=True)
        with finder.cost_field(origin, algorithm="delta-stepping") as field:
            cost = field.cost_to(site)
        assert cost == pytest.approx(golden.total_cost, rel=1e-5)


@pytest.mark.skipif(not GPU, reason="CUDA GPU not available")
class TestFimCostField:
    def test_self_symmetry(self):
        # MEASURED (2026-08-19): the FIM/eikonal field value itself -- what
        # CostField reads via _FimTree.peek_dist -- is only symmetric to
        # ~1-4 % on this backend, not the 1e-9-scale symmetry the exact
        # Dijkstra/delta kernels give. That is an inherent property of the
        # iterative eikonal solver (worse on the highly discontinuous
        # `_unique` raster, better on a smoother one), reproduced identically
        # with plain find_route() in both directions -- i.e. it is NOT a
        # CostField/reverse-tree defect. Never compare against
        # Path.total_cost either: that is a discrete retrace of the
        # continuous field and diverges further still (see
        # PathFinder._total_cost_basis).
        rng = np.random.default_rng(7)
        raster = rng.integers(50, 150, (60, 60)).astype(np.uint16)
        a = _xy(5, 30, 60)
        b = _xy(45, 12, 60)
        finder_ab = _make_finder(raster, a, b, graph_api="raster_fim")
        with finder_ab.cost_field(a, algorithm="fim") as field:
            cost_ab = field.cost_to(b)
        finder_ba = _make_finder(raster, b, a, graph_api="raster_fim")
        with finder_ba.cost_field(b, algorithm="fim") as field2:
            cost_ba = field2.cost_to(a)
        assert cost_ab == pytest.approx(cost_ba, rel=0.05)
