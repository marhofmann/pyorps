"""SearchSession: exact reuse of unconstrained search fields on edits.

Golden answers come from a fresh ``PathFinder.find_route`` chain on the
same points — reuse is only a speedup (plan acceptance criteria).
"""
from __future__ import annotations

import numpy as np
import pytest
from rasterio.transform import from_origin

from pyorps.core.exceptions import NoPathFoundError
from pyorps.core.types import IMPASSABLE_CELL_COST
from pyorps.graph.path_finder import PathFinder

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


def _indices(path):
    return [int(i) for i in path.path_indices]


def _golden_chain(finder, points, algorithm, **kwargs):
    all_indices = []
    for start, end in zip(points[:-1], points[1:]):
        path = finder.find_route(
            source=start, target=end, algorithm=algorithm,
            calculate_metrics=False, **kwargs)
        idxs = _indices(path)
        if all_indices and idxs and all_indices[-1] == idxs[0]:
            idxs = idxs[1:]
        all_indices.extend(idxs)
    return all_indices


def _uniform(n=80):
    return np.ones((n, n), dtype=np.uint16)


def _corridor_barrier(n=80):
    """Cheap corridor around a high-cost blob; first target's disk may
    miss a point on the far side of the blob."""
    raster = np.full((n, n), 8, dtype=np.uint16)
    raster[20:60, 35:45] = 200
    raster[0:15, :] = 2
    return raster


def _walled(n=80):
    raster = np.ones((n, n), dtype=np.uint16)
    raster[10:70, 40] = IMPASSABLE_CELL_COST
    raster[40, 40] = 1  # gap
    return raster


# ---------------------------------------------------------------------------
# Dijkstra
# ---------------------------------------------------------------------------

class TestDijkstraSession:
    def test_factory_returns_session(self):
        raster = _uniform(40)
        src, tgt = _xy(5, 20, 40), _xy(30, 20, 40)
        finder = _make_finder(raster, src, tgt)
        session = finder.search_session(algorithm="dijkstra")
        assert session is not None
        session.close()

    def test_route_matches_find_route(self):
        raster = _corridor_barrier(80)
        src, tgt = _xy(5, 40, 80), _xy(70, 40, 80)
        finder = _make_finder(raster, src, tgt)
        session = finder.search_session(algorithm="dijkstra")
        path = session.route([src, tgt])
        golden = finder.find_route(source=src, target=tgt,
                                   algorithm="dijkstra",
                                   calculate_metrics=False)
        assert _indices(path) == _indices(golden)

    def test_move_target_inside_disk_matches_and_does_not_expand(self):
        raster = _uniform(60)
        src = _xy(5, 30, 60)
        tgt = _xy(40, 30, 60)
        tgt2 = _xy(25, 30, 60)  # on the S–T axis, already settled
        finder = _make_finder(raster, src, tgt)
        session = finder.search_session(algorithm="dijkstra")
        session.route([src, tgt])
        updated = session.update([src, tgt2])
        golden = finder.find_route(source=src, target=tgt2,
                                   algorithm="dijkstra",
                                   calculate_metrics=False)
        assert _indices(updated) == _indices(golden)
        assert session.expansion_count == 0

    def test_move_target_outside_disk_matches(self):
        raster = _corridor_barrier(80)
        src = _xy(5, 40, 80)
        tgt = _xy(20, 40, 80)
        tgt2 = _xy(70, 8, 80)  # around the blob, outside the first disk
        finder = _make_finder(raster, src, tgt)
        session = finder.search_session(algorithm="dijkstra")
        session.route([src, tgt])
        updated = session.update([src, tgt2])
        golden = finder.find_route(source=src, target=tgt2,
                                   algorithm="dijkstra",
                                   calculate_metrics=False)
        assert _indices(updated) == _indices(golden)
        assert session.expansion_count > 0

    def test_move_source_matches(self):
        raster = _walled(80)
        src, tgt = _xy(5, 40, 80), _xy(70, 40, 80)
        src2 = _xy(5, 10, 80)
        finder = _make_finder(raster, src, tgt)
        session = finder.search_session(algorithm="dijkstra")
        session.route([src, tgt])
        updated = session.update([src2, tgt])
        golden = finder.find_route(source=src2, target=tgt,
                                   algorithm="dijkstra",
                                   calculate_metrics=False)
        assert _indices(updated) == _indices(golden)

    @pytest.mark.parametrize("edit", [
        "insert_waypoint",
        "move_middle",
        "delete_waypoint",
    ])
    def test_waypoint_edits_match_fresh_chain(self, edit):
        raster = _corridor_barrier(80)
        src, tgt = _xy(5, 40, 80), _xy(70, 40, 80)
        mid = _xy(40, 8, 80)
        mid2 = _xy(50, 8, 80)
        finder = _make_finder(raster, src, tgt)
        session = finder.search_session(algorithm="dijkstra")
        session.route([src, tgt])
        if edit == "insert_waypoint":
            points = [src, mid, tgt]
        elif edit == "move_middle":
            session.update([src, mid, tgt])
            points = [src, mid2, tgt]
        else:
            session.update([src, mid, tgt])
            points = [src, tgt]
        updated = session.update(points)
        golden = _golden_chain(finder, points, "dijkstra")
        assert _indices(updated) == golden

    def test_unchanged_leg_is_not_researched(self):
        raster = _uniform(60)
        src = _xy(5, 30, 60)
        mid = _xy(25, 30, 60)
        tgt = _xy(45, 30, 60)
        tgt2 = _xy(50, 30, 60)
        finder = _make_finder(raster, src, tgt)
        session = finder.search_session(algorithm="dijkstra")
        session.route([src, mid, tgt])
        session.update([src, mid, tgt2])
        assert session.last_dirty_legs == ((mid, tgt2),)

    def test_window_exit_rebuilds_and_matches(self):
        raster = _uniform(200)
        src = _xy(10, 10, 200)
        tgt = _xy(20, 10, 200)
        tgt2 = _xy(180, 180, 200)
        finder = _make_finder(raster, src, tgt, buffer_m=15)
        session = finder.search_session(algorithm="dijkstra")
        session.route([src, tgt])
        updated = session.update([src, tgt2])
        fresh = _make_finder(raster, src, tgt2, buffer_m=15)
        golden = fresh.find_route(source=src, target=tgt2,
                                  algorithm="dijkstra",
                                  calculate_metrics=False)
        assert _indices(updated) == _indices(golden)

    def test_close_releases_memory(self):
        raster = _uniform(40)
        src, tgt = _xy(5, 20, 40), _xy(30, 20, 40)
        finder = _make_finder(raster, src, tgt)
        session = finder.search_session(algorithm="dijkstra")
        session.route([src, tgt])
        assert session.memory_bytes > 0
        session.close()
        assert session.memory_bytes == 0

    def test_neighborhood_change_invalidates(self):
        raster = _uniform(40)
        src, tgt = _xy(5, 20, 40), _xy(30, 20, 40)
        finder = _make_finder(raster, src, tgt, neighborhood="r1")
        session = finder.search_session(algorithm="dijkstra")
        session.route([src, tgt])
        finder.neighborhood_str = "r2"
        with pytest.raises(ValueError, match="invalidat"):
            session.update([src, _xy(28, 20, 40)])

    def test_find_route_without_session_still_resets(self):
        """Default find_route must not keep session state on the shared solver."""
        raster = _corridor_barrier(60)
        src, tgt = _xy(5, 30, 60), _xy(50, 30, 60)
        finder = _make_finder(raster, src, tgt)
        a = finder.find_route(source=src, target=tgt, algorithm="dijkstra",
                              calculate_metrics=False)
        b = finder.find_route(source=src, target=tgt, algorithm="dijkstra",
                              calculate_metrics=False)
        assert _indices(a) == _indices(b)

    def test_release_device_resources_closes_session(self):
        raster = _uniform(40)
        src, tgt = _xy(5, 20, 40), _xy(30, 20, 40)
        finder = _make_finder(raster, src, tgt)
        session = finder.search_session(algorithm="dijkstra")
        session.route([src, tgt])
        finder.release_device_resources()
        assert session.memory_bytes == 0


# ---------------------------------------------------------------------------
# CPU delta-stepping (full-field v1)
# ---------------------------------------------------------------------------

class TestDeltaSteppingSession:
    def test_route_and_target_move_match_find_route(self):
        raster = _corridor_barrier(80)
        src, tgt = _xy(5, 40, 80), _xy(70, 40, 80)
        tgt2 = _xy(70, 8, 80)
        finder = _make_finder(raster, src, tgt)
        session = finder.search_session(algorithm="delta-stepping", delta=5.0)
        path = session.route([src, tgt])
        golden = finder.find_route(source=src, target=tgt,
                                   algorithm="delta-stepping",
                                   calculate_metrics=False, delta=5.0)
        assert _indices(path) == _indices(golden)
        updated = session.update([src, tgt2])
        golden2 = finder.find_route(source=src, target=tgt2,
                                    algorithm="delta-stepping",
                                    calculate_metrics=False, delta=5.0)
        assert _indices(updated) == _indices(golden2)
        assert session.expansion_count == 0  # full field: extract only

    def test_finite_unsettled_is_not_extracted(self):
        """A finite label past the cutoff must not count as settled.

        Full-field session solves have an infinite cutoff, so this pins
        the predicate itself against a synthetic targeted leftover.
        """
        from pyorps.graph.search_session import _delta_settled
        assert _delta_settled(dist=10.0, cutoff=20.0) is True
        assert _delta_settled(dist=20.0, cutoff=20.0) is False
        assert _delta_settled(dist=25.0, cutoff=20.0) is False
        assert _delta_settled(dist=float("inf"), cutoff=20.0) is False

    def test_waypoint_insert_matches_chain(self):
        raster = _walled(80)
        src, tgt = _xy(5, 40, 80), _xy(70, 40, 80)
        mid = _xy(40, 10, 80)
        finder = _make_finder(raster, src, tgt)
        session = finder.search_session(algorithm="delta-stepping", delta=5.0)
        session.route([src, tgt])
        updated = session.update([src, mid, tgt])
        golden = _golden_chain(finder, [src, mid, tgt], "delta-stepping",
                               delta=5.0)
        assert _indices(updated) == golden


# ---------------------------------------------------------------------------
# GPU / FIM (skipped without CUDA)
# ---------------------------------------------------------------------------

def _unique(n=80):
    """Strictly patterned costs so competing shortest paths are rare."""
    rows = np.arange(n, dtype=np.int32)[:, None]
    cols = np.arange(n, dtype=np.int32)[None, :]
    return (1 + (rows * 31 + cols * 17) % 250).astype(np.uint16)


@pytest.mark.skipif(not GPU, reason="CUDA GPU not available")
class TestGpuSession:
    def test_route_and_edit_match_find_route(self):
        raster = _unique(80)
        src, tgt = _xy(5, 40, 80), _xy(70, 40, 80)
        tgt2 = _xy(70, 8, 80)
        finder = _make_finder(raster, src, tgt, graph_api="raster_gpu")
        session = finder.search_session(algorithm="delta-stepping")
        path = session.route([src, tgt])
        golden = finder.find_route(source=src, target=tgt,
                                   calculate_metrics=False)
        assert _indices(path) == _indices(golden)
        updated = session.update([src, tgt2])
        golden2 = finder.find_route(source=src, target=tgt2,
                                    calculate_metrics=False)
        assert _indices(updated) == _indices(golden2)


# ---------------------------------------------------------------------------
# Reverse trees (reuse_reverse=True)
# ---------------------------------------------------------------------------

class TestReverseTrees:
    def test_reverse_off_by_default_is_bit_identical(self):
        raster = _walled(80)
        src, tgt = _xy(5, 40, 80), _xy(70, 40, 80)
        src2 = _xy(5, 10, 80)
        finder = _make_finder(raster, src, tgt)
        session = finder.search_session(algorithm="dijkstra")
        session.route([src, tgt])
        updated = session.update([src2, tgt])
        golden = finder.find_route(source=src2, target=tgt,
                                   algorithm="dijkstra",
                                   calculate_metrics=False)
        assert _indices(updated) == _indices(golden)

    def test_source_move_costs_nothing_after_backward_tree_exists(self):
        raster = _uniform(60)
        s1 = _xy(5, 30, 60)
        s2 = _xy(45, 5, 60)
        s3 = _xy(40, 8, 60)  # near s2 -- inside its backward disk
        tgt = _xy(30, 55, 60)
        finder = _make_finder(raster, s1, tgt)
        session = finder.search_session(algorithm="dijkstra", reuse_reverse=True)
        session.route([s1, tgt])
        session.update([s2, tgt])
        assert session.expansion_count > 0  # cold backward build
        session.update([s3, tgt])
        assert session.expansion_count == 0

    def test_source_move_is_cheaper_than_cold(self):
        raster = _uniform(60)
        s1 = _xy(5, 30, 60)
        s2 = _xy(45, 5, 60)
        s3 = _xy(40, 8, 60)
        tgt = _xy(30, 55, 60)
        finder = _make_finder(raster, s1, tgt)
        session = finder.search_session(algorithm="dijkstra", reuse_reverse=True)
        session.route([s1, tgt])
        session.update([s2, tgt])
        third_move_expansions = None
        session.update([s3, tgt])
        third_move_expansions = session.expansion_count

        fresh_finder = _make_finder(raster, s1, tgt)
        fresh_session = fresh_finder.search_session(algorithm="dijkstra")
        fresh_session.route([s3, tgt])
        assert third_move_expansions < fresh_session.expansion_count

    def test_reverse_indices_match_on_tie_free_raster(self):
        raster = _unique(80)
        src, tgt = _xy(5, 40, 80), _xy(70, 40, 80)
        src2 = _xy(5, 10, 80)
        finder = _make_finder(raster, src, tgt)
        session = finder.search_session(algorithm="dijkstra", reuse_reverse=True)
        session.route([src, tgt])
        updated = session.update([src2, tgt])
        golden = finder.find_route(source=src2, target=tgt,
                                   algorithm="dijkstra",
                                   calculate_metrics=False)
        assert _indices(updated) == _indices(golden)

    def test_reverse_cost_matches_on_tie_rich_raster(self):
        raster = _walled(80)
        src, tgt = _xy(5, 40, 80), _xy(70, 40, 80)
        src2 = _xy(5, 10, 80)
        finder = _make_finder(raster, src, tgt)
        session = finder.search_session(algorithm="dijkstra", reuse_reverse=True)
        session.route([src, tgt])
        updated = session.update([src2, tgt])
        golden = finder.find_route(source=src2, target=tgt,
                                   algorithm="dijkstra",
                                   calculate_metrics=True)
        got = finder._create_path_result(
            np.asarray(_indices(updated), dtype=np.uint32),
            src2, tgt, "dijkstra", True)
        assert got.total_cost == pytest.approx(golden.total_cost, rel=1e-6)

    def test_backward_tree_not_built_when_forward_is_warm(self):
        raster = _uniform(60)
        s1 = _xy(5, 30, 60)
        tgt = _xy(45, 30, 60)
        tgt2 = _xy(50, 30, 60)
        finder = _make_finder(raster, s1, tgt)
        session = finder.search_session(algorithm="dijkstra", reuse_reverse=True)
        session.route([s1, tgt])
        session.update([s1, tgt2])
        assert session._backward == {}

    def test_gc_keeps_ends_and_starts(self):
        raster = _uniform(60)
        s1 = _xy(5, 30, 60)
        s2 = _xy(10, 30, 60)
        tgt = _xy(45, 30, 60)
        finder = _make_finder(raster, s1, tgt)
        session = finder.search_session(algorithm="dijkstra", reuse_reverse=True)
        session.route([s1, tgt])
        session.update([s2, tgt])
        points = session._points
        assert set(session._backward) <= set(points[1:])
        assert set(session._forward) <= set(points[:-1])

    def test_max_trees_caps_resident_trees(self):
        raster = _uniform(60)
        s0 = _xy(5, 30, 60)
        tgt = _xy(45, 30, 60)
        finder = _make_finder(raster, s0, tgt)
        session = finder.search_session(algorithm="dijkstra", reuse_reverse=True,
                                        max_trees=2)
        session.route([s0, tgt])
        sources = [_xy(c, 30, 60) for c in (8, 12, 16, 20, 24)]
        for s in sources:
            updated = session.update([s, tgt])
            golden = finder.find_route(source=s, target=tgt, algorithm="dijkstra",
                                       calculate_metrics=False)
            assert _indices(updated) == _indices(golden)
            assert len(session._forward) + len(session._backward) <= 2

    def test_memory_bytes_counts_both_dicts(self):
        # Insert a waypoint rather than moving an existing point: gc keeps
        # the original forward[s1] tree (s1 is still a "start" in the new
        # route) while the new leg (mid, tgt) is cold and gets a NEW
        # backward tree -- resident tree count goes from 1 to 2, so
        # memory_bytes must grow (moving a point instead can evict as many
        # trees as it adds, leaving the count -- and the byte total on a
        # same-size window -- unchanged).
        raster = _uniform(60)
        s1 = _xy(5, 30, 60)
        mid = _xy(30, 30, 60)
        tgt = _xy(45, 30, 60)
        finder = _make_finder(raster, s1, tgt)
        session = finder.search_session(algorithm="dijkstra", reuse_reverse=True)
        session.route([s1, tgt])
        before = session.memory_bytes
        session.update([s1, mid, tgt])
        assert session.memory_bytes > before
        session.close()
        assert session.memory_bytes == 0

    def test_reverse_rejected_on_half_step_set(self):
        from pyorps.utils.neighborhood import get_neighborhood_steps
        # r1-directed=False keeps only steps with dc >= 0 (never left), so
        # reachability under this restricted step set requires
        # target_col >= source_col for every leg -- pick points accordingly.
        raster = _uniform(60)
        s1 = _xy(5, 30, 60)
        s2 = _xy(15, 45, 60)
        tgt = _xy(50, 10, 60)
        finder = _make_finder(raster, s1, tgt)
        finder.steps = get_neighborhood_steps("r1", directed=False)
        session = finder.search_session(algorithm="dijkstra", reuse_reverse=True)
        assert session._can_reverse is False
        session.route([s1, tgt])
        updated = session.update([s2, tgt])
        golden = finder.find_route(source=s2, target=tgt, algorithm="dijkstra",
                                   calculate_metrics=False)
        assert _indices(updated) == _indices(golden)

    def test_search_until_and_settle_all_agree(self):
        from pyorps.utils._dijkstra import make_dijkstra_solver
        raster = np.random.default_rng(3).integers(1, 300, (60, 60)).astype(
            np.uint16)
        from pyorps.utils.neighborhood import get_neighborhood_steps
        steps = get_neighborhood_steps("r2", directed=True)

        s1 = make_dijkstra_solver(raster, steps, max_value=65535,
                                  dem=None, gradient_luts=None)
        s1.reset_root(0)
        s1.search_until(400)
        s1.search_until(1500)
        s1.settle_all()
        full1 = s1.dist_array()

        s2 = make_dijkstra_solver(raster, steps, max_value=65535,
                                  dem=None, gradient_luts=None)
        s2.reset_root(0)
        s2.settle_all()
        full2 = s2.dist_array()

        np.testing.assert_allclose(full1, full2)


@pytest.mark.skipif(not GPU, reason="CUDA GPU not available")
class TestFimSession:
    def test_route_and_edit_match_find_route(self):
        raster = _uniform(80)
        src, tgt = _xy(5, 40, 80), _xy(70, 40, 80)
        tgt2 = _xy(70, 10, 80)
        finder = _make_finder(raster, src, tgt, graph_api="raster_fim")
        session = finder.search_session(algorithm="fim")
        path = session.route([src, tgt])
        golden = finder.find_route(source=src, target=tgt,
                                   calculate_metrics=False)
        assert _indices(path) == _indices(golden)
        updated = session.update([src, tgt2])
        golden2 = finder.find_route(source=src, target=tgt2,
                                    calculate_metrics=False)
        assert _indices(updated) == _indices(golden2)
