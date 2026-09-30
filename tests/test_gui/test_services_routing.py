"""Unit tests: pyorps.gui.services.routing — C13/C14, F1, F5, F8."""
import pytest
from shapely.geometry import LineString

from pyorps.gui.services import routing

from conftest import SOURCE, TARGET, TEST_CRS


# ------------------------------------------------------------- C14 matrix
@pytest.mark.parametrize("algorithm,hardware,expected", [
    ("dijkstra", "cpu", ("cython", "dijkstra")),
    ("delta-stepping", "cpu", ("cython", "delta-stepping")),
    ("delta-stepping-circular", "cpu", ("cython", "delta-stepping-circular")),
    ("dijkstra", "gpu", ("raster_gpu", "dijkstra")),
    ("delta-stepping", "gpu", ("raster_gpu", "delta-stepping")),
    ("bidirectional_dijkstra", "cpu", ("networkit", "bidirectional_dijkstra")),
    # no GPU implementation -> falls back to the CPU backend
    ("bidirectional_dijkstra", "gpu", ("networkit", "bidirectional_dijkstra")),
    ("astar", "cpu", ("networkit", "astar")),
    ("bellman_ford", "cpu", ("igraph", "bellman_ford")),
    # unknown algorithm -> safe default
    ("banana", "cpu", ("cython", "delta-stepping")),
])
def test_resolve_backend_matrix(algorithm, hardware, expected):
    assert routing.resolve_backend(algorithm, hardware) == expected


def test_valid_algorithms_filtered_by_hardware():
    assert "bidirectional_dijkstra" in routing.valid_algorithms("cpu")
    assert "bidirectional_dijkstra" not in routing.valid_algorithms("gpu")
    assert "dijkstra" in routing.valid_algorithms("gpu")


def test_recommend_gpu_threshold():
    assert not routing.recommend_gpu(1_000_000)
    assert routing.recommend_gpu(5_000_000)


# ------------------------------------------------------------------ F1 buffer
def test_default_search_buffer_never_none():
    buf = routing.default_search_buffer([(0, 0)], [(100, 0)])
    assert buf == 1000.0  # floor
    buf = routing.default_search_buffer([(0, 0)], [(2000, 0)])
    assert buf == pytest.approx(3000.0)  # 1.5 x euclid


# --------------------------------------------------------------------- pairs
def test_make_pairs_cross_and_pairwise():
    assert routing.make_pairs([1, 2], ["a"], False) == [(1, "a"), (2, "a")]
    assert routing.make_pairs([1, 2], ["a", "b"], True) == [(1, "a"), (2, "b")]


def test_make_pairs_pairwise_mismatch_raises():
    from pyorps.core.exceptions import PairwiseError

    with pytest.raises(PairwiseError):
        routing.make_pairs([1, 2], ["a"], True)


# ------------------------------------------------------------------ F5 helper
def test_simplify_line_reduces_vertices():
    line = LineString([(0, 0), (1, 0.01), (2, -0.01), (3, 0), (10, 0)])
    simplified = routing.simplify_line(line, 1.0)
    assert len(simplified.coords) < len(line.coords)
    assert simplified.coords[0] == (0, 0) and simplified.coords[-1] == (10, 0)


# ----------------------------------------------------- chained routing (C13)
def test_route_through_points_requires_two(finder):
    with pytest.raises(ValueError, match="at least"):
        routing.route_through_points(finder, [SOURCE])


def test_route_through_points_passes_waypoint(finder):
    waypoint = (500100.0, 5599950.0)
    line, rc = routing.route_through_points(
        finder, [SOURCE, waypoint, TARGET], algorithm="dijkstra")
    assert line.geom_type == "LineString"
    assert rc.total_length > 0
    # the stitched line passes through (near) the waypoint
    assert line.distance(LineString([waypoint, waypoint]).centroid) < 5


def test_run_routing_end_to_end(raster_path):
    finder, results, failed = routing.run_routing(
        raster_path, sources=[SOURCE], targets=[TARGET],
        algorithm="dijkstra", hardware="cpu", search_buffer_m=80)
    assert failed == []
    assert len(results) == 1
    built = results[0]
    assert built.line.geom_type == "LineString"
    assert built.cost.total_cost > 0
    assert built.params["graph_api"] == "cython"
    assert built.params["use_astar"] is False           # F8
    assert built.params["search_space_buffer_m"] == 80
    assert built.control_points == [SOURCE, TARGET]
    assert built.session is not None
    assert built.session.has_route
    assert built.session.memory_bytes > 0


def test_route_through_points_session_matches_oneshot(finder):
    session = finder.search_session(algorithm="dijkstra")
    line_s, _rc_s = routing.route_through_points(
        finder, [SOURCE, TARGET], algorithm="dijkstra", session=session)
    line, _rc = routing.route_through_points(
        finder, [SOURCE, TARGET], algorithm="dijkstra")
    assert list(line_s.coords) == list(line.coords)
    moved = (SOURCE[0] + 10.0, SOURCE[1] - 10.0)
    line_u, _ = routing.route_through_points(
        finder, [SOURCE, moved], algorithm="dijkstra", session=session)
    assert line_u.coords[-1] != line_s.coords[-1]


def test_run_routing_defaults_buffer_when_blank(raster_path):
    finder, results, failed = routing.run_routing(
        raster_path, sources=[SOURCE], targets=[TARGET],
        algorithm="dijkstra", search_buffer_m=None)
    # F1: the buffer was computed, not left blank
    assert results[0].params["search_space_buffer_m"] >= 1000


def test_finalize_and_recompute_moves_session(state, raster_path):
    from pyorps.gui.callbacks.edit import recompute_variant
    from pyorps.gui.callbacks.routes import finalize_routing

    raster = state.add_layer(
        "cost", "raster", crs=TEST_CRS,
        meta={"source_path": raster_path})
    finder, results, failed = routing.run_routing(
        raster_path, sources=[SOURCE], targets=[TARGET],
        algorithm="dijkstra", hardware="cpu", search_buffer_m=80)
    notices = []
    finalize_routing(
        state, finder, results, failed,
        meta={"raster_layer_id": raster.id, "n_sources": 1, "n_targets": 1},
        notices=notices)
    route = state.layers_of_kind("route")[0]
    session = state.search_sessions[route.id]
    new_pts = [SOURCE, (TARGET[0] - 20.0, TARGET[1] + 20.0)]
    new_layer, notices = recompute_variant(
        state, route, new_pts, "moved target", notices)
    assert new_layer is not None
    assert route.id not in state.search_sessions
    assert state.search_sessions[new_layer.id] is session


def test_run_routing_requires_points(raster_path):
    with pytest.raises(ValueError, match="must not be None"):
        routing.run_routing(raster_path, sources=[], targets=[(1, 2)])
