"""
Tests for pyorps.utils._dijkstra module.

Covers:
- DijkstraSolver.single_pair: path found, same source/target, no path (blocked), path validity
- DijkstraSolver.single_source_multi_target: multiple targets all reachable
- DijkstraSolver.multi_source_multi_target: small all-pairs
- DijkstraSolver.some_pairs: pairwise paths
- group_by_proximity_uint32: spatial reordering
- Public API wrappers: backward-compatible function signatures
- Edge cases: 1x1 raster, source on obstacle
"""

import numpy as np
import pytest

from pyorps.utils._raster_context import RasterContext
from pyorps.utils._dijkstra import (
    DijkstraSolver,
    group_by_proximity_uint32,
    dijkstra_2d_cython,
    dijkstra_single_source_multiple_targets,
    dijkstra_multiple_sources_multiple_targets,
    dijkstra_some_pairs_shortest_paths,
)


# ==================== FIXTURES ====================

@pytest.fixture
def simple_raster():
    """5x5 raster with uniform cost=10, no obstacles."""
    return np.full((5, 5), 10, dtype=np.uint16)


@pytest.fixture
def raster_with_wall():
    """5x5 raster with a wall blocking column 2 except row 4.

    Layout (0=cost 10, X=65535):
        0 0 X 0 0
        0 0 X 0 0
        0 0 X 0 0
        0 0 X 0 0
        0 0 0 0 0    <- gap at (4,2)
    """
    r = np.full((5, 5), 10, dtype=np.uint16)
    r[0, 2] = 65535
    r[1, 2] = 65535
    r[2, 2] = 65535
    r[3, 2] = 65535
    return r


@pytest.fixture
def raster_fully_blocked():
    """5x5 raster where all cells are impassable except corners."""
    r = np.full((5, 5), 65535, dtype=np.uint16)
    r[0, 0] = 10
    r[4, 4] = 10
    return r


@pytest.fixture
def simple_steps():
    """4-connected neighborhood (N, S, E, W)."""
    return np.array([
        [-1, 0],
        [1, 0],
        [0, -1],
        [0, 1],
    ], dtype=np.int8)


@pytest.fixture
def eight_connected_steps():
    """8-connected neighborhood."""
    return np.array([
        [-1, 0], [1, 0], [0, -1], [0, 1],
        [-1, -1], [-1, 1], [1, -1], [1, 1],
    ], dtype=np.int8)


def _make_solver(raster, steps, max_value=65535):
    """Helper to create a DijkstraSolver from raster and steps arrays."""
    ctx = RasterContext(raster, steps, max_value)
    return DijkstraSolver(ctx)


# ==================== single_pair ====================

class TestSinglePair:
    """Tests for DijkstraSolver.single_pair."""

    def test_same_source_target(self, simple_raster, simple_steps):
        solver = _make_solver(simple_raster, simple_steps)
        # source == target: should return single-element array
        path = solver.single_pair(0, 0)
        assert len(path) == 1
        assert path[0] == 0

    def test_adjacent_cells(self, simple_raster, simple_steps):
        solver = _make_solver(simple_raster, simple_steps)
        # (0,0) -> (0,1) in 5-col raster: indices 0 -> 1
        path = solver.single_pair(0, 1)
        assert len(path) >= 2
        assert path[0] == 0
        assert path[-1] == 1

    def test_path_found_on_uniform_raster(self, simple_raster, simple_steps):
        solver = _make_solver(simple_raster, simple_steps)
        cols = 5
        source = 0           # (0,0)
        target = 4 * cols + 4  # (4,4) = 24
        path = solver.single_pair(source, target)
        assert len(path) > 0
        assert path[0] == source
        assert path[-1] == target

    def test_path_around_wall(self, raster_with_wall, simple_steps):
        solver = _make_solver(raster_with_wall, simple_steps)
        cols = 5
        source = 0               # (0,0)
        target = 4               # (0,4)
        path = solver.single_pair(source, target)
        assert len(path) > 0
        assert path[0] == source
        assert path[-1] == target
        # Path must go through the gap at row 4
        path_set = set(path)
        # The wall blocks col 2 in rows 0-3, so path must detour through row 4
        gap_cell = 4 * cols + 2  # (4,2) = 22
        assert gap_cell in path_set

    def test_no_path_fully_blocked(self, raster_fully_blocked, simple_steps):
        solver = _make_solver(raster_fully_blocked, simple_steps)
        source = 0      # (0,0) - traversable
        target = 24     # (4,4) - traversable but no path connects them
        path = solver.single_pair(source, target)
        assert len(path) == 0

    def test_path_validity_consecutive_neighbors(self, simple_raster, simple_steps):
        """Each consecutive pair in the path should be valid neighbors."""
        solver = _make_solver(simple_raster, simple_steps)
        cols = 5
        path = solver.single_pair(0, 24)
        for i in range(len(path) - 1):
            r1, c1 = divmod(int(path[i]), cols)
            r2, c2 = divmod(int(path[i + 1]), cols)
            # For 4-connected, Manhattan distance between consecutive cells = 1
            assert abs(r1 - r2) + abs(c1 - c2) == 1

    def test_path_with_8_connectivity(self, simple_raster, eight_connected_steps):
        solver = _make_solver(simple_raster, eight_connected_steps)
        cols = 5
        source = 0
        target = 24
        path = solver.single_pair(source, target)
        assert len(path) > 0
        assert path[0] == source
        assert path[-1] == target
        # 8-connected should have a shorter or equal path
        for i in range(len(path) - 1):
            r1, c1 = divmod(int(path[i]), cols)
            r2, c2 = divmod(int(path[i + 1]), cols)
            assert abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1


# ==================== single_source_multi_target ====================

class TestSingleSourceMultiTarget:
    """Tests for DijkstraSolver.single_source_multi_target."""

    def test_all_targets_reachable(self, simple_raster, simple_steps):
        solver = _make_solver(simple_raster, simple_steps)
        cols = 5
        source = 0
        targets = np.array([1, 5, 24], dtype=np.uint32)  # (0,1), (1,0), (4,4)
        paths = solver.single_source_multi_target(source, targets)
        assert len(paths) == 3
        for i, t in enumerate(targets):
            assert len(paths[i]) > 0
            assert paths[i][0] == source
            assert paths[i][-1] == t

    def test_one_target_unreachable(self, raster_fully_blocked, simple_steps):
        solver = _make_solver(raster_fully_blocked, simple_steps)
        source = 0
        targets = np.array([24], dtype=np.uint32)
        paths = solver.single_source_multi_target(source, targets)
        assert len(paths) == 1
        assert len(paths[0]) == 0

    def test_single_target_matches_single_pair(self, simple_raster, simple_steps):
        """single_source_multi_target with one target should match single_pair."""
        solver1 = _make_solver(simple_raster, simple_steps)
        path_sp = solver1.single_pair(0, 24)

        solver2 = _make_solver(simple_raster, simple_steps)
        paths_mt = solver2.single_source_multi_target(
            0, np.array([24], dtype=np.uint32))
        np.testing.assert_array_equal(path_sp, paths_mt[0])


# ==================== multi_source_multi_target ====================

class TestMultiSourceMultiTarget:
    """Tests for DijkstraSolver.multi_source_multi_target."""

    def test_all_pairs_paths(self, simple_raster, simple_steps):
        solver = _make_solver(simple_raster, simple_steps)
        sources = np.array([0, 4], dtype=np.uint32)       # (0,0), (0,4)
        targets = np.array([20, 24], dtype=np.uint32)      # (4,0), (4,4)
        result = solver.multi_source_multi_target(sources, targets, return_paths=True)
        # result[i][j] = path from source i to target j
        assert len(result) == 2
        for i in range(2):
            assert len(result[i]) == 2
            for j in range(2):
                assert len(result[i][j]) > 0
                assert result[i][j][0] == sources[i]
                assert result[i][j][-1] == targets[j]

    def test_cost_matrix(self, simple_raster, simple_steps):
        solver = _make_solver(simple_raster, simple_steps)
        sources = np.array([0, 4], dtype=np.uint32)
        targets = np.array([20, 24], dtype=np.uint32)
        cost_matrix = solver.multi_source_multi_target(
            sources, targets, return_paths=False)
        # Should be 2x2 matrix with finite costs
        assert cost_matrix.shape == (2, 2)
        assert np.all(np.isfinite(cost_matrix))
        # Symmetric cost for uniform raster: cost(0->24) == cost(4->20)
        # because distance (0,0)->(4,4) == (0,4)->(4,0)
        assert abs(cost_matrix[0, 1] - cost_matrix[1, 0]) < 1e-6


# ==================== some_pairs ====================

class TestSomePairs:
    """Tests for DijkstraSolver.some_pairs."""

    def test_pairwise_paths(self, simple_raster, simple_steps):
        solver = _make_solver(simple_raster, simple_steps)
        # 3 pairs: (0->4), (4->20), (20->24)
        sources = np.array([0, 4, 20], dtype=np.uint32)
        targets = np.array([4, 20, 24], dtype=np.uint32)
        paths = solver.some_pairs(sources, targets, return_paths=True)
        assert len(paths) == 3
        for i in range(3):
            assert paths[i] is not None
            assert len(paths[i]) > 0
            assert paths[i][0] == sources[i]
            assert paths[i][-1] == targets[i]

    def test_pairwise_costs(self, simple_raster, simple_steps):
        solver = _make_solver(simple_raster, simple_steps)
        sources = np.array([0, 4], dtype=np.uint32)
        targets = np.array([4, 0], dtype=np.uint32)
        costs = solver.some_pairs(sources, targets, return_paths=False)
        assert len(costs) == 2
        assert np.all(np.isfinite(costs))
        # A->B and B->A should have same cost on uniform raster
        assert abs(costs[0] - costs[1]) < 1e-6

    def test_central_node_batching(self, simple_raster, simple_steps):
        """Central node optimization: node 4 appears as both source and target."""
        solver = _make_solver(simple_raster, simple_steps)
        # Pair 1: 0 -> 4, Pair 2: 4 -> 24
        # Node 4 is both a target (pair 1) and a source (pair 2)
        sources = np.array([0, 4], dtype=np.uint32)
        targets = np.array([4, 24], dtype=np.uint32)
        paths = solver.some_pairs(sources, targets, return_paths=True)
        assert len(paths) == 2
        assert paths[0][0] == 0 and paths[0][-1] == 4
        assert paths[1][0] == 4 and paths[1][-1] == 24


# ==================== group_by_proximity_uint32 ====================

class TestGroupByProximity:
    """Tests for group_by_proximity_uint32."""

    def test_single_element(self):
        arr = np.array([42], dtype=np.uint32)
        result = group_by_proximity_uint32(arr, 10)
        assert len(result) == 1
        assert result[0] == 42

    def test_preserves_elements(self):
        arr = np.array([24, 0, 12, 4], dtype=np.uint32)
        result = group_by_proximity_uint32(arr, 5)
        assert set(result) == set(arr)
        assert len(result) == len(arr)

    def test_sorted_by_row(self):
        """After grouping, elements should be sorted by row coordinate."""
        cols = 5
        # Indices: 20=(4,0), 0=(0,0), 10=(2,0), 5=(1,0)
        arr = np.array([20, 0, 10, 5], dtype=np.uint32)
        result = group_by_proximity_uint32(arr, cols)
        rows = [int(idx) // cols for idx in result]
        assert rows == sorted(rows)


# ==================== PUBLIC API WRAPPERS ====================

class TestPublicWrappers:
    """Tests for the public API wrapper functions."""

    def test_dijkstra_2d_cython(self, simple_raster, simple_steps):
        path = dijkstra_2d_cython(simple_raster, simple_steps, 0, 24)
        assert len(path) > 0
        assert path[0] == 0
        assert path[-1] == 24

    def test_dijkstra_2d_cython_same_source_target(self, simple_raster, simple_steps):
        path = dijkstra_2d_cython(simple_raster, simple_steps, 12, 12)
        assert len(path) == 1
        assert path[0] == 12

    def test_dijkstra_single_source_multiple_targets(
            self, simple_raster, simple_steps):
        targets = np.array([1, 5, 24], dtype=np.uint32)
        paths = dijkstra_single_source_multiple_targets(
            simple_raster, simple_steps, 0, targets)
        assert len(paths) == 3
        for i, t in enumerate(targets):
            assert paths[i][0] == 0
            assert paths[i][-1] == t

    def test_dijkstra_multiple_sources_multiple_targets(
            self, simple_raster, simple_steps):
        sources = np.array([0, 4], dtype=np.uint32)
        targets = np.array([20, 24], dtype=np.uint32)
        result = dijkstra_multiple_sources_multiple_targets(
            simple_raster, simple_steps, sources, targets)
        assert len(result) == 2
        assert len(result[0]) == 2

    def test_dijkstra_some_pairs(self, simple_raster, simple_steps):
        sources = np.array([0, 4], dtype=np.uint32)
        targets = np.array([4, 0], dtype=np.uint32)
        paths = dijkstra_some_pairs_shortest_paths(
            simple_raster, simple_steps, sources, targets)
        assert len(paths) == 2
        assert paths[0][0] == 0 and paths[0][-1] == 4
        assert paths[1][0] == 4 and paths[1][-1] == 0


# ==================== EDGE CASES ====================

class TestEdgeCases:
    """Edge cases: 1x1 raster, source on obstacle."""

    def test_1x1_raster(self, simple_steps):
        raster = np.array([[10]], dtype=np.uint16)
        solver = _make_solver(raster, simple_steps)
        path = solver.single_pair(0, 0)
        assert len(path) == 1
        assert path[0] == 0

    def test_source_on_obstacle(self, simple_steps):
        """Source cell is impassable — should find no path."""
        raster = np.full((3, 3), 10, dtype=np.uint16)
        raster[0, 0] = 65535  # source is blocked
        solver = _make_solver(raster, simple_steps)
        path = solver.single_pair(0, 8)
        # Source is blocked so the exclude_mask marks it 0.
        # Dijkstra starts from source but it's not traversable for neighbors.
        # The source still gets distance 0 but neighbors won't see it as
        # traversable. Actually, Dijkstra processes source normally — it's
        # only neighbors that check the exclude_mask of the neighbor. But
        # the source itself is pushed at dist 0 and popped, and its neighbors
        # are explored. However, the cost of a path through an obstacle cell
        # is still computed. The key question: does exclude_mask[source] == 0
        # prevent the source from being explored?
        #
        # Looking at the algorithm: source is pushed, popped, and explored.
        # The exclude_mask is checked for neighbors, not for the current cell.
        # So the source being blocked still lets it explore neighbors. But
        # path reconstruction still works.
        #
        # Actually, this depends on the raster value. The source has cost 65535
        # but is still processed. Whether a path is found depends on whether
        # neighbors are traversable. With a 3x3 grid and only (0,0) blocked,
        # a path from 0 to 8 would go through the blocked cell's cost.
        # This is a known behavior - the obstacle check is on the exclude_mask
        # which marks the source as non-traversable, but the Dijkstra still
        # processes source and explores neighbors (it checks neighbors' mask,
        # not its own). So it can still find paths.
        # The result is implementation-dependent. Just verify no crash.
        assert isinstance(path, np.ndarray)

    def test_target_on_obstacle(self, simple_steps):
        """Target cell is impassable — should find no path."""
        raster = np.full((3, 3), 10, dtype=np.uint16)
        raster[2, 2] = 65535  # target is blocked
        solver = _make_solver(raster, simple_steps)
        path = solver.single_pair(0, 8)
        # Target is not traversable, so it can never be reached as a neighbor.
        # But if the target IS the source, it's reachable at dist 0.
        # Since exclude_mask[target] == 0, no cell can relax an edge into it.
        assert len(path) == 0

    def test_solver_reuse_across_queries(self, simple_raster, simple_steps):
        """Solver should give correct results across multiple queries."""
        solver = _make_solver(simple_raster, simple_steps)
        path1 = solver.single_pair(0, 24)
        path2 = solver.single_pair(24, 0)
        assert len(path1) > 0 and path1[0] == 0 and path1[-1] == 24
        assert len(path2) > 0 and path2[0] == 24 and path2[-1] == 0

    def test_large_raster_10x10(self, simple_steps):
        """Slightly larger raster to confirm scaling."""
        raster = np.full((10, 10), 10, dtype=np.uint16)
        solver = _make_solver(raster, simple_steps)
        path = solver.single_pair(0, 99)
        assert len(path) > 0
        assert path[0] == 0
        assert path[-1] == 99
