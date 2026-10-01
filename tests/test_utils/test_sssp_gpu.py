"""
Tests for GPU-accelerated SSSP (sssp_raster_gpu).

Validates that the GPU delta-stepping SSSP produces the same shortest-path
costs as NetworkIt Dijkstra. Both use the same edge list (from construct_edges)
so any difference in internal representation is eliminated.

All tests are guarded with skipUnless so they are skipped gracefully
on machines without an NVIDIA GPU or CuPy installed.
"""

import unittest
import numpy as np

from pyorps.utils.traversal_gpu import GPU_AVAILABLE
from pyorps.utils.traversal import construct_edges

try:
    from networkit import Graph as NKGraph
    from networkit.distance import Dijkstra as NKDijkstra
    NK_AVAILABLE = True
except ImportError:
    NK_AVAILABLE = False


def _needs_gpu_and_nk(cls):
    """Skip class unless both GPU and NetworkIt are available."""
    if not GPU_AVAILABLE:
        return unittest.skip("CUDA GPU or CuPy not available")(cls)
    if not NK_AVAILABLE:
        return unittest.skip("NetworkIt not installed")(cls)
    return cls


def _build_nk_graph(raster, steps):
    """Build a NetworkIt weighted directed graph from raster edge list."""
    from_nodes, to_nodes, costs = construct_edges(raster, steps, True)
    n = raster.shape[0] * raster.shape[1]
    g = NKGraph(n, weighted=True, directed=False)
    g.addEdges(
        (costs.astype(np.float64),
         (np.asarray(from_nodes, dtype=np.uint64),
          np.asarray(to_nodes, dtype=np.uint64))),
        addMissing=False,
    )
    return g, from_nodes, to_nodes, costs


def _nk_dijkstra(graph, source, target):
    """Run NetworkIt Dijkstra, return (distance, path)."""
    dij = NKDijkstra(graph, source, storePaths=True, target=target)
    dij.run()
    return dij.distance(target), dij.getPath(target)


def _edge_cost_of_path(path, from_nodes, to_nodes, costs):
    """Compute total edge cost of a path using the edge list."""
    if len(path) < 2:
        return 0.0
    edge_cost = {}
    for i in range(len(from_nodes)):
        u, v, c = int(from_nodes[i]), int(to_nodes[i]), float(costs[i])
        key = (min(u, v), max(u, v))
        if key not in edge_cost or c < edge_cost[key]:
            edge_cost[key] = c
    total = 0.0
    for i in range(len(path) - 1):
        u, v = int(path[i]), int(path[i + 1])
        key = (min(u, v), max(u, v))
        if key not in edge_cost:
            return float('inf')
        total += edge_cost[key]
    return total


@_needs_gpu_and_nk
class TestSSSPGpuVsNetworkit(unittest.TestCase):
    """Validate sssp_raster_gpu distances against NetworkIt Dijkstra."""

    def setUp(self):
        from pyorps.utils.sssp_gpu import sssp_raster_gpu
        self.sssp = sssp_raster_gpu
        self.steps_4 = np.array([
            [0, 1], [1, 0], [0, -1], [-1, 0]
        ], dtype=np.int8)
        self.steps_8 = np.array([
            [0, 1], [1, 0], [0, -1], [-1, 0],
            [1, 1], [-1, 1], [1, -1], [-1, -1]
        ], dtype=np.int8)

    def _run_comparison(self, raster, steps, source, target,
                        cost_tol_pct=0.01, msg=""):
        """Run GPU SSSP and NetworkIt Dijkstra, compare path costs."""
        size = raster.shape[1]

        # NetworkIt reference
        graph, from_nodes, to_nodes, costs = _build_nk_graph(raster, steps)
        nk_dist, nk_path = _nk_dijkstra(graph, int(source), int(target))

        # GPU SSSP
        target_arr = np.array([target], dtype=np.int32)
        gpu_dist, gpu_pred = self.sssp(
            raster, steps, source,
            return_predecessor=True,
            target_indices=target_arr,
        )

        # If NK says no path, GPU should agree
        if nk_dist == float('inf') or len(nk_path) == 0:
            self.assertTrue(
                np.isinf(gpu_dist[target]) or gpu_dist[target] >= 1e29,
                f"{msg}NK found no path but GPU dist={gpu_dist[target]:.4f}")
            return

        # GPU must also find finite distance
        self.assertFalse(
            np.isinf(gpu_dist[target]) or gpu_dist[target] >= 1e29,
            f"{msg}NK dist={nk_dist:.4f} but GPU dist is inf")

        # Reconstruct GPU path from predecessor
        gpu_path = self._walk_pred(gpu_pred, source, target, len(gpu_dist))
        self.assertGreater(
            len(gpu_path), 0,
            f"{msg}NK found path but GPU pred chain is broken")

        # Compare edge-based costs
        gpu_edge_cost = _edge_cost_of_path(
            gpu_path, from_nodes, to_nodes, costs)
        self.assertNotEqual(
            gpu_edge_cost, float('inf'),
            f"{msg}GPU path contains edges not in the graph")

        rel_diff = abs(gpu_edge_cost - nk_dist) / max(nk_dist, 1e-10)
        self.assertLessEqual(
            rel_diff, cost_tol_pct,
            f"{msg}Cost mismatch: NK={nk_dist:.6f}, "
            f"GPU(edge)={gpu_edge_cost:.6f}, rel_diff={rel_diff:.6f}")

    def _walk_pred(self, pred, source, target, n):
        """Walk predecessor array from target to source."""
        path = []
        current = int(target)
        visited = set()
        for _ in range(n):
            path.append(current)
            if current == int(source):
                break
            if current in visited:
                return []
            visited.add(current)
            p = int(pred[current])
            if p < 0:
                return []
            current = p
        else:
            return []
        path.reverse()
        return path

    # === Core correctness tests ===

    def test_50x50_random_4conn(self):
        """50x50 random raster, 4-connected."""
        rng = np.random.RandomState(42)
        size = 50
        raster = rng.randint(1, 100, (size, size), dtype=np.uint16)
        raster[rng.random((size, size)) < 0.1] = 65535
        raster[0, 0] = 5
        raster[size - 1, size - 1] = 5
        self._run_comparison(
            raster, self.steps_4, 0, size * size - 1,
            msg="50x50 4-conn: ")

    def test_50x50_random_8conn(self):
        """50x50 random raster, 8-connected."""
        rng = np.random.RandomState(77)
        size = 50
        raster = rng.randint(1, 100, (size, size), dtype=np.uint16)
        raster[rng.random((size, size)) < 0.1] = 65535
        raster[0, 0] = 5
        raster[size - 1, size - 1] = 5
        self._run_comparison(
            raster, self.steps_8, 0, size * size - 1,
            msg="50x50 8-conn: ")

    def test_200x200_random(self):
        """200x200 random raster with obstacles."""
        rng = np.random.RandomState(1234)
        size = 200
        raster = rng.randint(1, 500, (size, size), dtype=np.uint16)
        raster[rng.random((size, size)) < 0.05] = 65535
        raster[0, 0] = 10
        raster[size - 1, size - 1] = 10
        self._run_comparison(
            raster, self.steps_4, 0, size * size - 1,
            msg="200x200: ")

    def test_500x500_random(self):
        """500x500 random raster — larger-scale consistency."""
        rng = np.random.RandomState(9999)
        size = 500
        raster = rng.randint(1, 1000, (size, size), dtype=np.uint16)
        raster[rng.random((size, size)) < 0.03] = 65535
        raster[0, 0] = 10
        raster[size - 1, size - 1] = 10
        self._run_comparison(
            raster, self.steps_4, 0, size * size - 1,
            msg="500x500: ")

    def test_uniform_exact_match(self):
        """Uniform raster — costs must match exactly."""
        size = 100
        raster = np.full((size, size), 5, dtype=np.uint16)
        self._run_comparison(
            raster, self.steps_4, 0, size * size - 1,
            cost_tol_pct=0.0001, msg="100x100 uniform: ")

    def test_gradient_raster(self):
        """Cost gradient: cheap top-left, expensive bottom-right."""
        size = 100
        raster = np.zeros((size, size), dtype=np.uint16)
        for r in range(size):
            for c in range(size):
                raster[r, c] = max(1, min(65534, r + c + 1))
        self._run_comparison(
            raster, self.steps_4, 0, size * size - 1,
            msg="100x100 gradient: ")

    def test_no_path(self):
        """Isolated target — GPU must return inf distance."""
        raster = np.array([
            [1, 1, 65535],
            [1, 1, 65535],
            [65535, 65535, 1]
        ], dtype=np.uint16)
        target_arr = np.array([8], dtype=np.int32)
        gpu_dist = self.sssp(
            raster, self.steps_4, 0, target_indices=target_arr)
        self.assertTrue(
            np.isinf(gpu_dist[8]) or gpu_dist[8] >= 1e29,
            f"Isolated target should be unreachable, got {gpu_dist[8]}")


@_needs_gpu_and_nk
class TestSSSPGpuMultiTarget(unittest.TestCase):
    """Validate multi-target GPU SSSP against per-target NetworkIt."""

    def setUp(self):
        from pyorps.utils.sssp_gpu import sssp_raster_gpu
        self.sssp = sssp_raster_gpu
        self.steps_4 = np.array([
            [0, 1], [1, 0], [0, -1], [-1, 0]
        ], dtype=np.int8)

    def test_multi_target_costs(self):
        """Multi-target SSSP: each target distance must match NK."""
        rng = np.random.RandomState(555)
        size = 80
        raster = rng.randint(1, 200, (size, size), dtype=np.uint16)
        raster[rng.random((size, size)) < 0.05] = 65535
        raster[0, 0] = 5

        targets_list = [
            size * size - 1,
            size - 1,
            (size - 1) * size,
            size * (size // 2) + size // 2,
        ]
        for t in targets_list:
            r, c = divmod(t, size)
            raster[r, c] = max(1, min(raster[r, c], 65534))

        target_arr = np.array(targets_list, dtype=np.int32)
        gpu_dist, gpu_pred = self.sssp(
            raster, self.steps_4, 0,
            return_predecessor=True,
            target_indices=target_arr)

        graph, from_nodes, to_nodes, costs = _build_nk_graph(
            raster, self.steps_4)

        for target in targets_list:
            nk_dist, nk_path = _nk_dijkstra(graph, 0, target)
            if nk_dist == float('inf'):
                self.assertTrue(
                    np.isinf(gpu_dist[target]) or gpu_dist[target] >= 1e29,
                    f"Target {target}: NK unreachable but GPU={gpu_dist[target]}")
                continue

            # Walk GPU predecessor
            path = []
            cur = target
            visited = set()
            for _ in range(size * size):
                path.append(cur)
                if cur == 0:
                    break
                if cur in visited:
                    break
                visited.add(cur)
                p = int(gpu_pred[cur])
                if p < 0:
                    break
                cur = p
            path.reverse()

            gpu_edge_cost = _edge_cost_of_path(
                path, from_nodes, to_nodes, costs)
            rel_diff = abs(gpu_edge_cost - nk_dist) / max(nk_dist, 1e-10)
            self.assertLessEqual(
                rel_diff, 0.01,
                f"Target {target}: NK={nk_dist:.6f}, "
                f"GPU={gpu_edge_cost:.6f}, rel_diff={rel_diff:.6f}")


@_needs_gpu_and_nk
class TestRasterGPUApiVsNetworkit(unittest.TestCase):
    """Validate RasterGPUAPI.shortest_path() against NetworkIt Dijkstra."""

    def setUp(self):
        from pyorps.graph.api.raster_gpu_api import RasterGPUAPI
        self.RasterGPUAPI = RasterGPUAPI
        self.steps_4 = np.array([
            [0, 1], [1, 0], [0, -1], [-1, 0]
        ], dtype=np.int8)

    def test_single_path(self):
        """RasterGPUAPI single source-target path matches NK."""
        rng = np.random.RandomState(42)
        size = 100
        raster = rng.randint(1, 100, (size, size), dtype=np.uint16)
        raster[rng.random((size, size)) < 0.1] = 65535
        raster[0, 0] = 5
        raster[size - 1, size - 1] = 5

        api = self.RasterGPUAPI(raster, self.steps_4, ignore_max=True)
        gpu_path = api.shortest_path(0, size * size - 1)

        graph, from_nodes, to_nodes, costs = _build_nk_graph(
            raster, self.steps_4)
        nk_dist, _ = _nk_dijkstra(graph, 0, size * size - 1)

        if nk_dist == float('inf'):
            self.assertEqual(len(gpu_path), 0)
            return

        self.assertGreater(len(gpu_path), 0)
        self.assertEqual(gpu_path[0], 0)
        self.assertEqual(gpu_path[-1], size * size - 1)

        gpu_cost = _edge_cost_of_path(gpu_path, from_nodes, to_nodes, costs)
        rel_diff = abs(gpu_cost - nk_dist) / max(nk_dist, 1e-10)
        self.assertLessEqual(rel_diff, 0.01,
                             f"NK={nk_dist:.6f}, GPU={gpu_cost:.6f}")

    def test_single_to_multi(self):
        """RasterGPUAPI single source to multiple targets."""
        rng = np.random.RandomState(88)
        size = 60
        raster = rng.randint(1, 100, (size, size), dtype=np.uint16)
        raster[0, 0] = 5

        targets = [size * size - 1, size - 1, (size - 1) * size]
        for t in targets:
            r, c = divmod(t, size)
            raster[r, c] = max(1, min(raster[r, c], 65534))

        api = self.RasterGPUAPI(raster, self.steps_4, ignore_max=True)
        gpu_paths = api.shortest_path(0, targets)

        graph, from_nodes, to_nodes, costs = _build_nk_graph(
            raster, self.steps_4)

        self.assertEqual(len(gpu_paths), len(targets))
        for i, target in enumerate(targets):
            nk_dist, _ = _nk_dijkstra(graph, 0, target)
            if nk_dist == float('inf'):
                self.assertEqual(len(gpu_paths[i]), 0)
                continue
            gpu_cost = _edge_cost_of_path(
                gpu_paths[i], from_nodes, to_nodes, costs)
            rel_diff = abs(gpu_cost - nk_dist) / max(nk_dist, 1e-10)
            self.assertLessEqual(
                rel_diff, 0.01,
                f"Target {target}: NK={nk_dist:.6f}, GPU={gpu_cost:.6f}")

    def test_multi_source_single_target(self):
        """RasterGPUAPI multiple sources to single target (symmetric)."""
        size = 50
        raster = np.full((size, size), 10, dtype=np.uint16)
        sources = [0, size - 1, (size - 1) * size]
        target = size * size - 1

        api = self.RasterGPUAPI(raster, self.steps_4, ignore_max=True)
        gpu_paths = api.shortest_path(sources, target)

        graph, from_nodes, to_nodes, costs = _build_nk_graph(
            raster, self.steps_4)

        self.assertEqual(len(gpu_paths), len(sources))
        for i, source in enumerate(sources):
            nk_dist, _ = _nk_dijkstra(graph, source, target)
            if nk_dist == float('inf'):
                continue
            gpu_cost = _edge_cost_of_path(
                gpu_paths[i], from_nodes, to_nodes, costs)
            rel_diff = abs(gpu_cost - nk_dist) / max(nk_dist, 1e-10)
            self.assertLessEqual(
                rel_diff, 0.01,
                f"Source {source}: NK={nk_dist:.6f}, GPU={gpu_cost:.6f}")

    def test_pairwise(self):
        """RasterGPUAPI pairwise source-target paths."""
        rng = np.random.RandomState(33)
        size = 50
        raster = rng.randint(1, 50, (size, size), dtype=np.uint16)
        sources = [0, size - 1, (size - 1) * size]
        targets = [size * size - 1, 0, size - 1]

        api = self.RasterGPUAPI(raster, self.steps_4, ignore_max=True)
        gpu_paths = api.shortest_path(sources, targets, pairwise=True)

        graph, from_nodes, to_nodes, costs = _build_nk_graph(
            raster, self.steps_4)

        self.assertEqual(len(gpu_paths), len(sources))
        for i in range(len(sources)):
            nk_dist, _ = _nk_dijkstra(graph, sources[i], targets[i])
            if nk_dist == float('inf'):
                continue
            gpu_cost = _edge_cost_of_path(
                gpu_paths[i], from_nodes, to_nodes, costs)
            rel_diff = abs(gpu_cost - nk_dist) / max(nk_dist, 1e-10)
            self.assertLessEqual(
                rel_diff, 0.01,
                f"Pair {i}: NK={nk_dist:.6f}, GPU={gpu_cost:.6f}")


if __name__ == "__main__":
    unittest.main()
