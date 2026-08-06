"""
GPU-accelerated graph API using raster-direct delta-stepping SSSP.

This backend extends GraphAPI directly (like CythonAPI) and operates on the
raster grid without constructing an explicit edge list or graph object.
Neighbors are computed on-the-fly during frontier expansion.

Requirements:
    - NVIDIA GPU with CUDA support
    - cupy-cuda12x >= 13.0.0

Usage:
    pf = PathFinder(..., graph_api="raster_gpu")
"""

from __future__ import annotations

from typing import Union, List, Optional, Any

import numpy as np
from numpy import ndarray
from numbers import Real

from pyorps.core.exceptions import (
    NoPathFoundError, AlgorithmNotImplementedError, PairwiseError
)
from pyorps.core.types import NodeList, NodePathList
from pyorps.graph.api.graph_api import GraphAPI

# Availability check — only depends on CuPy (no cuGraph/cuDF)
try:
    from pyorps.utils.sssp_gpu import sssp_raster_gpu
    from pyorps.utils.traversal_gpu import GPU_AVAILABLE as _gpu_flag
    RASTER_GPU_AVAILABLE = _gpu_flag
except ImportError:
    RASTER_GPU_AVAILABLE = False


class RasterGPUAPI(GraphAPI):
    """
    Graph API using raster-direct GPU delta-stepping SSSP.

    No edge list construction, no graph object. The raster IS the implicit
    graph — neighbor costs are computed on-the-fly during frontier expansion.
    """

    def __init__(
            self,
            raster_data: ndarray,
            steps: ndarray,
            ignore_max: bool = True,
            dem_data: Optional[ndarray] = None,
            delta: Union[float, str] = "auto",
            margin: float = 1.00001,
            gradient_luts=None,
            **kwargs,
    ):
        if not RASTER_GPU_AVAILABLE:
            raise ImportError(
                "Raster GPU backend requires CuPy with CUDA support. "
                "Install with: pip install cupy-cuda12x"
            )

        super().__init__(raster_data, steps, ignore_max, dem_data)

        # Per-edge gradient terms (feasibility plan): only active when the
        # slope-response LUT pair is provided — a bare dem_data keeps the
        # legacy behavior (it merely disables the symmetric-search trick).
        self.gradient_luts = gradient_luts
        if gradient_luts is not None and dem_data is None:
            raise ValueError(
                "gradient_luts require dem_data aligned to the raster")

        self._delta = delta
        self._margin = margin
        self._n_nodes = int(raster_data.shape[0] * raster_data.shape[1])

        # No graph construction — these are reported as zero
        self.edge_construction_time = 0.0
        self.graph_creation_time = 0.0
        self.graph = None

    def _run_sssp(
            self,
            source: int,
            target_indices: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run SSSP and return (dist, pred) as NumPy arrays."""
        targets = None
        if target_indices is not None:
            targets = np.asarray(target_indices, dtype=np.int32)

        dist, pred = sssp_raster_gpu(
            self.raster_data,
            self.steps,
            source_idx=int(source),
            delta=self._delta,
            ignore_max=self.ignore_max,
            target_indices=targets,
            margin=self._margin,
            return_predecessor=True,
            dem=self.dem_data if self.gradient_luts is not None else None,
            gradient_luts=self.gradient_luts,
        )
        return dist, pred

    def _extract_path(
            self,
            pred: np.ndarray,
            dist: np.ndarray,
            source: int,
            target: int,
    ) -> NodeList:
        """Reconstruct a single path from the predecessor array on CPU."""
        source = int(source)
        target = int(target)

        if target < 0 or target >= len(dist):
            raise NoPathFoundError(source=source, target=target)

        if np.isinf(dist[target]) or dist[target] >= 1e29:
            raise NoPathFoundError(source=source, target=target)

        # Walk predecessor chain: target → ... → source
        path = []
        current = target
        max_steps = self._n_nodes
        visited = set()

        while current != source and max_steps > 0:
            path.append(current)
            if current in visited:
                raise NoPathFoundError(source=source, target=target)
            visited.add(current)

            p = pred[current]
            if p < 0:
                raise NoPathFoundError(source=source, target=target)
            current = int(p)
            max_steps -= 1

        path.append(source)
        path.reverse()
        return path

    def _shortest_path_single_to_single(self, source_indices, target_indices):
        """Single source, single target."""
        target_arr = np.array([int(target_indices)], dtype=np.int32)
        dist, pred = self._run_sssp(source_indices, target_arr)
        return self._extract_path(pred, dist, source_indices,
                                  target_indices)

    def _shortest_path_single_to_multi(self, source_indices, target_indices):
        """Single source, multiple targets."""
        target_arr = np.asarray(target_indices, dtype=np.int32)
        dist, pred = self._run_sssp(source_indices, target_arr)
        paths = []
        for t in target_indices:
            try:
                paths.append(
                    self._extract_path(pred, dist, source_indices, t)
                )
            except NoPathFoundError:
                paths.append([])
        return paths

    def _shortest_path_multi_to_single(self, source_indices, target_indices):
        """Multiple sources, single target."""
        # For symmetric costs (no DEM): single SSSP from target, reverse
        if self.dem_data is None:
            source_arr = np.asarray(source_indices, dtype=np.int32)
            dist, pred = self._run_sssp(target_indices, source_arr)
            paths = []
            for s in source_indices:
                try:
                    path = self._extract_path(
                        pred, dist, target_indices, s
                    )
                    paths.append(path[::-1])
                except NoPathFoundError:
                    paths.append([])
            return paths
        # Asymmetric costs: per-source SSSP
        paths = []
        target_arr = np.array([int(target_indices)], dtype=np.int32)
        for source in source_indices:
            dist, pred = self._run_sssp(source, target_arr)
            try:
                paths.append(
                    self._extract_path(
                        pred, dist, source, target_indices
                    )
                )
            except NoPathFoundError:
                paths.append([])
        return paths

    def _shortest_path_multi_to_multi(self, source_indices, target_indices,
                                      pairwise):
        """Multiple sources, multiple targets."""
        if pairwise:
            if len(source_indices) != len(target_indices):
                raise PairwiseError()
            paths = []
            for source, target in zip(source_indices, target_indices):
                target_arr = np.array([int(target)], dtype=np.int32)
                dist, pred = self._run_sssp(source, target_arr)
                try:
                    paths.append(
                        self._extract_path(pred, dist, source, target)
                    )
                except NoPathFoundError:
                    paths.append([])
            return paths
        # All pairs: one SSSP per source, extract all targets
        target_arr = np.asarray(target_indices, dtype=np.int32)
        paths = []
        for source in source_indices:
            dist, pred = self._run_sssp(source, target_arr)
            for target in target_indices:
                try:
                    paths.append(
                        self._extract_path(pred, dist, source, target)
                    )
                except NoPathFoundError:
                    paths.append([])
        return paths

    def shortest_path(
            self,
            source_indices: Union[int, List[int], ndarray],
            target_indices: Union[int, List[int], ndarray],
            algorithm: str = "delta-stepping",
            **kwargs,
    ) -> Union[NodeList, NodePathList]:
        """
        Find shortest path(s) using raster-direct GPU delta-stepping.

        Parameters:
            source_indices: Source node index/indices
            target_indices: Target node index/indices
            algorithm: Only "delta-stepping" and "dijkstra" are accepted
            **kwargs: pairwise (bool) for pairwise computation

        Returns:
            Path or list of paths as node index lists
        """
        if algorithm.lower() not in ("delta-stepping", "dijkstra"):
            raise AlgorithmNotImplementedError(
                algorithm, graph_library="raster_gpu"
            )

        source_has_len = hasattr(source_indices, "__len__")
        target_has_len = hasattr(target_indices, "__len__")

        if not source_has_len and not target_has_len:
            return self._shortest_path_single_to_single(
                source_indices, target_indices)
        if not source_has_len and target_has_len:
            return self._shortest_path_single_to_multi(
                source_indices, target_indices)
        if source_has_len and not target_has_len:
            return self._shortest_path_multi_to_single(
                source_indices, target_indices)
        return self._shortest_path_multi_to_multi(
            source_indices, target_indices, kwargs.get("pairwise", False))
