"""Constrained path finder with infrastructure-aware routing."""

import math
import time
import warnings

import numpy as np
from shapely.geometry import Point, LineString

from pyorps.core.constrained_path import Tower, ConstrainedPath
from pyorps.core.infrastructure_profile import InfrastructureProfile
from pyorps.graph.path_finder import PathFinder


class ConstrainedPathFinder(PathFinder):
    """PathFinder extension for infrastructure-constrained routing.

    Supports turn-angle limits and intermediate structure (tower) placement
    via coupled extended-state Dijkstra. Inherits all PathFinder parameters
    and capabilities.

    Uses exact span tracking (float) to avoid quantization drift when
    span_bin_size > step_distance, enabling correct coupled route+tower
    optimization at any raster resolution.
    """

    SUPPORTED_BACKENDS = ("cython", "cython_parallel", "raster_gpu",
                          "raster_gpu_v3", "raster_gpu_v4")

    def __init__(self, dataset_source, source_coords, target_coords,
                 profile, graph_api="cython", neighborhood_str="r2",
                 dsm=None, **kwargs):
        """Initialize constrained path finder.

        Parameters:
            dataset_source: Raster or vector data source (same as PathFinder).
            source_coords: Source coordinates (same as PathFinder).
            target_coords: Target coordinates (same as PathFinder).
            profile: InfrastructureProfile, dict, or path to YAML/JSON.
            graph_api: "cython" or "raster_gpu".
            neighborhood_str: Neighborhood string (e.g. "r1", "r2").
            dsm: Digital Surface Model (includes trees/buildings). When both
                 dem and dsm are provided, obstacle_heights = dsm - dem is
                 computed automatically for clearance checking.
            **kwargs: All other PathFinder parameters (search_space_buffer_m,
                      cost_assumptions, dem, etc.).
        """
        if graph_api not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"ConstrainedPathFinder requires graph_api in "
                f"{self.SUPPORTED_BACKENDS}, got '{graph_api}'"
            )

        # Load profile before super().__init__ so it's available if needed
        if isinstance(profile, str):
            self._profile = InfrastructureProfile.load(profile)
        elif isinstance(profile, dict):
            self._profile = InfrastructureProfile.from_dict(profile)
        else:
            self._profile = profile

        # Store the requested backend for constrained routing
        self._constrained_backend = graph_api
        self._dsm_source = dsm

        # Initialize parent — sets up raster_handler, steps, etc.
        # Both "cython" and "raster_gpu" produce directed steps in parent
        super().__init__(
            dataset_source=dataset_source,
            source_coords=source_coords,
            target_coords=target_coords,
            graph_api=graph_api,
            neighborhood_str=neighborhood_str,
            **kwargs,
        )

        # Load DSM and compute obstacle heights (DSM - DEM)
        self._obstacle_data = None
        self._load_obstacle_data()

        # Feasibility objective integration (plan Phase 8): the metric
        # pipeline already produced the combined weight raster through the
        # parent; here we validate what the constrained kernels support
        # and apply the reserved "smoothness"/"tower" weights host-side.
        self._tower_cost_raster = None
        if self.objective is not None:
            if self.objective.has_gradient_terms:
                raise NotImplementedError(
                    "Constrained routing keeps the PROFILE's gradient "
                    "model (gradient_cost_params / max_gradient_percent); "
                    "objective gradient terms are not wired into the "
                    "constrained kernels yet.")
            if self._constrained_backend != "cython":
                raise NotImplementedError(
                    "Feasibility objectives on constrained routing are "
                    "supported on the 'cython' backend (per-cell tower "
                    "costs); parallel/GPU constrained variants are "
                    "pending.")
            if self._profile.has_clearance_constraints:
                raise NotImplementedError(
                    "Feasibility objectives with clearance constraints "
                    "are not wired yet (the clearance kernels lack the "
                    "per-cell tower-cost raster).")
            if self.dem_dataset is not None:
                warnings.warn(
                    "Constrained routing with a DEM uses the profile's "
                    "gradient model, not the objective's 3D stretch / "
                    "response LUTs.", UserWarning, stacklevel=2)

        # Precompute LUTs using parent's steps
        self._angle_cost_lut, self._angle_valid_lut = (
            self._profile.precompute_angle_lut(self.steps)
        )
        cell_size = abs(self.raster_handler.window_transform.a)
        self._cell_size = cell_size
        self._step_distances = self._profile.compute_step_distances(
            self.steps, cell_size
        )

        # Coarsen span bins for coupled Dijkstra performance.
        # Since exact span is tracked via float (span_dist), bins are only
        # used for state deduplication — not for span enforcement.
        # Using min_span as bin size gives just ceil(max/min) bins (typically
        # 2) instead of ceil(max/bin_size) (typically 40). This reduces the
        # state space ~20x, often the difference between sparse and dense.
        if self._profile.has_span_constraints:
            effective_bin = self._profile.min_span_m
            self._effective_span_bin_size = effective_bin
            self._effective_n_span_bins = max(
                2, math.ceil(self._profile.max_span_m / effective_bin)
            )
        else:
            self._effective_span_bin_size = 1e6
            self._effective_n_span_bins = 1

        # Rotated square offsets for "exact" area cost mode (set for all
        # profiles so backends can rely on the attributes existing)
        self._area_offsets = None
        self._area_offset_starts = None
        self._area_offset_counts = None

        if self._profile.has_span_constraints:
            self._tower_terrain_costs = (
                self._profile.precompute_tower_terrain_costs()
            )
            self._tower_angle_costs = (
                self._profile.precompute_tower_angle_costs(self.steps)
            )
            if (self._profile.tower_area_cost_mode == "exact"
                    and self._profile.tower_ground_area_m2 > 1.0):
                self._precompute_area_offsets()
        else:
            self._tower_terrain_costs = np.zeros(65536, dtype=np.float64)
            self._tower_angle_costs = np.zeros(
                (len(self.steps), len(self.steps)), dtype=np.float64
            )

        # Reserved objective weights: "smoothness" scales the turn-angle
        # penalty LUT, "tower" scales the tower cost terms. Default 1.0
        # (the profile LUTs ARE the steering magnitudes); explicit 0
        # disables the term. Hard limits (angle_valid_lut) are
        # constraints, never scaled.
        if self.objective is not None:
            w_smooth = self.objective.weights.get("smoothness", 1.0)
            w_tower = self.objective.weights.get("tower", 1.0)
            if w_smooth != 1.0:
                self._angle_cost_lut = self._angle_cost_lut.copy()
                finite = np.isfinite(self._angle_cost_lut)
                self._angle_cost_lut[finite] *= w_smooth
            if w_tower != 1.0:
                self._tower_terrain_costs = (
                    self._tower_terrain_costs * w_tower)
                self._tower_angle_costs = self._tower_angle_costs * w_tower

    def _load_obstacle_data(self):
        """Load DSM and compute obstacle heights (DSM - DEM).

        When both a DSM source and a DEM raster handler are available,
        computes obstacle_heights = dsm - dem (trees, buildings) and
        resamples both to match the cost raster shape.
        """
        if (self._dsm_source is not None and
                hasattr(self, 'dem_raster_handler') and
                self.dem_raster_handler is not None):
            from pyorps.io.geo_dataset import (
                initialize_geo_dataset, RasterDataset)
            from pyorps.raster.handler import RasterHandler
            dsm_dataset = initialize_geo_dataset(
                self._dsm_source, crs=self.dataset.crs)
            if isinstance(dsm_dataset, RasterDataset):
                dsm_dataset.load_data()
                dsm_handler = RasterHandler(
                    dsm_dataset,
                    self.source_coords, self.target_coords,
                    self.search_space_buffer_m, apply_mask=False)
                dsm_raw = dsm_handler.data
                if len(dsm_raw.shape) > 2:
                    dsm_raw = dsm_raw[0]
                dem_raw = self.dem_raster_handler.data
                if len(dem_raw.shape) > 2:
                    dem_raw = dem_raw[0]
                # Resample both to cost raster shape
                target_shape = self.raster_handler.data[0].shape
                if dsm_raw.shape != target_shape:
                    from scipy.ndimage import zoom
                    zf = (target_shape[0] / dsm_raw.shape[0],
                          target_shape[1] / dsm_raw.shape[1])
                    dsm_raw = zoom(dsm_raw, zf, order=1)
                if dem_raw.shape != target_shape:
                    from scipy.ndimage import zoom
                    zf = (target_shape[0] / dem_raw.shape[0],
                          target_shape[1] / dem_raw.shape[1])
                    dem_raw = zoom(dem_raw, zf, order=1)
                # Obstacle heights = surface - ground (trees, buildings)
                obstacle = (dsm_raw.astype(np.float32) -
                            dem_raw.astype(np.float32))
                obstacle[obstacle < 0] = 0  # clamp negatives
                self._obstacle_data = obstacle

    def _precompute_area_offsets(self):
        """Precompute rotated square pixel offsets per direction pair.

        For each (d_in, d_out) pair, computes the set of (dr, dc) pixel
        offsets that fall within a square of side sqrt(ground_area_m2),
        rotated to bisect the incoming and outgoing direction angles.

        Results are packed into flat arrays for Cython consumption:
        - area_offsets: int32 flat [dr0, dc0, dr1, dc1, ...] all pairs
        - area_offset_starts: int32[n_dirs*n_dirs] start index per pair
        - area_offset_counts: int32[n_dirs*n_dirs] offset count per pair
        """
        side_m = math.sqrt(self._profile.tower_ground_area_m2)
        half_side_px = side_m / (2.0 * self._cell_size)

        n_dirs = len(self.steps)
        angles = [math.atan2(float(dc), float(dr))
                  for dr, dc in self.steps]

        all_offsets = []
        starts = np.zeros(n_dirs * n_dirs, dtype=np.int32)
        counts = np.zeros(n_dirs * n_dirs, dtype=np.int32)

        # Search radius: diagonal of the square in pixels
        r_max = int(math.ceil(half_side_px * math.sqrt(2))) + 1

        for d_in in range(n_dirs):
            for d_out in range(n_dirs):
                # Bisector angle
                bisector = (angles[d_in] + angles[d_out]) / 2.0
                diff = angles[d_out] - angles[d_in]
                if abs(math.atan2(math.sin(diff), math.cos(diff))) > math.pi:
                    bisector += math.pi

                cos_b = math.cos(-bisector)
                sin_b = math.sin(-bisector)
                offsets = []
                for dr in range(-r_max, r_max + 1):
                    for dc in range(-r_max, r_max + 1):
                        # Inverse rotate to check if in unit square
                        local_r = dr * cos_b - dc * sin_b
                        local_c = dr * sin_b + dc * cos_b
                        if (abs(local_r) <= half_side_px
                                and abs(local_c) <= half_side_px):
                            offsets.append((dr, dc))

                pair_idx = d_in * n_dirs + d_out
                starts[pair_idx] = len(all_offsets)
                counts[pair_idx] = len(offsets)
                all_offsets.extend(offsets)

        if all_offsets:
            flat = np.array(all_offsets, dtype=np.int32).flatten()
        else:
            flat = np.zeros(0, dtype=np.int32)
        self._area_offsets = flat
        self._area_offset_starts = starts
        self._area_offset_counts = counts

    @property
    def profile(self):
        """The infrastructure profile used for constrained routing."""
        return self._profile

    def find_route(self, source=None, target=None, **kwargs):
        """Find constrained optimal route with tower placement.

        Parameters:
            source: Override source coordinates (default: use init coords).
            target: Override target coordinates (default: use init coords).

        Returns:
            ConstrainedPath with towers, cost breakdown, and geometry.
        """
        t_start = time.time()

        # Resolve coordinates (allow overrides like parent)
        src = source if source is not None else self.source_coords
        tgt = target if target is not None else self.target_coords

        if src is None or tgt is None:
            raise ValueError("Source and target coordinates must not be None!")

        src = PathFinder.normalize_coordinates(src)
        tgt = PathFinder.normalize_coordinates(tgt)

        # Convert CRS coordinates to local raster row/col
        src_indices = self.raster_handler.coords_to_indices([src])
        tgt_indices = self.raster_handler.coords_to_indices([tgt])
        source_row, source_col = int(src_indices[0][0]), int(src_indices[0][1])
        target_row, target_col = int(tgt_indices[0][0]), int(tgt_indices[0][1])

        # Get raster data (first band, windowed section)
        raster = self.raster_handler.data[0]

        backend = self._constrained_backend

        # Per-cell tower-cost raster (objective mode): tower foundation
        # costs come from the LAND USE (the stack's cost layer), not from
        # the combined feasibility values in the search raster.
        self._tower_cost_raster = self._build_tower_cost_raster()

        # Coupled approach: full extended-state Dijkstra
        route_result = self._find_route_coupled(
            raster, source_row, source_col, target_row, target_col,
            backend,
        )

        t_pathfinding = time.time() - t_start

        # Handle 3-tuple (variable height) or 2-tuple (fixed height) returns
        if len(route_result) == 3:
            path_indices, tower_cell_indices, tower_heights_arr = route_result
        else:
            path_indices, tower_cell_indices = route_result
            tower_heights_arr = None

        result = self._build_constrained_path(
            path_indices, tower_cell_indices, raster, src, tgt,
            t_pathfinding, tower_heights_arr=tower_heights_arr,
        )

        # Honest per-metric reporting (Phase 6 evaluator) in objective mode
        if (self.objective is not None and self.metric_stack is not None
                and len(result.path_indices) > 1):
            idx = np.asarray(result.path_indices, dtype=np.int64)
            cols_n = raster.shape[1]
            self._evaluate_objective_metrics(idx // cols_n, idx % cols_n,
                                             result)

        # Store in parent's PathCollection so save_paths() works
        self.paths.add(result)

        return result

    def _build_tower_cost_raster(self):
        """Per-cell tower foundation costs from the stack's cost layer.

        ``tower_cost[cell] = LUT[round(cost_layer[cell])]`` (the LUT is
        already "tower"-weight-scaled); forbidden cells carry INFINITY.
        Without an objective (or without span constraints) returns None —
        the kernels then use the legacy value-keyed LUT, bit-identically.
        """
        if self.objective is None or self.metric_stack is None:
            return None
        if not self._profile.has_span_constraints:
            return None
        sub = self.metric_stack.window(self.raster_handler.window)
        cost_layer = sub["cost"]
        indices = np.clip(np.rint(cost_layer), 0, 65535).astype(np.int64)
        tower = self._tower_terrain_costs[indices].astype(np.float32)
        tower[sub.forbidden_mask] = np.inf
        return tower

    def _prepare_dem_data(self, raster):
        """Extract and resample DEM data, build dem_kwargs.

        Returns:
            Tuple of (dem_data, dem_kwargs) where dem_data is the resampled
            DEM array (or None) and dem_kwargs is a dict of gradient params.
        """
        dem_data = None
        if hasattr(self, 'dem_raster_handler') and self.dem_raster_handler is not None:
            dem_raw = self.dem_raster_handler.data
            if len(dem_raw.shape) > 2:
                dem_data = dem_raw[0].astype(np.float32)
            else:
                dem_data = dem_raw.astype(np.float32)
            # Resample DEM to match cost raster shape (resolutions may differ)
            target_shape = (raster.shape[0], raster.shape[1])
            if dem_data.shape != target_shape:
                from scipy.ndimage import zoom
                zoom_factors = (target_shape[0] / dem_data.shape[0],
                                target_shape[1] / dem_data.shape[1])
                dem_data = zoom(dem_data, zoom_factors, order=1).astype(
                    np.float32)

        # Gradient parameters from profile
        dem_kwargs = {}
        if dem_data is not None:
            dem_kwargs['dem_data'] = dem_data
            dem_kwargs['cell_size'] = self._cell_size
            dem_kwargs['max_gradient_pct'] = (
                self._profile.max_gradient_percent
                if self._profile.max_gradient_percent is not None
                else 100.0
            )
            dem_kwargs['gradient_scale'] = (
                self._profile.gradient_cost_params.get('scale', 2.0)
                if self._profile.gradient_cost_params is not None
                else 2.0
            )

        return dem_data, dem_kwargs

    def _build_height_arrays(self):
        """Build heights and premiums arrays from profile.

        Returns:
            Tuple of (heights, premiums) as float32 numpy arrays.
        """
        if self._profile.has_variable_height:
            heights = np.array(
                self._profile.effective_tower_heights_m,
                dtype=np.float32)
            premiums = self._profile.precompute_height_premium()
        else:
            h = self._profile.tower_height_m or 25.0
            heights = np.array([h], dtype=np.float32)
            premiums = np.array([0.0], dtype=np.float32)
        return heights, premiums

    def _build_gpu_kwargs(self, raster, source_row, source_col,
                          target_row, target_col, n_span_bins,
                          span_bin_size, min_span, max_span,
                          heights, premiums):
        """Build the common gpu_kwargs dict shared by GPU backends.

        Returns:
            Dict of keyword arguments for GPU constrained SSSP functions.
        """
        return dict(
            raster=raster,
            source_row=source_row, source_col=source_col,
            target_row=target_row, target_col=target_col,
            steps=self.steps,
            angle_cost_lut=self._angle_cost_lut.astype(np.float32),
            angle_valid_lut=self._angle_valid_lut.astype(np.uint8),
            step_distances=self._step_distances.astype(np.float32),
            tower_terrain_costs=self._tower_terrain_costs.astype(
                np.float32),
            tower_angle_costs=self._tower_angle_costs.astype(
                np.float32),
            n_span_bins=n_span_bins, span_bin_size=span_bin_size,
            min_span=min_span, max_span=max_span,
            tower_heights=heights,
            height_premiums=premiums,
            n_heights=len(heights),
        )

    def _add_dem_to_kwargs(self, gpu_kwargs, dem_data, dem_kwargs):
        """Add DEM/clearance/gradient/obstacle params to gpu_kwargs.

        Mutates and returns gpu_kwargs.
        """
        if dem_data is not None:
            gpu_kwargs.update(
                dem=dem_data,
                cell_size=self._cell_size,
                conductor_weight_per_m=(
                    self._profile.conductor_weight_per_m or 0.0),
                conductor_tension=(
                    self._profile.conductor_tension_n or 1.0),
                min_clearance=(
                    self._profile.min_clearance_m or 0.0),
                max_gradient_pct=dem_kwargs.get(
                    'max_gradient_pct', 100.0),
                gradient_scale=dem_kwargs.get(
                    'gradient_scale', 2.0),
            )
            if self._obstacle_data is not None:
                gpu_kwargs['obstacle_heights'] = self._obstacle_data
        return gpu_kwargs

    def _add_area_to_kwargs(self, gpu_kwargs):
        """Add area_offsets if present to gpu_kwargs.

        Mutates and returns gpu_kwargs.
        """
        if self._area_offsets is not None:
            gpu_kwargs.update(
                area_offsets=self._area_offsets,
                area_offset_starts=self._area_offset_starts,
                area_offset_counts=self._area_offset_counts,
            )
        return gpu_kwargs

    def _run_gpu_backend(self, gpu_func, raster, source_row, source_col,
                         target_row, target_col, n_span_bins, span_bin_size,
                         min_span, max_span, dem_data, dem_kwargs,
                         extra_kwargs=None):
        """Run a GPU backend with standard kwargs setup.

        Returns:
            Result from the GPU function call.
        """
        heights, premiums = self._build_height_arrays()
        gpu_kwargs = self._build_gpu_kwargs(
            raster, source_row, source_col, target_row, target_col,
            n_span_bins, span_bin_size, min_span, max_span,
            heights, premiums,
        )
        if extra_kwargs:
            gpu_kwargs.update(extra_kwargs)
        self._add_dem_to_kwargs(gpu_kwargs, dem_data, dem_kwargs)
        self._add_area_to_kwargs(gpu_kwargs)
        return gpu_func(**gpu_kwargs)

    def _run_cython_backend(self, raster, source_row, source_col,
                            target_row, target_col, n_span_bins,
                            span_bin_size, min_span, max_span,
                            dem_data, dem_kwargs, backend):
        """Run the Cython backend for constrained routing.

        Dispatches to clearance-aware or basic algorithm depending on
        whether DEM data and clearance constraints are present.

        Returns:
            Result tuple from the Cython algorithm.
        """
        from pyorps.utils.constrained_path_algorithms import (
            constrained_dijkstra_2d,
            constrained_delta_stepping_2d,
            constrained_delta_stepping_height_2d,
            constrained_delta_stepping_lazy,
        )

        # Helper: build clearance kwargs and dispatch to dense or lazy.
        # Uses constrained_delta_stepping_height_2d (which auto-selects
        # dense vs sparse based on state space size) for all clearance
        # calls, wrapping single-height into 1-element arrays.
        # Area cost offset arrays (None when not using "exact" mode)
        area_kwargs = {}
        if self._area_offsets is not None:
            area_kwargs = dict(
                area_offsets=self._area_offsets,
                area_offset_starts=self._area_offset_starts,
                area_offset_counts=self._area_offset_counts,
            )

        def _run_clearance(height_arr, premium_arr, obs_data):
            kwargs = dict(
                raster=raster,
                source_row=source_row, source_col=source_col,
                target_row=target_row, target_col=target_col,
                steps=self.steps,
                angle_cost_lut=self._angle_cost_lut.astype(np.float32),
                angle_valid_lut=self._angle_valid_lut.astype(np.uint8),
                step_distances=self._step_distances.astype(np.float32),
                tower_terrain_costs=self._tower_terrain_costs.astype(
                    np.float32),
                tower_angle_costs=self._tower_angle_costs.astype(np.float32),
                n_span_bins=n_span_bins, span_bin_size=span_bin_size,
                min_span=min_span, max_span=max_span,
                dem_data=dem_data,
                cell_size=self._cell_size,
                tower_heights=height_arr,
                height_premiums=premium_arr,
                conductor_weight_per_m=self._profile.conductor_weight_per_m,
                conductor_tension=self._profile.conductor_tension_n,
                min_clearance_val=self._profile.min_clearance_m,
                max_gradient_pct=dem_kwargs['max_gradient_pct'],
                gradient_scale=dem_kwargs['gradient_scale'],
                **area_kwargs,
            )
            if obs_data is not None:
                kwargs['obstacle_heights'] = obs_data

            n_h = len(height_arr)
            total_states = (raster.shape[0] * raster.shape[1]
                            * len(self.steps) * n_span_bins * n_h)
            if total_states > 500_000_000:
                return constrained_delta_stepping_lazy(**kwargs)
            return constrained_delta_stepping_height_2d(**kwargs)

        # Variable-height clearance: try fixed (base) height first.
        # If base height passes all clearances, that result is already optimal
        # (height premiums can only add cost). Only run the expensive
        # variable-height search when fixed height fails to find a path.
        if (dem_data is not None and self._profile.has_clearance_constraints
                and self._profile.has_variable_height):
            base_height = min(self._profile.effective_tower_heights_m)
            fixed_result = _run_clearance(
                np.array([base_height], dtype=np.float32),
                np.array([0.0], dtype=np.float32),
                self._obstacle_data,
            )
            # If fixed-height found a path, it's optimal — skip variable search
            if len(fixed_result[0]) > 0:
                return fixed_result

            # Fixed height failed — run variable-height search
            heights = np.array(
                self._profile.effective_tower_heights_m, dtype=np.float32)
            premiums = self._profile.precompute_height_premium()
            return _run_clearance(heights, premiums, self._obstacle_data)

        # Fixed-height clearance algorithm
        if (dem_data is not None and self._profile.has_clearance_constraints):
            effective_height = self._profile.tower_height_m
            if (self._profile.tower_heights_m is not None and
                    len(self._profile.tower_heights_m) == 1):
                effective_height = self._profile.tower_heights_m[0]
            return _run_clearance(
                np.array([effective_height], dtype=np.float32),
                np.array([0.0], dtype=np.float32),
                self._obstacle_data,
            )

        algo_func = (constrained_delta_stepping_2d if backend == "cython_parallel"
                     else constrained_dijkstra_2d)

        tower_kwargs = {}
        if (self._tower_cost_raster is not None
                and backend != "cython_parallel"):
            tower_kwargs["tower_cost_raster"] = self._tower_cost_raster

        return algo_func(
            raster=raster,
            source_row=source_row, source_col=source_col,
            target_row=target_row, target_col=target_col,
            steps=self.steps,
            angle_cost_lut=self._angle_cost_lut.astype(np.float32),
            angle_valid_lut=self._angle_valid_lut.astype(np.uint8),
            step_distances=self._step_distances.astype(np.float32),
            tower_terrain_costs=self._tower_terrain_costs.astype(np.float32),
            tower_angle_costs=self._tower_angle_costs.astype(np.float32),
            n_span_bins=n_span_bins, span_bin_size=span_bin_size,
            min_span=min_span, max_span=max_span,
            **dem_kwargs,
            **area_kwargs,
            **tower_kwargs,
        )

    def _find_route_coupled(self, raster, source_row, source_col,
                            target_row, target_col, backend):
        """Full coupled Dijkstra with span bins in the state."""
        n_span_bins = self._effective_n_span_bins
        span_bin_size = self._effective_span_bin_size
        min_span = (self._profile.min_span_m
                    if self._profile.has_span_constraints else 0.0)
        max_span = (self._profile.max_span_m
                    if self._profile.has_span_constraints else 1e6)

        # Extract DEM data if available (aligned via parent's dem_raster_handler)
        dem_data, dem_kwargs = self._prepare_dem_data(raster)

        if backend == "raster_gpu":
            try:
                from pyorps.utils.constrained_sssp_gpu_v2 import (
                    constrained_sssp_raster_gpu_v2,
                )
                return self._run_gpu_backend(
                    constrained_sssp_raster_gpu_v2,
                    raster, source_row, source_col, target_row, target_col,
                    n_span_bins, span_bin_size, min_span, max_span,
                    dem_data, dem_kwargs, extra_kwargs={'sparse': 'auto'},
                )
            except (ImportError, RuntimeError) as e:
                warnings.warn(
                    f"GPU v2 unavailable ({e}), falling back to Cython")
                backend = "cython"

        if backend == "raster_gpu_v3":
            try:
                from pyorps.utils.constrained_sssp_gpu_v3 import (
                    constrained_sssp_raster_gpu_v3,
                )
                return self._run_gpu_backend(
                    constrained_sssp_raster_gpu_v3,
                    raster, source_row, source_col, target_row, target_col,
                    n_span_bins, span_bin_size, min_span, max_span,
                    dem_data, dem_kwargs,
                    extra_kwargs={'max_visited_fraction': 1.0},
                )
            except (ImportError, RuntimeError) as e:
                warnings.warn(
                    f"GPU v3 unavailable ({e}), falling back to Cython")
                backend = "cython"

        if backend == "raster_gpu_v4":
            try:
                from pyorps.utils.constrained_sssp_gpu_v4 import (
                    constrained_sssp_raster_gpu_v4,
                )
                return self._run_gpu_backend(
                    constrained_sssp_raster_gpu_v4,
                    raster, source_row, source_col, target_row, target_col,
                    n_span_bins, span_bin_size, min_span, max_span,
                    dem_data, dem_kwargs,
                    extra_kwargs={'max_visited_fraction': 0.4},
                )
            except (ImportError, RuntimeError) as e:
                warnings.warn(
                    f"GPU v4 unavailable ({e}), falling back to Cython")
                backend = "cython"

        return self._run_cython_backend(
            raster, source_row, source_col, target_row, target_col,
            n_span_bins, span_bin_size, min_span, max_span,
            dem_data, dem_kwargs, backend,
        )

    def _enforce_terminal_tower_spans(self, path_indices, tower_cell_indices,
                                       tower_set, min_span_m, idx_to_coord):
        """Enforce minimum span at terminal towers (start/end).

        Checks if the nearest Dijkstra tower is too close to the start
        or end of the path and removes it if so. Adds start/end cells
        as terminal towers.

        Returns:
            Tuple of (tower_set, terminal_cells) where tower_set is the
            updated set of tower cell indices and terminal_cells is the
            set of start/end cell indices.
        """
        terminal_cells = set()
        start_cell = int(path_indices[0])
        end_cell = int(path_indices[-1])
        start_coord = idx_to_coord(start_cell)
        end_coord = idx_to_coord(end_cell)

        # Check if nearest Dijkstra tower is too close to start
        for tc in tower_cell_indices:
            tc_int = int(tc)
            if tc_int == start_cell:
                continue
            tc_coord = idx_to_coord(tc_int)
            dist = math.sqrt((tc_coord[0] - start_coord[0])**2 +
                             (tc_coord[1] - start_coord[1])**2)
            if dist < min_span_m:
                tower_set.discard(tc_int)
            break  # only check the first non-start tower

        # Check if nearest Dijkstra tower is too close to end
        for tc in reversed(tower_cell_indices):
            tc_int = int(tc)
            if tc_int == end_cell:
                continue
            tc_coord = idx_to_coord(tc_int)
            dist = math.sqrt((tc_coord[0] - end_coord[0])**2 +
                             (tc_coord[1] - end_coord[1])**2)
            if dist < min_span_m:
                tower_set.discard(tc_int)
            break  # only check the last non-end tower

        tower_set.add(start_cell)
        tower_set.add(end_cell)
        terminal_cells.add(start_cell)
        terminal_cells.add(end_cell)

        return tower_set, terminal_cells

    def _compute_tower_angle_and_bisector(self, pos, path_indices, x, y,
                                          idx_to_coord):
        """Compute turn angle and bisector at a tower position.

        Returns:
            Tuple of (turn_deg, bisector_rad).
        """
        turn_deg = 0.0
        bisector_rad = 0.0
        if 0 < pos < len(path_indices) - 1:
            prev = idx_to_coord(path_indices[pos - 1])
            curr = (x, y)
            nxt = idx_to_coord(path_indices[pos + 1])
            d_in = math.atan2(curr[1] - prev[1], curr[0] - prev[0])
            d_out = math.atan2(nxt[1] - curr[1], nxt[0] - curr[0])
            turn_deg = abs(math.degrees(
                math.atan2(math.sin(d_out - d_in), math.cos(d_out - d_in))
            ))
            # Bisector angle: halfway between incoming and outgoing
            bisector_rad = (d_in + d_out) / 2.0
            # Ensure bisector is on the correct side (acute bisector)
            if abs(d_out - d_in) > math.pi:
                bisector_rad += math.pi
        elif pos == 0 and len(path_indices) > 1:
            # Terminal start: align with outgoing segment
            nxt = idx_to_coord(path_indices[pos + 1])
            bisector_rad = math.atan2(nxt[1] - y, nxt[0] - x)
        elif pos == len(path_indices) - 1 and len(path_indices) > 1:
            # Terminal end: align with incoming segment
            prev = idx_to_coord(path_indices[pos - 1])
            bisector_rad = math.atan2(y - prev[1], x - prev[0])
        return turn_deg, bisector_rad

    def _classify_tower_type(self, is_terminal, turn_deg):
        """Determine tower type and angle cost from angle.

        Terminal towers are always classified as "terminal".
        Non-terminal towers are classified based on angle_types config.

        Returns:
            Tuple of (tower_type, angle_cost).
        """
        tower_type = "suspension"
        angle_cost = 0.0
        if is_terminal:
            tower_type = "terminal"
            angle_cost = (self._profile.terminal_tower_cost
                          if self._profile.terminal_tower_cost > 0
                          else 0.0)
        else:
            angle_types = self._profile.tower_cost_params.get(
                "angle_types", {})
            if angle_types:
                sorted_types = sorted(
                    angle_types.items(),
                    key=lambda t: t[1]["max_angle_deg"],
                )
                for tname, tconfig in sorted_types:
                    if turn_deg <= tconfig["max_angle_deg"]:
                        tower_type = tname
                        angle_cost = tconfig["base_cost"]
                        break
        return tower_type, angle_cost

    def _resolve_tower_height(self, is_terminal, cell_idx, tower_height_map):
        """Resolve the height for a tower.

        Terminal towers use terminal_tower_height_m if available.
        Intermediate towers use algorithm-selected height or fixed height.

        Returns:
            Tower height in meters (float or None).
        """
        if is_terminal and self._profile.terminal_tower_height_m is not None:
            return self._profile.terminal_tower_height_m

        tower_h = tower_height_map.get(cell_idx, None)
        # Fallback: use base height from profile
        if tower_h is None and self._profile.has_variable_height:
            tower_h = min(self._profile.effective_tower_heights_m)
        elif tower_h is None and self._profile.tower_height_m is not None:
            tower_h = self._profile.tower_height_m
        return tower_h

    def _build_tower_object(self, pos, path_indices, tower_positions,
                            tower_set, tower_height_map, terminal_cells,
                            ncols, raster, idx_to_coord, tid):
        """Build a single Tower from position data.

        Handles angle computation, type classification, span computation,
        and height selection.

        Returns:
            Tower instance.
        """
        cell_idx = int(path_indices[pos])
        x, y = idx_to_coord(cell_idx)
        r, c = cell_idx // ncols, cell_idx % ncols
        terrain_cost = float(self._tower_terrain_costs[raster[r, c]])

        # Compute turn angle and bisector at tower
        turn_deg, bisector_rad = self._compute_tower_angle_and_bisector(
            pos, path_indices, x, y, idx_to_coord)

        # Terminal towers are dead-end structures at start/end
        is_terminal = cell_idx in terminal_cells

        # Determine tower type from angle (terminals are always dead-end)
        tower_type, angle_cost = self._classify_tower_type(
            is_terminal, turn_deg)

        # Spans to previous/next tower
        span_prev = None
        if tid > 0:
            prev_pos = tower_positions[tid - 1]
            prev_coord = idx_to_coord(path_indices[prev_pos])
            span_prev = math.sqrt(
                (x - prev_coord[0])**2 + (y - prev_coord[1])**2
            )

        span_next = None
        if tid < len(tower_positions) - 1:
            next_pos = tower_positions[tid + 1]
            next_coord = idx_to_coord(path_indices[next_pos])
            span_next = math.sqrt(
                (x - next_coord[0])**2 + (y - next_coord[1])**2
            )

        # Height: terminal towers use terminal_tower_height_m,
        # intermediate towers use algorithm-selected height or fixed height.
        tower_h = self._resolve_tower_height(
            is_terminal, cell_idx, tower_height_map)
        ground_area = self._profile.tower_ground_area_m2
        return Tower(
            location=Point(x, y),
            cell_index=cell_idx,
            tower_type=tower_type,
            turn_angle_deg=turn_deg,
            terrain_cost=terrain_cost,
            angle_cost=angle_cost,
            total_cost=terrain_cost + angle_cost,
            span_to_previous_m=span_prev,
            span_to_next_m=span_next,
            tower_id=tid,
            height_m=tower_h,
            bisector_angle_rad=bisector_rad,
            ground_area_m2=ground_area,
        )

    def _compute_span_statistics(self, towers):
        """Compute span distances and tower type counts/costs.

        Returns:
            Dict with keys: spans, min_span_actual_m, max_span_actual_m,
            avg_span_m, tower_type_counts, tower_type_costs.
        """
        spans = []
        for i in range(len(towers) - 1):
            dx = towers[i+1].location.x - towers[i].location.x
            dy = towers[i+1].location.y - towers[i].location.y
            spans.append(math.sqrt(dx**2 + dy**2))

        tower_type_counts = {}
        tower_type_costs = {}
        for t in towers:
            tower_type_counts[t.tower_type] = (
                tower_type_counts.get(t.tower_type, 0) + 1
            )
            tower_type_costs[t.tower_type] = (
                tower_type_costs.get(t.tower_type, 0) + t.total_cost
            )

        return {
            "spans": spans,
            "min_span_actual_m": min(spans) if spans else 0.0,
            "max_span_actual_m": max(spans) if spans else 0.0,
            "avg_span_m": sum(spans) / len(spans) if spans else 0.0,
            "tower_type_counts": tower_type_counts,
            "tower_type_costs": tower_type_costs,
        }

    def _build_constrained_path(self, path_indices, tower_cell_indices,
                                 raster, source, target, t_pathfinding,
                                 tower_heights_arr=None):
        """Convert raw kernel output into ConstrainedPath with towers."""
        ncols = raster.shape[1]
        transform = self.raster_handler.window_transform

        def idx_to_coord(idx):
            r, c = int(idx) // ncols, int(idx) % ncols
            x, y = transform * (c + 0.5, r + 0.5)
            return x, y

        path_coords = np.array([idx_to_coord(i) for i in path_indices])
        path_geometry = LineString(path_coords) if len(path_coords) > 1 else None

        # Build tower set from Dijkstra output + start/end anchors.
        # Turn towers are already placed by the Dijkstra (mandatory tower
        # at every direction change), so only start/end need post-processing.
        # Build cell->height mapping from variable-height algorithm output
        tower_height_map = {}
        if tower_heights_arr is not None and len(tower_heights_arr) > 0:
            for i, tc in enumerate(tower_cell_indices):
                if i < len(tower_heights_arr):
                    tower_height_map[int(tc)] = float(tower_heights_arr[i])

        tower_set = set(int(t) for t in tower_cell_indices)

        # Terminal towers (start/end anchors) connect to substation/transformer.
        # If the first/last Dijkstra tower is already within min_span of the
        # endpoint, replace it with the terminal instead of adding a new one.
        terminal_cells = set()
        min_span_m = (self._profile.min_span_m
                      if self._profile.has_span_constraints else 0.0)
        if len(path_indices) > 0 and self._profile.has_span_constraints:
            tower_set, terminal_cells = self._enforce_terminal_tower_spans(
                path_indices, tower_cell_indices, tower_set, min_span_m,
                idx_to_coord,
            )

        towers = []
        tower_positions = [
            i for i, idx in enumerate(path_indices) if int(idx) in tower_set
        ]

        for tid, pos in enumerate(tower_positions):
            towers.append(self._build_tower_object(
                pos, path_indices, tower_positions, tower_set,
                tower_height_map, terminal_cells, ncols, raster,
                idx_to_coord, tid,
            ))

        # Cost breakdown
        total_tower_cost = sum(t.total_cost for t in towers)
        total_terrain_cost = sum(
            float(raster[int(idx) // ncols, int(idx) % ncols])
            for idx in path_indices
        )

        stats = self._compute_span_statistics(towers)

        return ConstrainedPath(
            source=source,
            target=target,
            algorithm="constrained-dijkstra",
            graph_api=self._constrained_backend,
            path_indices=path_indices,
            path_coords=path_coords,
            path_geometry=path_geometry,
            euclidean_distance=math.sqrt(
                (path_coords[-1][0] - path_coords[0][0])**2 +
                (path_coords[-1][1] - path_coords[0][1])**2
            ) if len(path_coords) > 1 else 0.0,
            runtimes={"pathfinding": t_pathfinding},
            path_id=len(self.paths),
            search_space_buffer_m=self.search_space_buffer_m,
            neighborhood=self.neighborhood_str,
            profile_name=self._profile.name,
            towers=towers,
            n_towers=len(towers),
            total_terrain_cost=total_terrain_cost,
            total_tower_cost=total_tower_cost,
            total_angle_penalty_cost=0.0,
            cost_breakdown={
                "terrain": total_terrain_cost,
                "towers": total_tower_cost,
                "angle_penalties": 0.0,
            },
            spans=stats["spans"],
            min_span_actual_m=stats["min_span_actual_m"],
            max_span_actual_m=stats["max_span_actual_m"],
            avg_span_m=stats["avg_span_m"],
            turn_angles=[t.turn_angle_deg for t in towers],
            max_turn_angle_deg=(
                max(t.turn_angle_deg for t in towers) if towers else 0.0
            ),
            tower_type_counts=stats["tower_type_counts"],
            tower_type_costs=stats["tower_type_costs"],
        )
