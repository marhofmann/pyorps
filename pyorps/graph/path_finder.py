"""
PYORPS: An Open-Source Tool for Automated Power Line Routing

Reference:
[1] Hofmann, M., Stetz, T., Kammer, F., Repo, S.: 'PYORPS: An Open-Source Tool for
    Automated Power Line Routing', CIRED 2025 - 28th Conference and Exhibition on
    Electricity Distribution, 16 - 19 June 2025, Geneva, Switzerland
"""
from collections.abc import Generator
from contextlib import contextmanager
from time import time
from typing import Any
from warnings import warn

import numpy as np
from geopandas import GeoDataFrame, GeoSeries
from numpy import (
    array,
    asarray,
    iinfo,
    int32,
    ndarray,
    ravel_multi_index,
    sqrt,
    uint16,
    uint32,
    unravel_index,
)
from rasterio.transform import Affine, from_bounds
from rasterio.windows import bounds as window_bounds
from rasterio.windows import transform as window_transform
from shapely.geometry import LineString, MultiPoint, Point, box

from pyorps.core.exceptions import NoPathFoundError, RasterShapeError
from pyorps.core.metric_stack import MetricStack
from pyorps.core.objective import Objective

# Project imports
from pyorps.core.path import Path, PathCollection
from pyorps.core.types import (
    BboxType,
    CoordinateInput,
    CoordinateList,
    CoordinateTuple,
    CostAssumptionsType,
    IMPASSABLE_CELL_COST,
    GeometryMaskType,
    InputDataType,
    Node,
    NodeList,
    NodePathList,
    NormalizedCoordinate,
)
from pyorps.graph.api.graph_api import GraphAPI
from pyorps.io.geo_dataset import (
    InMemoryRasterDataset,
    RasterDataset,
    VectorDataset,
    initialize_geo_dataset,
)
from pyorps.raster.handler import RasterHandler
from pyorps.raster.rasterizer import GeoRasterizer
from pyorps.utils._raster_context import NO_EXCLUSION_VALUE
from pyorps.utils.neighborhood import get_neighborhood_steps
from pyorps.utils.traversal import (
    calculate_path_metrics_numba,
    check_max_values,
    find_nearest_valid_positions_numba,
)

# Maximum number of cells that can be safely indexed with uint32.
# Rasters exceeding this limit would cause silent index overflow in the
# Cython shortest-path kernels.
MAX_SAFE_CELLS = iinfo(uint32).max  # 2**32 - 1 = 4_294_967_295

#: The phases of PathFinder.runtimes that sum to runtimes["total"]. Every other
#: key is either a component of one of these ("import_time_graph_api",
#: "edge_construction" and "graph_creation" decompose "graph_build") or an
#: absolute timestamp ("shortest_path_start_time"), and must never be added.
RUNTIME_PHASES = ("raster_loading", "graph_build", "shortest_path",
                  "path_metrics")

#: Phases recorded in runtimes but DELIBERATELY excluded from the total.
#: "corridor_build" times build_corridor_graph, which is not part of routing a
#: Path; adding it would inflate the total of every Path found afterwards on
#: the same finder.
NON_PATH_RUNTIME_PHASES = ("corridor_build",)


@contextmanager
def timed(name: str, timings_dict: dict[str, float] | None) -> Generator:
    """
    Simple context manager for timing code blocks.

    Parameters:
        name: The name of the code block to be timed and used as a key within the
            timings_dict
        timings_dict: Dictionary to add the timing information to with the specified
            name as key

    Returns:
        A context manager that times the code block
    """
    start_time = time()
    try:
        yield
    finally:
        timings_dict[name] = time() - start_time


def get_graph_api_class(graph_api: str) -> type:
    """
    Return the graph API class based on the selected graph API using pattern matching.

    Parameters:
        graph_api (str): The name of the graph API to use ("networkit", "igraph",
        "networkx", "rustworkx", "cython", "raster_gpu" or
        "raster_fim" — the latter solves the continuous eikonal equation on
        the GPU instead of a graph search). "cugraph" is accepted but raises:
        the backend is not implemented, and cuGraph SSSP measured 5-25x
        slower than the Cython kernel on raster grids.
        Respective graph library must be
        installed! Networkit is a dependency of pyorps and will be installed
        automatically.

    Returns:
        class: The corresponding graph API class.

    Raises:
        ImportError: If the specified graph API module cannot be imported.
        ValueError: If the specified graph API is not supported.
    """
    match graph_api.lower():
        case "networkit":
            from pyorps.graph.api.networkit_api import NetworkitAPI
            return NetworkitAPI
        case "igraph":
            from pyorps.graph.api.igraph_api import IGraphAPI
            return IGraphAPI
        case "rustworkx":
            from pyorps.graph.api.rustworkx_api import RustworkxAPI
            return RustworkxAPI
        case "networkx":
            from pyorps.graph.api.networkx_api import NetworkxAPI
            return NetworkxAPI
        case "cython":
            from pyorps.graph.api.cython_api import CythonAPI
            return CythonAPI
        case "cugraph":
            # There is no cugraph_api module and there never was one in this
            # repository; the import here could only ever raise
            # ModuleNotFoundError, which reads as a broken install rather
            # than an unimplemented backend. Measured verdict on the idea
            # (project history): cuGraph SSSP ran 5-25x SLOWER than the
            # Cython kernel on raster grids, because it is built for
            # billion-edge social graphs and computes all distances instead
            # of stopping at the target.
            raise NotImplementedError(
                "The 'cugraph' backend is not implemented. Use "
                "graph_api='raster_gpu' for GPU routing (raster-direct "
                "delta-stepping), or 'raster_fim' for the eikonal solver.")
        case "raster_gpu":
            from pyorps.graph.api.raster_gpu_api import RasterGPUAPI
            return RasterGPUAPI
        case "raster_fim":
            from pyorps.graph.api.raster_fim_api import RasterFIMAPI
            return RasterFIMAPI
        case _:
            raise ValueError(f"Unsupported graph API: {graph_api}")


class PathFinder:
    """
    A class that encapsulates RasterReader and graph-based routing capabilities.

    This class provides functionality to:
    1. Read raster data using RasterReader or create a raster using GeoRasterizer
    2. Create a graph representation of the raster with a defined Graph library
    3. Find the shortest paths between coordinates
    4. Convert resulting paths of graph node indices back to coordinates
    5. Create GeoDataFrames of paths and export to other geo-formats for further
    analysis

    The class supports various graph APIs to create a graph from a raster.

    Runtime accounting (``self.runtimes``, seconds, copied onto every Path):
        raster_loading            one-time: loading/rasterizing the search
                                  raster; measured in __init__ and repeated
                                  verbatim in every later path's runtimes
        graph_build               building the backend handle FOR THIS ROUTE
                                  (edge construction included) — 0.0 when a
                                  previously built graph was reused
        import_time_graph_api,
        edge_construction,
        graph_creation            components of the last graph_build; never
                                  added to "total" on their own
        shortest_path             the search itself, graph construction
                                  excluded
        path_metrics              post-processing (0.0 when
                                  calculate_metrics=False)
        total                     sum of RUNTIME_PHASES
        shortest_path_start_time  absolute epoch timestamp of the search start
    """

    # Backing stores for the lazily-materialized objective state (item
    # 1.10); class-level defaults keep the properties safe for instances
    # built without running __init__ (test doubles, __new__).
    _raster_handler = None
    _combine_result_value = None
    _objective_dirty = False
    _applying_objective = False
    _category_cache = None
    #: search_space_buffer_m exactly as the caller passed it (see __init__).
    _explicit_buffer_m = None
    corridor_first = True
    #: Last graph built by build_corridor_graph (the shared-trench
    #: sense of 'corridor', not the search window).
    corridor_graph = None

    def __init__(
            self,
            dataset_source: InputDataType,
            source_coords: CoordinateInput | None,
            target_coords: CoordinateInput | None,
            search_space_buffer_m: float | None = None,
            neighborhood_str: str | int | None = "r2",
            steps: ndarray[int] | None = None,
            ignore_max_cost: bool = True,
            graph_api: str = "cython",
            cost_assumptions: CostAssumptionsType | None = None,
            datasets_to_modify: list[dict[str, Any]] | None = None,
            crs: str | None = None,
            bbox: BboxType | None = None,
            mask: GeometryMaskType | None = None,
            transform: Affine | None = None,
            raster_save_path: str | None = None,
            dem: InputDataType | None = None,
            dem_kwargs: dict[str, Any] | None = None,
            use_gpu: bool = False,
            objective: "Objective | dict[str, float] | None" = None,
            gradient_options: dict[str, Any] | None = None,
            metric_layers: dict[str, Any] | None = None,
            weight_precision: str = "uint16",
            corridor_first: bool = True,
            **kwargs
    ):
        """
        Initialize the RasterGraph with a dataset source and routing parameters.

        Parameters:
            dataset_source: Either:
                          - Path to a file (str)
                          - Tuple of (data_array, crs, transform)
                          - GeoDataset object
                          - Dictionary with url/layer for WFS
            source_coords: CoordinateInput
                Can be: tuple, list of tuples, array of arrays, shapely Point,
                shapely MultiPoint, GeoSeries of points, or GeoDataFrame of points.
            target_coords: CoordinateInput
                Can be: tuple, list of tuples, array of arrays, shapely Point,
                shapely MultiPoint, GeoSeries of points, or GeoDataFrame of points.
            search_space_buffer_m: Buffer in metres around the convex hull of the
                source and target coordinates; the search window is that buffered
                hull, clipped to the raster.
                ``None`` (the default) does NOT mean the full raster -- it calls
                :meth:`RasterHandler.estimate_buffer_width`, which samples the
                raster and returns something between 200 m and 4000 m sized for
                ONE source/target pair. That is the wrong shape for a
                :class:`~pyorps.graph.search_session.CostField` querying
                candidates spread across the raster.
                ``0`` means a zero-width buffer, i.e. the hull itself -- the
                SMALLEST search space, not the largest. With a single pair of
                points (or collinear ones) the hull has no area and ``0`` raises.
                For the whole raster pass
                :func:`~pyorps.graph.search_session.full_window_buffer_m`.
            neighborhood_str: Neighborhood type. Defaults to "r2".
            steps: Steps which define the neighborhood. If None,
                will be created from neighborhood_str.
            ignore_max_cost: Whether to ignore all cells in the raster
                which have the maximum cost value or not
            graph_api: Graph API to use.
                Available graph libraries:
                    "cython" (default), "networkit", "rustworkx", "igraph",
                    "networkx", "cugraph", "raster_gpu" (GPU delta-stepping)
                    and "raster_fim" (GPU eikonal solver — continuous
                    least-cost paths, no neighborhood; requires CUDA)
            cost_assumptions: Cost assumptions to use for rasterization.
                Required if dataset_source is vector data.
            datasets_to_modify: List of datasets to use to modify the raster using
                GeoRasterizer.modify_raster_from_dataset
            crs: The coordinate reference system to be used as project crs (crs of
                the dataset_source and all other datasets will be converted to this crs)
            bbox: The bounding box to be used as project bounding box. Defines the
                area in which path finding is processed.
            mask:  Defines the area in which path finding is processed similar to the
                bbox parameter. In this case a more complex Polygon, a Multipolygon or
                even a GeoSeries/GeoDataFrame with multiple Polygons can be used to
                define the search space for path finding.
            transform: Affine transformation describing the transform of a
                RasterDataset. Can be used ia a raster dataset is passed directly to
                dataset_source.
            raster_save_path: Path to save the raster dataset to.
            use_gpu: If True, use GPU-accelerated edge construction for graph
                library backends (networkit, rustworkx, igraph, networkx). Requires
                CuPy and an NVIDIA GPU. Falls back to CPU automatically if
                unavailable. Has no effect on cython or cugraph backends.
            objective: Optional feasibility objective (an Objective or a
                weights dict like ``{"cost": 1.0, "landscape": 800.0}``).
                Activates the metric pipeline: multi-metric cost assumptions
                are rasterized into a MetricStack, combined under the
                objective and quantized into the search raster. When None
                (default), routing behaves exactly as before.
            gradient_options: Optional gradient-response configuration
                (dict, see pyorps.GradientOptions) when ``objective`` is a
                plain dict. In-search gradient terms require the kernel
                phases of the feasibility plan and are rejected until then.
            weight_precision: "uint16" (default — the quantized search
                raster all backends accept) or "float32" — a LOSSLESS
                combined surface for extreme weight dynamic ranges (when
                the combine diagnostics warn about quantization collapse).
                float32 requires an objective and one of the
                FLOAT_BACKENDS (library backends, raster_gpu or
                raster_fim — the eikonal solver consumes float32
                natively).
            corridor_first: Restrict the stages BEFORE the search to the
                search corridor (performance-plan item 2.2): read only the
                search window out of a raster/DEM file, burn only the
                corridor of a vector input, and scalarize only the search
                window of a metric stack. Every one of those steps is
                applied only where it is provably identical inside the
                searched window and is skipped otherwise, so leaving it on
                cannot change a route; it exists as a switch so that the
                two paths can be compared directly in tests. All of it
                requires an explicit ``search_space_buffer_m`` — see
                :meth:`corridor_geometry`.
            **kwargs: Additional keyword arguments to pass to the rasterize function
                of the RasterHandler (if a VectorDataset or a source to a VectorDataset
                has been provided with dataset_source) or to the load function of the
                RasterDataset (if a source to a RasterDataset has been provided with
                dataset_source). On the vector path this includes the sub-cell
                forbidden-feature knobs ``widen_thin_forbidden`` and
                ``all_touched`` (both default False, i.e. a plain GDAL burn)
                and ``on_thin_features`` (default 'warn'). A forbidden feature
                narrower than one cell vanishes from the cost surface at some
                sub-pixel alignments and then cannot block a route; the default
                DETECTS that and warns, because both repairs also seal
                legitimate sub-cell OPENINGS (measured over four gate-width
                sets: 38-100 % of the gates for widening, 31-100 % for
                all_touched, on a thin wall) and the only fix that is
                not a trade is a finer cell size. See
                :meth:`pyorps.raster.GeoRasterizer.rasterize` and the
                "Thin Barriers & Forbidden Zones" documentation page.

        Minimal example:
        >>> from pyorps import PathFinder
        >>> source = (472000, 5593400)
        >>> target = (472800, 5594000)
        >>> raster_path = r"./data/raster/sample_raster.tiff"
        >>> path_finder = PathFinder(
        >>>     dataset_source=raster_path,
        >>>     source_coords=source,
        >>>     target_coords=target,
        >>> )
        >>> path_finder.find_route()
        Path(path_id=0, source=(472000, 5593400), target=(472800, 5594000),
             total_length=1192.43, total_cost=133578.05)

        """
        self.source_coords = PathFinder.normalize_coordinates(source_coords)
        self.target_coords = PathFinder.normalize_coordinates(target_coords)
        self.search_space_buffer_m = search_space_buffer_m
        # The buffer AS THE CALLER GAVE IT. search_space_buffer_m is
        # overwritten later with whatever the estimator picked, so keying the
        # corridor-first steps on it would make them switch on halfway
        # through an object's life — the first route would then see a
        # different search raster than the second one for the same
        # objective. The explicit value never changes, so the behaviour of
        # a finder is fixed at construction.
        self._explicit_buffer_m = search_space_buffer_m
        self.corridor_first = bool(corridor_first)
        self.neighborhood_str = neighborhood_str
        self.graph_api_name = graph_api
        self.ignore_max_cost = ignore_max_cost
        self.use_gpu = use_gpu
        self.objective = self._normalize_objective(objective, gradient_options)
        self.metric_layers = metric_layers
        self.metric_stack = None
        self._objective_dirty = False
        self._applying_objective = False
        self._category_cache = None
        self._combine_result = None
        self._gradient_dem = None
        if metric_layers and self.objective is None:
            raise ValueError(
                "metric_layers require an objective — pass objective= "
                "to activate the metric pipeline.")
        self.weight_precision = weight_precision
        self._validate_precision_support()

        if steps is None and neighborhood_str:
            directed = self.graph_api_name in ("cython", "raster_gpu", "cugraph")
            self.steps = get_neighborhood_steps(neighborhood_str, directed=directed)
        else:
            self.steps = steps

        self.runtimes = {}
        self._total_cost_basis_warned = False
        self.paths = PathCollection()  # Initialize PathCollection instead of list

        # Initialize as None (to be lazily loaded/created)
        self.raster_handler = None
        self.geo_rasterizer = None
        self._graph_api = None
        self.path_gdf = None
        self.dem_dataset = None
        self.dem_raster_handler = None
        self.dem_kwargs = dem_kwargs
        self._search_sessions: list = []

        # Load the dataset
        self.dataset = initialize_geo_dataset(dataset_source, crs, bbox, mask,
                                              transform)
        # Initialize DEM dataset if provided
        if dem is not None:
            self.dem_dataset = initialize_geo_dataset(
                dem,
                crs=self.dataset.crs,  # Use same CRS as main dataset
                bbox=bbox,
                mask=mask,
                transform=transform
            )

        self._validate_gradient_support()

        if self.source_coords is not None and self.target_coords is not None:
            self.create_raster_handler(cost_assumptions, datasets_to_modify,
                                       raster_save_path, dem_kwargs, **kwargs)

    @staticmethod
    def _normalize_objective(
            objective: "Objective | dict[str, float] | None",
            gradient_options: dict[str, Any] | None,
    ) -> Objective | None:
        """Coerce the objective inputs into a validated Objective or None."""
        if objective is None:
            if gradient_options is not None:
                raise ValueError(
                    "gradient_options requires an objective — pass "
                    "objective= as well.")
            return None
        if isinstance(objective, Objective):
            if gradient_options is not None:
                raise ValueError(
                    "Pass gradient options inside the Objective instance "
                    "OR use a plain weights dict with gradient_options=, "
                    "not both.")
            result = objective
        else:
            result = Objective(objective, gradient_options)
        return result

    #: Backends whose search kernels consume the gradient LUT pair.
    GRADIENT_BACKENDS = ("cython", "networkit", "networkx", "igraph",
                         "rustworkx", "raster_gpu")

    #: Backends that accept lossless float32 weight rasters (Phase 9).
    #: The cython kernels remain uint16-pinned (RasterContext).
    FLOAT_BACKENDS = ("networkit", "networkx", "igraph", "rustworkx",
                      "raster_gpu", "raster_fim")

    #: Corridor buffer assumed when the caller left ``search_space_buffer_m``
    #: unset. ``RasterHandler.estimate_buffer_width`` clamps its result to
    #: [200, 4000] m, so 4000 m is the WIDEST corridor the estimator can ever
    #: return: a corridor built with it always contains the one the estimator
    #: would have chosen. That asymmetry is the whole justification — a
    #: too-wide corridor only costs time, a too-narrow one silently changes
    #: the answer. See :meth:`corridor_geometry` for why the burn and the
    #: windowed read still refuse to use it.
    CORRIDOR_FALLBACK_BUFFER_M = 4000.0

    #: Largest coordinate drift (metres, accumulated across the whole corridor
    #: raster) accepted when certifying that a corridor burn lands on the same
    #: grid as the full-extent burn. ``GeoRasterizer.rasterize`` reconstructs
    #: the pixel size as ``(east - west) / columns``; that division cannot
    #: always reproduce the full grid's pixel size bit-for-bit, because the
    #: quantum of a coordinate near 5.6e6 is ~1e-9 m while the quantum of the
    #: quotient is ~2e-16. Measured worst case over 5000 random extents at
    #: 0.5-10 m resolution: 9.2e-10 m. Extents whose pixel size IS exactly
    #: representable (round bounds — precisely the case where scan conversion
    #: has genuine pixel-centre ties) reproduce the grid exactly. One
    #: micrometre sits three orders of magnitude below that worst case and six
    #: below the millimetre precision of cadastral geometry.
    CORRIDOR_GRID_TOLERANCE_M = 1e-6

    def corridor_geometry(self, buffer_m: float | None = None):
        """The search corridor — the polygon ``RasterHandler`` windows to.

        Reproduces ``RasterHandler._init_from_metadata`` exactly: a single
        source/target pair gives the buffered straight line between them,
        anything else the buffered convex hull of all endpoints, always with
        ``quad_segs=32``. Having it here as well is what lets the stages in
        FRONT of the handler — the file read, the burn, the objective
        combine — see the same corridor the search will use (plan item 2.2).

        Parameters:
            buffer_m: Buffer width. Defaults to ``search_space_buffer_m``,
                and to :attr:`CORRIDOR_FALLBACK_BUFFER_M` when that is None.

        Returns:
            A shapely polygon, or None when the corridor is undefined
            (missing coordinates, or a non-positive buffer, which is the
            documented "use the entire raster" request).

        Note:
            The default is deliberately NOT used by the corridor-first
            steps. When ``search_space_buffer_m`` is None the handler calls
            ``estimate_buffer_width``, which samples the raster it is given —
            and it samples it at indices clamped to the raster SHAPE
            (``handler.py:388-403`` clips row indices against the height and
            then indexes ``[cols, rows]``), so the estimate is a function of
            the extent, not only of the terrain. Shrinking the extent could
            therefore change the estimated buffer, hence the searched window,
            hence the route. Every corridor-first step in this class requires
            an explicit buffer for that reason; the fallback exists for
            callers that want a corridor for a step which does not feed the
            estimator (e.g. narrowing a network fetch).
        """
        source = self.source_coords
        target = self.target_coords
        if source is None or target is None:
            return None
        if buffer_m is None:
            buffer_m = self.search_space_buffer_m
        if buffer_m is None:
            buffer_m = self.CORRIDOR_FALLBACK_BUFFER_M
        if not buffer_m or buffer_m <= 0:
            return None

        is_single_source = self._is_single_coordinate(source)
        is_single_target = self._is_single_coordinate(target)
        if is_single_source and is_single_target:
            geometry = LineString([tuple(source), tuple(target)])
        else:
            points = []
            for coords, single in ((source, is_single_source),
                                   (target, is_single_target)):
                if single:
                    points.append(tuple(coords))
                else:
                    points.extend(tuple(c) for c in coords)
            geometry = MultiPoint(points).convex_hull
        return geometry.buffer(distance=buffer_m, quad_segs=32)

    def corridor_bounds(
            self,
            buffer_m: float | None = None
    ) -> tuple[float, float, float, float] | None:
        """``(minx, miny, maxx, maxy)`` of :meth:`corridor_geometry`."""
        geometry = self.corridor_geometry(buffer_m)
        if geometry is None or geometry.is_empty:
            return None
        return geometry.bounds

    @staticmethod
    def _is_single_coordinate(coords) -> bool:
        """Mirror of RasterHandler's single-vs-many coordinate test."""
        if isinstance(coords, tuple):
            return True
        return (isinstance(coords, list) and len(coords) == 2
                and not isinstance(coords[0], (list, tuple)))

    def _validate_precision_support(self) -> None:
        """Fail fast on unsupported weight_precision configurations."""
        if self.weight_precision not in ("uint16", "float32"):
            raise ValueError(
                f"weight_precision must be 'uint16' or 'float32', got "
                f"{self.weight_precision!r}")
        if self.weight_precision == "uint16":
            return
        if self.objective is None:
            raise ValueError(
                "weight_precision='float32' requires an objective — the "
                "lossless path exists for combined feasibility surfaces.")
        if self.graph_api_name not in self.FLOAT_BACKENDS:
            raise NotImplementedError(
                f"weight_precision='float32' is supported on "
                f"{self.FLOAT_BACKENDS}; the '{self.graph_api_name}' "
                f"kernels are uint16-pinned.")
        if self.search_space_buffer_m is None:
            raise ValueError(
                "weight_precision='float32' requires an explicit "
                "search_space_buffer_m (the buffer estimator assumes "
                "integer rasters).")

    def _uses_gradient_kernels(self) -> bool:
        """True when the search must apply per-edge slope terms.

        With an objective and a DEM, the 3D length stretch applies
        unconditionally (geometry, plan section 0) — even without explicit
        gradient responses.
        """
        return self.objective is not None and (
            self.dem_dataset is not None
            or self.objective.has_gradient_terms)

    def _validate_gradient_support(self) -> None:
        """Fail fast when gradient/stretch terms cannot reach the search."""
        if not self._uses_gradient_kernels():
            return
        if self.dem_dataset is None:
            raise ValueError(
                "The objective contains gradient terms (a 'gradient' "
                "weight, a multiplier response or max_gradient_pct) but "
                "no DEM was provided — pass dem=.")
        if self.graph_api_name not in self.GRADIENT_BACKENDS:
            raise NotImplementedError(
                f"With a DEM, an objective applies per-edge slope terms "
                f"(3D stretch always; responses when configured) — "
                f"supported on {self.GRADIENT_BACKENDS}, "
                f"not on '{self.graph_api_name}'.")

    def set_objective(
            self,
            objective: "Objective | dict[str, float]",
            gradient_options: dict[str, Any] | None = None,
    ) -> None:
        """Swap the feasibility objective and recombine the search raster.

        Cheap for the raster-direct backends (cython / raster_gpu): only
        the combine + quantize step reruns and the graph handle is
        invalidated — no graph rebuild happens until the next find_route.
        """
        normalized = self._normalize_objective(objective, gradient_options)
        if normalized is None:
            raise ValueError("set_objective requires an objective")
        if self.metric_stack is None:
            raise ValueError(
                "set_objective requires a PathFinder constructed with "
                "objective= (the metric pipeline is inactive).")
        previous = self.objective
        self.objective = normalized
        try:
            self._validate_gradient_support()
        except (ValueError, NotImplementedError):
            self.objective = previous
            raise
        self._apply_objective_to_stack()

    @property
    def raster_handler(self) -> RasterHandler | None:
        """The search-raster handler, recombined on demand (item 1.10)."""
        self._ensure_objective_applied()
        return self._raster_handler

    @raster_handler.setter
    def raster_handler(self, value: RasterHandler | None) -> None:
        # Every handler replacement funnels through here, so dropping the
        # category cache at this one point is what makes it sound: without it
        # the cache keys on a buffer address, and a replacement handler that
        # reuses the freed address would be served the previous raster's value
        # set - silently dropping metres from length_by_category.
        self._category_cache = None
        self._raster_handler = value

    @property
    def _combine_result(self):
        """The active combine diagnostics, recombined on demand."""
        self._ensure_objective_applied()
        return self._combine_result_value

    @_combine_result.setter
    def _combine_result(self, value) -> None:
        self._combine_result_value = value

    def _defer_objective_restore(self, objective: Objective) -> None:
        """Restore the objective now, defer its combine to first use.

        ``find_route_ensemble`` restores the caller's objective in a
        ``finally`` block even when the caller never routes again; a full
        combine plus handler rebuild there is pure waste (item 1.10). The
        objective itself is restored immediately, so everything that reads
        ``self.objective`` sees the caller's choice — only the derived
        state (combined raster, handler, graph, aligned DEM) is marked
        stale and rebuilt by ``_ensure_objective_applied`` on the first
        access to ``raster_handler`` or ``_combine_result``.
        """
        self.objective = objective
        self._objective_dirty = True
        self._graph_api = None
        self._gradient_dem = None

    def _ensure_objective_applied(self) -> None:
        """Materialize a deferred objective restore, at most once."""
        if not self._objective_dirty or self._applying_objective:
            return
        self._applying_objective = True
        try:
            self._apply_objective_to_stack()
        finally:
            self._applying_objective = False

    def _stack_search_window(self, stack):
        """The window the RasterHandler will cut, on the stack's own grid.

        Returned only when cutting it FIRST provably leaves the handler with
        exactly the same cells it would have windowed out of the full stack:

        * an explicit ``search_space_buffer_m`` is required, because
          otherwise the handler estimates the buffer from the raster it is
          handed and a smaller raster can produce a different estimate
          (:meth:`corridor_geometry`);
        * a legacy single-raster alias is left alone — windowing it would
          materialize the float layers and destroy ``combine``'s zero-copy
          pass-through of the untouched uint16 raster;
        * the handler's own window computation is re-run on the sub-grid and
          must come back as "the whole sub-raster". That check is what makes
          the pre-window exact rather than merely plausible: ``rowcol`` on
          the shifted transform is not guaranteed to floor identically, and
          if it does not, we simply do not pre-window.

        Returns None whenever any of that fails — the caller then combines
        the full extent, i.e. today's behaviour.
        """
        if not getattr(self, "corridor_first", True):
            return None
        buffer_m = self._explicit_buffer_m
        if buffer_m is None or buffer_m <= 0:
            return None
        if stack is None or stack.shape is None or stack.is_legacy_alias:
            return None
        corridor = self.corridor_geometry(buffer_m)
        if corridor is None or corridor.is_empty:
            return None

        window = RasterHandler.window_from_bounds(
            corridor.bounds, stack.transform, stack.shape)
        height, width = int(window.height), int(window.width)
        if height <= 0 or width <= 0:
            return None
        if (height, width) == tuple(stack.shape):
            return None  # nothing to win, and nothing to risk

        sub_transform = window_transform(window, stack.transform)
        check = RasterHandler.window_from_bounds(
            corridor.bounds, sub_transform, (height, width))
        if (int(check.col_off), int(check.row_off),
                int(check.width), int(check.height)) != (0, 0, width, height):
            return None
        return window

    def _apply_objective_to_stack(self) -> None:
        """Combine the stack under the current objective, rebuild handler.

        Window first, then combine (performance-plan item 2.6). The handler
        discards everything outside the search window anyway, so scalarizing
        the full extent first is pure waste — 5-30x per objective variant,
        plus up to ~1 GB of transient float32 at municipality scale. The
        pre-window is taken only when it is certified to select exactly the
        cells the handler would have selected (see
        :meth:`_stack_search_window`); otherwise the full stack is combined,
        exactly as before.

        SEMANTIC CONSEQUENCE, stated plainly: the uint16 quantization scale
        is ``65534 / max(F)`` over the traversable cells of whatever extent
        ``combine`` is called on. Taking the maximum over the search window
        instead of the whole data extent therefore produces a DIFFERENT — and
        normally FINER — quantization than before this change, so the
        combined weight raster is not bit-identical to the one a full-extent
        combine produced. It is an exact linear scalarization of the same
        objective, and it is exact with respect to the window that is
        actually searched; what changes is the rounding grid, which becomes
        no coarser. The scale in force is recorded on
        ``CombineResult.scale`` and copied onto
        ``Path.objective_spec["quantization_scale"]``. Nothing else in the
        pipeline is affected: the float layers, the per-metric evaluation and
        the reported feasibility are all computed from the unquantized
        layers.

        The narrowing is applied to ``self.metric_stack`` itself rather than
        to a private copy. The corridor is a function of the endpoints and
        the explicit buffer, both fixed for the finder's lifetime, so the
        second pass is a no-op and every objective variant sees the same
        extent. Doing it this way is also what keeps the invariant everything
        else relies on — ``raster_handler.window`` indexes into
        ``metric_stack`` — true for subclasses and callers that were never
        told about this change (``ConstrainedPathFinder._build_tower_cost_
        raster`` is one). The bands are numpy VIEWS, so nothing is copied.
        """
        use_float = self.weight_precision == "float32"
        window = self._stack_search_window(self.metric_stack)
        if window is not None:
            self.metric_stack = self.metric_stack.window(window)
        stack = self.metric_stack
        result = stack.combine(self.objective, quantize=not use_float)
        self._combine_result = result
        self._gradient_dem = None  # rebuilt lazily at graph creation
        dataset = InMemoryRasterDataset(
            result.weights, stack.crs, stack.transform)
        handler_kwargs = {}
        if use_float:
            # Outside-buffer cells become +inf (the float forbidden
            # encoding) instead of the integer dtype maximum.
            handler_kwargs["outside_value"] = np.float32(np.inf)
        self.raster_handler = RasterHandler(
            dataset,
            self.source_coords,
            self.target_coords,
            self.search_space_buffer_m,
            **handler_kwargs,
        )
        self._graph_api = None
        self._objective_dirty = False

    def _create_metric_raster_handler(
            self,
            cost_assumptions: CostAssumptionsType | None,
            datasets_to_modify: list[dict[str, Any]] | None,
            raster_save_path: str | None,
            **kwargs
    ) -> None:
        """Metric-pipeline variant of create_raster_handler (objective set).

        Vector input is rasterized into a multi-band MetricStack; raster
        input becomes a zero-copy legacy alias stack. The stack is built
        unmasked, so objective changes always recombine from pristine
        layers; ``_apply_objective_to_stack`` narrows it once to the search
        corridor (item 2.6, a view — nothing is copied) and the
        RasterHandler still owns the buffer masking of the combined weight
        raster. ``raster_save_path`` saves the stack AFTER that narrowing,
        i.e. the corridor rather than the whole input extent.

        Note that the corridor burn (item 2.2) does not apply here:
        ``rasterize_metrics`` has no ``bounding_box`` parameter, so the
        metric bands are still burned over the whole data extent. Only the
        combine step is corridor-sized on this path.
        """
        if datasets_to_modify:
            raise NotImplementedError(
                "datasets_to_modify overlays on the metric pipeline land "
                "with Phase 4 of the feasibility plan (per-metric "
                "'applies_to' targeting).")

        if isinstance(self.dataset, VectorDataset):
            if cost_assumptions is None:
                msg = "Cost assumptions must be provided when using vector data"
                raise ValueError(msg)
            # The sub-cell forbidden-feature knobs (all_touched /
            # widen_thin_forbidden / on_thin_features) are listed here so the
            # objective pipeline exposes the same barrier semantics as the
            # legacy one - a barrier that vanishes from the cost band is
            # invisible to the search on BOTH paths.
            allowed = {"resolution_in_m", "geometry_buffer_m",
                       "include_category", "preprocessing_function",
                       "preprocessing_kwargs", "all_touched",
                       "widen_thin_forbidden", "on_thin_features"}
            unknown = set(kwargs) - allowed
            if unknown:
                raise ValueError(
                    f"Keyword argument(s) {sorted(unknown)} are not "
                    f"supported together with objective= "
                    f"(rasterize_metrics accepts {sorted(allowed)}).")
            self.geo_rasterizer = GeoRasterizer(self.dataset, cost_assumptions)
            self.metric_stack = self.geo_rasterizer.rasterize_metrics(**kwargs)
        elif isinstance(self.dataset, RasterDataset):
            if cost_assumptions is not None:
                raise NotImplementedError(
                    "Applying cost assumptions to a raster input together "
                    "with objective= lands with Phase 4 of the "
                    "feasibility plan.")
            self.dataset.load_data(**kwargs)
            band = self.dataset.data
            if band is not None and band.ndim == 3:
                band = band[0]
            self.metric_stack = MetricStack.from_single_raster(
                band, self.dataset.transform, self.dataset.crs)
        else:
            raise ValueError(f"Unsupported dataset type: {type(self.dataset)}")

        if self.metric_layers:
            self._ingest_metric_layers()

        self._apply_objective_to_stack()

        if raster_save_path is not None:
            self.metric_stack.save(raster_save_path)

    def _ingest_metric_layers(self) -> None:
        """Add user-provided metric layers to the stack.

        Spec forms per layer name:
            "path.tif"                       — prebuilt raster, reprojected
            ndarray                          — array on the stack grid
            {"derive": "slope_from_dem", ...}— per-cell terrain slope (%)
            {"source": path|ndarray, "transform": ..., "crs": ...,
             "hard_max": ..., "hard_min": ...}
        """
        from pyorps.core.metric_stack import reproject_to_grid

        stack = self.metric_stack
        if stack.shape is None:
            raise ValueError("Cannot add metric layers to an empty stack")

        for name, spec in self.metric_layers.items():
            if isinstance(spec, (str, ndarray)):
                spec = {"source": spec}
            elif not isinstance(spec, dict):
                raise ValueError(
                    f"metric_layers['{name}'] must be a path, an array or "
                    f"a spec dict, got {type(spec)}")

            derive = spec.get("derive")
            # ``owned`` says whether the array was produced for this call and
            # nobody else holds it — then the stack can adopt it instead of
            # copying a full-extent float32 band (item 2.6).
            if derive == "slope_from_dem":
                self._ensure_stack_dem()
                values = stack.derive_terrain_slope()
                owned = True
            elif derive is not None:
                raise ValueError(
                    f"Unknown derive '{derive}' for metric layer "
                    f"'{name}' (supported: 'slope_from_dem')")
            elif "source" in spec:
                values, owned = self._load_layer_source(name, spec)
            else:
                raise ValueError(
                    f"metric_layers['{name}'] needs 'source' or 'derive'")

            stack.add_layer(name, values,
                            hard_max=spec.get("hard_max"),
                            hard_min=spec.get("hard_min"),
                            copy=not owned)

    def _ensure_stack_dem(self) -> None:
        """Reproject the DEM dataset onto the stack grid (once)."""
        from pyorps.core.metric_stack import reproject_to_grid

        stack = self.metric_stack
        if stack.dem is not None:
            return
        if self.dem_dataset is None:
            raise ValueError(
                "Deriving terrain_slope requires a DEM — pass dem=.")
        if self.dem_dataset.data is None:
            self.dem_dataset.load_data(**(self.dem_kwargs or {}))
        dem_src = self.dem_dataset.data
        if dem_src.ndim > 2:
            dem_src = dem_src[0]
        # Legacy assumption: an unspecified DEM CRS matches the stack CRS.
        src_crs = self.dem_dataset.crs or stack.crs
        aligned = reproject_to_grid(
            dem_src, self.dem_dataset.transform, src_crs,
            stack.shape, stack.transform, stack.crs)
        stack.attach_dem(aligned)

    def _load_layer_source(self, name: str, spec: dict) -> tuple[ndarray, bool]:
        """Load a prebuilt layer source onto the stack grid.

        Returns:
            ``(values, owned)`` — ``owned`` is True when the array was
            allocated here, so the stack may adopt it without copying
            (item 2.6). It is False for a caller-supplied array handed
            through unchanged, which must never be written into.
        """
        from pyorps.core.metric_stack import reproject_to_grid

        stack = self.metric_stack
        source = spec["source"]
        if isinstance(source, ndarray):
            if source.shape == stack.shape:
                return source, False
            if "transform" not in spec:
                raise ValueError(
                    f"metric_layers['{name}'] array shape {source.shape} "
                    f"differs from the stack {stack.shape} — provide "
                    f"'transform' (and optionally 'crs') for "
                    f"reprojection.")
            return reproject_to_grid(
                source, spec["transform"], spec.get("crs", stack.crs),
                stack.shape, stack.transform, stack.crs), True
        if isinstance(source, str):
            from rasterio import open as rio_open
            with rio_open(source) as src:
                data = src.read(1)
                aligned = reproject_to_grid(
                    data, src.transform, src.crs,
                    stack.shape, stack.transform, stack.crs)
            # Uncovered area contributes 0 (coverage gaps are not
            # forbidden — forbidden comes from values, 65535/inf).
            uncovered = ~np.isfinite(aligned)
            if uncovered.any():
                warn(f"Metric layer '{name}': {int(uncovered.sum())} "
                     f"cell(s) outside the source coverage default to 0.",
                     UserWarning, stacklevel=2)
                aligned[uncovered] = 0.0
            return aligned, True
        raise ValueError(
            f"metric_layers['{name}']['source'] must be a path or an "
            f"array, got {type(source)}")

    @staticmethod
    def normalize_coordinates(
            input_data: CoordinateInput | None
    ) -> NormalizedCoordinate | None:
        """
        Normalize different coordinate formats into tuples or lists of tuples.

        Parameters:
            input_data: Can be a tuple, a list of tuples, an array of arrays, a shapely
                Point, a shapely MultiPoint, a GeoSeries of points, or a GeoDataFrame of
                points.

        Returns:
            CoordinateOutput: A single coordinate tuple (x, y) or list of coordinate
                tuples [(x1, y1), (x2, y2), ...]
        """
        if input_data is None:
            coordinate_output = None
        # Case: Input is a tuple with two elements
        elif isinstance(input_data, tuple) and len(input_data) == 2:
            coordinate_output = input_data
        # Case: Input is a shapely Point
        elif isinstance(input_data, Point):
            coordinate_output = input_data.x, input_data.y
        # Case: Input is a shapely MultiPoint
        elif isinstance(input_data, MultiPoint):
            coordinate_output = [(p.x, p.y) for p in input_data.geoms]
        # Case: Input is a GeoSeries
        elif isinstance(input_data, GeoSeries):
            coordinate_output = PathFinder._point_or_multipoints(input_data)
        # Case: Input is a GeoDataFrame
        elif isinstance(input_data, GeoDataFrame):
            coordinate_output = PathFinder._point_or_multipoints(input_data.geometry)
        # Case: Input is a list of tuples
        elif isinstance(input_data, list):
            if all(isinstance(item, tuple) and len(item) == 2 for item in input_data):
                coordinate_output = input_data
            elif all(isinstance(item, list) and len(item) == 2 for item in input_data):
                coordinate_output = [(float(i[0]), float(i[1])) for i in input_data]
            else:
                coordinate_output = PathFinder._point_or_multipoints(input_data)
        # Case: Input is a numpy array
        elif isinstance(input_data, ndarray):
            if len(input_data.shape) == 2 and input_data.shape[1] == 2:
                coordinate_output = [(float(c[0]), float(c[1])) for c in input_data]
            else:
                coordinate_output = PathFinder._point_or_multipoints(input_data)
        else:
            # If input doesn't match any expected format
            raise ValueError("Input data cannot be interpreted as coordinates")
        if isinstance(coordinate_output, list) and len(coordinate_output) == 1:
            return coordinate_output[0]
        return coordinate_output

    @staticmethod
    def _point_or_multipoints(input_data: CoordinateInput) -> NormalizedCoordinate:
        """
        Converts a Points or a Multipoint to a NormalizedCoordinate

        Parameters:
            input_data: Point or Multipoint to be converted to a NormalizedCoordinate

        Returns:
            A NormalizedCoordinate of the input_data
        """
        if len(input_data) == 0:
            return []
        if all(isinstance(item, Point) for item in input_data):
            return PathFinder._get_point_coordinates(input_data)
        if all(isinstance(item, MultiPoint) for item in input_data):
            return PathFinder._get_multipoint_coordinates(input_data)
        raise ValueError("Input data cannot be interpreted as coordinates")

    @staticmethod
    def _get_multipoint_coordinates(input_data):
        """
        Extracts coordinates from a collection of MultiPoint geometries

        Parameters:
            input_data: Collection of MultiPoint objects

        Returns:
            List of (x, y) coordinate tuples extracted from all points
            within all MultiPoint geometries
        """
        # Iterate through each MultiPoint item and extract coordinates from each
        # point geometry
        return [(p.x, p.y) for item in input_data for p in item.geoms]

    @staticmethod
    def _get_point_coordinates(input_data):
        """
        Extracts coordinates from a collection of Point geometries

        Parameters:
            input_data: Collection of Point objects

        Returns:
            List of (x, y) coordinate tuples from the Point objects
        """
        # Extract x, y coordinates from each Point object
        return [(point.x, point.y) for point in input_data]

    def _can_read_source_window(self, dataset, read_kwargs) -> bool:
        """True when RasterHandler can take the window straight off disk.

        Item 2.2/2.3: PathFinder used to call ``load_data()`` on every raster
        input before handing it to the handler, which pulled the whole
        GeoTIFF into memory (3.2 GB for a state-wide 40000x40000 raster) so
        that the handler could slice a few hundred MB out of it. The handler
        can read the window itself — but only if we do NOT pre-load it.

        Bit-identity: the file's grid is a property of the file, so the
        window is computed from the same header either way, and
        ``read_window`` asks rasterio for the stored pixels of exactly that
        window with no resampling. The cells are therefore the same cells
        with the same values; only the ones outside the window are never
        materialized.

        Refused when: the corridor is switched off, the buffer is not
        explicit (the estimator needs the pixels — see
        :meth:`corridor_geometry`), the caller passed read options that the
        windowed read would silently drop, the data is already in memory, or
        the dataset has no windowed-read API.
        """
        if not getattr(self, "corridor_first", True):
            return False
        buffer_m = self._explicit_buffer_m
        if buffer_m is None or buffer_m <= 0:
            return False
        if read_kwargs:
            return False
        if getattr(dataset, "data", None) is not None:
            return False
        return (callable(getattr(dataset, "load_metadata", None))
                and callable(getattr(dataset, "read_window", None)))

    def _with_corridor_bounding_box(self, rasterize_kwargs: dict) -> dict:
        """Add ``bounding_box=`` to a rasterize() call when it is certified."""
        if rasterize_kwargs.get("bounding_box") is not None:
            return rasterize_kwargs
        bounding_box = self._certified_corridor_bounding_box(
            resolution_in_m=rasterize_kwargs.get("resolution_in_m", 1.0),
            geometry_buffer_m=rasterize_kwargs.get("geometry_buffer_m", 0),
            preprocessing_function=rasterize_kwargs.get(
                "preprocessing_function"),
        )
        if bounding_box is None:
            return rasterize_kwargs
        return {**rasterize_kwargs, "bounding_box": bounding_box}

    def _certified_corridor_bounding_box(
            self,
            resolution_in_m: float = 1.0,
            geometry_buffer_m: float = 0,
            preprocessing_function: Any | None = None,
    ):
        """Burn extent for the corridor, or None when it is not provably safe.

        Item 2.2: only the corridor is ever searched, yet the burn covers the
        whole data extent — 64 M cells against 13 M for a 5 km route in an
        8x8 km extent, 13x at municipality scale.

        The catch the plan does not mention is that
        ``GeoRasterizer.rasterize`` derives BOTH the output shape and the
        transform from the extent it is given, so a smaller extent normally
        lands on a different pixel grid — every cell boundary moves by a
        sub-pixel amount, class assignments flip along polygon edges, and the
        route changes. Shrinking the burn is only exact if the corridor grid
        is the same grid, restricted.

        So this method does not guess: it reconstructs the grid the
        full-extent burn WOULD have produced, snaps the corridor to whole
        cells of that grid, and then checks — before burning anything — that
        the extent it is about to pass reproduces
            * the same output shape as the snapped window, and
            * the same affine, to within
              :attr:`CORRIDOR_GRID_TOLERANCE_M` accumulated over the raster,
              and
            * a handler window covering the corridor raster completely,
        falling back to the unchanged full-extent burn if any check fails.
        The certificate is arithmetic on the header only; it never touches a
        pixel.

        Returns:
            A shapely box to pass as ``bounding_box=``, or None.
        """
        if not getattr(self, "corridor_first", True):
            return None
        buffer_m = self._explicit_buffer_m
        if buffer_m is None or buffer_m <= 0:
            # The buffer estimator samples the raster it is handed, at
            # indices clamped to its SHAPE — a smaller burn could change the
            # estimated buffer and therefore the answer.
            return None
        if preprocessing_function is not None:
            # The hook runs inside rasterize() and may move geometries, so
            # the bounds we would compute here are not the bounds it burns.
            return None
        corridor = self.corridor_geometry(buffer_m)
        if corridor is None or corridor.is_empty:
            return None

        rasterizer = self.geo_rasterizer
        try:
            base = getattr(rasterizer, "base_data", None)
            if base is None or len(base) == 0:
                return None
            crs = getattr(base, "crs", None)
            if crs is None or crs.is_geographic:
                # rasterize() sizes a geographic frame from a reprojected
                # copy but anchors the transform on the original bounds;
                # that path is inconsistent enough already without a second
                # extent in it.
                return None

            full_bounds = self._full_burn_bounds(base, geometry_buffer_m)
            full_shape = rasterizer._calculate_out_shape_from_geodataframe(
                GeoDataFrame(geometry=[box(*full_bounds)], crs=crs),
                resolution_in_m)
            rows, cols = int(full_shape[0]), int(full_shape[1])
            if rows <= 0 or cols <= 0:
                return None
            full_transform = from_bounds(*full_bounds, cols, rows)

            window = RasterHandler.window_from_bounds(
                corridor.bounds, full_transform, (rows, cols))
            width, height = int(window.width), int(window.height)
            if width <= 0 or height <= 0:
                return None
            if width >= cols and height >= rows:
                return None  # the corridor already covers the whole burn

            candidate = box(*window_bounds(window, full_transform))
            predicted_shape = (
                rasterizer._calculate_out_shape_from_bounding_box(
                    candidate, resolution_in_m))
            if (int(predicted_shape[0]), int(predicted_shape[1])) != (height,
                                                                      width):
                return None
            predicted = from_bounds(*candidate.bounds, width, height)
            expected = window_transform(window, full_transform)
            if not self._same_pixel_grid(predicted, expected, width, height):
                return None

            handler_window = RasterHandler.window_from_bounds(
                corridor.bounds, predicted, (height, width))
            if (int(handler_window.col_off), int(handler_window.row_off),
                    int(handler_window.width),
                    int(handler_window.height)) != (0, 0, width, height):
                return None
        except (AttributeError, TypeError, ValueError):
            # Any rasterizer/geometry shape we did not anticipate: burn the
            # full extent, exactly as before.
            return None
        return candidate

    @staticmethod
    def _full_burn_bounds(base, geometry_buffer_m: float):
        """Bounds ``rasterize()`` would compute for the full extent.

        It burns ``create_buffer(base_data, geometry_buffer_m)`` sorted by
        cost; sorting does not move bounds and the buffer call is the same
        one, so this is the identical frame's ``total_bounds``.
        """
        if geometry_buffer_m and geometry_buffer_m > 0:
            return tuple(base.buffer(geometry_buffer_m).total_bounds)
        return tuple(base.total_bounds)

    @classmethod
    def _same_pixel_grid(cls, predicted: Affine, expected: Affine,
                         cols: int, rows: int) -> bool:
        """Do two affines place every pixel of a cols x rows raster alike?"""
        tolerance = cls.CORRIDOR_GRID_TOLERANCE_M
        return (predicted.b == 0.0 and predicted.d == 0.0
                and expected.b == 0.0 and expected.d == 0.0
                and abs(predicted.c - expected.c) <= tolerance
                and abs(predicted.f - expected.f) <= tolerance
                and abs(predicted.a - expected.a) * cols <= tolerance
                and abs(predicted.e - expected.e) * rows <= tolerance)

    def create_raster_handler(
            self,
            cost_assumptions: CostAssumptionsType | None = None,
            datasets_to_modify: list[dict[str, Any]] | None = None,
            raster_save_path: str | None = None,
            dem_kwargs: dict[str, Any] | None = None,
            **kwargs
    ) -> RasterHandler:
        """
        Create a RasterReader object for the specified file and parameters.

        Parameters:
            cost_assumptions: Cost assumptions to use for rasterization.
                Required if dataset_source is vector data
            datasets_to_modify: List of datasets to use to modify the raster using
                GeoRasterizer.modify_raster_from_dataset
            raster_save_path: Path to save the raster dataset to.

        Returns:
            RasterReader: The created RasterReader object
        """
        # Using timed context manager instead of manual timing
        with timed("raster_loading", self.runtimes):
            if self.objective is not None:
                # Feasibility pipeline: MetricStack -> combine -> quantize
                self._create_metric_raster_handler(
                    cost_assumptions, datasets_to_modify, raster_save_path,
                    **kwargs)
                return self._finish_raster_handler_setup(dem_kwargs)
            # Check if we have vector data but no cost_assumptions
            if isinstance(self.dataset, VectorDataset) and cost_assumptions is None:
                msg = "Cost assumptions must be provided when using vector data"
                raise ValueError(msg)

            # Process the dataset based on its type and parameters
            if isinstance(self.dataset, VectorDataset) and cost_assumptions is not None:
                # Create a GeoRasterizer and rasterize the vector data
                self.geo_rasterizer = GeoRasterizer(self.dataset, cost_assumptions)
                # Item 2.2: burn the corridor instead of the data extent when
                # the corridor provably lands on the same pixel grid. Note
                # that a saved raster (raster_save_path / save_raster) then
                # covers the corridor rather than the whole input extent —
                # the searched area, which is what it was always meant to
                # document.
                self.geo_rasterizer.rasterize(
                    **self._with_corridor_bounding_box(kwargs))

                # Apply any additional dataset modifications
                if datasets_to_modify:
                    for dataset_params in datasets_to_modify:
                        self.geo_rasterizer.modify_raster_from_dataset(**dataset_params)

                if raster_save_path is not None:
                    self.geo_rasterizer.save_raster(save_path=raster_save_path)

                # Create RasterHandler with the rasterized data
                self.raster_handler = RasterHandler(
                    self.geo_rasterizer.raster_dataset,
                    self.source_coords,
                    self.target_coords,
                    self.search_space_buffer_m
                )
            elif isinstance(self.dataset, RasterDataset):
                if cost_assumptions is not None:
                    # If we have a raster but also cost assumptions, use GeoRasterizer
                    # to modify it
                    self.dataset.load_data(**kwargs)
                    self.geo_rasterizer = GeoRasterizer(self.dataset, cost_assumptions)

                    # Apply any additional dataset modifications
                    if datasets_to_modify:
                        for params in datasets_to_modify:
                            self.geo_rasterizer.modify_raster_from_dataset(**params)
                    if raster_save_path is not None:
                        self.geo_rasterizer.save_raster(raster_save_path)

                    # Create RasterHandler with the modified raster
                    self.raster_handler = RasterHandler(
                        self.geo_rasterizer.raster_dataset,
                        self.source_coords,
                        self.target_coords,
                        self.search_space_buffer_m
                    )
                else:
                    # Direct use of the raster without modifications. Item
                    # 2.2/2.3: leave the read to the handler when it can take
                    # the search window straight out of the file instead of
                    # loading the whole raster first.
                    if not self._can_read_source_window(self.dataset, kwargs):
                        self.dataset.load_data(**kwargs)

                    self.raster_handler = RasterHandler(
                        self.dataset,
                        self.source_coords,
                        self.target_coords,
                        self.search_space_buffer_m
                    )
                    if raster_save_path is not None:
                        self.raster_handler.save_section_as_raster(raster_save_path)
            else:
                raise ValueError(f"Unsupported dataset type: {type(self.dataset)}")
        return self._finish_raster_handler_setup(dem_kwargs)

    def _finish_raster_handler_setup(
            self,
            dem_kwargs: dict[str, Any] | None
    ) -> RasterHandler:
        """Shared tail of raster-handler creation: buffer warning + DEM."""
        if self.search_space_buffer_m is None:
            self.search_space_buffer_m = self.raster_handler.search_space_buffer_m
            shape = getattr(self.raster_handler, 'data', None)
            if shape is not None:
                shape = getattr(shape, 'shape', None)
            if shape is not None and len(shape) >= 2 and shape[0] * shape[1] > 1_000_000:
                rows, cols = shape[:2]
                import warnings
                warnings.warn(
                    f"No search_space_buffer_m set — using full raster "
                    f"({rows}x{cols} = {rows*cols:,} cells). This may cause "
                    f"excessive memory usage. Set search_space_buffer_m for "
                    f"production use.",
                    ResourceWarning,
                    stacklevel=2
                )
        # Create DEM raster handler if DEM dataset exists
        if self.dem_dataset is not None and isinstance(self.dem_dataset, RasterDataset):
            if dem_kwargs is None:
                dem_kwargs = {}

            # Load DEM data — unless the handler can read just the window
            # (item 2.2/2.3). The DEM handler uses the buffer the main
            # handler settled on, so its window is unchanged either way.
            if not self._can_read_source_window(self.dem_dataset, dem_kwargs):
                self.dem_dataset.load_data(**dem_kwargs)

            # Create DEM RasterHandler with same parameters
            self.dem_raster_handler = RasterHandler(
                self.dem_dataset,
                self.source_coords,
                self.target_coords,
                self.search_space_buffer_m,
                apply_mask=False
            )

        return self.raster_handler

    def create_graph(self, band_index: int = 0) -> Any:
        """
        Create a graph from the raster data.

        Timed as the "graph_build" phase of self.runtimes (see the class
        docstring); "import_time_graph_api", "edge_construction" and
        "graph_creation" decompose it.

        Parameters:
            band_index: Index of the raster band to use. Defaults to 0.

        Returns:
            The created graph object.
        """
        with timed("graph_build", self.runtimes):
            return self._build_graph(band_index)

    def _build_graph(self, band_index: int) -> Any:
        """Body of create_graph, wrapped by the graph_build timer."""
        # Importing the specified graph API using the timed context manager
        with timed("import_time_graph_api", self.runtimes):
            graph_api_class_constructor = get_graph_api_class(self.graph_api_name)

        # Get raster data for the specified band
        raster_data = self.raster_handler.data[band_index]

        # Validate raster size to prevent uint32 index overflow in Cython
        rows, cols = raster_data.shape[:2]
        total_cells = rows * cols
        if total_cells > MAX_SAFE_CELLS:
            raise ValueError(
                f"Raster has {total_cells:,} cells ({rows} x {cols}), which "
                f"exceeds the maximum of {MAX_SAFE_CELLS:,} cells supported "
                f"by uint32 indexing. Use a coarser resolution or a smaller "
                f"search area."
            )

        # Get DEM data if available
        if self.dem_raster_handler is not None:
            if len(self.dem_raster_handler.data.shape) > 2:
                dem_data = self.dem_raster_handler.data[0]
            else:
                dem_data = self.dem_raster_handler.data
        else:
            dem_data = None

        # Objective + DEM: replace the raw DEM with one reprojected onto
        # the search window grid and build the slope-response LUTs (the 3D
        # stretch applies unconditionally; responses when configured).
        gradient_luts = None
        # getattr: PathFinder is also constructed via __new__ (test doubles,
        # subclasses that skip super().__init__), where only the attributes a
        # caller sets exist.
        if (getattr(self, "objective", None) is not None
                and self.dem_raster_handler is not None):
            dem_data, gradient_luts = self._prepare_gradient_inputs(
                raster_data)

        # Build extra kwargs for backends that support them
        extra_kwargs = {}
        if self.graph_api_name == "cugraph":
            extra_kwargs["dem_kwargs"] = self.dem_kwargs
        elif self.graph_api_name not in ("cython",):
            extra_kwargs["use_gpu"] = self.use_gpu
            extra_kwargs["dem_kwargs"] = self.dem_kwargs
        if gradient_luts is not None:
            extra_kwargs["gradient_luts"] = gradient_luts
        if self.graph_api_name == "raster_fim" and dem_data is not None:
            # Tier A needs metres: the eikonal metric M = c^2 (I + grad_z
            # grad_z^T) is built from an elevation gradient in metres of
            # rise per metre of run. RasterFIMAPI cross-checks this
            # against GradientLUTs.inv_horiz_m.
            extra_kwargs["cell_size"] = float(
                abs(self.raster_handler.window_transform.a))

        # Create graph using the graph API
        self._graph_api = graph_api_class_constructor(raster_data,
                                                      self.steps,
                                                      ignore_max=self.ignore_max_cost,
                                                      dem_data=dem_data,
                                                      **extra_kwargs)

        # Save edge construction and graph creation times
        if (hasattr(self._graph_api, 'edge_construction_time') and
                hasattr(self._graph_api, 'graph_creation_time')):
            self.runtimes["edge_construction"] = self._graph_api.edge_construction_time
            self.runtimes["graph_creation"] = self._graph_api.graph_creation_time
            return self._graph_api.graph
        self.runtimes["edge_construction"] = 0.0
        self.runtimes["graph_creation"] = 0.0
        return None

    def find_route_ensemble(
            self,
            objectives: "dict[str, Objective | dict] | list",
            source: CoordinateInput | None = None,
            target: CoordinateInput | None = None,
            **kwargs
    ) -> "RouteEnsemble":
        """Route the same source-target pair under several objectives.

        Variants run SEQUENTIALLY (deliberately — routing shares the
        machine with other work); on the raster-direct backends each
        additional variant costs only combine + search, no graph rebuild.
        The finder's active objective is restored afterwards; its combine
        is deferred to the next access that needs the search raster, so a
        caller that only reads the ensemble pays no restore at all.

        Parameters:
            objectives: Mapping name -> Objective/weights-dict, or a list
                of objectives (auto-named ``variant_0`` ...).
            source / target: Optional single-pair override.
            **kwargs: Forwarded to :meth:`find_route`.

        Returns:
            :class:`pyorps.core.ensemble.RouteEnsemble` — use
            ``.to_dataframe()`` for the comparison table and
            ``.pareto_front([...])`` for the non-dominated subset.
        """
        from pyorps.core.ensemble import EnsembleError, RouteEnsemble

        if self.metric_stack is None:
            raise ValueError(
                "find_route_ensemble requires a PathFinder constructed "
                "with objective= (the metric pipeline is inactive).")
        if isinstance(objectives, list):
            objectives = {f"variant_{i}": obj
                          for i, obj in enumerate(objectives)}
        if not objectives:
            raise ValueError("objectives must not be empty")

        ensemble = RouteEnsemble()
        original_objective = self.objective
        try:
            for name, objective in objectives.items():
                self.set_objective(objective)
                path = self.find_route(source=source, target=target,
                                       **kwargs)
                if isinstance(path, PathCollection):
                    raise EnsembleError(
                        "find_route_ensemble supports a single "
                        "source-target pair — route multiple pairs in an "
                        "outer loop instead.")
                ensemble.add(name, path)
        finally:
            if (original_objective is not None
                    and self.objective is not original_objective):
                self._defer_objective_restore(original_objective)
        return ensemble

    def compare_optimal(
            self,
            metrics: tuple[str, ...] = ("cost",),
            source: CoordinateInput | None = None,
            target: CoordinateInput | None = None,
            **kwargs
    ):
        """Price the current objective against single-metric optima.

        Runs the current objective plus one pure ``{metric: 1.0}`` route
        per requested metric, and reports per-metric deltas — "your
        policy costs +X EUR and saves Y exposure" (plan section 9).

        Returns:
            DataFrame: variant rows plus one ``delta current - <m>``
            row per compared metric (positive = the current policy pays
            more of that metric than the single-metric optimum).
        """
        if self.objective is None:
            raise ValueError(
                "compare_optimal requires an active objective")
        variants = {"current": self.objective}
        for metric in metrics:
            variants[f"{metric}-optimal"] = {metric: 1.0}

        ensemble = self.find_route_ensemble(variants, source=source,
                                            target=target, **kwargs)
        table = ensemble.to_dataframe()

        metric_columns = [c for c in table.columns
                          if c in ensemble["current"].metrics]
        for metric in metrics:
            delta = (table.loc["current", metric_columns]
                     - table.loc[f"{metric}-optimal", metric_columns])
            table.loc[f"delta current - {metric}-optimal",
                      metric_columns] = delta
        return table

    def _prepare_gradient_inputs(self, raster_data):
        """Reproject the DEM onto the search window grid and build LUTs.

        Uses rasterio.warp.reproject (bilinear) with the true transforms of
        both grids — correct for any DEM resolution or extent, unlike the
        legacy shape-zoom approach. Non-finite DEM cells (nodata voids)
        make the corresponding weight cells forbidden and are filled with
        the mean height, so no NaN can ever reach a kernel.

        Returns:
            (dem_aligned float32, GradientLUTs)
        """
        from rasterio.warp import Resampling, reproject

        if self.dem_raster_handler is None:
            raise ValueError(
                "Gradient terms require a DEM — pass dem= to PathFinder.")

        dem_src = self.dem_raster_handler.data
        if dem_src.ndim > 2:
            dem_src = dem_src[0]
        dst_crs = self.raster_handler.raster_dataset.crs
        # Legacy assumption: an unspecified DEM CRS matches the raster CRS.
        src_crs = self.dem_dataset.crs or dst_crs
        dem_aligned = np.full(raster_data.shape, np.nan, dtype=np.float32)
        reproject(
            source=np.ascontiguousarray(dem_src, dtype=np.float32),
            destination=dem_aligned,
            src_transform=self.dem_raster_handler.window_transform,
            src_crs=src_crs,
            dst_transform=self.raster_handler.window_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )

        invalid = ~np.isfinite(dem_aligned)
        if invalid.any():
            finite = dem_aligned[~invalid]
            fill = float(finite.mean()) if finite.size else 0.0
            raster_data[invalid] = uint16(65535)  # forbid uncovered cells
            dem_aligned[invalid] = fill

        cell_size = float(abs(self.raster_handler.window_transform.a))
        quant_scale = (self._combine_result.scale
                       if self._combine_result is not None else 1.0)
        gradient_luts = self.objective.build_gradient_luts(
            self.steps, cell_size, quant_scale)
        self._gradient_dem = dem_aligned  # reused by the metric evaluator
        return dem_aligned, gradient_luts

    @property
    def graph_api(self) -> GraphAPI:
        """The backend handle, built on first use.

        Building is attributed to the "graph_build" phase and never to
        "shortest_path"; a route that reuses an existing graph is charged
        0.0 for it.
        """
        if self._graph_api is None:
            self.create_graph()
        else:
            self.runtimes["graph_build"] = 0.0
        return self._graph_api

    def release_device_resources(self, free_pool: bool = True) -> None:
        """Give back any GPU memory the backend is holding.

        The raster-direct GPU backends keep a device session alive between
        queries - that is the point of plan item 1.2, and it is worth
        roughly 18 B/cell (~0.42 GiB on a 25 M-cell window). Refcounting
        releases it when the PathFinder is dropped, but a long-lived owner
        never drops one: an interactive session holds a finder across
        minutes of idle time, pinning VRAM on a card shared with other
        work.

        Call this when a routing job is finished but the finder is being
        kept. The next query rebuilds the session and pays one re-upload;
        nothing else changes, and no result differs. Backends with no
        device state (cython, the graph libraries) are unaffected.
        """
        for session in getattr(self, "_search_sessions", []):
            close_session = getattr(session, "close", None)
            if close_session is not None:
                close_session()
        self._search_sessions = []
        api = self._graph_api
        close = getattr(api, "close", None)
        if close is None:
            return
        try:
            close(free_pool=free_pool)
        except TypeError:          # a backend whose close() takes no kwargs
            close()

    def search_session(self, algorithm: str = "dijkstra", **kwargs):
        """Open an incremental :class:`~pyorps.graph.search_session.SearchSession`.

        Retains the settled search field so later source/target/waypoint
        edits extract or resume instead of starting from scratch.
        ``find_route`` stays one-shot; this is the opt-in retention seam.
        """
        from pyorps.graph.search_session import SearchSession
        if self.raster_handler is None:
            self.create_raster_handler()
        session = SearchSession(self, algorithm=algorithm, **kwargs)
        sessions = getattr(self, "_search_sessions", None)
        if sessions is None:
            self._search_sessions = []
            sessions = self._search_sessions
        sessions.append(session)
        return session

    def cost_fields(self, origins, algorithm: str = "auto", **kwargs):
        """Settle ONE field per fixed terminal, then price every candidate.

        The entry point for substation siting and anything shaped like
        it. A candidate position may be anywhere; the turbines and grid
        connection points do not move, so the search is rooted at THEM
        and read in the other direction. Ten terminals against forty
        million candidates is ten searches, not forty million::

            with finder.cost_fields(turbines + pccs, labels=names) as f:
                costs = f.costs_to(candidates)     # (n_terminals, n)
                leg = f.path_to("WT0", best_site)  # cells, coords, m, EUR

        Fields are settled with delta-stepping unless told otherwise, and
        are held in memory when they fit and paged through disk when they
        do not -- see :class:`~pyorps.graph.search_session.CostFieldSet`
        for the labels, cache and spill controls.

        Use ``cost_field`` for a single root, and ``search_session`` when
        one route's control points are being dragged.
        """
        from pyorps.graph.search_session import CostFieldSet
        if self.raster_handler is None:
            self.create_raster_handler()
        fields = CostFieldSet(self, origins, algorithm=algorithm, **kwargs)
        sessions = getattr(self, "_search_sessions", None)
        if sessions is None:
            self._search_sessions = []
            sessions = self._search_sessions
        sessions.append(fields)
        return fields

    def cost_field(self, origin, algorithm: str = "auto", **kwargs):
        """Open a rooted :class:`~pyorps.graph.search_session.CostField`.

        One search field from a terminal that does NOT move, then O(1)
        pricing of arbitrarily many moving endpoints against it --
        substation siting, service areas, k-candidate ranking. The raster
        graph is undirected with symmetric weights, so a field rooted at
        the fixed terminal answers every query in both directions; the
        field checks that precondition and raises when it does not hold.

        Use ``search_session`` instead when the route is one polyline
        whose control points are being dragged.

        Parameters:
            origin: The fixed terminal, ``(x, y)`` in the raster CRS.
            algorithm: ``"auto"`` (default) picks the fastest FULL-FIELD
                kernel -- delta-stepping on the CPU, the GPU kernel when
                ``graph_api="raster_gpu"``. That is what a field is
                almost always for, and on 41.8 M cells at r2 on an idle
                16-core machine it is 4.7 s against 38.1 s for the
                serial Dijkstra, or 2.6 s on the GPU. The parallel
                kernel is exact: verified bit-identical at 1, 8 and 12
                threads and against the float64 Dijkstra over 35.6 M
                reachable cells. Pass ``"dijkstra"`` for the one case it
                does not cover -- pricing two or three candidates, where
                only Dijkstra can stop early.
            **kwargs: Passed to the kernel, e.g. ``delta=`` and
                ``num_threads=`` for delta-stepping. Under ``"auto"``
                the thread count defaults to three quarters of the
                cores: the kernel's bucket barrier collapses if no core
                is left free, and that leaves the machine usable.

        Registered on ``_search_sessions`` so
        ``release_device_resources`` frees it.
        """
        from pyorps.graph.search_session import CostField
        if self.raster_handler is None:
            self.create_raster_handler()
        field = CostField(self, origin, algorithm=algorithm, **kwargs)
        sessions = getattr(self, "_search_sessions", None)
        if sessions is None:
            self._search_sessions = []
            sessions = self._search_sessions
        sessions.append(field)
        return field

    def get_node_indices_from_coords(
            self,
            coords: CoordinateTuple | CoordinateList
    ) -> Node | NodeList | NodePathList:
        """
        Convert coordinates to node indices.

        Parameters:
            coords: Either:
                - A single coordinate pair (x, y)
                - A list of coordinate pairs [(x1, y1), (x2, y2), ...]

        Returns:
            List of node indices.
        """
        # Check if coords is a single coordinate pair and not a list
        if not isinstance(coords, list):
            coords = [coords]

        # Convert coordinates to 2D indices
        indices_2d = self.raster_handler.coords_to_indices(coords)

        # Correct positions with max cost if needed (using Numba-optimized method)
        indices_2d = self._correct_max_cost_positions(indices_2d)

        # Get shape of raster
        if len(self.raster_handler.data.shape) == 3:
            _, rows, cols = self.raster_handler.data.shape
        elif len(self.raster_handler.data.shape) == 2:
            rows, cols = self.raster_handler.data.shape
        else:
            raise RasterShapeError(self.raster_handler.data.shape)

        # Convert 2D indices to 1D node indices using ravel_multi_index
        node_indices = ravel_multi_index(
            (indices_2d[:, 0], indices_2d[:, 1]), (rows, cols))

        if len(coords) == 1:
            result = node_indices[0]
        else:
            result = node_indices
        return result

    def _correct_max_cost_positions(self, indices_2d: ndarray) -> ndarray:
        """
        Check and correct positions that have maximum cost value (uint16 max) using
        Numba-optimized functions.

        If positions in the raster have the maximum value (65535 for uint16),
        find the nearest position that doesn't have the maximum value.

        Parameters:
            indices_2d: Array of (row, col) indices to check and potentially correct

        Returns:
            Corrected array of (row, col) indices
        """
        if not self.ignore_max_cost:
            return indices_2d

        # Float32 weight rasters (Phase 9): forbidden = non-finite
        if np.issubdtype(self.raster_handler.data.dtype, np.floating):
            return self._correct_forbidden_positions_float(indices_2d)

        # Get the maximum value for uint16
        max_value = iinfo(uint16).max  # 65535

        # Get raster data (handle different shapes)
        if len(self.raster_handler.data.shape) == 3:
            raster_data = self.raster_handler.data[0]  # Use first band
            _, rows, cols = self.raster_handler.data.shape
        elif len(self.raster_handler.data.shape) == 2:
            raster_data = self.raster_handler.data
            rows, cols = self.raster_handler.data.shape
        else:
            raise RasterShapeError(self.raster_handler.data.shape)

        # Ensure indices are in the right format for Numba
        indices_2d = asarray(indices_2d, dtype=int32)

        # Check which positions have max values using Numba function [[11]]
        has_max_values, invalid_mask, invalid_indices = check_max_values(
            raster_data, indices_2d, max_value
        )

        if not has_max_values:
            return indices_2d

        # Find nearest valid positions for all invalid positions at once
        corrected_positions = find_nearest_valid_positions_numba(
            raster_data, invalid_indices, max_value
        )

        # Create corrected indices array
        corrected_indices = indices_2d.copy()

        # Collect corrections and emit a single summary warning
        corrections = []
        invalid_idx = 0
        for i in range(len(indices_2d)):
            if invalid_mask[i]:
                original_row, original_col = indices_2d[i]
                new_row, new_col = corrected_positions[invalid_idx]

                if new_row != original_row or new_col != original_col:
                    original_coords = self.raster_handler.indices_to_coords(
                        [(original_row, original_col)]
                    )[0]
                    corrected_coords = self.raster_handler.indices_to_coords(
                        [(new_row, new_col)]
                    )[0]

                    corrections.append((
                        original_coords, original_row, original_col,
                        corrected_coords, new_row, new_col,
                    ))
                    corrected_indices[i] = [new_row, new_col]

                invalid_idx += 1

        if corrections:
            header = (
                f"{len(corrections)} position(s) had maximum cost value "
                f"({max_value}) and were corrected:\n"
            )
            col_w = [14, 14, 14, 14]  # widths for table columns
            table = (
                f"  {'Orig Coords':>{col_w[0]}}  {'Orig Idx':>{col_w[1]}}  "
                f"{'Corr Coords':>{col_w[2]}}  {'Corr Idx':>{col_w[3]}}\n"
            )
            for (oc, or_, oc_, cc, nr, nc) in corrections:
                oc_str = f"({oc[0]:.1f}, {oc[1]:.1f})"
                cc_str = f"({cc[0]:.1f}, {cc[1]:.1f})"
                oi_str = f"[{or_}, {oc_}]"
                ci_str = f"[{nr}, {nc}]"
                table += (
                    f"  {oc_str:>{col_w[0]}}  {oi_str:>{col_w[1]}}  "
                    f"  ->  {cc_str:>{col_w[2]}}  {ci_str:>{col_w[3]}}\n"
                )
            warn(header + table, UserWarning, stacklevel=2)

        return corrected_indices

    def _correct_forbidden_positions_float(self, indices_2d: ndarray) -> ndarray:
        """Snap endpoints on non-finite (forbidden) float cells to the
        nearest finite cell — the float twin of the uint16 correction."""
        raster_data = self.raster_handler.data
        if raster_data.ndim == 3:
            raster_data = raster_data[0]
        corrected = asarray(indices_2d, dtype=int32).copy()
        finite_cells = None
        for i, (row, col) in enumerate(corrected):
            if np.isfinite(raster_data[row, col]):
                continue
            if finite_cells is None:
                finite_cells = np.argwhere(np.isfinite(raster_data))
                if finite_cells.size == 0:
                    return corrected
            d2 = ((finite_cells[:, 0] - row) ** 2
                  + (finite_cells[:, 1] - col) ** 2)
            nearest = finite_cells[int(d2.argmin())]
            warn(f"Position [{row}, {col}] lies on a forbidden cell and "
                 f"was corrected to [{nearest[0]}, {nearest[1]}].",
                 UserWarning, stacklevel=2)
            corrected[i] = nearest
        return corrected

    def get_coords_from_node_indices(
            self,
            node_indices: Node | NodeList,
    ) -> CoordinateList:
        """
        Convert node indices to coordinates.

        Parameters:
            node_indices: List of node indices.

        Returns:
            List of coordinates (x, y).
        """
        # Get shape of raster
        if len(self.raster_handler.data.shape) == 3:
            _, rows, cols = self.raster_handler.data.shape
        elif len(self.raster_handler.data.shape) == 2:
            rows, cols = self.raster_handler.data.shape
        else:
            raise RasterShapeError(self.raster_handler.data.shape)

        # Convert 1D indices to 2D indices using unravel_index
        indices_2d = array(unravel_index(node_indices, (rows, cols))).T

        # Convert 2D indices to coordinates
        coords = self.raster_handler.indices_to_coords(indices_2d)
        return coords

    def find_route(
            self,
            source: CoordinateInput | None = None,
            target: CoordinateInput | None = None,
            algorithm: str = "dijkstra",
            calculate_metrics: bool = True,
            pairwise: bool = False,
            raster_parameters: dict[str, Any] | None = None,
            **kwargs
    ) -> Path | PathCollection:
        """
        Find the shortest path between source and target coordinates.

        Parameter:
            source: CoordinateInput - Source coordinates. If None, uses the
                source_coords provided at initialization. Can be: tuple, list of
                tuples, array of arrays, shapely Point,
                shapely MultiPoint, GeoSeries of points, or GeoDataFrame of points.
            target: Target coordinates. If None, uses the target_coords provided at
                initialization. Can be a single pair (x, y) or a list of pairs
                [(x1, y1), (x2, y2), ...].
            algorithm: Algorithm to use for shortest path. Defaults to "dijkstra".
            calculate_metrics: Whether to calculate path metrics. Defaults to True.
            pairwise: Whether to calculate paths pairwise (requires equal number of
                sources and targets). Default is False.
        Returns:
            Path: When a single source-target pair is provided.
            PathCollection: When multiple source-target pairs or a single
                source with multiple targets are provided.
        """
        # Per-call objective override (kept out of the shortest_path kwargs)
        objective = kwargs.pop("objective", None)
        if objective is not None:
            self.set_objective(objective)

        # Get source and target coords
        if source is None:
            source = self.source_coords
        else:
            source = PathFinder.normalize_coordinates(source)

        if target is None:
            target = self.target_coords
        else:
            target = PathFinder.normalize_coordinates(target)

        if source is None or target is None:
            raise ValueError("Source and target coordinates must not be None!")

        if self.raster_handler is None:
            self.create_raster_handler(**(raster_parameters or {}))

        # Convert coordinates to node indices
        source_indices = self.get_node_indices_from_coords(source)
        target_indices = self.get_node_indices_from_coords(target)

        # Build the graph BEFORE the search timer starts — dereferencing the
        # property inside the timed block charged edge construction (minutes
        # on the library backends) to "shortest_path".
        graph_api = self.graph_api

        # Time the shortest path calculation
        self.runtimes["shortest_path_start_time"] = time()

        # Find the shortest path using the graph API
        with timed("shortest_path", self.runtimes):
            path_indices = graph_api.shortest_path(
                source_indices=source_indices,
                target_indices=target_indices,
                algorithm=algorithm,
                pairwise=pairwise,
                **kwargs
            )

        if len(path_indices) == 0:
            msg = (" In some cases, this happens if source or target are within a "
                   "pixel with max cost and ignore_max is set to True! "
                   "Either change the coordinates of source or target, change the "
                   "cost value to a vlue smaller than the maximum or set ignore_max "
                   "to False!")
            raise NoPathFoundError(source_indices, target_indices, msg)

        # Case 1: Single source, single target -> single path
        if (not isinstance(path_indices[0], list) and
                not isinstance(path_indices[0], ndarray)):
            return self._create_path_result(path_indices, source, target, algorithm,
                                            calculate_metrics)
        # Case 2 & 3: Multiple paths
        # For single source + multiple targets OR multiple sources +
        # multiple targets
        results = self._extract_path_results(path_indices, algorithm,
                                             calculate_metrics)
        return results

    def _extract_path_results(self, path_indices, algorithm, calculate_metrics):
        results = PathCollection()
        for path in path_indices:
            if not path or len(path) < 2:
                continue
            source = self.get_coords_from_node_indices(path[0])[0]
            target = self.get_coords_from_node_indices(path[-1])[0]
            path = self._create_path_result(path, source, target, algorithm,
                                            calculate_metrics)
            results.add(path)
        return results

    def _create_path_result(self, path_indices, source, target, algorithm,
                            calculate_metrics):
        """
        Helper method to create a path result dictionary from path indices.

        Parameters:
            path_indices: List of node indices for the path
            source: Source coordinate(s)
            target: Target coordinate(s)
            algorithm: The routing algorithm used
            calculate_metrics: Whether to calculate metrics

        Returns:
            Dictionary containing path information
        """
        # Convert path indices to coordinates
        path_coords = self.get_coords_from_node_indices(path_indices)
        if path_coords is None or len(path_coords) == 0:
            path_coords = [tuple(source), tuple(target)]
        elif len(path_coords) < 2:
            path_coords = [path_coords[0], path_coords[0]]

        # Calculate the Euclidean distance
        euclidean_distance = sqrt((path_coords[0][0] - path_coords[-1][0]) ** 2 +
                                  (path_coords[0][1] - path_coords[-1][1]) ** 2)

        # Create LineString from path coordinates
        path_geometry = LineString(path_coords)

        # Create path object using the Path dataclass
        path_id = len(self.paths)
        path = Path(
            source=source,
            target=target,
            algorithm=algorithm,
            graph_api=self.graph_api_name,
            path_indices=path_indices,
            path_coords=path_coords,
            path_geometry=path_geometry,
            euclidean_distance=euclidean_distance,
            runtimes=self.runtimes.copy(),
            path_id=path_id,
            search_space_buffer_m=self.search_space_buffer_m,
            neighborhood=self.neighborhood_str
        )

        # Attach cost labels if cost assumptions are available
        if self.geo_rasterizer is not None:
            path.cost_labels = (
                self.geo_rasterizer.cost_manager.build_cost_labels()
            )

        # Attach objective provenance when the metric pipeline is active
        if self.objective is not None and self._combine_result is not None:
            path.objective_spec = {
                **self.objective.to_dict(),
                "quantization_scale": self._combine_result.scale,
                "resolution": self._combine_result.resolution,
                "legacy_passthrough": self._combine_result.legacy_passthrough,
            }

        # Calculate path metrics if requested
        if calculate_metrics:
            with timed("path_metrics", self.runtimes):
                self.calculate_path_metrics(path_indices, path)
        else:
            self.runtimes["path_metrics"] = 0.0

        # Every phase counted exactly once, post-processing included; the
        # Path carries the completed accounting, not a mid-run snapshot.
        self.runtimes["total"] = sum(
            self.runtimes.get(phase, 0.0) for phase in RUNTIME_PHASES)
        path.runtimes = self.runtimes.copy()

        # Store path in PathCollection
        self.paths.add(path)

        return path

    def _total_cost_basis(self) -> tuple[str, list[str]]:
        """Describe what Path.total_cost reflects and where it diverges.

        total_cost is always recomputed as ``sum(cell value x 2D length)``
        over the search raster, with the 2D length in CRS units (metres).
        Even in the plain 2D discrete case that is the minimized quantity
        only up to the cell-size factor — the kernels accumulate in cell
        units (cost_factor = sqrt(dr^2+dc^2)) while reporting is metric,
        the same convention Path.feasibility already follows. The returned
        list names every term of the actual search objective the number
        does NOT contain.

        Returns:
            (basis, divergences) — *basis* names the quantity total_cost was
            computed from, *divergences* is empty when it is the quantity
            the search minimized.
        """
        basis = "sum(cell value x 2D length) over the search raster"
        divergences = []
        if self.graph_api_name == "raster_fim":
            divergences.append(
                "the eikonal field value T[target] that was actually "
                "minimized — the discrete recompute prices the continuous "
                "path upward")
        if self.dem_raster_handler is not None:
            term = ("the 3D length stretch and the slope response that the "
                    "search applied to every edge")
            if self.objective is None:
                term += " (Path.total_length is the 2D length here as well)"
            divergences.append(term)
        if self.objective is not None:
            scale = (self._combine_result.scale
                     if self._combine_result is not None else 1.0)
            divergences.append(
                f"the objective's units — the raster holds the quantized "
                f"combined surface (scale {scale:g}), so the number is "
                f"objective units x m, not EUR")
        return basis, divergences

    def _report_total_cost_basis(self, path, computed: bool) -> None:
        """Record (and once per finder, warn about) the total_cost basis.

        Changing what total_cost *means* is a product decision owned by the
        dual-metric plan (2026-06-09: construction_cost vs
        routing_objective). Until that lands the number stays exactly as it
        was; this only makes the mismatch visible instead of silent.
        """
        basis, divergences = self._total_cost_basis()
        if not computed:
            path.total_cost_basis = (
                "not computed (float32 weight raster) — see Path.metrics")
            return
        path.total_cost_basis = basis
        if not divergences or getattr(self, "_total_cost_basis_warned", False):
            return
        self._total_cost_basis_warned = True
        warn(f"Path.total_cost is a recompute of {basis} and does not "
             f"reflect: {'; '.join(divergences)}. Read Path.metrics / "
             f"Path.feasibility for what the search minimized. Every result "
             f"carries this statement on Path.total_cost_basis.",
             UserWarning, stacklevel=3)

    @staticmethod
    def _category_stamp(raster_data) -> tuple:
        """Identity of the exact buffer a category table was derived from."""
        return (raster_data.__array_interface__["data"][0],
                raster_data.shape, raster_data.strides,
                raster_data.dtype.str)

    def _cached_categories(self, raster_data):
        """The sorted unique raster values, or None when not cached.

        The table depends only on the raster, yet the reporting kernel
        rebuilt it for every path (plan item 1.1). Soundness rests on two
        things, not one: the key (buffer address plus shape/strides/dtype)
        and the unconditional drop in the ``raster_handler`` setter, which
        every handler replacement goes through. The setter is what defeats
        address reuse; the key alone cannot.

        In-place writes into a handler's buffer are NOT witnessed by the
        key. One exists: ``_prepare_gradient_inputs`` stamps 65535 into
        forbidden cells at graph-build time, adding a value to the raster's
        value set. It is safe only because graph build always precedes the
        first ``calculate_path_metrics`` on that buffer. Any new
        post-construction writer must call
        ``invalidate_category_cache()``, or reported metres will be
        silently dropped from ``length_by_category``.
        """
        cached = self._category_cache
        if cached is None:
            return None
        stamp, _array, categories = cached
        if stamp != self._category_stamp(raster_data):
            return None
        return categories

    def _store_categories(self, raster_data, categories) -> None:
        """Remember the category table for this exact raster buffer."""
        self._category_cache = (self._category_stamp(raster_data),
                                raster_data, categories)

    def invalidate_category_cache(self) -> None:
        """Forget the cached category table.

        Call this after writing into the search raster in place; the cache
        key cannot witness such an edit.
        """
        self._category_cache = None

    def calculate_path_metrics(self, path_indices, path):
        """
        Calculate metrics about the path and add directly to the Path object.

        Sets Path.total_cost_basis: total_cost is recomputed from the 2D
        search raster and is NOT the value the search minimized whenever a
        DEM, a gradient response, an objective quantization or the
        continuous eikonal backend is in play.

        Parameters:
            path_indices: List of node indices for the path.
            path: Path object to update with metrics.
        """
        # Ensure path_indices is a numpy array
        path_indices = array(path_indices, dtype=uint32)

        # Get the raster data (costs)
        raster_data = self.raster_handler.data[0]
        cols = raster_data.shape[1]
        rows_idx = path_indices // cols
        cols_idx = path_indices % cols

        # Legacy category metrics require the uint16 raster; the float32
        # precision mode (Phase 9) relies on the evaluator instead.
        if raster_data.dtype == np.uint16:
            # Calculate metrics using Numba-accelerated function; the
            # category table is derived once per raster, not per path.
            categories = self._cached_categories(raster_data)
            total_length, cat, length = calculate_path_metrics_numba(
                raster_data, path_indices, categories)
            self._store_categories(raster_data, cat)

            # The kernel is pure cell space (1.0 per orthogonal step, sqrt(2)
            # per diagonal); it never sees the transform. Convert to CRS
            # units here so the legacy fields carry the same metre semantics
            # the evaluator branch already produces (see the cell_size
            # argument at _evaluate_objective_metrics below) - without it
            # total_length is a cell count printed with an "m" suffix and
            # total_cost is EUR/m x cells. Same square-cell assumption as
            # everywhere else in pyorps: distances use |transform.a|.
            cell_size = float(abs(self.raster_handler.window_transform.a))
            path.total_length = total_length * cell_size
            # Only the lengths. `cat` is the per-raster category table that
            # _store_categories just cached and every later path over this
            # buffer reuses, so scaling it would corrupt them all; it holds
            # cost values anyway, which are already per-metre.
            length = length * cell_size

            # Convert to regular Python dictionary
            path.length_by_category = dict(zip(cat, length))
            tot = path.total_length
            l_by_cat = path.length_by_category.items()
            # Calculate percentages
            path.length_by_category_percent = {
                k: (v / tot) * 100 if tot > 0 else 0
                for k, v in l_by_cat}

            # Calculate total cost (distance-weighted: category × length)
            path.total_cost = sum(cat * length for cat, length in l_by_cat)

            # Calculate raw cell cost (sum of raster values along path)
            path.total_cell_cost = float(
                raster_data[rows_idx, cols_idx].sum())

        self._report_total_cost_basis(
            path, computed=raster_data.dtype == np.uint16)

        # Feasibility-objective reporting: honest per-metric totals from
        # the float layers (the legacy fields above stay untouched).
        if self.objective is not None and self.metric_stack is not None:
            self._evaluate_objective_metrics(rows_idx, cols_idx, path)

    def _evaluate_objective_metrics(self, rows_idx, cols_idx, path):
        """Populate Path.metrics/feasibility from the metric stack."""
        from pyorps.utils.metric_eval import evaluate_path_metrics

        sub = self.metric_stack.window(self.raster_handler.window)
        layers = {name: sub[name] for name in sub.layer_names}
        dem = self._gradient_dem  # aligned at graph creation (or None)

        evaluation = evaluate_path_metrics(
            rows_idx, cols_idx, layers, self.objective,
            cell_size=float(abs(self.raster_handler.window_transform.a)),
            dem=dem,
            category=sub.category,
            category_labels=sub.category_labels,
        )

        path.metrics = evaluation.metrics
        path.feasibility = evaluation.feasibility
        path.total_length_2d = evaluation.total_length_2d
        path.total_length_3d = evaluation.total_length_3d
        path.mean_gradient_pct = evaluation.mean_gradient_pct
        path.max_gradient_pct = evaluation.max_gradient_pct
        path.length_by_class = evaluation.length_by_class or None
        if dem is not None or path.total_length is None:
            # v3 plan section 0: with a DEM the reported length is the
            # true 3D length (2D retained in total_length_2d). In float
            # precision mode the legacy 2D metric is skipped entirely,
            # so the evaluator supplies the length either way.
            path.total_length = evaluation.total_length_3d

    # ------------------------------------------------------------- corridors
    #
    # NOTE ON THE WORD "CORRIDOR". Everywhere else in this class a corridor is
    # the SEARCH-WINDOW polygon around the source/target set
    # (`corridor_geometry`, `corridor_bounds`, `corridor_first`,
    # `_certified_corridor_bounding_box`). The three methods below use the
    # other sense: a corridor is a stretch of ground that several routes have
    # in common, i.e. one trench serving more than one connection. They have
    # nothing to do with the search window.

    def build_corridor_graph(
            self,
            terminals: CoordinateInput | None = None,
            *,
            k_per_pair: int = 1,
            min_shared_length_m: float = 0.0,
            max_pair_distance: float | None = None,
            validate_pair_costs: bool = False,
    ) -> "CorridorGraph":
        """Derive the graph of shared trenches between a set of terminals.

        Not the search window (see the note above this method): the object
        returned here has the terminals plus every cell where least-cost routes
        MERGE or DIVERGE as its nodes, and the stretches between them as its
        edges, each carrying its construction cost once however many
        connections run over it.

        One multi-source sweep seeds every terminal at distance 0 and labels
        each cell with the terminal that reaches it most cheaply, which
        partitions the raster into routing-metric Voronoi regions. Every step
        whose two ends fall in different regions is a candidate connection
        priced at ``dist[u] + step(u->v) + dist[v]``; keeping the cheapest per
        terminal pair is Mehlhorn's terminal distance network [1]. The routes
        those steps stand for are then overlaid and cut where their membership
        changes, which is the contraction.

        There is no tolerance parameter anywhere in the construction. A raster
        route is an ordered list of cells over one grid, so "these two routes
        coincide here" is an identity, not a proximity test - which is the
        property a corridor definition derived from terrain has to have if it
        is to be reproducible.

        Parameters:
            terminals: Points to connect. Accepts everything ``source``/
                ``target`` accept. When None, the finder's own source and
                target coordinates are concatenated.
            k_per_pair: Keep the k cheapest boundary steps per terminal pair
                rather than only the cheapest. Note what that actually buys:
                the k cheapest crossings of ONE Voronoi boundary are usually
                adjacent cells on the same stretch, so the extra routes are
                near-duplicates of the first, not genuinely different
                corridors, and they inflate the segment count without adding
                structure. Raise it only when the boundary is long and you
                want the optimiser to see more than one crossing of it.
            min_shared_length_m: Discard a shared run shorter than this and
                revert those steps to per-route membership, so that two routes
                grazing for a couple of cells do not generate a junction. 0
                (the default) keeps the construction exact.
            max_pair_distance: Drop candidate connections costing more than
                this (in the search's CELL units). None keeps all of them.
            validate_pair_costs: Also run the pairwise search for every
                recovered pair and record the gap in ``provenance``. The
                distance network is an approximation of the true pairwise
                metric and the size of that gap is a reportable number, not a
                bug - but it costs one Dijkstra per terminal.

        Returns:
            A :class:`~pyorps.core.corridor.CorridorGraph` with
            ``construction='distance_network'``.

        Raises:
            NotImplementedError: on any backend but ``cython``. ``raster_gpu``
                would need multi-source seeding in ``GpuSsspSession.solve``
                (it takes one source today) and ``raster_fim`` would need a
                region label propagated through the FIM sweep kernels; both
                are real work, and falling back silently to a different edge
                model would make the corridor disagree with ``find_route``.

        References:
            [1] Mehlhorn, K.: 'A faster approximation algorithm for the Steiner
                problem in graphs', Inf. Process. Lett., 1988, 27, (3),
                pp. 125-128
        """
        from pyorps.graph.corridor import corridor_graph_from_routes

        if type(self) is not PathFinder:
            # ConstrainedPathFinder overrides find_route to run the
            # extended-state kernels and never touches self.graph_api, so the
            # backend check below would pass while this method quietly built a
            # plain terrain corridor with none of the span, angle, tower or
            # clearance constraints the subclass exists to enforce. Refuse
            # rather than export an unconstrained graph as if it were one.
            raise NotImplementedError(
                f"build_corridor_graph is implemented for PathFinder only; "
                f"{type(self).__name__} routes with a different edge model "
                f"(extended state, not a plain cell graph) and the "
                f"multi-source sweep has no equivalent for it. The corridor "
                f"would silently drop every constraint.")

        if self.graph_api_name != "cython":
            raise NotImplementedError(
                f"build_corridor_graph is implemented for the 'cython' "
                f"backend only; this finder uses '{self.graph_api_name}'. "
                f"The multi-source label propagation the construction needs "
                f"does not exist on the GPU backends (raster_gpu seeds one "
                f"source per solve, raster_fim carries no region label), and "
                f"substituting a different edge model would make the corridor "
                f"disagree with find_route.")

        if terminals is None:
            source = self.source_coords
            target = self.target_coords
            if source is None or target is None:
                raise ValueError(
                    "build_corridor_graph needs terminals: pass terminals= or "
                    "construct the finder with source and target coordinates.")
            merged: list = []
            for coords in (source, target):
                if PathFinder._is_single_coordinate(coords):
                    merged.append(tuple(coords))
                else:
                    merged.extend(tuple(c) for c in coords)
            terminal_coords = merged
        else:
            terminal_coords = PathFinder.normalize_coordinates(terminals)
            if PathFinder._is_single_coordinate(terminal_coords):
                terminal_coords = [tuple(terminal_coords)]

        # Deduplicate while preserving order: two terminals on one cell would
        # seed a region the second one can never own, and the numbering has to
        # stay a function of the input.
        seen: set = set()
        unique_coords = []
        for coord in terminal_coords:
            key = (float(coord[0]), float(coord[1]))
            if key in seen:
                continue
            seen.add(key)
            unique_coords.append(key)
        if len(unique_coords) < 2:
            raise ValueError(
                f"build_corridor_graph needs at least two distinct terminals, "
                f"got {len(unique_coords)}")

        if self.raster_handler is None:
            self.create_raster_handler()

        # Dereferenced BEFORE the corridor timer for the same reason find_route
        # does it: building the backend is the "graph_build" phase and must not
        # be charged to the corridor.
        graph_api = self.graph_api

        node_indices = self.get_node_indices_from_coords(unique_coords)
        terminal_cells = {i: int(cell) for i, cell in enumerate(node_indices)}
        if len(set(terminal_cells.values())) != len(terminal_cells):
            # Distinct coordinates can still land on one cell, and
            # _correct_max_cost_positions can push two of them onto the same
            # replacement. Collapse rather than fail: the caller gets a graph
            # whose terminal table says which inputs merged.
            collapsed: dict[int, int] = {}
            for tid in sorted(terminal_cells):
                cell = terminal_cells[tid]
                if cell not in collapsed.values():
                    collapsed[tid] = cell
            warn(f"{len(terminal_cells) - len(collapsed)} terminal(s) resolved "
                 f"onto a cell already taken by another terminal and were "
                 f"dropped from the corridor graph.", UserWarning, stacklevel=2)
            terminal_cells = collapsed

        with timed("corridor_build", self.runtimes):
            result = self._build_corridor_graph(
                graph_api, terminal_cells, k_per_pair, max_pair_distance,
                validate_pair_costs)
        routes, provenance, solver = result

        raster_data = self.raster_handler.data[0]
        graph = corridor_graph_from_routes(
            routes,
            raster_data,
            self.raster_handler.window_transform,
            crs=self.dataset.crs,
            steps=self.steps,
            max_value=IMPASSABLE_CELL_COST,
            ignore_max_cost=self.ignore_max_cost,
            dem=graph_api.dem_data,
            gradient_luts=graph_api.gradient_luts,
            terminal_cells=terminal_cells,
            min_shared_length_m=min_shared_length_m,
            price_routing_cost=True,
            # Reuse the sweep's own solver: a second one over the same window
            # would allocate another 17 B/cell and rescan the exclude mask for
            # nothing, and the search window here is routinely tens of
            # millions of cells.
            pricer=solver,
            provenance=provenance,
        )
        graph.construction = "distance_network"
        graph.runtimes = {"corridor_build": self.runtimes.get(
            "corridor_build", 0.0)}
        self.corridor_graph = graph
        return graph

    def _build_corridor_graph(self, graph_api, terminal_cells, k_per_pair,
                              max_pair_distance, validate_pair_costs):
        """Multi-source sweep, boundary reduction and route recovery.

        Split out of build_corridor_graph so the timer wraps exactly the work
        and nothing else.
        """
        from pyorps.utils._dijkstra import make_multi_source_solver

        terminal_ids = sorted(terminal_cells)
        seeds = np.array([terminal_cells[t] for t in terminal_ids],
                         dtype=np.uint32)

        max_value = (IMPASSABLE_CELL_COST if self.ignore_max_cost
                     else NO_EXCLUSION_VALUE)
        solver = make_multi_source_solver(
            graph_api.raster_data, self.steps, max_value=max_value,
            dem=graph_api.dem_data, gradient_luts=graph_api.gradient_luts)

        notes: list[str] = []
        if solver.has_zero_cost_steps():
            notes.append(
                "the search raster holds zero-cost cells, so the "
                "lexicographic tie-break is not provably independent of the "
                "terminal order here")

        settled = solver.solve(seeds)
        boundary = solver.boundary_steps()

        # k cheapest per unordered terminal pair.
        order = np.argsort(boundary["cost"], kind="stable")
        per_pair: dict[tuple[int, int], list[int]] = {}
        for position in order:
            cost = float(boundary["cost"][position])
            if max_pair_distance is not None and cost > max_pair_distance:
                break
            pair = (terminal_ids[int(boundary["region_u"][position])],
                    terminal_ids[int(boundary["region_v"][position])])
            bucket = per_pair.setdefault(pair, [])
            if len(bucket) < k_per_pair:
                bucket.append(int(position))

        routes: dict = {}
        recovered_costs: dict[tuple[int, int], float] = {}
        for pair, positions in per_pair.items():
            for rank, position in enumerate(positions):
                left = solver.path_to_root(
                    np.uint32(boundary["cell_u"][position]))
                right = solver.path_to_root(
                    np.uint32(boundary["cell_v"][position]))
                if left.size == 0 or right.size == 0:
                    continue
                cells = np.concatenate([left, right[::-1]])
                key = pair if rank == 0 else (*pair, rank)
                routes[key] = cells
                if rank == 0:
                    recovered_costs[pair] = float(boundary["cost"][position])

        if not routes:
            reason = (" No two terminals share a region boundary, so no pair "
                      "is connected in the search window. Widen the search "
                      "space buffer or check the exclusion mask.")
            if max_pair_distance is not None and boundary["cost"].size:
                reason = (f" Every candidate connection costs more than "
                          f"max_pair_distance={max_pair_distance} (the "
                          f"cheapest is {float(boundary['cost'].min()):.6g} "
                          f"in CELL units); raise it or pass None.")
            raise NoPathFoundError(
                list(terminal_cells.values()), list(terminal_cells.values()),
                reason)

        unreached = [t for t in terminal_ids
                     if not any(t in key[:2] for key in routes)]
        if unreached:
            notes.append(
                f"terminals {unreached} appear in no pair: they are isolated "
                f"in the search window, or every route to them runs through "
                f"another terminal's region")

        provenance: dict = {
            "backend": "cython",
            "neighborhood": self.neighborhood_str,
            "k_per_pair": int(k_per_pair),
            "settled_cells": int(settled),
            "boundary_candidates": int(len(boundary["cost"])),
            "n_terminals": len(terminal_ids),
            "n_pairs": len(recovered_costs),
            "notes": notes,
        }

        if validate_pair_costs:
            provenance["pair_cost_gap"] = self._pair_cost_gap(
                terminal_cells, recovered_costs, graph_api, max_value)

        return routes, provenance, solver

    def _pair_cost_gap(self, terminal_cells, recovered_costs, graph_api,
                       max_value):
        """How much the distance network overprices each recovered pair.

        The distance network connects a and b through the cheapest step on
        their shared Voronoi boundary. That is a real route, so it is an upper
        bound on the true a-b distance, and it is TIGHT whenever the true
        shortest path crosses the bisector - which it does for any pair whose
        regions touch along that path. Where it is not tight the gap is a
        property of the construction worth reporting, not a defect.
        """
        from pyorps.utils._dijkstra import make_dijkstra_solver

        reference = make_dijkstra_solver(
            graph_api.raster_data, self.steps, max_value=max_value,
            dem=graph_api.dem_data, gradient_luts=graph_api.gradient_luts)

        gaps: dict[str, float] = {}
        by_source: dict[int, list[int]] = {}
        for a, b in recovered_costs:
            by_source.setdefault(a, []).append(b)

        for a, others in by_source.items():
            reference.reset_root(np.uint32(terminal_cells[a]))
            for b in others:
                reference.search_until(np.uint32(terminal_cells[b]))
                true_cost = float(reference.peek_dist(
                    np.uint32(terminal_cells[b])))
                if not np.isfinite(true_cost) or true_cost <= 0.0:
                    continue
                excess = (recovered_costs[(a, b)] - true_cost) / true_cost
                gaps[f"{a}-{b}"] = excess

        # NOT floored at 0: a negative excess would mean the "upper bound"
        # came back cheaper than the true shortest path, which is impossible
        # for a real route and is exactly the breakage worth witnessing.
        values = list(gaps.values())
        return {"per_pair": gaps,
                "max_excess": max(values) if values else 0.0,
                "min_excess": min(values) if values else 0.0,
                "mean_excess": float(np.mean(values)) if values else 0.0}

    def save_corridor_graph(self, save_file_path: str | None = None,
                            corridor_graph: "CorridorGraph | None" = None
                            ) -> None:
        """Write the corridor graph as two layers of one GeoPackage.

        Layers ``corridor_segments`` (one row per trench, with its length,
        construction cost and how many routes use it) and ``corridor_nodes``
        (terminals and derived junctions). Mirrors :meth:`save_paths`.
        """
        graph = corridor_graph or getattr(self, "corridor_graph", None)
        if graph is None:
            raise ValueError(
                "No corridor graph to save - call build_corridor_graph first "
                "or pass corridor_graph=.")
        if not save_file_path:
            return
        graph.save(save_file_path)

    def plot_corridor_graph(
            self,
            corridor_graph: "CorridorGraph | None" = None,
            ax=None,
            figsize: tuple[int, int] = (12, 10),
            show_raster: bool = True,
            shared_color: str = "#c1121f",
            single_color: str = "#4c6ef5",
            junction_color: str = "#c1121f",
            terminal_color: str = "#111111",
            title: str | None = None,
    ):
        """Draw the corridor graph over the search raster.

        Shared segments are drawn thicker and in ``shared_color``, and the
        derived junctions are marked: that picture is the whole argument, so it
        is worth one method. Mirrors :meth:`plot_paths` in taking an optional
        axes and returning it.
        """
        import matplotlib.pyplot as plt

        from pyorps.core.corridor import NODE_TERMINAL

        graph = corridor_graph or getattr(self, "corridor_graph", None)
        if graph is None:
            raise ValueError(
                "No corridor graph to plot - call build_corridor_graph first "
                "or pass corridor_graph=.")

        if ax is None:
            _fig, ax = plt.subplots(figsize=figsize)

        if show_raster and self.raster_handler is not None:
            raster_data = self.raster_handler.data[0]
            left, bottom, right, top = window_bounds(
                self.raster_handler.window,
                self.raster_handler.raster_dataset.transform)
            values = np.asarray(raster_data, dtype=float)
            values[~np.isfinite(values)] = np.nan
            # Forbidden cells are drawn as their own class. Leaving them in the
            # ramp makes 65535 the whole dynamic range and flattens every real
            # cost difference to white, which is exactly the terrain the reader
            # is being asked to look at.
            forbidden = values >= float(IMPASSABLE_CELL_COST)
            passable = values.copy()
            passable[forbidden] = np.nan
            if np.isfinite(passable).any():
                low, high = np.nanpercentile(passable, (2, 98))
                if high <= low:
                    low, high = np.nanmin(passable), np.nanmax(passable)
            else:
                low, high = 0.0, 1.0
            ax.imshow(passable, extent=(left, right, bottom, top),
                      cmap="Greys", interpolation="nearest", alpha=0.6,
                      vmin=low, vmax=high)
            if forbidden.any():
                overlay = np.zeros(values.shape + (4,), dtype=float)
                overlay[forbidden] = (0.35, 0.10, 0.10, 0.55)
                ax.imshow(overlay, extent=(left, right, bottom, top),
                          interpolation="nearest")

        widest = max((s.use_count for s in graph.segments.values()),
                     default=1)
        for segment in graph.segments.values():
            xs, ys = zip(*segment.coords)
            shared = segment.use_count > 1
            ax.plot(xs, ys,
                    color=shared_color if shared else single_color,
                    linewidth=1.2 + 2.2 * (segment.use_count - 1) / max(
                        widest - 1, 1),
                    solid_capstyle="round", zorder=3 if shared else 2)

        for node in graph.nodes.values():
            if node.kind == NODE_TERMINAL:
                ax.plot(node.coords[0], node.coords[1], marker="s",
                        color=terminal_color, markersize=7, zorder=5)
            else:
                ax.plot(node.coords[0], node.coords[1], marker="o",
                        color=junction_color, markersize=5,
                        markeredgecolor="white", markeredgewidth=0.8, zorder=4)

        ax.set_title(title or (
            f"Corridor graph: {len(graph.segments)} segments, "
            f"{len(graph.junctions)} derived junctions"))
        ax.set_aspect("equal")
        return ax

    def get_path(self, path_id=None, source=None, target=None):
        """
        Retrieve a stored path by ID, or by source AND target.

        Parameters:
            path_id: Numerical ID of the path
            source: Source coordinates to search for
            target: Target coordinates to search for

        Returns:
            Path object or None if not found
        """
        return self.paths.get(path_id, source, target)

    def create_path_geodataframe(self):
        """
        Create a GeoDataFrame containing all stored paths.

        Returns:
            GeoDataFrame containing path data, or None if no paths available
        """
        # Check if there are any paths
        if not self.paths:
            return None

        # Use the PathCollection method to get all path records
        records = self.paths.to_geodataframe_records()

        # Create GeoDataFrame directly from records
        self.path_gdf = GeoDataFrame(records, geometry="geometry", crs=self.dataset.crs)
        return self.path_gdf

    def save_paths(self, save_file_path: str | None = None) -> None:
        """
        Save all calculated paths to a file in a GIS-compatible format.

        This method creates a GeoDataFrame containing all paths from the PathCollection
        and saves it to the specified file. The file format is automatically determined
        from the file extension (e.g., '.shp' for Shapefile, '.gpkg' for GeoPackage).

        Parameters:
            save_file_path: Path to save the paths file. If None, no file is saved.
                Common formats include:
                - Shapefile (.shp)
                - GeoPackage (.gpkg)
                - GeoJSON (.geojson)
                - CSV (.csv)

        Returns:
            None


        Notes:
            - The saved file includes all path attributes (ID, length, cost data)
            - The geometries are saved as LineString features with the CRS from the
            source dataset
            - If no paths have been calculated, an empty GeoDataFrame will be created
            first
        """
        if self.path_gdf is None:
            self.create_path_geodataframe()
        if save_file_path is not None and save_file_path != '':
            self.path_gdf.to_file(save_file_path)

    def save_raster(self, save_path: str | None = None) -> None:
        """
        Save the raster data used for path calculations to a GeoTIFF file.

        This method exports the current raster data to the specified file location.
        The raster contains the cost values used for path calculations, including
        any modifications from additional datasets. The exported file includes
        complete geo referencing information and preserves the original CRS.

        Parameters:
            save_path: Path where the raster file should be saved. If None, uses
                the default filename "pyorps_raster.tiff" in the current directory.

        Returns:
            None

        Notes:
            - The saved raster includes all cost modifications from additional datasets
            - The file is saved in GeoTIFF format which preserves geo referencing
            information
            - If the PathFinder uses a GeoRasterizer, the complete raster is saved
            - Otherwise, only the section loaded in the RasterHandler is saved
            - For large areas, the resulting file size may be substantial
        """
        if save_path is None:
            save_path = "pyorps_raster.tiff"
        if self.geo_rasterizer is not None:
            self.geo_rasterizer.save_raster(save_path)
        else:
            self.raster_handler.save_section_as_raster(save_path)

    def plot_paths(self,
                   paths: Path | PathCollection | list[Path] | None = None,
                   plot_all: bool = True,
                   subplots: bool = True,
                   subplot_size: tuple[int, int] = (10, 8),
                   source_color: str = 'green',
                   target_color: str = 'red',
                   path_colors: str | list[str] | None = None,
                   source_marker: str = 'o',
                   target_marker: str = 'x',
                   path_line_width: int = 2,
                   show_raster: bool = True,
                   title: str | list[str] | None = None,
                   sup_title: str | None = None,
                   path_id: int | list[int] | None = None,
                   reverse_colors: bool = False) -> Any | list[Any]:
        """
        Plot paths with customizable styling and layout options.

        This method visualizes the calculated paths, allowing for detailed customization
        of the plot appearance. It delegates to the PathPlotter class to handle the
        actual visualization.

        Parameters:
            paths: Specific path(s) to plot. If None, uses all paths in this PathFinder
                instance. Can be a single Path object, a list of Path objects, or a
                PathCollection.
            plot_all: If True, plots all paths. If False, plots only the path with
                path_id.
            subplots: If True and multiple paths are plotted, creates separate subplots
                for each path.
            subplot_size: Size of each individual subplot in inches (width, height).
            source_color: Color for source markers.
            target_color: Color for target markers.
            path_colors: Colors for path lines. Can be a single color or a list of
                colors. If None, default color scheme is used.
            source_marker: Marker style for source points.
            target_marker: Marker style for target points.
            path_line_width: Line width for the paths.
            show_raster: Whether to display the raster data as background.
            title: Title for the plot or individual subplot titles if a list is
                provided.
            sup_title: Overall title for the figure (when using multiple subplots).
            path_id: ID of specific path to plot when plot_all is False.
                Can be a single ID or a list of IDs.
            reverse_colors: Whether to reverse the color scheme for raster data
                (dark=low cost, bright=high cost).

        Returns:
            The matplotlib axes object(s) for the plot. Returns a list of axes if
            multiple subplots are created, otherwise returns a single axes object.

        Runtime Notes:
            - The plotting operation itself is generally quick (0.1-0.5 seconds)
            - Most time is spent on data preparation in the initial PathFinder setup
            - When plotting many paths, using subplots=True can improve readability
            - Displaying the raster background (show_raster=True) adds minimal overhead
              once the PathFinder is initialized
        """
        from pyorps.utils.plotting import PathPlotter

        # Determine which paths to plot based on the input
        if paths is None:
            # Use all paths from this PathFinder instance
            path_collection = self.paths
        elif isinstance(paths, Path):
            # Create a collection with a single path
            path_collection = PathCollection()
            path_collection.add(paths)
        elif isinstance(paths, list):
            # Create a collection from a list of paths
            path_collection = PathCollection()
            for path in paths:
                path_collection.add(path, replace=False)
        else:
            # Assume it's already a PathCollection
            path_collection = paths

        # Create PathPlotter and delegate the plotting
        plotter = PathPlotter(paths=path_collection, raster_handler=self.raster_handler)
        return plotter.plot_paths(
            plot_all=plot_all,
            subplots=subplots,
            subplotsize=subplot_size,
            source_color=source_color,
            target_color=target_color,
            path_colors=path_colors,
            source_marker=source_marker,
            target_marker=target_marker,
            path_linewidth=path_line_width,
            show_raster=show_raster,
            title=title,
            suptitle=sup_title,
            path_id=path_id,
            reverse_colors=reverse_colors
        )
