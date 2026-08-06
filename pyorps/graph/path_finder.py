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
from rasterio.transform import Affine
from shapely.geometry import LineString, MultiPoint, Point

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
        "networkx" or "rustworkx). Respective graph library must be installed!
        Networkit is a dependency of pyorps and will be installed automatically.

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
            from pyorps.graph.api.cugraph_api import CuGraphAPI
            return CuGraphAPI
        case "raster_gpu":
            from pyorps.graph.api.raster_gpu_api import RasterGPUAPI
            return RasterGPUAPI
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
    """

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
            search_space_buffer_m: Buffer around the source and target coordinates in
                meters. If set to 0, the entire Raster will be considered!
                WARNING: Setting to None uses the full raster which can cause
                excessive memory usage and long computation times for large rasters.
                Always set an appropriate buffer for production use.
            neighborhood_str: Neighborhood type. Defaults to "r2".
            steps: Steps which define the neighborhood. If None,
                will be created from neighborhood_str.
            ignore_max_cost: Whether to ignore all cells in the raster
                which have the maximum cost value or not
            graph_api: Graph API to use.
                Available graph libraries:
                    "cython" (default), "networkit", "rustworkx", "igraph", "networkx"
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
                FLOAT_BACKENDS (library backends or raster_gpu).
            **kwargs: Additional keyword arguments to pass to the rasterize function
                of the RasterHandler (if a VectorDataset or a source to a VectorDataset
                has been provided with dataset_source) or to the load function of the
                RasterDataset (if a source to a RasterDataset has been provided with
                dataset_source).

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
        self.neighborhood_str = neighborhood_str
        self.graph_api_name = graph_api
        self.ignore_max_cost = ignore_max_cost
        self.use_gpu = use_gpu
        self.objective = self._normalize_objective(objective, gradient_options)
        self.metric_layers = metric_layers
        self.metric_stack = None
        self._combine_result = None
        self._gradient_dem = None
        if metric_layers and self.objective is None:
            raise ValueError(
                "metric_layers require an objective — pass objective= "
                "to activate the metric pipeline.")
        self.weight_precision = weight_precision
        self._validate_precision_support()

        if steps is None and neighborhood_str:
            directed = self.graph_api_name in ("cython", "raster_gpu", "raster_gpu_v3", "cugraph")
            self.steps = get_neighborhood_steps(neighborhood_str, directed=directed)
        else:
            self.steps = steps

        self.runtimes = {}
        self.paths = PathCollection()  # Initialize PathCollection instead of list

        # Initialize as None (to be lazily loaded/created)
        self.raster_handler = None
        self.geo_rasterizer = None
        self._graph_api = None
        self.path_gdf = None
        self.dem_dataset = None
        self.dem_raster_handler = None
        self.dem_kwargs = dem_kwargs

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
                      "raster_gpu")

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

    def _apply_objective_to_stack(self) -> None:
        """Combine the stack under the current objective, rebuild handler."""
        use_float = self.weight_precision == "float32"
        result = self.metric_stack.combine(self.objective,
                                           quantize=not use_float)
        self._combine_result = result
        self._gradient_dem = None  # rebuilt lazily at graph creation
        dataset = InMemoryRasterDataset(
            result.weights, self.metric_stack.crs, self.metric_stack.transform)
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

    def _create_metric_raster_handler(
            self,
            cost_assumptions: CostAssumptionsType | None,
            datasets_to_modify: list[dict[str, Any]] | None,
            raster_save_path: str | None,
            **kwargs
    ) -> None:
        """Metric-pipeline variant of create_raster_handler (objective set).

        Vector input is rasterized into a multi-band MetricStack; raster
        input becomes a zero-copy legacy alias stack. The stack stays at
        full extent and unmasked — the RasterHandler owns windowing and
        buffer masking of the combined weight raster, so objective changes
        recombine from pristine layers.
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
            allowed = {"resolution_in_m", "geometry_buffer_m",
                       "include_category", "preprocessing_function",
                       "preprocessing_kwargs"}
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
            if derive == "slope_from_dem":
                self._ensure_stack_dem()
                values = stack.derive_terrain_slope()
            elif derive is not None:
                raise ValueError(
                    f"Unknown derive '{derive}' for metric layer "
                    f"'{name}' (supported: 'slope_from_dem')")
            elif "source" in spec:
                values = self._load_layer_source(name, spec)
            else:
                raise ValueError(
                    f"metric_layers['{name}'] needs 'source' or 'derive'")

            stack.add_layer(name, values,
                            hard_max=spec.get("hard_max"),
                            hard_min=spec.get("hard_min"))

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

    def _load_layer_source(self, name: str, spec: dict) -> ndarray:
        """Load a prebuilt layer source onto the stack grid."""
        from pyorps.core.metric_stack import reproject_to_grid

        stack = self.metric_stack
        source = spec["source"]
        if isinstance(source, ndarray):
            if source.shape == stack.shape:
                return source
            if "transform" not in spec:
                raise ValueError(
                    f"metric_layers['{name}'] array shape {source.shape} "
                    f"differs from the stack {stack.shape} — provide "
                    f"'transform' (and optionally 'crs') for "
                    f"reprojection.")
            return reproject_to_grid(
                source, spec["transform"], spec.get("crs", stack.crs),
                stack.shape, stack.transform, stack.crs)
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
            return aligned
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
                self.geo_rasterizer.rasterize(**kwargs)

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
                    # Direct use of the raster without modifications
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

            # Load DEM data
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

        Parameters:
            band_index: Index of the raster band to use. Defaults to 0.

        Returns:
            The created graph object.
        """
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
        if (self.objective is not None
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
        The finder's active objective is restored afterwards.

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
            if original_objective is not None:
                self.set_objective(original_objective)
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
        if self._graph_api is None:
            self.create_graph()
            # Overwrite the shortest_path_start_time, to make sure, that graph creation
            # is not part of it
            self.runtimes["shortest_path_start_time"] = time()
        return self._graph_api

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

        # Time the shortest path calculation
        self.runtimes["shortest_path_start_time"] = time()

        # Find the shortest path using the graph API
        with timed("shortest_path", self.runtimes):
            path_indices = self.graph_api.shortest_path(
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
            if not path:
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

        # Calculate the Euclidean distance
        euclidean_distance = sqrt((path_coords[0][0] - path_coords[-1][0]) ** 2 +
                                  (path_coords[0][1] - path_coords[-1][1]) ** 2)

        # Create LineString from path coordinates
        path_geometry = LineString(path_coords)

        # Calculate total runtime based on the graph API used
        if self.graph_api_name == "cython":
            self.runtimes["total"] = self.runtimes.get("raster_loading", 0) + \
                                     self.runtimes.get("shortest_path", 0.0)
        else:
            self.runtimes["total"] = self.runtimes.get("raster_loading", 0) + \
                                     self.runtimes.get("graph_creation", 0) + \
                                     self.runtimes.get("edge_construction", 0) + \
                                     self.runtimes.get("import_time_graph_api", 0) + \
                                     self.runtimes.get("shortest_path", 0.0)

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

        # Store path in PathCollection
        self.paths.add(path)

        return path

    def calculate_path_metrics(self, path_indices, path):
        """
        Calculate metrics about the path and add directly to the Path object.

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
            # Calculate metrics using Numba-accelerated function
            path.total_length, cat, length = calculate_path_metrics_numba(
                raster_data, path_indices)

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
