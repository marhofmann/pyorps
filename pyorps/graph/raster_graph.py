import time
from typing import Optional
from contextlib import contextmanager
from importlib import import_module

import numpy as np
import geopandas as gpd
from shapely.geometry import LineString
from rasterio.transform import Affine


# Project imports
from ..core.path import Path, PathCollection
from ..core.types import BboxType, GeometryMaskType
from ..raster.rasterizer import GeoRasterizer
from ..raster.handler import RasterHandler
from ..utils.neighborhood import get_neighborhood_steps
from ..io.geo_dataset import get_geo_dataset, VectorDataset, RasterDataset
from ..utils.traversal import calculate_path_metrics_numba


@contextmanager
def timed(name, timings_dict):
    """Simple context manager for timing code blocks."""
    start_time = time.time()
    try:
        yield
    finally:
        timings_dict[name] = time.time() - start_time


def get_graph_api_class(graph_api: str):
    """
    Dynamically import and return the graph API class based on the selected graph API.

    Args:
        graph_api (str): The name of the graph API to use (e.g., "networkit", "igraph", "cython").

    Returns:
        class: The corresponding graph API class.

    Raises:
        ImportError: If the specified graph API module cannot be imported.
        ValueError: If the specified graph API is not supported.
    """
    graph_api_classes = {
        "networkit": ("pyorps.graph.api.networkit_api", "NetworkitAPI"),
        "igraph": ("pyorps.graph.api.igraph_api", "IgraphAPI"),
        "rustworkx": ("pyorps.graph.api.rustworkx_api", "RustworkxAPI"),
        "networkx": ("pyorps.graph.api.networkx_api", "NetworkxAPI"),
        "cugraph": ("pyorps.graph.api.cugraph_api", "CugraphAPI"),
        "dask_cugraph": ("pyorps.graph.api.dask_cugraph_api", "DaskCugraphAPI"),
        "cython": ("pyorps.graph.api.cython_api", "CythonAPI"),
    }

    if graph_api not in graph_api_classes:
        raise ValueError(f"Unsupported graph API: {graph_api}")

    module_path, class_name = graph_api_classes[graph_api]
    try:
        module = import_module(module_path)
        return getattr(module, class_name)
    except ImportError as e:
        raise ImportError(f"Failed to import {graph_api}: {e}")


class RasterGraph:
    """
    A class that encapsulates RasterReader and graph-based routing capabilities.

    This class provides functionality to:
    1. Read raster data using RasterReader or create
    2. Create a graph representation of the raster
    3. Find shortest paths between coordinates
    4. Convert between coordinates and graph node indices
    5. Create GeoDataFrames of paths

    The class supports various graph APIs including a special "cython" mode that
    performs direct path finding on the raster without creating a graph.
    """

    def __init__(self,
                 dataset_source,
                 source_coords,
                 target_coords,
                 search_space_buffer_m,
                 neighborhood_str="r2",
                 steps=None,
                 graph_api="networkit",
                 cost_assumptions=None,
                 datasets_to_modify=None,
                 crs: Optional[str] = None,
                 bbox: Optional[BboxType] = None,
                 mask: Optional[GeometryMaskType] = None,
                 transform: Optional[Affine] = None,
                 **kwargs):
        """
        Initialize the RasterGraph with a dataset source and routing parameters.

        Args:
            dataset_source: Either:
                          - Path to a file (str)
                          - Tuple of (data_array, crs, transform)
                          - GeoDataset object
                          - Dictionary with url/layer for WFS
            source_coords (tuple or list): Either:
                - A single coordinate pair (x, y)
                - A list of coordinate pairs [(x1, y1), (x2, y2), ...]
            target_coords (tuple or list): Either:
                - A single coordinate pair (x, y)
                - A list of coordinate pairs [(x1, y1), (x2, y2), ...]
            search_space_buffer_m (float): Buffer around the source and target coordinates in meters.
            neighborhood_str (str, optional): Neighborhood type. Defaults to "r2".
            steps (ndarray, optional): Steps which define the neighborhood. If None, will be created from
            neighborhood_str.
            graph_api (str, optional): Graph API to use. Defaults to "networkit".
            cost_assumptions (optional): Cost assumptions to use for rasterization. Required if dataset_source is
            vector data.
            datasets_to_modify (list, optional): List of datasets to use to modify the raster using
            GeoRasterizer.modify_raster_from_dataset
        """
        self.source_coords = source_coords
        self.target_coords = target_coords
        self.search_space_buffer_m = search_space_buffer_m
        self.neighborhood_str = neighborhood_str
        self.graph_api_name = graph_api

        if steps is None and neighborhood_str:
            directed = True if self.graph_api_name == "cython" else False
            self.steps = get_neighborhood_steps(neighborhood_str, directed=directed)
        else:
            self.steps = steps

        self.runtimes = {}
        self.paths = PathCollection()  # Initialize PathCollection instead of list
        self.path_data = None  # For backward compatibility

        # Initialize as None (to be lazily loaded)
        self.raster_handler = None
        self.geo_rasterizer = None
        self._graph_api = None

        # Load the dataset
        self.dataset = get_geo_dataset(dataset_source, crs, bbox, mask, transform)
        self.create_raster_handler(cost_assumptions, datasets_to_modify, **kwargs)

    def create_raster_handler(self, cost_assumptions, datasets_to_modify, **kwargs):
        """
        Create a RasterReader object for the specified file and parameters.

        Returns:
            RasterReader: The created RasterReader object
        """
        # Using timed context manager instead of manual timing
        with timed("raster_loading", self.runtimes):
            # Check if we have vector data but no cost_assumptions
            if isinstance(self.dataset, VectorDataset) and cost_assumptions is None:
                raise ValueError("Cost assumptions must be provided when using vector data")

            # Process the dataset based on its type and parameters
            if isinstance(self.dataset, VectorDataset) and cost_assumptions is not None:
                # Create a GeoRasterizer and rasterize the vector data
                self.geo_rasterizer = GeoRasterizer(self.dataset, cost_assumptions)
                self.geo_rasterizer.rasterize(**kwargs)

                # Apply any additional dataset modifications
                if datasets_to_modify:
                    for dataset_params in datasets_to_modify:
                        self.geo_rasterizer.modify_raster_from_dataset(**dataset_params)

                # Create RasterHandler with the rasterized data
                self.raster_handler = RasterHandler(
                    self.geo_rasterizer.raster_dataset,
                    self.source_coords,
                    self.target_coords,
                    self.search_space_buffer_m
                )
            elif isinstance(self.dataset, RasterDataset):
                if cost_assumptions is not None:
                    # If we have a raster but also cost assumptions, use GeoRasterizer to modify it
                    self.geo_rasterizer = GeoRasterizer(self.dataset, cost_assumptions)

                    # Apply any additional dataset modifications
                    if datasets_to_modify:
                        for dataset_params in datasets_to_modify:
                            self.geo_rasterizer.modify_raster_from_dataset(**dataset_params)

                    # Create RasterHandler with the modified raster
                    self.raster_handler = RasterHandler(
                        self.geo_rasterizer.raster_dataset,
                        self.source_coords,
                        self.target_coords,
                        self.search_space_buffer_m
                    )
                else:
                    # Direct use of the raster without modifications
                    self.raster_handler = RasterHandler(
                        self.dataset,
                        self.source_coords,
                        self.target_coords,
                        self.search_space_buffer_m
                    )
            else:
                raise ValueError(f"Unsupported dataset type: {type(self.dataset)}")

        return self.raster_handler

    @property
    def raster(self):
        return self.raster_handler.data

    def create_graph(self, band_index: int = 0):
        """
        Create a graph from the raster data.

        Args:
            band_index (int, optional): Index of the raster band to use. Defaults to 0.

        Returns:
            The created graph object.
        """
        # Importing the specified graph API using the timed context manager
        with timed("import_time_graph_api", self.runtimes):
            graph_api_class_constructor = get_graph_api_class(self.graph_api_name)

        # Get raster data for the specified band
        raster_data = self.raster[band_index]

        # Create graph using the graph API
        self._graph_api = graph_api_class_constructor(raster_data, self.steps)
        # Save edge construction and graph creation times
        if hasattr(self._graph_api, 'edge_construction_time') and hasattr(self._graph_api, 'graph_creation_time'):
            self.runtimes["edge_construction"] = self._graph_api.edge_construction_time
            self.runtimes["graph_creation"] = self._graph_api.graph_creation_time
            return self._graph_api.graph
        else:
            self.runtimes["edge_construction"] = 0.0
            self.runtimes["graph_creation"] = 0.0
            return None

    @property
    def graph_api(self):
        if self._graph_api is None:
            self.create_graph()
            # Overwrite the shortest_path_start_time, to make sure, that graph creation is not part of it
            self.runtimes["shortest_path_start_time"] = time.time()
        return self._graph_api

    def get_node_indices_from_coords(self, coords):
        """
        Convert coordinates to node indices.

        Args:
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

        # Get shape of raster
        _, rows, cols = self.raster_handler.data.shape

        # Convert 2D indices to 1D node indices using ravel_multi_index
        node_indices = np.ravel_multi_index((indices_2d[:, 0], indices_2d[:, 1]), (rows, cols))
        return node_indices

    def get_coords_from_node_indices(self, node_indices):
        """
        Convert node indices to coordinates.

        Args:
            node_indices: List of node indices.

        Returns:
            List of coordinates (x, y).
        """
        # Get shape of raster
        _, rows, cols = self.raster_handler.data.shape

        # Convert 1D indices to 2D indices using unravel_index
        indices_2d = np.array(np.unravel_index(node_indices, (rows, cols))).T

        # Convert 2D indices to coordinates
        coords = self.raster_handler.indices_to_coords(indices_2d)
        return coords

    def find_route(self, source=None, target=None, algorithm="dijkstra", calculate_metrics=True, pairwise=False):
        """
        Find the shortest path between source and target coordinates.

        Args:
            source: Source coordinates. If None, uses the source_coords provided at initialization.
                Can be a single pair (x, y) or a list of pairs [(x1, y1), (x2, y2), ...].
            target: Target coordinates. If None, uses the target_coords provided at initialization.
                Can be a single pair (x, y) or a list of pairs [(x1, y1), (x2, y2), ...].
            algorithm: Algorithm to use for shortest path. Defaults to "dijkstra".
            calculate_metrics: Whether to calculate path metrics. Defaults to True.
            pairwise: Whether to calculate paths pairwise (requires equal number of sources and targets).
                Default is False.
        Returns:
            Dictionary or list of dictionaries containing path information
        """
        # Get source and target coords
        if source is None:
            source = self.source_coords
        if target is None:
            target = self.target_coords

        # Convert coordinates to node indices
        source_indices = self.get_node_indices_from_coords(source)
        target_indices = self.get_node_indices_from_coords(target)

        # Time the shortest path calculation
        self.runtimes["shortest_path_start_time"] = time.time()

        # Find shortest path using the graph API
        with timed("shortest_path", self.runtimes):
            path_indices = self.graph_api.shortest_path(
                source_indices=source_indices,
                target_indices=target_indices,
                algorithm=algorithm,
                pairwise=pairwise
            )

        # Check if we have multiple paths or a single path
        is_source_list = isinstance(source, list)
        is_target_list = isinstance(target, list)

        # Case 1: Single source, single target -> single path
        if not is_source_list and not is_target_list:
            return self._create_path_result(path_indices, source, target, algorithm, calculate_metrics)

        # Case 2 & 3: Multiple paths
        # For single source + multiple targets OR multiple sources + multiple targets
        results = []

        # If path_indices is a list of paths (multiple source-target pairs)
        if isinstance(path_indices, list) and all(
                isinstance(p, list) or isinstance(p, np.ndarray) for p in path_indices):
            for i, path in enumerate(path_indices):
                if not path:
                    continue
                source = self.get_coords_from_node_indices(path[0])[0]
                target = self.get_coords_from_node_indices(path[-1])[0]
                results.append(self._create_path_result(path, source, target, algorithm, calculate_metrics))
        else:
            # In case the graph API returns a single path even for multiple inputs
            results.append(self._create_path_result(path_indices, source, target, algorithm, calculate_metrics))

        return results

    def _create_path_result(self, path_indices, source, target, algorithm, calculate_metrics):
        """
        Helper method to create a path result dictionary from path indices.

        Args:
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

        # Calculate the euclidean distance
        euclidean_distance = np.sqrt((path_coords[0][0] - path_coords[0][1]) ** 2 +
                                     (path_coords[-1][0] - path_coords[-1][1]) ** 2)

        # Create LineString from path coordinates
        path_geometry = LineString(path_coords)

        # Calculate total runtime based on the graph API used
        if self.graph_api_name == "cython":
            self.runtimes["total"] = self.runtimes.get("raster_loading", 0) + \
                                     self.runtimes["shortest_path"]
        else:
            self.runtimes["total"] = self.runtimes.get("raster_loading", 0) + \
                                     self.runtimes.get("graph_creation", 0) + \
                                     self.runtimes.get("edge_construction", 0) + \
                                     self.runtimes.get("import_time_graph_api", 0) + \
                                     self.runtimes["shortest_path"]

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
            path_id=path_id
        )

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

        Args:
            path_indices: List of node indices for the path.
            path: Path object to update with metrics.
        """
        # Ensure path_indices is a numpy array
        path_indices = np.array(path_indices, dtype=np.uint32)

        # Get the raster data (costs)
        raster_data = self.raster_handler.data[0]

        # Calculate metrics using Numba-accelerated function
        path.total_length, categories, lengths = calculate_path_metrics_numba(raster_data, path_indices)

        # Convert to regular Python dictionary
        path.length_by_category = dict(zip(categories, lengths))
        tot = path.total_length
        l_by_cat = path.length_by_category.items()
        # Calculate percentages
        path.length_by_category_percent = {k: (v / tot) * 100 if tot > 0 else 0 for k, v in l_by_cat}

        # Calculate total cost
        path.total_cost = sum(cat * length for cat, length in l_by_cat)

    def get_path(self, path_id=None, source=None, target=None):
        """
        Retrieve a stored path by ID, or by source AND target.

        Args:
            path_id: Numerical ID of the path
            source: Source coordinates to search for
            target: Target coordinates to search for

        Returns:
            Path object or None if not found
        """
        return self.paths.get(path_id, source, target)

    def create_path_geodataframe(self, save_path=None):
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
        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=self.dataset.crs)
        if save_path is not None:
            gdf.to_file(save_path)
        return gdf

    def plot_paths(self, plot_all=True, subplots=True, subplotsize=(10, 8),
                   source_color='green', target_color='red', path_colors=None,
                   source_marker='o', target_marker='x', path_linewidth=2,
                   show_raster=True, title=None, path_id=None):
        from ..utils.plotting import PathPlotter

        plotter = PathPlotter(self.paths, self.raster_handler)
        plotter.plot_paths(plot_all=plot_all, subplots=subplots, subplotsize=subplotsize,
                           source_color=source_color, target_color=target_color, path_colors=path_colors,
                           source_marker=source_marker, target_marker=target_marker, path_linewidth=path_linewidth,
                           show_raster=show_raster, title=title, path_id=path_id)

