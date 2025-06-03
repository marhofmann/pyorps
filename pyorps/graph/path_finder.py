from time import time
from typing import Optional, Union, Any, Generator
from contextlib import contextmanager

from numpy import array, ndarray, ravel_multi_index, unravel_index, sqrt, uint32 
from geopandas import GeoDataFrame, GeoSeries
from shapely.geometry import LineString, Point, MultiPoint
from rasterio.transform import Affine

# Project imports
from pyorps.core.path import Path, PathCollection
from pyorps.core.types import (BboxType, GeometryMaskType, InputDataType,
                               CostAssumptionsType, CoordinateInput, Node, NodeList,
                               NodePathList, NormalizedCoordinate, CoordinateTuple,
                               CoordinateList)
from pyorps.graph.api.graph_api import GraphAPI
from pyorps.raster.rasterizer import GeoRasterizer
from pyorps.raster.handler import RasterHandler
from pyorps.utils.neighborhood import get_neighborhood_steps
from pyorps.io.geo_dataset import initialize_geo_dataset, VectorDataset, RasterDataset
from pyorps.utils.traversal import calculate_path_metrics_numba


@contextmanager
def timed(name: str, timings_dict: Optional[dict[str, float]]) -> Generator:
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
    match graph_api:
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
            source_coords: Optional[CoordinateInput],
            target_coords: Optional[CoordinateInput],
            search_space_buffer_m: Optional[float] = None,
            neighborhood_str: Optional[Union[str, int]] = "r2",
            steps: Optional[ndarray[int]] = None,
            ignore_max_cost: bool =True,
            graph_api: str = "networkit",
            cost_assumptions: Optional[CostAssumptionsType] = None,
            datasets_to_modify: Optional[list[dict[str, Any]]] = None,
            crs: Optional[str] = None,
            bbox: Optional[BboxType] = None,
            mask: Optional[GeometryMaskType] = None,
            transform: Optional[Affine] = None,
            raster_save_path: Optional[str] = None,
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
            search_space_buffer_m: Buffer around the source and target
            coordinates in meters.
            neighborhood_str: Neighborhood type. Defaults to "r2".
            steps: Steps which define the neighborhood. If None,
                will be created from neighborhood_str.
            ignore_max_cost: Whether to ignore all cells in the raster
                which have the maximum cost value or not
            graph_api: Graph API to use.
                Available graph libraries:
                    "networkit" (default), "rustworkx", "igraph", "networkx"
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

        if steps is None and neighborhood_str:
            directed = True if self.graph_api_name == "cython" else False
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

        # Load the dataset
        self.dataset = initialize_geo_dataset(dataset_source, crs, bbox, mask,
                                              transform)
        if self.source_coords is not None and self.target_coords is not None:
            self.create_raster_handler(cost_assumptions, datasets_to_modify,
                                       raster_save_path, **kwargs)

    @staticmethod
    def normalize_coordinates(
            input_data: Optional[CoordinateInput]
    ) -> Optional[NormalizedCoordinate]:
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
        else:
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
        elif all(isinstance(item, Point) for item in input_data):
            return [(point.x, point.y) for point in input_data]
        elif all(isinstance(item, MultiPoint) for item in input_data):
            return [(p.x, p.y) for item in input_data for p in item.geoms]
        else:
            raise ValueError("Input data cannot be interpreted as coordinates")

    def create_raster_handler(
            self,
            cost_assumptions: Optional[CostAssumptionsType] = None,
            datasets_to_modify: Optional[list[dict[str, Any]]] = None,
            raster_save_path: Optional[str] = None,
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
            # Check if we have vector data but no cost_assumptions
            if isinstance(self.dataset, VectorDataset) and cost_assumptions is None:
                msg = "Cost assumptions must be provided when using vector data"
                raise ValueError(msg)

            # Process the dataset based on its type and parameters
            if isinstance(self.dataset, VectorDataset) and cost_assumptions is not None:
                # Create a GeoRasterizer and rasterize the vector data
                self.geo_rasterizer = GeoRasterizer(self.dataset, cost_assumptions)
                self.geo_rasterizer.rasterize(save_path=raster_save_path, **kwargs)

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
        if self.search_space_buffer_m is None:
            self.search_space_buffer_m = self.raster_handler.search_space_buffer_m
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

        # Create graph using the graph API
        self._graph_api = graph_api_class_constructor(raster_data, self.steps,
                                                      ignore_max=self.ignore_max_cost)
        # Save edge construction and graph creation times
        if (hasattr(self._graph_api, 'edge_construction_time') and
                hasattr(self._graph_api, 'graph_creation_time')):
            self.runtimes["edge_construction"] = self._graph_api.edge_construction_time
            self.runtimes["graph_creation"] = self._graph_api.graph_creation_time
            return self._graph_api.graph
        else:
            self.runtimes["edge_construction"] = 0.0
            self.runtimes["graph_creation"] = 0.0
            return None

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
            coords: Union[CoordinateTuple, CoordinateList]
    ) -> Union[Node, NodeList, NodePathList]:
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

        # Get shape of raster
        _, rows, cols = self.raster_handler.data.shape

        # Convert 2D indices to 1D node indices using ravel_multi_index
        node_indices = ravel_multi_index(
            (indices_2d[:, 0], indices_2d[:, 1]), (rows, cols))

        if len(coords) == 1:
            result = node_indices[0]
        else:
            result = node_indices
        return result

    def get_coords_from_node_indices(
            self,
            node_indices: Union[Node, NodeList],
    ) -> CoordinateList:
        """
        Convert node indices to coordinates.

        Parameters:
            node_indices: List of node indices.

        Returns:
            List of coordinates (x, y).
        """
        # Get shape of raster
        _, rows, cols = self.raster_handler.data.shape

        # Convert 1D indices to 2D indices using unravel_index
        indices_2d = array(unravel_index(node_indices, (rows, cols))).T

        # Convert 2D indices to coordinates
        coords = self.raster_handler.indices_to_coords(indices_2d)
        return coords

    def find_route(
            self,
            source: Optional[CoordinateInput] = None,
            target: Optional[CoordinateInput] = None,
            algorithm: str = "dijkstra",
            calculate_metrics: bool = True,
            pairwise: bool = False,
            raster_parameters: Optional[dict[str, Any]] = None,
            **kwargs
    ) -> Union[Path, PathCollection]:
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
            Dictionary or list of dictionaries containing path information
        """
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
            raise ValueError(f"Source and target coordinates must not be None!")

        if self.raster_handler is None:
            self.create_raster_handler(**raster_parameters)

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

        # Case 1: Single source, single target -> single path
        if not isinstance(path_indices[0], list):
            return self._create_path_result(path_indices, source, target, algorithm,
                                            calculate_metrics)
        else:
            # Case 2 & 3: Multiple paths
            # For single source + multiple targets OR multiple sources +
            # multiple targets
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
        euclidean_distance = sqrt((path_coords[0][0] - path_coords[0][1]) ** 2 +
                                  (path_coords[-1][0] - path_coords[-1][1]) ** 2)

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

        # Calculate metrics using Numba-accelerated function
        path.total_length, cat, length = calculate_path_metrics_numba(raster_data,
                                                                      path_indices)

        # Convert to regular Python dictionary
        path.length_by_category = dict(zip(cat, length))
        tot = path.total_length
        l_by_cat = path.length_by_category.items()
        # Calculate percentages
        path.length_by_category_percent = {k: (v / tot) * 100 if tot > 0 else 0
                                           for k, v in l_by_cat}

        # Calculate total cost
        path.total_cost = sum(cat * length for cat, length in l_by_cat)

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

    def save_paths(self, save_file_path: Optional[str] = None) -> None:
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

    def save_raster(self, save_path: Optional[str] = None) -> None:
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
                   paths: Optional[Union[Path, PathCollection, list[Path]]] = None,
                   plot_all: bool = True,
                   subplots: bool = True,
                   subplot_size: tuple[int, int] = (10, 8),
                   source_color: str = 'green',
                   target_color: str = 'red',
                   path_colors: Optional[Union[str, list[str]]] = None,
                   source_marker: str = 'o',
                   target_marker: str = 'x',
                   path_line_width: int = 2,
                   show_raster: bool = True,
                   title: Optional[Union[str, list[str]]] = None,
                   sup_title: Optional[str] = None,
                   path_id: Optional[Union[int, list[int]]] = None,
                   reverse_colors: bool = False) -> Union[Any, list[Any]]:
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
