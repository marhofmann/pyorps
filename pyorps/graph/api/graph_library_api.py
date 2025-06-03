"""
This file contains the abstract base class for the interface to the graph libraries.
All specific graph library interfaces should inherit from this class. The workflow of
the specific interfaces are determined by the respective graph library. The workflow
of the graph libraries can vary!

- For rustworkx and igraph the nodes need to be created before the edges can be added
- For networkit and networkx the edges can be added on the fly when adding the nodes

- For rustworkx and igraph the edges can only be added as a list of tuples. This means
that the edge information as retrieved by numpy arrays, need to be converted into a
list, which leads to a much higher (more than double) memory usage!
- For networkit and networkx edges can be added as a sparse matrix or as numpy arrays

Please see the specific interfaces to the specific graph libraries for more details!
"""
from typing import Optional, Any, Union, List
from abc import abstractmethod
import numpy as np
from time import time

from .graph_api import GraphAPI
from pyorps.core.exceptions import NoPathFoundError, PairwiseError, PairwiseError
from pyorps.core.types import SourceTargetType, Node, NodeList, NodePathList
from pyorps.utils.traversal import construct_edges


class GraphLibraryAPI(GraphAPI):
    """
    Base class for all graph library-based APIs.

    This class extends GraphAPI with common functionality needed by standard graph
    libraries that require edge data to be explicitly provided and a graph to be
    constructed.
    """

    def __init__(
            self,
            raster_data: np.ndarray[int],
            steps: np.ndarray[int],
            from_nodes: Optional[np.ndarray] = None,
            to_nodes: Optional[np.ndarray] = None,
            cost: Optional[np.ndarray] = None,
            ignore_max: Optional[bool] = True,
            **kwargs
    ):
        """
        Initialize the graph library API.

        Parameters:
            raster_data: 2D numpy array representing the raster
            steps: Array defining the neighborhood connections
            from_nodes: Source node indices for edges
            to_nodes: Target node indices for edges
            cost: Edge weights
            ignore_max: Ignore edges whose weights are equal to the maximum value in
            the raster data
        """
        super().__init__(raster_data, steps)

        self.edge_construction_time = 0.0
        if from_nodes is None or to_nodes is None:
            before_constructing_edge_data = time()
            from_nodes, to_nodes, cost = construct_edges(
                self.raster_data,
                self.steps,
                ignore_max
            )
            self.edge_construction_time = time() - before_constructing_edge_data

        before_graph_creation = time()
        self.graph = self.create_graph(from_nodes, to_nodes, cost, **kwargs)
        self.graph_creation_time = time() - before_graph_creation

    @staticmethod
    def _ensure_path_endpoints(
            path: list[int],
            source: int,
            target: int
    ) -> list[int]:
        """
        Ensures the path starts with the source node and ends with the target node.

        Parameters:
            path: List of node IDs representing a path
            source: ID of the source node that should be at the start of the path
            target: ID of the target node that should be at the end of the path

        Returns:
            list of node IDs with source and target at endpoints if needed
        """
        if len(path) > 0:
            if path[0] != source:
                path.insert(0, source)
            if path[-1] != target:
                path.append(target)
        return path

    @abstractmethod
    def create_graph(
            self,
            from_nodes: np.ndarray[int],
            to_nodes: np.ndarray[int],
            cost: Optional[np.ndarray[int]] = None,
            **kwargs
    ) -> Any:
        """
        Creates a graph object with the graph library specified in the selected interface.

        Parameters:
            from_nodes: The starting node indices from the edge data
            to_nodes: The ending node indices from the edge data
            cost: The weight of the edge data
            kwargs: Additional parameters for the underlying graph library

        Returns:
            The graph object
        """

    @abstractmethod
    def get_number_of_nodes(self) -> int:
        """
        Returns the number of nodes in the graph.

        Returns:
            The number of nodes
        """

    @abstractmethod
    def get_number_of_edges(self) -> int:
        """
        Returns the number of edges in the graph.

        Returns:
            The number of edges
        """

    @abstractmethod
    def remove_isolates(self) -> None:
        """
        If the graph object was initialized with the maximum number of nodes, this
        function helps to reduce the occupied memory by removing nodes without any
        edge (degree == 0).

        Returns:
            None
        """

    @abstractmethod
    def get_nodes(self) -> Union[List[int], np.ndarray]:
        """
        This method returns the nodes in the graph as a list or numpy array of node
        indices.

        Returns:
            List or array of node indices of the nodes in the graph
        """

    def shortest_path(
            self,
            source_indices: Optional[SourceTargetType],
            target_indices: Optional[SourceTargetType],
            algorithm: str = "dijkstra",
            **kwargs
    ) -> Union[NodeList, NodePathList]:
        """
        This method applies the specified shortest path algorithm on the created graph
        object and finds the shortest path between source(s) and target(s) as a list of
        node indices.

        Parameters:
            source_indices: Index or indices of source node(s) (int or list[int])
            target_indices: Index or indices of target node(s) (int or list[int])
            algorithm: Algorithm to use for shortest path computation.
                Options depend on the specific library implementation.
            kwargs: Additional parameters for specific algorithms library including
            specific keywords and:
                "pairwise": If True, compute pairwise shortest paths between
                source_indices and target_indices.
                Only allowed if len(source_indices) == len(target_indices)

        Returns:
            List of node indices representing the shortest path(s)
        """
        source_has_len = hasattr(source_indices, '__len__')
        target_has_len = hasattr(target_indices, '__len__')

        # Single source, single target
        if not source_has_len and not target_has_len:
            return self._compute_single_path(source_indices,
                                             target_indices,
                                             algorithm,
                                             **kwargs)

        # Single source, multiple targets
        elif not source_has_len and target_has_len:
            return self._compute_single_source_multiple_targets(source_indices,
                                                                target_indices,
                                                                algorithm,
                                                                **kwargs)
        # Multiple sources, single target
        elif source_has_len and not target_has_len:
            paths = self._compute_single_source_multiple_targets(target_indices,
                                                                 source_indices,
                                                                 algorithm,
                                                                 **kwargs)
            return [p[::-1] for p in paths]

        # Multiple sources, multiple targets (all pairs or pairwise)
        else:
            # Check for pairwise computation
            pairwise = kwargs.get('pairwise', False)
            if pairwise:
                if len(source_indices) != len(target_indices):
                    raise PairwiseError()
                return self._pairwise_shortest_path(source_indices, target_indices,
                                                    algorithm, **kwargs)
            else:
                return self._all_pairs_shortest_path(source_indices, target_indices,
                                                     algorithm, **kwargs)

    @abstractmethod
    def _compute_single_path(
            self,
            source: Node,
            target: Node,
            algorithm: str,
            **kwargs
    ) -> NodeList:
        """
        Computes shortest path between a single source and target.

        Parameters:
            source: Source node identifier
            target: Target node identifier
            algorithm: Algorithm to use for computation
            kwargs: Additional algorithm-specific parameters

        Returns:
            List of node identifiers representing the shortest path
        """

    @abstractmethod
    def _compute_single_source_multiple_targets(
            self,
            source: Node,
            targets: NodeList,
            algorithm: str,
            **kwargs
    ) -> NodePathList:
        """
        Computes shortest paths from a single source to multiple targets.

        Parameters:
            source: Source node identifier
            targets: List of target node identifiers
            algorithm: Algorithm to use for computation
            kwargs: Additional algorithm-specific parameters

        Returns:
            List of paths from the source to each target
        """

    def _pairwise_shortest_path(
            self,
            sources: NodeList,
            targets: NodeList,
            algorithm: str,
            **kwargs
    ) -> NodePathList:
        """
        Default implementation for pairwise shortest path computation.
        Subclasses can override this for library-specific optimizations.

        Parameters:
            sources: List of source node identifiers
            targets: List of target node identifiers
            algorithm: Algorithm to use for computation
            kwargs: Additional algorithm-specific parameters

        Returns:
            List of paths, each connecting corresponding source-target pairs
        """
        paths = []
        for source, target in zip(sources, targets):
            try:
                path = self._compute_single_path(source, target, algorithm, **kwargs)
                paths.append(path)
            except NoPathFoundError:
                paths.append([])
        return paths

    def _compute_all_pairs_shortest_paths(
            self,
            sources: NodeList,
            targets: NodeList,
            algorithm: str,
            **kwargs
    ) -> NodePathList:
        """
        Computes paths individually for each source-target pair using the specified
        algorithm. Returns empty paths for unreachable targets.

        Parameters:
            sources: List of source node identifiers
            targets: List of target node identifiers
            algorithm: Algorithm to use for computation
            kwargs: Additional algorithm-specific parameters

        Returns:
            List of paths for all source-target combinations
        """
        paths = []
        for source in sources:
            for target in targets:
                try:
                    path = self._compute_single_path(source, target, algorithm,
                                                     **kwargs)
                    paths.append(path)
                except NoPathFoundError:
                    paths.append([])
        return paths

    @abstractmethod
    def _all_pairs_shortest_path(
            self,
            sources: NodeList,
            targets: NodeList,
            algorithm: str,
            **kwargs
    ) -> NodePathList:
        """
        Computes shortest paths between all pairs of sources and targets.

        Parameters:
            sources: List of source node identifiers
            targets: List of target node identifiers
            algorithm: Algorithm to use for computation.
            kwargs: Additional algorithm-specific parameters

        Returns:
            List of paths for all source-target combinations
        """

    def get_a_star_heuristic(
            self,
            target: Node,
            source: Optional[Node] = None,
            **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the A* heuristic based on the Euclidean distance from the target node.

        Parameters:
            target: The index of the target node in the raster data
            source: Optional source node for calculating area-specific minimum values
            kwargs: Additional parameters, including optional heu_weight for scaling

        Returns:
            tuple containing:
            - An array of node indices in the graph
            - An array of heuristic values corresponding to each node
        """
        # Retrieve the current nodes in the graph
        nodes = self.get_nodes()

        # Convert node indices to 2D coordinates (x, y) based on the raster data shape
        x_nodes, y_nodes = np.unravel_index(nodes, self.raster_data.shape)

        # Convert the target index to its corresponding 2D coordinates
        x_target, y_target = np.unravel_index(target, self.raster_data.shape)

        # Calculate the Euclidean distance from each node to the target node
        x_square = np.power(x_target - x_nodes, 2)
        y_square = np.power(y_target - y_nodes, 2)
        euclidean_distance = np.sqrt(x_square + y_square)

        # Use localized min value between source and target
        if source is not None:
            buffer_radius = kwargs.get('buffer_radius', 200)
            # Convert the source index to its 2D coordinates
            x_source, y_source = np.unravel_index(source, self.raster_data.shape)

            # Create a bounding box around the source-target line with buffer
            min_x = max(0, min(x_source, x_target) - buffer_radius)
            max_x = min(self.raster_data.shape[0] - 1,
                        max(x_source, x_target) + buffer_radius)
            min_y = max(0, min(y_source, y_target) - buffer_radius)
            max_y = min(self.raster_data.shape[1] - 1,
                        max(y_source, y_target) + buffer_radius)

            # Extract the subset of raster data within the bounding box
            subset_data = self.raster_data[min_x:max_x + 1, min_y:max_y + 1]

            # Use minimum value in the area
            min_value = np.min(subset_data)
        else:
            min_value = self.raster_data.min()

        # Calculate the heuristic by scaling the Euclidean distance
        heuristic = euclidean_distance * min_value

        # Apply any heuristic weight scaling
        if 'heu_weight' in kwargs:
            heuristic *= kwargs['heu_weight']

        return nodes, heuristic

    def get_advanced_a_star_heuristic(
            self,
            target: Node,
            source: Optional[Node] = None,
            **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the A* heuristic based on the Euclidean distance from the target node.

        Parameters:
            target: The index of the target node in the raster data
            source: Optional source node for calculating area-specific minimum values
            kwargs: Additional parameters, including optional heu_weight for scaling

        Returns:
            tuple containing:
            - An array of node indices in the graph
            - An array of heuristic values corresponding to each node
        """
        # Retrieve the current nodes in the graph
        nodes = self.get_nodes()

        # Convert node indices to 2D coordinates (x, y) based on the raster data shape
        x_nodes, y_nodes = np.unravel_index(nodes, self.raster_data.shape)

        # Convert the target index to its corresponding 2D coordinates
        x_target, y_target = np.unravel_index(target, self.raster_data.shape)

        # Calculate the Euclidean distance from each node to the target node
        euclidean_distance = np.sqrt(
            np.square(x_target - x_nodes) + np.square(y_target - y_nodes)
        )

        min_value = None

        # Use Bresenham's algorithm to find cells along the source-target line
        if source is not None:
            # Convert the source index to its 2D coordinates
            x_source, y_source = np.unravel_index(source, self.raster_data.shape)

            # Get line coordinates using optimized Bresenham implementation
            line_coords = self._vectorized_bresenham(x_source, y_source, x_target,
                                                     y_target)

            # If buffer is requested, expand the line cells
            if buffer_radius :=  kwargs.get('buffer_radius', 0):
                # Create a mask of the entire raster to track which pixels are in the
                # buffer. This avoids duplicates without needing to use np.unique
                mask = np.zeros(self.raster_data.shape, dtype=bool)

                # Extract x and y coordinates from line_coords
                x_line, y_line = line_coords[:, 0], line_coords[:, 1]

                # Create buffer offsets using meshgrid
                y_offsets, x_offsets = np.meshgrid(
                    np.arange(-buffer_radius, buffer_radius + 1),
                    np.arange(-buffer_radius, buffer_radius + 1)
                )

                # Reshape to 1D arrays
                x_offsets = x_offsets.flatten()
                y_offsets = y_offsets.flatten()

                # For each line point, add a buffer around it
                for x, y in zip(x_line, y_line):
                    # Calculate all buffer coordinates at once
                    buffer_x = x + x_offsets
                    buffer_y = y + y_offsets

                    # Filter out coordinates outside the raster bounds
                    valid_indices = (
                            (buffer_x >= 0) & (buffer_x < self.raster_data.shape[0]) &
                            (buffer_y >= 0) & (buffer_y < self.raster_data.shape[1])
                    )

                    # Mark valid buffer pixels in the mask
                    mask[buffer_x[valid_indices], buffer_y[valid_indices]] = True

                # Get coordinates of all marked pixels
                buffer_coords = np.column_stack(np.where(mask))
            else:
                buffer_coords = line_coords

            # Extract values from the raster data
            cell_values = self.raster_data[buffer_coords[:, 0], buffer_coords[:, 1]]

            # Otherwise just use minimum value in the area
            min_value = np.min(cell_values)

        # If no source provided or no valid min_value was calculated, fall back to
        # global minimum
        if min_value is None:
            min_value = self.raster_data.min()

        # Calculate the heuristic by scaling the Euclidean distance
        heuristic = euclidean_distance * min_value

        # Apply any heuristic weight scaling
        if 'heu_weight' in kwargs:
            heuristic *= kwargs['heu_weight']

        return nodes, heuristic

    def _vectorized_bresenham(self, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        """
        Optimized implementation of Bresenham's line algorithm that avoids generating
        duplicate coordinates.

        Parameters:
            x0: x-coordinate of the first point
            y0: y-coordinate of the first point
            x1: x-coordinate of the second point
            y1: y-coordinate of the second point

        Returns:
            Array of (x,y) coordinates of cells that the line passes through
        """
        # Determine if the line is steep (more vertical than horizontal)
        steep = abs(y1 - y0) > abs(x1 - x0)

        # If steep, swap x and y coordinates to ensure we step along the longer
        # dimension
        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1

        # Ensure we always iterate from left to right
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        # Calculate differences and step direction
        dx = x1 - x0
        dy = abs(y1 - y0)
        y_step = 1 if y0 < y1 else -1

        # Calculate points along the line using the Bresenham algorithm
        # Pre-allocate arrays for the coordinates (one point per x step guarantees no
        # duplicates)
        n_points = dx + 1
        x_coords = np.arange(x0, x1 + 1, dtype=np.int32)
        y_coords = np.zeros(n_points, dtype=np.int32)

        # Calculate y-coordinates based on error terms
        error = 0
        y = y0

        for i in range(n_points):
            y_coords[i] = y
            error += dy
            if error * 2 >= dx:
                y += y_step
                error -= dx

        # Combine into coordinate pairs and handle the steep case
        if steep:
            return np.column_stack((y_coords, x_coords))
        else:
            return np.column_stack((x_coords, y_coords))
