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
from pyorps.core.exceptions import NoPathFoundError
from pyorps.utils.traversal import construct_edges


class GraphLibraryAPI(GraphAPI):
    """
    Base class for all graph library-based APIs.

    This class extends GraphAPI with common functionality needed by standard graph
    libraries that require edge data to be explicitly provided and a graph to be
    constructed.
    """

    def __init__(self,
                 raster_data: np.ndarray[int],
                 steps: np.ndarray[int],
                 from_nodes: Optional[np.ndarray] = None,
                 to_nodes: Optional[np.ndarray] = None,
                 cost: Optional[np.ndarray] = None,
                 ignore_max: Optional[bool] = True,
                 **kwargs):
        """
        Initialize the graph library API.

        Args:
            raster_data: 2D numpy array representing the raster
            steps: Array defining the neighborhood connections
            from_nodes: Source node indices for each edge
            to_nodes: Target node indices for each edge
            cost: Edge weights
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
    def _ensure_path_endpoints(path, source, target):
        """
        Ensures the path starts with the source node and ends with the target node.
        """
        if len(path) > 0:
            if path[0] != source:
                path.insert(0, source)
            if path[-1] != target:
                path.append(target)
        return path

    @abstractmethod
    def create_graph(self, from_nodes: np.ndarray[int], to_nodes: np.ndarray[int],
                     cost: Optional[np.ndarray[int]] = None, **kwargs) -> Any:
        """
        Creates a graph object with the graph library specified in the selected
        interface.

        Args:
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

        :return: The number of Nodes
        """

    @abstractmethod
    def get_number_of_edges(self) -> int:
        """
        Returns the number of edges in the graph.

        :return: The number of Edges
        """

    @abstractmethod
    def remove_isolates(self) -> None:
        """
        If the graph object was initialized with the maximum number of nodes, this
        function helps to reduce the occupied memory by removing nodes without any
        edge (degree == 0).

        :return: None
        """

    @abstractmethod
    def get_nodes(self) -> Union[List[int], np.ndarray]:
        """
        This method returns the nodes in the graph as a list or numpy array of node
        indices.

        :return:  list[int] | ndarray[int]
            The list of node indices of the nodes in the graph
        """

    def shortest_path(self, source_indices, target_indices, algorithm="dijkstra",
                      **kwargs):
        """
        This method applies the specified shortest path algorithm on the created graph
        object and finds the shortest path between source and target(s) as a list of
        node indices.

        Parameters:
        -----------
        source_indices : int or list[int]
            Index or indices of source node(s)
        target_indices : int or list[int]
            Index or indices of target node(s)
        algorithm : str, default="dijkstra"
            Algorithm to use for shortest path computation.
            Options depend on the specific library implementation.
        **kwargs:
            pairwise : bool
                If True, compute pairwise shortest paths between source_indices and
                target_indices.
                Only allowed if len(source_indices) == len(target_indices)
            Additional parameters for specific algorithms

        Returns:
        --------
        list[int] or list[list[int]]:
            List of node indices representing the shortest path(s)
        """
        # Convert single indices to lists for uniform handling
        if not isinstance(source_indices, (list, tuple, np.ndarray)):
            source_indices = [source_indices]
        if not isinstance(target_indices, (list, tuple, np.ndarray)):
            target_indices = [target_indices]

        # Check for pairwise computation
        pairwise = kwargs.get('pairwise', False)
        if pairwise:
            if len(source_indices) != len(target_indices):
                msg = ("Source and target lists must have the same length for "
                       "pairwise computation")
                raise ValueError(msg)
            return self._pairwise_shortest_path(source_indices, target_indices,
                                                algorithm, **kwargs)

        # Single source, single target
        if len(source_indices) == 1 and len(target_indices) == 1:
            source = source_indices[0]
            target = target_indices[0]
            return self._compute_single_path(source, target, algorithm, **kwargs)

        # Single source, multiple targets
        elif len(source_indices) == 1:
            source = source_indices[0]
            return self._compute_single_source_multiple_targets(source, target_indices,
                                                                algorithm, **kwargs)

        # Multiple sources, multiple targets (all pairs)
        else:
            return self._all_pairs_shortest_path(source_indices, target_indices,
                                                 algorithm, **kwargs)

    @abstractmethod
    def _compute_single_path(self, source, target, algorithm, **kwargs):
        """
        Computes shortest path between a single source and target.
        """

    @abstractmethod
    def _compute_single_source_multiple_targets(self, source, targets, algorithm,
                                                **kwargs):
        """
        Computes shortest paths from a single source to multiple targets.
        """

    def _pairwise_shortest_path(self, sources, targets, algorithm, **kwargs):
        """
        Default implementation for pairwise shortest path computation.
        Subclasses can override this for library-specific optimizations.
        """
        paths = []
        for source, target in zip(sources, targets):
            try:
                path = self._compute_single_path(source, target, algorithm, **kwargs)
                paths.append(path)
            except NoPathFoundError:
                paths.append([])
        return paths

    def _compute_all_pairs_shortest_paths(self, sources, targets, algorithm, **kwargs):
        """
        Computes paths individually for each source-target pair using the specified
        algorithm. Returns empty paths for unreachable targets.
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
    def _all_pairs_shortest_path(self, sources, targets, algorithm, **kwargs):
        """
        Computes shortest paths between all pairs of sources and targets.
        """

    def get_a_star_heuristic(self, target: int,
                             **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the A* heuristic based on the Euclidean distance from the target node.

        :param target: int
            The index of the target node in the raster data.

        :return: tuple[np.ndarray[int], np.ndarray[float]]
            A tuple containing:
            - An array of node indices (nodes) in the graph.
            - An array of heuristic values corresponding to each node, calculated as the
              Euclidean distance to the target node multiplied by the minimum value in
              the raster data.
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

        # Get the minimum value from the raster data for scaling the heuristic
        min_value = self.raster_data.min()

        # Calculate the heuristic by scaling the Euclidean distance
        heuristic = euclidean_distance * min_value

        if 'heu_weight' in kwargs:
            heuristic *= kwargs['heu_weight']
        return nodes, heuristic
