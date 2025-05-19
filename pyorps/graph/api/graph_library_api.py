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
from typing import Optional, Any
from abc import abstractmethod
import numpy as np
from time import time

from .graph_api import GraphAPI

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
    def shortest_path(self, source_indices, target_indices, algorithm="dijkstra",
                      **kwargs) -> list[int]:
        """
        This method applies the specified shortest path algorithm on the created graph
        object and finds the shortest path between source and target(s) as a list of
        node indices.

        :return: list[int]
            The list of node indices of the shortest path between source and target
        """

    @abstractmethod
    def get_nodes(self) -> list[int] | np.ndarray[int]:
        """
        This method returns the nodes in the graph as a list or numpy array of node
        indices.

        :return:  list[int] | ndarray[int]
            The list of node indices of the nodes in the graph
        """

    def get_a_star_heuristic(self, target: int,
                             **kwargs) -> tuple[np.ndarray[int], np.ndarray[float]]:
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

