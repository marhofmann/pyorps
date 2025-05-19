# Third party
import rustworkx as rx
from numpy import where, ndarray, ravel_multi_index, max as np_max
from typing import Union, Optional, List

# Project files
from pyorps.core.exceptions import NoPathFoundError, AlgorthmNotImplementedError
from .graph_library_api import GraphLibraryAPI


class RustworkxAPI(GraphLibraryAPI):

    def create_graph(self, from_nodes: ndarray[int], to_nodes: ndarray[int],
                     cost: Optional[ndarray[int]] = None,
                     **kwargs) -> rx.PyGraph:
        """
        Creates a graph object using rustworkx.

        Args:
            from_nodes: The starting node indices from the edge data.
            to_nodes: The ending node indices from the edge data.
            cost: The weight of the edge data.
            kwargs: Additional parameters for the underlying graph library.

        Returns:
            The rustworkx graph object.
        """
        # Get total number of nodes needed for the graph
        if (n := kwargs.get('n', None)) is not None:
            max_node = n - 1
        else:
            max_node = np_max([np_max(from_nodes), np_max(to_nodes)])

        # Initialize graph
        self.graph = rx.PyGraph()
        self.graph.add_nodes_from(range(max_node + 1))

        # Add edges with costs
        if cost is not None:
            self.graph.add_edges_from(list(zip(from_nodes, to_nodes, cost)))
        else:
            # Add edges with default weight of 1.0
            # Rustworkx only takes a list of tuples instead of edges!
            edge_list = list(zip(from_nodes, to_nodes, [1.0] * len(from_nodes)))
            self.graph.add_edges_from(edge_list)

        if kwargs.get('remove_isolated_nodes', False):
            self.remove_isolates()

        return self.graph

    def init_chunk_size(self, chunk_size: Optional[int] = None) -> int:
        """
        Initializes the chunk size for batch operations.

        Args:
            chunk_size: The size of chunks to use. Defaults to 1 if None.

        Returns:
            The initialized chunk size.
        """
        if chunk_size is None:
            self.chunk_size = 1
        else:
            self.chunk_size = chunk_size
        return self.chunk_size

    def get_graph(self) -> rx.PyGraph:
        """
        Creates a graph with nodes based on the raster dimensions.

        Returns:
            A rustworkx graph with nodes but no edges.
        """
        nr_of_nodes = self.raster_data.shape[0] * self.raster_data.shape[1]
        nodes = range(nr_of_nodes)
        rxgraph = rx.PyGraph()
        rxgraph.add_nodes_from(nodes)
        return rxgraph

    def get_number_of_nodes(self) -> int:
        """Returns the number of nodes in the graph."""
        return self.graph.num_nodes()

    def get_number_of_edges(self) -> int:
        """Returns the number of edges in the graph."""
        return self.graph.num_edges()

    def get_nodes(self) -> Union[list[int], ndarray[int]]:
        """Returns the list of node indices in the graph."""
        return self.graph.nodes()

    def add_edge(self, from_node: int, to_node: int, cost: float) -> None:
        """Adds a single edge to the graph."""
        self.graph.add_edge(from_node, to_node, cost)

    def add_edges(self, from_nodes: Union[ndarray[int], list[int]],
                  to_nodes: Union[ndarray[int], list[int]],
                  cost: Union[ndarray[float], list[float]]) -> None:
        """Adds multiple edges to the graph."""
        self.graph.add_edges_from(list(zip(from_nodes, to_nodes, cost)))

    def get_graph_from_sparse_matrix(
            self,
            from_nodes: Union[ndarray[int], list[int]],
            to_nodes: Union[ndarray[int], list[int]],
            cost: Union[ndarray[float], list[float]]) -> rx.PyGraph:
        """Creates a graph from edge data."""
        self.graph.add_edges_from(list(zip(from_nodes, to_nodes, cost)))
        return self.graph

    def remove_isolates(self) -> None:
        """Removes isolated nodes (nodes with no edges) from the graph."""
        indices_max_values = where(self.raster_data == 65535)
        nodes_max_values = ravel_multi_index(indices_max_values, self.raster_data.shape)
        self.graph.remove_nodes_from(nodes_max_values)

    def _compute_single_path(self, source: int, target: int, algorithm: str,
                             **kwargs) -> List[int]:
        """
        Computes shortest path between a single source and target.

        Args:
            source: Source node index.
            target: Target node index.
            algorithm: The algorithm to use (e.g., "dijkstra", "astar").
            **kwargs: Additional arguments for the algorithm.

        Returns:
            The shortest path as a list of node indices.
        """
        def weight_fn(edge_weight):
            return edge_weight

        try:
            if algorithm == "dijkstra":
                path = rx.dijkstra_shortest_paths(self.graph, source, target,
                                                  weight_fn=weight_fn)
                path = list(path[target])
            elif algorithm == "astar":
                # Get heuristic function or use default as heuristic
                heuristic_function = kwargs.get('heu', None)

                if heuristic_function is None:
                    nodes, heuristic = self.get_a_star_heuristic(target, **kwargs)
                    heuristic_dict = dict(zip(nodes, heuristic))

                    def heuristic_function(node):
                        return heuristic_dict[node]

                def goal_reached(node):
                    return node == target

                path = rx.astar_shortest_path(self.graph, source,
                                              goal_fn=goal_reached,
                                              edge_cost_fn=weight_fn,
                                              estimate_cost_fn=heuristic_function)
                path = list(path)
            elif algorithm == "bellman_ford":
                path = rx.bellman_ford_shortest_paths(self.graph, source, target=target,
                                                      weight_fn=weight_fn)
                path = list(path[target])
            else:
                raise AlgorthmNotImplementedError(algorithm, self.__class__.__name__)

            return self._ensure_path_endpoints(path, source, target)
        except rx.NoPathFound:
            raise NoPathFoundError(source=source, target=target)

    def _compute_single_source_multiple_targets(self, source: int,
                                                targets: List[int],
                                                algorithm: str,
                                                **kwargs) -> List[List[int]]:
        """
        Computes shortest paths from a single source to multiple targets.
        """
        paths = []
        for target in targets:
            try:
                path = self._compute_single_path(source, target, algorithm, **kwargs)
                paths.append(path)
            except NoPathFoundError:
                paths.append([])
        return paths

    def _pairwise_shortest_path(self, sources: List[int], targets: List[int],
                                algorithm: str, **kwargs) -> List[List[int]]:
        """
        Computes pairwise shortest paths between corresponding sources and targets.
        """
        paths = []
        for source, target in zip(sources, targets):
            try:
                path = self._compute_single_path(source, target, algorithm, **kwargs)
                paths.append(path)
            except NoPathFoundError:
                paths.append([])
        return paths

    def _all_pairs_shortest_path(self, sources: List[int], targets: List[int],
                                 algorithm: str, **kwargs) -> List[List[int]]:
        """
        Computes shortest paths between all pairs of sources and targets.
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
