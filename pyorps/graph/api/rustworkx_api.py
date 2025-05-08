# Third party
import rustworkx as rx
from numpy import where, ndarray, ravel_multi_index, max, array, unravel_index, power, sqrt
from typing import Union, Optional, Any, List

# Project files
from pyorps.core.exceptions import NoPathFoundError, AlgorthmNotImplementedError
from .graph_library_api import GraphLibraryAPI


class RustworkxAPI(GraphLibraryAPI):

    def create_graph(self, from_nodes: ndarray[int], to_nodes: ndarray[int], cost: Optional[ndarray[int]] = None,
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
            max_node = max([max(from_nodes), max(to_nodes)])

        # Initialize graph
        self.graph = rx.PyGraph()
        self.graph.add_nodes_from(range(max_node + 1))

        # Add edges with costs
        if cost is not None:
            self.graph.add_edges_from(list(zip(from_nodes, to_nodes, cost)))
        else:
            # Add edges with default weight of 1.0
            self.graph.add_edges_from(list(zip(from_nodes, to_nodes, [1.0] * len(from_nodes))))

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
        """
        Returns the number of nodes in the graph.

        Returns:
            The number of nodes.
        """
        return self.graph.num_nodes()

    def get_number_of_edges(self) -> int:
        """
        Returns the number of edges in the graph.

        Returns:
            The number of edges.
        """
        return self.graph.num_edges()

    def get_nodes(self) -> Union[list[int], ndarray[int]]:
        """
        This method returns the nodes in the graph as a list or numpy array of node indices.

        :return: list[int]
            The list of node indices of the nodes in the graph
        """
        return self.graph.nodes()

    def add_edge(self, from_node: int, to_node: int, cost: float) -> None:
        """
        Adds a single edge to the graph.

        Args:
            from_node: The source node.
            to_node: The target node.
            cost: The edge weight.
        """
        self.graph.add_edge(from_node, to_node, cost)

    def add_edges(self, from_nodes: Union[ndarray[int], list[int]], to_nodes: Union[ndarray[int], list[int]],
                  cost: Union[ndarray[float], list[float]]) -> None:
        """
        Adds multiple edges to the graph.

        Args:
            from_nodes: Source node indices.
            to_nodes: Target node indices.
            cost: Edge weights.
        """
        self.graph.add_edges_from(list(zip(from_nodes, to_nodes, cost)))

    def get_graph_from_sparse_matrix(self,
                                     from_nodes: Union[ndarray[int], list[int]],
                                     to_nodes: Union[ndarray[int], list[int]],
                                     cost: Union[ndarray[float], list[float]]) -> rx.PyGraph:
        """
        Creates a graph from edge data.

        Args:
            from_nodes: Source node indices.
            to_nodes: Target node indices.
            cost: Edge weights.

        Returns:
            The rustworkx graph.
        """
        self.graph.add_edges_from(list(zip(from_nodes, to_nodes, cost)))
        return self.graph

    def remove_isolates(self) -> None:
        """
        Removes isolated nodes (nodes with no edges) from the graph.
        """
        indices_max_values = where(self.raster_data == 65535)
        nodes_max_values = ravel_multi_index(indices_max_values, self.raster_data.shape)
        self.graph.remove_nodes_from(nodes_max_values)

    @staticmethod
    def _ensure_path_endpoints(path, source, target):
        """
        Ensures the path starts with the source node and ends with the target node.

        Args:
            path: The path to check.
            source: The source node.
            target: The target node.

        Returns:
            The path with source and target as endpoints.
        """
        if len(path) > 0:
            if path[0] != source:
                path.insert(0, source)
            if path[-1] != target:
                path.append(target)
        return path

    def _compute_single_path(self, source, target, algorithm, **kwargs):
        """
        Computes shortest path between a single source and target.

        Args:
            source: Source node index.
            target: Target node index.
            algorithm: The algorithm to use (e.g., "dijkstra", "astar").
            **kwargs: Additional arguments for the algorithm.

        Returns:
            The shortest path as a list of node indices.

        Raises:
            NoPathFoundError: If no path exists between source and target.
            AlgorthmNotImplementedError: If the specified algorithm is not implemented.
        """
        try:
            if algorithm == "dijkstra":
                path = rx.dijkstra_shortest_paths(self.graph, source, target,
                                                  weight_fn=lambda edge_weight: edge_weight)
                path = list(path[target])
            elif algorithm == "astar":
                # Get heuristic function or use default manhattan distance as heuristic
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
                                              edge_cost_fn=lambda edge_weight: edge_weight,
                                              estimate_cost_fn=heuristic_function)
                path = list(path)
            elif algorithm == "bellman_ford":
                path = rx.bellman_ford_shortest_paths(self.graph, source, target=target,
                                                      weight_fn=lambda edge_weight: edge_weight)
                path = list(path[target])
            else:
                raise AlgorthmNotImplementedError(algorithm, self.__class__.__name__)

            return path
        except rx.NoPathFound:
            raise NoPathFoundError(source=source, target=target)

    def _compute_single_source_multiple_targets(self, source, targets, algorithm, **kwargs):
        """
        Computes shortest paths from a single source to multiple targets.

        Args:
            source: Source node index.
            targets: List of target node indices.
            algorithm: The algorithm to use (e.g., "dijkstra", "astar").
            **kwargs: Additional arguments for the algorithm.

        Returns:
            List of shortest paths, one for each target.
        """
        paths = []
        for target in targets:
            try:
                path = self._compute_single_path(source, target, algorithm, **kwargs)
                paths.append(path)
            except NoPathFoundError:
                paths.append([])
        return paths

    def _pairwise_shortest_path(self, sources, targets, algorithm, **kwargs):
        """
        Computes pairwise shortest paths between corresponding sources and targets.

        Args:
            sources: List of source node indices.
            targets: List of target node indices.
            algorithm: The algorithm to use (e.g., "dijkstra", "astar").
            **kwargs: Additional arguments for the algorithm.

        Returns:
            List of shortest paths, one for each source-target pair.
        """
        paths = []
        for source, target in zip(sources, targets):
            try:
                path = self._compute_single_path(source, target, algorithm, **kwargs)
                paths.append(path)
            except NoPathFoundError:
                paths.append([])
        return paths

    def _all_pairs_shortest_path(self, sources, targets, algorithm, **kwargs):
        """
        Computes shortest paths between all pairs of sources and targets.

        Args:
            sources: List of source node indices.
            targets: List of target node indices.
            algorithm: The algorithm to use (e.g., "dijkstra", "astar").
            **kwargs: Additional arguments for the algorithm.

        Returns:
            List of shortest paths, one for each source-target pair.
        """
        paths = []
        for source in sources:
            for target in targets:
                try:
                    path = self._compute_single_path(source, target, algorithm, **kwargs)
                    paths.append(path)
                except NoPathFoundError:
                    paths.append([])
        return paths

    def shortest_path(self, source_indices, target_indices, algorithm="dijkstra", **kwargs):
        """
        This method applies the specified shortest path algorithm on the created graph object and finds the shortest
        path between source and target(s) as a list of node indices.

        Parameters:
        -----------
        source_indices : int or list[int]
            Index or indices of source node(s)
        target_indices : int or list[int]
            Index or indices of target node(s)
        algorithm : str, default="dijkstra"
            Algorithm to use for shortest path computation.
            Options: "dijkstra", "bidirectional_dijkstra", "astar"
        **kwargs:
            pairwise : bool
                If True, compute pairwise shortest paths between source_indices and target_indices.
                Only allowed if len(source_indices) == len(target_indices)
            heuristic : callable, optional
                A function that takes a node index and returns an estimate of the distance
                to the target. Only used when algorithm="astar".

        Returns:
        --------
        list[int] or list[list[int]]:
            List of node indices representing the shortest path(s)
        """
        # Convert single indices to lists for uniform handling
        if not isinstance(source_indices, (list, tuple, ndarray)):
            source_indices = [source_indices]
        if not isinstance(target_indices, (list, tuple, ndarray)):
            target_indices = [target_indices]

        # Check for pairwise computation
        pairwise = kwargs.get('pairwise', False)
        if pairwise:
            if len(source_indices) != len(target_indices):
                raise ValueError("Source and target lists must have the same length for pairwise computation")
            return self._pairwise_shortest_path(source_indices, target_indices, algorithm)

        # Single source, single target
        if len(source_indices) == 1 and len(target_indices) == 1:
            source = source_indices[0]
            target = target_indices[0]
            return self._compute_single_path(source, target, algorithm, **kwargs)
        # Multiple sources, single target (special case handling)
        elif len(source_indices) > 1 and len(target_indices) == 1:
            target = target_indices[0]
            paths = []
            for source in source_indices:
                try:
                    path = self._compute_single_path(source, target, algorithm, **kwargs)
                    paths.append(path)
                except NoPathFoundError:
                    paths.append([])
            return paths
        # Single source, multiple targets
        elif len(source_indices) == 1:
            source = source_indices[0]
            return self._compute_single_source_multiple_targets(source, target_indices, algorithm, **kwargs)

        # Multiple sources, multiple targets (all pairs)
        else:
            return self._all_pairs_shortest_path(source_indices, target_indices, algorithm, **kwargs)


