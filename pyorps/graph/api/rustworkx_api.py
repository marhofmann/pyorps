# Third party
import rustworkx as rx
from numpy import where, ndarray, ravel_multi_index, max as np_max
from typing import Optional

# Project files
from pyorps.core.exceptions import NoPathFoundError, AlgorthmNotImplementedError
from pyorps.core.types import Node, NodeList, NodePathList
from pyorps.graph.api.graph_library_api import GraphLibraryAPI


class RustworkxAPI(GraphLibraryAPI):

    def create_graph(
            self,
            from_nodes: NodeList,
            to_nodes: NodeList,
            cost: Optional[ndarray[int]] = None,
            **kwargs
    ) -> rx.PyGraph:
        """
        Creates a graph object using rustworkx.

        Parameters:
            from_nodes: The starting node indices from the edge data
            to_nodes: The ending node indices from the edge data
            cost: The weight of the edge data
            kwargs: Additional parameters for the underlying graph library

        Returns:
            The graph object
        """
        # Get total number of nodes needed for the graph
        if n := kwargs.get('n', None):
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

    def get_number_of_nodes(self) -> int:
        """
        Returns the number of nodes in the graph.

        Returns:
            The number of nodes
        """
        return self.graph.num_nodes()

    def get_number_of_edges(self) -> int:
        """
        Returns the number of edges in the graph.

        Returns:
            The number of edges
        """
        return self.graph.num_edges()

    def get_nodes(self) -> NodeList:
        """
        This method returns the nodes in the graph as a list or numpy array of node
        indices.

        Returns:
            List or array of node indices of the nodes in the graph
        """
        return self.graph.nodes()

    def remove_isolates(self) -> None:
        """
        If the graph object was initialized with the maximum number of nodes, this
        function helps to reduce the occupied memory by removing nodes without any
        edge (degree == 0).

        Returns:
            None
        """
        indices_max_values = where(self.raster_data == 65535)
        nodes_max_values = ravel_multi_index(indices_max_values, self.raster_data.shape)
        self.graph.remove_nodes_from(nodes_max_values)

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
                    nodes, heuristic = self.get_a_star_heuristic(target,
                                                                 source,
                                                                 **kwargs)
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
        paths = []
        for target in targets:
            try:
                path = self._compute_single_path(source, target, algorithm, **kwargs)
                paths.append(path)
            except NoPathFoundError:
                paths.append([])
        return paths

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
