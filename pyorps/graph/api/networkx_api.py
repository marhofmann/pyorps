from typing import Optional

# Third party
import networkx as nx
from numpy import ndarray

# Project files
from pyorps.core.types import Node, NodeList, NodePathList
from pyorps.core.exceptions import NoPathFoundError, AlgorthmNotImplementedError
from pyorps.graph.api.graph_library_api import GraphLibraryAPI


class NetworkxAPI(GraphLibraryAPI):

    def create_graph(
            self,
            from_nodes: NodeList,
            to_nodes: NodeList,
            cost: Optional[ndarray[int]] = None,
            **kwargs
    ) -> nx.Graph:
        """
        Creates a graph object with the networkx library.

        Parameters:
            from_nodes: The starting node indices from the edge data
            to_nodes: The ending node indices from the edge data
            cost: The weight of the edge data
            kwargs: Additional parameters for the underlying graph library

        Returns:
            The graph object
        """
        directed = kwargs.get('directed', False)
        self.graph = nx.DiGraph() if directed else nx.Graph()

        if cost is not None:
            self.graph.add_weighted_edges_from(zip(from_nodes, to_nodes, cost))
        else:
            self.graph.add_edges_from(zip(from_nodes, to_nodes))

        if kwargs.get('remove_isolated_nodes', False):
            self.remove_isolates()

        return self.graph

    def get_number_of_nodes(self) -> int:
        """
        Returns the number of nodes in the graph.

        Returns:
            The number of nodes
        """
        return self.graph.number_of_nodes()

    def get_number_of_edges(self):
        """
        Returns the number of edges in the graph.

        Returns:
            The number of edges
        """
        return self.graph.number_of_edges()

    def remove_isolates(self):
        """
        If the graph object was initialized with the maximum number of nodes, this
        function helps to reduce the occupied memory by removing nodes without any
        edge (degree == 0).

        Returns:
            None
        """
        self.graph.remove_nodes_from(list(nx.isolates(self.graph)))

    def get_nodes(self) -> NodeList:
        """
        This method returns the nodes in the graph as a list or numpy array of node
        indices.

        Returns:
            List or array of node indices of the nodes in the graph
        """
        return list(self.graph)

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
        try:
            if algorithm == "dijkstra":
                path = nx.dijkstra_path(self.graph, source, target, weight='weight')

            elif algorithm == "bidirectional_dijkstra":
                _, path = nx.bidirectional_dijkstra(self.graph, source, target,
                                                    weight='weight')

            elif algorithm == "astar":
                heuristic_function = kwargs.get('heu', None)

                if heuristic_function is None:
                    nodes, heuristic = self.get_a_star_heuristic(target, **kwargs)
                    heuristic_dict = dict(zip(nodes, heuristic))

                    def heuristic_function(node, _target):
                        return heuristic_dict[node]

                path = nx.astar_path(self.graph, source, target, heuristic_function,
                                     weight='weight')

            else:
                raise AlgorthmNotImplementedError(algorithm, self.__class__.__name__)

        except nx.NetworkXNoPath:
            raise NoPathFoundError(source=source, target=target)

        path = self._ensure_path_endpoints(path, source, target)
        return path

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

        if algorithm == "dijkstra":
            # Use single-source Dijkstra for efficiency
            _, paths_dict = nx.single_source_dijkstra(self.graph, source,
                                                      weight='weight')

            for target in targets:
                if target in paths_dict:
                    path = paths_dict[target]
                    path = self._ensure_path_endpoints(path, source, target)
                    paths.append(path)
                else:
                    paths.append([])

            return paths

        elif algorithm in ["bidirectional_dijkstra", "astar"]:
            # Run individual algorithm for each target
            for target in targets:
                try:
                    path = self._compute_single_path(source, target, algorithm,
                                                     **kwargs)
                    paths.append(path)
                except NoPathFoundError:
                    paths.append([])
            return paths

        else:
            raise AlgorthmNotImplementedError(algorithm, self.__class__.__name__)

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
        if algorithm == "dijkstra":
            paths = []

            # For each source, compute paths to all targets
            for source in sources:
                for target in targets:
                    try:
                        _, path = nx.single_source_dijkstra(self.graph, source, target,
                                                            weight='weight')
                        path = self._ensure_path_endpoints(path, source, target)
                        paths.append(path)
                    except nx.NetworkXNoPath:
                        paths.append([])

            return paths

        else:
            # For other algorithms, compute each path individually
            return self._compute_all_pairs_shortest_paths(sources, targets, algorithm,
                                                          **kwargs)
