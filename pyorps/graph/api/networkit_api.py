from typing import Optional

# Third party
import numpy as np
from numpy import ndarray
from networkit import Graph
from networkit.distance import Dijkstra, BidirectionalDijkstra, AStar

# Project files
from pyorps.core.exceptions import NoPathFoundError, AlgorthmNotImplementedError
from pyorps.core.types import Node, NodeList, NodePathList
from pyorps.graph.api.graph_library_api import GraphLibraryAPI


class NetworkitAPI(GraphLibraryAPI):

    def create_graph(
            self,
            from_nodes: NodeList,
            to_nodes: NodeList,
            cost: Optional[ndarray[int]] = None,
            **kwargs
    ) -> Graph:
        """
        Creates a graph object with the networkit library.

        Parameters:
            from_nodes: The starting node indices from the edge data
            to_nodes: The ending node indices from the edge data
            cost: The weight of the edge data
            kwargs: Additional parameters for the underlying graph library

        Returns:
            The graph object
        """
        if (n := kwargs.get('n', None)) is not None:
            self.graph = Graph(n, weighted=True, directed=False)
        else:
            n = max([max(from_nodes), max(to_nodes)]) + 1
            self.graph = Graph(n=n, weighted=True, directed=False)
        if cost is not None:
            self.graph.addEdges((cost.astype(np.float64, copy=False),
                                 (from_nodes, to_nodes)), addMissing=False)
        else:
            self.graph.addEdges((from_nodes, to_nodes), addMissing=False)
        if kwargs.get('remove_isolated_nodes', False):
            self.remove_isolates()
        return self.graph

    def get_number_of_nodes(self):
        """
        Returns the number of nodes in the graph.

        Returns:
            The number of nodes
        """
        return self.graph.numberOfNodes()

    def get_number_of_edges(self):
        """
        Returns the number of edges in the graph.

        Returns:
            The number of edges
        """
        return self.graph.numberOfEdges()

    def remove_isolates(self):
        """
        If the graph object was initialized with the maximum number of nodes, this
        function helps to reduce the occupied memory by removing nodes without any
        edge (degree == 0).

        Returns:
            None
        """
        for n in self.graph.iterNodes():
            if self.graph.isIsolated(n):
                self.graph.removeNode(n)

    def get_nodes(self) -> NodeList:
        """
        This method returns the nodes in the graph as a list or numpy array of node
        indices.

        Returns:
            List or array of node indices of the nodes in the graph
        """
        return [i for i in self.graph.iterNodes()]

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
        if algorithm == "dijkstra":
            dijkstra = Dijkstra(self.graph, source, storePaths=True, target=target)
            dijkstra.run()
            path = dijkstra.getPath(target)

        elif algorithm == "bidirectional_dijkstra":
            bidir_dijkstra = BidirectionalDijkstra(self.graph, source, target)
            bidir_dijkstra.run()
            path = bidir_dijkstra.getPath()

        elif algorithm == "astar":
            # Use provided heuristic function or default to zero heuristic
            heuristic_function = kwargs.get('heu', None)
            if heuristic_function is None:
                _, heuristic = self.get_a_star_heuristic(target, source, **kwargs)
                heuristic_function = list(heuristic)

            astar = AStar(self.graph, heuristic_function, source, target)
            astar.run()
            path = astar.getPath()

        else:
            raise AlgorthmNotImplementedError(algorithm, self.__class__.__name__)

        if len(path) == 0:
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
        if algorithm == "dijkstra" or algorithm == "bidirectional_dijkstra":
            return self._compute_multi_target_dijkstra(source, targets)

        elif algorithm == "astar":
            # If using A* with multiple targets, run individual A* or fall back to
            # Dijkstra depending on whether a heuristic is provided
            if 'heuristic' in kwargs:
                paths = []
                for target in targets:
                    try:
                        path = self._compute_single_path(source, target, algorithm,
                                                         **kwargs)
                        paths.append(path)
                    except NoPathFoundError:
                        paths.append([])
                return paths
            else:
                return self._compute_multi_target_dijkstra(source, targets)
        else:
            raise AlgorthmNotImplementedError(algorithm, self.__class__.__name__)

    def _compute_multi_target_dijkstra(self, source, targets):
        # Use MultiTargetDijkstra for efficient computation
        dijkstra = Dijkstra(self.graph, source, storePaths=True)
        dijkstra.run()
        paths = []
        for target in targets:
            path = dijkstra.getPath(target)

            # For multi-target we add empty paths for unreachable targets
            if len(path) == 0:
                paths.append([])
                continue

            path = self._ensure_path_endpoints(path, source, target)
            paths.append(path)
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
        if algorithm == "dijkstra":
            # Use APSP for efficient all-pairs computation
            paths = []

            # Run Dijkstra once for each source
            for source in sources:
                dijkstra = Dijkstra(self.graph, source, storePaths=True)
                dijkstra.run()

                for target in targets:
                    path = dijkstra.getPath(target)

                    # For all-pairs we add empty paths for unreachable targets
                    if len(path) == 0:
                        paths.append([])
                        continue

                    path = self._ensure_path_endpoints(path, source, target)
                    paths.append(path)

            return paths
        else:
            # For other algorithms, use helper function to compute paths individually
            return self._compute_all_pairs_shortest_paths(sources, targets, algorithm,
                                                          **kwargs)
