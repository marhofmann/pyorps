from typing import Optional

# Third party
import igraph as ig
from numpy import float64, ndarray, max as np_max

# Project files
from pyorps.core.exceptions import NoPathFoundError, AlgorthmNotImplementedError
from pyorps.core.types import Node, NodeList, NodePathList
from pyorps.graph.api.graph_library_api import GraphLibraryAPI


class IGraphAPI(GraphLibraryAPI):

    def create_graph(
            self,
            from_nodes: NodeList,
            to_nodes: NodeList,
            cost: Optional[ndarray[int]] = None,
            **kwargs
    ) -> ig.Graph:
        """
        Creates a graph object with the igraph library.

        Parameters:
            from_nodes: The starting node indices from the edge data
            to_nodes: The ending node indices from the edge data
            cost: The weight of the edge data
            kwargs: Additional parameters for the underlying graph library

        Returns:
            The graph object
        """
        # Determine number of nodes
        if (n := kwargs.get('n', None)) is not None:
            n_vertices = n
        else:
            n_vertices = np_max([np_max(from_nodes), np_max(to_nodes)]) + 1

        # Create graph
        self.graph = ig.Graph(n=int(n_vertices), directed=False)

        # Create edges
        edges = list(zip(from_nodes, to_nodes))

        # Add edges with weights if provided, otherwise without weights
        if cost is not None:
            weights = cost.astype(float64, copy=False)
            self.graph.add_edges(edges)
            self.graph.es['weight'] = weights
        else:
            self.graph.add_edges(edges)

        # Remove isolated nodes if requested
        if kwargs.get('remove_isolated_nodes', False):
            self.remove_isolates()

        return self.graph

    def get_number_of_nodes(self):
        """
        Returns the number of nodes in the graph.

        Returns:
            The number of nodes
        """
        return self.graph.vcount()

    def get_number_of_edges(self):
        """
        Returns the number of edges in the graph.

        Returns:
            The number of edges
        """
        return self.graph.ecount()

    def remove_isolates(self):
        """
        If the graph object was initialized with the maximum number of nodes, this
        function helps to reduce the occupied memory by removing nodes without any
        edge (degree == 0).

        Returns:
            None
        """
        # Get vertices with degree 0
        isolated_vertices = [v.index for v in self.graph.vs if v.degree() == 0]
        # Delete them from highest index to lowest to avoid reindexing issues
        isolated_vertices.sort(reverse=True)
        for v in isolated_vertices:
            self.graph.delete_vertices(v)

    def get_nodes(self) -> NodeList:
        """
        This method returns the nodes in the graph as a list or numpy array of node
        indices.

        Returns:
            List or array of node indices of the nodes in the graph
        """
        return [v.index for v in self.graph.vs()]

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
        if 'weight' in self.graph.es.attributes():
            weights = self.graph.es.get_attribute_values('weight')
        else:
            weights = None

        if algorithm == "dijkstra":
            path = self.graph.get_shortest_paths(source, target, weights=weights,
                                                 output="vpath")[0]

        elif algorithm == "bellman_ford":
            path = self.graph.get_shortest_paths(source, target, weights=weights,
                                                 output="vpath",
                                                 algorithm="bellman_ford")[0]

        elif algorithm == "astar":
            heuristic_function = kwargs.get('heu', None)

            if heuristic_function is None:
                _, heuristic = self.get_a_star_heuristic(target, **kwargs)

                def heuristic_function(_graph, node, _target):
                    return heuristic[node]

            path = self.graph.get_shortest_path_astar(source, target,
                                                      heuristics=heuristic_function,
                                                      weights=weights,
                                                      output="vpath", mode="all")

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
        if 'weight' in self.graph.es.attributes():
            weights = self.graph.es.get_attribute_values('weight')
        else:
            weights = None

        if algorithm in ["dijkstra", "bellman_ford"]:
            # igraph can compute paths to multiple targets at once
            algo = "dijkstra" if algorithm == "dijkstra" else "bellman_ford"
            all_paths = self.graph.get_shortest_paths(source, targets,
                                                      weights=weights, output="vpath",
                                                      algorithm=algo)

            # Process each path
            paths = []
            for i, path in enumerate(all_paths):
                if len(path) == 0:
                    paths.append([])  # No path found
                else:
                    paths.append(self._ensure_path_endpoints(path, source, targets[i]))
            return paths

        elif algorithm == "astar":
            # For A*, handle each target separately
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
        if algorithm in ["dijkstra", "bellman_ford"]:
            paths = []
            if 'weight' in self.graph.es.attributes():
                weights = self.graph.es.get_attribute_values('weight')
            else:
                weights = None
            algo = "dijkstra" if algorithm == "dijkstra" else "bellman_ford"

            for source in sources:
                # For each source, compute paths to all targets at once
                target_paths = self.graph.get_shortest_paths(source, targets,
                                                             weights=weights,
                                                             output="vpath",
                                                             algorithm=algo)

                # Process each path
                for i, path in enumerate(target_paths):
                    if len(path) == 0:
                        paths.append([])  # No path found
                    else:
                        paths.append(self._ensure_path_endpoints(path, source,
                                                                 targets[i]))

            return paths

        else:
            # For other algorithms, compute each path individually
            return self._compute_all_pairs_shortest_paths(sources, targets, algorithm,
                                                          **kwargs)
