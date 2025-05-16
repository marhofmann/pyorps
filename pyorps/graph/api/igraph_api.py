from typing import Union

# Third party
import igraph as ig
from numpy import float64, ndarray, max

# Project files
from pyorps.core.exceptions import NoPathFoundError, AlgorthmNotImplementedError
from .graph_library_api import *


class IGraphAPI(GraphLibraryAPI):

    def create_graph(self, from_nodes: ndarray[int], to_nodes: ndarray[int],
                     cost: Optional[ndarray[int]] = None,
                     **kwargs) -> Any:
        """
        Creates a graph object with the igraph library.

        :param from_nodes: The starting node indices from the edge data.
        :param to_nodes: The ending node indices from the edge data.
        :param cost: The weight of the edge data.
        :param kwargs: Additional parameters for the underlying graph library.
        :return: The graph object.
        """
        # Determine number of nodes
        if (n := kwargs.get('n', None)) is not None:
            n_vertices = n
        else:
            n_vertices = max([max(from_nodes), max(to_nodes)]) + 1

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

        :return: The number of Nodes
        """
        return self.graph.vcount()

    def get_number_of_edges(self):
        """
        Returns the number of edges in the graph.

        :return: The number of Edges
        """
        return self.graph.ecount()

    def remove_isolates(self):
        """
        Removes nodes without any edge (degree == 0).

        :return: None
        """
        # Get vertices with degree 0
        isolated_vertices = [v.index for v in self.graph.vs if v.degree() == 0]
        # Delete them from highest index to lowest to avoid reindexing issues
        isolated_vertices.sort(reverse=True)
        for v in isolated_vertices:
            self.graph.delete_vertices(v)

    def get_nodes(self) -> Union[list[int], ndarray[int]]:
        """
        This method returns the nodes in the graph as a list or numpy array of node
        indices.

        :return: list[int]
            The list of node indices of the nodes in the graph
        """
        return [v.index for v in self.graph.vs()]

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

    def _compute_all_pairs_shortest_paths(self, sources, targets, algorithm, **kwargs):
        """
        Computes paths individually for each source-target pair using the specified
        algorithm.
        Returns empty paths for unreachable targets.
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
            Options: "dijkstra", "bellman_ford", "astar"
        **kwargs:
            pairwise : bool
                If True, compute pairwise shortest paths between source_indices and
                target_indices.
                Only allowed if len(source_indices) == len(target_indices)
            heuristic : callable, optional
                A function that takes two node indices (u, target) and returns an
                estimate of the distance between them. Only used when algorithm="astar".

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
                msg = ("Source and target lists must have the same length for pairwise "
                       "computation")
                raise ValueError()
            return self._pairwise_shortest_path(source_indices, target_indices, algorithm)

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

    def _compute_single_path(self, source, target, algorithm, **kwargs):
        """
        Computes shortest path between a single source and target using the specified
        algorithm.
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

    def _compute_single_source_multiple_targets(self, source, targets, algorithm,
                                                **kwargs):
        """
        Computes shortest paths from a single source to multiple targets.
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

    def _pairwise_shortest_path(self, sources, targets, algorithm, **kwargs):
        """
        Computes pairwise shortest paths between corresponding sources and targets.
        """
        # For igraph, we need to compute each path individually
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
