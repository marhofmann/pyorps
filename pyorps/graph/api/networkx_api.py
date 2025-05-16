from typing import Union

# Third party
import networkx as nx

# Project files
from pyorps.core.exceptions import NoPathFoundError, AlgorthmNotImplementedError
from .graph_library_api import *


class NetworkxAPI(GraphLibraryAPI):

    def create_graph(self, from_nodes: np.ndarray[int], to_nodes: np.ndarray[int],
                     cost: Optional[np.ndarray[int]] = None,
                     **kwargs) -> Any:
        """
        Creates a graph object with the graph library specified in the selected
        interface.

        :param from_nodes: The starting node indices from the edge data.
        :param to_nodes: The ending node indices from the edge data.
        :param cost: The weight of the edge data.
        :param kwargs: Additional parameters for the underlying graph library.
        :return: The graph object.
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

    def get_number_of_nodes(self):
        return self.graph.number_of_nodes()

    def get_number_of_edges(self):
        return self.graph.number_of_edges()

    def remove_isolates(self):
        self.graph.remove_nodes_from(list(nx.isolates(self.graph)))

    def get_nodes(self) -> Union[list[int], np.ndarray[int]]:
        """
        This method returns the nodes in the graph as a list or numpy array of node
        indices.

        :return: list[int]
            The list of node indices of the nodes in the graph
        """
        return list(self.graph)

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
        object and finds the shortest
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
                If True, compute pairwise shortest paths between source_indices and
                target_indices.
                Only allowed if len(source_indices) == len(target_indices)
            heuristic : callable, optional
                A function that takes two node indices (u, target) and returns an
                estimate of the distance
                between them. Only used when algorithm="astar".

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
                msg = ("Source and target lists must have the same length for pairwise "
                       "computation")
                raise ValueError(msg)
            return self._pairwise_shortest_path(source_indices, target_indices,
                                                algorithm)

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

    def _compute_single_source_multiple_targets(self, source, targets, algorithm,
                                                **kwargs):
        """
        Computes shortest paths from a single source to multiple targets.
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

    def _pairwise_shortest_path(self, sources, targets, algorithm, **kwargs):
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

    def _all_pairs_shortest_path(self, sources, targets, algorithm, **kwargs):
        """
        Computes shortest paths between all pairs of sources and targets.
        """
        if algorithm == "dijkstra":
            paths = []

            # For each source, compute paths to all targets
            for source in sources:
                for target in targets:
                    _, path = nx.single_source_dijkstra(self.graph, source, target,
                                                        weight='weight')
                    path = self._ensure_path_endpoints(path, source, target)
                    paths.append(path)

            return paths

        else:
            # For other algorithms, compute each path individually
            return self._compute_all_pairs_shortest_paths(sources, targets, algorithm,
                                                          **kwargs)
