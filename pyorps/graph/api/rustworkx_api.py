
# Third party
import rustworkx as rx
from numpy import where

# Project files
from pyorps.core.exceptions import NoPathFoundError, AlgorthmNotImplementedError
from .graph_library_api import *


class RustworkxAPI(GraphAPI):
    def init_chunk_size(self, chunk_size: Optional[int] = None) -> int:
        if chunk_size is None:
            self.chunk_size = 1
        else:
            self.chunk_size = chunk_size
        return self.chunk_size

    def get_graph(self) -> rx.PyGraph:
        nr_of_nodes = self.raster.shape[0] * self.raster.shape[1]
        nodes = range(nr_of_nodes)
        rxgraph = rx.PyGraph()
        rxgraph.add_nodes_from(nodes)
        return rxgraph

    def get_number_of_nodes(self) -> int:
        return self.graph.num_nodes()

    def get_number_of_edges(self) -> int:
        return self.graph.num_edges()

    def add_edge(self, from_node: int, to_node: int, cost: float) -> None:
        self.graph.add_edge(from_node, to_node, cost)

    def add_edges(self, from_nodes: Union[ndarray[int], list[int]], to_nodes: Union[ndarray[int], list[int]],
                  cost: Union[ndarray[float], list[float]]) -> None:
        self.graph.add_edges_from(list(zip(from_nodes, to_nodes, cost)))

    def get_graph_from_sparse_matrix(self,
                                     from_nodes: Union[ndarray[int], list[int]],
                                     to_nodes: Union[ndarray[int], list[int]],
                                     cost: Union[ndarray[float], list[float]]) -> None:
        self.graph.add_edges_from(list(zip(from_nodes, to_nodes, cost)))
        return self.graph

    def remove_isolates(self) -> None:
        indices_max_values = where(self.raster == 65535)
        nodes_max_values = ravel_multi_index(indices_max_values, self.raster.shape)
        self.graph.remove_nodes_from(nodes_max_values)

    def shortest_path(self) -> list[int]:
        if self.algorithm == "dijkstra":
            return self.dijkstra_shortest_path()
        elif self.algorithm == "astar":
            return self.astar_shortest_path()
        else:
            raise AlgorthmNotImplementedError(self.algorithm, self.__class__.__name__)

    def astar_shortest_path(self) -> list[int]:

        def goal_reached(node):
            return node == self.targets

        node_path = rx.astar_shortest_path(self.graph, self.source,
                                           goal_fn=goal_reached,
                                           edge_cost_fn=lambda edge_weight: edge_weight,
                                           estimate_cost_fn=lambda node: 0.0)
        return node_path

    def dijkstra_shortest_path(self) -> list[int]:
        paths = rx.dijkstra_shortest_paths(self.graph, self.source, self.targets,
                                           weight_fn=lambda edge_weight: edge_weight)
        if len(paths) == 0:
            raise NoPathFoundError(self.source, self.targets)
        node_path = paths[self.targets]
        return node_path
