import warnings

# Third party
from numpy import array, where
import igraph as ig
# Project files
from pyorps.graph.exceptions import AlgorthmNotImplementedError, NoPathFoundError
from .graph_api import *


class IgraphAPI(GraphAPI):

    def init_chunk_size(self, chunk_size: Optional[int] = None) -> int:
        if chunk_size is None or chunk_size <= 0:
            self.chunk_size = 1
        else:
            self.chunk_size = chunk_size
        return self.chunk_size

    def get_graph(self):
        nr_of_nodes = self.raster.shape[0] * self.raster.shape[1]
        ig_graph = ig.Graph()
        ig_graph.add_vertices(nr_of_nodes)
        return ig_graph

    def get_number_of_nodes(self):
        return self.graph.vcount()

    def get_number_of_edges(self):
        return self.graph.ecount()

    def add_edge(self, from_node, to_node, cost):
        self.graph.add_edge(from_node, to_node, weight=cost)

    def add_edges(self, from_nodes, to_nodes, cost):
        self.graph.add_edges(list(zip(from_nodes, to_nodes)), {"weight": cost})

    def get_graph_from_sparse_matrix(self, from_nodes, to_nodes, cost):
        pass

    def remove_isolates(self):
        degree_view = array(self.graph.degree(self.graph.vs()))
        isolates = where(degree_view == 0)[0]
        self.graph.delete_vertices(isolates)

    def shortest_path(self):
        if self.algorithm == "dijkstra":
            return self.dijkstra_shortest_path()
        else:
            raise AlgorthmNotImplementedError(self.algorithm, self.__class__.__name__)

    def dijkstra_shortest_path(self):
        warnings.filterwarnings("error")
        try:
            node_path = self.graph.get_shortest_paths(self.source, self.target, weights="weight")
        except (RuntimeWarning, SystemError):
            raise NoPathFoundError(self.source, self.target)
        return node_path[0]
