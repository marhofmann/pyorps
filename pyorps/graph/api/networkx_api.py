from typing import Optional

# Third party
import networkx as nx
from numpy import concatenate

# Project files
from .graph_api import *
from pyorps.graph.exceptions import AlgorthmNotImplementedError, NoPathFoundError


class NetworkxAPI(GraphAPI):

    def init_chunk_size(self, chunk_size: Optional[int] = None) -> int:
        if chunk_size is None:
            self.chunk_size = 0
        else:
            self.chunk_size = chunk_size
        return self.chunk_size

    def get_graph(self):
        return nx.Graph()

    def get_number_of_nodes(self):
        return self.graph.number_of_nodes()

    def get_number_of_edges(self):
        return self.graph.number_of_edges()

    def add_edge(self, from_node, to_node, cost):
        self.graph.add_edge(from_node, to_node, weight=cost)

    def add_edges(self, from_nodes, to_nodes, cost):
        self.graph.add_weighted_edges_from(zip(from_nodes, to_nodes, cost))

    def get_graph_from_sparse_matrix(self, from_nodes, to_nodes, cost):
        # make scipy optional dependency!
        from scipy.sparse import coo_matrix
        if self.undirected:
            fn = concatenate([from_nodes, to_nodes])
            tn = concatenate([to_nodes, from_nodes])
            co = concatenate([cost, cost])
        else:
            fn, tn, co = from_nodes, to_nodes, cost
        sparse_matrix = coo_matrix((co, (fn, tn)))
        return nx.from_scipy_sparse_array(sparse_matrix)

    def remove_isolates(self):
        self.graph.remove_nodes_from(list(nx.isolates(self.graph)))

    def shortest_path(self) -> list[int]:
        if self.algorithm == "dijkstra":
            return self.dijkstra_shortest_path()
        else:
            raise AlgorthmNotImplementedError(self.algorithm, self.__class__.__name__)

    def dijkstra_shortest_path(self):
        try:
            node_path = nx.dijkstra_path(self.graph, self.source, self.target)
        except nx.NetworkXNoPath:
            raise NoPathFoundError(self.source, self.target)
        return node_path
