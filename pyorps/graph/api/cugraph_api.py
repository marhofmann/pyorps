
# Third party
from numpy import int32
import cugraph
import cudf
import gc

# Project files
from .graph_api import GraphAPI
from pyorps.graph.exceptions import AlgorthmNotImplementedError, NoPathFoundError


class CugraphAPI(GraphAPI):


    def shortest_path(self) -> list[int]:
        if self.algorithm == "dijkstra":
            return self.dijkstra_shortest_path()
        else:
            raise AlgorthmNotImplementedError(self.algorithm, self.__class__.__name__)

    def init_chunk_size(self, chunk_size):
        if chunk_size is None:
            return 10_000_000_000
        else:
            return chunk_size

    def add_edges(self, from_nodes, to_nodes, cost):
        self.graph.add_weighted_edges_from(zip(from_nodes, to_nodes, cost))

    def add_edge(self, from_node, to_node, cost):
        self.graph.add_edge(from_node, to_node, weight=cost)

    def get_graph(self):
        return cugraph.Graph()

    def get_graph_from_sparse_matrix(self, from_nodes, to_nodes, cost):
        self.edge_list =cudf.DataFrame({
            "source": from_nodes.astype(int32),
            "destination": to_nodes.astype(int32),
            "weight": cost
            })
        del from_nodes, to_nodes, cost
        gc.collect()
        self.graph.from_cudf_edgelist(input_df=self.edge_list, weight="weight", symmetrize=True, renumber=False)
        return self.graph

    def get_number_of_nodes(self):
        return self.graph.number_of_nodes()

    def get_number_of_edges(self):
        return self.graph.number_of_edges()

    def remove_isolates(self):
        pass

    def dijkstra_shortest_path(self):
        paths_cudf = cugraph.shortest_path(self.graph, source=self.source)
        if self.targets not in paths_cudf.vertex:
            raise NoPathFoundError(self.source, self.targets)
        target = self.targets
        node_path = [target]
        while target != self.source:
            target = paths_cudf.loc[paths_cudf.vertex == target, 'predecessor'].iloc[0, 0]
            node_path.append(target)
        return node_path

