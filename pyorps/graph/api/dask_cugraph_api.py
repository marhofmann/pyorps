from gc import enable

# Third party
import cugraph
import dask_cudf
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
import cugraph.dask as dask_cugraph
import cugraph.dask.comms.comms as Comms
from pyarrow import table
from pyarrow.parquet import write_table
from ..benchmarking import PerfCounter
import pandas as pd

import cudf
import ucp
import cupy as cp
import dask

# Project files
from .graph_api import GraphAPI
from ..raster.traversal2d import construct_edges_for_sparse_matrix
from ..raster.dask_traversal2d import construct_edges_for_dask_cudf, run_cuda_kernels, construct_edges_for_dask_cudf_undelayed
from pyorps.graph.exceptions import AlgorthmNotImplementedError


class DaskCugraphAPI(GraphAPI):


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

    #def create_graph(self):
    #    #self.cu_df_df = run_cuda_kernels(self.raster, self.steps)
    #    pass

    def get_graph_from_sparse_matrix(self, from_nodes, to_nodes, cost):
        pa_table = table({
            "source": from_nodes.astype(int),
            "destination": to_nodes.astype(int),
            "weight": cost
        })
        write_table(pa_table, "graph_data.parquet")
        return self.graph

    def get_number_of_nodes(self):
        return None

    def get_number_of_edges(self):
        return None

    def remove_isolates(self):
        pass

    def dijkstra_shortest_path(self):
        #from_nodes, to_nodes, cost = construct_edges_for_sparse_matrix(self.raster, self.steps, self.ignore_max_cost)

        cluster = LocalCUDACluster()
        client = Client(cluster)
        Comms.initialize(p2p=True)

        # Version2
        chunksize = dask_cugraph.get_chunksize("graph_data.parquet")
        dask_edge_list = dask_cudf.read_parquet("graph_data.parquet", blocksize=chunksize)

        #Version3
        #edges_list = list()
        #raster = cp.asarray(self.raster)
        #for step in self.steps:
        #    edges = dask.delayed(construct_edges_for_dask_cudf_undelayed)(raster, cp.asarray(step))
        #    edges_list.append(edges)
        #cudf_edge_list= cudf.concat(list(dask.compute(*edges_list)))
        #dask_edge_list = dask_cudf.from_cudf(cudf_edge_list, npartitions=4)

        # Version1
        #edges_data = construct_edges_for_dask_cudf(self.raster, self.steps)

        # Version4
        #dask_edge_list = dask_cudf.from_cudf(self.cu_df_df, npartitions=4)

        #dask_edge_list = dask_cudf.DataFrame.from_dict({
        #    "source": from_nodes.astype(int),
        #    "destination": to_nodes.astype(int),
        #    "weight": cost
        #}, npartitions=4)

        self.graph = cugraph.Graph(directed=True)
        self.graph.from_dask_cudf_edgelist(input_ddf=dask_edge_list, weight="weight", renumber=True)
        del dask_edge_list
        paths_dask_cudf = dask_cugraph.sssp(input_graph=self.graph, source=self.source)
        self.number_of_nodes = self.graph.number_of_nodes()
        self.number_of_edges = self.graph.number_of_edges()
        paths_cudf = paths_dask_cudf.compute()
        Comms.destroy()
        client.close()
        cluster.close()

        with PerfCounter("Get path nodes from cudf"):
            target = self.targets
            node_path = [target]
            vert_pred_dict = dict(zip(paths_cudf.vertex.to_numpy(), paths_cudf.predecessor.to_numpy()))

            while target != self.source:
                target = vert_pred_dict[target]
                node_path.append(target)


        print(f"Graph created with {self.number_of_nodes:_} nodes and {self.number_of_edges:_} edges")
        return node_path
