import networkx as nx
import geopandas as gpd
from substation_planning import set_node_indices_to_line_data


def get_minimal_cost_tree(source_gdf, target_gdf, line_gdf):
    source_gdf['node_type'] = 'slack'
    target_gdf['node_type'] = 'load_gens'
    nodes = gpd.pd.concat([source_gdf, target_gdf], ignore_index=True)
    set_node_indices_to_line_data(nodes, line_gdf)
    line_gdf['length_km'] = line_gdf.geometry.length / 1000.0

    # Keep only unique source-target pairs (also checks reverse order)
    unique_lines = dict()
    for lidx, source_idx, target_idx in zip(line_gdf.index, line_gdf['source_index'], line_gdf['target_index']):
        if ((source_idx, target_idx) in unique_lines or
                (target_idx, source_idx) in unique_lines or source_idx == target_idx):
            continue
        else:
            unique_lines[(source_idx, target_idx)] = lidx
    line_gdf = line_gdf.loc[list(unique_lines.values())]

    graph = nx.Graph()
    zipper = zip(line_gdf['source_index'], line_gdf['target_index'], line_gdf['length_km'])
    graph.add_edges_from([(f, t, {"weight": c}) for f, t, c in zipper])
    tree = nx.minimum_spanning_tree(graph)
    lines = []
    for u, v in tree.edges():
        line = line_gdf.loc[((line_gdf.source_index == u) & (line_gdf.target_index == v)) |
                            ((line_gdf.source_index == v) & (line_gdf.target_index == u))].index.values[0]
        lines.append(line)
    tree_gdf = line_gdf.loc[lines]
    tree_gdf.to_file("min_spanning_length.geojson")


if __name__ == '__main__':
    source_path = r"C:\Users\mhnn82\Documents\3_python_projects\pyorps\case_studies\substation_planning\shape\sources_substation.shp"
    targets_path = r"C:\Users\mhnn82\Documents\3_python_projects\pyorps\case_studies\substation_planning\shape\targets_substations.shp"
    save_path = r"C:\Users\mhnn82\Documents\3_python_projects\pyorps\case_studies\substation_planning\results"

    sources_gdf = gpd.read_file(source_path).to_crs("EPSG:32632")
    targets_gdf = gpd.read_file(targets_path).to_crs("EPSG:32632")
    line_data = gpd.read_file(r"C:\Users\mhnn82\Documents\3_python_projects\pyorps\case_studies\substation_planning"
                              r"\meshed_network_data.geojson").to_crs("EPSG:32632")
    print(sum(line_data["runtime_total"].unique()))
    #get_minimal_cost_tree(sources_gdf, targets_gdf, line_data)