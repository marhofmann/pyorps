import geopandas as gpd

from pyorps.main import run
from pyorps.raster_graph import RasterGraph
from pyorps.raster.neighborhood import get_r2_neighborhood_directed
import power_grid_model as pgm

from pathlib import Path
import numpy as np
import pandas as pd

from power_grid_model import LoadGenType, ComponentType, DatasetType, ComponentAttributeFilterOptions
from power_grid_model import PowerGridModel, CalculationMethod, CalculationType
from power_grid_model import initialize_array, attribute_dtype
from power_grid_model.validation import assert_valid_input_data, validate_input_data
from power_grid_model.utils import json_serialize
import json
import pandapower as pp
from pandapower.topology import create_nxgraph
from networkx import cycle_basis


def read_and_concat_geojsons(directory):
    """
    Read all GeoJSON files from a directory and concatenate them into a single GeoDataFrame.

    Parameters:
    -----------
    directory : str or Path
        Path to the directory containing GeoJSON files

    Returns:
    --------
    GeoDataFrame
        Combined GeoDataFrame containing all data from the GeoJSON files
    """
    # Convert to Path object if string
    directory = Path(directory)

    # Get all .geojson files in the directory
    geojson_files = list(directory.glob('*.geojson'))

    if not geojson_files:
        print(f"No GeoJSON files found in {directory}")
        return None

    # Read each GeoJSON file into a GeoDataFrame
    gdfs = []
    for file in geojson_files:
        try:
            gdf = gpd.read_file(file)
            gdfs.append(gdf.to_crs("EPSG:32632"))
            print(f"Read {file.name} with {len(gdf)} features")
        except Exception as e:
            print(f"Error reading {file.name}: {e}")

    # Concatenate all GeoDataFrames
    if gdfs:
        combined_gdf = gpd.pd.concat(gdfs, ignore_index=True)
        print(f"Combined GeoDataFrame contains {len(combined_gdf)} features")
        return combined_gdf
    else:
        print("No valid GeoJSON files were read")
        return None


def set_node_indices_to_line_data(nodes, lines):
    coords_dict = {i: (geom.x, geom.y) for i, geom in nodes.geometry.items()}
    lines['source_index'] = lines.geometry.apply(find_matching_coord, coords_dict=coords_dict, tolerance=2, index=0)
    lines['target_index'] = lines.geometry.apply(find_matching_coord, coords_dict=coords_dict, tolerance=2, index=-1)


def find_matching_coord(line_coords, coords_dict, tolerance, index=0):
    x, y = line_coords.coords[index]
    for i, coords in coords_dict.items():
        cx, cy = coords
        if abs(x - cx) < tolerance and abs(y - cy) < tolerance:
            return i
    return None


def get_n_nearest_neighbors_generator(gdf, n):
    """Generator that yields the n nearest neighbors for each point"""
    geoms = gdf.geometry
    edges = []
    for source_index, source in zip(gdf.index, gdf.geometry):
        distances = geoms.distance(source).sort_values(ascending=True)
        n_targets = distances.head(n + 1).index.tolist()
        n_targets.remove(source_index)
        for t in list(n_targets):
            if (source_index, t) in edges or (t, source_index) in edges:
                n_targets.remove(t)
            else:
                edges.append((t, source_index))
        print(source_index, n_targets)
        targets_coords = [tuple(geom.coords[0]) for geom in gdf.loc[n_targets, 'geometry'].values]
        source = [tuple(gdf.at[source_index, "geometry"].coords[0])]
        yield source, targets_coords


def create_meshed_network(raster_path, sources, targets, save_path):
    source_coords = [tuple(geom.coords[0]) for geom in sources.geometry.values]
    target_coords = [tuple(geom.coords[0]) for geom in targets.geometry.values]
    save_path_primary_lines = f"{save_path}_primary_lines.geojson"
    steps = get_r2_neighborhood_directed()
    print("\n\nCreating meshed network - Starting with primary substation lines!")
    #find_and_save_paths(raster_path, source_coords[0], target_coords, save_path_primary_lines, steps)
    print(f"Primary lines from substation saved to {save_path_primary_lines}!")
    print("\nConnecting secondary substations with lines...")

    neighbours = get_n_nearest_neighbors_generator(targets, 5)
    for source, targets in neighbours:
        if targets:
            save_path_secondary_lines = f"{save_path}_secondary_lines{source}.geojson"
            find_and_save_paths(raster_path, source, targets, save_path_secondary_lines, steps)

    print("Done - meshed network created!")


def find_and_save_paths(raster_path, source, targets, save_path, steps):
    raster_graph = RasterGraph(
        raster_source=raster_path,
        source_coords=source,
        target_coords=targets,
        search_space_buffer_m=3000,
        neighborhood_str="r2",
        steps=steps,
        graph_api="cython"
    )
    # Find route
    print(f"\nFinding routes for source {source} ...")
    raster_graph.find_route(algorithm="dijkstra")
    path_gdf = raster_graph.create_path_geodataframe()
    path_gdf.to_file(save_path)
    print(f"Saved routes to {save_path}\n")


def create_pandapower_model_from_geodata(
        source_gdf,
        target_gdf,
        line_gdf,
        line_std_type=None):
    if line_std_type is None:
        line_std_type = "NA2XS2Y 1x240 RM/25 12/20 kV"
    line_gdf, nodes = prepare_geodata_for_net_creation(source_gdf, target_gdf, line_gdf)
    net = pp.create_empty_network()
    pp.create_buses(net, vn_kv=20, nr_buses=nodes.shape[0], index=nodes.index)
    slacks = nodes.loc[nodes.node_type == "slack"]
    load_buses = nodes.loc[nodes.node_type != "slack"]
    for slack in slacks.index.values:
        pp.create_ext_grid(net, bus=slack)
    geodata = [list(geom.coords) for geom in line_gdf.geometry.values]
    pp.create_lines(net, from_buses=line_gdf.source_index.values, to_buses=line_gdf.target_index.values,
                           std_type=line_std_type, length_km=line_gdf.length_km, geodata=geodata)
    net.line['cost'] = line_gdf['path_cost'].values
    for bus, line in zip(net.line.from_bus, net.line.index):
        pp.create_switch(net, bus=bus, element=line, et='l')
    s = np.repeat(1.11, len(load_buses.index.values))
    p, q = pp.pq_from_cosphi(s, cosphi=0.9, qmode='overexcited', pmode='load')
    pp.create_loads(net, buses=load_buses.index.values, p_mw=p, q_mvar=q)
    pp.runpp(net)
    pp.to_json(net, "substation_planning_pp_grid_1_mva.json")
    return net


def prepare_geodata_for_net_creation(source_gdf, target_gdf, line_gdf):
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

    lines = line_gdf.loc[list(unique_lines.values())]
    return lines, nodes


def create_power_grid_from_geodata(
        source_gdf,
        target_gdf,
        line_gdf,
        line_params=None):

    if line_params is None:
        line_params = dict()

    # Standard type - NA2XS2Y 1x240 RM/25 12/20 kV
    default_line_params = dict()
    default_line_params['r_ohm_per_km'] = 0.122
    default_line_params['x_ohm_per_km'] = 0.112
    default_line_params['c_nf_per_km'] = 304.0
    default_line_params['max_i_ka'] = 0.421

    line_gdf, nodes = prepare_geodata_for_net_creation(source_gdf, target_gdf, line_gdf)

    # node
    nr_of_nodes = nodes.shape[0]
    node = initialize_array(DatasetType.input, ComponentType.node, nr_of_nodes)
    node["id"] = nodes.index.values
    node["u_rated"] = np.repeat(20e3, nr_of_nodes)

    # line
    nr_of_lines = line_gdf.shape[0]
    line = initialize_array(DatasetType.input, ComponentType.line, nr_of_lines)
    line["id"] = np.arange(nr_of_nodes + 1, nr_of_lines + nr_of_nodes + 1, 1)
    line["from_node"] = line_gdf.source_index.values.astype(int)
    line["to_node"] = line_gdf.target_index.values.astype(int)
    line["from_status"] = np.repeat(1, nr_of_lines)
    line["to_status"] = np.repeat(1, nr_of_lines)
    line["r1"] = line_params.get('r_ohm_per_km', default_line_params['r_ohm_per_km']) * line_gdf.length_km
    line["x1"] = line_params.get('x_ohm_per_km', default_line_params['x_ohm_per_km']) * line_gdf.length_km
    line["c1"] = line_params.get('c_nf_per_km', default_line_params['c_nf_per_km']) * line_gdf.length_km * 10e-9
    line["tan1"] = np.repeat(0, nr_of_lines)
    line["i_n"] = np.repeat(line_params.get('max_i_ka', default_line_params['max_i_ka']) * 10e2, nr_of_lines)

    # load
    load_gens = nodes.loc[nodes.node_type == 'load_gens']
    number_of_load_gens = load_gens.shape[0]
    sym_load = initialize_array(DatasetType.input, ComponentType.sym_load, number_of_load_gens)
    id_offset = nr_of_lines + nr_of_nodes
    sym_load["id"] = np.arange(id_offset + 1, id_offset + number_of_load_gens + 1, 1)
    sym_load["node"] = load_gens.index.values
    sym_load["status"] = np.repeat(1, number_of_load_gens)
    sym_load["type"] = [LoadGenType.const_power for _ in range(number_of_load_gens)]
    sym_load["p_specified"] = np.repeat(1 * 10e6, number_of_load_gens)
    sym_load["q_specified"] = np.repeat(0.2 * 10e6, number_of_load_gens)

    # source
    slacks = nodes.loc[nodes.node_type == 'slack']
    number_of_slacks = slacks.shape[0]
    id_offset += number_of_load_gens
    source = initialize_array(DatasetType.input, ComponentType.source, number_of_slacks)
    source["id"] = np.arange(id_offset + 1, id_offset + number_of_slacks + 1, 1)
    source["node"] = slacks.index.values
    source["status"] = np.repeat(1, number_of_slacks)
    source["u_ref"] = np.repeat(1, number_of_slacks)

    # all
    input_data = {
        ComponentType.node: node,
        ComponentType.line: line,
        ComponentType.sym_load: sym_load,
        ComponentType.source: source,
    }

    assert_valid_input_data(input_data=input_data, calculation_type=CalculationType.power_flow, symmetric=True)
    model = PowerGridModel(input_data)
    model.calculate_power_flow(symmetric=True)
    output = json_serialize(input_data)
    with open("pgm_model.json", 'w') as f:
        json.dump(output, f)
    return input_data


def get_switches_on_path(net, sp):
    path_lines = net.line.loc[net.line.from_bus.isin(sp) & net.line.to_bus.isin(sp)].index.values
    switches = net.res_switch.loc[net.switch.element.isin(path_lines) &
                                  net.switch.bus.isin(sp) &
                                  net.switch.closed &
                                  (net.switch.et == 'l')]
    return switches


def open_simple_cycles_lowest_current(net, nxg=None):
    if nxg is None:
        nxg = create_nxgraph(net, multi=False)
    iterations = 0
    while cycles := cycle_basis(nxg):
        all_nodes_in_cycles = [node for cycle in cycles for node in cycle]
        switches = get_switches_on_path(net, all_nodes_in_cycles)

        min_switch = switches.i_ka.idxmin()
        net.switch.at[min_switch, 'closed'] = False
        pp.runpp(net)
        nxg = create_nxgraph(net, multi=False)
        iterations += 1
    print(f"open_simple_cycles: Optimization took {iterations} iterations")


if "__main__" == __name__:
    line_data_directory = r"C:\Users\mhnn82\Documents\3_python_projects\pyorps\case_studies\substation_planning"
    line_data_concatenated = r"C:\Users\mhnn82\Documents\3_python_projects\pyorps\case_studies\substation_planning\shape\meshed_network_data.geojson"
    source_path = r"C:\Users\mhnn82\Documents\3_python_projects\pyorps\case_studies\substation_planning\shape\sources_substation.shp"
    targets_path = r"C:\Users\mhnn82\Documents\3_python_projects\pyorps\case_studies\substation_planning\shape\targets_substations.shp"
    save_path = r"C:\Users\mhnn82\Documents\3_python_projects\pyorps\case_studies\substation_planning\results"
    #
    line_params = {
        "r_ohm_per_km": 0.122,
        "x_ohm_per_km": 0.112,
        "c_nf_per_km": 304,
        "max_i_ka": 0.421,
    }
    sources_gdf = gpd.read_file(source_path).to_crs("EPSG:32632")
    targets_gdf = gpd.read_file(targets_path).to_crs("EPSG:32632")
    #line_data = read_and_concat_geojsons(line_data_directory)
    #line_data.to_file("meshed_network_data.geojson")
    line_data = gpd.read_file(line_data_concatenated).to_crs("EPSG:32632")
    net = create_pandapower_model_from_geodata(source_gdf=sources_gdf, target_gdf=targets_gdf, line_gdf=line_data)
    #print()
    #net.switch.closed = True
    #open_switches = [16, 210, 208, 17, 209, 12, 4, 107, 2, 105, 104, 190, 90, 5, 88, 87, 86, 187, 184, 182, 183, 6,
    #                 75, 74, 73, 72, 181, 175, 171, 165, 31, 154, 172, 51, 50, 49, 48, 7, 157, 153, 185, 81, 80, 79,
    #                 78, 77, 180, 71, 70, 69, 68, 67, 9, 173, 162, 159, 156, 176, 174, 10, 55, 54, 53, 52, 169, 161,
    #                 160, 164, 163, 41, 40, 39, 1, 37, 168, 206, 103, 13, 101, 100, 99, 205, 203, 222, 14, 135, 134,
    #                 133, 132, 235, 230, 223, 141, 3, 139, 138, 137, 241, 240, 20, 218, 216, 213, 66, 65, 64, 63, 21,
    #                 189, 177, 167, 36, 61, 60, 59, 58, 57, 166, 170, 46, 45, 44, 23, 42, 236, 234, 151, 24, 149, 148,
    #                 147, 227, 27, 228, 25, 145, 144, 143, 142, 238, 26, 231, 233, 232, 225, 239, 224, 29, 221, 152,
    #                 32,
    #                 195, 194, 193, 192, 33, 204, 201, 199, 202, 200, 188, 198, 207, 95, 94, 106, 92, 91, 85, 84, 83,
    #                 82, 127, 122, 116, 111, 98, 97, 96, 128, 123, 117, 112, 121, 129, 124, 118, 113, 109, 130, 125,
    #                 119, 114, 110, 131, 108, 120, 115, 214, 212, 211, 217, 215, 219]
    #open_switches = [185, 9, 6, 5, 81, 80, 2, 78, 77, 172, 171, 7, 50, 49, 48, 47, 157, 153, 165, 164, 163, 162,
    # 161, 41, 40, 39, 38, 37, 11, 198, 17, 196, 95, 94, 93, 92, 91, 190, 186, 206, 103, 102, 101, 13, 99, 34, 203, 222, 136, 135, 134, 133, 0, 235, 230, 219, 218, 126, 125, 124, 123, 122, 217, 212, 209, 210, 208, 207, 108, 107, 106, 16, 104, 188, 214, 213, 211, 115, 3, 113, 112, 111, 216, 215, 120, 119, 118, 18, 116, 223, 4, 140, 139, 138, 137, 29, 240, 20, 221, 131, 130, 129, 128, 127, 179, 66, 65, 64, 63, 62, 36, 177, 174, 23, 178, 61, 60, 59, 58, 22, 176, 166, 170, 169, 46, 45, 44, 43, 42, 236, 234, 151, 150, 149, 148, 24, 227, 229, 26, 146, 145, 144, 25, 142, 238, 237, 231, 233, 232, 27, 239, 224, 30, 160, 159, 31, 152, 32, 155, 154, 33, 194, 193, 192, 191, 204, 201, 199, 202, 1, 187, 182, 181, 71, 180, 69, 68, 67, 173, 76, 183, 74, 73, 72, 175, 85, 84, 83, 82, 86, 52, 98, 97, 96, 184, 53, 121, 109, 88, 79, 110, 89, 55, 90, 56]
    net = pp.from_json(r"C:\Users\mhnn82\Documents\3_python_projects\pyorps\case_studies\substation_planning\10000_6926664_19_apparent_power_slack_mva_cost.json")
    open_switches = net.switch.loc[~net.switch.closed].index.values
    #net.switch.loc[open_switches, 'closed'] = False
    #pp.runpp(net)
    #print(net.res_ext_grid)
    #net.res_line.loc[net.switch.loc[open_switches, 'element'], 'loading_percent'] = 0.0
    line_data['loading_percent'] = 0.0
    for fb, tb, loading in zip(net.line.from_bus, net.line.to_bus, net.res_line.loading_percent):
        line_data.loc[((line_data.source_index == fb) & (line_data.target_index == tb) |
                       (line_data.source_index == tb) & (line_data.target_index == fb)), "loading_percent"] = loading
    line_data.to_file(save_path + r"\10000_6926664_19_apparent_power_slack_mva_cost.geojson")



    #input_data = create_power_grid_from_geodata(sources_gdf, targets_gdf, line_data)
    #ga = GeneticAlgorithm(
    #    input_data=input_data,
    #    population_size=500,
    #    tournament_size=5,
    #    nr_of_generations=50,
    #    crossover_rate=0.5,
    #    mutation_rate=0.5,
    #    max_trafo_loading=110
    #)
    #
    ## Run the algorithm
    #final_result, best_config = ga.run()
    #
    ## Print the best configuration
    #print(f"Best configuration found: {best_config}")
    #print(f"Total losses: {ga.get_losses(final_result)} MW")


