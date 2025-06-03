
"""
PYORPS Case Study: Multi-Criteria Optimization for Distribution Grid Planning

This Python file implements a case study demonstrating how PYORPS can be integrated
into distribution grid planning for multi-criteria optimization. The study focuses on
connecting a photovoltaic (PV) power plant to the medium-voltage (MV) grid
MV-Oberrhein using underground cables, considering technical constraints alongside
economic and environmental aspects.
The code leverages publicly available land registry data to create a cost raster that
represent different terrain types and their associated construction costs. It then
uses PYORPS for pathfinding with varying neighborhood structures (R0, R1, R2, R3) to
find optimal cable routes between the PV plant and potential points of common coupling
(PCC). After identifying possible routes, the code evaluates their technical
feasibility through power flow simulations, checking voltage constraints and
equipment loading. The final  result identifies the most cost-effective and
technically valid connection options for various neighborhood searches.

Key components:
- GIS data processing and cost raster creation
- Route optimization with multiple neighborhood structures
- Technical validation through power flow analysis
- Economic evaluation of connection options
"""

from typing import Optional, Union, Any

import pandapower as pp
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon, LineString
from numpy import ndarray

import pyorps


# =============================================================================
# GIS and Geo-processing Functions
# =============================================================================
# This section contains functions for working with geographic data, including
# converting network bus data to geospatial formats, finding closest connection
# points, and creating bounding boxes for spatial analysis.


def get_bus_geoseries(
        net: pp.pandapowerNet,
        net_crs: str,
        buses: Optional[list[int]] = None
) -> gpd.GeoSeries:
    """
    Convert bus geodata to a GeoSeries.

    Args:
        net: pandapower network
        net_crs: coordinate reference system of the network
        buses: optional list of bus indices to include (defaults to all buses)

    Returns:
        GeoSeries containing bus points with bus indices as index
    """
    if not buses:
        buses = net.bus.index
    return net.bus.loc[buses].geo.geojson.as_geoseries.to_crs(crs=net_crs)


def find_closest_connection_points(
        net: pp.pandapowerNet,
        sources: gpd.GeoSeries,
        nr_of_buses: int,
        net_crs: str,
        sources_crs: str
) -> dict[Any, gpd.GeoSeries]:
    """
    Find closest network connection points for given source locations.

    Args:
        net: pandapower network
        sources: GeoSeries of source point geometries
        nr_of_buses: number of closest buses to find
        net_crs: coordinate reference system of the network
        sources_crs: coordinate reference system of the sources

    Returns:
        dictionary mapping source indices to GeoSeries of nearest buses
    """
    buses = net.bus.loc[net.bus.index.isin(net.trafo.hv_bus) &
                        (net.bus.vn_kv == 20)].index.to_list()

    bus_geoseries = get_bus_geoseries(net, net_crs=net_crs, buses=buses)
    bus_geoseries = bus_geoseries.to_crs(sources_crs)
    closest_connection_points = dict()

    for source_index, source in zip(sources.index, sources.geometry):
        # find the nearest buses to the sources
        distances = bus_geoseries.geometry.distance(source).sort_values(ascending=True)
        closest_connection_points[source_index] = bus_geoseries.loc[distances.iloc[0:nr_of_buses].index]

    return closest_connection_points


def create_bbox_from_net_data(
        net: pp.pandapowerNet,
        buffer: float = 1000,
        create_gdf: bool = False,
        net_crs: Optional[str] = None,
        reference_crs: Optional[str] = None,
) -> Union[tuple[float, float, float, float], gpd.GeoDataFrame]:
    """
    Create a bounding box from network bus geodata.

    Args:
        net: pandapower network
        buffer: buffer distance to add around the bounding box (meters)
        create_gdf: Whether to return a GeoDataFrame (True) or joust the bounding
        coordinates as a tuple (False)
        net_crs: Coordinate reference system of the coordinates in the network
        reference_crs: Coordinate reference system of the output coordinates

    Returns:
        Either tuple of (minx, miny, maxx, maxy) or GeoDataFrame with the bounding box
        polygon
    """
    bus_geoseries = get_bus_geoseries(net, net_crs=net_crs).to_crs(reference_crs)
    minx, miny, maxx, maxy = bus_geoseries.total_bounds
    minx, miny = minx - buffer, miny - buffer
    maxx, maxy = maxx + buffer, maxy + buffer

    if create_gdf:
        bbox = Polygon([(maxx, maxy), (minx, maxy), (minx, miny), (maxx, miny)])
        bbox_gdf = gpd.GeoDataFrame(index=[0], geometry=[bbox], crs=reference_crs)
        return bbox_gdf
    else:
        return minx, miny, maxx, maxy


# =============================================================================
# Cost and Route Analysis Functions
# =============================================================================
# These functions handle the preparation of raster data with cost values for
# different terrain types and calculate the total cost of potential routes.
# The cost model incorporates construction expenses for different terrains.


def prepare_raster_data(
        net: pp.pandapowerNet,
        net_crs: str,
        gis_data_crs: str,
        gis_data_source: str | dict[str, str],
        save_path: str
) -> None:
    """
    Prepare raster data for route optimization.

    Args:
        net: pandapower network
        net_crs: coordinate reference system of the network
        gis_data_crs: coordinate reference system of the GIS data
        gis_data_source: path to the GIS data file
        save_path: path to save the raster data

    Returns:
        tuple of (raster, transform)
    """
    bbox = create_bbox_from_net_data(net, create_gdf=True, net_crs=net_crs,
                                     reference_crs=gis_data_crs)
    gis_dataset = pyorps.initialize_geo_dataset(gis_data_source, bbox=bbox)
    gis_dataset.load_data()

    # For the cost_assumptions dictionary:
    # High values (65535) represent areas that should be avoided when routing
    # Lower values represent preferred routing areas with varying costs
    # Cost values reflect construction difficulty and environmental impact

    # Define cost assumptions for different terrain types
    cost_assumptions = {
        'objektname': {
            # Residential area - avoid
            'Wohnbaufläche': 65535,
            # Industrial area - avoid
            'Industrie- und Gewerbefläche': 65535,
            # Special functional area - avoid
            'Fläche besonderer funktionaler Prägung': 65535,
            # Quarry/mining - avoid
            'Tagebau/Grube/Steinbruch': 65535,
            # Cemetery - avoid
            'Friedhof': 65535,
            # Waste dump - avoid
            'Halde': 65535,
            # Swamp - avoid
            'Sumpf': 65535,
            # Airport - avoid
            'Flugverkehr': 65535,

            # Road traffic
            'Straßenverkehr': 178,
            # Sports/leisure area
            'Sport-, Freizeit- und Erholungsfläche': 107,
            # Path
            'Weg': 97,
            # Agricultural land
            'Landwirtschaft': 107,
            # Forest
            'Wald': 365,
            # Flowing water
            'Fließgewässer': 155,
            # Woodland
            'Gehölz': 365,
            # Mixed use area
            'Fläche gemischter Nutzung': 107,
            # Square/plaza
            'Platz': 152,
            # Wasteland/non-vegetated area
            'Unland/Vegetationslose Fläche': 92,
            # Standing water
            'Stehendes Gewässer': 155,
            # Railway
            'Bahnverkehr': 415,
        }
    }

    geo_rasterizer = pyorps.GeoRasterizer(gis_dataset, cost_assumptions, bbox)
    geo_rasterizer.rasterize(save_path=save_path)


def get_cost(
        lines: gpd.GeoDataFrame,
        line: int,
        ignore_category: Any=65535
):
    """
    Calculate the cost of a line based on terrain types it crosses.

    Args:
        lines: GeoDataFrame containing line data
        line: Index of the line to calculate cost for
        ignore_category: Cost category to ignore (typically used for avoided areas)

    Returns:
        Total cost of the line
    """
    cost = 0.0
    for cat in [92, 97, 107, 152, 155, 178, 365, 65535]:
        if cat == ignore_category:
            continue
        category_length = lines.at[line, 'length_cost_' + str(cat)]
        if pd.isnull(category_length):
            continue
        cost += category_length * int(cat)
    return cost

# =============================================================================
# Main Case Study Functions
# =============================================================================


def find_routes_for_connection_lines(
        net: pp.pandapowerNet,
        raster_path : str,
        result_path: str,
        sources: gpd.GeoSeries,
        net_crs: str,
        gis_data_crs: str
) -> gpd.GeoDataFrame:
    """
    Execute the main MV-Oberrhein case study with route finding.

    Args:
        net: pandapower network
        raster_path: Path to the rasterized cost data
        result_path: Path to save the resulting routes
        sources: GeoSeries containing source points (PV converter station)
        net_crs: Coordinate reference system of the network
        gis_data_crs: Coordinate reference system of the GIS data

    Returns:
        GeoDataFrame containing all found paths
    """
    all_paths = pyorps.PathCollection()
    all_targets = find_closest_connection_points(net, sources, nr_of_buses=20,
                                                 net_crs=net_crs,
                                                 sources_crs=gis_data_crs)
    targets = all_targets[0]

    # Find routes with different neighborhood settings
    for r in ['r0', 'r1', 'r2', 'r3']:
        # The R3 neighborhood must be processed in clusters due to memory constraints
        # Each cluster contains 2 potential connection points to keep the graph size
        # manageable
        if r == 'r3':
            for target_indices in [[133, 141],
                                   [138, 170],
                                   [1, 2],
                                   [101, 106]]:
                ts = targets.loc[target_indices]
                path_finder = pyorps.PathFinder(raster_path, source_coords=sources,
                                                target_coords=ts, neighborhood_str=r,
                                                ignore_max_cost=False)
                paths = path_finder.find_route()
                for path in paths:
                    all_paths.add(path)
        else:
            ts = targets.loc[[1, 2, 133, 141, 138, 170, 101, 106]]
            path_finder = pyorps.PathFinder(raster_path, source_coords=sources,
                                            target_coords=ts, neighborhood_str=r,
                                            ignore_max_cost=False)
            paths = path_finder.find_route()
            for path in paths:
                all_paths.add(path)
    all_paths_gdf = gpd.GeoDataFrame(all_paths.to_geodataframe_records(),
                                     crs=gis_data_crs)
    all_paths_gdf.to_file(result_path)
    return all_paths_gdf


def analyze_mv_oberrhein(
        result_path: str,
        net_crs: str,
        gis_data_crs: str
) -> None:
    """
    Analyze results from the MV-Oberrhein case study.

    Args:
        result_path: Path to the file containing route results
        net_crs: Coordinate reference system of the network
        gis_data_crs: Coordinate reference system of the GIS data

    Returns:
        GeoDataFrame with the best connection lines for each neighborhood
    """
    print("RUNNING PYORPS CASE-STUDY 'MV-OBERRHEIN'\n")
    lines = gpd.read_file(result_path)

    # Initialize network and bus data
    net = pp.networks.mv_oberrhein(scenario='generation', separation_by_sub=True,
                                   include_substations=True)[1]
    trafo_stations = net.bus.loc[net.bus.index.isin(net.trafo.hv_bus) &
                                 (net.bus.vn_kv == 20)].index.to_list()

    # Change grid topology by moving a sectioning point (otherwise all neighborhoods
    # lead to the same PCC)
    net.switch.at[107, 'closed'] = True
    net.switch.at[242, 'closed'] = False

    bus_geoseries = get_bus_geoseries(net, net_crs=net_crs, buses=trafo_stations)
    bus_geoseries = bus_geoseries.to_crs(gis_data_crs)

    # Create test PV generator
    line_type = "NA2XS2Y 1x95 RM/25 12/20 kV"
    from_bus = pp.create_bus(net, vn_kv=20, name='PV-Site')
    new_sgen = pp.create_sgen_from_cosphi(net, bus=from_bus, sn_mva=1.0,
                                          mode="underexcited", cos_phi=0.90)

    # Set some parameters to all static generators
    net.ext_grid.vm_pu = 1.04
    p_mw, q_mvar = pp.pq_from_cosphi(net.sgen.sn_mva, cosphi=0.90,
                                     qmode="underexcited", pmode="gen")
    net.sgen.p_mw = p_mw
    net.sgen.q_mvar = q_mvar

    # As this case study only serves to demonstrate the capabilities of PYORPS,
    # we deliberately selected a very low simultaneity factor for all PV systems
    # connected to the medium voltage distribution grid. This decision was made because
    # high levels of simultaneous PV generation would exceed the technical limitations
    # of the existing grid infrastructure - even without any additional PV system!
    net.sgen.scaling = 0.7
    print(f"Created PV-Plant with:\n{net.sgen.loc[new_sgen]}\n\n")
    original_sgen_p_mw = net.sgen.p_mw.values

    # Initialize containers
    geometries = []
    buses_memo = set()
    data = dict()

    # Analyze different neighborhood routing results
    neighborhood_groups = lines.groupby('neighborhood')
    for neighborhood_str in ['r0', 'r1', 'r2', 'r3']:
        neighborhood_lines = neighborhood_groups.get_group(neighborhood_str)

        # Process each line in the neighborhood results
        for line_idx, geom in zip(neighborhood_lines.index, neighborhood_lines.geometry):
            line_result_data = evaluate_connection_line(net, line_idx, from_bus, line_type, geom, neighborhood_lines,
                                                        bus_geoseries, original_sgen_p_mw, buses_memo)
            data[len(data)] = line_result_data
            geometries.append(geom)

    # Create and save results
    result = pd.DataFrame.from_dict(data, orient='index')
    result_gdf = gpd.GeoDataFrame(result, geometry=geometries, crs=gis_data_crs).sort_values(['Bus', 'Neighborhood'])

    # Find valid options meeting voltage and loading criteria
    # - Equipment loading < 50% of rated capacity
    # - Bus voltages within 0.95-1.05 p.u.
    # - Voltage rise < 0.02 p.u. with PV generation
    valid_options = result_gdf.loc[(result_gdf['Overall dV [p.u.]'] < 0.02) &
                                   (result_gdf['Max. V [p.u.]'] < 1.05) &
                                   (result_gdf['Min. V [p.u.]'] > 0.95) &
                                   (result_gdf['Max. line loading [%]'] < 50) &
                                   (result_gdf['Transformer loading [%]'] < 50)]
    print(valid_options.loc[:, ['Neighborhood', 'Bus', 'Overall dV [p.u.]', 'Line cost [€]', 'Line length [km]']])
    print(valid_options.sort_values(['Line cost [€]', 'Overall dV [p.u.]'], ascending=True).loc[:,
          ['Neighborhood', 'Bus', 'Overall dV [p.u.]', 'Line cost [€]', 'Line length [km]']])

    # Find best lines for each neighborhood
    best_lines = []
    for neighborhood in ['r0', 'r1', 'r2', 'r3']:
        best_lines.append(valid_options.loc[valid_options['Neighborhood'] == neighborhood, 'Line cost [€]'].idxmin())
    best_lines_gdf = valid_options.loc[best_lines]
    best_lines_gdf.to_file(r"results\best_lines.geojson")

    print_result = best_lines_gdf.loc[:, ['Neighborhood', 'Bus', 'Overall dV [p.u.]',
                                          'Line cost [€]', 'Line length [km]']]
    print(f"\n\n{50* '='}\nCheapest and technically valid power lines:\n{print_result}")
    return best_lines_gdf


def evaluate_connection_line(
        net: pp.pandapowerNet,
        line_idx: int,
        from_bus: int,
        line_type: str,
        geom: LineString,
        neighborhood_lines: gpd.GeoDataFrame,
        bus_geoseries: gpd.GeoSeries,
        original_sgen_p_mw: Union[pd.Series, ndarray],
        buses_memo: set[int]
):
    """
    Evaluate the technical performance of a connection line.
    1. Create a temporary line in the network model
    2. Run power flow simulation
    3. Check voltage profiles and line/transformer loading
    4. Calculate voltage rise between zero and full generation
    5. Compare results against technical criteria (voltage limits, equipment loading)

    Args:
        net: pandapower network
        line_idx: Index of the line in the neighborhood_lines GeoDataFrame
        from_bus: Source bus (PV plant)
        line_type: Type of line to use
        geom: Geometry of the line
        neighborhood_lines: GeoDataFrame containing all lines
        bus_geoseries: GeoSeries containing bus locations
        original_sgen_p_mw: Original static generator power values
        buses_memo: Set of already processed buses

    Returns:
        Dictionary with line performance metrics
    """
    line_coords = list(geom.coords)
    to_bus_in_gs = bus_geoseries.sindex.nearest(Point(line_coords[-1]))[1][0]
    to_bus = bus_geoseries.index.values[to_bus_in_gs]

    buses_memo.add(to_bus)
    length = geom.length / 1000
    # Add line to network and run power flow
    new_line = pp.create_line(net, from_bus=from_bus, to_bus=to_bus, std_type=line_type, length_km=length,
                              geodata=line_coords)
    pp.runpp(net)
    # Collect results
    all_vm = net.res_bus.loc[net.res_bus.index.isin(net.sgen.bus)].vm_pu.values
    bus_voltage = net.res_bus.at[from_bus, 'vm_pu']
    line_vdm = bus_voltage - net.res_bus.at[to_bus, 'vm_pu']
    line_loading = net.res_line.at[new_line, 'loading_percent']
    trafo_loading = net.res_trafo.at[142, 'loading_percent']
    max_line_loading = net.res_line.loading_percent.max()
    min_vm_pu = net.res_bus.vm_pu.min()
    max_vm_pu = net.res_bus.vm_pu.max()
    # Calculate voltage change
    net.sgen.p_mw = 0
    pp.runpp(net)
    sgen_buses = net.res_bus.index.isin(net.sgen.bus)
    overall_vdm_max = (all_vm - net.res_bus.loc[sgen_buses].vm_pu.values).max()
    net.sgen.p_mw = original_sgen_p_mw
    # Store line performance data
    line_result_data = dict()
    line_result_data['Bus'] = to_bus
    line_result_data['Line length [km]'] = length
    line_result_data['Voltage @ PV-Site [p.u.]'] = bus_voltage
    line_result_data['Line dV [p.u.]'] = line_vdm
    line_result_data['Overall dV [p.u.]'] = overall_vdm_max
    line_result_data['Min. V [p.u.]'] = min_vm_pu
    line_result_data['Max. V [p.u.]'] = max_vm_pu
    line_result_data['Line loading [%]'] = line_loading
    line_result_data['Max. line loading [%]'] = max_line_loading
    line_result_data['Transformer loading [%]'] = trafo_loading
    line_result_data['Line cost [€]'] = get_cost(neighborhood_lines, line_idx)
    line_result_data['Neighborhood'] = neighborhood_lines.at[line_idx, 'neighborhood']
    ed = neighborhood_lines.at[line_idx, 'euclidean_distance']
    line_result_data['Euclidean distance [km]'] = ed
    ssb = neighborhood_lines.at[line_idx, 'search_space_buffer_m']
    line_result_data['Search space buffer [m]'] = ssb
    rt = neighborhood_lines.at[line_idx, 'runtime_total']
    line_result_data['Total runtime [s]'] = rt
    rsp = neighborhood_lines.at[line_idx, 'runtime_shortest_path']
    line_result_data['Time for path finding [s]'] = rsp
    rpm = neighborhood_lines.at[line_idx, 'runtime_path_metrics']
    line_result_data['Time for path metrics [s]'] = rpm
    net.line.drop(new_line, inplace=True)
    return line_result_data


if __name__ == "__main__":
    # Choose which function to run
    url = r"https://owsproxy.lgl-bw.de/owsproxy/wfs/WFS_LGL-BW_ALKIS?version=2.0.0"
    layer = "Tatsächliche Nutzung"
    request = {
        'url': url,
        'layer': layer
    }
    result_path = r"results\all_paths.geojson"
    raster_path = r"data\raster\mv_oberrhein.tiff"
    net_crs = "epsg:4326"
    gis_data_crs = "epsg:25832"
    source = gpd.read_file(r"data\shapes\sources.geojson")

    mv_oberrhein = pp.networks.mv_oberrhein(scenario='generation',
                                            separation_by_sub=True,
                                            include_substations=True)[1]

    # Prepare raster file
    prepare_raster_data(mv_oberrhein, net_crs, gis_data_crs, request, raster_path)
    # Run route finding
    find_routes_for_connection_lines(mv_oberrhein, raster_path, result_path, source,
                                     net_crs, gis_data_crs)
    # Run analysis of routes
    analyze_mv_oberrhein(result_path, net_crs, gis_data_crs)
