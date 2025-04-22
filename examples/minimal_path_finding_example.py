from pyorps import RasterGraph

cost_file_path = r"data\cost_assumptions\cost_assumptions.csv"
base_file_url = "https://www.gds.hessen.de/wfs2/aaa-suite/cgi-bin/alkis/vereinf/wfs"
box_tuple = (471228, 5592632, 474176, 5594879)

# Example usage
base_file = {
    "url": base_file_url,
    "layer": "ave_Nutzung",
}

delta = 800
source_coords = (box_tuple[0] + delta, box_tuple[1] + delta)
target_coords = (box_tuple[2] - delta, box_tuple[3] - delta)

raster_graph = RasterGraph(
    dataset_source=base_file,
    source_coords=source_coords,
    target_coords=target_coords,
    search_space_buffer_m=500,
    neighborhood_str="r2",
    graph_api='networkit',
    cost_assumptions=cost_file_path,
    bbox=box_tuple,
)
route_result = raster_graph.find_route()
raster_graph.plot_paths()
route_result
