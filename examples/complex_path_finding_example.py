#%%
from pyorps import RasterGraph
#%%
cost_file_path = r"data\cost_assumptions\cost_assumptions.csv"
base_file_url = "https://www.gds.hessen.de/wfs2/aaa-suite/cgi-bin/alkis/vereinf/wfs"
geometry_buffer_m = 20
#%%
water_protection_cost_assumptions = {
    "ZONE": {
        'Schutzzone I': 65535,
        'Schutzzone II': 5000,
        'Qualitative Schutzzone I': 65535,
        'Qualitative Schutzzone II': 5000,
        'Quantitative Schutzzone A': 65535,
        'Quantitative Schutzzone B': 2000,
        '': 1200

    }
}
#%%
datasets_to_modify = [
    {
        "input_data": "data/shapes/additional_data/Natura2000_end2021_rev1.gpkg",
        "cost_assumptions": 1200,
        "geometry_buffer_m": geometry_buffer_m,
        "layer": "NaturaSite_polygon"
    },
    {
        "input_data": "data/shapes/additional_data/ffh.shp",
        "cost_assumptions": 1200,
        "geometry_buffer_m": geometry_buffer_m,
    },
    {
        "input_data": "data/shapes/additional_data/vsg.shp",
        "cost_assumptions": 1.1,
        "geometry_buffer_m": geometry_buffer_m,
        "multiply": True
    },
    {
        "input_data": "data/shapes/additional_data/LSG_Hessen_UTM.shp",
        "cost_assumptions": 1.1,
        "geometry_buffer_m": geometry_buffer_m,
        "multiply": True
    },
    {
        "input_data": "data/shapes/additional_data/TWS_HQS_TK25.shp",
        "cost_assumptions": water_protection_cost_assumptions,
        "geometry_buffer_m": geometry_buffer_m,
    },
]

#%%
box_tuple = (471228, 5592632, 474176, 5594879)
#%%
# Example usage
base_file = {
    "url": base_file_url,
    "layer": "ave_Nutzung",
}
#%%
delta = 200
source_coords = (box_tuple[0] + delta, box_tuple[1] + delta)
target_coords = [(box_tuple[2] - delta, box_tuple[3] - delta),
                 (box_tuple[2] - delta, box_tuple[3] - 10 * delta)]

#%%
raster_graph = RasterGraph(
    dataset_source=base_file,
    source_coords=source_coords,
    target_coords=target_coords,
    search_space_buffer_m=2000,  # Use search-specific buffer
    neighborhood_str="r2",
    graph_api='networkit',
    cost_assumptions=cost_file_path,
    bbox=box_tuple,
    datasets_to_modify=datasets_to_modify,
    geometry_buffer_m=geometry_buffer_m
)
#%%
route_result = raster_graph.find_route()
#%%
print(route_result)
#%%
raster_graph.plot_paths()
#%%
raster_graph.geo_rasterizer.save_raster(r'data\results\example_raster_with_complex_cost_assumptions.tiff')
path_gdf = raster_graph.create_path_geodataframe(save_path=r'data\results\example_multiple_paths.geojson')
