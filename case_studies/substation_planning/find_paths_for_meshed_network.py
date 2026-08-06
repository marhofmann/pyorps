
from substation_planning import *


if "__main__" == __name__:
    raster_path = r"C:\Users\mhnn82\Documents\3_python_projects\infrastructure_route_planning\QGIS\Hessen\new_raster.tiff"
    source_path = r"C:\Users\mhnn82\Documents\3_python_projects\pyorps\case_studies\substation_planning\shape\sources_substation.shp"
    targets_path = r"C:\Users\mhnn82\Documents\3_python_projects\pyorps\case_studies\substation_planning\shape\targets_substations.shp"
    save_path = r"C:\Users\mhnn82\Documents\3_python_projects\pyorps\case_studies\substation_planning\results"

    sources_gdf = gpd.read_file(source_path).to_crs("EPSG:32632")
    targets_gdf = gpd.read_file(targets_path).to_crs("EPSG:32632")

    #create_meshed_network(raster_path, sources_gdf, targets_gdf, save_path)
    line_data = read_and_concat_geojsons(r"C:\Users\mhnn82\Documents\3_python_projects\pyorps\case_studies\substation_planning\shape")
    line_data.to_file(r"C:\Users\mhnn82\Documents\3_python_projects\pyorps\case_studies\substation_planning\shape"
                      r"\meshed_network_data.geojson")