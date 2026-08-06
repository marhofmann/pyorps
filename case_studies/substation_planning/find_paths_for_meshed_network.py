
from substation_planning import *


if "__main__" == __name__:
    raster_path = r"<local-path>"
    source_path = r"<local-path>"
    targets_path = r"<local-path>"
    save_path = r"<local-path>"

    sources_gdf = gpd.read_file(source_path).to_crs("EPSG:32632")
    targets_gdf = gpd.read_file(targets_path).to_crs("EPSG:32632")

    #create_meshed_network(raster_path, sources_gdf, targets_gdf, save_path)
    line_data = read_and_concat_geojsons(r"<local-path>")
    line_data.to_file(r"<local-path>"
                      r"\meshed_network_data.geojson")