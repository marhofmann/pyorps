
from os.path import join
import geopandas as gpd

from pyorps import PathFinder


def set_street_bez(gdf):
    all_streets = gdf['nutzart'] == 'Straßenverkehr'
    gdf.loc[all_streets & gdf['name'].str.contains('A '), 'bez'] = "Autobahn"
    gdf.loc[all_streets & gdf['name'].str.contains('L '), 'bez'] = "Landesstr."
    gdf.loc[all_streets & gdf['name'].str.contains('B '), 'bez'] = "Bundesstr."


def fancy_function(gdf, conditions_dict, buffer_distance):
    set_street_bez(gdf)


agrar_factor = 1.5
oberflaechennutzung_kosten = {
    ('nutzart', 'bez'): {
        # Wald - unverändert lassen wie gewünscht
        "Wald": {"Nadelholz": 365, "Laub- und Nadelholz": 402, "Laubholz": 438,
                 "": 365},

        # Straßenverkehr - gestaffelt nach Straßentyp
        "Straßenverkehr": {
            "Landesstr.": 378,
            "Bundesstr.": 400,
            "Autobahn": 430,
            "": 378
        },

        # Wege
        "Weg": {"Fußweg": 300, "Rad- und Fußweg": 300, "": 300},
        # Asphalt Gehweg vs. Verbundpflaster [[11]]

        # Landwirtschaft - Pflug nicht berücksichtigt, Mindestkosten "ohne Oberfläche"
        "Landwirtschaft": {
            "Ackerland": int(125 * agrar_factor),
            "Grünland": int(125 * agrar_factor),
            "Gartenbauland": int(166 * agrar_factor),
            # [[11]]
            "Streuobstwiese": int(166 * agrar_factor),
            # [[11]]
            "Obst- und Nussplantage": int(166 * agrar_factor),
            # Oberfläche [[11]]
            "Streuobstacker": int(125 * agrar_factor),
            "Baumschule": int(166 * agrar_factor),
            "Brachland": int(125 * agrar_factor),
            "": int(125 * agrar_factor)
        },

        # Gewässer - gestaffelt: Graben < Bach < Kanal < Fluss
        "Fließgewässer": {
            "Graben": 125,
            "Bach": 211,
            "Kanal": 295,
            "Fluss": 590,
            "": 295
        },

        "Stehendes Gewässer": {
            "Teich": 295,
            "Speicherbecken": 590,
            "Stausee": 590,
            "Baggersee": 590,
            "": 295
        },

        # Weitere Kategorien - Mindestkosten 125€
        "Sport-, Freizeit- und Erholungsfläche": {"Grünanlage": 125, "": 125},
        # Ohne Oberfläche [[11]]
        "Gehölz": {"": 180},  # Bodenklasse 7 ohne Oberfläche [[11]]

        "Platz": {"Parkplatz": 300, "Rastplatz": 300, "": 211},

        "Flugverkehr": {"Segelfluggelände": 125, "Sonderlandeplatz": 125, "": 125},
        # Ohne Oberfläche [[11]]
        "Bahnverkehr": {"": 480},  # Spülbohrung für Kreuzung [[11]]
        "Heide": {"": 125},  # Ohne Oberfläche [[11]]
        "Unland/Vegetationslose Fläche": {"": 125},  # Ohne Oberfläche [[11]]

        # Verbotene Flächen - unverändert
        "Fläche gemischter Nutzung": {"": 65535},
        "Fläche besonderer funktionaler Prägung": {"": 65535},
        "Wohnbaufläche": {"": 65535},
        "Sumpf": {"": 65535},
        "Industrie- und Gewerbefläche": {"": 65535},
        "Tagebau, Grube, Steinbruch": {"": 65535},
        "Friedhof": {"": 65535},
        "Moor": {"": 65535},
        "Halde": {"": 65535},
        "Schiffsverkehr": {"": 65535}
    }
}

soil_classes_factors = {
    "AUSGANGSGE": {
        # Bodenklasse 5 - Faktor 1.0 (Basis)
        "Lösslehm, Löss": 1.0,
        "vorwiegend Lösslehm mit Gesteinsbeimengungen": 1.0,
        "Schluff- und Tonsteine, Sandsteine": 1.0,
        "Ton- und Schluffsteine und Arkosen, örtl. carbonathaltig": 1.0,
        "Löss": 1.0,
        "Lösslehm über dichtem Untergrund": 1.0,
        "Terrassensand und -kies": 1.0,
        "Dünensand, Terrassensand und -kies": 1.0,
        "carbonathaltiger Hochflutlehm": 1.0,
        "Verschiedene Torfarten": 1.0,
        "Auenlehm": 1.0,
        "Trachytische Aschen": 1.0,
        "Lösslehm, örtl. mit Gesteinsbeimengungen": 1.0,
        "Lösslehm mit Gesteinsbeimengungen": 1.0,
        "carbonathaltiger Dünensand": 1.0,

        # Bodenklasse 7 - Faktor 1.3 (basierend auf Kostenverhältnis 166€/125€ = 1.33)
        "Gabbro, Diorit, Amphibolit, Melaphyr, Basalt": 1.3,
        "Sandsteine": 1.3,
        "Grauwacken, Sandsteine, Konglomerate, Quarzite, Kieselschiefer": 1.3,
        "Kalkstein, Mergel, Dolomit": 1.3,
        "Tonschiefer, Grauwackenschiefer, Phyllit": 1.3,
        "Schalstein, Diabas": 1.3,
        "Kalkstein, Mergel, Dolomit, Ton- und Schluffsteine und Arkosen": 1.3,
        "Quarzite, Sandsteine": 1.3,
        "Granodiorit, Quarzporphyr, Glimmer- und Quarzitschiefer, Gneis": 1.3,
        "Basalt, Basalttuff": 1.3,
        "Basalt, Lösslehm, Löss": 1.3,
        "Basalt, Lösslehm": 1.3,
        "Lösslehm, Basalt": 1.3,
        "Ton- und Schluffsteine, Arkosen, Kalkstein, Mergel, Dolomit": 1.3
    }
}


# Define a nested dictionary for specific cost assumptions for the
# water protection zones
water_protection_cost_assumptions = {
    "ZONE": {
        # Innermost protection zone (effectively forbidden)
        'Schutzzone I': 100,
        # High protection zone
        'Schutzzone II': 1.1,
        # Qualitative protection zone I (effectively forbidden)
        'Qualitative Schutzzone I': 100,
        # Qualitative protection zone II
        'Qualitative Schutzzone II': 1.1,
        # Quantitative protection zone A (effectively forbidden)
        'Quantitative Schutzzone A': 100,
        # Quantitative protection zone B
        'Quantitative Schutzzone B': 1.1,
        # Default value for unspecified zones
        '': 1.05
    }
}


# WFS-Request for base file
base_file_url = "https://www.gds.hessen.de/wfs2/aaa-suite/cgi-bin/alkis/vereinf/wfs"
base_file = {
    "url": base_file_url,
    "layer": "ave_Nutzung",
}

local_directory = r"data/shapes"

mask_path = join(local_directory, r"praeferenzraum\masked.shp")

soil_classes = join(local_directory,
                    r"additional_data\Bodeneinheiten_Bodenuebersicht_500000.shp",)
nature_reserves = join(local_directory,
                       r"additional_data\Natura2000_end2021_rev1_epsg3035.shp")
drinking_water_protection = join(local_directory, "additional_data/TWS_HQS_TK25.shp")
wind_energy_area = join(local_directory, r"additional_data\RTW_WINDENERGIE16_F.shp")

source_path = join(local_directory, r"source_and_target\sources_HE.shp")
target_path = join(local_directory, r"source_and_target\targets_HE.shp")

source = gpd.read_file(source_path).loc[[0]]
target = gpd.read_file(target_path).loc[[5]]

mask = gpd.read_file(mask_path)
buffers = [0, 20, 40, 60, 80, 100, 125]
for geometry_buffer_m in buffers:
    # Create a list with dictionaries for the datasets which should be used to modify
    # the base data
    datasets_to_modify = [
        {
            "input_data": soil_classes,
            "cost_assumptions": soil_classes_factors,
            "multiply": True
        },
        {
            "input_data": nature_reserves,
            "cost_assumptions": 1.25,
            "geometry_buffer_m": geometry_buffer_m,
            "multiply": True,
        },
        {
            "input_data": drinking_water_protection,
            "cost_assumptions": water_protection_cost_assumptions,
            "geometry_buffer_m": geometry_buffer_m,
            "multiply": True,
        },
        {
            "input_data": wind_energy_area,
            "cost_assumptions": 1.25,
            "geometry_buffer_m": geometry_buffer_m,
            "multiply": True,
        },
    ]
    raster_path = join("data/raster", f"RML_buffer_{geometry_buffer_m}_m.tiff")
    path_finder = PathFinder(
        source_coords=source,
        target_coords=target,
        dataset_source=base_file,
        graph_api='cython',
        ignore_max_cost=True,
        datasets_to_modify=datasets_to_modify,
        cost_assumptions=oberflaechennutzung_kosten,
        search_space_buffer_m=25_000,
        mask=mask,
        raster_save_path=raster_path,
        fancy_function=fancy_function,
        #fancy_function_kwargs=fancy_function_kwargs
    )
    #path_finder.find_route()
    #path_finder.save_paths(f"RML_buffer_{geometry_buffer_m}_m.geojson")
    print("Trassenplanung abgeschlossen:")
    #print(path_finder.paths)
    print("\n\n")
    break
