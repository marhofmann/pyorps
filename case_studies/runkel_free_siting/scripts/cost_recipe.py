"""Land-use cost recipe of the CIRED 2026 study, as data (EUR per metre of trench; 65535 = forbidden).

Copied from ``TopoMILP/data/cired2026/pyorps_scripts/create_raster_for_windfarm_layout.py`` (the
``mod1`` version, with the road classes priced by their buffers). These are the stand-in class costs
the owner chose to keep ("keep all stand-ins and say so openly", 2026-10-05); they are NOT sourced
prices of this study. The recipe is data here so the build script, the provenance record and the
tests all read the same numbers.
"""
from __future__ import annotations

FORBIDDEN = 65535

#: (nutzart, bez) -> cost; "" is the class default when the sub-class is empty or unlisted
LAND_USE_COSTS: dict[str, dict[str, int]] = {
    "Wald": {"Nadelholz": 405, "Laub- und Nadelholz": 425, "Laubholz": 475, "": 405},
    "Straßenverkehr": {"Landstr.": 450, "Bundesstr.": 500, "Autobahn": 750, "": 340},
    "Weg": {"Fußweg": 220, "Rad- und Fußweg": 220, "": 220},
    "Landwirtschaft": {"Ackerland": 437, "Grünland": 437, "Gartenbauland": 581,
                       "Streuobstwiese": 581, "Obst- und Nussplantage": 581,
                       "Streuobstacker": 468, "Baumschule": 581, "Brachland": 380, "": 437},
    "Fließgewässer": {"Graben": 332, "Bach": 346, "Kanal": 358, "Fluss": 586, "": 332},
    "Stehendes Gewässer": {"Teich": 350, "Speicherbecken": 590, "Stausee": 590,
                           "Baggersee": 590, "": 400},
    "Sport-, Freizeit- und Erholungsfläche": {"Grünanlage": 320, "": FORBIDDEN},
    "Gehölz": {"": 380},
    "Platz": {"Parkplatz": 310, "Rastplatz": 320, "": 310},
    "Flugverkehr": {"Segelfluggelände": 200, "Sonderlandeplatz": 300, "": FORBIDDEN},
    "Bahnverkehr": {"": 800},
    "Heide": {"": 125},
    "Unland/Vegetationslose Fläche": {"": 125},
    "Fläche gemischter Nutzung": {"": FORBIDDEN},
    "Fläche besonderer funktionaler Prägung": {"": FORBIDDEN},
    "Wohnbaufläche": {"": FORBIDDEN},
    "Sumpf": {"": FORBIDDEN},
    "Industrie- und Gewerbefläche": {"": FORBIDDEN},
    "Tagebau, Grube, Steinbruch": {"": FORBIDDEN},
    "Friedhof": {"": FORBIDDEN},
    "Moor": {"": FORBIDDEN},
    "Halde": {"": FORBIDDEN},
    "Schiffsverkehr": {"": FORBIDDEN},
}

#: road name prefix -> (class label, buffer in metres) as in the CIRED 2026 script
ROAD_BUFFERS = (("A ", "Autobahn", 10), ("B ", "Bundesstr.", 4), ("L ", "Landstr.", 2))

#: water protection zone text -> code stored in the protection flag raster (bits 8-11)
WATER_ZONE_CODES = {"Schutzzone I": 1, "Schutzzone II": 2, "Schutzzone III": 3,
                    "Schutzzone IIIA": 4, "Schutzzone IIIB": 5}

#: flag bits of the protection raster (uint16)
FLAG_BITS = {"nature_reserves": 0, "landscape_protection": 1, "natura2000_habitats": 2,
             "bird_protection_areas": 3, "biotopes_protected": 4,
             "biotope_complexes_protected": 5}
WATER_ZONE_SHIFT = 8        # bits 8..11 hold the water-zone code (0 = none, 1 = most restrictive)


def street_classes(gdf) -> None:
    """Classify main roads by name prefix and buffer them (in place), as CIRED 2026 did."""
    streets = gdf["nutzart"] == "Straßenverkehr"
    for prefix, label, buf in ROAD_BUFFERS:
        sel = streets & gdf["name"].str.contains(prefix, na=False)
        gdf.loc[sel, "bez"] = label
        gdf.loc[sel, "geometry"] = gdf.loc[sel, "geometry"].buffer(buf)


def class_cost(nutzart, bez) -> int:
    """The recipe's cost for one land-use row; a class the recipe does not list is forbidden."""
    table = LAND_USE_COSTS.get(nutzart)
    if table is None:
        return FORBIDDEN
    key = bez if isinstance(bez, str) and bez in table else ""
    return table.get(key, table.get("", FORBIDDEN))


def forbidden_mask(gdf):
    """Boolean Series: rows of an ALKIS frame whose recipe cost is forbidden. Road rows are never
    forbidden, so the road buffers do not matter here. Independent of the pyorps rasterizer."""
    import pandas as pd

    return pd.Series([class_cost(n, b) == FORBIDDEN for n, b in zip(gdf["nutzart"], gdf["bez"])],
                     index=gdf.index)
