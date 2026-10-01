"""
PYORPS GUI: presets — known WFS servers and notebook cost-model defaults.

All values are lifted verbatim from
``examples/prepare_data_for_distribution_grid_planning.ipynb`` (Appendix B of
the GUI design plan): the base ALKIS land-use costs keyed by
``('nutzart', 'bez')`` (including the post-refinement street classes of cell
41), the water-protection / soil / landscape / nature modifiers, and the
street-buffer preprocessor.
"""
from __future__ import annotations

import copy
from typing import Callable

#: uint16 exclusion sentinel — "forbidden", first-class in the GUI (F11).
FORBIDDEN = 65535

DEFAULT_PROJECT_CRS = "EPSG:25832"

# ---------------------------------------------------- coverage bounding boxes
#: approximate WGS84 [min_lon, min_lat, max_lon, max_lat] per German state, used
#: to filter the dataset dropdown to services that actually cover the current
#: map view (viewport filtering). Whole-Germany services use ``DE``.
STATE_BBOX: dict[str, list[float]] = {
    "DE": [5.8, 47.2, 15.1, 55.1],       # whole Germany (nationwide services)
    "BW": [7.5, 47.5, 10.5, 49.8],
    "BY": [8.9, 47.2, 13.9, 50.6],
    "BE": [13.0, 52.3, 13.8, 52.7],
    "BB": [11.2, 51.3, 14.8, 53.6],
    "HB": [8.4, 53.0, 9.0, 53.6],
    "HH": [9.7, 53.3, 10.4, 53.8],
    "HE": [7.7, 49.3, 10.3, 51.7],
    "MV": [10.5, 53.1, 14.5, 54.7],
    "NI": [6.5, 51.2, 11.7, 53.9],
    "NW": [5.8, 50.3, 9.5, 52.6],
    "RP": [6.1, 48.9, 8.6, 50.95],
    "SL": [6.3, 49.1, 7.5, 49.7],
    "SN": [11.8, 50.1, 15.1, 51.7],
    "ST": [10.5, 50.9, 13.3, 53.1],
    "SH": [7.8, 53.3, 11.4, 55.1],
    "TH": [9.8, 50.2, 12.7, 51.7],
}

STATE_NAME = {
    "DE": "Deutschland", "BW": "Baden-Württemberg", "BY": "Bayern",
    "BE": "Berlin", "BB": "Brandenburg", "HB": "Bremen", "HH": "Hamburg",
    "HE": "Hessen", "MV": "Mecklenburg-Vorpommern", "NI": "Niedersachsen",
    "NW": "Nordrhein-Westfalen", "RP": "Rheinland-Pfalz", "SL": "Saarland",
    "SN": "Sachsen", "ST": "Sachsen-Anhalt", "SH": "Schleswig-Holstein",
    "TH": "Thüringen",
}

# --------------------------------------------------------------- map services
# Each entry: name, url, layer (WFS feature type / WMS layer; "" = discover via
# GetCapabilities), service ("wfs"|"wms"|"wcs"), category, state (STATE_BBOX
# key), attribution. All are free/open (dl-de/by-2-0 or state open-data licence,
# June-2024 Geobasis "Open Data" release). Land-use layer names vary per state —
# where unsure the layer is "" so the "List layers" button discovers it.
DE_ATTR = "© GeoBasis-DE / BKG (dl-de/by-2-0)"

MAP_SERVICES: list[dict] = [
    # ---- nationwide (BKG open data) ------------------------------------
    {"name": "BKG Digitales Geländemodell DGM200 (DEM, terrain)",
     "url": "https://sgx.geodatenzentrum.de/wms_dgm200", "layer": "relief",
     "service": "wms", "category": "Terrain (DEM/DGM)", "state": "DE",
     "attribution": DE_ATTR},
    {"name": "BKG DGM200 elevation coverage (WCS → DEM raster)",
     "url": "https://sgx.geodatenzentrum.de/wcs_dgm200_inspire",
     "layer": "dgm200_inspire__EL.GridCoverage", "service": "wcs",
     "category": "Terrain (DEM/DGM)", "state": "DE", "attribution": DE_ATTR,
     "crs": "EPSG:25832", "axis": ["E", "N"], "res": 200},
    # ---- high-resolution state DEM (DGM, terrain) / DSM (DOM, surface) ----
    # All verified WCS 2.0.1 endpoints (GetCapabilities/DescribeCoverage;
    # HE + NW also GetCoverage-smoke-tested). "res" = metres per pixel,
    # "max_px" = documented per-request pixel limit (requests are tiled to
    # the study area, so bigger areas need a smaller area or the overview).
    {"name": "Hessen DGM1 elevation 1 m (WCS → DEM raster)",
     "url": "https://inspire-hessen.de/raster/dgm1/ows", "layer": "he_dgm1",
     "service": "wcs", "category": "Terrain (DEM/DGM)", "state": "HE",
     "attribution": "© HVBG (dl-de/zero-2-0)",
     "crs": "EPSG:25832", "axis": ["E", "N"], "res": 1},
    {"name": "Hessen DOM1 surface model 1 m (WCS → DSM raster)",
     "url": "https://inspire-hessen.de/raster/dom1/ows", "layer": "dom1",
     "service": "wcs", "category": "Surface (DSM/DOM)", "state": "HE",
     "attribution": "© HVBG (dl-de/zero-2-0)",
     "crs": "EPSG:25832", "axis": ["E", "N"], "res": 1},
    {"name": "NRW DGM elevation 1 m (WCS → DEM raster)",
     "url": "https://www.wcs.nrw.de/geobasis/wcs_nw_dgm", "layer": "nw_dgm",
     "service": "wcs", "category": "Terrain (DEM/DGM)", "state": "NW",
     "attribution": "© GeoBasis NRW (dl-de/zero-2-0)",
     "crs": "EPSG:25832", "axis": ["x", "y"], "res": 1,
     "max_px": 2000 * 2000},
    {"name": "NRW DOM surface model 1 m (WCS → DSM raster)",
     "url": "https://www.wcs.nrw.de/geobasis/wcs_nw_dom", "layer": "nw_dom",
     "service": "wcs", "category": "Surface (DSM/DOM)", "state": "NW",
     "attribution": "© GeoBasis NRW (dl-de/zero-2-0)",
     "crs": "EPSG:25832", "axis": ["x", "y"], "res": 1,
     "max_px": 2000 * 2000},
    {"name": "Brandenburg/Berlin DGM elevation 1 m (WCS → DEM raster)",
     "url": "https://isk.geobasis-bb.de/ows/dgm_wcs", "layer": "bb_dgm",
     "service": "wcs", "category": "Terrain (DEM/DGM)", "state": "BB",
     "attribution": "© GeoBasis-DE/LGB (dl-de/by-2-0)",
     "crs": "EPSG:25833", "axis": ["x", "y"], "res": 1},
    {"name": "Brandenburg/Berlin bDOM surface model 1 m (WCS → DSM raster)",
     "url": "https://isk.geobasis-bb.de/ows/bdom_wcs", "layer": "bb_bdom",
     "service": "wcs", "category": "Surface (DSM/DOM)", "state": "BB",
     "attribution": "© GeoBasis-DE/LGB (dl-de/by-2-0)",
     "crs": "EPSG:25833", "axis": ["x", "y"], "res": 1},
    {"name": "Sachsen-Anhalt DGM1 elevation 1 m (WCS → DEM raster)",
     "url": "https://www.geodatenportal.sachsen-anhalt.de/wss/service/"
            "ST_LVermGeo_DGM1_WCS_OpenData/guest", "layer": "Coverage1",
     "service": "wcs", "category": "Terrain (DEM/DGM)", "state": "ST",
     "attribution": "© GeoBasis-DE / LVermGeo ST (dl-de/by-2-0)",
     "crs": "EPSG:25832", "axis": ["x", "y"], "res": 1},
    {"name": "Baden-Württemberg DGM1 elevation 1 m (WCS → DEM raster)",
     "url": "https://owsproxy.lgl-bw.de/owsproxy/wcs/"
            "WCS_INSP_BW_Hoehe_Coverage_DGM1",
     "layer": "EL.ElevationGridCoverage", "service": "wcs",
     "category": "Terrain (DEM/DGM)", "state": "BW",
     "attribution": "© LGL-BW (dl-de/by-2-0)",
     "crs": "EPSG:25832", "axis": ["E", "N"], "res": 1},
    {"name": "BKG TopPlusOpen (topographic basemap)",
     "url": "https://sgx.geodatenzentrum.de/wms_topplus_open",
     "layer": "web", "service": "wms", "category": "Topographic", "state": "DE",
     "attribution": DE_ATTR},
    {"name": "BKG Digitale Topographische Karte DTK250",
     "url": "https://sgx.geodatenzentrum.de/wms_dtk250", "layer": "dtk250",
     "service": "wms", "category": "Topographic", "state": "DE",
     "attribution": DE_ATTR},
    # ---- ALKIS "tatsächliche Nutzung" (land use) per state -------------
    {"name": "Baden-Württemberg ALKIS Tatsächliche Nutzung",
     "url": "https://owsproxy.lgl-bw.de/owsproxy/wfs/WFS_LGL-BW_ALKIS"
            "?version=2.0.0", "layer": "nora:v_al_tatsaechliche_nutzung",
     "service": "wfs", "category": "ALKIS land use", "state": "BW",
     "attribution": "© LGL-BW (dl-de/by-2-0)"},
    {"name": "Bayern ALKIS Tatsächliche Nutzung (ave)",
     "url": "https://geoservices.bayern.de/wfs/v1/ogc_alkis_ave.cgi",
     "layer": "ave:Nutzung", "service": "wfs", "category": "ALKIS land use",
     "state": "BY", "attribution": "© LDBV Bayern (dl-de/by-2-0)"},
    {"name": "Berlin ALKIS Nutzung",
     "url": "https://fbinter.stadt-berlin.de/fb/wfs/data/senstadt/s_wfs_alkis",
     "layer": "", "service": "wfs", "category": "ALKIS land use", "state": "BE",
     "attribution": "© Geoportal Berlin (dl-de/by-2-0)"},
    {"name": "Brandenburg ALKIS (vereinfacht)",
     "url": "https://isk.geobasis-bb.de/ows/alkis_wfs", "layer": "",
     "service": "wfs", "category": "ALKIS land use", "state": "BB",
     "attribution": "© GeoBasis-DE/LGB (dl-de/by-2-0)"},
    {"name": "Bremen ALKIS (LGLN)",
     "url": "https://opendata.lgln.niedersachsen.de/doorman/noauth/"
            "alkishb_wfs_nas", "layer": "", "service": "wfs",
     "category": "ALKIS land use", "state": "HB",
     "attribution": "© LGLN (dl-de/by-2-0)"},
    {"name": "Hamburg ALKIS Flurstücke/Nutzung",
     "url": "https://geodienste.hamburg.de/HH_WFS_INSPIRE_Flurstuecke",
     "layer": "", "service": "wfs", "category": "ALKIS land use", "state": "HH",
     "attribution": "© LGV Hamburg (dl-de/by-2-0)"},
    {"name": "Hessen ALKIS Nutzung (ave:Nutzung)",
     "url": "https://www.gds.hessen.de/wfs2/aaa-suite/cgi-bin/alkis/vereinf/wfs",
     "layer": "ave:Nutzung", "service": "wfs", "category": "ALKIS land use",
     "state": "HE", "attribution": "© HVBG (dl-de/by-2-0)"},
    {"name": "Mecklenburg-Vorpommern ALKIS (vereinfacht)",
     "url": "https://www.geodaten-mv.de/dienste/alkis_wfs_sf", "layer": "",
     "service": "wfs", "category": "ALKIS land use", "state": "MV",
     "attribution": "© GeoBasis-DE/M-V (dl-de/by-2-0)"},
    {"name": "Niedersachsen ALKIS (vereinfacht)",
     "url": "https://opendata.lgln.niedersachsen.de/doorman/noauth/"
            "alkis_wfs_sf", "layer": "", "service": "wfs",
     "category": "ALKIS land use", "state": "NI",
     "attribution": "© LGLN (dl-de/by-2-0)"},
    {"name": "Nordrhein-Westfalen ALKIS Nutzung (vereinfacht)",
     "url": "https://www.wfs.nrw.de/geobasis/wfs_nw_alkis_vereinfacht",
     "layer": "ave:Nutzung", "service": "wfs", "category": "ALKIS land use",
     "state": "NW", "attribution": "© GeoBasis-DE/BezReg Köln (dl-de/by-2-0)"},
    {"name": "Rheinland-Pfalz ALKIS Nutzung",
     "url": "https://geo5.service24.rlp.de/wfs/alkis_rp.fcgi",
     "layer": "ave:Nutzung", "service": "wfs", "category": "ALKIS land use",
     "state": "RP", "attribution": "© LVermGeo RLP (dl-de/by-2-0)"},
    {"name": "Saarland ALKIS Nutzung",
     "url": "https://geoportal.saarland.de/registry/wfs/325", "layer": "",
     "service": "wfs", "category": "ALKIS land use", "state": "SL",
     "attribution": "© LVGL Saarland (dl-de/by-2-0)"},
    {"name": "Sachsen ALKIS Nutzung (vereinfacht)",
     "url": "https://geodienste.sachsen.de/aaa/public_alkis/vereinf/wfs",
     "layer": "ave:Nutzung", "service": "wfs", "category": "ALKIS land use",
     "state": "SN", "attribution": "© GeoSN (dl-de/by-2-0)"},
    {"name": "Sachsen-Anhalt ALKIS Nutzung (OpenData)",
     "url": "https://www.geodatenportal.sachsen-anhalt.de/wss/service/"
            "ST_LVermGeo_ALKIS_WFS_OpenData/wfs", "layer": "", "service": "wfs",
     "category": "ALKIS land use", "state": "ST",
     "attribution": "© LVermGeo LSA (dl-de/by-2-0)"},
    {"name": "Schleswig-Holstein ALKIS Nutzung (vereinfacht)",
     "url": "https://service.gdi-sh.de/WFS_SH_ALKIS_vereinf_OpenGBD",
     "layer": "ave:Nutzung", "service": "wfs", "category": "ALKIS land use",
     "state": "SH", "attribution": "© GDI-SH (dl-de/by-2-0)"},
    {"name": "Thüringen ALKIS Nutzung (adv)",
     "url": "https://www.geoproxy.geoportal-th.de/geoproxy/services/"
            "adv_alkis_wfs", "layer": "ave:Nutzung", "service": "wfs",
     "category": "ALKIS land use", "state": "TH",
     "attribution": "© TLBG (dl-de/by-2-0)"},
    # ---- Hessen thematic layers (used by the notebook cost model) ------
    {"name": "Hessen Trinkwasserschutz (TWS_HQS_TK25)",
     "url": "https://geodienste-umwelt.hessen.de/arcgis/services/inspire/"
            "bewirtschaftungsgebiete/MapServer/WFSServer",
     "layer": "TWS_HQS_TK25", "service": "wfs",
     "category": "Water protection", "state": "HE",
     "attribution": "© HLNUG"},
    {"name": "Hessen Boden (Bodenübersicht 500k)",
     "url": "https://geodienste-umwelt.hessen.de/arcgis/services/inspire/"
            "boden/MapServer/WFSServer",
     "layer": "Bodeneinheiten_Bodenuebersicht_500000", "service": "wfs",
     "category": "Soil", "state": "HE", "attribution": "© HLNUG"},
    {"name": "Hessen Naturschutzgebiete",
     "url": "https://geodienste-umwelt.hessen.de/arcgis/services/inspire/"
            "schutzgebiete/MapServer/WFSServer", "layer": "Naturschutzgebiete",
     "service": "wfs", "category": "Nature/Landscape", "state": "HE",
     "attribution": "© HLNUG"},
    {"name": "Hessen Landschaftsschutzgebiete",
     "url": "https://geodienste-umwelt.hessen.de/arcgis/services/inspire/"
            "schutzgebiete/MapServer/WFSServer",
     "layer": "Landschaftsschutzgebiete", "service": "wfs",
     "category": "Nature/Landscape", "state": "HE", "attribution": "© HLNUG"},
]


def _bbox_intersects(a: list[float], b: list[float]) -> bool:
    """True if two [min_lon, min_lat, max_lon, max_lat] boxes overlap."""
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def services_in_view(view_bbox: list[float] | None = None,
                     services: list[dict] | None = None) -> list[dict]:
    """Every map service whose coverage OVERLAPS ``view_bbox`` (WGS84).

    A service qualifies when its coverage bbox intersects the current view —
    i.e. at least part of its data lives in the window on screen ("at least a
    single tile in the view"). Nationwide (``state='DE'``) services always
    qualify. ``view_bbox=None`` (map not moved yet) returns everything.
    """
    services = MAP_SERVICES if services is None else services
    if view_bbox is None:
        return list(services)
    result = []
    for svc in services:
        bbox = STATE_BBOX.get(svc.get("state", "DE"), STATE_BBOX["DE"])
        if svc.get("state") == "DE" or _bbox_intersects(bbox, view_bbox):
            result.append(svc)
    return result


def find_service(url: str, layer: str | None = None) -> dict | None:
    """The MAP_SERVICES entry matching ``url`` (and ``layer`` if given)."""
    for svc in MAP_SERVICES:
        if svc["url"] == url and (layer is None or svc.get("layer") == layer):
            return svc
    for svc in MAP_SERVICES:              # fall back to url-only match
        if svc["url"] == url:
            return svc
    return None


#: default nationwide DEM (BKG DGM200 WCS) for the "Load DEM for area" button.
#: The WCS coverage is served in EPSG:25832 with E/N axes — the GetCoverage
#: SUBSET must use those, not the map's Web-Mercator bounds.
DEFAULT_DEM = {"url": "https://sgx.geodatenzentrum.de/wcs_dgm200_inspire",
               "coverage": "dgm200_inspire__EL.GridCoverage",
               "crs": "EPSG:25832", "axis": ["E", "N"]}

#: WFS-only view of MAP_SERVICES (the editable WFS preset dropdown). Kept as a
#: name for backward compatibility with older code/tests.
WFS_PRESETS: list[dict] = [s for s in MAP_SERVICES if s["service"] == "wfs"]

# --------------------------------------------- base land-use costs (cell 24+41)
LAND_USE_COSTS: dict = {
    ("nutzart", "bez"): {
        "Wald": {
            "Nadelholz": 405, "Laub- und Nadelholz": 425,
            "Laubholz": 475, "": 405,
        },
        # cell 41: refined street classes (after set_street_type_and_buffer)
        "Straßenverkehr": {
            "Landesstr.": 450, "Bundesstr.": 500, "Autobahn": 750, "": 340,
        },
        "Weg": {"Fußweg": 300, "Rad- und Fußweg": 300, "": 300},
        "Landwirtschaft": {
            "Ackerland": 437, "Grünland": 437, "Gartenbauland": 581,
            "Streuobstwiese": 581, "Obst- und Nussplantage": 581,
            "Streuobstacker": 468, "Baumschule": 581, "Brachland": 380,
            "": 437,
        },
        "Fließgewässer": {
            "Graben": 332, "Bach": 346, "Kanal": 358, "Fluss": 586, "": 332,
        },
        "Stehendes Gewässer": {
            "Teich": 350, "Speicherbecken": 590, "Stausee": 590,
            "Baggersee": 590, "": 400,
        },
        "Sport-, Freizeit- und Erholungsfläche": {
            "Grünanlage": 320, "": FORBIDDEN,
        },
        "Gehölz": {"": 380},
        "Platz": {"Parkplatz": 310, "Rastplatz": 320, "": 310},
        "Flugverkehr": {
            "Segelfluggelände": 200, "Sonderlandeplatz": 300, "": FORBIDDEN,
        },
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
}


def default_land_use_costs() -> dict:
    """A deep copy safe to hand to the editable cost grid."""
    return copy.deepcopy(LAND_USE_COSTS)


# ---------------------------------------------------------- modifier presets
#: water protection multipliers by zone (cell 62) — multiply.
WATER_PROTECTION_MULTIPLIERS = {1: 100, 2: 2, 3: 1.5, 4: 1.2}

#: soil condition factors by AUSGANGSGESTEIN, DIN 18300 (cell 84) — multiply.
SOIL_CONDITION_FACTORS = {
    "AUSGANGSGESTEIN": {
        # class 1-3 — easy excavation
        "Lösslehm, Löss": 1.0,
        "vorwiegend Lösslehm mit Gesteinsbeimengungen": 1.0,
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
        # class 4-5 — cohesive soils
        "Schluff- und Tonsteine, Sandsteine": 1.05,
        "Ton- und Schluffsteine und Arkosen, örtl. carbonathaltig": 1.05,
        # class 6 — soft rock
        "Sandsteine": 1.15,
        "Grauwacken, Sandsteine, Konglomerate, Quarzite, Kieselschiefer": 1.15,
        "Kalkstein, Mergel, Dolomit": 1.15,
        "Tonschiefer, Grauwackenschiefer, Phyllit": 1.15,
        "Schalstein, Diabas": 1.15,
        "Kalkstein, Mergel, Dolomit, Ton- und Schluffsteine und Arkosen": 1.15,
        "Quarzite, Sandsteine": 1.15,
        "Ton- und Schluffsteine, Arkosen, Kalkstein, Mergel, Dolomit": 1.15,
        # class 7 — hard rock
        "Gabbro, Diorit, Amphibolit, Melaphyr, Basalt": 1.3,
        "Granodiorit, Quarzporphyr, Glimmer- und Quarzitschiefer, Gneis": 1.3,
        "Basalt, Basalttuff": 1.3,
        "Basalt, Lösslehm, Löss": 1.3,
        "Basalt, Lösslehm": 1.3,
        "Lösslehm, Basalt": 1.3,
    }
}

#: landscape protection: flat +25% (multiply); nature protection: no-go.
LANDSCAPE_PROTECTION_FACTOR = 1.25
NATURE_PROTECTION_COST = FORBIDDEN


# ------------------------------------------------------------- preprocessors
def make_street_buffer_preprocessor(buffer_a: float = 10, buffer_b: float = 4,
                                    buffer_l: float = 2) -> Callable:
    """The notebook's ``set_street_type_and_buffer`` with editable buffers.

    Classifies 'Straßenverkehr' rows by road-name prefix (A/B/L) into
    Autobahn/Bundesstr./Landstr. via the ``bez`` column and buffers their
    geometry by the per-class construction width (metres).
    """
    def set_street_type_and_buffer(gdf):
        if "nutzart" not in gdf.columns or "name" not in gdf.columns:
            return gdf
        streets = gdf["nutzart"] == "Straßenverkehr"
        for prefix, road_type, buffer_m in zip(
                ["A ", "B ", "L "],
                ["Autobahn", "Bundesstr.", "Landstr."],
                [buffer_a, buffer_b, buffer_l]):
            mask = streets & gdf["name"].str.contains(prefix, na=False)
            gdf.loc[mask, "bez"] = road_type
            gdf.loc[mask, "geometry"] = gdf.loc[mask, "geometry"].buffer(
                buffer_m)
        return gdf

    return set_street_type_and_buffer


#: named preprocessors offered in the Cost tab.
PREPROCESSORS: dict[str, dict] = {
    "none": {"label": "None", "factory": None},
    "street_buffer": {
        "label": "Street type + buffer (ALKIS A/B/L roads)",
        "factory": make_street_buffer_preprocessor,
    },
}

#: operations for the customizable preprocessing steps (Feature 4).
PREPROC_STEP_OPS = ["buffer", "set", "keep", "drop"]


def step_conditions(step: dict) -> list[dict]:
    """The condition group of a step: the ``conditions`` list if present,
    else the legacy single ``(column, operator, value)`` triple as one entry."""
    conditions = step.get("conditions")
    if isinstance(conditions, list) and conditions:
        return conditions
    return [{"column": step.get("column"),
             "operator": step.get("operator") or "all",
             "value": step.get("value")}]


def step_mask(gdf, step: dict):
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Boolean row mask of a step's condition GROUP (or None = all rows).

    Each condition is a ``(column, operator, value)`` triple resolved by
    :func:`~pyorps.gui.services.cost_model.condition_mask`; multiple
    conditions are combined by the step's group operator ``combine``
    (``"&"`` = all must match, ``"|"`` = any matches). An "all" condition
    inside a group is the identity for ``&`` and matches everything for
    ``|``.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    from .services.cost_model import condition_mask

    combine = "|" if str(step.get("combine") or "&").strip() == "|" else "&"
    combined = None
    for cond in step_conditions(step):
        mask = condition_mask(gdf, cond.get("column"),
                              cond.get("operator") or "all",
                              cond.get("value"))
        if mask is None:                     # "all rows"
            if combine == "|":
                return None                  # anything ORed with all = all
            continue                         # identity for AND
        combined = mask if combined is None else (
            (combined & mask) if combine == "&" else (combined | mask))
    return combined


def _condition_text(cond: dict) -> str:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """One condition as python-like text: ("nutzart" == "Wald")."""
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    operator = (cond.get("operator") or "all").strip()
    column = cond.get("column") or ""
    if operator in ("", "all"):
        return "(all features)"
    if operator == "is-empty":
        return f'("{column}" is empty)'
    if operator == "in":
        values = [v.strip() for v in str(cond.get("value") or "").split(",")
                  if v.strip()]
        return f'("{column}" in {values})'
    value = cond.get("value")
    if operator in ("<", "<=", ">", ">="):
        return f'("{column}" {operator} {value})'
    return f'("{column}" {operator} "{value}")'


def step_label(step: dict) -> str:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Plain-english/python-like mask text for a step, e.g.
    ``(("nutzart" == "Wald") & ("bez" == "Nadelholz")) -> buffer=2m``.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    conditions = step_conditions(step)
    combine = "|" if str(step.get("combine") or "&").strip() == "|" else "&"
    parts = [_condition_text(c) for c in conditions]
    mask = parts[0] if len(parts) == 1 else \
        "(" + f" {combine} ".join(parts) + ")"
    op = (step.get("op") or "").strip().lower()
    arg = step.get("arg")
    if op == "buffer":
        action = f"buffer={arg}m"
    elif op == "set":
        action = f'set "{step.get("target") or "?"}"={arg}'
    else:
        action = op or "?"
    return f"{mask} -> {action}"


def make_steps_preprocessor(steps: list[dict]) -> Callable:
    """Build a gdf->gdf function running ordered custom steps (Feature 4).

    Each ``step`` is ``{op, column, operator, value, target, arg}`` — or,
    built by the condition-group editor, ``{op, conditions: [{column,
    operator, value}, ...], combine: "&"|"|", target, arg}``. The condition
    (group) selects the affected rows (via :func:`step_mask` /
    :func:`pyorps.gui.services.cost_model.condition_mask`; ``op='all'`` = all
    rows), then ``op`` acts on them:

    - ``buffer`` — buffer the selected geometries by ``arg`` metres (the named
      example: "add a buffer to a decisive group of tiles").
    - ``set``    — set column ``target`` to ``arg`` on the selected rows
      (reclassify before rasterizing, e.g. force a land-use class).
    - ``keep``   — drop everything the condition does NOT match.
    - ``drop``   — drop the rows the condition matches.

    Steps run in list order so later steps see earlier edits. Unknown/blank
    ``op`` rows are skipped so a half-filled grid row can't crash a build.
    """
    def _to_float(value, default=0.0):
        try:
            return float(str(value).replace(",", "."))
        except (TypeError, ValueError):
            return default

    def run_steps(gdf):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        for step in steps or []:
            op = (step.get("op") or "").strip().lower()
            if op not in PREPROC_STEP_OPS:
                continue
            mask = step_mask(gdf, step)
            if op == "keep":
                gdf = gdf if mask is None else gdf[mask].copy()
                continue
            if op == "drop":
                if mask is not None:
                    gdf = gdf[~mask].copy()
                continue
            # buffer / set operate on the selected rows in place
            geom = gdf.geometry.name
            idx = gdf.index if mask is None else gdf.index[mask]
            if len(idx) == 0:
                continue
            if op == "buffer":
                gdf.loc[idx, geom] = gdf.loc[idx, geom].buffer(
                    _to_float(step.get("arg")))
            elif op == "set":
                target = (step.get("target") or "").strip()
                if target:
                    gdf.loc[idx, target] = step.get("arg")
        return gdf

    return run_steps
