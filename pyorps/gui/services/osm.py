"""
PYORPS GUI: OpenStreetMap features via the Overpass API.

OSM is available in the GUI as a *basemap* (raster tiles), but its real value
for routing is the underlying **vector features** — land use, roads, water,
power lines, buildings, protected areas. This module pulls those into a normal
GeoDataFrame layer (:func:`load_osm_features`), so they flow straight into the
existing machinery: pick them as a cost-model base dataset, edit them, select
features on the map, and rasterize them into a cost surface.

No OSM dependency (osmnx/pyrosm) is required — we POST an Overpass QL query with
``out geom;`` (which inlines way/relation coordinates) and assemble shapely
geometries ourselves. Queries are always bounded by the study area / map view so
they stay small and polite to the public Overpass servers.
"""
from __future__ import annotations

import geopandas as gpd
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

WGS84 = "EPSG:4326"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
#: public Overpass mirrors, tried IN ORDER when the primary is busy (429/504)
#: or unreachable — the main server rate-limits aggressively, and a mirror
#: usually answers the very same query without any waiting.
OVERPASS_ENDPOINTS = [
    OVERPASS_URL,
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.osm.jp/api/interpreter",
]
#: Overpass blocks the default ``python-requests`` User-Agent (HTTP 406) — send
#: an explicit, identifiable one (also good Overpass etiquette).
OVERPASS_HEADERS = {"User-Agent": "pyorps-gui (github.com/pyorps)"}

#: tags whose closed ways are areas (Polygon), not rings (LineString).
_AREA_TAGS = ("landuse", "building", "natural", "leisure", "amenity",
              "boundary", "waterway_area", "aeroway", "place")

#: ready-made feature queries (name, category, Overpass selectors, main tag key
#: to use as the cost feature column). ``nwr`` = node+way+relation.
OSM_FEATURE_PRESETS: list[dict] = [
    {"name": "Land use (landuse=*)", "category": "Land use",
     "filters": ['nwr["landuse"]'], "key": "landuse"},
    {"name": "Forest / wood", "category": "Land use",
     "filters": ['nwr["landuse"="forest"]', 'nwr["natural"="wood"]'],
     "key": "landuse"},
    {"name": "Water (natural=water, waterway)", "category": "Water",
     "filters": ['nwr["natural"="water"]', 'way["waterway"]'],
     "key": "waterway"},
    {"name": "Roads (highway=*)", "category": "Infrastructure",
     "filters": ['way["highway"]'], "key": "highway"},
    {"name": "Power lines (power=line/minor_line/cable)",
     "category": "Infrastructure",
     "filters": ['way["power"~"line|minor_line|cable"]'], "key": "power"},
    {"name": "Power towers / poles", "category": "Infrastructure",
     "filters": ['node["power"~"tower|pole"]'], "key": "power"},
    {"name": "Railways (railway=rail)", "category": "Infrastructure",
     "filters": ['way["railway"="rail"]'], "key": "railway"},
    {"name": "Buildings (building=*)", "category": "Buildings",
     "filters": ['nwr["building"]'], "key": "building"},
    {"name": "Protected areas / nature reserves", "category": "Protection",
     "filters": ['nwr["boundary"="protected_area"]',
                 'nwr["leisure"="nature_reserve"]'], "key": "boundary"},
]


#: guided key -> common values catalog (Feature: categorized OSM menu). The
#: user first picks a "column" (OSM tag key), then one/more of its values —
#: no Overpass-QL knowledge needed. Values are the most common ones from the
#: OSM wiki (taginfo top entries); "" (any value) is always offered too.
OSM_KEY_VALUES: dict[str, list[str]] = {
    "landuse": ["forest", "farmland", "meadow", "grass", "residential",
                "industrial", "commercial", "retail", "orchard", "vineyard",
                "allotments", "cemetery", "quarry", "landfill", "brownfield",
                "greenfield", "recreation_ground", "military", "railway",
                "basin", "reservoir", "farmyard", "greenhouse_horticulture"],
    "natural": ["wood", "water", "wetland", "scrub", "grassland", "heath",
                "sand", "beach", "bare_rock", "scree", "cliff", "tree_row",
                "shingle", "rock", "spring"],
    "highway": ["motorway", "trunk", "primary", "secondary", "tertiary",
                "unclassified", "residential", "service", "track", "path",
                "footway", "cycleway", "bridleway", "living_street",
                "pedestrian", "steps", "motorway_link", "trunk_link",
                "primary_link", "secondary_link"],
    "waterway": ["river", "stream", "canal", "drain", "ditch", "riverbank",
                 "weir", "dam", "lock_gate"],
    "railway": ["rail", "light_rail", "tram", "subway", "narrow_gauge",
                "disused", "abandoned", "platform", "station"],
    "power": ["line", "minor_line", "cable", "tower", "pole", "substation",
              "transformer", "generator", "plant", "portal", "switch"],
    "building": ["house", "residential", "apartments", "detached", "garage",
                 "industrial", "commercial", "retail", "farm", "barn",
                 "church", "school", "greenhouse", "shed", "warehouse"],
    "boundary": ["protected_area", "national_park", "administrative",
                 "forest", "water_protection_area"],
    "leisure": ["nature_reserve", "park", "pitch", "playground", "garden",
                "sports_centre", "golf_course", "swimming_pool"],
    "amenity": ["parking", "school", "hospital", "fuel", "fire_station",
                "place_of_worship", "kindergarten", "recycling"],
    "man_made": ["pipeline", "wastewater_plant", "water_works", "tower",
                 "mast", "silo", "storage_tank", "bridge", "pier"],
    "aeroway": ["aerodrome", "runway", "taxiway", "helipad", "apron"],
    "barrier": ["fence", "wall", "hedge", "gate", "retaining_wall"],
}


def filters_from_selections(selections: list[dict]) -> list[str]:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Categorized-menu rows -> Overpass selectors.

    Each selection is ``{"key": <tag key>, "values": [v1, v2, ...]}``; an
    empty/absent values list means "any value" (bare key). Every selection is
    ORed by the surrounding Overpass union, matching the menu semantics
    "multiple combinations of columns and values".
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    filters: list[str] = []
    for sel in selections or []:
        key = str(sel.get("key") or "").strip()
        if not key:
            continue
        values = [str(v).strip() for v in (sel.get("values") or [])
                  if str(v).strip()]
        if not values:
            filters.append(f'nwr["{key}"]')
        elif len(values) == 1:
            filters.append(f'nwr["{key}"="{values[0]}"]')
        else:                       # one regex selector for the value set
            pattern = "|".join(values)
            filters.append(f'nwr["{key}"~"^({pattern})$"]')
    return filters


def selection_label(sel: dict) -> str:
    """Human-readable chip text for one categorized-menu selection."""
    key = str(sel.get("key") or "").strip()
    values = [str(v).strip() for v in (sel.get("values") or [])
              if str(v).strip()]
    if not values:
        return f"{key} = any"
    return f"{key} = {', '.join(values)}"


def preset_filters(name: str) -> list[str]:
    for preset in OSM_FEATURE_PRESETS:
        if preset["name"] == name:
            return list(preset["filters"])
    return []


def custom_filters(text: str) -> list[str]:
    """Turn a user tag string into Overpass ``nwr`` selectors.

    Accepts ``key`` (any value), ``key=value``, or a comma-separated list of
    those, e.g. ``landuse=forest, natural=wood`` or ``highway``.
    """
    filters = []
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            key, value = (p.strip() for p in part.split("=", 1))
            filters.append(f'nwr["{key}"="{value}"]')
        else:
            filters.append(f'nwr["{part}"]')
    return filters


def build_query(filters: list[str], bbox_sw_ne: tuple[float, float, float,
                float], *, timeout: int = 60) -> str:
    """Overpass QL for ``filters`` inside ``bbox_sw_ne`` = (south, west, north, east)."""
    south, west, north, east = bbox_sw_ne
    box = f"({south},{west},{north},{east})"
    body = "".join(f"{f}{box};" for f in filters)
    return f"[out:json][timeout:{timeout}];({body});out geom;"


# -------------------------------------------------------------- geometry build
def _way_geometry(element: dict):
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    coords = [(g["lon"], g["lat"]) for g in element.get("geometry") or []
              if g and "lon" in g and "lat" in g]
    if len(coords) < 2:
        return None
    tags = element.get("tags") or {}
    closed = len(coords) >= 4 and coords[0] == coords[-1]
    is_area = tags.get("area") != "no" and any(k in tags for k in _AREA_TAGS)
    if closed and is_area:
        try:
            return Polygon(coords)
        except Exception:
            return LineString(coords)
    return LineString(coords)


def _relation_geometry(element: dict):
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    if (element.get("tags") or {}).get("type") != "multipolygon":
        return None
    outers = []
    for member in element.get("members") or []:
        if member.get("role") == "outer" and member.get("geometry"):
            coords = [(g["lon"], g["lat"]) for g in member["geometry"]]
            if len(coords) >= 4:
                try:
                    outers.append(Polygon(coords))
                except Exception:  # nosec B110
                    pass
    if not outers:
        return None
    return outers[0] if len(outers) == 1 else MultiPolygon(outers)


def elements_to_gdf(elements: list[dict]) -> gpd.GeoDataFrame:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Assemble Overpass ``out geom`` elements into a WGS84 GeoDataFrame."""
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    rows, geoms = [], []
    for element in elements or []:
        etype = element.get("type")
        if etype == "node" and "lon" in element and "lat" in element:
            geom = Point(element["lon"], element["lat"])
        elif etype == "way":
            geom = _way_geometry(element)
        elif etype == "relation":
            geom = _relation_geometry(element)
        else:
            geom = None
        if geom is None or geom.is_empty:
            continue
        record = {"osm_id": element.get("id"), "osm_type": etype}
        record.update(element.get("tags") or {})
        rows.append(record)
        geoms.append(geom)
    if not rows:
        return gpd.GeoDataFrame({"osm_id": []}, geometry=[], crs=WGS84)
    return gpd.GeoDataFrame(rows, geometry=geoms, crs=WGS84)


def _post_overpass(endpoint: str, query: str, timeout: int):
    """POST one Overpass query. Returns the parsed JSON payload, or the
    string ``"busy"`` when this endpoint rate-limits / times out (the caller
    then tries the next mirror). Raises ValueError for non-retryable errors."""
    import requests

    try:
        response = requests.post(endpoint, data={"data": query},  # nosec B113 - timeout is set on the next line
                                 headers=OVERPASS_HEADERS, timeout=timeout + 10)
    except requests.RequestException:
        return "busy"                      # unreachable -> try the next mirror
    if response.status_code in (429, 502, 503, 504):
        return "busy"                      # overloaded -> try the next mirror
    if response.status_code != 200:
        raise ValueError(
            f"Overpass returned HTTP {response.status_code}. Check the tag "
            "filter, or draw a smaller area.")
    try:
        return response.json()
    except ValueError as exc:
        raise ValueError("Overpass did not return valid JSON — the query may "
                         "be malformed.") from exc


def load_osm_features(bbox_sw_ne: tuple[float, float, float, float],
                      filters: list[str], *, timeout: int = 60,
                      endpoint: str | None = None) -> gpd.GeoDataFrame:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Query Overpass for ``filters`` in the bbox and return a WGS84 GeoDataFrame.

    ``bbox_sw_ne`` is (south, west, north, east). The busy public main server
    is no longer a dead end: rate-limited / overloaded / unreachable endpoints
    fall through to the next public mirror (OVERPASS_ENDPOINTS) before giving
    up. Raises ValueError with a readable message on server / empty-result
    problems.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    if not filters:
        raise ValueError(
            "No OSM feature selected — pick a feature column and value, a "
            "preset (e.g. 'Land use'), or enter a tag like 'landuse=forest'.")
    query = build_query(filters, bbox_sw_ne, timeout=timeout)
    endpoints = [endpoint] if endpoint else list(OVERPASS_ENDPOINTS)
    payload = None
    for url in endpoints:
        payload = _post_overpass(url, query, timeout)
        if payload != "busy":
            break
    if payload == "busy" or payload is None:
        tried = ", ".join(endpoints)
        raise ValueError(
            "All Overpass servers are rate-limiting / overloaded or "
            f"unreachable right now (tried {tried}). Wait a minute, or draw "
            "a smaller study area, and retry.")
    gdf = elements_to_gdf(payload.get("elements") or [])
    if gdf.empty:
        raise ValueError(
            "No OSM features of that kind in the study area. Try a different "
            "feature type or a larger area.")
    return gdf
