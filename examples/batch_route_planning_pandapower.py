"""Batch route planning between MV station pairs of a pandapower grid.

Loads a pandapower 3 grid model, extracts the MV station busbar coordinates
from ``net.bus.geo``, builds one multi-layer cost raster covering the full
grid (ALKIS land use with road buffering + drinking water protection + soil
conditions + landscape/nature protection — same defaults as
``prepare_data_for_distribution_grid_planning.ipynb``), and routes the
candidate station pairs with a single PathFinder using the GPU SSSP V4
backend.

Candidate pair selection: instead of routing all N(N-1)/2 unordered pairs,
the script builds a Delaunay triangulation on the MV station coordinates and
keeps only triangulation edges whose Euclidean length is at most
``MAX_PAIR_DISTANCE_M`` (5 km). This mirrors typical target-grid-planning
candidate sets — neighboring stations only, no long-range duplicates — and
cuts routing cost dramatically on dense grids.

Routing efficiency: the cost raster is built once and the underlying graph
is reused across all pairs. Pairs are grouped by source so one
single-source delta-stepping wave covers all targets of a given source.
"""

import json
from pathlib import Path as FilePath

import geopandas as gpd
import numpy as np
import pandas as pd
import pandapower as pp
from rasterio.features import geometry_mask
from scipy.spatial import Delaunay
from shapely.geometry import shape

from pyorps import (
    CostAssumptions,
    GeoRasterizer,
    PathFinder,
    detect_feature_columns,
    initialize_geo_dataset,
)
from pyorps.core.exceptions import NoPathFoundError


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

NET_JSON = FilePath(
    "../gis2pp/output/example_2025_eamex/example_2025.json"
)
OUTPUT_DIR = FilePath("output/example_2025_eamex/route_planning")

# Optional post-routing LineString simplification. Set to None to disable.
# When enabled, every routed path is simplified for export but cost and
# per-category metrics are kept from the un-simplified routed line (a
# simplification shortcut can cross forbidden / high-cost cells that the
# routed line carefully avoided, which would falsely deflate the cost).
# Tolerance is in CRS units (metres on EPSG:25832).
SIMPLIFY = {"method": "douglas_peucker", "tolerance": 1.0}

# Buffer (metres) applied to every feature when rasterizing the cost map.
# A small buffer (~ SIMPLIFY tolerance) keeps low-cost corridors wide enough
# that a simplified-line shortcut of size `tolerance` is unlikely to leave
# the corridor and end up rendered through a forbidden / high-cost cell.
GEOMETRY_BUFFER_M = 1.0

# OpenStreetMap streets/paths overlay — used to fill gaps where ALKIS lacks a
# street or path. Applied at the same cost as an ALKIS path (~281 €/m for
# 20 kV cable + conduits + "ohne Oberfläche" surface) and only on cells that
# ALKIS has NOT already classified as street / path / square / rail /
# forbidden (the ALKIS "official" dataset always wins).
OSM_STREETS_PATH = FilePath(
    r"C:\Users\mhnn82\Documents\8_example\2_Projekte\example"
    r"\Zielnetzplanung\Daten\PYORPS-Routes\open_street_map_streets.geojson"
)
OSM_BUFFER_M = 2.0
OSM_PATH_COST = 281
OSM_STREET_PATH_TAGS = frozenset({
    # streets
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link", "unclassified", "residential",
    "service", "living_street", "pedestrian",
    # paths
    "track", "path", "footway", "cycleway", "bridleway", "steps",
})
# ALKIS nutzart values whose geometries must NEVER be overwritten by OSM, even
# if rasterisation stored a different (higher) cost there because a forest or
# field polygon happened to overlap the street polygon. We mask by the vector
# geometries directly so the "ALKIS wins" rule holds regardless of which
# overlapping polygon the "higher cost wins" rasterisation picked. (Cells that
# are already forbidden — uint16 == 65535 — are additionally protected via a
# raster-value check inside overlay_osm_streets.)
ALKIS_INFRASTRUCTURE_CATEGORIES = frozenset({
    "Straßenverkehr", "Weg", "Platz", "Bahnverkehr", "Flugverkehr",
})

# Raster filename includes both buffers so re-running with a different buffer
# (ALKIS or OSM) or without the OSM overlay does not silently reuse a
# previously cached raster.
RASTER_PATH = (
    OUTPUT_DIR
    / f"cost_raster_buf{GEOMETRY_BUFFER_M:g}m_osm{OSM_BUFFER_M:g}m_v2.tiff"
)

_paths_stem = f"all_mv_routes_osm{OSM_BUFFER_M:g}m_v2"
if SIMPLIFY is not None:
    _paths_stem += f"_simplified_{SIMPLIFY['method']}_{SIMPLIFY['tolerance']}"
PATHS_PATH = OUTPUT_DIR / f"{_paths_stem}.geojson"

# Projected CRS used for raster + routing (Hessen — ETRS89 / UTM zone 32N).
WORKING_CRS = "EPSG:25832"
# CRS that net.bus.geo is stored in. The gis2pp example export already uses
# ETRS89 / UTM 32N, so it matches WORKING_CRS; set this to "EPSG:4326" for
# grids whose bus geo is stored as WGS 84 lon/lat.
GEODATA_CRS = "EPSG:25832"

# Buffer around the convex hull of the stations (meters in WORKING_CRS).
BBOX_BUFFER_M = 1000.0

# Voltage band (kV) considered "MV" when picking station buses.
MV_VOLTAGE_MIN_KV = 1.0
MV_VOLTAGE_MAX_KV = 60.0

# Maximum Euclidean length of a candidate routing pair (meters). Delaunay
# edges longer than this are discarded — keeps candidate set close to the
# typical neighbor-station distances seen in target grid planning.
MAX_PAIR_DISTANCE_M = 5000.0


# --------------------------------------------------------------------------- #
# Cost-assumption defaults (from prepare_data_for_distribution_grid_planning.ipynb)
# --------------------------------------------------------------------------- #

BASE_WFS_REQUEST = {
    "url": "https://www.gds.hessen.de/wfs2/aaa-suite/cgi-bin/alkis/vereinf/wfs",
    "layer": "ave_Nutzung",
}

DRINKING_WATER_WFS = {
    "url": ("https://geodienste-umwelt.hessen.de/arcgis/services/inspire/"
            "bewirtschaftungsgebiete/MapServer/WFSServer"),
    "layer": "TWS_HQS_TK25",
}
SOIL_CONDITIONS_WFS = {
    "url": ("https://geodienste-umwelt.hessen.de/arcgis/services/inspire/"
            "boden/MapServer/WFSServer"),
    "layer": "Bodeneinheiten_Bodenuebersicht_500000",
}
NATURE_PROTECTION_WFS = {
    "url": ("https://geodienste-umwelt.hessen.de/arcgis/services/inspire/"
            "schutzgebiete/MapServer/WFSServer"),
    "layer": "Naturschutzgebiete",
}
LANDSCAPE_PROTECTION_WFS = {
    "url": ("https://geodienste-umwelt.hessen.de/arcgis/services/inspire/"
            "schutzgebiete/MapServer/WFSServer"),
    "layer": "Landschaftsschutzgebiete",
}

# Per-metre cable installation costs (20 kV, including cable + conduits) anchored
# to company figures:
#     Cable (20 kV):           56 €/m
#     "ohne Oberfläche" (unpaved trench, no surface restoration): 225 €/m
#     "mit Oberfläche" (paved trench, surface restoration):       328 €/m
#     Leerrohre (conduits):    17 €/m cable + 34 €/m surface
# Streets and paths are billed at the "ohne Oberfläche" rate per user request
# (do not penalise routing along streets/paths for surface restoration), so
# their base cost is 56 + 225 + 17 + 34 ≈ 280 €/m. Forest, grassland and
# farmland are charged a much higher per-metre figure on top of the natural
# unpaved rate to cover tree clearing, crop damage, access restrictions and
# permitting — this keeps routing on the street network and only forces it off
# when the detour would otherwise be huge. Highways stay expensive (heavy
# traffic, deep reinforcement, complex permitting). Forbidden zones keep their
# uint16 sentinel (65535).
LAND_USE_COSTS = {
    ("nutzart", "bez"): {
        "Wald": {
            "Nadelholz": 600, "Laub- und Nadelholz": 625,
            "Laubholz": 700, "": 600,
        },
        "Straßenverkehr": {
            "Landesstr.": 281, "Bundesstr.": 281,
            "Autobahn": 750, "": 281,
        },
        "Weg": {"Fußweg": 281, "Rad- und Fußweg": 281, "": 281},
        "Landwirtschaft": {
            "Ackerland": 650, "Grünland": 650, "Gartenbauland": 750,
            "Streuobstwiese": 750, "Obst- und Nussplantage": 750,
            "Streuobstacker": 700, "Baumschule": 750,
            "Brachland": 550, "": 650,
        },
        "Fließgewässer": {
            "Graben": 400, "Bach": 500, "Kanal": 600,
            "Fluss": 1000, "": 400,
        },
        "Stehendes Gewässer": {
            "Teich": 500, "Speicherbecken": 1000, "Stausee": 1000,
            "Baggersee": 1000, "": 600,
        },
        "Sport-, Freizeit- und Erholungsfläche": {"Grünanlage": 400, "": 65535},
        "Gehölz": {"": 500},
        "Platz": {"Parkplatz": 281, "Rastplatz": 281, "": 281},
        "Flugverkehr": {
            "Segelfluggelände": 400, "Sonderlandeplatz": 400, "": 65535,
        },
        "Bahnverkehr": {"": 800},
        "Heide": {"": 500},
        "Unland/Vegetationslose Fläche": {"": 400},
        "Fläche gemischter Nutzung": {"": 65535},
        "Fläche besonderer funktionaler Prägung": {"": 65535},
        "Wohnbaufläche": {"": 65535},
        "Sumpf": {"": 65535},
        "Industrie- und Gewerbefläche": {"": 65535},
        "Tagebau, Grube, Steinbruch": {"": 65535},
        "Friedhof": {"": 65535},
        "Moor": {"": 65535},
        "Halde": {"": 65535},
        "Schiffsverkehr": {"": 65535},
    }
}

SOIL_CONDITION_FACTORS = {
    "AUSGANGSGESTEIN": {
        # DIN 18300 class 1–3 — easy excavation
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
        # Class 4–5 — cohesive soils
        "Schluff- und Tonsteine, Sandsteine": 1.05,
        "Ton- und Schluffsteine und Arkosen, örtl. carbonathaltig": 1.05,
        # Class 6 — soft rock
        "Sandsteine": 1.15,
        "Grauwacken, Sandsteine, Konglomerate, Quarzite, Kieselschiefer": 1.15,
        "Kalkstein, Mergel, Dolomit": 1.15,
        "Tonschiefer, Grauwackenschiefer, Phyllit": 1.15,
        "Schalstein, Diabas": 1.15,
        "Kalkstein, Mergel, Dolomit, Ton- und Schluffsteine und Arkosen": 1.15,
        "Quarzite, Sandsteine": 1.15,
        "Ton- und Schluffsteine, Arkosen, Kalkstein, Mergel, Dolomit": 1.15,
        # Class 7 — hard rock
        "Gabbro, Diorit, Amphibolit, Melaphyr, Basalt": 1.3,
        "Granodiorit, Quarzporphyr, Glimmer- und Quarzitschiefer, Gneis": 1.3,
        "Basalt, Basalttuff": 1.3,
        "Basalt, Lösslehm, Löss": 1.3,
        "Basalt, Lösslehm": 1.3,
        "Lösslehm, Basalt": 1.3,
    }
}


def _alkis_infrastructure_mask(rasterizer: GeoRasterizer) -> np.ndarray:
    """Boolean raster mask covering ALKIS infrastructure polygons.

    Uses the geometries from ``rasterizer.base_dataset.data`` — which the
    rasterise step already mutated in-place with the per-road-class buffers
    from ``set_street_type_and_buffer`` — and additionally inflates them by
    ``GEOMETRY_BUFFER_M`` so the mask matches exactly the cells that the
    base ALKIS rasterisation touched.
    """
    base_gdf = rasterizer.base_dataset.data
    infra = base_gdf[base_gdf["nutzart"].isin(ALKIS_INFRASTRUCTURE_CATEGORIES)]
    infra = infra[infra.geometry.notna() & ~infra.geometry.is_empty]
    if infra.empty:
        return np.zeros(rasterizer.raster.shape, dtype=bool)
    geoms = infra.geometry
    if GEOMETRY_BUFFER_M > 0:
        geoms = geoms.buffer(GEOMETRY_BUFFER_M)
    return geometry_mask(
        geoms.values,
        transform=rasterizer.transform,
        invert=True,
        out_shape=rasterizer.raster.shape,
    )


def overlay_osm_streets(rasterizer: GeoRasterizer, bbox) -> None:
    """Fill ALKIS street/path gaps using OSM highways at the path cost.

    ALKIS always wins: cells inside any ALKIS infrastructure polygon
    (street, path, square, rail, airfield) are left untouched regardless of
    what cost the "higher cost wins" rasterisation actually stored for them.
    Cells that are already forbidden (uint16 == 65535) are also preserved.
    Every other cell covered by a buffered OSM street/path geometry is set
    to ``OSM_PATH_COST``.
    """
    if not OSM_STREETS_PATH.exists():
        print(f"  OSM file not found at {OSM_STREETS_PATH} — skipping overlay")
        return

    osm = gpd.read_file(str(OSM_STREETS_PATH))
    osm = osm.to_crs(rasterizer.crs)
    minx, miny, maxx, maxy = bbox
    osm = osm.cx[minx:maxx, miny:maxy]
    osm = osm[osm["highway"].isin(OSM_STREET_PATH_TAGS)]
    osm = osm[osm.geometry.notna() & ~osm.geometry.is_empty]
    if osm.empty:
        print("  no OSM streets/paths within bbox — skipping overlay")
        return
    print(
        f"  OSM streets/paths: {len(osm):,} linestrings within bbox "
        f"(buffer = {OSM_BUFFER_M} m)"
    )
    osm = osm.assign(geometry=osm.geometry.buffer(OSM_BUFFER_M))

    osm_mask = geometry_mask(
        osm.geometry.values,
        transform=rasterizer.transform,
        invert=True,
        out_shape=rasterizer.raster.shape,
    )
    alkis_infra_mask = _alkis_infrastructure_mask(rasterizer)
    forbidden_mask = rasterizer.raster == 65535
    protected_mask = alkis_infra_mask | forbidden_mask
    apply_mask = osm_mask & ~protected_mask
    n_applied = int(apply_mask.sum())
    n_blocked_infra = int((osm_mask & alkis_infra_mask).sum())
    n_blocked_forbidden = int(
        (osm_mask & forbidden_mask & ~alkis_infra_mask).sum()
    )
    rasterizer.raster[apply_mask] = OSM_PATH_COST
    print(
        f"  OSM overlay: set {n_applied:,} cells to {OSM_PATH_COST}; "
        f"kept {n_blocked_infra:,} ALKIS-infrastructure cells, "
        f"{n_blocked_forbidden:,} forbidden cells."
    )


def set_street_type_and_buffer(gdf):
    """Differentiate roads by class prefix and apply construction buffers."""
    streets = gdf["nutzart"] == "Straßenverkehr"
    for prefix, road_type, buffer_m in zip(
        ["A ", "B ", "L "],
        ["Autobahn", "Bundesstr.", "Landstr."],
        [10, 4, 2],
    ):
        mask = streets & gdf["name"].str.contains(prefix, na=False)
        gdf.loc[mask, "bez"] = road_type
        gdf.loc[mask, "geometry"] = gdf.loc[mask, "geometry"].buffer(buffer_m)


# --------------------------------------------------------------------------- #
# MV station extraction
# --------------------------------------------------------------------------- #

def _parse_bus_geo(geo):
    if geo is None or (isinstance(geo, float) and pd.isna(geo)):
        return None
    if hasattr(geo, "x") and hasattr(geo, "y"):
        return geo
    if isinstance(geo, str):
        return shape(json.loads(geo))
    if isinstance(geo, dict):
        return shape(geo)
    raise TypeError(f"Unsupported net.bus.geo entry type: {type(geo)}")


def extract_mv_stations(net) -> gpd.GeoDataFrame:
    """Return MV station busbars as a GeoDataFrame in WORKING_CRS.

    Stations = ext_grid buses ∪ transformer HV-side buses, restricted to buses
    whose nominal voltage falls in the configured MV band.
    """
    station_buses: set[int] = set(net.ext_grid.bus.tolist())
    if len(net.trafo):
        station_buses.update(net.trafo.hv_bus.tolist())

    mv_mask = net.bus.vn_kv.between(MV_VOLTAGE_MIN_KV, MV_VOLTAGE_MAX_KV)
    station_buses = [b for b in station_buses if mv_mask.loc[b]]

    rows = []
    for bus_id in station_buses:
        geom = _parse_bus_geo(net.bus.geo.loc[bus_id])
        if geom is None:
            continue
        rows.append((bus_id, net.bus.name.loc[bus_id], geom))

    if not rows:
        raise RuntimeError(
            "No MV station bus has geographic coordinates in net.bus.geo."
        )

    bus_ids, names, geoms = zip(*rows)
    return gpd.GeoDataFrame(
        {"bus": list(bus_ids), "name": list(names)},
        geometry=list(geoms),
        crs=GEODATA_CRS,
    ).to_crs(WORKING_CRS)


# --------------------------------------------------------------------------- #
# Cost raster
# --------------------------------------------------------------------------- #

def build_cost_raster(bbox: tuple[float, float, float, float]) -> None:
    """Build the multi-layer cost raster covering ``bbox`` and save to disk."""
    print(f"Building cost raster for bbox {bbox}")

    base_ds = initialize_geo_dataset(BASE_WFS_REQUEST, bbox=bbox)
    base_ds.load_data()
    print(f"  base ALKIS land use: {len(base_ds.data):,} features")

    rasterizer = GeoRasterizer(base_ds, CostAssumptions(LAND_USE_COSTS))
    print(
        f"  rasterizing base land use (street buffering applied, "
        f"global geometry buffer = {GEOMETRY_BUFFER_M} m)..."
    )
    rasterizer.rasterize(
        preprocessing_function=set_street_type_and_buffer,
        geometry_buffer_m=GEOMETRY_BUFFER_M,
    )

    print("  overlaying OpenStreetMap streets/paths (ALKIS wins ties)...")
    overlay_osm_streets(rasterizer, bbox)

    # Drinking water protection — multipliers keyed on the auto-detected feature.
    water_ds = initialize_geo_dataset(DRINKING_WATER_WFS, bbox=bbox)
    water_ds.load_data()
    main_feature_tws, _ = detect_feature_columns(
        water_ds.data, max_features_per_column=4
    )
    water_multipliers = CostAssumptions({
        main_feature_tws: {1: 100, 2: 2, 3: 1.5, 4: 1.2}
    })

    datasets_to_modify = [
        {"input_data": DRINKING_WATER_WFS,
         "cost_assumptions": water_multipliers,
         "multiply": True,
         "geometry_buffer_m": GEOMETRY_BUFFER_M},
        {"input_data": SOIL_CONDITIONS_WFS,
         "cost_assumptions": CostAssumptions(SOIL_CONDITION_FACTORS),
         "multiply": True,
         "geometry_buffer_m": GEOMETRY_BUFFER_M},
        {"input_data": LANDSCAPE_PROTECTION_WFS,
         "cost_assumptions": 1.25,
         "multiply": True,
         "geometry_buffer_m": GEOMETRY_BUFFER_M},
        # Nature protection zones are treated as no-go (max uint16).
        {"input_data": NATURE_PROTECTION_WFS,
         "cost_assumptions": 65535,
         "multiply": False,
         "geometry_buffer_m": GEOMETRY_BUFFER_M},
    ]
    for i, params in enumerate(datasets_to_modify, start=1):
        action = "multiply" if params["multiply"] else "override"
        print(f"  applying modification {i}/4 ({action})...")
        rasterizer.modify_raster_from_dataset(**params)

    RASTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    rasterizer.save_raster(save_path=str(RASTER_PATH))
    print(f"  saved cost raster -> {RASTER_PATH}")


# --------------------------------------------------------------------------- #
# Candidate pair selection (Delaunay triangulation + distance cutoff)
# --------------------------------------------------------------------------- #

def select_candidate_pairs(
    coords: list[tuple[float, float]],
    max_distance_m: float,
) -> list[tuple[int, int]]:
    """Return unordered ``(i, j)`` index pairs with ``i < j`` from the Delaunay
    triangulation of ``coords``, filtered to Euclidean length ≤ ``max_distance_m``.

    For fewer than 3 stations a Delaunay triangulation is undefined, so the
    function falls back to the complete graph (also distance-filtered).
    """
    pts = np.asarray(coords, dtype=float)
    n = len(pts)

    if n < 3:
        edges = {(i, j) for i in range(n) for j in range(i + 1, n)}
    else:
        tri = Delaunay(pts)
        edges: set[tuple[int, int]] = set()
        for simplex in tri.simplices:
            a, b, c = int(simplex[0]), int(simplex[1]), int(simplex[2])
            for u, v in ((a, b), (b, c), (a, c)):
                edges.add((u, v) if u < v else (v, u))

    keep: list[tuple[int, int]] = []
    for i, j in edges:
        if float(np.linalg.norm(pts[i] - pts[j])) <= max_distance_m:
            keep.append((i, j))
    keep.sort()
    return keep


# --------------------------------------------------------------------------- #
# Batch routing
# --------------------------------------------------------------------------- #

def route_all_pairs(stations: gpd.GeoDataFrame) -> PathFinder:
    coords = [(p.x, p.y) for p in stations.geometry]
    n = len(coords)

    pairs = select_candidate_pairs(coords, MAX_PAIR_DISTANCE_M)
    full_pairs = n * (n - 1) // 2
    print(
        f"Selected {len(pairs)} Delaunay candidate pairs "
        f"(<= {MAX_PAIR_DISTANCE_M:.0f} m) out of {full_pairs} possible "
        f"unordered pairs across {n} stations."
    )
    if not pairs:
        raise RuntimeError(
            "No station pair survived the Delaunay + distance filter — "
            "loosen MAX_PAIR_DISTANCE_M."
        )

    # Group targets by source so we issue one single-source delta-stepping wave
    # per source against all of its candidate neighbors.
    targets_by_source: dict[int, list[int]] = {}
    for i, j in pairs:
        targets_by_source.setdefault(i, []).append(j)

    # Use all station coords as both source and target so the raster window
    # (computed in create_raster_handler) covers the entire station set.
    path_finder = PathFinder(
        source_coords=coords,
        target_coords=coords,
        dataset_source=str(RASTER_PATH),
        search_space_buffer_m=BBOX_BUFFER_M,
        neighborhood_str="r3",
    )

    # Each call reuses the cached graph; paths accumulate in path_finder.paths.
    # On NoPathFoundError the batched multi-target call aborts mid-flight, so we
    # fall back to per-target calls for that source to keep the surviving pairs.
    sources_with_targets = sorted(targets_by_source)
    skipped: list[tuple[int, int]] = []
    for k, src_idx in enumerate(sources_with_targets, start=1):
        tgt_idxs = targets_by_source[src_idx]
        print(
            f"  source {k}/{len(sources_with_targets)} "
            f"(station {src_idx}) -> {len(tgt_idxs)} targets"
        )
        try:
            path_finder.find_route(
                source=coords[src_idx],
                target=[coords[j] for j in tgt_idxs],
                algorithm="delta-stepping",
                simplify=SIMPLIFY,
            )
        except NoPathFoundError:
            print(
                f"    batch failed for source {src_idx}; "
                f"retrying targets individually."
            )
            for j in tgt_idxs:
                try:
                    path_finder.find_route(
                        source=coords[src_idx],
                        target=coords[j],
                        algorithm="delta-stepping",
                        simplify=SIMPLIFY,
                    )
                except NoPathFoundError:
                    print(
                        f"    skipped pair ({src_idx}, {j}) — no path found."
                    )
                    skipped.append((src_idx, j))

    if skipped:
        print(f"Skipped {len(skipped)} pair(s) with no feasible path:")
        for s, t in skipped:
            print(f"  ({s}, {t})")

    return path_finder


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading pandapower grid from {NET_JSON}")
    net = pp.from_json(str(NET_JSON))

    stations = extract_mv_stations(net)
    print(f"Found {len(stations)} MV station busbars with geodata")

    minx, miny, maxx, maxy = stations.total_bounds
    bbox = (
        minx - BBOX_BUFFER_M, miny - BBOX_BUFFER_M,
        maxx + BBOX_BUFFER_M, maxy + BBOX_BUFFER_M,
    )

    if not RASTER_PATH.exists():
        build_cost_raster(bbox)
    else:
        print(f"Re-using existing cost raster: {RASTER_PATH}")

    path_finder = route_all_pairs(stations)

    print(f"Saving {len(path_finder.paths)} paths -> {PATHS_PATH}")
    path_finder.save_paths(str(PATHS_PATH))
    print("Done.")


if __name__ == "__main__":
    main()
