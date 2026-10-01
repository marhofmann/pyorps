"""Route every LV cable box to its nearest transformer station (example).

A companion to ``batch_route_planning_pandapower.py``. Where that script routes
between neighbouring MV *station pairs*, this one connects each LV **cable box**
to the **nearest transformer station** with a single least-cost path.

Cable boxes are LV (0.4 kV) buses that are either ``VERT-`` cable distributors
(Kabelverteiler) or plain busbars (``bus_type == "bb"``), excluding any bus that
belongs to a LV/MV station (``substation_name`` starting with ``ST`` or
``trafo_station == True``). Transformer stations are taken from the batch
script's ``extract_mv_stations`` (ext_grid ∪ transformer HV buses in the MV
band).

Everything else — the cost raster, the grid model, the CRS, the routing
settings (``r3`` neighbourhood, 1 km search-space buffer, delta-stepping on the
GPU SSSP V4 backend, Douglas–Peucker simplification) — is reused verbatim from
``batch_route_planning_pandapower.py`` by importing its constants and helpers,
so the two scripts can never drift apart.

Assignment is by **Euclidean-nearest** station. Boxes are grouped by their
assigned station so one single-source delta-stepping wave covers all boxes of a
station, reusing the cached graph across the whole batch.
"""

import geopandas as gpd
import numpy as np
import pandapower as pp
from scipy.spatial import cKDTree

from pyorps import PathFinder
from pyorps.core.exceptions import NoPathFoundError

# Reuse configuration and helpers from the sibling batch script verbatim so the
# raster, grid model and routing settings stay identical between both scripts.
from batch_route_planning_pandapower import (
    BBOX_BUFFER_M,
    GEODATA_CRS,
    NET_JSON,
    OUTPUT_DIR,
    RASTER_PATH,
    SIMPLIFY,
    WORKING_CRS,
    _parse_bus_geo,
    build_cost_raster,
    extract_mv_stations,
)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

_routes_stem = "cablebox_to_station_routes"
if SIMPLIFY is not None:
    _routes_stem += f"_simplified_{SIMPLIFY['method']}_{SIMPLIFY['tolerance']}"
ROUTES_PATH = OUTPUT_DIR / f"{_routes_stem}.geojson"


# --------------------------------------------------------------------------- #
# Cable-box extraction
# --------------------------------------------------------------------------- #

def extract_cable_boxes(net) -> gpd.GeoDataFrame:
    """Return LV cable boxes as a GeoDataFrame in WORKING_CRS.

    Cable boxes = LV (0.4 kV) buses that are ``VERT-`` distributors or ``bb``
    busbars, excluding any bus that belongs to a LV/MV station (``ST`` name or
    ``trafo_station``). All matching buses with geographic coordinates are
    returned; the in-service breakdown is printed for information.
    """
    lv = net.bus[net.bus.vn_kv == 0.4]
    sub = lv.substation_name.fillna("").astype(str)

    is_vert = sub.str.startswith("VERT")
    is_bb = (lv.bus_type == "bb").fillna(False)
    belongs_to_station = sub.str.startswith("ST") | (lv.trafo_station == True)  # noqa: E712

    boxes = lv[(is_vert | is_bb) & ~belongs_to_station]

    rows = []
    for bus_id, name, geo in zip(boxes.index, boxes.name, boxes.geo):
        geom = _parse_bus_geo(geo)
        if geom is None:
            continue
        rows.append((bus_id, name, geom))

    if not rows:
        raise RuntimeError("No LV cable box has geographic coordinates.")

    n_in_service = int((boxes.in_service == True).sum())  # noqa: E712
    print(
        f"  cable boxes: {len(rows)} with geo "
        f"({len(boxes)} matched, {n_in_service} in service)"
    )

    bus_ids, names, geoms = zip(*rows)
    return gpd.GeoDataFrame(
        {"bus": list(bus_ids), "name": list(names)},
        geometry=list(geoms),
        crs=GEODATA_CRS,
    ).to_crs(WORKING_CRS)


# --------------------------------------------------------------------------- #
# Nearest-station assignment + routing
# --------------------------------------------------------------------------- #

def route_boxes_to_nearest_station(
    boxes: gpd.GeoDataFrame,
    stations: gpd.GeoDataFrame,
) -> PathFinder:
    """Route each cable box to its Euclidean-nearest transformer station."""
    box_coords = [(p.x, p.y) for p in boxes.geometry]
    station_coords = [(p.x, p.y) for p in stations.geometry]

    # Euclidean-nearest station for each box.
    tree = cKDTree(np.asarray(station_coords, dtype=float))
    _, nearest_station = tree.query(np.asarray(box_coords, dtype=float), k=1)

    # Group box indices by assigned station so each station is routed with one
    # single-source delta-stepping wave against all of its boxes.
    boxes_by_station: dict[int, list[int]] = {}
    for box_idx, st_idx in enumerate(nearest_station):
        boxes_by_station.setdefault(int(st_idx), []).append(box_idx)

    print(
        f"Routing {len(box_coords)} cable boxes to "
        f"{len(boxes_by_station)} of {len(station_coords)} stations "
        f"(Euclidean-nearest assignment)."
    )

    # One PathFinder over all coordinates so the raster window covers the full
    # extent; the cached graph is reused across every find_route call.
    all_coords = station_coords + box_coords
    path_finder = PathFinder(
        source_coords=all_coords,
        target_coords=all_coords,
        dataset_source=str(RASTER_PATH),
        search_space_buffer_m=BBOX_BUFFER_M,
        neighborhood_str="r3",
    )

    skipped: list[int] = []
    stations_to_route = sorted(boxes_by_station)
    for k, st_idx in enumerate(stations_to_route, start=1):
        box_idxs = boxes_by_station[st_idx]
        print(
            f"  station {k}/{len(stations_to_route)} "
            f"(idx {st_idx}) -> {len(box_idxs)} cable boxes"
        )
        try:
            path_finder.find_route(
                source=station_coords[st_idx],
                target=[box_coords[b] for b in box_idxs],
                algorithm="delta-stepping",
                simplify=SIMPLIFY,
            )
        except NoPathFoundError:
            print(
                f"    batch failed for station {st_idx}; "
                f"retrying boxes individually."
            )
            for b in box_idxs:
                try:
                    path_finder.find_route(
                        source=station_coords[st_idx],
                        target=box_coords[b],
                        algorithm="delta-stepping",
                        simplify=SIMPLIFY,
                    )
                except NoPathFoundError:
                    print(
                        f"    skipped box idx {b} "
                        f"(bus {boxes.bus.iloc[b]}) — no path found."
                    )
                    skipped.append(b)

    if skipped:
        print(f"Skipped {len(skipped)} cable box(es) with no feasible path:")
        for b in skipped:
            print(f"  box idx {b} (bus {boxes.bus.iloc[b]})")

    return path_finder


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading pandapower grid from {NET_JSON}")
    net = pp.from_json(str(NET_JSON))

    stations = extract_mv_stations(net)
    print(f"Found {len(stations)} MV/transformer stations with geodata")

    boxes = extract_cable_boxes(net)
    print(f"Found {len(boxes)} cable boxes with geodata")

    if not RASTER_PATH.exists():
        # Build over the union of station + box extents so the raster covers
        # every routing endpoint.
        minx = min(stations.total_bounds[0], boxes.total_bounds[0])
        miny = min(stations.total_bounds[1], boxes.total_bounds[1])
        maxx = max(stations.total_bounds[2], boxes.total_bounds[2])
        maxy = max(stations.total_bounds[3], boxes.total_bounds[3])
        bbox = (
            minx - BBOX_BUFFER_M, miny - BBOX_BUFFER_M,
            maxx + BBOX_BUFFER_M, maxy + BBOX_BUFFER_M,
        )
        build_cost_raster(bbox)
    else:
        print(f"Re-using existing cost raster: {RASTER_PATH}")

    path_finder = route_boxes_to_nearest_station(boxes, stations)

    print(f"Saving {len(path_finder.paths)} paths -> {ROUTES_PATH}")
    path_finder.save_paths(str(ROUTES_PATH))
    print("Done.")


if __name__ == "__main__":
    main()
