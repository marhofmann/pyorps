"""Preprocess least-cost MV connection candidates for large (> 1 MVA) elements.

For each large-scale load / generator that must connect at the 20 kV level, this
script routes ONE least-cost underground-cable candidate to each of the *k*
nearest 20 kV stations, so a downstream grid planner can compare the candidates
and pick the cheapest feasible point of common coupling (PCC). It runs ONCE as
preprocessing, before planning, and caches its result to disk.

It consumes the two GeoPackage layers exported upstream by gis2pp:

* ``mv_connection_requests`` -- one row per > 1 MVA element (the PHYSICAL site
  location, *not* a bus location), with the columns documented in
  ``REQUIRED_MANIFEST_COLUMNS``.
* ``mv_stations`` -- one row per 20 kV bus (the PCC identifier), with the columns
  documented in ``REQUIRED_STATION_COLUMNS``.

This is application glue ON TOP of :class:`pyorps.PathFinder`; the pyorps core
library is not modified. The PathFinder wiring mirrors powergridforge's proven
``_run_pathfinder`` (one source + many targets in a single one-to-many call,
``search_space_buffer_m=None`` auto-estimate, ``ignore_max_cost=True`` so nodes
inside max-cost cells snap to the nearest reachable cell).

Cost data source (chosen by CLI / kwargs, in priority order):

1. ``--cost-raster PATH``  : a pre-rasterized cost GeoTIFF (CRS embedded). The
   routing CRS is read from the raster. Mirrors powergridforge's ``raster_path``.
2. ``--cost-vector PATH --cost-assumptions PATH`` : a vector dataset plus cost
   assumptions (CSV / JSON / Excel / dict) that pyorps rasterizes internally.
   The routing CRS is the manifest CRS; pyorps reprojects + rasterizes per
   element bbox.
3. Neither given -> a UNIFORM-COST raster auto-generated over the bounding box of
   (elements + their candidate stations) in the manifest CRS. Logged loudly --
   this exists only so the script is runnable end-to-end for testing.

Run as a module for the CLI, or import :func:`preprocess_mv_connections`.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from scipy.spatial import cKDTree

from pyorps import PathFinder
from pyorps.core.exceptions import NoPathFoundError

log = logging.getLogger("pyorps.preprocess_mv_connections")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DEFAULT_MANIFEST_LAYER = "mv_connection_requests"
DEFAULT_STATIONS_LAYER = "mv_stations"
DEFAULT_OUT_DIR = Path("output/mv_connection_candidates")
CANDIDATES_LAYER = "candidates"

# Columns the upstream gis2pp export guarantees. We fail loudly if any is absent
# rather than silently routing with wrong assumptions.
REQUIRED_MANIFEST_COLUMNS = (
    "element_id", "kind", "tech", "s_mva", "default_pcc_bus", "year",
    "scenario_id",
)
REQUIRED_STATION_COLUMNS = ("bus_id",)

# Columns of the failures CSV, pinned so the file always carries a header even
# when there are zero failures (an empty header-less CSV is not parseable).
FAILURE_COLUMNS = (
    "element_id", "kind", "tech", "s_mva", "year", "scenario_id", "reason",
)

# Uniform-cost fallback raster parameters (testing only).
UNIFORM_COST_VALUE = 1            # cost per cell; far below the 65535 sentinel
UNIFORM_RESOLUTION_M = 10.0       # cell size of the generated raster
UNIFORM_MARGIN_M = 1000.0         # margin around the element/station bbox

# Margin (m) added around each element + its candidate stations when pyorps has
# to rasterize a vector cost source for that element.
COST_VECTOR_MARGIN_M = 1000.0

# Hard ceiling mirroring pyorps' own uint32 node-index limit, used to refuse an
# absurdly large uniform-fallback raster with a clear message instead of a
# cryptic pyorps error deep in graph construction.
MAX_RASTER_CELLS = np.iinfo(np.uint32).max


# --------------------------------------------------------------------------- #
# I/O + validation
# --------------------------------------------------------------------------- #

def _read_layer(path: str | Path, layer: str, role: str) -> gpd.GeoDataFrame:
    """Read ``layer`` from a GeoPackage, falling back to the default layer.

    Some exports write the layer under its documented name; others write a
    single unnamed layer. We try the named layer first and fall back so the
    script works for both, logging which path was taken.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{role} file not found: {path}")
    try:
        gdf = gpd.read_file(str(path), layer=layer)
        log.info("Loaded %s layer '%s' from %s (%d rows)", role, layer, path, len(gdf))
    except Exception:  # layer not present -> read the single/default layer
        gdf = gpd.read_file(str(path))
        log.info(
            "Loaded %s from %s (layer '%s' not found; used default layer, %d rows)",
            role, path, layer, len(gdf),
        )
    if gdf.crs is None:
        raise ValueError(f"{role} layer has no CRS defined: {path}")
    return gdf


def _validate_columns(gdf: gpd.GeoDataFrame, required, role: str) -> None:
    missing = [c for c in required if c not in gdf.columns]
    if missing:
        raise ValueError(
            f"{role} layer is missing required column(s) {missing}. "
            f"Present columns: {list(gdf.columns)}"
        )
    if gdf.geometry.isna().any():
        n = int(gdf.geometry.isna().sum())
        raise ValueError(f"{role} layer has {n} row(s) with no geometry.")


# --------------------------------------------------------------------------- #
# Cost source resolution
# --------------------------------------------------------------------------- #

def _generate_uniform_raster(
    bbox: tuple[float, float, float, float],
    crs,
    resolution_m: float,
    out_path: Path,
) -> Path:
    """Write a uniform-cost GeoTIFF covering ``bbox`` (+ no extra margin) to disk."""
    minx, miny, maxx, maxy = bbox
    width = max(1, math.ceil((maxx - minx) / resolution_m))
    height = max(1, math.ceil((maxy - miny) / resolution_m))
    n_cells = width * height
    if n_cells > MAX_RASTER_CELLS:
        raise ValueError(
            f"Uniform fallback raster would be {height}x{width} = {n_cells:,} "
            f"cells, exceeding the {MAX_RASTER_CELLS:,} uint32 limit. Provide a "
            f"--cost-raster / --cost-vector instead, or a coarser resolution."
        )
    transform = from_origin(minx, maxy, resolution_m, resolution_m)
    data = np.full((height, width), UNIFORM_COST_VALUE, dtype=np.uint16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        str(out_path), "w", driver="GTiff", height=height, width=width,
        count=1, dtype="uint16", crs=crs, transform=transform, compress="deflate",
    ) as dst:
        dst.write(data, 1)
    log.warning(
        "==================================================================\n"
        "  USING UNIFORM-COST FALLBACK RASTER (testing only!)\n"
        "  No --cost-raster / --cost-vector was provided. Routes minimise\n"
        "  geometric length only; they carry NO real terrain cost.\n"
        "  Wrote %dx%d uniform raster (cost=%d) -> %s\n"
        "==================================================================",
        height, width, UNIFORM_COST_VALUE, out_path,
    )
    return out_path


def _resolve_cost_source(
    manifest: gpd.GeoDataFrame,
    stations: gpd.GeoDataFrame,
    cost_raster: str | Path | None,
    cost_vector: str | Path | None,
    cost_assumptions: str | Path | dict | None,
    out_dir: Path,
    uniform_resolution_m: float,
    uniform_margin_m: float,
):
    """Determine the routing CRS and the PathFinder cost-source kwargs.

    Returns ``(routing_crs, pf_source_kwargs, mode)`` where ``pf_source_kwargs``
    is a dict of kwargs forwarded to ``PathFinder`` that pins the cost source
    (``dataset_source`` and, for vector data, ``cost_assumptions`` + ``crs``).
    ``crs``/``bbox`` are deliberately omitted for raster file paths, which carry
    their own metadata (mirrors powergridforge).
    """
    if cost_raster is not None:
        with rasterio.open(str(cost_raster)) as src:
            routing_crs = src.crs
        if routing_crs is None:
            raise ValueError(f"Cost raster has no CRS: {cost_raster}")
        log.info("Cost source: pre-rasterized GeoTIFF %s (CRS %s)", cost_raster, routing_crs)
        return routing_crs, {"dataset_source": str(cost_raster)}, "raster"

    if cost_vector is not None:
        if cost_assumptions is None:
            raise ValueError("--cost-vector requires --cost-assumptions.")
        # Rasterize the vector into the manifest CRS; pyorps reprojects + crops
        # per element bbox. crs is passed because the source is NOT a raster file.
        routing_crs = manifest.crs
        log.info(
            "Cost source: vector %s + cost assumptions %s (routing CRS %s)",
            cost_vector, cost_assumptions, routing_crs,
        )
        # pyorps' CostAssumptions accepts a str path or a dict, not a Path object.
        ca = str(cost_assumptions) if isinstance(cost_assumptions, Path) else cost_assumptions
        kwargs = {
            "dataset_source": str(cost_vector),
            "cost_assumptions": ca,
            "crs": str(routing_crs),
        }
        return routing_crs, kwargs, "vector"

    # Uniform fallback over the bbox of (elements + stations) in the manifest CRS.
    routing_crs = manifest.crs
    all_pts = pd.concat([manifest.geometry, stations.to_crs(routing_crs).geometry])
    minx, miny, maxx, maxy = all_pts.total_bounds
    bbox = (
        minx - uniform_margin_m, miny - uniform_margin_m,
        maxx + uniform_margin_m, maxy + uniform_margin_m,
    )
    raster_path = _generate_uniform_raster(
        bbox, routing_crs, uniform_resolution_m, out_dir / "uniform_cost_raster.tiff"
    )
    return routing_crs, {"dataset_source": str(raster_path)}, "uniform"


# --------------------------------------------------------------------------- #
# Candidate selection + path -> station assignment
# --------------------------------------------------------------------------- #

def _k_nearest(element_xy, station_xy: np.ndarray, k: int) -> np.ndarray:
    """Return the indices of the ``k`` Euclidean-nearest stations (fewer if k>N)."""
    n = len(station_xy)
    kk = min(k, n)
    tree = cKDTree(station_xy)
    _, idx = tree.query(element_xy, k=kk)
    return np.atleast_1d(np.asarray(idx, dtype=int))


def _assign_paths_to_stations(paths, cand_xy: np.ndarray) -> dict[int, int]:
    """Greedily map each routed Path to its nearest *un-claimed* candidate station.

    Returned paths are matched to stations by their (snapped) END coordinate,
    NOT by position: a one-to-many PathFinder call silently drops unreachable
    targets, so index alignment would mis-attribute the survivors. Greedy
    nearest-unclaimed assignment is robust to that and to two stations sharing a
    pixel. Returns ``{path_index -> local_candidate_index}``.
    """
    if not paths:
        return {}
    targets = np.array([[p.target[0], p.target[1]] for p in paths], dtype=float)
    dist = np.linalg.norm(targets[:, None, :] - cand_xy[None, :, :], axis=2)
    order = np.dstack(np.unravel_index(np.argsort(dist, axis=None), dist.shape))[0]
    assignment: dict[int, int] = {}
    claimed: set[int] = set()
    for pi, si in order:
        pi, si = int(pi), int(si)
        if pi in assignment or si in claimed:
            continue
        assignment[pi] = si
        claimed.add(si)
        if len(assignment) == len(paths):
            break
    return assignment


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #

def _route_one_element(
    row, element_xy, stations_xy, stations_bus, stations_name, default_pcc_bus,
    pf_source_kwargs, mode, k, neighborhood, graph_api, algorithm,
):
    """Route one element to its k nearest stations; return (records, warnings)."""
    cand_idx = _k_nearest(element_xy, stations_xy, k)
    if len(cand_idx) < k:
        log.info(
            "Element %s: only %d station(s) available (k=%d) -> using all.",
            row["element_id"], len(cand_idx), k,
        )
    cand_xy = stations_xy[cand_idx]
    cand_bus = stations_bus[cand_idx]

    pf_kwargs = dict(
        source_coords=tuple(element_xy),
        target_coords=[tuple(xy) for xy in cand_xy],
        search_space_buffer_m=None,        # pyorps auto-estimates
        graph_api=graph_api,
        neighborhood_str=neighborhood,
        ignore_max_cost=True,
        **pf_source_kwargs,
    )
    # Vector cost sources are rasterized per element bbox (raster files are not).
    if mode == "vector":
        pts = np.vstack([element_xy, cand_xy])
        minx, miny = pts.min(axis=0) - COST_VECTOR_MARGIN_M
        maxx, maxy = pts.max(axis=0) + COST_VECTOR_MARGIN_M
        pf_kwargs["bbox"] = (float(minx), float(miny), float(maxx), float(maxy))

    pf = PathFinder(**pf_kwargs)
    # One single-source shortest-path wave reaches all k targets at once
    # (pairwise=False default). delta-stepping is the parallel OpenMP variant and
    # builds an explicit `!= 65535` exclude-mask, so it both runs faster on large
    # rasters and avoids the cython-dijkstra wall-with-gap impassable-cell bug.
    result = pf.find_route(algorithm=algorithm, calculate_metrics=True)

    # find_route returns a single Path or a PathCollection (one source, many
    # targets). Detect via the Path-only `total_length` attribute.
    if hasattr(result, "__iter__") and not hasattr(result, "total_length"):
        paths = list(result)
    else:
        paths = [result]

    assignment = _assign_paths_to_stations(paths, cand_xy)

    records = []
    for pi, si in assignment.items():
        p = paths[pi]
        bus_id = int(cand_bus[si])
        records.append({
            "element_id": row["element_id"],
            "kind": row.get("kind"),
            "tech": row.get("tech"),
            "s_mva": float(row["s_mva"]) if pd.notna(row.get("s_mva")) else None,
            "year": int(row["year"]) if pd.notna(row.get("year")) else None,
            "scenario_id": row.get("scenario_id"),
            "target_bus_id": bus_id,
            "target_name": (None if stations_name is None
                            else stations_name[cand_idx[si]]),
            "euclid_m": float(np.linalg.norm(element_xy - cand_xy[si])),
            "route_length_m": (float(p.total_length)
                               if p.total_length is not None else None),
            "route_cost": (float(p.total_cost)
                           if p.total_cost is not None else None),
            "length_by_category_json": json.dumps(
                {str(c): v for c, v in (p.length_by_category or {}).items()}
            ),
            "is_default_pcc": bool(bus_id == default_pcc_bus),
            "default_pcc_bus": (int(default_pcc_bus)
                                if pd.notna(default_pcc_bus) else None),
            "geometry": p.path_geometry,
        })

    # Stations among the k nearest that returned no route: warn, never drop silently.
    matched_local = set(assignment.values())
    warnings = []
    for li in range(len(cand_idx)):
        if li not in matched_local:
            bus_id = int(cand_bus[li])
            log.warning(
                "Element %s: no route to station bus_id=%d (unreachable target).",
                row["element_id"], bus_id,
            )
            warnings.append(bus_id)

    # Rank by route cost ascending (None costs sort last); rank 1 = cheapest.
    records.sort(key=lambda r: (r["route_cost"] is None,
                                r["route_cost"] if r["route_cost"] is not None else 0.0))
    for rank, rec in enumerate(records, start=1):
        rec["rank"] = rank

    return records, warnings


def preprocess_mv_connections(
    manifest_path: str | Path,
    stations_path: str | Path,
    *,
    cost_raster: str | Path | None = None,
    cost_vector: str | Path | None = None,
    cost_assumptions: str | Path | dict | None = None,
    k: int = 4,
    neighborhood: str = "r2",
    graph_api: str = "cython",
    algorithm: str = "delta-stepping",
    out_dir: str | Path = DEFAULT_OUT_DIR,
    force: bool = False,
    manifest_layer: str = DEFAULT_MANIFEST_LAYER,
    stations_layer: str = DEFAULT_STATIONS_LAYER,
    uniform_resolution_m: float = UNIFORM_RESOLUTION_M,
    uniform_margin_m: float = UNIFORM_MARGIN_M,
) -> Path:
    """Preprocess least-cost MV connection candidates and cache them to disk.

    Returns the path to the written candidates GeoPackage. Idempotent: if that
    file already exists and ``force`` is False, recomputation is skipped.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = _read_layer(manifest_path, manifest_layer, "manifest")
    stations = _read_layer(stations_path, stations_layer, "stations")
    _validate_columns(manifest, REQUIRED_MANIFEST_COLUMNS, "manifest")
    _validate_columns(stations, REQUIRED_STATION_COLUMNS, "stations")

    # Output naming from the scenario/year carried by the manifest.
    scen_label = _single_label(manifest["scenario_id"], "scenario_id")
    year_label = _single_label(manifest["year"], "year")
    stem = f"mv_connection_candidates_{scen_label}_{year_label}"
    gpkg_path = out_dir / f"{stem}.gpkg"
    csv_path = out_dir / f"{stem}.csv"
    failures_path = out_dir / f"{stem}_failures.csv"

    if gpkg_path.exists() and not force:
        log.info("Output exists, skipping (pass force=True to recompute): %s", gpkg_path)
        return gpkg_path

    # Resolve the cost source + routing CRS, then project every coordinate into it.
    routing_crs, pf_source_kwargs, mode = _resolve_cost_source(
        manifest, stations, cost_raster, cost_vector, cost_assumptions,
        out_dir, uniform_resolution_m, uniform_margin_m,
    )
    manifest_r = manifest.to_crs(routing_crs)
    stations_r = stations.to_crs(routing_crs)
    log.info(
        "Routing CRS=%s | %d element(s), %d station(s) | k=%d, neighborhood=%s, "
        "graph_api=%s, algorithm=%s, mode=%s",
        routing_crs, len(manifest_r), len(stations_r), k, neighborhood, graph_api,
        algorithm, mode,
    )

    stations_xy = np.array([[p.x, p.y] for p in stations_r.geometry], dtype=float)
    stations_bus = stations_r["bus_id"].to_numpy()
    stations_name = (stations_r["name"].tolist() if "name" in stations_r.columns
                     else None)

    all_records: list[dict] = []
    failures: list[dict] = []
    n_dropped_targets = 0

    for _, row in manifest_r.iterrows():
        element_xy = np.array([row.geometry.x, row.geometry.y], dtype=float)
        try:
            records, dropped = _route_one_element(
                row, element_xy, stations_xy, stations_bus, stations_name,
                row["default_pcc_bus"], pf_source_kwargs, mode,
                k, neighborhood, graph_api, algorithm,
            )
            n_dropped_targets += len(dropped)
        except NoPathFoundError as exc:
            log.warning("Element %s: NO ROUTE FOUND to any of its %d nearest "
                        "stations (%s).", row["element_id"], k, exc)
            records = []
            dropped = []
        except Exception as exc:  # one element must never abort the batch
            log.warning("Element %s: routing failed (%s: %s).",
                        row["element_id"], type(exc).__name__, exc)
            records = []
            dropped = []

        if records:
            all_records.extend(records)
        else:
            failures.append({
                "element_id": row["element_id"],
                "kind": row.get("kind"),
                "tech": row.get("tech"),
                "s_mva": float(row["s_mva"]) if pd.notna(row.get("s_mva")) else None,
                "year": int(row["year"]) if pd.notna(row.get("year")) else None,
                "scenario_id": row.get("scenario_id"),
                "reason": "no feasible route to any of the k nearest stations",
            })

    # --- Write outputs ----------------------------------------------------- #
    if all_records:
        candidates = gpd.GeoDataFrame(all_records, geometry="geometry", crs=routing_crs)
        candidates.to_file(str(gpkg_path), layer=CANDIDATES_LAYER, driver="GPKG")
        candidates.drop(columns="geometry").to_csv(csv_path, index=False)
        log.info("Wrote %d candidate route(s) -> %s (+ %s)",
                 len(candidates), gpkg_path, csv_path.name)
    else:
        # No candidates at all: still emit an empty gpkg so callers can rely on it.
        empty = gpd.GeoDataFrame(
            {"element_id": [], "target_bus_id": []}, geometry=[], crs=routing_crs
        )
        empty.to_file(str(gpkg_path), layer=CANDIDATES_LAYER, driver="GPKG")
        log.warning("No candidate routes were produced for any element.")

    pd.DataFrame(failures, columns=list(FAILURE_COLUMNS)).to_csv(
        failures_path, index=False
    )

    # --- Summary ----------------------------------------------------------- #
    log.info(
        "Done. elements=%d | candidates=%d | failed elements=%d | dropped targets=%d",
        len(manifest_r), len(all_records), len(failures), n_dropped_targets,
    )
    if failures:
        log.info("Failed elements (no route): %s",
                 ", ".join(str(f["element_id"]) for f in failures))
    return gpkg_path


def _single_label(series: pd.Series, name: str) -> str:
    """Return the single unique value of ``series`` as a label, or 'multi'."""
    uniq = pd.unique(series.dropna())
    if len(uniq) == 1:
        return str(uniq[0])
    log.warning("Manifest mixes %d distinct %s values %s -> using 'multi' in "
                "output filename.", len(uniq), name, list(uniq))
    return "multi"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifest", required=True,
                   help="GeoPackage with the mv_connection_requests layer.")
    p.add_argument("--stations", required=True,
                   help="GeoPackage with the mv_stations layer.")
    p.add_argument("--cost-raster", default=None,
                   help="Pre-rasterized cost GeoTIFF (CRS embedded).")
    p.add_argument("--cost-vector", default=None,
                   help="Vector cost dataset (requires --cost-assumptions).")
    p.add_argument("--cost-assumptions", default=None,
                   help="Cost assumptions CSV/JSON/Excel for --cost-vector.")
    p.add_argument("--k", type=int, default=4,
                   help="Number of nearest stations to route to (default: 4).")
    p.add_argument("--neighborhood", default="r2",
                   help="pyorps neighborhood string (default: r2).")
    p.add_argument("--graph-api", default="cython",
                   help="pyorps graph backend (default: cython).")
    p.add_argument("--algorithm", default="delta-stepping",
                   help="Shortest-path algorithm (default: delta-stepping; "
                        "e.g. dijkstra).")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                   help="Output directory for the candidate artifacts.")
    p.add_argument("--manifest-layer", default=DEFAULT_MANIFEST_LAYER)
    p.add_argument("--stations-layer", default=DEFAULT_STATIONS_LAYER)
    p.add_argument("--force", action="store_true",
                   help="Recompute even if the output gpkg already exists.")
    return p


def main(argv=None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    out = preprocess_mv_connections(
        manifest_path=args.manifest,
        stations_path=args.stations,
        cost_raster=args.cost_raster,
        cost_vector=args.cost_vector,
        cost_assumptions=args.cost_assumptions,
        k=args.k,
        neighborhood=args.neighborhood,
        graph_api=args.graph_api,
        algorithm=args.algorithm,
        out_dir=args.out_dir,
        force=args.force,
        manifest_layer=args.manifest_layer,
        stations_layer=args.stations_layer,
    )
    print(out)


if __name__ == "__main__":
    main()
