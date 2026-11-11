"""Smoke test for public-PyPI pyorps compatibility with batch_route_planning_pandapower.

Exercises every pyorps API used by the batch script, against the existing
cost raster on disk and the pandapower net — but only routes ONE station
pair, so it finishes in seconds.

The test fails fast (with a clear message) if any symbol is missing, any
keyword argument has been renamed, or PathFinder raises.

``check_api_surface()`` (symbols, keyword arguments, attributes) needs no data.
``run_single_pair()`` needs the batch script's cost raster and pandapower net
and exits with a message when they are missing.
"""

import inspect
import json
from pathlib import Path

from shapely.geometry import shape

import pyorps

# ---- 1. Import every pyorps symbol the batch script imports. ----
from pyorps import (
    CostAssumptions,
    GeoRasterizer,
    PathFinder,
    detect_feature_columns,
    initialize_geo_dataset,
)
from pyorps.core.exceptions import NoPathFoundError

# Same locations as batch_route_planning_pandapower.py, resolved from this file so the
# script works from any working directory.
ROOT = Path(__file__).resolve().parent.parent
ROUTE_PLANNING = ROOT / "output" / "example_2025_eamex" / "route_planning"
RASTER = ROUTE_PLANNING / "cost_raster_buf1m_osm2m_v2.tiff"
NET = ROOT.parent / "gis2pp" / "output" / "example_2025_eamex" / "example_2025.json"
OUT = ROUTE_PLANNING / "_pypi_compat_smoke.geojson"


# ---- 2. Verify all keyword arguments used by the batch script. ----
def assert_kwargs(callable_, *required):
    sig = inspect.signature(callable_)
    params = set(sig.parameters)
    missing = [p for p in required if p not in params]
    if missing:
        raise AssertionError(
            f"{callable_.__qualname__} is missing kwargs: {missing}\n"
            f"  available: {sorted(params)}"
        )


def check_api_surface() -> None:
    """Parts 1-3: the symbols, keyword arguments and attributes the batch script relies on."""
    print(f"pyorps version: {pyorps.__version__}")
    print(f"  CostAssumptions          -> {CostAssumptions}")
    print(f"  GeoRasterizer            -> {GeoRasterizer}")
    print(f"  PathFinder               -> {PathFinder}")
    print(f"  detect_feature_columns   -> {detect_feature_columns}")
    print(f"  initialize_geo_dataset   -> {initialize_geo_dataset}")
    print(f"  NoPathFoundError         -> {NoPathFoundError}")

    assert_kwargs(
        PathFinder.__init__,
        "source_coords", "target_coords", "dataset_source",
        "search_space_buffer_m", "neighborhood_str",
    )
    # find_route names ``simplify``: the routed LineString is simplified before the metrics are reported.
    assert_kwargs(
        PathFinder.find_route,
        "source", "target", "algorithm", "simplify",
    )
    assert_kwargs(
        GeoRasterizer.rasterize,
        "preprocessing_function", "geometry_buffer_m",
    )
    assert_kwargs(
        GeoRasterizer.modify_raster_from_dataset,
        "input_data", "cost_assumptions", "multiply", "geometry_buffer_m",
    )
    assert_kwargs(GeoRasterizer.save_raster, "save_path")
    assert_kwargs(PathFinder.save_paths, "save_file_path")
    print("API signatures OK.")

    # ---- 3. Verify GeoRasterizer attribute surface used by the batch script. ----
    for attr in ("raster", "transform", "crs", "base_dataset"):
        if not (hasattr(GeoRasterizer, attr) or attr in GeoRasterizer.__init__.__code__.co_names):
            # crs is a property; raster/transform/base_dataset are set in __init__.
            # We can't always introspect instance attrs from the class, so just
            # warn rather than fail here.
            print(f"  warn: GeoRasterizer.{attr} not found on class (may be set on instances)")


def parse_geo(g):
    if isinstance(g, str):
        return shape(json.loads(g))
    if isinstance(g, dict):
        return shape(g)
    return g


def run_single_pair() -> None:
    """Part 4: run a single routing call using the existing cost raster."""
    import pandapower as pp  # imported here: parts 1-3 work without it

    if not RASTER.exists():
        raise SystemExit(f"Cost raster missing — run the full script once first: {RASTER}")
    if not NET.exists():
        raise SystemExit(f"pandapower net missing: {NET}")

    net = pp.from_json(str(NET))
    # Pick any two MV stations whose geo is set. Use the first two ext_grid /
    # trafo HV buses that have coordinates.
    candidate_buses = list(net.ext_grid.bus.tolist())
    if len(net.trafo):
        candidate_buses += net.trafo.hv_bus.tolist()
    mv_band = net.bus.vn_kv.between(1.0, 60.0)
    candidate_buses = [b for b in candidate_buses if mv_band.loc[b]]
    coords = []
    for b in candidate_buses:
        g = parse_geo(net.bus.geo.loc[b])
        if g is not None and hasattr(g, "x"):
            coords.append((g.x, g.y))
            if len(coords) == 2:
                break

    if len(coords) < 2:
        raise SystemExit("Could not find two MV stations with coordinates.")

    source, target = coords
    print(f"Routing one pair: {source} -> {target}")

    pf = PathFinder(
        source_coords=coords,
        target_coords=coords,
        dataset_source=str(RASTER),
        search_space_buffer_m=1000.0,
        neighborhood_str="r3",
    )
    pf.find_route(
        source=source,
        target=target,
        algorithm="delta-stepping",
        simplify={"method": "douglas_peucker", "tolerance": 1.0},
    )
    print(f"PathFinder produced {len(pf.paths)} path(s).")
    pf.save_paths(str(OUT))
    print(f"Saved to {OUT}")
    print(f"ALL OK — public pyorps {pyorps.__version__} is compatible with the batch script.")


def main() -> None:
    check_api_surface()
    run_single_pair()


if __name__ == "__main__":
    main()
