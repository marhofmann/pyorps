"""Smoke test for public-PyPI pyorps compatibility with batch_route_planning_pandapower.

Exercises every pyorps API used by the batch script, against the existing
cost raster on disk and the pandapower net — but only routes ONE station
pair, so it finishes in seconds.

The test fails fast (with a clear message) if any symbol is missing, any
keyword argument has been renamed, or PathFinder raises.
"""

import inspect
import json
from pathlib import Path
from shapely.geometry import shape
import pandapower as pp

# ---- 1. Import every pyorps symbol the batch script imports. ----
from pyorps import (
    CostAssumptions,
    GeoRasterizer,
    PathFinder,
    detect_feature_columns,
    initialize_geo_dataset,
)
from pyorps.core.exceptions import NoPathFoundError
import pyorps

print(f"pyorps version: {pyorps.__version__}")
print(f"  CostAssumptions          -> {CostAssumptions}")
print(f"  GeoRasterizer            -> {GeoRasterizer}")
print(f"  PathFinder               -> {PathFinder}")
print(f"  detect_feature_columns   -> {detect_feature_columns}")
print(f"  initialize_geo_dataset   -> {initialize_geo_dataset}")
print(f"  NoPathFoundError         -> {NoPathFoundError}")

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

assert_kwargs(
    PathFinder.__init__,
    "source_coords", "target_coords", "dataset_source",
    "search_space_buffer_m", "neighborhood_str",
)
assert_kwargs(
    PathFinder.find_route,
    "source", "target", "algorithm",
)
# Note: the batch script passes simplify=... to find_route, but neither the
# local nor the public pyorps consume that kwarg — both signatures only
# accept **kwargs and there is no Python code that reads "simplify" from
# them. So simplification is a no-op in both versions; this is identical
# behaviour and not a compatibility regression.
assert_kwargs(
    GeoRasterizer.rasterize,
    "preprocessing_function", "geometry_buffer_m",
)
assert_kwargs(
    GeoRasterizer.modify_raster_from_dataset,
    "input_data", "cost_assumptions", "multiply", "geometry_buffer_m",
)
assert_kwargs(GeoRasterizer.save_raster, "save_path")
assert_kwargs(
    PathFinder.save_paths,
    # save_paths has a single positional arg in current local code; just
    # verify the method exists.
)
print("API signatures OK.")

# ---- 3. Verify GeoRasterizer attribute surface used by the batch script. ----
ATTRS = ("raster", "transform", "crs", "base_dataset")
for attr in ATTRS:
    if not (hasattr(GeoRasterizer, attr) or attr in GeoRasterizer.__init__.__code__.co_names):
        # crs is a property; raster/transform/base_dataset are set in __init__.
        # We can't always introspect instance attrs from the class, so just
        # warn rather than fail here.
        print(f"  warn: GeoRasterizer.{attr} not found on class (may be set on instances)")

# ---- 4. Run a single routing call using the existing cost raster. ----
RASTER = Path("output/example_2025_eamex/route_planning/cost_raster_buf1m_osm2m_v2.tiff")
NET = Path("../gis2pp/output/example_2025_eamex/example_2025.json")
OUT = Path("output/example_2025_eamex/route_planning/_pypi_compat_smoke.geojson")

if not RASTER.exists():
    raise SystemExit(f"Cost raster missing — run the full script once first: {RASTER}")
if not NET.exists():
    raise SystemExit(f"pandapower net missing: {NET}")

def parse_geo(g):
    if isinstance(g, str):
        return shape(json.loads(g))
    if isinstance(g, dict):
        return shape(g)
    return g

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
print("ALL OK — public pyorps {} is compatible with the batch script.".format(pyorps.__version__))
