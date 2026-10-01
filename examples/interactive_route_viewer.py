"""
Interactive route viewer (pyorps.webviz) — LEGACY, see examples/pyorps_gui.py
=============================================================================

.. deprecated::
    ``pyorps.webviz`` is deprecated in favour of the rebuilt ``pyorps.gui``
    workbench — run ``python examples/pyorps_gui.py`` instead (blank-map
    workflow with study-area drawing, cost calibration, rasterization, route
    building and non-destructive editing). This legacy script keeps working
    for one release.

Opens an interactive Dash + Leaflet map that mirrors the ``mv_oberrhein`` case
study end to end:

1. **Shapes from a WFS server** — pulls ALKIS "Tatsächliche Nutzung" (actual land
   use) polygons for the routing area straight from the LGL Baden-Württemberg
   WFS.
2. **Cost raster from cost assumptions** — rasterizes those polygons into a cost
   surface using the per-``objektname`` construction-cost assumptions
   (``GeoRasterizer`` + the same cost dict used in the case study).
3. **Multi-target routing** — routes the single PV source to its eight candidate
   points of common coupling (the real mv_oberrhein targets), yielding a
   ``PathCollection``.

The viewer then shows the cost raster on OSM/Esri/Carto basemaps, overlays the
land-use polygons (click one to read its ``objektname`` and other ALKIS
attributes), draws the routes and their source/target markers, and lets you edit
a route with live cost feedback or re-route it through dropped waypoints.

Requires the optional viz dependencies and internet access to the WFS::

    pip install "pyorps[viz]"      # or: uv pip install -e ".[viz]"

Run::

    python examples/interactive_route_viewer.py                 # browser
    python examples/interactive_route_viewer.py --desktop       # native window
    python examples/interactive_route_viewer.py --max-targets 3 # fewer routes / smaller area
    python examples/interactive_route_viewer.py --offline       # synthetic raster, no WFS
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import geopandas as gpd
from shapely.geometry import MultiPoint, Point

import pyorps
from pyorps import GeoRasterizer, PathFinder
from pyorps.webviz import RouteViewer

# --------------------------------------------------------------------------- #
# Configuration — taken verbatim from case_studies/mv_oberrhein
# --------------------------------------------------------------------------- #

WORKING_CRS = "EPSG:25832"          # ETRS89 / UTM zone 32N (the ALKIS + routing CRS)

WFS_REQUEST = {
    "url": "https://owsproxy.lgl-bw.de/owsproxy/wfs/WFS_LGL-BW_ALKIS?version=2.0.0",
    "layer": "Tatsächliche Nutzung",
}

# Per-land-use construction costs (€/m proxy). 65535 = forbidden / no-go.
COST_ASSUMPTIONS = {
    "objektname": {
        "Wohnbaufläche": 65535,
        "Industrie- und Gewerbefläche": 65535,
        "Fläche besonderer funktionaler Prägung": 65535,
        "Tagebau/Grube/Steinbruch": 65535,
        "Friedhof": 65535,
        "Halde": 65535,
        "Sumpf": 65535,
        "Flugverkehr": 65535,
        "Straßenverkehr": 178,
        "Sport-, Freizeit- und Erholungsfläche": 107,
        "Weg": 97,
        "Landwirtschaft": 285,
        "Wald": 365,
        "Fließgewässer": 155,
        "Gehölz": 365,
        "Fläche gemischter Nutzung": 107,
        "Platz": 152,
        "Unland/Vegetationslose Fläche": 92,
        "Stehendes Gewässer": 155,
        "Bahnverkehr": 415,
    }
}

# The single PV source and its eight candidate PCC targets (EPSG:25832) — the
# actual mv_oberrhein multi-target set (see case_studies/mv_oberrhein).
SOURCE_XY = (412873.7, 5362043.3)
TARGET_XY = [
    (413856.3, 5361543.8),
    (413439.3, 5361147.8),
    (412809.2, 5360104.8),
    (412102.2, 5360518.8),
    (412501.2, 5364561.8),
    (412027.2, 5364635.8),
    (409661.2, 5362818.8),
    (409629.2, 5362577.8),
]

BBOX_BUFFER_M = 400.0               # land-use area padding around source + targets
SEARCH_BUFFER_M = 400.0             # per-route search-space buffer


def _cache_dir() -> Path:
    path = Path(tempfile.gettempdir()) / "pyorps_viz_example"
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_viewer(max_targets: int | None = None, *, add_shapes: bool = True,
                 rebuild: bool = False) -> tuple[RouteViewer, PathFinder]:
    """Build the MV-Oberrhein viewer (WFS shapes -> cost raster -> routes)."""
    targets_xy = TARGET_XY[:max_targets] if max_targets else TARGET_XY
    source = gpd.GeoDataFrame(
        {"id": [0]}, geometry=[Point(*SOURCE_XY)], crs=WORKING_CRS)
    targets = gpd.GeoDataFrame(
        {"id": list(range(len(targets_xy)))},
        geometry=[Point(*xy) for xy in targets_xy], crs=WORKING_CRS)

    # bbox covering source + targets + padding, as a GeoDataFrame in WORKING_CRS.
    all_pts = MultiPoint([Point(*SOURCE_XY)] + [Point(*xy) for xy in targets_xy])
    bbox_poly = all_pts.convex_hull.buffer(BBOX_BUFFER_M)
    bbox_gdf = gpd.GeoDataFrame(index=[0], geometry=[bbox_poly], crs=WORKING_CRS)

    # 1) Pull ALKIS land use from the WFS for exactly this area.
    print("Fetching ALKIS land use from the LGL-BW WFS ...")
    gis_dataset = pyorps.initialize_geo_dataset(WFS_REQUEST, bbox=bbox_gdf)
    gis_dataset.load_data()
    print(f"  {len(gis_dataset.data):,} land-use polygons")

    # 2) Rasterize into a cost surface using the per-objektname cost assumptions.
    raster_path = _cache_dir() / f"mv_oberrhein_cost_{len(targets_xy)}t.tiff"
    if rebuild or not raster_path.exists():
        print("Rasterizing cost surface from cost assumptions ...")
        GeoRasterizer(gis_dataset, COST_ASSUMPTIONS, bbox_gdf).rasterize(
            save_path=str(raster_path))
    else:
        print(f"Re-using cached cost raster: {raster_path}")

    # 3) Route the source to every target (single-source, multi-target).
    print(f"Routing source -> {len(targets_xy)} targets ...")
    finder = PathFinder(str(raster_path), source_coords=source,
                        target_coords=targets, neighborhood_str="r2",
                        search_space_buffer_m=SEARCH_BUFFER_M,
                        ignore_max_cost=False)
    finder.find_route()
    print(f"  found {len(finder.paths)} route(s)")

    # 4) Build the viewer (raster + routes + endpoints + live cost + re-route)
    #    and add the land-use polygons as a clickable overlay.
    viewer = RouteViewer.from_path_finder(finder)
    if add_shapes:
        viewer.add_shapes(gis_dataset.data, name="ALKIS land use (objektname)",
                          color="#7f7f7f")
    return viewer, finder


def _synthetic_viewer() -> RouteViewer:
    """Offline fallback: a synthetic gradient raster + one route (no WFS)."""
    from pyorps.raster.handler import create_test_tiff

    tif = _cache_dir() / "synthetic_cost.tif"
    create_test_tiff(str(tif), width=300, height=300, pattern="gradient",
                     crs="EPSG:32632")
    finder = PathFinder(str(tif), source_coords=(500030.0, 5599970.0),
                        target_coords=(500270.0, 5599730.0),
                        search_space_buffer_m=120)
    finder.find_route()
    viewer = RouteViewer.from_path_finder(finder)
    viewer.add_shapes(
        gpd.GeoDataFrame({"name": ["search buffer"]},
                         geometry=[finder.raster_handler.buffer_geometry],
                         crs=finder.dataset.crs),
        name="Search buffer", color="#00aaff")
    return viewer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-targets", type=int, default=None,
                        help="Route only the first N targets (smaller/faster area).")
    parser.add_argument("--no-shapes", action="store_true",
                        help="Do not overlay the land-use polygons.")
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild the cost raster even if cached.")
    parser.add_argument("--offline", action="store_true",
                        help="Skip the WFS and use a synthetic raster instead.")
    parser.add_argument("--desktop", action="store_true",
                        help="Open a native desktop window.")
    args = parser.parse_args()

    if args.offline:
        print("Offline mode — building a synthetic example (no WFS).")
        viewer = _synthetic_viewer()
    else:
        viewer, _ = build_viewer(max_targets=args.max_targets,
                                 add_shapes=not args.no_shapes,
                                 rebuild=args.rebuild)

    # In the sidebar try both edit modes:
    #   * "Live cost feedback": drag route vertices — length/cost update live.
    #   * "Re-route through waypoints": drop markers, press Re-route.
    print("Opening viewer — Ctrl+C to stop.")
    viewer.launch(desktop=args.desktop)


if __name__ == "__main__":
    main()
