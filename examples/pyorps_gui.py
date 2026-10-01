"""
PYORPS GUI — the interactive route-planning workbench (pyorps.gui)
==================================================================

Starts the rebuilt GUI on a blank map (this replaces the deprecated
``pyorps.webviz`` RouteViewer). Everything happens in the browser:

1. **Data tab** — draw a study area (rectangle/polygon), then load vector
   data from local files or WFS servers (Hessen/BW ALKIS presets included),
   clipped to the area. Save/open whole projects from here too.
2. **Cost tab** — pick the feature column(s) (e.g. ``nutzart`` + ``bez``),
   seed the editable cost table (notebook defaults for ALKIS), add modifier
   layers (water protection multipliers, soil factors, nature = forbidden),
   choose preprocessing (street-type buffers).
3. **Raster tab** — set resolution/dtype/fill (with a live size estimate),
   build the cost raster, or load a pre-computed ``.tif``.
4. **Routes tab** — place sources/targets/waypoints by clicking the map,
   pick algorithm + CPU/GPU, run; waypoints are routed as a chain
   source → w1 → … → target. Optional: constrained overhead-line routing
   with an editable tower profile (110/220/380 kV presets).
5. **Edit tab** — select a route, move its source/target or add waypoints by
   clicking; every edit creates a NEW route with lineage back to its parent
   (originals are never overwritten). Export any variant.

Requires the ``gui`` extra::

    pip install "pyorps[gui]"      # or: uv pip install -e ".[gui]"

Run::

    python examples/pyorps_gui.py                # browser (http://127.0.0.1:8050)
    python examples/pyorps_gui.py --desktop      # native window (pywebview)
    python examples/pyorps_gui.py --raster c.tif # pre-load a cost raster
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--desktop", action="store_true",
                        help="open in a desktop window (pywebview)")
    parser.add_argument("--raster", default=None,
                        help="optional cost raster (.tif) to pre-load")
    parser.add_argument("--routes", default=None,
                        help="optional routes file (.geojson/.gpkg) to "
                             "pre-load")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    from pyorps.gui import ProjectState, launch

    state = ProjectState()
    if args.raster:
        from pyorps.gui.callbacks.raster import add_raster_layer

        layer, notices = add_raster_layer(state, args.raster, [])
        if layer is None:
            for notice in notices:
                print(f"[{notice['severity']}] {notice['title']}: "
                      f"{notice.get('meaning', '')}")
        else:
            print(f"pre-loaded raster: {args.raster}")
    if args.routes:
        import geopandas as gpd

        from pyorps.gui.services import geo

        gdf = gpd.read_file(args.routes)
        for i, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.geom_type != "LineString":
                continue
            name = str(row.get("name") or f"Route {i + 1}")
            feature = geo.linestring_to_wgs84_feature(
                geom, gdf.crs, properties={"name": name})
            state.add_layer(
                name, "route",
                gdf=gpd.GeoDataFrame([{"name": name}], geometry=[geom],
                                     crs=gdf.crs),
                crs=gdf.crs,
                geojson={"type": "FeatureCollection",
                         "features": [feature]},
                meta={"origin": "imported", "params": {},
                      "control_points": [list(geom.coords[0]),
                                         list(geom.coords[-1])]})
        print(f"pre-loaded routes: {args.routes}")

    launch(state, port=args.port, desktop=args.desktop)


if __name__ == "__main__":
    main()
