"""
PYORPS webviz CLI: launch the interactive viewer from the command line.

Examples::

    # Raster + source/target, compute a route and open the viewer
    python -m pyorps.webviz --raster cost.tif --source 472000,5593400 \\
        --target 472800,5594000

    # Just browse a raster (+ optional shapes), no routing
    python -m pyorps.webviz --raster cost.tif --shapes landuse.gpkg --no-route

    # Open in a native desktop window instead of the browser
    python -m pyorps.webviz --raster cost.tif --source 472000,5593400 \\
        --target 472800,5594000 --desktop
"""
from __future__ import annotations

import argparse


def _coord(text: str) -> tuple[float, float]:
    x, y = text.split(",")
    return float(x), float(y)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pyorps.webviz",
        description="Launch the interactive PYORPS route viewer.")
    parser.add_argument("--raster", required=True, help="Cost raster (GeoTIFF).")
    parser.add_argument("--source", type=_coord, help="Source 'x,y' in raster CRS.")
    parser.add_argument("--target", type=_coord, help="Target 'x,y' in raster CRS.")
    parser.add_argument("--shapes", action="append", default=[],
                        help="Input shape file(s) to overlay (repeatable).")
    parser.add_argument("--buffer", type=float, default=None,
                        help="Search-space buffer in metres.")
    parser.add_argument("--colormap", default="viridis", help="Raster colormap.")
    parser.add_argument("--no-route", action="store_true",
                        help="Only browse the raster; do not compute a route.")
    parser.add_argument("--desktop", action="store_true",
                        help="Open a native window instead of the browser.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args(argv)

    from pyorps.webviz import RouteViewer

    do_route = not args.no_route and args.source is not None and args.target is not None

    if do_route:
        from pyorps import PathFinder
        finder = PathFinder(args.raster, source_coords=args.source,
                            target_coords=args.target,
                            search_space_buffer_m=args.buffer)
        finder.find_route()
        viewer = RouteViewer.from_path_finder(finder, colormap=args.colormap)
    else:
        viewer = RouteViewer(colormap=args.colormap)
        viewer.add_cost_raster(args.raster, colormap=args.colormap)

    for shp in args.shapes:
        viewer.add_shapes(shp, name=shp)

    print(f"Launching PYORPS viewer on http://{args.host}:{args.port} ...")
    viewer.launch(desktop=args.desktop, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
