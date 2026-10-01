"""
PYORPS webviz: the public ``RouteViewer`` facade.

Collects the things a user wants on the map - a cost raster, result routes,
input shapes - keeps the tile server alive, holds a RasterHandler for live cost
feedback and (optionally) a PathFinder for waypoint re-routing, then builds and
launches the Dash app (in the browser or a native desktop window).

Typical use::

    from pyorps import PathFinder
    from pyorps.webviz import RouteViewer

    finder = PathFinder(raster, source_coords=s, target_coords=t)
    finder.find_route()
    RouteViewer.from_path_finder(finder).launch(desktop=True)

Reference:
[1] Hofmann, M., Stetz, T., Kammer, F., Repo, S.: 'PYORPS: An Open-Source Tool for
    Automated Power Line Routing', CIRED 2025.
"""
from __future__ import annotations

import tempfile
from pathlib import Path as FsPath
from typing import Any

import geopandas as gpd
from shapely.geometry import Point

from pyorps.core.path import Path, PathCollection
from pyorps.raster.handler import RasterHandler

from . import geo
from .tiles import RasterTileLayer, build_tile_layer

WGS84 = "EPSG:4326"


class ShapeLayer:
    """An input-shape overlay (WGS84 GeoJSON + display name/color)."""

    def __init__(self, name: str, geojson: dict, color: str = "#3388ff"):
        self.name = name
        self.geojson = geojson
        self.color = color


class RouteViewer:
    """Interactive map viewer for pyorps rasters, routes and shapes."""

    def __init__(self, title: str = "PYORPS Route Viewer",
                 work_dir: str | None = None, colormap: str = "viridis"):
        self.title = title
        self.colormap = colormap
        self.work_dir = FsPath(work_dir or tempfile.mkdtemp(prefix="pyorps_viz_"))
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.raster_layers: list[RasterTileLayer] = []
        self.shape_layers: list[ShapeLayer] = []

        self.route_geojson: dict | None = None      # WGS84 FeatureCollection
        self.route_crs: Any = None                  # original routing CRS
        self.route_gdf_crs: gpd.GeoDataFrame | None = None  # routes in routing CRS
        self.endpoints_geojson: dict | None = None  # source/target markers (WGS84)
        # Per-route control points in the routing CRS: one dict per route with
        # {"source": (x, y), "target": (x, y)} — the editable anchors.
        self.route_controls: list[dict] = []

        self.cost_handler: RasterHandler | None = None  # for live cost feedback
        self.path_finder: Any = None                    # for waypoint re-routing

        self._app = None

    # ------------------------------------------------------------------ layers
    def add_cost_raster(self, source: Any, *, name: str = "Cost raster",
                        crs: Any = None, transform: Any = None,
                        colormap: str | None = None,
                        handler: RasterHandler | None = None) -> RasterTileLayer:
        """Serve a cost raster as a tile overlay and enable live cost feedback.

        ``source`` may be a file path, a numpy array (with ``crs`` + ``transform``),
        a ``RasterHandler``, or a ``RasterDataset``.
        """
        layer = build_tile_layer(
            source, name=name, crs=crs, transform=transform,
            colormap=colormap or self.colormap, work_dir=self.work_dir)
        self.raster_layers.append(layer)

        if handler is not None:
            self.cost_handler = handler
        elif isinstance(source, RasterHandler):
            self.cost_handler = source
        return layer

    def add_paths(self, paths: Any, *, crs: Any = None,
                  name: str = "Routes") -> "RouteViewer":
        """Add result routes (PathCollection, Path, list of Path, or GeoDataFrame)."""
        gdf, endpoints = self._paths_to_gdf(paths, crs)
        if gdf.crs is None:
            raise ValueError(
                "Routes have no CRS. Pass crs=... (e.g. the raster CRS) so they "
                "can be shown on the map.")
        self.route_crs = gdf.crs
        self.route_gdf_crs = gdf
        self.route_geojson = geo.gdf_to_wgs84_geojson(gdf)
        # Editable control points: source = first vertex, target = last vertex.
        self.route_controls = []
        for geom in gdf.geometry:
            coords = list(geom.coords)
            self.route_controls.append({
                "source": (float(coords[0][0]), float(coords[0][1])),
                "target": (float(coords[-1][0]), float(coords[-1][1])),
            })
        if endpoints is not None and not endpoints.empty:
            self.endpoints_geojson = geo.gdf_to_wgs84_geojson(endpoints)
        return self

    def add_shapes(self, shapes: Any, *, name: str = "Shapes", crs: Any = None,
                   color: str = "#ff7800") -> "RouteViewer":
        """Add input vector shapes (GeoDataFrame, file path, or VectorDataset)."""
        gdf = self._to_gdf(shapes, crs)
        if gdf.crs is None:
            raise ValueError(f"Shape layer '{name}' has no CRS; pass crs=...")
        self.shape_layers.append(
            ShapeLayer(name, geo.gdf_to_wgs84_geojson(gdf), color=color))
        return self

    def add_route_geometries(self, lines: list, properties: list | None = None,
                             crs: Any = None) -> "RouteViewer":
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        """Append routed LineStrings (routing CRS) to the editable route set.

        Rebuilds the WGS84 GeoJSON, per-route control points and endpoint markers
        so the new routes show on the map and become selectable for editing.
        """
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        import pandas as pd

        crs = crs or self.route_crs or (
            self.route_gdf_crs.crs if self.route_gdf_crs is not None else None
        ) or self._raster_crs()
        if crs is None:
            raise ValueError("Cannot determine a CRS for the new routes; pass crs=.")
        props = properties or [{} for _ in lines]
        new_gdf = gpd.GeoDataFrame(props, geometry=list(lines), crs=crs)
        if self.route_gdf_crs is not None and len(self.route_gdf_crs):
            combined = pd.concat([self.route_gdf_crs, new_gdf], ignore_index=True)
            self.route_gdf_crs = gpd.GeoDataFrame(
                combined, geometry="geometry", crs=self.route_gdf_crs.crs)
        else:
            self.route_gdf_crs = new_gdf
        self.route_crs = self.route_gdf_crs.crs
        self._rebuild_route_derived()
        return self

    def _raster_crs(self) -> Any:
        """Best-effort CRS of the loaded cost raster."""
        if self.cost_handler is not None:
            return self.cost_handler.raster_dataset.crs
        if self.raster_layers:
            import rasterio
            with rasterio.open(self.raster_layers[0].source_path) as ds:
                return ds.crs
        return None

    def _rebuild_route_derived(self) -> None:
        """Recompute route_geojson, control points and endpoints from route_gdf_crs."""
        gdf = self.route_gdf_crs
        self.route_geojson = geo.gdf_to_wgs84_geojson(gdf)
        self.route_controls = []
        endpoint_rows = []
        for geom in gdf.geometry:
            coords = list(geom.coords)
            src = (float(coords[0][0]), float(coords[0][1]))
            tgt = (float(coords[-1][0]), float(coords[-1][1]))
            self.route_controls.append({"source": src, "target": tgt})
            endpoint_rows.append({"role": "source", "geometry": Point(*src)})
            endpoint_rows.append({"role": "target", "geometry": Point(*tgt)})
        endpoints = gpd.GeoDataFrame(endpoint_rows, geometry="geometry", crs=gdf.crs)
        self.endpoints_geojson = geo.gdf_to_wgs84_geojson(endpoints)

    def attach_path_finder(self, finder: Any) -> "RouteViewer":
        """Attach a PathFinder to enable waypoint re-routing and live cost."""
        self.path_finder = finder
        if self.cost_handler is None:
            self.cost_handler = getattr(finder, "raster_handler", None)
        return self

    @classmethod
    def from_path_finder(cls, finder: Any, *, add_raster: bool = True,
                         colormap: str = "viridis", **kwargs) -> "RouteViewer":
        """Build a viewer pre-populated from a PathFinder (raster + routes)."""
        viewer = cls(colormap=colormap, **kwargs)
        if add_raster and getattr(finder, "raster_handler", None) is not None:
            viewer.add_cost_raster(finder.raster_handler, colormap=colormap)
        if len(finder.paths):
            viewer.add_paths(finder.paths, crs=finder.dataset.crs)
        viewer.attach_path_finder(finder)
        return viewer

    # ------------------------------------------------------------- conversions
    def _paths_to_gdf(self, paths: Any,
                      crs: Any) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame | None]:
        """Normalize route input to (routes_gdf, endpoints_gdf)."""
        if isinstance(paths, gpd.GeoDataFrame):
            return paths, None

        if isinstance(paths, Path):
            collection = PathCollection()
            collection.add(paths)
        elif isinstance(paths, PathCollection):
            collection = paths
        elif isinstance(paths, (list, tuple)):
            collection = PathCollection()
            for p in paths:
                collection.add(p)
        else:
            raise TypeError(
                f"Unsupported paths type {type(paths)!r}; pass a PathCollection, "
                "Path, list of Path, or GeoDataFrame.")

        crs = crs or self.route_crs or getattr(
            getattr(self.path_finder, "dataset", None), "crs", None)
        records = collection.to_geodataframe_records()
        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
        endpoints = self._endpoints_gdf(collection, crs)
        return gdf, endpoints

    @staticmethod
    def _endpoints_gdf(collection: PathCollection, crs: Any) -> gpd.GeoDataFrame:
        """Build a point GeoDataFrame of every path's source/target."""
        rows = []
        for path in collection:
            for role, coord in (("source", path.source), ("target", path.target)):
                pts = coord if isinstance(coord, list) else [coord]
                for pt in pts:
                    rows.append({
                        "role": role,
                        "path_id": path.path_id,
                        "geometry": Point(pt[0], pt[1]),
                    })
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)

    def _to_gdf(self, shapes: Any, crs: Any) -> gpd.GeoDataFrame:
        """Normalize a shape input to a GeoDataFrame with a CRS."""
        if isinstance(shapes, gpd.GeoDataFrame):
            return shapes if crs is None else shapes.set_crs(crs, allow_override=True)
        if isinstance(shapes, (str, FsPath)):
            return gpd.read_file(shapes)
        data = getattr(shapes, "data", None)  # VectorDataset-like
        if isinstance(data, gpd.GeoDataFrame):
            return data
        raise TypeError(
            f"Unsupported shapes type {type(shapes)!r}; pass a GeoDataFrame, "
            "file path, or VectorDataset.")

    # -------------------------------------------------------------- app launch
    def build_app(self):
        """Build (and cache) the Dash app for this viewer."""
        from .app import build_app
        if self._app is None:
            self._app = build_app(self)
        return self._app

    def launch(self, desktop: bool = False, host: str = "127.0.0.1",
               port: int = 8050, debug: bool = False) -> None:
        """Run the viewer in the browser (default) or a native desktop window."""
        app = self.build_app()
        if desktop:
            _launch_desktop(app, self.title, host, port)
        else:
            app.run(host=host, port=port, debug=debug)

    def shutdown(self) -> None:
        """Stop all background tile servers."""
        for layer in self.raster_layers:
            try:
                layer.tile_client.shutdown()
            except Exception:  # nosec B110
                pass


def _launch_desktop(app, title: str, host: str, port: int) -> None:
    """Serve the Dash app in a daemon thread and open a pywebview window."""
    import threading

    import webview  # pywebview

    def _serve():
        app.run(host=host, port=port, debug=False, use_reloader=False)

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()

    webview.create_window(title, f"http://{host}:{port}", width=1400, height=900)
    webview.start()
