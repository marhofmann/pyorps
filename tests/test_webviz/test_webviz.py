"""
Tests for pyorps.webviz (interactive Dash + Leaflet viewer).

These require the optional ``viz`` dependencies and are skipped otherwise::

    pip install "pyorps[viz]"
"""
import json
import math

import numpy as np
import pytest

pytest.importorskip("dash")
pytest.importorskip("dash_leaflet")
pytest.importorskip("localtileserver")
requests = pytest.importorskip("requests")

import geopandas as gpd
from shapely.geometry import LineString

from pyorps import PathFinder
from pyorps.raster.handler import create_test_tiff


SOURCE = (500020.0, 5599980.0)
TARGET = (500180.0, 5599820.0)


@pytest.fixture(scope="module")
def raster_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("webviz") / "cost.tif"
    create_test_tiff(str(path), width=200, height=200, pattern="gradient",
                     crs="EPSG:32632")
    return str(path)


@pytest.fixture(scope="module")
def finder(raster_path):
    pf = PathFinder(raster_path, source_coords=SOURCE, target_coords=TARGET,
                    search_space_buffer_m=80, graph_api="cython")
    pf.find_route()
    return pf


# ------------------------------------------------------------------- geo utils
def test_gdf_wgs84_roundtrip(finder):
    from pyorps.webviz import geo

    gdf = gpd.GeoDataFrame(finder.paths.to_geodataframe_records(),
                           geometry="geometry", crs=finder.dataset.crs)
    gj = geo.gdf_to_wgs84_geojson(gdf)
    assert gj["type"] == "FeatureCollection"
    coords = gj["features"][0]["geometry"]["coordinates"]
    # WGS84 lon/lat for the test area (~9E, 50.5N)
    assert 8 < coords[0][0] < 10 and 50 < coords[0][1] < 51

    line_crs = geo.wgs84_linestring_to_crs(coords, finder.dataset.crs)
    # reprojected back into the routing CRS -> near the original coordinates
    assert abs(line_crs.coords[0][0] - SOURCE[0]) < 5


def test_gdf_wgs84_handles_timestamp_columns():
    """Real WFS/ALKIS layers carry datetime columns; they must serialize."""
    import pandas as pd
    from shapely.geometry import Point
    from pyorps.webviz import geo

    gdf = gpd.GeoDataFrame(
        {"name": ["a"], "checked": [pd.Timestamp("2024-01-02")]},
        geometry=[Point(500000, 5600000)], crs="EPSG:32632")
    gj = geo.gdf_to_wgs84_geojson(gdf)
    props = gj["features"][0]["properties"]
    assert props["name"] == "a"
    assert isinstance(props["checked"], str) and "2024-01-02" in props["checked"]


# ---------------------------------------------------------------- cost metric
def test_cost_matches_pyorps_metric(finder):
    from pyorps.webviz import cost

    path = finder.paths.all[0]
    rc = cost.evaluate_route_cost(path.path_geometry, finder.raster_handler)
    # The webviz evaluator reuses pyorps' own numba kernel on an 8-connected
    # discretization of the same routed geometry -> identical totals.
    assert rc.total_length == pytest.approx(path.total_length, rel=1e-6)
    assert rc.total_cost == pytest.approx(path.total_cost, rel=1e-6)
    assert rc.total_cell_cost == pytest.approx(path.total_cell_cost, rel=1e-6)
    assert not rc.crosses_forbidden


def test_cost_flags_forbidden(finder):
    from pyorps.webviz import cost

    handler = finder.raster_handler
    forbidden = np.iinfo(handler.data.dtype).max
    # Force a couple of cells to the exclusion sentinel and route a line over them.
    handler.data[0, 5, 5] = forbidden
    handler.data[0, 5, 6] = forbidden
    xs = handler.indices_to_coords([(5, 4), (5, 7)])
    line = LineString([tuple(xs[0]), tuple(xs[1])])
    rc = cost.evaluate_route_cost(line, handler)
    assert rc.crosses_forbidden
    assert rc.n_forbidden_cells >= 1


# --------------------------------------------------------------- tile serving
def test_tile_layer_serves_png(raster_path, tmp_path):
    from pyorps.webviz.tiles import build_tile_layer

    layer = build_tile_layer(raster_path, name="cost", work_dir=tmp_path)
    try:
        assert layer.tile_url.startswith("http://localhost:")
        (s, w), (n, e) = layer.bounds
        lat, lon = (s + n) / 2, (w + e) / 2
        z = 15
        nn = 2 ** z
        x = int((lon + 180) / 360 * nn)
        y = int((1 - math.log(math.tan(math.radians(lat)) +
                              1 / math.cos(math.radians(lat))) / math.pi) / 2 * nn)
        resp = requests.get(layer.tile_url.format(z=z, x=x, y=y), timeout=20)
        assert resp.status_code == 200
        assert resp.headers.get("content-type") == "image/png"
        assert len(resp.content) > 0
    finally:
        layer.tile_client.shutdown()


# ------------------------------------------------------------------- full app
def test_route_layer_is_direct_map_child(finder):
    """route-layer must be a direct Map child (not nested in LayersControl),
    otherwise its data updates (new/edited routes) do not re-render."""
    import dash_leaflet as dl
    from pyorps.webviz import RouteViewer

    viewer = RouteViewer.from_path_finder(finder)
    app = viewer.build_app()
    try:
        def walk(comp):
            yield comp
            kids = getattr(comp, "children", None)
            if kids is None:
                return
            if not isinstance(kids, (list, tuple)):
                kids = [kids]
            for k in kids:
                if hasattr(k, "children") or hasattr(k, "id"):
                    yield from walk(k)

        the_map = next(c for c in walk(app.layout) if isinstance(c, dl.Map))
        direct_ids = [getattr(ch, "id", None) for ch in the_map.children]
        assert "route-layer" in direct_ids
        assert "active-route" in direct_ids
    finally:
        viewer.shutdown()


def test_no_marker_drag_input(finder):
    """Drag doesn't sync back in dash-leaflet; ensure we rely on no such input."""
    import json as _json
    from pyorps.webviz import RouteViewer

    viewer = RouteViewer.from_path_finder(finder)
    app = viewer.build_app()
    try:
        assert not any("ctrl-marker" in _json.dumps(cb.get("inputs", []))
                       for cb in app.callback_map.values())
    finally:
        viewer.shutdown()


def test_app_builds_and_registers_callbacks(finder):
    from pyorps.webviz import RouteViewer

    viewer = RouteViewer.from_path_finder(finder)
    buffer_gdf = gpd.GeoDataFrame(
        {"name": ["buf"]}, geometry=[finder.raster_handler.buffer_geometry],
        crs=finder.dataset.crs)
    viewer.add_shapes(buffer_gdf, name="buffer")
    app = viewer.build_app()
    try:
        client = app.server.test_client()
        assert client.get("/").status_code == 200
        assert client.get("/_dash-layout").status_code == 200

        deps = json.loads(client.get("/_dash-dependencies").data)
        outputs = {d["output"] for d in deps}
        assert any("attr-panel.children" in o for o in outputs)
        assert any("cost-stats.children" in o for o in outputs)
        assert any("save-status.children" in o for o in outputs)
        assert any("raster-tile" in o for o in outputs)   # opacity control
    finally:
        viewer.shutdown()


# --------------------------------------------------------------- re-routing
def test_reroute_through_waypoint_detours(finder):
    from pyorps.webviz import RouteViewer
    from pyorps.webviz.reroute import reroute_through_waypoints

    viewer = RouteViewer.from_path_finder(finder)
    base_line = viewer.route_gdf_crs.geometry.iloc[0]
    waypoint = ((SOURCE[0] + TARGET[0]) / 2 + 30, (SOURCE[1] + TARGET[1]) / 2 + 30)
    new_line, rc = reroute_through_waypoints(finder, base_line, [waypoint])
    assert new_line.length > 0
    # Forcing a detour through an off-route waypoint cannot be cheaper.
    assert rc.total_length >= finder.paths.all[0].total_length - 1e-6


def test_control_points_derived_from_route(finder):
    from pyorps.webviz import RouteViewer

    viewer = RouteViewer.from_path_finder(finder)
    assert len(viewer.route_controls) == len(finder.paths)
    ctrl = viewer.route_controls[0]
    assert abs(ctrl["source"][0] - SOURCE[0]) < 2
    assert abs(ctrl["target"][0] - TARGET[0]) < 2


def test_route_through_points_respects_waypoint_order(finder):
    from pyorps.webviz.reroute import route_through_points

    src = (SOURCE[0], SOURCE[1])
    tgt = (TARGET[0], TARGET[1])
    wa = (src[0] + 40, src[1] - 20)
    wb = (tgt[0] - 40, tgt[1] + 20)
    line_ab, _ = route_through_points(finder, [src, wa, wb, tgt])
    line_ba, _ = route_through_points(finder, [src, wb, wa, tgt])
    # Both are valid least-cost chains but the visiting order differs, so the
    # stitched geometries must differ.
    assert list(line_ab.coords) != list(line_ba.coords)
    assert line_ab.length > 0 and line_ba.length > 0


def test_route_through_points_needs_two_points(finder):
    import pytest as _pytest
    from pyorps.webviz.reroute import route_through_points

    with _pytest.raises(ValueError):
        route_through_points(finder, [(SOURCE[0], SOURCE[1])])


# --------------------------------------------------------------- route builder
def test_resolve_backend_matrix():
    from pyorps.webviz.builder import resolve_backend

    assert resolve_backend("dijkstra", "cpu") == ("cython", "dijkstra")
    assert resolve_backend("delta-stepping", "cpu") == ("cython", "delta-stepping")
    assert resolve_backend("dijkstra", "gpu") == ("raster_gpu", "dijkstra")
    assert resolve_backend("delta-stepping", "gpu") == ("raster_gpu", "delta-stepping")
    # bidirectional has no GPU path -> CPU networkit either way
    assert resolve_backend("bidirectional_dijkstra", "cpu") == (
        "networkit", "bidirectional_dijkstra")
    assert resolve_backend("bidirectional_dijkstra", "gpu") == (
        "networkit", "bidirectional_dijkstra")


def test_make_pairs():
    from pyorps.webviz.builder import make_pairs

    assert make_pairs([1, 2], [3, 4], pairwise=False) == [(1, 3), (1, 4), (2, 3), (2, 4)]
    assert make_pairs([1, 2], [3, 4], pairwise=True) == [(1, 3), (2, 4)]


def test_run_routing_single_source_multi_target(raster_path):
    from pyorps.webviz.builder import run_routing

    s = [SOURCE]
    t = [(SOURCE[0] + 150, SOURCE[1] - 120), (SOURCE[0] + 120, SOURCE[1] - 150)]
    finder, built = run_routing(raster_path, sources=s, targets=t,
                                algorithm="delta-stepping", hardware="cpu",
                                search_buffer_m=120)
    assert len(built) == 2
    assert all(b.cost.total_length > 0 for b in built)
    assert finder is not None


def test_run_routing_with_waypoint_and_append(raster_path):
    from pyorps.webviz import RouteViewer
    from pyorps.webviz.builder import run_routing

    viewer = RouteViewer()
    viewer.add_cost_raster(raster_path, name="cost")
    try:
        s = [SOURCE]
        t = [(SOURCE[0] + 150, SOURCE[1] - 120)]
        wp = [(SOURCE[0] + 60, SOURCE[1] - 80)]
        _finder, built = run_routing(raster_path, sources=s, targets=t,
                                     waypoints=wp, algorithm="dijkstra",
                                     hardware="cpu", search_buffer_m=120)
        assert len(built) == 1
        viewer.add_route_geometries([b.line for b in built], [{"kind": "built"}])
        assert len(viewer.route_controls) == 1
        assert len(viewer.route_geojson["features"]) == 1
        # endpoints = source + target per route
        assert len(viewer.endpoints_geojson["features"]) == 2
    finally:
        viewer.shutdown()
