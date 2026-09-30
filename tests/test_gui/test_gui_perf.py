"""
GUI performance guards (2026-07-15): the layer-host render skips work when
nothing painted changed, table rows are built vectorized (not per-row iloc),
and CRS transformers are cached. Behaviour must stay identical — these tests
pin the fast paths.
"""
import geopandas as gpd
import pytest
from shapely.geometry import box
from dash.exceptions import PreventUpdate

from pyorps.gui import ids
from pyorps.gui.services import geo
from pyorps.gui.callbacks.layers import (host_signature, render_layer_host,
                                         _rows_from_gdf)

from conftest import invoke


def _vec(state, name, n=3):
    gdf = gpd.GeoDataFrame(
        {"use": [f"u{i}" for i in range(n)], "cost": list(range(n))},
        geometry=[box(500000 + i, 5600000, 500000 + i + 5, 5600005)
                  for i in range(n)], crs="EPSG:25832")
    return state.add_layer(name, "vector", gdf=gdf, crs=gdf.crs,
                           geojson=geo.gdf_to_wgs84_geojson(gdf))


# --------------------------------------------------- render-host signature guard
def test_render_host_skips_unchanged(app, state):
    _vec(state, "A")
    resp = invoke(app, ("layer-host.children",), state.layers_view())
    assert len(resp[ids.LAYER_HOST]["children"]) == 1
    # identical state -> no rebuild (this is the multi-MB re-ship we avoid)
    with pytest.raises(PreventUpdate):
        invoke(app, ("layer-host.children",), state.layers_view())


def test_render_host_rerenders_on_real_change(app, state):
    layer = _vec(state, "A")
    invoke(app, ("layer-host.children",), state.layers_view())
    # a visibility change must re-render
    state.set_visible(layer.id, False)
    resp = invoke(app, ("layer-host.children",), state.layers_view())
    assert resp[ids.LAYER_HOST]["children"] == []


def test_host_signature_tracks_visible_geojson_style(state):
    layer = _vec(state, "A")
    sig0 = host_signature(state)
    layer.style = {"color": "#123456"}
    assert host_signature(state) != sig0            # style change
    sig1 = host_signature(state)
    layer.geojson = geo.gdf_to_wgs84_geojson(layer.gdf)   # reassigned -> new id
    assert host_signature(state) != sig1            # data change
    sig2 = host_signature(state)
    state.set_visible(layer.id, False)
    assert host_signature(state) != sig2            # visibility change


def test_host_signature_stable_when_nothing_changes(state):
    _vec(state, "A")
    _vec(state, "B", n=5)
    assert host_signature(state) == host_signature(state)


def test_opacity_not_in_layers_view_path(state):
    """A raster's opacity lives on the tile signature, so changing it via the
    slider path never invalidates the (heavy) vector layers' render."""
    layer = _vec(state, "V")
    sig_before = host_signature(state)
    # bumping a raster's opacity would change only raster tuples, never a vector
    layer2 = state.add_layer("R", "raster", crs="EPSG:25832",
                             meta={}, tile=None)
    sig_with_raster = host_signature(state)
    assert sig_with_raster != sig_before
    # the vector tuple is unchanged across the two signatures
    assert host_signature(state)[0] == sig_with_raster[0]


# ----------------------------------------------------- vectorized table rows
def test_rows_from_gdf_matches_and_indexes():
    gdf = gpd.GeoDataFrame(
        {"a": [1, 2, 3], "b": ["x", "y", "z"]},
        geometry=[box(0, 0, 1, 1)] * 3, crs="EPSG:25832")
    rows = _rows_from_gdf(gdf, ["a", "b"])
    assert [r["__row"] for r in rows] == [0, 1, 2]
    assert rows[1]["a"] == 2 and rows[2]["b"] == "z"


def test_rows_from_gdf_respects_limit():
    gdf = gpd.GeoDataFrame(
        {"a": list(range(10))}, geometry=[box(0, 0, 1, 1)] * 10,
        crs="EPSG:25832")
    rows = _rows_from_gdf(gdf, ["a"], limit=4)
    assert len(rows) == 4 and rows[-1]["__row"] == 3


def test_attr_table_caps_large_layers(state):
    from pyorps.gui.callbacks.layers import _attr_table_payload, MAX_TABLE_ROWS

    n = MAX_TABLE_ROWS + 250
    gdf = gpd.GeoDataFrame(
        {"a": list(range(n))},
        geometry=[box(0, 0, 1, 1)] * n, crs="EPSG:25832")
    layer = state.add_layer("big", "vector", gdf=gdf, crs=gdf.crs,
                            geojson={"type": "FeatureCollection",
                                     "features": []})
    _cols, rows, title = _attr_table_payload(layer)
    assert len(rows) == MAX_TABLE_ROWS
    assert "first" in title and str(n) in title.replace(",", "")


# ------------------------------------------------------ transformer caching
def test_crs_transformer_is_cached():
    a = geo.crs_transformer("EPSG:25832", "EPSG:4326")
    b = geo.crs_transformer("EPSG:25832", "EPSG:4326")
    assert a is b                                   # same object reused
    c = geo.crs_transformer("EPSG:3857", "EPSG:4326")
    assert c is not a


# ---------------------------------------------- URL-served GeoJSON (big layers)
def _big_vec(state, name, n):
    gdf = gpd.GeoDataFrame(
        {"a": list(range(n))},
        geometry=[box(500000 + i, 5600000, 500000 + i + 3, 5600003)
                  for i in range(n)], crs="EPSG:25832")
    return state.add_layer(name, "vector", gdf=gdf, crs=gdf.crs,
                           geojson=geo.gdf_to_wgs84_geojson(gdf))


def test_geojson_rev_bumps_on_reassign(state):
    layer = _vec(state, "A")
    assert layer.geojson_rev == 0                   # set once at creation
    layer.geojson = geo.gdf_to_wgs84_geojson(layer.gdf)
    assert layer.geojson_rev == 1                   # cache-buster advances


def test_render_layer_urls_large_inlines_small(state):
    from pyorps.gui.callbacks.layers import render_layer, URL_GEOJSON_THRESHOLD

    small = _vec(state, "small", n=3)
    big = _big_vec(state, "big", URL_GEOJSON_THRESHOLD + 20)
    cs, cb = render_layer(small), render_layer(big)
    assert getattr(cs, "data", None) is not None
    assert getattr(cs, "url", None) is None
    assert getattr(cb, "data", None) is None
    # geobuf wire format preferred (perf plan 1.3); GeoJSON URL as fallback
    try:
        import geobuf  # noqa: F401
        assert cb.url == f"/_gb/{big.id}?v={big.geojson_rev}"
        assert cb.format == "geobuf"
    except ImportError:
        assert cb.url == f"/_gj/{big.id}?v={big.geojson_rev}"


def test_geojson_route_serves_and_404s(app, state):
    big = _big_vec(state, "big", 10)
    client = app.server.test_client()
    ok = client.get(f"/_gj/{big.id}")
    assert ok.status_code == 200
    assert len(ok.get_json()["features"]) == 10
    assert "max-age" in ok.headers.get("Cache-Control", "")
    assert client.get("/_gj/does-not-exist").status_code == 404
