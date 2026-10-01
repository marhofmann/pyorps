"""
OpenStreetMap features via the Overpass API (2026-07-15) — the "edit base" /
cost source the basemap tiles couldn't provide. Network is always mocked.
"""
import geopandas as gpd
import pytest

from pyorps.gui import ids
from pyorps.gui.services import osm

from conftest import invoke

ELEMENTS = [
    {"type": "node", "id": 1, "lat": 50.05, "lon": 8.1,
     "tags": {"power": "tower"}},
    {"type": "way", "id": 2, "tags": {"highway": "primary"},
     "geometry": [{"lat": 50.0, "lon": 8.0}, {"lat": 50.01, "lon": 8.02}]},
    {"type": "way", "id": 3, "tags": {"landuse": "forest"},
     "geometry": [{"lat": 50.0, "lon": 8.0}, {"lat": 50.0, "lon": 8.05},
                  {"lat": 50.03, "lon": 8.05}, {"lat": 50.0, "lon": 8.0}]},
    {"type": "relation", "id": 4,
     "tags": {"type": "multipolygon", "natural": "water"},
     "members": [{"type": "way", "role": "outer",
                  "geometry": [{"lat": 50.1, "lon": 8.1},
                               {"lat": 50.1, "lon": 8.15},
                               {"lat": 50.12, "lon": 8.15},
                               {"lat": 50.1, "lon": 8.1}]}]},
]


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload


# --------------------------------------------------------------- query build
def test_build_query_and_filters():
    q = osm.build_query(['nwr["landuse"]'], (50.0, 8.0, 50.1, 8.2), timeout=25)
    assert 'nwr["landuse"](50.0,8.0,50.1,8.2);' in q
    assert "[out:json][timeout:25]" in q and "out geom;" in q


def test_custom_filters_variants():
    assert osm.custom_filters("landuse=forest, highway") == \
        ['nwr["landuse"="forest"]', 'nwr["highway"]']
    assert osm.custom_filters("building") == ['nwr["building"]']
    assert osm.custom_filters("  ") == []


def test_preset_filters_exist():
    assert osm.preset_filters("Roads (highway=*)") == ['way["highway"]']
    assert osm.preset_filters("nope") == []


# ------------------------------------------------------------ geometry build
def test_elements_to_gdf_mixed_geometries():
    gdf = osm.elements_to_gdf(ELEMENTS)
    assert set(gdf.geometry.type) == {"Point", "LineString", "Polygon"}
    assert str(gdf.crs) == "EPSG:4326"
    # tags become columns, plus osm provenance
    for col in ("osm_id", "osm_type", "landuse", "highway", "power"):
        assert col in gdf.columns
    # a closed non-area way stays a line, not a polygon
    assert gdf.loc[gdf["osm_id"] == 2, "geometry"].iloc[0].geom_type == \
        "LineString"


def test_elements_to_gdf_empty():
    assert osm.elements_to_gdf([]).empty


# --------------------------------------------------------------- Overpass I/O
def test_load_osm_features_success(monkeypatch):
    import requests

    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: _FakeResponse({"elements": ELEMENTS}))
    gdf = osm.load_osm_features((50.0, 8.0, 50.2, 8.3), ['nwr["landuse"]'])
    assert len(gdf) == 4


def test_load_osm_features_empty_is_friendly(monkeypatch):
    import requests

    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: _FakeResponse({"elements": []}))
    with pytest.raises(ValueError, match="No OSM features"):
        osm.load_osm_features((50.0, 8.0, 50.2, 8.3), ['nwr["landuse"]'])


def test_load_osm_features_rate_limit(monkeypatch):
    import requests

    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: _FakeResponse({}, status=429))
    with pytest.raises(ValueError, match="rate-limiting|overloaded"):
        osm.load_osm_features((50.0, 8.0, 50.2, 8.3), ['nwr["landuse"]'])


def test_load_osm_features_needs_a_filter():
    with pytest.raises(ValueError, match="No OSM feature selected"):
        osm.load_osm_features((50.0, 8.0, 50.2, 8.3), [])


# ----------------------------------------------------------------- callbacks
def _study_area():
    return {"type": "Feature", "properties": {},
            "geometry": {"type": "Polygon", "coordinates": [[
                [8.0, 50.0], [8.2, 50.0], [8.2, 50.2], [8.0, 50.2],
                [8.0, 50.0]]]}}


def test_load_osm_callback_registers_vector_layer(app, state, monkeypatch):
    monkeypatch.setattr(osm, "load_osm_features",
                        lambda *a, **k: osm.elements_to_gdf(ELEMENTS))
    state.study_area = _study_area()
    resp = invoke(app, ("layers-view.data", ids.OSM_LOAD_BTN),
                  1, "Land use (landuse=*)", "", [], [],
                  triggered=[f"{ids.OSM_LOAD_BTN}.n_clicks"])
    vectors = state.layers_of_kind("vector")
    assert len(vectors) == 1
    layer = vectors[0]
    assert layer.name.startswith("OSM:")
    # reprojected into the routing CRS (metres), not left in WGS84
    assert str(layer.crs) == str(state.project_crs)
    assert layer.meta["source"]["kind"] == "osm"
    assert resp[ids.NOTICES]["data"][-1]["title"].startswith("Loaded")


def test_load_osm_callback_requires_study_area(app, state, monkeypatch):
    monkeypatch.setattr(osm, "load_osm_features",
                        lambda *a, **k: osm.elements_to_gdf(ELEMENTS))
    state.study_area = None
    resp = invoke(app, ("layers-view.data", ids.OSM_LOAD_BTN),
                  1, "Land use (landuse=*)", "", [], [],
                  triggered=[f"{ids.OSM_LOAD_BTN}.n_clicks"])
    assert resp[ids.NOTICES]["data"][-1]["title"] == "Draw a study area first"


def test_load_osm_callback_custom_tag(app, state, monkeypatch):
    captured = {}

    def fake(bbox, filters, **k):
        captured["filters"] = filters
        return osm.elements_to_gdf(ELEMENTS)

    monkeypatch.setattr(osm, "load_osm_features", fake)
    state.study_area = _study_area()
    invoke(app, ("layers-view.data", ids.OSM_LOAD_BTN),
           1, None, "landuse=forest", [], [],
           triggered=[f"{ids.OSM_LOAD_BTN}.n_clicks"])
    assert captured["filters"] == ['nwr["landuse"="forest"]']
