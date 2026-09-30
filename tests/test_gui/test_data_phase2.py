"""Phase 2: study-area drawing (C4), local/WFS loading, dataset list."""
import json

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from pyorps.gui import ids
from pyorps.gui.services import data_io

from conftest import invoke

# a small WGS84 polygon near Frankfurt (drawn-area payload from EditControl)
AREA_FEATURE = {
    "type": "Feature", "properties": {"type": "polygon"},
    "geometry": {"type": "Polygon", "coordinates": [[
        [8.6, 50.0], [8.8, 50.0], [8.8, 50.2], [8.6, 50.2], [8.6, 50.0]]]},
}
DRAWN = {"type": "FeatureCollection", "features": [AREA_FEATURE]}


# --------------------------------------------------------------- service unit
def test_study_area_projection_roundtrip():
    polygon = data_io.study_area_polygon(AREA_FEATURE, "EPSG:25832")
    minx, miny, maxx, maxy = polygon.bounds
    # ~14 km x 22 km box in UTM32
    assert 10_000 < (maxx - minx) < 20_000
    assert 15_000 < (maxy - miny) < 30_000
    assert data_io.study_area_polygon(None, "EPSG:25832") is None
    assert data_io.study_area_bounds(None, "EPSG:25832") is None


def _write_geojson(tmp_path, crs="EPSG:25832"):
    gdf = gpd.GeoDataFrame(
        {"use": ["forest", "road"]},
        geometry=[Polygon([(500000, 5530000), (500100, 5530000),
                           (500100, 5530100)]),
                  Polygon([(500200, 5530200), (500300, 5530200),
                           (500300, 5530300)])],
        crs=crs)
    path = tmp_path / "data.geojson"
    gdf.to_file(path, driver="GeoJSON")
    return str(path), gdf


def test_load_local_vector_reprojects(tmp_path):
    path, _ = _write_geojson(tmp_path, crs="EPSG:32632")
    gdf = data_io.load_local_vector(path, target_crs="EPSG:25832")
    assert str(gdf.crs) == "EPSG:25832"
    assert len(gdf) == 2


def test_clip_to_area(tmp_path):
    path, gdf = _write_geojson(tmp_path)
    area = Polygon([(499990, 5529990), (500150, 5529990),
                    (500150, 5530150), (499990, 5530150)])
    loaded = data_io.load_local_vector(path, target_crs="EPSG:25832")
    clipped = data_io.clip_to_area(loaded, area)
    assert len(clipped) == 1
    assert clipped.iloc[0]["use"] == "forest"


def test_dataset_summary(tmp_path):
    path, _ = _write_geojson(tmp_path)
    gdf = data_io.load_local_vector(path)
    summary = data_io.dataset_summary(gdf)
    assert summary["n_features"] == 2
    assert "use" in summary["columns"]


def test_load_wfs_vector_through_real_dataset(monkeypatch, tmp_path):
    """Drive the REAL WFSVectorDataset.load_data (only the network call is
    stubbed): a shapely-Polygon study-area mask must not crash pyorps'
    ``mask.total_bounds`` bbox derivation (regression, 2026-07-14)."""
    from shapely.geometry import Polygon as ShapelyPolygon

    path, gdf = _write_geojson(tmp_path)
    seen = {}

    def fake_load_from_wfs(url, layer, bbox=None, **kwargs):
        seen["url"], seen["layer"], seen["bbox"] = url, layer, bbox
        return gpd.read_file(path)

    import pyorps.io.geo_dataset as geo_dataset_module
    monkeypatch.setattr(geo_dataset_module, "load_from_wfs",
                        fake_load_from_wfs)

    mask = ShapelyPolygon([(499990, 5529990), (500400, 5529990),
                           (500400, 5530400), (499990, 5530400)])
    out = data_io.load_wfs_vector("https://x.test/wfs", "lyr", mask=mask,
                                  target_crs="EPSG:25832")
    assert len(out) == 2 and str(out.crs) == "EPSG:25832"
    assert seen["url"] == "https://x.test/wfs"
    # the request bbox was derived from the polygon's bounds
    assert seen["bbox"] == pytest.approx((499990, 5529990, 500400, 5530400))


# --------------------------------------------------------- headless callbacks
def test_draw_sets_study_area(app, state):
    resp = invoke(app, ("study-area-info.children", ids.DRAW_CONTROL),
                  DRAWN, "area", "Manual costs", 65535, "override", [], [],
                  triggered=[f"{ids.DRAW_CONTROL}.geojson"])
    assert state.study_area == AREA_FEATURE
    assert state.get("study-area") is not None
    info = resp[ids.STUDY_AREA_INFO]["children"]
    assert "km" in info
    notices = resp[ids.NOTICES]["data"]
    assert notices[-1]["title"] == "Study area set"


def test_clear_study_area(app, state):
    invoke(app, ("study-area-info.children", ids.DRAW_CONTROL), DRAWN,
           "area", "Manual costs", 65535, "override", [], [],
           triggered=[f"{ids.DRAW_CONTROL}.geojson"])
    resp = invoke(app, ("study-area-info.children", ids.STUDY_AREA_CLEAR), 1,
                  triggered=[f"{ids.STUDY_AREA_CLEAR}.n_clicks"])
    assert state.study_area is None
    assert state.get("study-area") is None
    toolbar = resp[ids.DRAW_CONTROL]["editToolbar"]
    assert toolbar["action"] == "clear all"


def test_project_crs_validation(app, state):
    invoke(app, ("notices.data", ids.PROJECT_CRS), "EPSG:4326", [],
           triggered=[f"{ids.PROJECT_CRS}.value"])
    # geographic CRS rejected -> state unchanged
    assert state.project_crs == "EPSG:25832"
    invoke(app, ("notices.data", ids.PROJECT_CRS), "EPSG:32632", [],
           triggered=[f"{ids.PROJECT_CRS}.value"])
    assert state.project_crs == "EPSG:32632"


def test_local_load_missing_file_shows_notice(app, state):
    resp = invoke(app, ("dataset-list.children", ids.LOCAL_LOAD_BTN),
                  1, "C:/nope/gone.geojson", None, True, [],
                  triggered=[f"{ids.LOCAL_LOAD_BTN}.n_clicks"])
    notices = resp[ids.NOTICES]["data"]
    assert notices[-1]["title"] == "File not found"
    assert state.layers_of_kind("vector") == []


def test_local_load_adds_layer(app, state, tmp_path):
    path, _ = _write_geojson(tmp_path)
    resp = invoke(app, ("dataset-list.children", ids.LOCAL_LOAD_BTN),
                  1, path, None, False, [],
                  triggered=[f"{ids.LOCAL_LOAD_BTN}.n_clicks"])
    layers = state.layers_of_kind("vector")
    assert len(layers) == 1
    assert layers[0].geojson["type"] == "FeatureCollection"
    rendered = json.dumps(resp[ids.DATASET_LIST]["children"])
    assert "2 features" in rendered
    assert resp[ids.NOTICES]["data"][-1]["severity"] == "success"


def test_wfs_preset_fills_fields(app):
    from pyorps.gui.presets import WFS_PRESETS

    preset = WFS_PRESETS[0]
    resp = invoke(app, (f"{ids.WFS_URL}.value",),
                  f'{preset["url"]}|{preset["layer"]}',
                  triggered=[f"{ids.WFS_PRESET}.value"])
    assert resp[ids.WFS_URL]["value"] == preset["url"]
    assert resp[ids.WFS_LAYER]["value"] == preset["layer"]


def test_wfs_load_requires_area_and_valid_url(app, state):
    resp = invoke(app, ("dataset-list.children", ids.WFS_LOAD_BTN),
                  1, "not a url", "layer", True, [],
                  triggered=[f"{ids.WFS_LOAD_BTN}.n_clicks"])
    assert resp[ids.NOTICES]["data"][-1]["title"] == \
        "That WFS URL looks invalid"
    resp = invoke(app, ("dataset-list.children", ids.WFS_LOAD_BTN),
                  1, "https://x.test/wfs", "layer", True, [],
                  triggered=[f"{ids.WFS_LOAD_BTN}.n_clicks"])
    assert resp[ids.NOTICES]["data"][-1]["title"] == \
        "Draw a study area first"


def test_wfs_connection_error_translated(app, state, monkeypatch):
    from pyorps.core.exceptions import WFSConnectionError

    def boom(*args, **kwargs):
        raise WFSConnectionError("server down")

    monkeypatch.setattr(data_io, "load_wfs_vector", boom)
    from pyorps.gui.callbacks import data as data_cb
    monkeypatch.setattr(data_cb.data_io, "load_wfs_vector", boom)
    resp = invoke(app, ("dataset-list.children", ids.WFS_LOAD_BTN),
                  1, "https://x.test/wfs", "layer", False, [],
                  triggered=[f"{ids.WFS_LOAD_BTN}.n_clicks"])
    assert resp[ids.NOTICES]["data"][-1]["title"] == \
        "Can't reach the WFS server"


def test_load_raster_via_raster_tab(app, state, raster_path):
    resp = invoke(app, ("layers-view.data", ids.RASTER_LOAD_BTN),
                  1, raster_path, "viridis", [],
                  triggered=[f"{ids.RASTER_LOAD_BTN}.n_clicks"])
    rasters = state.layers_of_kind("raster")
    assert len(rasters) == 1
    assert rasters[0].tile.tile_url.startswith("http://localhost:")
    view = resp[ids.MAP_VIEW]["data"]
    assert view["fit_bounds"] is not None
    # opacity slider updates the served tiles directly (no host rebuild)
    tid = {"type": ids.TYPE_RASTER_TILE, "id": rasters[0].id}
    invoke(app, (ids.TYPE_RASTER_TILE, ids.RASTER_OPACITY), 0.4, [tid],
           outputs_list=[{"id": tid, "property": "opacity"}],
           triggered=[f"{ids.RASTER_OPACITY}.value"])
    assert rasters[0].opacity == 0.4
