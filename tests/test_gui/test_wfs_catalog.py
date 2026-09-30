"""
Germany-wide map-service catalog (2026-07-15):

- coverage-bbox viewport filtering (only servers covering the map view listed);
- WFS/WMS GetCapabilities discovery (namespace-agnostic, no owslib);
- WMS overlay layers + BKG DGM DEM download (WCS);
- the Data-tab callbacks that wire them up.
"""
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from pyorps.gui import ids, presets
from pyorps.gui.services import catalog

from conftest import invoke

WFS_CAPS = b"""<wfs:WFS_Capabilities xmlns:wfs="http://www.opengis.net/wfs/2.0">
<wfs:FeatureTypeList>
 <wfs:FeatureType><wfs:Name>ave:ave_Nutzung</wfs:Name>
   <wfs:Title>Tatsaechliche Nutzung</wfs:Title></wfs:FeatureType>
 <wfs:FeatureType><wfs:Name>ave:ave_Flurstueck</wfs:Name></wfs:FeatureType>
</wfs:FeatureTypeList></wfs:WFS_Capabilities>"""

WMS_CAPS = b"""<WMS_Capabilities xmlns="http://www.opengis.net/wms">
<Capability><Layer><Title>root</Title>
 <Layer><Name>web</Name><Title>TopPlusOpen</Title></Layer>
 <Layer><Name>web_grau</Name><Title>TopPlusOpen Grau</Title></Layer>
</Layer></Capability></WMS_Capabilities>"""


# ------------------------------------------------------- presets + filtering
def test_all_16_states_plus_nationwide_present():
    wfs = [s for s in presets.MAP_SERVICES if s["service"] == "wfs"
           and s["category"] == "ALKIS land use"]
    states = {s["state"] for s in wfs}
    assert states == set(presets.STATE_NAME) - {"DE"}          # all 16 states
    # nationwide DEM + topo services
    kinds = {s["service"] for s in presets.MAP_SERVICES}
    assert {"wfs", "wms", "wcs"} <= kinds


def test_services_in_view_filters_by_coverage():
    central_hessen = [8.5, 50.1, 9.2, 50.7]
    shown = {s["state"]
             for s in presets.services_in_view(central_hessen)}
    assert "HE" in shown and "DE" in shown        # local + nationwide
    for distant in ("MV", "SH", "BE", "SL"):      # clearly outside the view
        assert distant not in shown
    # no bbox -> everything
    assert len(presets.services_in_view(None)) == len(presets.MAP_SERVICES)


def test_find_service_and_default_dem():
    svc = presets.find_service(
        "https://geodienste.sachsen.de/aaa/public_alkis/vereinf/wfs")
    assert svc["state"] == "SN"
    assert presets.DEFAULT_DEM["coverage"] == "dgm200_inspire__EL.GridCoverage"
    assert presets.DEFAULT_DEM["crs"] == "EPSG:25832"


# --------------------------------------------------------- capabilities XML
def test_wfs_feature_types_parses_namespaced_xml(monkeypatch):
    monkeypatch.setattr(catalog, "_fetch", lambda url, timeout: WFS_CAPS)
    types = catalog.wfs_feature_types("http://x/wfs")
    assert [t["name"] for t in types] == ["ave:ave_Nutzung",
                                          "ave:ave_Flurstueck"]
    assert types[0]["title"] == "Tatsaechliche Nutzung"


def test_wfs_feature_types_errors_are_friendly(monkeypatch):
    monkeypatch.setattr(catalog, "_fetch",
                        lambda url, timeout: b"<html>nope</html>")
    with pytest.raises(ValueError, match="no feature types|valid capabilities"):
        catalog.wfs_feature_types("http://x/wfs")


def test_wms_layers_and_descriptor(monkeypatch):
    monkeypatch.setattr(catalog, "_fetch", lambda url, timeout: WMS_CAPS)
    layers = catalog.wms_layers("http://x/wms")
    assert {l["name"] for l in layers} == {"web", "web_grau"}
    d = catalog.wms_tile_url("http://x/wms", "web")
    assert d["layers"] == "web" and d["transparent"] is True


def test_load_dem_raster_saves_tiff(monkeypatch, tmp_path):
    tiff = (tmp_path / "src.tif")
    with rasterio.open(tiff, "w", driver="GTiff", height=2, width=2, count=1,
                       dtype="float32", crs="EPSG:25832",
                       transform=from_origin(0, 10, 5, 5)) as dst:
        dst.write(np.ones((2, 2), "float32"), 1)
    monkeypatch.setattr(catalog, "_fetch",
                        lambda url, timeout: tiff.read_bytes())
    out = catalog.load_dem_raster("http://x/wcs", "DGM200",
                                  (0, 0, 10, 10), work_dir=tmp_path)
    with rasterio.open(out) as src:
        assert src.width == 2


def test_load_dem_raster_reports_service_error(monkeypatch, tmp_path):
    err = (b'<ExceptionReport><ExceptionText>bbox too large'
           b'</ExceptionText></ExceptionReport>')
    monkeypatch.setattr(catalog, "_fetch", lambda url, timeout: err)
    with pytest.raises(ValueError, match="bbox too large"):
        catalog.load_dem_raster("http://x/wcs", "DGM200", (0, 0, 1, 1),
                                work_dir=tmp_path)


# --------------------------------------------------------------- callbacks
def test_filter_presets_by_viewport(app):
    # bounds are Leaflet [[south, west], [north, east]] — central Hessen
    bounds = [[50.1, 8.5], [50.7, 9.2]]
    resp = invoke(app, (f"{ids.WFS_PRESET}.options",), bounds, "", True)
    labels = [o["label"] for o in resp[ids.WFS_PRESET]["options"]]
    joined = " ".join(labels)
    assert "Hessen" in joined
    assert "Mecklenburg" not in joined and "Schleswig" not in joined
    # category filter
    resp = invoke(app, (f"{ids.WFS_PRESET}.options",), None,
                  "ALKIS land use", False)
    assert all("Custom" in o["label"] or "ALKIS" not in o["label"]
               or o["value"] for o in resp[ids.WFS_PRESET]["options"])


def test_list_feature_types_callback(app, monkeypatch):
    monkeypatch.setattr(
        catalog, "wfs_feature_types",
        lambda url, **k: [{"name": "ave_Nutzung", "title": "Nutzung"}])
    resp = invoke(app, (f"{ids.WFS_LAYER_SELECT}.options", ids.WFS_CAPS_BTN),
                  1, "http://x/wfs", [],
                  triggered=[f"{ids.WFS_CAPS_BTN}.n_clicks"])
    assert resp[ids.WFS_LAYER_SELECT]["options"][0]["value"] == "ave_Nutzung"
    assert "found" in resp[ids.NOTICES]["data"][-1]["title"]


def test_pick_feature_type_fills_layer(app):
    resp = invoke(app, (f"{ids.WFS_LAYER}.value", ids.WFS_LAYER_SELECT),
                  "ave_Nutzung", triggered=[f"{ids.WFS_LAYER_SELECT}.value"])
    assert resp[ids.WFS_LAYER]["value"] == "ave_Nutzung"


def test_load_wms_overlay_callback(app, state):
    from pyorps.gui.callbacks.layers import render_layer

    preset = "wms|https://sgx.geodatenzentrum.de/wms_topplus_open|web"
    resp = invoke(app, ("layers-view.data", ids.OVERLAY_LOAD_BTN),
                  1, preset, [],
                  triggered=[f"{ids.OVERLAY_LOAD_BTN}.n_clicks"])
    wms_layers = state.layers_of_kind("wms")
    assert len(wms_layers) == 1
    layer = wms_layers[0]
    assert layer.meta["wms"]["layers"] == "web"
    assert "bounds" not in layer.meta["wms"]          # no study area -> nationwide
    # renders as a real dash-leaflet WMSTileLayer
    component = render_layer(layer)
    assert component.layers == "web"
    assert component.url.endswith("wms_topplus_open")
    assert resp[ids.LAYERS_VIEW]["data"][-1]["kind"] == "wms"


def test_wms_overlay_covers_study_area(app, state):
    from pyorps.gui.callbacks.layers import render_layer

    state.study_area = {"type": "Feature", "properties": {},
                        "geometry": {"type": "Polygon", "coordinates": [[
                            [9.0, 50.5], [9.1, 50.5], [9.1, 50.6],
                            [9.0, 50.6], [9.0, 50.5]]]}}
    preset = "wms|https://sgx.geodatenzentrum.de/wms_topplus_open|web"
    invoke(app, ("layers-view.data", ids.OVERLAY_LOAD_BTN), 1, preset, [],
           triggered=[f"{ids.OVERLAY_LOAD_BTN}.n_clicks"])
    layer = state.layers_of_kind("wms")[0]
    (south, west), (north, east) = layer.meta["wms"]["bounds"]  # [[s,w],[n,e]]
    # expanded OUTWARD so every edge tile intersecting the area loads (task 59)
    assert south <= 50.5 and west <= 9.0
    assert north >= 50.6 and east >= 9.1
    # still bounded near the study area, not nationwide (2% of 0.1° span)
    assert south == pytest.approx(50.498, abs=1e-3)
    assert render_layer(layer).bounds == layer.meta["wms"]["bounds"]


def test_dem_get_coverage_uses_native_axes(monkeypatch, tmp_path):
    captured = {}

    def fake_fetch(url, timeout):
        captured["url"] = url
        with open(tmp_path / "z.tif", "wb"):
            pass
        return b"II" + b"\x00" * 40

    monkeypatch.setattr(catalog, "_fetch", fake_fetch)
    catalog.load_dem_raster("http://x/wcs", "dgm200_inspire__EL.GridCoverage",
                            (400000, 5600000, 401000, 5601000),
                            axis_labels=("E", "N"), work_dir=tmp_path)
    assert "SUBSET=E(400000,401000)" in captured["url"]
    assert "SUBSET=N(5600000,5601000)" in captured["url"]
    assert "%28" not in captured["url"]               # parens stay literal


def test_services_in_view_intersection():
    # any service whose coverage overlaps the view qualifies (>=1 tile in view)
    hessen = [8.6, 50.3, 9.0, 50.5]
    shown = {s["state"] for s in presets.services_in_view(hessen)}
    assert "HE" in shown and "DE" in shown
    for distant in ("MV", "SH", "BE"):                  # no overlap at all
        assert distant not in shown


def test_load_dem_callback(app, state, monkeypatch, tmp_path):
    # a real small tif for the (mocked) DEM download to return
    tiff = tmp_path / "dem.tif"
    with rasterio.open(tiff, "w", driver="GTiff", height=4, width=4, count=1,
                       dtype="float32", crs="EPSG:25832",
                       transform=from_origin(500000, 5600000, 50, 50)) as dst:
        dst.write(np.linspace(100, 200, 16).reshape(4, 4).astype("float32"), 1)
    monkeypatch.setattr(catalog, "load_dem_raster",
                        lambda *a, **k: str(tiff))
    state.study_area = {"type": "Feature", "properties": {},
                        "geometry": {"type": "Polygon", "coordinates": [[
                            [9.0, 50.5], [9.05, 50.5], [9.05, 50.55],
                            [9.0, 50.55], [9.0, 50.5]]]}}
    resp = invoke(app, ("layers-view.data", ids.DEM_LOAD_BTN), 1, None, [],
                  triggered=[f"{ids.DEM_LOAD_BTN}.n_clicks"])
    rasters = state.layers_of_kind("raster")
    assert any(r.name == "DEM (DGM)" for r in rasters)


def test_load_dem_without_study_area_warns(app, state):
    state.study_area = None
    resp = invoke(app, ("layers-view.data", ids.DEM_LOAD_BTN), 1, None, [],
                  triggered=[f"{ids.DEM_LOAD_BTN}.n_clicks"])
    assert resp[ids.NOTICES]["data"][-1]["title"] == "Draw a study area first"
