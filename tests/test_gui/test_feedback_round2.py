"""User-feedback round 2 (2026-07-14): dedupe/refresh, WFS empty answer,
warning noise filter, go-fix flash, browse dialogs, feature analysis,
route auto-select + styling."""
import json
import warnings

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from pyorps.gui import ids
from pyorps.gui.services import cost_model, data_io
from pyorps.gui.services.errors import guard

from conftest import SOURCE, TARGET, invoke
from test_routes_phase4 import _draft


def _write_geojson(tmp_path, n=2):
    gdf = gpd.GeoDataFrame(
        {"use": ["forest", "road"][:n]},
        geometry=[Polygon([(500000 + i * 200, 5530000),
                           (500100 + i * 200, 5530000),
                           (500100 + i * 200, 5530100)])
                  for i in range(n)],
        crs="EPSG:25832")
    path = tmp_path / "data.geojson"
    gdf.to_file(path, driver="GeoJSON")
    return str(path)


# ------------------------------------------------- dedupe + refresh (issue 1)
def test_same_local_source_loads_only_once(app, state, tmp_path):
    path = _write_geojson(tmp_path)
    for click in (1, 2):
        invoke(app, ("dataset-list.children", ids.LOCAL_LOAD_BTN),
               click, path, None, False, [],
               triggered=[f"{ids.LOCAL_LOAD_BTN}.n_clicks"])
    layers = state.layers_of_kind("vector")
    assert len(layers) == 1                       # unique, not duplicated
    assert layers[0].meta["source"]["kind"] == "local"


def test_refresh_button_reloads_dataset(app, state, tmp_path):
    path = _write_geojson(tmp_path)
    invoke(app, ("dataset-list.children", ids.LOCAL_LOAD_BTN),
           1, path, None, False, [],
           triggered=[f"{ids.LOCAL_LOAD_BTN}.n_clicks"])
    layer = state.layers_of_kind("vector")[0]
    # the file grows -> refresh picks it up
    gdf = gpd.GeoDataFrame(
        {"use": ["forest", "road", "water"]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1)])] * 3, crs="EPSG:25832")
    gdf.to_file(path, driver="GeoJSON")
    resp = invoke(
        app, ("dataset-list.children", "dataset-refresh"), [1], [],
        triggered=[{"id": {"type": ids.TYPE_DATASET_REFRESH,
                           "index": layer.id},
                    "property": "n_clicks", "value": 1}])
    assert len(state.layers_of_kind("vector")) == 1
    assert layer.meta["summary"]["n_features"] == 3
    assert resp[ids.NOTICES]["data"][-1]["title"].startswith("Refreshed")


# ------------------------------------------- WFS empty answer (issue 2)
def test_wfs_no_data_gives_friendly_error(monkeypatch, tmp_path):
    import pyorps.io.geo_dataset as geo_dataset_module

    monkeypatch.setattr(geo_dataset_module, "load_from_wfs",
                        lambda *a, **k: None)
    from pyorps.core.exceptions import WFSResponseParsingError

    with pytest.raises(WFSResponseParsingError, match="no data"):
        data_io.load_wfs_vector("https://x.test/wfs", "lyr",
                                mask=Polygon([(0, 0), (1, 0), (1, 1)]),
                                target_crs="EPSG:25832")


# ------------------------------------------- warning noise filter (issue 4)
def test_no_overview_warning_is_suppressed():
    def fn():
        warnings.warn("The dataset has no Overviews. rio-tiler performances "
                      "might be impacted.")
        warnings.warn("No search_space_buffer_m set — using full raster")
        return 1

    result, notices = guard(fn, notices=[])
    assert result == 1
    titles = [n["title"] for n in notices]
    assert titles == ["No search buffer set — routing the whole raster"]


# -------------------------------------------- feature analysis (issue 5)
def test_feature_analysis_counts():
    gdf = gpd.GeoDataFrame(
        {"nutzart": ["Wald", "Wald", "Weg"],
         "bez": ["Nadelholz", "Laubholz", ""]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1)])] * 3,
        crs="EPSG:25832")
    per_column, n_combo = cost_model.feature_analysis(
        gdf, ("nutzart", "bez"))
    assert dict(per_column) == {"nutzart": 2, "bez": 3}
    assert n_combo == 3                          # distinct (nutzart, bez)


def test_feature_info_callback(app, state):
    gdf = gpd.GeoDataFrame(
        {"nutzart": ["Wald", "Weg"], "bez": ["Nadelholz", ""]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1)])] * 2,
        crs="EPSG:25832")
    from pyorps.gui.services import geo

    layer = state.add_layer("d", "vector", gdf=gdf, crs=gdf.crs,
                            geojson=geo.gdf_to_wgs84_geojson(gdf))
    resp = invoke(app, (f"{ids.COST_FEATURE_INFO}.children",),
                  ["nutzart", "bez"], layer.id,
                  triggered=[f"{ids.COST_FEATURE_KEYS}.value"])
    rendered = json.dumps(resp)
    assert "nutzart: 2 categories" in rendered
    assert "distinct" in rendered


# ------------------------------------------------ browse dialogs (issue 3)
def test_browse_button_fills_input(app, monkeypatch):
    from pyorps.gui.services import dialogs

    monkeypatch.setattr(dialogs, "ask_open",
                        lambda kind, title="": ("C:/data/x.geojson", None))
    resp = invoke(app, (f"{ids.LOCAL_PATH}.value", "browse-local-path"), 1,
                  triggered=[f"browse-{ids.LOCAL_PATH}.n_clicks"])
    assert resp[ids.LOCAL_PATH]["value"] == "C:/data/x.geojson"


def test_browse_cancel_keeps_input(app, monkeypatch):
    from dash.exceptions import PreventUpdate

    from pyorps.gui.services import dialogs

    monkeypatch.setattr(dialogs, "ask_directory",
                        lambda title="": (None, None))
    resp = invoke(app,
                  (f"{ids.PROJECT_SAVE_PATH}.value", "browse-project-save"),
                  1,
                  triggered=[f"browse-{ids.PROJECT_SAVE_PATH}.n_clicks"])
    # no_update -> the component is absent from the response
    assert ids.PROJECT_SAVE_PATH not in resp


# --------------------------------- auto-select + style editing (issue 8)
@pytest.fixture()
def built(app, state, raster_path):
    invoke(app, ("layers-view.data", ids.RASTER_LOAD_BTN), 1, raster_path,
           "viridis", [], triggered=[f"{ids.RASTER_LOAD_BTN}.n_clicks"])
    raster_layer = state.layers_of_kind("raster")[0]
    resp = invoke(
        app, (f"{ids.ROUTING_STATUS}.children", ids.RUN_ROUTING_BTN),
        1, False, _draft({"source": [SOURCE], "target": [TARGET]}),
        raster_layer.id, "dijkstra", "cpu", "r1", 80, True, False,
        False, 1.0, 100, 0, False, [], [],
        triggered=[f"{ids.RUN_ROUTING_BTN}.n_clicks"])
    return resp


def test_run_auto_selects_new_route(app, state, built):
    route = state.layers_of_kind("route")[0]
    assert built[ids.EDIT_ROUTE_SELECT]["value"] == route.id
    assert state.active_route_id == route.id


def test_apply_style_and_rename(app, state, built):
    """Feature 2: name/colour/dash edited inline in the routes grid;
    group changed via the Move-to-group button."""
    route = state.layers_of_kind("route")[0]

    def cell_edit(col, data):
        invoke(app, ("layers-view.data", "routes-grid", "cellValueChanged"),
               {"data": {"id": route.id, **data}, "colId": col},
               triggered=[f"{ids.ROUTES_GRID}.cellValueChanged"])

    cell_edit("name", {"name": "Trasse Nord"})
    cell_edit("color", {"color": "#00ff00"})
    cell_edit("dash", {"dash": "dashed"})
    assert route.name == "Trasse Nord"
    assert route.style["color"] == "#00ff00"
    assert route.style["dashArray"] == "8 8"

    # move the route to a new group via the button
    invoke(app, ("layers-view.data", "route-move-btn"),
           1, [{"id": route.id}], "Gruppe A", [],
           triggered=[f"{ids.ROUTE_MOVE_BTN}.n_clicks"])
    assert route.meta["group"] == "Gruppe A"

    # the map render picks the style up
    from pyorps.gui.callbacks.layers import render_layer

    component = render_layer(route)
    assert component.options["style"]["dashArray"] == "8 8"
    assert component.options["style"]["color"] == "#00ff00"
    # solid removes the dashArray again
    cell_edit("dash", {"dash": "solid"})
    assert "dashArray" not in route.style
    # move back to ungrouped
    invoke(app, ("layers-view.data", "route-move-btn"),
           2, [{"id": route.id}], "", [],
           triggered=[f"{ids.ROUTE_MOVE_BTN}.n_clicks"])
    assert route.meta["group"] is None