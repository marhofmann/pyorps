"""Round-11 features: layer merge, hi-res DEM/DSM presets, categorized OSM
menu + Overpass fallback, condition-group preprocessing steps, multi-select
layer ops + route-group rows, rasterize dataset/table pairing, default
z-order bands, draft consumption + re-add, map-click route select, pretty
attributes."""
import json

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point, Polygon

from pyorps.gui import ids, presets
from pyorps.gui.services import data_io, osm

from conftest import invoke

CRS = "EPSG:32632"


def _gdf(names, crs=CRS, extra=None):
    rows = []
    for i, n in enumerate(names):
        row = {"nutzart": n, "bez": f"b{i}"}
        row.update(extra or {})
        rows.append(row)
    return gpd.GeoDataFrame(
        rows, geometry=[Point(500000 + i, 5599000 + i).buffer(5)
                        for i in range(len(names))], crs=crs)


# --------------------------------------------------------- merge vector layers
def test_merge_gdfs_aligns_columns_and_crs():
    a = _gdf(["Wald", "Weg"])
    b = _gdf(["Acker"], crs="EPSG:25832", extra={"only_b": "x"})
    merged = data_io.merge_gdfs([a, b])
    assert len(merged) == 3
    assert str(merged.crs) == CRS                    # first layer's CRS wins
    assert "only_b" in merged.columns                # union of columns
    assert merged["only_b"].isna().sum() == 2        # missing stays empty


def test_merge_gdfs_needs_two():
    with pytest.raises(ValueError, match="at least two"):
        data_io.merge_gdfs([_gdf(["Wald"])])


def test_merge_callback_creates_layer(app, state):
    la = state.add_layer("A", "vector", gdf=_gdf(["Wald"]), crs=CRS)
    lb = state.add_layer("B", "vector", gdf=_gdf(["Acker"]), crs=CRS)
    state.project_crs = CRS
    resp = invoke(app, ("layers-view.data", ids.MERGE_BTN),
                  1, [la.id, lb.id], "Cross-state", [],
                  triggered=[f"{ids.MERGE_BTN}.n_clicks"])
    merged = [ly for ly in state.layers_of_kind("vector")
              if ly.name == "Cross-state"]
    assert len(merged) == 1
    assert len(merged[0].gdf) == 2
    assert merged[0].meta["source"]["kind"] == "merge"
    assert resp[ids.NOTICES]["data"][-1]["severity"] == "success"
    # re-merging the same members UPDATES the layer instead of duplicating
    invoke(app, ("layers-view.data", ids.MERGE_BTN),
           2, [la.id, lb.id], "Cross-state", [],
           triggered=[f"{ids.MERGE_BTN}.n_clicks"])
    assert len([ly for ly in state.layers_of_kind("vector")
                if ly.meta.get("source", {}).get("kind") == "merge"]) == 1


def test_merge_callback_needs_two(app, state):
    la = state.add_layer("A", "vector", gdf=_gdf(["Wald"]), crs=CRS)
    resp = invoke(app, ("layers-view.data", ids.MERGE_BTN),
                  1, [la.id], "", [],
                  triggered=[f"{ids.MERGE_BTN}.n_clicks"])
    assert resp[ids.NOTICES]["data"][-1]["title"] == \
        "Pick at least two vector layers"


# ------------------------------------------------- hi-res DEM / DSM presets
def test_highres_elevation_presets_registered():
    wcs = [s for s in presets.MAP_SERVICES if s["service"] == "wcs"]
    by_state = {(s["state"], s["category"]): s for s in wcs}
    hessen = by_state[("HE", "Terrain (DEM/DGM)")]
    assert hessen["layer"] == "he_dgm1" and hessen["res"] == 1
    assert by_state[("HE", "Surface (DSM/DOM)")]["layer"] == "dom1"
    nrw = by_state[("NW", "Terrain (DEM/DGM)")]
    assert nrw["max_px"] == 2000 * 2000 and nrw["axis"] == ["x", "y"]
    bb = by_state[("BB", "Terrain (DEM/DGM)")]
    assert bb["crs"] == "EPSG:25833"                 # UTM 33N, not 32N!
    assert ("ST", "Terrain (DEM/DGM)") in by_state
    assert ("BW", "Terrain (DEM/DGM)") in by_state


def test_dem_request_blocked_over_service_limit(app, state):
    """A 1 m service with a documented per-request cap must refuse an area
    that exceeds it — with a readable notice, not a server error."""
    # ~10 km x 10 km study area (WGS84) >> NRW's 2000x2000 px cap at 1 m
    poly = Polygon([(7.0, 51.0), (7.14, 51.0), (7.14, 51.09), (7.0, 51.09)])
    state.study_area = {"type": "Feature", "properties": {},
                       "geometry": json.loads(
                           gpd.GeoSeries([poly]).to_json())[
                           "features"][0]["geometry"]}
    nrw = next(s for s in presets.MAP_SERVICES
               if s.get("layer") == "nw_dgm")
    preset = f"wcs|{nrw['url']}|{nrw['layer']}"
    resp = invoke(app, ("map-view.data", ids.DEM_LOAD_BTN),
                  1, preset, [],
                  triggered=[f"{ids.DEM_LOAD_BTN}.n_clicks"])
    notice = resp[ids.NOTICES]["data"][-1]
    assert notice["title"] == "Study area too large for this elevation service"
    assert notice["severity"] == "error"


# ----------------------------------------- categorized OSM menu + fallback
def test_osm_filters_from_selections():
    filters = osm.filters_from_selections([
        {"key": "landuse", "values": []},
        {"key": "highway", "values": ["primary"]},
        {"key": "natural", "values": ["wood", "water"]},
    ])
    assert filters == [
        'nwr["landuse"]',
        'nwr["highway"="primary"]',
        'nwr["natural"~"^(wood|water)$"]',
    ]
    assert osm.selection_label({"key": "landuse", "values": []}) == \
        "landuse = any"


def test_osm_key_values_catalog():
    assert "forest" in osm.OSM_KEY_VALUES["landuse"]
    assert "line" in osm.OSM_KEY_VALUES["power"]


def test_overpass_falls_back_to_mirror(monkeypatch):
    """A busy primary (504) must NOT be a dead end — the next mirror serves."""
    calls = []

    class Resp:
        def __init__(self, code, payload=None):
            self.status_code = code
            self._payload = payload

        def json(self):
            return self._payload

    def fake_post(url, **kwargs):
        calls.append(url)
        if len(calls) == 1:
            return Resp(504)
        return Resp(200, {"elements": [
            {"type": "node", "id": 1, "lat": 50.0, "lon": 9.0,
             "tags": {"power": "tower"}}]})

    import requests
    monkeypatch.setattr(requests, "post", fake_post)
    gdf = osm.load_osm_features((50, 9, 51, 10), ['nwr["power"]'])
    assert len(calls) == 2                     # primary busy -> mirror hit
    assert calls[0] == osm.OVERPASS_ENDPOINTS[0]
    assert calls[1] == osm.OVERPASS_ENDPOINTS[1]
    assert len(gdf) == 1


def test_overpass_all_busy_raises_readable(monkeypatch):
    class Resp:
        status_code = 504

        def json(self):  # pragma: no cover
            return {}

    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp())
    with pytest.raises(ValueError, match="All Overpass servers"):
        osm.load_osm_features((50, 9, 51, 10), ['nwr["power"]'])


def test_osm_menu_callbacks(app, state):
    # key -> values options
    resp = invoke(app, (f"{ids.OSM_VALUES}.options", ids.OSM_KEY), "landuse")
    values = [o["value"] for o in resp[ids.OSM_VALUES]["options"]]
    assert "forest" in values
    # add two combinations
    resp = invoke(app, (f"{ids.OSM_SELECTIONS}.data", ids.OSM_ADD_BTN),
                  1, "landuse", ["forest"], [],
                  triggered=[f"{ids.OSM_ADD_BTN}.n_clicks"])
    sels = resp[ids.OSM_SELECTIONS]["data"]
    assert sels == [{"key": "landuse", "values": ["forest"]}]
    # chips render with a remove button each
    resp = invoke(app, (f"{ids.OSM_SELECTION_LIST}.children",), sels)
    assert "landuse = forest" in json.dumps(resp[ids.OSM_SELECTION_LIST])
    # remove chip 0
    resp = invoke(app, (f"{ids.OSM_SELECTIONS}.data",
                        ids.TYPE_OSM_SEL_REMOVE),
                  [1], sels,
                  triggered=[{"id": {"type": ids.TYPE_OSM_SEL_REMOVE,
                                     "index": 0},
                              "property": "n_clicks", "value": 1}])
    assert resp[ids.OSM_SELECTIONS]["data"] == []


def test_load_osm_uses_selections(app, state, monkeypatch):
    captured = {}

    def fake(bbox, filters, **kwargs):
        captured["filters"] = filters
        return _gdf(["x"]).to_crs("EPSG:4326")

    monkeypatch.setattr(osm, "load_osm_features", fake)
    poly = Polygon([(9.0, 50.0), (9.01, 50.0), (9.01, 50.01), (9.0, 50.01)])
    state.study_area = {"type": "Polygon", "coordinates":
                        [list(poly.exterior.coords)]}
    invoke(app, ("layers-view.data", ids.OSM_LOAD_BTN),
           1, None, "", [], [{"key": "power", "values": ["line", "cable"]}],
           triggered=[f"{ids.OSM_LOAD_BTN}.n_clicks"])
    assert captured["filters"] == ['nwr["power"~"^(line|cable)$"]']


# --------------------------------------- condition-group preprocessing steps
def test_step_label_python_like():
    step = {"conditions": [
        {"column": "nutzart", "operator": "==", "value": "Wald"},
        {"column": "bez", "operator": "==", "value": "Nadelholz"}],
        "combine": "&", "op": "buffer", "arg": 2}
    assert presets.step_label(step) == \
        '(("nutzart" == "Wald") & ("bez" == "Nadelholz")) -> buffer=2m'
    single = {"conditions": [{"column": "zone", "operator": ">",
                              "value": 3}],
              "combine": "&", "op": "set", "target": "cost", "arg": 99}
    assert presets.step_label(single) == '("zone" > 3) -> set "cost"=99'


def test_step_mask_and_or():
    gdf = _gdf(["Wald", "Wald", "Acker"])
    gdf.loc[:, "bez"] = ["Nadelholz", "Laubholz", "Nadelholz"]
    both = {"conditions": [
        {"column": "nutzart", "operator": "==", "value": "Wald"},
        {"column": "bez", "operator": "==", "value": "Nadelholz"}],
        "combine": "&"}
    assert list(presets.step_mask(gdf, both)) == [True, False, False]
    either = dict(both, combine="|")
    assert list(presets.step_mask(gdf, either)) == [True, True, True]


def test_steps_preprocessor_group_buffer():
    gdf = _gdf(["Wald", "Acker"])
    gdf.loc[:, "bez"] = ["Nadelholz", "Nadelholz"]
    step = {"op": "buffer", "arg": "2", "combine": "&", "conditions": [
        {"column": "nutzart", "operator": "==", "value": "Wald"},
        {"column": "bez", "operator": "==", "value": "Nadelholz"}]}
    before = gdf.geometry.area.copy()
    out = presets.make_steps_preprocessor([step])(gdf.copy())
    assert out.geometry.iloc[0].area > before.iloc[0] * 1.5   # buffered
    assert abs(out.geometry.iloc[1].area - before.iloc[1]) < 1e-6


def test_ppb_builder_callbacks(app, state):
    layer = state.add_layer("D", "vector", gdf=_gdf(["Wald", "Acker"]),
                            crs=CRS)
    # columns of the picked dataset
    resp = invoke(app, (f"{ids.PPB_COLUMN}.options", ids.PPB_DATASET),
                  layer.id)
    cols = [o["value"] for o in resp[ids.PPB_COLUMN]["options"]]
    assert "nutzart" in cols and "geometry" not in cols
    # values present in the picked column (searchable dropdown data)
    resp = invoke(app, (f"{ids.PPB_VALUE}.options", ids.PPB_COLUMN),
                  "nutzart", layer.id)
    vals = [o["value"] for o in resp[ids.PPB_VALUE]["options"]]
    assert set(vals) == {"Wald", "Acker"}
    # add a condition
    resp = invoke(app, (f"{ids.PPB_CONDS}.data", ids.PPB_ADD_COND_BTN),
                  1, "nutzart", "==", "Wald", [],
                  triggered=[f"{ids.PPB_ADD_COND_BTN}.n_clicks"])
    conds = resp[ids.PPB_CONDS]["data"]
    assert conds == [{"column": "nutzart", "operator": "==",
                      "value": "Wald"}]
    # live preview shows the python-like mask
    resp = invoke(app, (f"{ids.PPB_PREVIEW}.children", ids.PPB_CONDS),
                  conds, "&", "buffer", "", "2")
    assert resp[ids.PPB_PREVIEW]["children"] == \
        '("nutzart" == "Wald") -> buffer=2m'
    # add the step -> lands in the grid with the label; conditions cleared
    resp = invoke(app, (f"{ids.PREPROC_GRID}.rowData",
                        ids.PPB_ADD_STEP_BTN),
                  1, conds, "&", "buffer", "", "2", layer.id, [],
                  triggered=[f"{ids.PPB_ADD_STEP_BTN}.n_clicks"])
    rows = resp[ids.PREPROC_GRID]["rowData"]
    assert len(rows) == 1
    assert rows[0]["dataset"] == "D"
    assert rows[0]["conditions"] == conds
    assert rows[0]["condition"] == '("nutzart" == "Wald") -> buffer=2m'
    assert resp[ids.PPB_CONDS]["data"] == []


# -------------------------- multi-select layer ops + route-group grid rows
def _routes_with_groups(state):
    line = LineString([(0, 0), (10, 10)])
    routes = []
    for i, group in enumerate(["G1", "G1", None]):
        layer = state.add_layer(
            f"Route {i + 1}", "route",
            gdf=gpd.GeoDataFrame([{"name": f"Route {i + 1}"}],
                                 geometry=[line], crs=CRS),
            crs=CRS, meta={"group": group,
                           "control_points": [[0, 0], [10, 10]]})
        routes.append(layer)
    return routes


def test_layers_grid_groups_routes(app, state):
    state.add_layer("Data", "vector", gdf=_gdf(["Wald"]), crs=CRS)
    _routes_with_groups(state)
    resp = invoke(app, (f"{ids.LAYERS_GRID}.rowData",), state.layers_view())
    rows = resp[ids.LAYERS_GRID]["rowData"]
    ids_shown = [r["id"] for r in rows]
    assert "group::G1" in ids_shown and "group::(ungrouped)" in ids_shown
    assert not any(r.get("kind") == "route" for r in rows)   # no single rows
    g1 = next(r for r in rows if r["id"] == "group::G1")
    assert g1["kind"] == "routes (2)" and g1["name"] == "G1"


def test_group_row_visibility_and_rename(app, state):
    routes = _routes_with_groups(state)
    event = [{"colId": "visible",
              "data": {"id": "group::G1", "name": "G1", "visible": False}}]
    invoke(app, ("layers-view.data", ids.LAYERS_GRID, "cellValueChanged"),
           event, triggered=[f"{ids.LAYERS_GRID}.cellValueChanged"])
    assert routes[0].visible is False and routes[1].visible is False
    assert routes[2].visible is True
    event = [{"colId": "name",
              "data": {"id": "group::G1", "name": "Renamed", "visible":
                       False}}]
    invoke(app, ("layers-view.data", ids.LAYERS_GRID, "cellValueChanged"),
           event, triggered=[f"{ids.LAYERS_GRID}.cellValueChanged"])
    assert routes[0].meta["group"] == "Renamed"
    assert routes[1].meta["group"] == "Renamed"


def test_multi_select_move_and_remove(app, state):
    a = state.add_layer("A", "vector", gdf=_gdf(["w"]), crs=CRS)
    b = state.add_layer("B", "vector", gdf=_gdf(["x"]), crs=CRS)
    c = state.add_layer("C", "vector", gdf=_gdf(["y"]), crs=CRS)
    # ctrl+click selection of A and B moves BOTH up one step: [C, A, B]
    resp = invoke(app, ("layers-view.data", "layer-up-btn"),
                  1, None, [{"id": a.id}, {"id": b.id}],
                  triggered=["layer-up-btn.n_clicks"])
    assert [v["id"] for v in resp[ids.LAYERS_VIEW]["data"]] == \
        [c.id, a.id, b.id]
    # removing a group row removes every route in the group
    routes = _routes_with_groups(state)
    resp = invoke(app, ("layers-view.data", ids.LAYER_REMOVE_BTN),
                  1, [{"id": "group::G1"}],
                  triggered=[f"{ids.LAYER_REMOVE_BTN}.n_clicks"])
    assert state.get(routes[0].id) is None
    assert state.get(routes[1].id) is None
    assert state.get(routes[2].id) is not None       # ungrouped survives


# ------------------------------------------ rasterize dataset/table pairing
def test_rasterize_inputs_compatibility(state):
    from pyorps.gui.callbacks.raster import _rasterize_inputs

    good = state.add_layer("Good", "vector", gdf=_gdf(["Wald"]), crs=CRS)
    bad = state.add_layer("Bad", "vector",
                          gdf=gpd.GeoDataFrame(
                              [{"other": 1}],
                              geometry=[Point(0, 0).buffer(1)], crs=CRS),
                          crs=CRS)
    state.cost_tables["t1"] = {"name": "T1", "dataset_id": good.id,
                               "feature_keys": ["nutzart"],
                               "rows": [{"nutzart": "Wald", "cost": 5,
                                         "forbidden": False}]}
    layer, keys, rows = _rasterize_inputs(state, {}, [], "", "t1")
    assert layer.id == good.id and keys == ("nutzart",)
    layer, keys, rows = _rasterize_inputs(state, {}, [], good.id, "t1")
    assert layer.id == good.id
    with pytest.raises(ValueError, match="does not belong"):
        _rasterize_inputs(state, {}, [], bad.id, "t1")
    with pytest.raises(ValueError, match="Seed a cost table"):
        _rasterize_inputs(state, {}, [], "", "current")


def test_rasterize_table_options_filtered(app, state):
    good = state.add_layer("Good", "vector", gdf=_gdf(["Wald"]), crs=CRS)
    bad = state.add_layer("Bad", "vector",
                          gdf=gpd.GeoDataFrame(
                              [{"other": 1}],
                              geometry=[Point(0, 0).buffer(1)], crs=CRS),
                          crs=CRS)
    state.cost_tables["t1"] = {"name": "T1", "dataset_id": good.id,
                               "feature_keys": ["nutzart"], "rows": []}
    resp = invoke(app, (f"{ids.RASTERIZE_TABLE}.options",),
                  state.layers_view(), {}, good.id)
    values = [o["value"] for o in resp[ids.RASTERIZE_TABLE]["options"]]
    assert "t1" in values
    resp = invoke(app, (f"{ids.RASTERIZE_TABLE}.options",),
                  state.layers_view(), {}, bad.id)
    values = [o["value"] for o in resp[ids.RASTERIZE_TABLE]["options"]]
    assert "t1" not in values and "current" in values


def test_seed_registers_cost_table(app, state):
    layer = state.add_layer("D", "vector", gdf=_gdf(["Wald", "Acker"]),
                            crs=CRS)
    invoke(app, (f"{ids.COST_GRID}.columnDefs", ids.COST_SEED_BTN),
           1, layer.id, ["nutzart"], [],
           triggered=[f"{ids.COST_SEED_BTN}.n_clicks"])
    assert len(state.cost_tables) == 1
    entry = next(iter(state.cost_tables.values()))
    assert entry["dataset_id"] == layer.id
    assert entry["feature_keys"] == ["nutzart"]
    assert entry["rows"]


# ------------------------------------------------------ default z-order bands
def test_default_zorder_bands(state):
    route = state.add_layer("R", "route", geojson={})
    raster_like = state.add_layer("Ras", "raster")
    state.add_layer("V", "vector", gdf=_gdf(["w"]), crs=CRS)
    state.add_layer("W", "wms")
    state.add_layer("SA", "study_area", geojson={})
    kinds = [ly.kind for ly in state.ordered_layers()]
    # background -> foreground: study area, vector, overlays, raster, routes
    assert kinds == ["study_area", "vector", "wms", "raster", "route"]
    # a SECOND raster lands on top of its band, still below the routes
    r2 = state.add_layer("Ras2", "raster")
    order = [ly.id for ly in state.ordered_layers()]
    assert order.index(r2.id) == order.index(raster_like.id) + 1
    assert order.index(r2.id) < order.index(route.id)


# ------------------------------- draft consumption + re-add group points
def test_finalize_clears_draft_only_on_full_success(state):
    from unittest.mock import MagicMock

    from pyorps.gui.callbacks.routes import _EMPTY_DRAFT, finalize_routing

    route_cost = MagicMock(total_length=10.0, total_cost=5.0,
                           total_cell_cost=5.0, geodesic_length_m=10.0,
                           n_forbidden_cells=0, crosses_forbidden=False)
    built = MagicMock(line=LineString([(0, 0), (1, 1)]), cost=route_cost,
                      params={"graph_api": "cython"},
                      control_points=[(0, 0), (1, 1)])
    meta = {"raster_layer_id": None, "algorithm": "dijkstra",
            "simplify_tol": None, "wp_names": [], "n_sources": 1,
            "n_targets": 1, "n_waypoints": 0}
    state.project_crs = CRS
    *_, draft = finalize_routing(state, None, [built], [], meta=meta,
                                 notices=[])
    assert draft == dict(_EMPTY_DRAFT)               # success -> consumed
    *_, draft = finalize_routing(state, None, [built],
                                 [((0, 0), (1, 1), "no path")], meta=meta,
                                 notices=[])
    from dash import no_update
    assert draft is no_update                        # failures keep the list
    *_, draft = finalize_routing(state, None, [built], [], meta=meta,
                                 notices=[], cancelled=True)
    assert draft is no_update                        # stopped keeps the list


def test_readd_group_points(app, state):
    line = LineString([(500000, 5599000), (500100, 5599100)])
    state.add_layer(
        "Route 1", "route",
        gdf=gpd.GeoDataFrame([{"name": "Route 1"}], geometry=[line],
                             crs=CRS),
        crs=CRS,
        meta={"group": "Run 1 (1x1)",
              "control_points": [[500000, 5599000], [500050, 5599050],
                                 [500100, 5599100]],
              "waypoint_names": ["wp-a"]})
    resp = invoke(app, ("route-draft.data", ids.READD_POINTS_BTN),
                  1, "Run 1 (1x1)", [],
                  triggered=[f"{ids.READD_POINTS_BTN}.n_clicks"])
    draft = resp[ids.ROUTE_DRAFT]["data"]
    assert len(draft["sources"]) == 1
    assert len(draft["targets"]) == 1
    assert len(draft["waypoints"]) == 1
    assert draft["waypoints"][0]["name"] == "wp-a"
    assert draft["sources"][0]["x"] == 500000.0
    assert 0 < draft["sources"][0]["lat"] < 90       # lat/lng filled in


# ------------------------------------------- map click selects the route
def test_map_click_selects_route(app, state):
    routes = _routes_with_groups(state)
    feature = {"type": "Feature", "properties": {},
               "geometry": {"type": "LineString",
                            "coordinates": [[0, 0], [1, 1]]}}
    resp = invoke(
        app, ("edit-route-select.value", "clickData"), [feature],
        triggered=[{"id": {"type": ids.TYPE_LAYER_GEOJSON,
                           "id": routes[0].id},
                    "property": "clickData", "value": feature}])
    assert resp[ids.EDIT_ROUTE_SELECT]["value"] == routes[0].id


# ------------------------------------------------------ pretty attributes
def test_attribute_labels_and_rounding():
    from pyorps.gui.callbacks.attrs import (field_label, format_value,
                                            properties_table)

    assert field_label("total_length_m") == "Total Length (in m)"
    assert field_label("unknown_col") == "unknown_col"
    assert format_value(1234.5678) == "1,234.57"
    assert format_value(65535) == "65,535"
    assert format_value(True) == "yes"
    assert format_value(None) == ""
    assert format_value("35390") == "35390"          # strings verbatim
    rendered = str(properties_table(
        {"total_length_m": 1234.5678, "total_cost": 99.999}, "Route 1"))
    assert "Total Length (in m)" in rendered
    assert "1,234.57" in rendered and "100.00" in rendered
