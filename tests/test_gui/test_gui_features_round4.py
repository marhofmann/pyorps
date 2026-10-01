"""
Round-4 GUI features (2026-07-15):

F1  click a feature of the selected layer -> select its row in the layer table
F2  cost-raster combination table (exact spatial overlay) + click-to-select
F3  selectable raster colormap
F4  customizable preprocessing steps (buffer/set/keep/drop)
F5  custom drawn cost layer (vector + per-polygon cost column)
F6  raster algebra (add/multiply/min/max/overlay/merge, auto-resample)

Service-level tests exercise the geometry/algebra directly; headless-callback
tests exercise the Dash wiring through the shared ``invoke`` harness.
"""
import json

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, mapping, Polygon

from dash.exceptions import PreventUpdate

from pyorps.gui import ids
from pyorps.gui.services import (cost_model, combinations, geo, manual_cost,
                                 raster_algebra, tiles)
from pyorps.gui.services.rasterize import ModifierSpec, build_cost_raster

from conftest import invoke


def _key_by_input(app, out_frag, input_pred):
    """Exact callback-map key by output fragment + an INPUT-only predicate.

    ``find_callback`` concatenates inputs+state, so it can't tell a callback
    apart from one whose wiring is a strict superset (e.g. set_colormap has
    ``raster-colormap`` as an *input* while run_rasterize/combine have it as
    *state*). Matching on ``entry["inputs"]`` only resolves those.
    """
    keys = [k for k, e in app.callback_map.items()
            if out_frag in k and any(input_pred(i) for i in e.get("inputs", []))]
    assert len(keys) == 1, f"{out_frag}: {keys}"
    return keys[0]


def _by_id(component_id):
    return lambda i: i.get("id") == component_id


def _pattern(type_):
    # pattern-matching input ids are stored as JSON strings, e.g.
    # '{"id":["ALL"],"type":"layer"}'
    return lambda i: isinstance(i.get("id"), str) and \
        f'"type":"{type_}"' in i["id"]


# --------------------------------------------------------------------- helpers
def _write_tif(path, arr, *, res=1.0, ox=0.0, oy=100.0, crs="EPSG:25832"):
    transform = from_origin(ox, oy, res, res)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0],
                       width=arr.shape[1], count=1, dtype="uint16", crs=crs,
                       transform=transform, nodata=65535) as dst:
        dst.write(arr, 1)
    return str(path)


def _landuse_gdf():
    """Three land-use tiles side by side (Wald/Nadelholz, Wald/Laubholz, Weg)."""
    return gpd.GeoDataFrame(
        {"nutzart": ["Wald", "Wald", "Weg"],
         "bez": ["Nadelholz", "Laubholz", ""]},
        geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10), box(20, 0, 30, 10)],
        crs="EPSG:25832")


LANDUSE_ASSUMPTIONS = {"Wald": {"Nadelholz": 405, "Laubholz": 475, "": 405},
                       "Weg": {"": 300}}


def _add_vector(state, name, gdf):
    return state.add_layer(name, "vector", gdf=gdf, crs=gdf.crs,
                           geojson=geo.gdf_to_wgs84_geojson(gdf))


# ============================================================ F3: colormap ===
def test_set_colormap_service_changes_url(tmp_path):
    path = _write_tif(tmp_path / "c.tif",
                      np.array([[10, 20], [30, 40]], dtype="uint16"))
    layer = tiles.build_tile_layer(path, work_dir=tmp_path, colormap="viridis")
    before = layer.tile_url
    url = tiles.set_colormap(layer, "jet")
    assert layer.colormap == "jet"
    assert "jet" in url and url != before
    # a no-op colormap keeps the url
    assert tiles.set_colormap(layer, "jet") == url
    layer.tile_client.shutdown()


def test_set_colormap_callback_recolours_all_rasters(app, state, tmp_path):
    from pyorps.gui.callbacks.raster import add_raster_layer

    path = _write_tif(tmp_path / "c.tif",
                      np.array([[1, 2], [3, 4]], dtype="uint16"))
    layer, _ = add_raster_layer(state, path, [])
    # colormap re-colours the served tiles in place (pattern-matching output),
    # never rebuilding the vector host
    tid = {"type": ids.TYPE_RASTER_TILE, "id": layer.id}
    invoke(app, (ids.TYPE_RASTER_TILE, ids.RASTER_COLORMAP), "turbo", [tid], [],
           outputs_list=[[{"id": tid, "property": "url"}],
                         {"id": ids.NOTICES, "property": "data"}],
           triggered=[f"{ids.RASTER_COLORMAP}.value"])
    assert layer.tile.colormap == "turbo"
    assert layer.meta["colormap"] == "turbo"


# ================================================ F4: preprocessing steps ===
def test_steps_preprocessor_operations():
    from pyorps.gui.presets import make_steps_preprocessor

    gdf = _landuse_gdf()
    grown = make_steps_preprocessor(
        [{"op": "buffer", "column": "nutzart", "operator": "==",
          "value": "Weg", "target": "", "arg": "5"}])(gdf.copy())
    assert grown.geometry.iloc[2].area > gdf.geometry.iloc[2].area
    assert grown.geometry.iloc[0].area == gdf.geometry.iloc[0].area

    kept = make_steps_preprocessor(
        [{"op": "keep", "column": "nutzart", "operator": "==",
          "value": "Wald", "target": "", "arg": ""}])(gdf.copy())
    assert set(kept["nutzart"]) == {"Wald"}


def test_build_cost_raster_applies_buffer_step(tmp_path):
    """A buffer step grows the covered (non-forbidden) area of the raster."""
    gdf = gpd.GeoDataFrame(
        {"nutzart": ["Wald"], "bez": [""]},
        geometry=[box(10, 10, 20, 20)], crs="EPSG:25832")
    common = dict(base_gdf=gdf, assumptions={"Wald": {"": 405}},
                  feature_keys=("nutzart", "bez"), resolution_in_m=1.0,
                  work_dir=tmp_path, use_cache=False)

    plain_path, _ = build_cost_raster(**common)
    step = [{"op": "buffer", "column": "nutzart", "operator": "all",
             "value": "", "target": "", "arg": "6"}]
    buff_path, log = build_cost_raster(preprocessor_steps=step, **common)

    def covered(path):
        with rasterio.open(path) as src:
            band = src.read(1)
        return int((band != 65535).sum())

    assert covered(buff_path) > covered(plain_path)
    assert any("custom step" in line for line in log)


def test_add_and_delete_preproc_step(app, state):
    resp = invoke(app, (f"{ids.PREPROC_GRID}.rowData", ids.PREPROC_ADD_BTN),
                  1, [], triggered=[f"{ids.PREPROC_ADD_BTN}.n_clicks"])
    rows = resp[ids.PREPROC_GRID]["rowData"]
    assert len(rows) == 1 and rows[0]["op"] == "buffer"
    resp = invoke(app, (f"{ids.PREPROC_GRID}.rowData", ids.PREPROC_DEL_BTN),
                  1, rows, [rows[0]],
                  triggered=[f"{ids.PREPROC_DEL_BTN}.n_clicks"])
    assert resp[ids.PREPROC_GRID]["rowData"] == []


# ================================================= F5: manual cost layer ===
def _drawn(*polys):
    return {"type": "FeatureCollection",
            "features": [{"type": "Feature", "properties": {},
                          "geometry": mapping(p)} for p in polys]}


def test_manual_layer_from_drawn():
    poly = Polygon([(9.0, 50.5), (9.001, 50.5), (9.001, 50.501),
                    (9.0, 50.501)])
    gdf = manual_cost.layer_from_drawn(
        _drawn(poly)["features"], "EPSG:25832", default_cost=1000,
        mode="override", name="Manual")
    assert str(gdf.crs) == "EPSG:25832"
    assert list(gdf.columns) == ["name", "cost", "mode", "geometry"]
    assert gdf["cost"].tolist() == [1000.0]

    with pytest.raises(ValueError, match="No polygons drawn"):
        manual_cost.layer_from_drawn([], "EPSG:25832", default_cost=1)


def test_create_manual_layer_callback(app, state):
    poly = Polygon([(9.0, 50.5), (9.002, 50.5), (9.002, 50.502),
                    (9.0, 50.502)])
    resp = invoke(app, ("layers-view.data", ids.MANUAL_CREATE_BTN),
                  1, _drawn(poly), "My costs", 5000, "override", [], [],
                  triggered=[f"{ids.MANUAL_CREATE_BTN}.n_clicks"])
    layers = state.layers_of_kind("vector")
    assert len(layers) == 1
    layer = layers[0]
    assert layer.meta["manual_cost"] is True
    assert layer.gdf["cost"].tolist() == [5000.0]
    assert resp[ids.MANUAL_GRID]["rowData"][0]["cost"] == 5000.0
    assert resp[ids.MANUAL_STATUS]["children"].startswith("Created")


def test_manual_polygon_live_sync_and_edit(app, state):
    """Task 33: draw target 'cost' live-syncs polygons; grid edits write back."""
    from pyorps.gui.services.manual_cost import MANUAL_LAYER_ID

    two = _drawn(box(0, 0, 1, 1), box(2, 2, 3, 3))
    # drawing with target='cost' builds the editable manual layer (on_draw is
    # the only callback with a study-area-info output + draw-control input)
    resp = invoke(app, ("study-area-info.children", ids.DRAW_CONTROL),
                  two, "cost", "Zone", 100, "override", [], [],
                  triggered=[f"{ids.DRAW_CONTROL}.geojson"])
    rows = resp[ids.MANUAL_GRID]["rowData"]
    assert len(rows) == 2 and rows[0]["cost"] == 100.0
    layer = state.get(MANUAL_LAYER_ID)
    assert layer is not None and len(layer.gdf) == 2

    # editing a cost cell in the Cost-tab grid writes back to the layer
    invoke(app, ("layers-view.data", ids.MANUAL_GRID, "cellValueChanged"),
           {"data": {"__row": 1, "cost": "777"}, "colId": "cost"},
           triggered=[f"{ids.MANUAL_GRID}.cellValueChanged"])
    assert float(state.get(MANUAL_LAYER_ID).gdf["cost"].iloc[1]) == 777.0

    # deleting a polygon (draw control now has one shape) shrinks the layer
    invoke(app, ("study-area-info.children", ids.DRAW_CONTROL),
           _drawn(box(0, 0, 1, 1)), "cost", "Zone", 100, "override",
           rows, [], triggered=[f"{ids.DRAW_CONTROL}.geojson"])
    assert len(state.get(MANUAL_LAYER_ID).gdf) == 1


def test_per_feature_override_modifier_expands(state):
    from pyorps.gui.callbacks.raster import _build_modifiers

    gdf = manual_cost.layer_from_drawn(
        _drawn(box(0, 0, 1, 1), box(2, 2, 3, 3))["features"], "EPSG:25832",
        default_cost=100, name="Manual")
    gdf.loc[0, "cost"] = 500
    gdf.loc[1, "cost"] = 2000
    _add_vector(state, "Manual", gdf)

    specs = _build_modifiers(state, [{"dataset": "Manual", "column": "cost",
                                      "operator": "all", "mode": "per-feature",
                                      "buffer_m": 0}])
    factors = sorted(int(s.factor) for s in specs)
    assert factors == [500, 2000]
    assert all(s.mode == "override" for s in specs)


def test_edit_manual_cost_in_table_writes_back(app, state):
    gdf = manual_cost.layer_from_drawn(
        _drawn(box(0, 0, 1, 1))["features"], "EPSG:25832", default_cost=100,
        name="Manual")
    layer = state.add_layer("Manual", "vector", gdf=gdf, crs=gdf.crs,
                            geojson=geo.gdf_to_wgs84_geojson(gdf),
                            meta={"manual_cost": True, "cost_column": "cost"})
    event = [{"colId": "cost", "data": {"__row": 0, "cost": "777"}}]
    invoke(app, ("notices.data", "layer-table-grid", "cellValueChanged"),
           event, [{"id": layer.id}],
           triggered=[f"{ids.LAYER_TABLE_GRID}.cellValueChanged"])
    assert float(state.get(layer.id).gdf["cost"].iloc[0]) == 777.0


# ==================================================== F6: raster algebra ===
@pytest.mark.parametrize("op,expected", [
    ("add", [[11, 22], [33, 65535]]),
    ("multiply", [[10, 40], [90, 65535]]),
    ("min", [[1, 2], [3, 4]]),
    ("max", [[10, 20], [30, 4]]),
    ("overlay", [[10, 20], [30, 4]]),
])
def test_combine_rasters_operations(tmp_path, op, expected):
    a = _write_tif(tmp_path / "a.tif",
                   np.array([[10, 20], [30, 65535]], dtype="uint16"))
    b = _write_tif(tmp_path / "b.tif",
                   np.array([[1, 2], [3, 4]], dtype="uint16"))
    out, _ = raster_algebra.combine_rasters([a, b], op, work_dir=tmp_path)
    with rasterio.open(out) as src:
        assert src.read(1).tolist() == expected


def test_combine_rasters_resamples_to_reference(tmp_path):
    a = _write_tif(tmp_path / "a.tif",
                   np.array([[10, 20], [30, 40]], dtype="uint16"))
    fine = np.array([[5, 5, 6, 6], [5, 5, 6, 6],
                     [7, 7, 8, 8], [7, 7, 8, 8]], dtype="uint16")
    b = _write_tif(tmp_path / "b.tif", fine, res=0.5)
    out, _ = raster_algebra.combine_rasters([a, b], "add", work_dir=tmp_path)
    with rasterio.open(out) as src:
        assert src.read(1).shape == (2, 2)              # reference grid


def test_combine_rasters_needs_two():
    with pytest.raises(ValueError, match="at least two"):
        raster_algebra.combine_rasters(["only-one.tif"], "add")


def test_combine_rasters_callback(app, state, tmp_path):
    from pyorps.gui.callbacks.raster import add_raster_layer

    a = _write_tif(tmp_path / "a.tif",
                   np.array([[10, 20], [30, 40]], dtype="uint16"))
    b = _write_tif(tmp_path / "b.tif",
                   np.array([[1, 2], [3, 4]], dtype="uint16"))
    la, _ = add_raster_layer(state, a, [], name="A")
    lb, _ = add_raster_layer(state, b, [], name="B")
    resp = invoke(app, ("layers-view.data", ids.RASTER_COMBINE_BTN),
                  1, [la.id, lb.id], "add", "viridis", [],
                  triggered=[f"{ids.RASTER_COMBINE_BTN}.n_clicks"])
    rasters = state.layers_of_kind("raster")
    assert len(rasters) == 3
    assert "combined_from" in rasters[-1].meta
    assert "add" in resp[ids.RASTER_COMBINE_STATUS]["children"]


def test_combine_rasters_callback_needs_two(app, state, tmp_path):
    from pyorps.gui.callbacks.raster import add_raster_layer

    a = _write_tif(tmp_path / "a.tif",
                   np.array([[10, 20]], dtype="uint16"))
    la, _ = add_raster_layer(state, a, [], name="A")
    resp = invoke(app, ("layers-view.data", ids.RASTER_COMBINE_BTN),
                  1, [la.id], "add", "viridis", [],
                  triggered=[f"{ids.RASTER_COMBINE_BTN}.n_clicks"])
    assert resp[ids.NOTICES]["data"][-1]["title"] == \
        "Select at least two rasters"


# ============================================= F2: combination table ===
def test_combination_table_overlay_and_locate():
    base = _landuse_gdf()
    zone = gpd.GeoDataFrame({"ZONE": ["Schutzzone I"]},
                            geometry=[box(0, 0, 15, 10)], crs="EPSG:25832")
    mod = ModifierSpec(gdf=zone, mode="multiply", factor=100,
                       name="TWS_HQS_TK25", condition="ZONE == Schutzzone I")
    combo = combinations.build_combination_table(
        base, ("nutzart", "bez"), LANDUSE_ASSUMPTIONS,
        base_layer_name="ave_nutzung", modifiers=[mod], base_crs=base.crs)

    by_cost = {int(r["cost"]): r for _, r in combo.iterrows()}
    assert set(by_cost) == {300, 475, 40500, 47500}
    # example c: Wald/Nadelholz + protection zone -> 405 x 100
    assert "TWS_HQS_TK25" in by_cost[40500]["modifiers"]
    assert by_cost[40500]["nutzart"] == "Wald"

    # click inside the modified Nadelholz region -> that combination row
    row = combinations.locate(combo, 5, 5)
    assert int(combo[combo["__row"] == row].iloc[0]["cost"]) == 40500
    # click in the untouched Laubholz strip (x 15..20) -> 475
    row = combinations.locate(combo, 17, 5)
    assert int(combo[combo["__row"] == row].iloc[0]["cost"]) == 475
    # a point outside every region
    assert combinations.locate(combo, 999, 999) is None


def test_combination_forbidden_base_unchanged_by_modifier():
    base = gpd.GeoDataFrame(
        {"nutzart": ["Wohnbaufläche"], "bez": [""]},
        geometry=[box(0, 0, 10, 10)], crs="EPSG:25832")
    zone = gpd.GeoDataFrame({"z": ["a"]}, geometry=[box(0, 0, 10, 10)],
                            crs="EPSG:25832")
    mod = ModifierSpec(gdf=zone, mode="multiply", factor=100, name="Z",
                       condition="all")
    combo = combinations.build_combination_table(
        base, ("nutzart", "bez"), {"Wohnbaufläche": {"": 65535}},
        modifiers=[mod], base_crs=base.crs)
    # forbidden stays forbidden (ignore_value default) and no modifier recorded
    assert combo["cost"].tolist() == [65535]
    assert combo.iloc[0]["modifiers"] == ""


def test_combination_row_order_and_colour():
    combo = combinations.build_combination_table(
        _landuse_gdf(), ("nutzart", "bez"), LANDUSE_ASSUMPTIONS,
        base_layer_name="ave_nutzung", base_crs="EPSG:25832")
    # __row equals the (cost-sorted) display position — so scrollTo lands right
    assert combo["__row"].tolist() == list(range(len(combo)))
    assert combo["cost"].tolist() == sorted(combo["cost"].tolist())
    # colour-coded grid: a safe styleConditions swatch column + per-row colour
    cds, rows, _ = combinations.grid_payload(combo, "R", colormap="jet")
    assert cds[0]["field"] == "_swatch"
    assert "styleConditions" in cds[0]["cellStyle"]
    assert all(r["_color"] is None or r["_color"].startswith("#") for r in rows)


def test_legend_items_map_cost_colour_combination():
    combo = combinations.build_combination_table(
        _landuse_gdf(), ("nutzart", "bez"), LANDUSE_ASSUMPTIONS,
        base_layer_name="ave_nutzung", base_crs="EPSG:25832")
    items = combinations.legend_items(combo, "jet")
    assert len(items) == len(combo)
    costs = [c for c, _clr, _lbl in items]
    assert costs == sorted(costs)
    # each entry ties a cost to a colour and a layer/feature label
    assert any("Wald" in lbl for _c, _clr, lbl in items)


def test_colormap_legend_callback(app, state):
    combo = combinations.build_combination_table(
        _landuse_gdf(), ("nutzart", "bez"), LANDUSE_ASSUMPTIONS,
        base_layer_name="ave_nutzung", base_crs="EPSG:25832")
    raster = state.add_layer("Cost raster", "raster", crs="EPSG:25832",
                             meta={"combination_gdf": combo, "vmin": 300,
                                   "vmax": 475})
    resp = invoke(app, (f"{ids.COLORMAP_LEGEND}.children", ids.LAYERS_GRID),
                  [{"id": raster.id}], "turbo", "",
                  triggered=[f"{ids.LAYERS_GRID}.selectedRows"])
    rendered = json.dumps(resp[ids.COLORMAP_LEGEND])
    assert "backgroundColor" in rendered and "Wald" in rendered


def test_open_raster_combination_table_callback(app, state):
    combo = combinations.build_combination_table(
        _landuse_gdf(), ("nutzart", "bez"), LANDUSE_ASSUMPTIONS,
        base_layer_name="ave_nutzung", base_crs="EPSG:25832")
    raster = state.add_layer("Cost raster", "raster", crs="EPSG:25832",
                             meta={"combination_gdf": combo})
    resp = invoke(app, (f"{ids.LAYER_TABLE_GRID}.rowData", ids.LAYER_VIEW_BTN),
                  1, [{"id": raster.id}],
                  triggered=[f"{ids.LAYER_VIEW_BTN}.n_clicks"])
    assert resp[ids.LAYER_TABLE_OFFCANVAS]["is_open"] is True
    fields = [c["field"] for c in resp[ids.LAYER_TABLE_GRID]["columnDefs"]]
    assert "cost" in fields and "modifiers" in fields
    assert len(resp[ids.LAYER_TABLE_GRID]["rowData"]) == len(combo)


def test_click_raster_selects_combination_row(app, state):
    # build the combo table in the raster CRS and click a WGS84 point over it
    lat, lng = 50.5, 9.0
    x, y = geo.point_wgs84_to_crs(lat, lng, "EPSG:25832")
    base = gpd.GeoDataFrame(
        {"nutzart": ["Wald"], "bez": ["Nadelholz"]},
        geometry=[box(x - 50, y - 50, x + 50, y + 50)], crs="EPSG:25832")
    combo = combinations.build_combination_table(
        base, ("nutzart", "bez"), LANDUSE_ASSUMPTIONS,
        base_layer_name="ave_nutzung", base_crs="EPSG:25832")
    raster = state.add_layer("Cost raster", "raster", crs="EPSG:25832",
                             meta={"combination_gdf": combo})
    click = {"latlng": {"lat": lat, "lng": lng}}
    key = _key_by_input(app, "paginationGoTo", _by_id(ids.MAP))
    resp = invoke(app, key, click, [{"id": raster.id}],
                  {"click_mode": "off"}, triggered=[f"{ids.MAP}.clickData"])
    selected = resp[ids.LAYER_TABLE_GRID]["selectedRows"]
    assert selected and int(selected[0]["cost"]) == 405     # Wald/Nadelholz
    assert selected[0]["bez"] == "Nadelholz"
    # page-jump: the combination row sits on its computed page (task 53) and
    # is scrolled into view within that page (task 61)
    assert resp[ids.LAYER_TABLE_GRID]["paginationGoTo"] == 0
    assert resp[ids.LAYER_TABLE_GRID]["scrollTo"] == {"rowIndex": 0}
    assert resp[ids.LAYER_TABLE_OFFCANVAS]["is_open"] is True


# =================================== F1: feature click -> table row ===
def test_feature_row_index_matches_by_id_and_geometry():
    from pyorps.gui.callbacks.layers import _feature_row_index

    gdf = _landuse_gdf()
    layer = type("L", (), {})()
    layer.geojson = geo.gdf_to_wgs84_geojson(gdf)
    feat = layer.geojson["features"][1]
    assert _feature_row_index(layer, feat) == 1
    # a feature carrying only geometry still resolves
    assert _feature_row_index(
        layer, {"geometry": feat["geometry"]}) == 1


def test_click_feature_selects_table_row(app, state):
    gdf = _landuse_gdf()
    layer = _add_vector(state, "Landuse", gdf)
    feat = layer.geojson["features"][2]
    key = _key_by_input(app, "paginationGoTo", _pattern(ids.TYPE_LAYER_GEOJSON))
    resp = invoke(
        app, key, [None], [{"id": layer.id}],
        triggered=[{"id": {"type": ids.TYPE_LAYER_GEOJSON, "id": layer.id},
                    "property": "clickData", "value": feat}])
    assert resp[ids.LAYER_TABLE_OFFCANVAS]["is_open"] is True
    selected = resp[ids.LAYER_TABLE_GRID]["selectedRows"]
    assert selected and selected[0]["__row"] == 2
    assert resp[ids.LAYER_TABLE_GRID]["paginationGoTo"] == 0
    assert resp[ids.LAYER_TABLE_GRID]["scrollTo"] == {"rowIndex": 2}


def test_click_feature_of_unselected_layer_ignored(app, state):
    gdf = _landuse_gdf()
    layer = _add_vector(state, "Landuse", gdf)
    other = _add_vector(state, "Other", gdf)
    feat = layer.geojson["features"][0]
    key = _key_by_input(app, "paginationGoTo", _pattern(ids.TYPE_LAYER_GEOJSON))
    with pytest.raises(PreventUpdate):
        invoke(app, key, [None], [{"id": other.id}],
               triggered=[{"id": {"type": ids.TYPE_LAYER_GEOJSON,
                                  "id": layer.id},
                           "property": "clickData", "value": feat}])
