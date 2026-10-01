"""
Round-8 GUI fixes (2026-07-16, live testing):

58  silence the numpy 'invalid value encountered in cast' RuntimeWarning
    (localtileserver/rio-tiler rendering the float DEM's NaN nodata)
59  WMS/DEM must fully cover the study area (expand the requested bbox outward)
60  OSM/Overpass errors → a clear notice, not the generic box
61  scroll the selected table row into view (see test_gui_features_round4)
62  the study-area shape must not become a full-area cost polygon + list remove
63  routes grid shows cost / length / source / target
64  auto-refresh toggle for route edits (see test_edit_phase5)
"""
import warnings

import pytest

from pyorps.gui import ids

from conftest import invoke


# ---------------------------------------------------------- 58 warning filter
def test_dem_cast_warning_is_filtered():
    import importlib

    import pyorps.gui.services.tiles as tiles_mod

    # pytest resets warning filters per test; reload re-registers tiles' filter
    # into the CURRENT filter list so the assertion is deterministic.
    importlib.reload(tiles_mod)
    assert any(
        getattr(f[1], "pattern", "")
        and "invalid value encountered in cast" in f[1].pattern
        for f in warnings.filters), "cast-warning filter not registered"


# --------------------------------------------------------- 59 coverage expand
def test_expand_bounds_grows_outward():
    from pyorps.gui.callbacks.data import _expand_bounds

    # small span → the absolute floor dominates
    assert _expand_bounds((0, 0, 1000, 1000), floor=300) == (
        -300, -300, 1300, 1300)
    # large span → the fractional margin dominates (2% of 100 000 = 2000)
    lo_x, lo_y, hi_x, hi_y = _expand_bounds((0, 0, 100_000, 100_000), floor=300)
    assert lo_x == -2000 and hi_x == 102_000


# --------------------------------------------------------- 60 OSM notices
def test_overpass_error_is_friendly_warning():
    from pyorps.gui.services import errors

    notice = errors.translate_exception(ValueError(
        "Overpass is rate-limiting / overloaded (HTTP 504). Wait a moment…"))
    assert notice.severity == "warning"
    assert notice.title == "OpenStreetMap server is busy"
    assert notice.focus_id == ids.TAB_DATA


def test_no_osm_features_is_warning():
    from pyorps.gui.services import errors

    notice = errors.translate_exception(ValueError(
        "No OSM features of that kind in the study area. Try a different…"))
    assert notice.severity == "warning"
    assert notice.title == "No OSM features in this area"


# --------------------------------------------------------- 62 manual cost
_AREA = {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [
    [[9.0, 50.5], [9.1, 50.5], [9.1, 50.6], [9.0, 50.6], [9.0, 50.5]]]}}
_COST_A = {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [
    [[9.20, 50.50], [9.25, 50.50], [9.25, 50.55], [9.20, 50.55],
     [9.20, 50.50]]]}}
_COST_B = {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [
    [[9.30, 50.50], [9.35, 50.50], [9.35, 50.55], [9.30, 50.55],
     [9.30, 50.50]]]}}


def _sync(state, features, prev=None):
    from pyorps.gui.services import manual_cost

    return manual_cost.sync_manual_layer(
        state, features, project_crs="EPSG:25832", default_cost=42,
        mode="override", name="Manual costs", prev_rows=prev or [])


def test_study_area_shape_is_not_a_cost_polygon(state):
    from shapely.geometry import shape

    state.study_area = _AREA
    state.study_area_geoms.add(shape(_AREA["geometry"]).wkt)
    layer, rows = _sync(state, [_AREA, _COST_A])
    # ONLY the drawn cost polygon, never the study-area rectangle
    assert layer is not None
    assert len(rows) == 1
    assert len(layer.gdf) == 1


def test_manual_remove_selected_polygon(app, state):
    from pyorps.gui.services import manual_cost

    _layer, rows = _sync(state, [_COST_A, _COST_B])
    assert len(rows) == 2
    drawn = {"type": "FeatureCollection", "features": [_COST_A, _COST_B]}
    resp = invoke(app, ("layers-view.data", ids.MANUAL_DEL_BTN),
                  1, drawn, rows, [rows[0]], "Manual costs", 42, "override",
                  triggered=[f"{ids.MANUAL_DEL_BTN}.n_clicks"])
    new_rows = resp[ids.MANUAL_GRID]["rowData"]
    assert len(new_rows) == 1
    assert len(state.get(manual_cost.MANUAL_LAYER_ID).gdf) == 1
    # a later redraw keeps the removed polygon out (excluded by geometry)
    _layer2, rows2 = _sync(state, [_COST_A, _COST_B])
    assert len(rows2) == 1


# --------------------------------------------------------- 63 routes grid
def test_route_row_has_cost_length_endpoints(state):
    from pyorps.gui.callbacks.groups import _route_row

    layer = state.add_layer(
        "R", "route",
        meta={"metrics": {"total_cost": 1234.6, "total_length_m": 500.4},
              "control_points": [[100.0, 200.0], [300.0, 400.0]]})
    row = _route_row(layer)
    assert row["cost"] == 1235
    assert row["length"] == 500
    assert row["from"] == "100, 200"
    assert row["to"] == "300, 400"
