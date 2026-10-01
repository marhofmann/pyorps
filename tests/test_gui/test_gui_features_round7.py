"""
Round-7 GUI features (2026-07-16):

47  persistent warning/error log + in-app viewer
48  auto-dismiss durations (warning 5 s, error 10 s)
49  the drawn study-area shape is invisible (no duplicate blue fill)
50  a WFS with no data in the area is a warning, not an error
52  background-map source / opacity / z-order editable in the Layers tab
53  deterministic page-jump for the attribute / combination table
54  the attribute table never hides the sidebar + a resize slider
55  move-waypoint edit + unified build/edit mode driven by the edit knob
56  hierarchical route tree (routes → groups → routes) in the Layers tab
57  step gating + workflow guide
"""
import json

import pytest

from pyorps.gui import ids

from conftest import invoke


# =============================================================== 47/48 notices
def test_auto_dismiss_durations():
    from pyorps.gui.callbacks.notifications import _DURATIONS, notice_toast

    assert _DURATIONS["warning"] == 5_000
    assert _DURATIONS["error"] == 10_000
    assert notice_toast({"severity": "warning", "title": "x"}).duration == 5_000
    assert notice_toast({"severity": "error", "title": "x"}).duration == 10_000


def test_logbook_writes_and_reads():
    from pyorps.gui.services import logbook

    logbook.log_notice({"severity": "warning", "title": "MARK-LOGBOOK-7A",
                        "meaning": "why", "details": "a\ntrace"})
    text = logbook.read_log()
    assert "MARK-LOGBOOK-7A" in text
    assert str(logbook.log_file_path()).endswith("pyorps_gui.log")


def test_render_notices_persists_to_log(app):
    from pyorps.gui.services import logbook

    invoke(app, ("notice-stack.children",),
           [{"id": "n7id", "severity": "error", "title": "MARK-RENDER-7B"}])
    assert "MARK-RENDER-7B" in logbook.read_log()


def test_log_viewer_callback_opens(app):
    resp = invoke(app, ("log-offcanvas.is_open", ids.LOG_VIEW_BTN),
                  1, None, False,
                  triggered=[f"{ids.LOG_VIEW_BTN}.n_clicks"])
    assert resp[ids.LOG_OFFCANVAS]["is_open"] is True
    assert "Saved to" in resp[ids.LOG_PATH_INFO]["children"]


# ============================================================= 49 study area
def test_draw_rubber_band_visible_created_shape_hidden():
    """The in-progress rubber band is VISIBLE (live edge preview) while the
    CREATED layer is restyled to invisible by assets/draw-preview.js — so the
    styled host layer never doubles (round 7), but drawing previews live."""
    from pathlib import Path

    from pyorps.gui.layout import DRAW_SHAPE_OPTIONS, build_map

    assert DRAW_SHAPE_OPTIONS["weight"] > 0
    assert DRAW_SHAPE_OPTIONS["opacity"] > 0
    control = next(c for c in build_map()._traverse()
                   if getattr(c, "id", None) == ids.DRAW_CONTROL)
    assert control.draw["rectangle"]["shapeOptions"]["opacity"] > 0
    assert control.draw["polygon"]["shapeOptions"]["opacity"] > 0

    # the created-layer hider must exist and patch the leaflet-draw funnel
    js = (Path(build_map.__code__.co_filename).parent
          / "assets" / "draw-preview.js").read_text(encoding="utf-8")
    assert "_fireCreatedEvent" in js
    assert "setStyle" in js


# ============================================================= 50 WFS warning
@pytest.mark.parametrize("message", [
    "The WFS returned no data for layer 'X' (bbox …).",
    "The WFS returned no features for layer 'Y' (bbox …).",
])
def test_wfs_empty_area_is_warning(message):
    from pyorps.core.exceptions import WFSResponseParsingError
    from pyorps.gui.services import errors

    notice = errors.translate_exception(WFSResponseParsingError(message))
    assert notice.severity == "warning"
    assert notice.focus_id == ids.TAB_DATA


def test_wfs_malformed_still_error():
    from pyorps.core.exceptions import WFSResponseParsingError
    from pyorps.gui.services import errors

    notice = errors.translate_exception(
        WFSResponseParsingError("invalid XML gibberish"))
    assert notice.severity == "error"


# ============================================================= 52 basemap
def test_update_basemap_source_opacity_zorder(app):
    from pyorps.gui.layout import (BASEMAP_BY_NAME, BASEMAP_Z_ABOVE,
                                   BASEMAP_Z_BELOW, BLANK_TILE)

    resp = invoke(app, ("basemap-tile.url", ids.BASEMAP_SELECT),
                  "OpenStreetMap", 1.0, "below",
                  triggered=[f"{ids.BASEMAP_SELECT}.value"])
    tile = resp[ids.BASEMAP_TILE]
    assert tile["url"] == BASEMAP_BY_NAME["OpenStreetMap"]["url"]
    assert tile["zIndex"] == BASEMAP_Z_BELOW

    resp = invoke(app, ("basemap-tile.url", ids.BASEMAP_SELECT),
                  "Esri World Imagery", 0.4, "above",
                  triggered=[f"{ids.BASEMAP_ZORDER}.value"])
    tile = resp[ids.BASEMAP_TILE]
    assert tile["zIndex"] == BASEMAP_Z_ABOVE
    assert tile["opacity"] == 0.4

    resp = invoke(app, ("basemap-tile.url", ids.BASEMAP_SELECT),
                  "", 1.0, "below", triggered=[f"{ids.BASEMAP_SELECT}.value"])
    assert resp[ids.BASEMAP_TILE]["url"] == BLANK_TILE


def test_raster_tile_carries_zindex(state, raster_path):
    from pyorps.gui.callbacks.layers import render_layer
    from pyorps.gui.callbacks.raster import add_raster_layer
    from pyorps.gui.layout import RASTER_TILE_ZINDEX

    layer, _ = add_raster_layer(state, raster_path, [])
    component = render_layer(layer)
    assert component.zIndex == RASTER_TILE_ZINDEX


# ============================================================= 53/54 table
def test_table_page_computation():
    from pyorps.gui.callbacks.layers import _table_page
    from pyorps.gui.layout import TABLE_PAGE_SIZE

    assert _table_page(0) == 0
    assert _table_page(TABLE_PAGE_SIZE - 1) == 0
    assert _table_page(TABLE_PAGE_SIZE) == 1
    assert _table_page(TABLE_PAGE_SIZE * 2 + 5) == 2


def test_table_resize_is_drag_based(state):
    """Round 9: the percent slider is gone — the panel height is dragged
    freely (px) via the .table-resizer strip + assets/table-resize.js; the
    sidebar stays free through the CSS width calc(100% - var(--sidebar-w))."""
    from pathlib import Path

    from pyorps.gui.layout import build_layout

    layout = build_layout(state)
    found = {getattr(c, "id", None) for c in layout._traverse()}
    assert ids.LAYER_TABLE_HEIGHT not in found      # slider removed
    strips = [c for c in layout._traverse()
              if "table-resizer" in (getattr(c, "className", "") or "")]
    assert len(strips) == 1
    js = (Path(build_layout.__code__.co_filename).parent
          / "assets" / "table-resize.js").read_text(encoding="utf-8")
    assert "table-resizer" in js and "localStorage" in js


# ============================================================= 56 route tree
def _add_route(state, name, group):
    return state.add_layer(name, "route",
                           meta={"group": group,
                                 "control_points": [[0, 0], [1, 1]]})


def test_route_tree_component_hierarchy(state):
    from pyorps.gui.callbacks.layers import route_tree_component

    assert "No routes" in str(route_tree_component(state))

    _add_route(state, "R1", "Run 1")
    _add_route(state, "R2", "Run 1")
    _add_route(state, "R3", None)
    rendered = str(route_tree_component(state))
    assert "Routes (3)" in rendered
    assert "Run 1 (2)" in rendered
    assert "(ungrouped) (1)" in rendered
    for name in ("R1", "R2", "R3"):
        assert name in rendered


def test_route_tree_click_selects(app, state):
    route = _add_route(state, "Only", "Run 1")
    resp = invoke(
        app, ("edit-route-select.value", ids.TYPE_ROUTE_TREE_ITEM), [1],
        triggered=[{"id": {"type": ids.TYPE_ROUTE_TREE_ITEM, "id": route.id},
                    "property": "n_clicks", "value": 1}])
    assert resp[ids.EDIT_ROUTE_SELECT]["value"] == route.id


# ============================================================= 57 gating/guide
def test_step_gating_and_guide(app, state, raster_path):
    from pyorps.gui.callbacks.raster import add_raster_layer

    # empty project → every dependent step is greyed out
    resp = invoke(app, ("workflow-guide.children",), [], {}, None, None, None,
                  triggered=[f"{ids.LAYERS_VIEW}.data"])
    assert resp[ids.COST_SEED_BTN]["disabled"] is True
    assert resp[ids.RASTERIZE_BTN]["disabled"] is True
    assert resp[ids.RUN_ROUTING_BTN]["disabled"] is True
    assert resp[ids.RASTER_COMBINE_BTN]["disabled"] is True

    add_raster_layer(state, raster_path, [])
    grid_state = {"dataset_id": "d1", "feature_keys": ["nutzart"]}
    resp = invoke(app, ("workflow-guide.children",), state.layers_view(),
                  grid_state, "d1", ["nutzart"], None,
                  triggered=[f"{ids.COST_GRID_STATE}.data"])
    assert resp[ids.COST_SEED_BTN]["disabled"] is False      # dataset + keys
    assert resp[ids.RASTERIZE_BTN]["disabled"] is False      # cost seeded
    assert resp[ids.RUN_ROUTING_BTN]["disabled"] is False    # a raster exists
    assert resp[ids.RASTER_COMBINE_BTN]["disabled"] is True  # needs 2 rasters
    # the guide (serialized component tree) is present
    assert "Workflow" in json.dumps(resp[ids.WORKFLOW_GUIDE])
