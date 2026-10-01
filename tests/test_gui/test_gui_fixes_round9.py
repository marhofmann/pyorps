"""
Round-9 GUI changes (2026-07-20):

- colormap names are valid rio-tiler keys (the "Spectral_r" recolor crash) and
  the matplotlib legend path resolves them case-insensitively
- the draw target follows the active tab (Cost tab → cost polygons, otherwise
  study area); the explicit draw-target radio is gone
- the Add-vector-data card shows only the selected source's parameters
  (local | wfs) and the load buttons carry a spinner status target
- the basemap picker moved onto the map (corner control) with an extended,
  provider-grouped catalog
- the section cards named in the task are collapsible <details>, closed by
  default; 'Run routing' moved to the always-visible 'New routing' card
"""
import pytest

from pyorps.gui import ids

from conftest import invoke


# =============================================================== colormaps
def test_colormaps_are_valid_rio_tiler_keys():
    """Every offered colormap must pass localtileserver's exact-match
    validation (rio-tiler keys are lowercase — 'Spectral_r' used to raise)."""
    palettes = pytest.importorskip("localtileserver.tiler.palettes")
    from pyorps.gui.services.tiles import COLORMAPS

    for name in COLORMAPS:
        palettes.palette_valid_or_raise(name)   # raises ValueError if invalid


def test_legend_cmap_resolves_case_insensitively():
    """The legend (matplotlib) must find the real colormap for the lowercase
    rio-tiler keys — not silently fall back to viridis."""
    from pyorps.gui.services.tiles import _mpl_cmap, cost_colors

    assert _mpl_cmap("spectral_r").name == "Spectral_r"
    assert _mpl_cmap("rdylgn_r").name == "RdYlGn_r"
    assert _mpl_cmap("no-such-cmap").name == "viridis"
    spectral = cost_colors("spectral_r", [10, 90], 0, 100)
    viridis = cost_colors("viridis", [10, 90], 0, 100)
    assert spectral[10] != viridis[10]


def test_set_colormap_service_normalizes_case(state, raster_path):
    """A stale mixed-case name (old project) still re-colours the tiles."""
    from pyorps.gui.services import tiles

    layer = tiles.build_tile_layer(raster_path, name="t",
                                   work_dir=state.work_dir)
    try:
        url = tiles.set_colormap(layer, "Spectral_r")
        assert "spectral_r" in url
        assert layer.colormap == "spectral_r"
    finally:
        shutdown = getattr(layer.tile_client, "shutdown", None)
        if callable(shutdown):
            shutdown()


# ===================================================== draw target by tab
def test_draw_target_follows_active_tab(app):
    resp = invoke(app, ("draw-target.data", ids.TABS), ids.TAB_COST,
                  triggered=[f"{ids.TABS}.active_tab"])
    assert resp[ids.DRAW_TARGET]["data"] == "cost"
    resp = invoke(app, ("draw-target.data", ids.TABS), ids.TAB_DATA,
                  triggered=[f"{ids.TABS}.active_tab"])
    assert resp[ids.DRAW_TARGET]["data"] == "area"


def test_draw_target_radio_removed_from_layout(state):
    from pyorps.gui.layout import build_layout

    found = {getattr(c, "id", None) for c in build_layout(state)._traverse()}
    assert ids.DRAW_TARGET_RADIO not in found


# ======================================================= data source panels
def test_source_panels_toggle(app):
    resp = invoke(app, ("data-local-panel.style", ids.DATA_SOURCE_TYPE),
                  "wfs", triggered=[f"{ids.DATA_SOURCE_TYPE}.value"])
    assert resp[ids.DATA_LOCAL_PANEL]["style"] == {"display": "none"}
    assert resp[ids.DATA_WFS_PANEL]["style"] == {}

    resp = invoke(app, ("data-local-panel.style", ids.DATA_SOURCE_TYPE),
                  "local", triggered=[f"{ids.DATA_SOURCE_TYPE}.value"])
    assert resp[ids.DATA_LOCAL_PANEL]["style"] == {}
    assert resp[ids.DATA_WFS_PANEL]["style"] == {"display": "none"}


def test_load_buttons_have_spinner_targets(app, state):
    """Both load callbacks write a status span that sits inside a
    dcc.Loading, so a spinner shows until the load finishes."""
    from pyorps.gui.layout import build_layout

    keys = "\n".join(app.callback_map)
    assert ids.LOCAL_LOAD_STATUS in keys
    assert ids.WFS_LOAD_STATUS in keys
    found = {getattr(c, "id", None) for c in build_layout(state)._traverse()}
    assert ids.LOCAL_LOAD_STATUS in found
    assert ids.WFS_LOAD_STATUS in found


# ============================================================ basemap picker
def test_basemap_catalog_extended_and_consistent():
    from pyorps.gui.layout import BASEMAP_BY_NAME, BASEMAPS, DEFAULT_BASEMAP

    names = [bm["name"] for bm in BASEMAPS]
    assert len(names) == len(set(names))          # no duplicates
    assert DEFAULT_BASEMAP in BASEMAP_BY_NAME
    assert len(BASEMAPS) >= 15                    # extended catalog
    for bm in BASEMAPS:
        assert bm["url"].startswith("http")
        assert bm["attribution"]
        assert bm["group"]


def test_basemap_select_lives_on_the_map(state):
    """One BASEMAP_SELECT, inside the map pane's corner control (the Layers
    tab card was removed) — the update_basemap callback wiring is untouched."""
    from pyorps.gui.layout import _basemap_control, build_layout

    all_ids = [getattr(c, "id", None)
               for c in build_layout(state)._traverse()]
    assert all_ids.count(ids.BASEMAP_SELECT) == 1
    control_ids = {getattr(c, "id", None)
                   for c in _basemap_control()._traverse()}
    assert {ids.BASEMAP_SELECT, ids.BASEMAP_OPACITY,
            ids.BASEMAP_ZORDER} <= control_ids


# ===================================================== collapsible sections
def test_section_cards_collapsible_closed_by_default(state):
    from dash import html

    from pyorps.gui.layout import build_layout

    details = [c for c in build_layout(state)._traverse()
               if isinstance(c, html.Details)
               and "gui-card-collapsible" in (getattr(c, "className", "")
                                              or "")]
    # Data 3 + Cost 3 + Raster 2 + Routes 4
    assert len(details) >= 12
    assert all(not getattr(d, "open", False) for d in details)


def test_run_routing_button_outside_collapsibles(state):
    """'Run routing' stays reachable: it must NOT sit inside any collapsible
    card (it moved to 'New routing' when 'Constrained routing' collapsed)."""
    from dash import html

    from pyorps.gui.layout import build_layout

    def contains(component, wanted_id):
        return any(getattr(c, "id", None) == wanted_id
                   for c in component._traverse())

    layout = build_layout(state)
    assert contains(layout, ids.RUN_ROUTING_BTN)
    for det in layout._traverse():
        if isinstance(det, html.Details) and "gui-card-collapsible" in (
                getattr(det, "className", "") or ""):
            assert not contains(det, ids.RUN_ROUTING_BTN)
