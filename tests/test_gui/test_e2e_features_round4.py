"""
Playwright E2E for the round-4 features (2026-07-15) — the browser interactions
that must actually work, not just the headless callbacks:

- new Cost/Raster controls render without a client-side JS error (the
  webviz-v1 failure class, and the dash-ag-grid getRowId freeze the project
  memory warns about);
- add a custom preprocessing step (F4);
- click a feature of the selected layer -> its row is selected in the layer
  table offcanvas (F1);
- open a raster's cost-combination table (F2);
- change the raster colormap and see tiles reload (F3);
- snapshot a drawn rectangle into a custom cost layer (F5);
- combine two rasters into a third (F6).

Skipped cleanly when playwright/chromium isn't available.
"""
from __future__ import annotations

import socket
import threading

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

pytest.importorskip("dash")
playwright_sync = pytest.importorskip("playwright.sync_api")

from werkzeug.serving import make_server

# a projected area near Frankfurt (EPSG:25832) so the map can sit over it
OX, OY = 475000.0, 5550000.0


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_tif(path, arr, res=20.0):
    transform = from_origin(OX, OY, res, res)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0],
                       width=arr.shape[1], count=1, dtype="uint16",
                       crs="EPSG:25832", transform=transform,
                       nodata=65535) as dst:
        dst.write(arr, 1)
    return str(path)


@pytest.fixture(scope="module")
def feat_server(tmp_path_factory):
    """Live GUI pre-seeded with a vector layer + two served rasters (one with a
    combination table), map centred on the data."""
    import geopandas as gpd
    from shapely.geometry import box

    from pyorps.gui import ids
    from pyorps.gui.app import build_app
    from pyorps.gui.callbacks.raster import add_raster_layer
    from pyorps.gui.services import combinations, geo
    from pyorps.gui.state import ProjectState

    tmp = tmp_path_factory.mktemp("e2e4")
    state = ProjectState()

    # vector land-use layer (unique stroke colour to select on the map)
    gdf = gpd.GeoDataFrame(
        {"nutzart": ["Wald", "Weg"], "bez": ["Nadelholz", ""]},
        geometry=[box(OX + 100, OY - 400, OX + 300, OY - 200),
                  box(OX + 400, OY - 400, OX + 600, OY - 200)],
        crs="EPSG:25832")
    vlayer = state.add_layer("Landuse", "vector", gdf=gdf, crs=gdf.crs,
                             geojson=geo.gdf_to_wgs84_geojson(gdf),
                             style={"color": "#aa00aa"})

    # two served rasters
    a = _make_tif(tmp / "a.tif",
                  np.array([[405, 405], [300, 300]], dtype="uint16"))
    b = _make_tif(tmp / "b.tif",
                  np.array([[2, 2], [3, 3]], dtype="uint16"))
    rlayer, _ = add_raster_layer(state, a, [], name="Cost raster A")
    add_raster_layer(state, b, [], name="Cost raster B")
    # attach a combination table so the 📋 table has content (F2)
    rlayer.meta["combination_gdf"] = combinations.build_combination_table(
        gdf, ("nutzart", "bez"),
        {"Wald": {"Nadelholz": 405, "": 405}, "Weg": {"": 300}},
        base_layer_name="Landuse", base_crs=gdf.crs)

    app = build_app(state)
    centre = [(vlayer.geojson["features"][0]["geometry"]["coordinates"][0][0][1]
               + 0.002), rlayer.tile.bounds[0][1] + 0.002]
    for component in app.layout._traverse():
        if getattr(component, "id", None) == ids.LAYERS_VIEW:
            component.data = state.layers_view()
        elif getattr(component, "id", None) == ids.MAP:
            component.center = [rlayer.tile.bounds[0][0] + 0.001,
                                rlayer.tile.bounds[0][1] + 0.001]
            component.zoom = 15

    port = _free_port()
    server = make_server("127.0.0.1", port, app.server)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}", app, state, vlayer, rlayer
    finally:
        server.shutdown()
        state.shutdown()


@pytest.fixture(scope="module")
def page(feat_server):
    url = feat_server[0]
    try:
        pw = playwright_sync.sync_playwright().start()
        browser = pw.chromium.launch()
    except Exception as exc:  # pragma: no cover - no browser installed
        pytest.skip(f"chromium not launchable: {exc}")
    page = browser.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(url)
    page.wait_for_selector(".leaflet-container", timeout=15_000)
    yield page, errors
    browser.close()
    pw.stop()


def _select_tab(page, label):
    page.locator("#sidebar-tabs a", has_text=label).click()


def _open_sections(page):
    """Expand every collapsible sidebar card (closed by default) so the
    controls inside are visible/clickable — what a user does via the caret."""
    page.eval_on_selector_all(
        "details.gui-card-collapsible",
        "els => els.forEach(e => { e.open = true; })")


def test_new_controls_render_without_js_error(page):
    """The new Cost/Raster controls mount cleanly (webviz-v1 failure class)."""
    pg, errors = page
    _select_tab(pg, "Cost")
    _open_sections(pg)
    pg.wait_for_selector("#preproc-grid", timeout=15_000)
    pg.wait_for_selector("#manual-cost-create-btn", timeout=15_000)
    _select_tab(pg, "Raster")
    _open_sections(pg)
    pg.wait_for_selector("#raster-colormap", timeout=15_000)
    pg.wait_for_selector("#raster-combine-btn", timeout=15_000)
    assert errors == [], f"client JS errors: {errors}"


def test_germany_wfs_controls_and_viewport_filter(page):
    """Germany-wide dataset controls render and the preset dropdown lists the
    servers covering the current view (map is centred on Hessen)."""
    pg, _ = page
    _select_tab(pg, "Data")
    _open_sections(pg)
    # the WFS parameters only show once the WFS source is selected
    pg.locator("#data-source-type label", has_text="WFS server").click()
    for sel in ("#wfs-category", "#wfs-in-view-only", "#wfs-caps-btn",
                "#wfs-layer-select", "#overlay-preset", "#dem-load-btn"):
        pg.wait_for_selector(sel, timeout=15_000)
    # the WFS preset dropdown lists German ALKIS servers for this view
    pg.locator("#wfs-preset").click()
    pg.wait_for_selector("[role=option]:has-text('Hessen')", timeout=15_000)
    pg.keyboard.press("Escape")


def test_osm_features_control_wired(page):
    """OSM Overpass controls render and the load button fires its callback.

    Runs before any study area is drawn, so clicking Load OSM yields the
    deterministic 'draw a study area' guard (no network needed); the parsing /
    happy path is covered by the headless tests."""
    pg, _ = page
    _select_tab(pg, "Data")
    _open_sections(pg)
    # round 11: the categorized key/value menu is the primary UI …
    pg.wait_for_selector("#osm-key", timeout=15_000)
    pg.wait_for_selector("#osm-add-btn", timeout=15_000)
    pg.wait_for_selector("#osm-load-btn", timeout=15_000)
    # … the preset/tag shortcuts moved into a collapsed 'advanced' accordion
    pg.click(".accordion-button:has-text('Presets & raw tags')")
    pg.wait_for_selector("#osm-tags", timeout=15_000)
    # a tag filter passes the "pick a feature" guard; with no study area yet the
    # callback then asks for one — proving the OSM button is wired end to end
    pg.fill("#osm-tags", "landuse")
    pg.click("#osm-load-btn")
    pg.wait_for_selector(
        ".notice-stack .toast:has-text('Draw a study area first')",
        timeout=15_000)


def test_add_preprocessing_step(page):
    """F4: '+ Step' adds an editable row to the preprocessing grid."""
    pg, _ = page
    _select_tab(pg, "Cost")
    _open_sections(pg)
    pg.click("#preproc-add-btn")
    row = pg.locator("#preproc-grid .ag-row")
    row.first.wait_for(timeout=15_000)
    assert row.count() >= 1
    assert "buffer" in pg.locator("#preproc-grid").inner_text()


def test_click_feature_selects_table_row(page, feat_server):
    """F1: select the layer, click its feature -> the layer table opens with
    that feature's row selected. Runs before any drawing so nothing overlaps
    the click target."""
    pg, _ = page
    _select_tab(pg, "Layers")
    lrow = pg.locator("#layers-grid .ag-row", has_text="Landuse")
    lrow.wait_for(timeout=15_000)
    lrow.click()
    path = pg.locator("path.leaflet-interactive[stroke='#aa00aa']").first
    path.wait_for(timeout=15_000)
    path.click()
    pg.wait_for_selector("#layer-table-offcanvas.show", timeout=15_000)
    pg.wait_for_selector("#layer-table-grid .ag-row-selected", timeout=15_000)


def test_open_raster_combination_table(page):
    """F2: the 📋 table of a built raster shows its cost/combination columns."""
    pg, _ = page
    # F1 left the offcanvas open; close it so it can't cover the layers grid
    if pg.locator("#layer-table-offcanvas.show").count():
        pg.locator("#layer-table-offcanvas .btn-close").click()
        pg.wait_for_selector("#layer-table-offcanvas.show", state="detached",
                             timeout=10_000)
    _select_tab(pg, "Layers")
    rrow = pg.locator("#layers-grid .ag-row", has_text="Cost raster A")
    rrow.wait_for(timeout=15_000)
    rrow.click()
    # make sure the raster row is the selected one before opening its table
    # (otherwise the offcanvas would keep showing the previous layer's table)
    pg.wait_for_selector(
        "#layers-grid .ag-row-selected:has-text('Cost raster A')",
        timeout=10_000)
    pg.click("#layer-view-btn")
    pg.wait_for_selector("#layer-table-offcanvas.show", timeout=15_000)
    # the combination table is shown once a 'modifiers' column header appears
    pg.wait_for_selector(
        "#layer-table-grid .ag-header-cell[col-id='modifiers']", timeout=15_000)
    header = pg.locator("#layer-table-grid .ag-header").inner_text()
    assert "cost" in header and "modifiers" in header
    pg.wait_for_selector("#layer-table-grid .ag-row", timeout=15_000)


def test_change_colormap_reloads_tiles(page):
    """F3: switching the colormap re-serves the raster tiles."""
    pg, _ = page
    _select_tab(pg, "Raster")
    _open_sections(pg)
    # a served raster tile is on the map
    pg.wait_for_selector("img.leaflet-tile[src*='localhost']", timeout=30_000)
    pg.select_option("#raster-colormap", "turbo")
    pg.wait_for_selector("img.leaflet-tile[src*='turbo']", timeout=30_000)


def test_create_manual_cost_layer(page, feat_server):
    """F5: draw a rectangle, snapshot it into a custom cost vector layer.

    The draw target follows the active tab (the old radio is gone): shapes
    drawn while the COST tab is open become cost polygons; the study area
    is drawn from any other tab. ``sync_manual_layer`` still excludes
    study-area geometry (task 62).
    """
    pg, _ = page
    state = feat_server[2]
    before = len(state.layers_of_kind("vector"))
    # opening the Cost tab arms the cost-polygon draw target (server-side
    # store write — give the round trip a beat before drawing)
    _select_tab(pg, "Cost")
    _open_sections(pg)
    pg.wait_for_timeout(500)
    pg.click("a.leaflet-draw-draw-rectangle")
    box = pg.locator(".leaflet-container").bounding_box()
    x0, y0 = box["x"] + box["width"] * 0.35, box["y"] + box["height"] * 0.35
    x1, y1 = box["x"] + box["width"] * 0.55, box["y"] + box["height"] * 0.55
    pg.mouse.move(x0, y0)
    pg.mouse.down()
    pg.mouse.move(x1, y1, steps=8)
    pg.mouse.up()
    # create/refresh the manual cost layer from the drawn shapes
    pg.fill("#manual-cost-name", "Drawn costs")
    pg.fill("#manual-cost-value", "1234")
    pg.click("#manual-cost-create-btn")
    pg.wait_for_function(
        "document.getElementById('manual-cost-status').innerText"
        ".includes('Created')", timeout=15_000)
    assert len(state.layers_of_kind("vector")) == before + 1
    # hand the draw tool back to the study area for any later test by
    # leaving the Cost tab (the target follows the active tab)
    _select_tab(pg, "Data")


def test_combine_rasters_controls_wired(page):
    """F6: the combine controls render and the button fires its callback.

    The actual add/multiply/… combination math + new-layer creation is covered
    exhaustively by the headless tests; here we only prove the browser wiring —
    driving the react-select multi picker deterministically across browsers is
    unreliable, so we exercise the callback through its guard path (clicking
    Combine with nothing selected yields the 'select at least two' notice).
    """
    pg, _ = page
    # the bottom offcanvas from the F1/F2 tests overlaps the combine controls
    if pg.locator("#layer-table-offcanvas.show").count():
        pg.locator("#layer-table-offcanvas .btn-close").click()
        pg.wait_for_selector("#layer-table-offcanvas.show", state="detached",
                             timeout=10_000)
    _select_tab(pg, "Raster")
    _open_sections(pg)
    pg.wait_for_selector("#raster-combine-select", timeout=15_000)
    pg.select_option("#raster-combine-op", "multiply")
    pg.click("#raster-combine-btn")
    # the notice stack accumulates toasts across tests — wait for THIS one
    pg.wait_for_selector(
        ".notice-stack .toast:has-text('Select at least two rasters')",
        timeout=15_000)
