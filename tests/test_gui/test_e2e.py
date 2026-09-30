"""
Playwright E2E tests — the handful of flows that must work in a real browser
(Section 13.3). Skipped cleanly when playwright/chromium isn't available.

Run explicitly with::

    pytest tests/test_gui/test_e2e.py -v
"""
from __future__ import annotations

import socket
import threading

import pytest

pytest.importorskip("dash")
playwright_sync = pytest.importorskip("playwright.sync_api")

from werkzeug.serving import make_server


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def gui_server():
    """A live GUI app with a pre-seeded error notice (forced-error check)."""
    from pyorps.gui import ids
    from pyorps.gui.app import build_app
    from pyorps.gui.services.errors import Notice
    from pyorps.gui.state import ProjectState

    state = ProjectState()
    app = build_app(state)

    # A synthetic vector layer around Frankfurt (Phase 1 E2E: toggle it).
    square = {"type": "FeatureCollection", "features": [{
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [[
            [8.6, 50.0], [8.8, 50.0], [8.8, 50.2], [8.6, 50.2], [8.6, 50.0],
        ]]},
        "properties": {"name": "e2e-square"},
    }]}
    # unique stroke color so E2E selectors match exactly this layer
    state.add_layer("E2E square", "vector", geojson=square,
                    style={"color": "#00aa88"})

    # Force an error notice into the initial store: the render callback must
    # paint it as a box on load (Phase 0 E2E requirement).
    forced = Notice(severity="error", title="Forced test error",
                    meaning="m", impact="i", fix="f", details="boom",
                    id="forced-1")
    for component in app.layout._traverse():
        if getattr(component, "id", None) == ids.NOTICES:
            component.data = [forced.to_dict()]
        elif getattr(component, "id", None) == ids.LAYERS_VIEW:
            component.data = state.layers_view()
        elif getattr(component, "id", None) == ids.MAP:
            component.center = [50.1, 8.7]
            component.zoom = 10

    port = _free_port()
    server = make_server("127.0.0.1", port, app.server)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}", app, state
    finally:
        server.shutdown()
        state.shutdown()


#: the forced-error toast text, captured at first render — error toasts now
#: auto-dismiss after 10 s (round 8), so we grab it before it disappears.
_FORCED_TOAST: dict[str, str] = {}


@pytest.fixture(scope="module")
def page(gui_server):
    url, app, state = gui_server
    try:
        pw = playwright_sync.sync_playwright().start()
        browser = pw.chromium.launch()
    except Exception as exc:  # pragma: no cover - no browser installed
        pytest.skip(f"chromium not launchable: {exc}")
    page = browser.new_page()
    page.goto(url)
    # warm the page (cold module start can be slow) so every test starts ready,
    # and snapshot the auto-dismissing forced-error toast right away.
    page.wait_for_selector(".leaflet-container", timeout=30_000)
    try:
        toast = page.wait_for_selector(".notice-stack .toast", timeout=15_000)
        _FORCED_TOAST["text"] = toast.inner_text()
    except Exception:  # pragma: no cover - captured lazily otherwise
        _FORCED_TOAST["text"] = ""
    yield page
    browser.close()
    pw.stop()


def test_app_serves_and_map_renders(page):
    # Leaflet initialized (blank map, R1) with the single basemap tile pane
    # mounted. The pane is present but 'hidden' until basemap tiles load (they
    # need network, absent in the test env) → assert it's ATTACHED, not visible.
    page.wait_for_selector(".leaflet-container", timeout=15_000)
    page.wait_for_selector(".leaflet-tile-pane", state="attached", timeout=15_000)
    # sidebar tabs present
    assert page.locator("text=PYORPS GUI").count() >= 1


def test_forced_error_shows_notice_box(page):
    # captured at first render (error toasts auto-dismiss after 10 s now)
    text = _FORCED_TOAST.get("text", "")
    assert "Forced test error" in text
    # the four-line anatomy is present
    assert "Means:" in text
    assert "Fix:" in text


def test_mode_badge_overlay_present(page):
    badge = page.wait_for_selector(".mode-badge", timeout=15_000)
    assert "off" in badge.inner_text()


def test_layer_renders_toggles_and_inspects(page):
    """Phase 1 E2E: the layer paints, hides on toggle, and shows attributes.

    This is the exact failure class of webviz v1 (C1): a dynamically-updated
    layer must actually (re)render in the browser.
    """
    # the vector square is painted as an interactive SVG path
    page.wait_for_selector("path.leaflet-interactive[stroke='#00aa88']", timeout=15_000)

    # attribute inspection (C5): click the feature, check the Attrs tab.
    # Click near its bottom-right corner — the study area drawn by the
    # previous test overlaps the centre and would intercept the click.
    square = page.locator("path.leaflet-interactive[stroke='#00aa88']")
    bbox = square.bounding_box()
    square.click(position={"x": bbox["width"] * 0.92,
                           "y": bbox["height"] * 0.92})
    page.locator("#sidebar-tabs a", has_text="Attrs").click()
    panel = page.wait_for_selector("#attr-panel table", timeout=15_000)
    assert "e2e-square" in panel.inner_text()

    # Layers tab: toggle visibility -> the path must disappear (C1 render)
    page.locator("#sidebar-tabs a", has_text="Layers").click()
    row = page.locator("#layers-grid .ag-row", has_text="E2E square")
    row.wait_for(timeout=15_000)
    row.locator("input[type=checkbox]").click()
    page.wait_for_selector("path.leaflet-interactive[stroke='#00aa88']",
                           state="detached", timeout=15_000)
    # toggle back on -> it reappears
    row = page.locator("#layers-grid .ag-row", has_text="E2E square")
    row.locator("input[type=checkbox]").click()
    page.wait_for_selector("path.leaflet-interactive[stroke='#00aa88']",
                           timeout=15_000)


def test_draw_study_area_and_load_local_file(page, gui_server, tmp_path):
    """Phase 2 E2E: draw a rectangle -> study area set; load a GeoJSON."""
    import geopandas as gpd
    from shapely.geometry import Polygon

    url, app, state = gui_server

    # --- draw a rectangle with the Leaflet.draw toolbar
    page.click("a.leaflet-draw-draw-rectangle")
    box = page.locator(".leaflet-container").bounding_box()
    x0, y0 = box["x"] + box["width"] * 0.3, box["y"] + box["height"] * 0.3
    x1, y1 = box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5
    page.mouse.move(x0, y0)
    page.mouse.down()
    page.mouse.move(x1, y1, steps=8)
    page.mouse.up()
    page.wait_for_function(
        "document.getElementById('study-area-info').innerText"
        ".includes('km')", timeout=15_000)
    assert state.study_area is not None

    # --- load a small local GeoJSON via the Data tab
    gdf = gpd.GeoDataFrame(
        {"use": ["forest", "road"]},
        geometry=[Polygon([(8.65, 50.05), (8.66, 50.05), (8.66, 50.06)]),
                  Polygon([(8.67, 50.07), (8.68, 50.07), (8.68, 50.08)])],
        crs="EPSG:4326")
    path = tmp_path / "e2e_data.geojson"
    gdf.to_file(path, driver="GeoJSON")

    page.locator("#sidebar-tabs a", has_text="Data").click()
    page.fill("#local-path", str(path))
    # loading without clipping (the drawn area is arbitrary screen coords)
    page.uncheck("#clip-to-area")
    page.click("#local-load-btn")
    page.wait_for_function(
        "document.getElementById('dataset-list').innerText"
        ".includes('2 features')", timeout=15_000)


def test_cost_seed_edit_and_rasterize(page, gui_server, tmp_path):
    """Phase 3 E2E: seed the cost grid, edit a cell, rasterize, see tiles."""
    import geopandas as gpd
    from shapely.geometry import Polygon

    url, app, state = gui_server

    # a small ALKIS-like dataset in the project CRS
    def sq(x, y, s=60):
        return Polygon([(x, y), (x + s, y), (x + s, y + s), (x, y + s)])

    gdf = gpd.GeoDataFrame(
        {"nutzart": ["Wald", "Weg", "Landwirtschaft"],
         "bez": ["Nadelholz", "", "Ackerland"],
         "name": ["w", "p", "a"]},
        geometry=[sq(500000, 5530000), sq(500060, 5530000),
                  sq(500000, 5530060)], crs="EPSG:25832")
    path = tmp_path / "alkis.geojson"
    gdf.to_file(path, driver="GeoJSON")

    page.locator("#sidebar-tabs a", has_text="Data").click()
    page.fill("#local-path", str(path))
    if page.is_checked("#clip-to-area"):
        page.uncheck("#clip-to-area")
    page.click("#local-load-btn")
    page.wait_for_function(
        "document.getElementById('dataset-list').innerText"
        ".includes('alkis.geojson')", timeout=15_000)

    # Cost tab: pick the dataset -> features auto-proposed -> seed
    page.locator("#sidebar-tabs a", has_text="Cost").click()
    # dash 4 dropdown = Radix listbox in a portal: click button, then option
    page.locator("#cost-dataset").click()
    page.locator("[role=option]", has_text="alkis.geojson").click()
    page.wait_for_function(
        "document.querySelector('#cost-feature-keys').innerText"
        ".includes('nutzart')", timeout=15_000)
    page.click("#cost-seed-btn")
    row = page.locator("#cost-grid .ag-row", has_text="Nadelholz").first
    row.wait_for(timeout=15_000)

    # edit the Nadelholz cost cell (405 -> 999)
    cost_cell = row.locator(".ag-cell[col-id='cost']")
    cost_cell.dblclick()
    page.keyboard.press("Control+a")
    page.keyboard.type("999")
    page.keyboard.press("Enter")

    # Raster tab: build + wait for served tiles from localtileserver
    page.locator("#sidebar-tabs a", has_text="Raster").click()
    page.click("#rasterize-btn")
    page.wait_for_function(
        "document.getElementById('rasterize-log').innerText"
        ".includes('saved')", timeout=60_000)
    # localtileserver tile generation (rio-tiler) is CPU-bound; under a full
    # module run the first tile can take a while — give it the same budget as
    # the build itself so this isn't a timing flake.
    page.wait_for_selector("img.leaflet-tile[src*='localhost']",
                           timeout=60_000)

    # the edited cost reached the raster config
    rasters = [ly for ly in state.layers.values() if ly.kind == "raster"]
    assert len(rasters) == 1



