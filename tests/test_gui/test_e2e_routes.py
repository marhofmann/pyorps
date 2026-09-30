"""
Playwright E2E: route building + editing in a real browser (Phases 4/5).

Runs against its own app instance with a pre-served cost raster and the map
centered on it, so the click coordinates are deterministic.
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
def route_server(tmp_path_factory):
    from pyorps.gui import ids
    from pyorps.gui.app import build_app
    from pyorps.gui.callbacks.raster import add_raster_layer
    from pyorps.gui.state import ProjectState
    from pyorps.raster.handler import create_test_tiff

    tmp = tmp_path_factory.mktemp("e2e_routes")
    raster = tmp / "cost.tif"
    create_test_tiff(str(raster), width=200, height=200, pattern="gradient",
                     crs="EPSG:32632")

    state = ProjectState()
    app = build_app(state)
    layer, _ = add_raster_layer(state, str(raster), [])

    for component in app.layout._traverse():
        cid = getattr(component, "id", None)
        if cid == ids.LAYERS_VIEW:
            component.data = state.layers_view()
        elif cid == ids.MAP:
            (s, w), (n, e) = layer.tile.bounds
            component.center = [(s + n) / 2, (w + e) / 2]
            # zoom 16 ≈ 1.5 m/px at this latitude: the 200 m raster spans
            # ~130 px, so ±40 px clicks stay well inside it
            component.zoom = 16

    port = _free_port()
    server = make_server("127.0.0.1", port, app.server)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{port}", app, state
    finally:
        server.shutdown()
        state.shutdown()


@pytest.fixture(scope="module")
def page(route_server):
    url, app, state = route_server
    try:
        pw = playwright_sync.sync_playwright().start()
        browser = pw.chromium.launch()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"chromium not launchable: {exc}")
    page = browser.new_page()
    page.goto(url)
    page.wait_for_selector(".leaflet-container", timeout=15_000)
    yield page
    browser.close()
    pw.stop()


def _open_sections(page):
    """Expand the collapsible sidebar cards (closed by default) so controls
    like the route selector in 'Computed routes' are visible/clickable."""
    page.eval_on_selector_all(
        "details.gui-card-collapsible",
        "els => els.forEach(e => { e.open = true; })")


def test_build_route_by_map_clicks(page, route_server):
    """Phase 4 E2E: radio mode -> map clicks -> run -> polyline in DOM."""
    url, app, state = route_server

    page.locator("#sidebar-tabs a", has_text="Routes").click()
    page.locator("#route-raster").click()
    page.locator("[role=option]").last.click()

    box = page.locator(".leaflet-container").bounding_box()
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2

    page.locator("#build-mode label", has_text="Source").click()
    page.wait_for_function(
        "document.getElementById('mode-badge').innerText"
        ".includes('SOURCE')", timeout=15_000)
    page.mouse.click(cx - 40, cy - 30)
    page.locator("#build-points-table .ag-row").first.wait_for(
        timeout=15_000)

    page.locator("#build-mode label", has_text="Target").click()
    page.wait_for_function(
        "document.getElementById('mode-badge').innerText"
        ".includes('TARGET')", timeout=15_000)
    page.mouse.click(cx + 40, cy + 30)
    page.wait_for_function(
        "document.querySelectorAll('#build-points-table .ag-row')"
        ".length >= 2", timeout=15_000)

    page.click("#run-routing-btn")
    page.wait_for_function(
        "document.getElementById('routing-status').innerText"
        ".includes('built')", timeout=60_000)
    # the routed line is painted with the route style (C1 render)
    page.wait_for_selector("path.leaflet-interactive[stroke='#e6194b']",
                           timeout=15_000)
    routes = [ly for ly in state.layers.values() if ly.kind == "route"]
    assert len(routes) == 1
    assert routes[0].meta["metrics"]["total_cost"] > 0


def test_edit_route_click_to_place_replaces(page, route_server):
    """Phase 5 E2E: click-to-place move recomputes the route and REPLACES it
    (task 44) — the edited route is the only one left.
    """
    url, app, state = route_server

    page.locator("#sidebar-tabs a", has_text="Routes").click()
    _open_sections(page)
    page.locator("#edit-route-select").click()
    page.locator("[role=option]").first.click()
    # active route drawn dashed-capable polyline + control markers appear
    page.wait_for_selector("path.leaflet-interactive[stroke='#ffe119']",
                           timeout=15_000)

    # turn on the edit knob, then the unified mode radio edits the route (task 55)
    page.locator("#edit-enable").check()
    page.locator("#build-mode label", has_text="Target").click()
    page.wait_for_function(
        "document.getElementById('mode-badge').innerText"
        ".includes('MOVES TARGET')", timeout=15_000)

    box = page.locator(".leaflet-container").bounding_box()
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    page.mouse.click(cx + 20, cy + 45)

    # the edited route replaces the old one (task 44): the grid shows the
    # 'moved target' route and it's the only route left
    page.wait_for_function(
        "document.getElementById('routes-grid').innerText"
        ".includes('moved target')", timeout=60_000)
    routes = [ly for ly in state.layers.values() if ly.kind == "route"]
    assert len(routes) == 1
    assert routes[0].meta["edit"] == "moved target"
