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


def test_undo_last_edit_brings_back_the_route_before_the_edit(page, route_server):
    """Round 13 E2E: edit the route by a map click, press "Undo last edit": the route's points, length and
    readout are the original again and there is still exactly one route."""
    import time

    url, app, state = route_server

    def routes():
        return [ly for ly in state.layers.values() if ly.kind == "route"]

    def wait_for(condition, what):
        deadline = time.monotonic() + 60
        while not condition():
            assert time.monotonic() < deadline, what
            time.sleep(0.2)

    page.locator("#sidebar-tabs a", has_text="Routes").click()
    _open_sections(page)
    page.locator("#edit-route-select").click()
    page.locator("[role=option]").first.click()
    page.wait_for_selector("path.leaflet-interactive[stroke='#ffe119']", timeout=15_000)
    assert page.locator("#edit-undo-btn").is_visible()

    original = routes()[0]
    before = {"id": original.id, "points": [list(p) for p in original.meta["control_points"]],
              "length": original.meta["metrics"]["total_length_m"]}

    if not page.locator("#edit-enable").is_checked():
        page.locator("#edit-enable").check()
    page.locator("#build-mode label", has_text="Target").click()
    page.wait_for_function(
        "document.getElementById('mode-badge').innerText.includes('MOVES TARGET')", timeout=15_000)
    box = page.locator(".leaflet-container").bounding_box()
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    page.mouse.click(cx - 25, cy + 50)

    wait_for(lambda: len(routes()) == 1 and routes()[0].id != before["id"], "the edit did not replace the route")
    assert routes()[0].meta["control_points"] != before["points"]
    edited_id = routes()[0].id
    page.wait_for_function("!document.getElementById('edit-undo-btn').disabled", timeout=15_000)

    page.click("#edit-undo-btn")

    wait_for(lambda: len(routes()) == 1 and routes()[0].id == before["id"], "the undo did not restore the route")
    restored = routes()[0]
    assert restored.meta["control_points"] == before["points"]
    assert restored.meta["metrics"]["total_length_m"] == before["length"]
    assert state.active_route_id == restored.id
    # the page shows the original again: its length in the readout, undo off, one route in the grid
    page.wait_for_function(
        "(len) => document.getElementById('edit-cost-readout').innerText.includes(len)",
        arg=f"{before['length']:,.0f} m", timeout=15_000)
    page.wait_for_function("document.getElementById('edit-undo-btn').disabled", timeout=15_000)
    page.wait_for_function(
        "(id) => document.querySelector(`#routes-grid .ag-row[row-id='${id}']`) !== null",
        arg=restored.id, timeout=15_000)
    page.wait_for_function(     # (the grid animates a removed row away)
        "(id) => document.querySelector(`#routes-grid .ag-row[row-id='${id}']`) === null",
        arg=edited_id, timeout=15_000)


def test_another_backend_and_algorithm_build_a_route_that_names_them(page, route_server):
    """Round 13 E2E: pick networkx (installed here) in the Backend dropdown and an algorithm it offers, build a
    route by map clicks and check that the stored route says which backend and algorithm ran."""
    import time

    pytest.importorskip("networkx")
    url, app, state = route_server
    routes_before = len([ly for ly in state.layers.values() if ly.kind == "route"])

    page.locator("#sidebar-tabs a", has_text="Routes").click()
    _open_sections(page)
    if page.locator("#edit-enable").is_checked():
        page.locator("#edit-enable").uncheck()
    page.locator("#new-route-btn").click()                    # leave the route the edit test left selected

    page.locator("#route-graph-api").click()
    page.locator("[role=option]", has_text="networkx").click()
    # changing the backend reset the algorithm to its default and filtered the list
    page.wait_for_function(
        "document.getElementById('route-algorithm').innerText.includes('dijkstra')", timeout=15_000)
    page.locator("#route-algorithm").click()
    page.locator("[role=option]", has_text="bidirectional dijkstra").click()
    page.wait_for_function(
        "document.getElementById('route-algorithm').innerText.includes('bidirectional')", timeout=15_000)

    box = page.locator(".leaflet-container").bounding_box()
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    page.locator("#build-mode label", has_text="Source").click()
    page.wait_for_function(
        "document.getElementById('mode-badge').innerText.includes('SOURCE')", timeout=15_000)
    page.mouse.click(cx - 35, cy + 25)
    page.locator("#build-points-table .ag-row").first.wait_for(timeout=15_000)
    page.locator("#build-mode label", has_text="Target").click()
    page.wait_for_function(
        "document.getElementById('mode-badge').innerText.includes('TARGET')", timeout=15_000)
    page.mouse.click(cx + 35, cy - 25)
    page.wait_for_function(
        "document.querySelectorAll('#build-points-table .ag-row').length >= 2", timeout=15_000)

    page.click("#run-routing-btn")
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        routes = [ly for ly in state.layers.values() if ly.kind == "route"]
        if len(routes) > routes_before:
            break
        time.sleep(0.5)
    routes = [ly for ly in state.layers.values() if ly.kind == "route"]
    assert len(routes) == routes_before + 1
    params = routes[-1].meta["params"]
    assert params["graph_api"] == "networkx"
    assert params["algorithm"] == "bidirectional_dijkstra"
    assert routes[-1].meta["metrics"]["total_cost"] > 0
    assert routes[-1].id not in state.search_sessions             # run through find_route, not a session
