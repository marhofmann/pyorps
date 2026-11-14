"""
Playwright E2E (round 15): the grids show their rows and sensible columns whenever they are visible.

The routes grid sits in the closed "Computed routes" card, far down the Routes tab. It used to be fitted while the card
was closed (columns squeezed to ~36 px, never fitted again) and, scrolled away, AG Grid's ``content-visibility: auto``
kept its old rendering, so the edited route never appeared in its text. ``assets/grid-fit.js`` now fits a grid when it
is first shown, and ``gui.css`` renders the grids also while they are scrolled away.
"""
from __future__ import annotations

import socket
import threading

import pytest

pytest.importorskip("dash")
playwright_sync = pytest.importorskip("playwright.sync_api")

from werkzeug.serving import make_server

#: no column of a visible grid is narrower (the narrowest column of the app has maxWidth 50 px)
MIN_COLUMN_PX = 40

#: the column widths of every visible grid: the grid host is rendered and no closed card is around it
_VISIBLE_GRID_COLUMNS = """() => [...document.querySelectorAll('.ag-root-wrapper')].map(w => {
    const host = w.closest('[id]');
    return {host, id: host && host.id};
}).filter(({host}) => host && host.checkVisibility() && !host.closest('details:not([open])')
                      && host.getBoundingClientRect().width > 0)
  .map(({host, id}) => ({id, widths: [...host.querySelectorAll('.ag-header-row-column .ag-header-cell')]
                                     .map(c => Math.round(c.getBoundingClientRect().width))}))"""

#: the routes grid shows ``text`` in a rendered row (innerText skips cells AG Grid still hides) with a readable name
_ROUTE_ROW_SHOWN = """(text) => {
    const grid = document.getElementById('routes-grid');
    if (!grid || !grid.innerText.includes(text)) return false;
    return [...grid.querySelectorAll(".ag-row .ag-cell[col-id='name']")].some(
        c => c.innerText.includes(text) && c.getBoundingClientRect().width > MIN_PX);
}""".replace("MIN_PX", str(MIN_COLUMN_PX))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def grid_server(tmp_path_factory):
    from pyorps.gui import ids
    from pyorps.gui.app import build_app
    from pyorps.gui.callbacks.raster import add_raster_layer
    from pyorps.gui.state import ProjectState
    from pyorps.raster.handler import create_test_tiff

    raster = tmp_path_factory.mktemp("e2e_grids") / "cost.tif"
    create_test_tiff(str(raster), width=200, height=200, pattern="gradient", crs="EPSG:32632")
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
            component.zoom = 16                 # the 200 m raster spans ~130 px: the clicks below stay inside it

    port = _free_port()
    server = make_server("127.0.0.1", port, app.server)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{port}", state
    finally:
        server.shutdown()
        state.shutdown()


@pytest.fixture(scope="module")
def page(grid_server):
    url, _state = grid_server
    try:
        pw = playwright_sync.sync_playwright().start()
        browser = pw.chromium.launch()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"chromium not launchable: {exc}")
    page = browser.new_page()
    warnings: list[str] = []
    page.on("console", lambda msg: warnings.append(msg.text) if msg.type in ("warning", "error") else None)
    page.goto(url)
    page.wait_for_selector(".leaflet-container", timeout=15_000)
    page.console_warnings = warnings
    yield page
    browser.close()
    pw.stop()


def _tab(page, name):
    page.locator("#sidebar-tabs a", has_text=name).click()


def _toggle_routes_card(page):
    page.locator("details.gui-card-collapsible > summary", has_text="Computed routes").click()


def _routes_card_open(page) -> bool:
    return page.evaluate("document.getElementById('routes-grid').closest('details').open")


def _map_centre(page):
    box = page.locator(".leaflet-container").bounding_box()
    return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2


def _expect_route_row(page, text):
    page.wait_for_function(_ROUTE_ROW_SHOWN, arg=text, timeout=30_000)


def _expect_sensible_columns(page):
    page.wait_for_timeout(300)                  # a fit runs right after the grid got its width
    grids = page.evaluate(_VISIBLE_GRID_COLUMNS)
    assert grids, "no visible grid"
    narrow = {g["id"]: g["widths"] for g in grids if any(width <= MIN_COLUMN_PX for width in g["widths"])}
    assert not narrow, f"visible grids with columns of {MIN_COLUMN_PX} px or less: {narrow}"


def test_routes_grid_shows_the_route_after_the_card_opens(page, grid_server):
    _url, state = grid_server
    _tab(page, "Routes")
    page.locator("#route-raster").click()
    page.locator("[role=option]").last.click()
    cx, cy = _map_centre(page)
    page.locator("#build-mode label", has_text="Source").click()
    page.wait_for_function("document.getElementById('mode-badge').innerText.includes('SOURCE')", timeout=15_000)
    page.mouse.click(cx - 40, cy - 30)
    page.locator("#build-points-table .ag-row").first.wait_for(timeout=15_000)
    page.locator("#build-mode label", has_text="Target").click()
    page.wait_for_function("document.getElementById('mode-badge').innerText.includes('TARGET')", timeout=15_000)
    page.mouse.click(cx + 40, cy + 30)
    page.wait_for_function("document.querySelectorAll('#build-points-table .ag-row').length >= 2", timeout=15_000)
    page.click("#run-routing-btn")
    page.wait_for_function("document.getElementById('routing-status').innerText.includes('built')", timeout=60_000)
    routes = [ly for ly in state.layers.values() if ly.kind == "route"]
    assert len(routes) == 1

    # the route arrived while the card was closed; opening it shows the row in readable columns
    assert not _routes_card_open(page)
    _toggle_routes_card(page)
    _expect_route_row(page, routes[0].name)
    _expect_sensible_columns(page)


def test_routes_grid_keeps_its_rows_after_a_tab_switch_and_closing_the_card(page, grid_server):
    _url, state = grid_server
    name = next(ly.name for ly in state.layers.values() if ly.kind == "route")
    _tab(page, "Layers")
    _expect_sensible_columns(page)              # the layers grid, shown for the first time now
    _tab(page, "Routes")
    _expect_route_row(page, name)
    _expect_sensible_columns(page)

    _toggle_routes_card(page)                   # close ...
    page.wait_for_function("!document.getElementById('routes-grid').closest('details').open", timeout=5_000)
    _toggle_routes_card(page)                   # ... and open again
    _expect_route_row(page, name)
    _expect_sensible_columns(page)


def test_routes_grid_shows_the_edited_route_while_scrolled_away(page, grid_server):
    _url, state = grid_server
    _tab(page, "Routes")
    # all cards open: the routes grid is then far below the window while the map click edits the route
    page.eval_on_selector_all("details.gui-card-collapsible", "els => els.forEach(e => { e.open = true; })")
    page.locator("#edit-route-select").click()
    page.locator("[role=option]").first.click()
    page.wait_for_selector("path.leaflet-interactive[stroke='#ffe119']", timeout=15_000)
    page.locator("#edit-enable").check()
    page.locator("#build-mode label", has_text="Target").click()
    page.wait_for_function("document.getElementById('mode-badge').innerText.includes('MOVES TARGET')",
                           timeout=15_000)
    cx, cy = _map_centre(page)
    page.mouse.click(cx + 20, cy + 45)

    _expect_route_row(page, "moved target")
    routes = [ly for ly in state.layers.values() if ly.kind == "route"]
    assert len(routes) == 1 and routes[0].meta["edit"] == "moved target"
    _expect_sensible_columns(page)


def test_routes_grid_shows_the_route_after_a_reload(page, grid_server):
    _url, state = grid_server
    name = next(ly.name for ly in state.layers.values() if ly.kind == "route")
    page.console_warnings.clear()
    page.reload()
    page.wait_for_selector(".leaflet-container", timeout=15_000)
    _tab(page, "Routes")
    if not _routes_card_open(page):
        _toggle_routes_card(page)
    _expect_route_row(page, name)
    _expect_sensible_columns(page)
    # no grid was fitted while it had no width (AG Grid warning #29)
    for tab in ("Cost", "Layers", "Data"):
        _tab(page, tab)
    page.wait_for_timeout(500)
    assert not [w for w in page.console_warnings if "sizeColumnsToFit" in w or "#29" in w]
