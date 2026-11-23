"""Playwright E2E, round 14: the app shell fits the window and the grids raise no deprecation warnings.

* Fix 2: at 1440 x 900 and 1280 x 720 the document is never taller than the window on any tab (sections closed
  or all open), and the last control of the Routes tab scrolls into view INSIDE the tab pane, without shifting
  the page.
* Fix 3: no AG Grid deprecation warning reaches the browser console on any tab, and the grids still select rows
  the way they did (click, ctrl-click, no checkbox column, a re-click keeps the selection of the layers grid).
"""
from __future__ import annotations

import socket
import threading

import pytest

pytest.importorskip("dash")
playwright_sync = pytest.importorskip("playwright.sync_api")

from werkzeug.serving import make_server

VIEWPORTS = [(1440, 900), (1280, 720)]
TABS = ["Data", "Cost", "Raster", "Routes", "Layers", "Attrs"]
PANE = ".gui-sidebar .tab-content"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def gui_url(tmp_path_factory):
    """A live app with three layers, so the layers grid has rows to select."""
    from pyorps.gui import ids
    from pyorps.gui.app import build_app
    from pyorps.gui.callbacks.raster import add_raster_layer
    from pyorps.gui.state import ProjectState
    from pyorps.raster.handler import create_test_tiff

    raster = tmp_path_factory.mktemp("e2e_round14") / "cost.tif"
    create_test_tiff(str(raster), width=100, height=100, pattern="gradient", crs="EPSG:32632")

    state = ProjectState()
    for i in range(2):
        square = {"type": "FeatureCollection", "features": [{
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[
                [8.6 + i, 50.0], [8.8 + i, 50.0], [8.8 + i, 50.2], [8.6 + i, 50.2], [8.6 + i, 50.0]]]},
            "properties": {"name": f"square-{i}"}}]}
        state.add_layer(f"Square {i}", "vector", geojson=square, style={"color": "#00aa88"})
    add_raster_layer(state, str(raster), [])
    app = build_app(state)
    for component in app.layout._traverse():
        cid = getattr(component, "id", None)
        if cid == ids.LAYERS_VIEW:
            component.data = state.layers_view()
        elif cid == ids.MAP:
            component.center = [50.1, 8.7]
            component.zoom = 9

    port = _free_port()
    server = make_server("127.0.0.1", port, app.server)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        state.shutdown()


@pytest.fixture(scope="module")
def browser():
    try:
        pw = playwright_sync.sync_playwright().start()
        launched = pw.chromium.launch()
    except Exception as exc:  # pragma: no cover - no browser installed
        pytest.skip(f"chromium not launchable: {exc}")
    yield launched
    launched.close()
    pw.stop()


def _open_page(browser, url, width, height):
    page = browser.new_page(viewport={"width": width, "height": height})
    messages: list[tuple[str, str]] = []
    page.on("console", lambda m: messages.append((m.type, m.text)))
    page.goto(url)
    page.wait_for_selector(".leaflet-container", timeout=30_000)
    page.wait_for_selector("#layers-grid .ag-row", state="attached", timeout=30_000)
    return page, messages


def _select_tab(page, label: str) -> None:
    page.locator("#sidebar-tabs .nav-link", has_text=label).first.click()
    page.wait_for_timeout(400)


def _open_sections(page) -> None:
    page.evaluate("document.querySelectorAll('details.gui-card-collapsible')"
                  ".forEach(d => { d.open = true; })")
    page.wait_for_timeout(300)


def _fits(page) -> dict:
    return page.evaluate("""() => ({
        scroll: document.documentElement.scrollHeight, inner: window.innerHeight,
        docTop: document.documentElement.scrollTop, bodyTop: document.body.scrollTop,
        shellTop: document.querySelector('.gui-shell').scrollTop,
        shellFits: document.querySelector('.gui-shell').scrollHeight <= window.innerHeight})""")


def _last_control(page):
    """Handle of the last visible control in the active tab pane."""
    return page.evaluate_handle("""() => {
        const pane = document.querySelector('.gui-sidebar .tab-content .tab-pane.active');
        const sel = 'button, input, select, textarea, summary, a[href]';
        const items = [...pane.querySelectorAll(sel)].filter(e => {
            const r = e.getBoundingClientRect();
            return r.width > 0 && r.height > 0 && !e.classList.contains('dash-dropdown-focus-target')
                && !e.closest('.visually-hidden') && !e.closest('.ag-root-wrapper');
        });
        return items[items.length - 1];
    }""").as_element()


@pytest.mark.parametrize(("width", "height"), VIEWPORTS)
def test_every_tab_fits_the_window_and_scrolls_inside_its_pane(browser, gui_url, width, height):
    page, _ = _open_page(browser, gui_url, width, height)
    try:
        for opened in (False, True):
            for tab in TABS:
                _select_tab(page, tab)
                if opened:
                    _open_sections(page)
                fit = _fits(page)
                where = f"{tab} tab, sections {'open' if opened else 'closed'}, {width}x{height}"
                assert fit["scroll"] <= fit["inner"], f"document taller than the window on the {where}: {fit}"
                assert fit["shellFits"], f"app shell overflows on the {where}"

        # the last control of the Routes tab is reachable by scrolling the pane, not the page
        _select_tab(page, "Routes")
        _open_sections(page)
        control = _last_control(page)
        assert control is not None
        control.scroll_into_view_if_needed()
        page.wait_for_timeout(200)
        box, pane = control.bounding_box(), page.locator(PANE).bounding_box()
        assert box["y"] >= pane["y"] - 1
        assert box["y"] + box["height"] <= pane["y"] + pane["height"] + 1
        fit = _fits(page)
        assert fit["docTop"] == fit["bodyTop"] == fit["shellTop"] == 0, f"the page moved: {fit}"
        assert fit["scroll"] <= fit["inner"]
        assert page.evaluate(f"document.querySelector('{PANE}').scrollTop") > 0   # the pane did the scrolling
    finally:
        page.close()


def test_the_detector_sees_the_old_defect(browser, gui_url):
    """Without the positioned shell the hidden labels of the Routes tab stretch the document again."""
    page, _ = _open_page(browser, gui_url, 1440, 900)
    try:
        page.add_style_tag(content=".gui-shell, .gui-sidebar .tab-content { position: static !important; }")
        _select_tab(page, "Routes")
        _open_sections(page)
        fit = _fits(page)
        assert fit["scroll"] > fit["inner"], fit
    finally:
        page.close()


@pytest.mark.parametrize(("width", "height"), VIEWPORTS)
def test_no_ag_grid_deprecation_warning_on_any_tab(browser, gui_url, width, height):
    page, messages = _open_page(browser, gui_url, width, height)
    try:
        for tab in TABS:
            _select_tab(page, tab)
            _open_sections(page)
        page.wait_for_timeout(500)
    finally:
        page.close()
    ag_grid = [text for kind, text in messages if "AG Grid" in text]
    deprecated = [text for text in ag_grid if "deprecated" in text.lower()]
    assert deprecated == [], deprecated


def _selected(page) -> int:
    return page.locator("#layers-grid .ag-row-selected").count()


def test_layers_grid_selection_behaves_as_before(browser, gui_url):
    page, _ = _open_page(browser, gui_url, 1440, 900)
    try:
        _select_tab(page, "Layers")
        cells = page.locator("#layers-grid .ag-center-cols-container .ag-cell[col-id='kind']")
        assert cells.count() >= 3
        assert page.locator("#layers-grid .ag-selection-checkbox").count() == 0   # no checkbox column
        cells.nth(0).click()
        page.wait_for_timeout(300)
        assert _selected(page) == 1
        cells.nth(1).click()
        page.wait_for_timeout(300)
        assert _selected(page) == 1                       # a plain click replaces the selection
        cells.nth(2).click(modifiers=["Control"])
        page.wait_for_timeout(300)
        assert _selected(page) == 2                       # ctrl-click adds a row
        cells.nth(2).click(modifiers=["Control"])
        page.wait_for_timeout(300)
        assert _selected(page) == 2                       # ... and never deselects it (the old suppressRowDeselection)
        cells.nth(1).click()
        page.wait_for_timeout(300)
        assert _selected(page) == 1
    finally:
        page.close()
