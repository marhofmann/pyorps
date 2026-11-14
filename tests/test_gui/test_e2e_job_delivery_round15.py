"""
Round 15 browser tests: a background job's outcome reaches the page although its finalize step is slow.

Before round 15 the poll tick that found an ended job also ran the finalize step; when that took longer than
the poll interval the browser dropped the tick's answer (a newer tick had been sent) and the newer tick
switched the poll off: the layer was in the project but never on the page. Here a rasterize run (forced into
the background by lowering ``rasterize.RASTERIZE_BACKGROUND_CELLS`` in this process, which serves the app) and
a cost-field run get a finalize step that is slowed to several poll intervals; the layer and the log or the
status must appear, the status line must have said "adding the result" meanwhile, Stop must be hidden and the
poll must stop. The server is threaded, as waitress is, so ticks are served while the finalize step runs.

Run explicitly with::

    pytest tests/test_gui/test_e2e_job_delivery_round15.py -v
"""
from __future__ import annotations

import socket
import threading
import time

import pytest

pytest.importorskip("dash")
playwright_sync = pytest.importorskip("playwright.sync_api")

from werkzeug.serving import make_server

#: the finalize step is held this long: several poll intervals (800 ms)
SLOW_S = 3.5
#: a poll that still fires sends a request within one interval; watch for a few
QUIET_S = 2.5
#: the receipt tick comes within one interval after the outcome
RECEIPT_S = 1.5

_WATCH_STATUS = """() => {
    window.__seen = {log: [], cf: []};
    const record = () => {
        for (const [key, id] of [['log', 'rasterize-log'], ['cf', 'cf-status']]) {
            const el = document.getElementById(id);
            if (!el) continue;
            const seen = window.__seen[key];
            if (seen[seen.length - 1] !== el.innerText) seen.push(el.innerText);
        }
    };
    new MutationObserver(record).observe(document.body, {subtree: true, childList: true, characterData: true});
    record();
}"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def gui_server():
    from pyorps.gui.app import build_app
    from pyorps.gui.state import ProjectState

    state = ProjectState()
    app = build_app(state)
    port = _free_port()
    server = make_server("127.0.0.1", port, app.server, threaded=True)     # ticks in parallel, like waitress
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}", app, state
    finally:
        server.shutdown()
        state.shutdown()


@pytest.fixture(scope="module")
def browser(gui_server):
    try:
        pw = playwright_sync.sync_playwright().start()
        chromium = pw.chromium.launch()
    except Exception as exc:  # pragma: no cover - no browser installed
        pytest.skip(f"chromium not launchable: {exc}")
    yield chromium
    chromium.close()
    pw.stop()


@pytest.fixture()
def page(browser, gui_server):
    url, _app, _state = gui_server
    tab = browser.new_page()
    tab.on("dialog", lambda dialog: dialog.accept())         # the unsaved-work guard of a reload
    tab.goto(url)
    tab.wait_for_selector(".leaflet-container", timeout=30_000)
    yield tab
    tab.close()


class PollRequests:
    """Counts the requests of one poll (its ``n_intervals`` Input) the page sends."""

    def __init__(self, page, poll_id: str):
        self.marker = f'{{"id":"{poll_id}","property":"n_intervals"'
        self.times: list[float] = []
        page.on("request", self._seen)

    def _seen(self, request):
        if request.method == "POST" and "_dash-update-component" in request.url:
            if self.marker in (request.post_data or ""):
                self.times.append(time.monotonic())

    def since(self, moment: float) -> int:
        return sum(t > moment for t in self.times)


def _slowed(monkeypatch, module, name):
    """Wrap ``module.name`` so it waits ``SLOW_S`` first; returns the list of its calls."""
    original = getattr(module, name)
    calls: list[str] = []

    def slow(*args, **kwargs):
        calls.append(threading.current_thread().name)
        time.sleep(SLOW_S)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, slow)
    return calls


def _assert_poll_stopped(page, polls: PollRequests):
    # the tick after the outcome carries the receipt and switches the poll off; let it come first
    page.wait_for_timeout(RECEIPT_S * 1000)
    settled = time.monotonic()
    page.wait_for_timeout(QUIET_S * 1000)
    assert polls.since(settled) == 0, "the poll still fires after the outcome was handed over"


def test_a_background_rasterize_run_with_a_slow_finalize_shows_its_layer_and_log(page, gui_server, tmp_path,
                                                                                 monkeypatch):
    import geopandas as gpd
    from shapely.geometry import Polygon

    from pyorps.gui.callbacks import raster as raster_cb
    from pyorps.gui.services import rasterize

    _url, _app, state = gui_server
    monkeypatch.setattr(rasterize, "RASTERIZE_BACKGROUND_CELLS", 100)           # every build is a job
    calls = _slowed(monkeypatch, raster_cb, "finalize_rasterize")
    polls = PollRequests(page, "rasterize-poll")

    def sq(x, y, s=60):
        return Polygon([(x, y), (x + s, y), (x + s, y + s), (x, y + s)])

    gdf = gpd.GeoDataFrame({"nutzart": ["Wald", "Weg", "Landwirtschaft"], "bez": ["Nadelholz", "", "Ackerland"]},
                           geometry=[sq(500000, 5530000), sq(500060, 5530000), sq(500000, 5530060)],
                           crs="EPSG:25832")
    path = tmp_path / "landuse.geojson"
    gdf.to_file(path, driver="GeoJSON")

    page.locator("#sidebar-tabs a", has_text="Data").click()
    page.fill("#local-path", str(path))
    if page.is_checked("#clip-to-area"):
        page.uncheck("#clip-to-area")
    page.click("#local-load-btn")
    page.wait_for_function("document.getElementById('dataset-list').innerText.includes('landuse.geojson')",
                           timeout=15_000)
    page.locator("#sidebar-tabs a", has_text="Cost").click()
    page.locator("#cost-dataset").click()
    page.locator("[role=option]", has_text="landuse.geojson").click()
    page.wait_for_function("document.querySelector('#cost-feature-keys').innerText.includes('nutzart')",
                           timeout=15_000)
    page.click("#cost-seed-btn")
    page.locator("#cost-grid .ag-row", has_text="Nadelholz").first.wait_for(timeout=15_000)

    page.locator("#sidebar-tabs a", has_text="Raster").click()
    page.evaluate(_WATCH_STATUS)
    page.click("#rasterize-btn")
    page.wait_for_function("document.getElementById('rasterize-log').innerText.includes('saved')",
                           timeout=60_000)
    page.wait_for_selector("img.leaflet-tile[src*='localhost']", timeout=60_000)

    seen = page.evaluate("window.__seen.log")
    assert any(text.startswith("Rasterizing...") for text in seen), seen               # the job ran
    assert any("adding the result to the project" in text for text in seen), seen      # the finalize step
    assert "saved" in seen[-1] and "combination table" in seen[-1]
    assert page.evaluate("getComputedStyle(document.getElementById('rasterize-stop-btn')).display") == "none"
    _assert_poll_stopped(page, polls)
    assert calls and len(calls) == 1 and not calls[0].startswith("pyorps-")       # once, in a request thread
    assert len(polls.times) >= 4                                                 # ticks did fire meanwhile
    assert state.jobs == {} and len(state.layers_of_kind("raster")) == 1
    assert "saved" in page.evaluate("document.getElementById('rasterize-log').innerText")   # and it stays


def test_a_cost_field_run_with_a_slow_finalize_shows_its_layer_and_status(page, gui_server, raster_path,
                                                                          monkeypatch):
    from conftest import SOURCE, TARGET

    from pyorps.gui.callbacks.raster import add_raster_layer
    from pyorps.gui.services import cost_field as cost_field_service

    url, _app, state = gui_server
    layer, _notices = add_raster_layer(state, raster_path, [], name="field raster")
    assert layer is not None
    before = len(state.layers_of_kind("raster"))
    calls = _slowed(monkeypatch, cost_field_service, "finalize_cost_field")
    page.goto(url)                                            # the page lists the new raster layer
    page.wait_for_selector(".leaflet-container", timeout=30_000)
    polls = PollRequests(page, "cf-poll")

    draft = {"sources": [{"lat": 0, "lng": 0, "x": SOURCE[0], "y": SOURCE[1]},
                         {"lat": 0, "lng": 0, "x": TARGET[0], "y": TARGET[1]}], "targets": [], "waypoints": []}
    page.locator("#sidebar-tabs a", has_text="Routes").click()
    page.wait_for_function("typeof (window.dash_clientside || {}).set_props === 'function'", timeout=15_000)
    page.evaluate("""([layerId, draft]) => {
        window.dash_clientside.set_props('route-raster', {value: layerId});
        window.dash_clientside.set_props('route-draft', {data: draft});
        document.getElementById('cf-run-btn').closest('details').open = true;
    }""", [layer.id, draft])
    page.wait_for_function("!document.getElementById('cf-run-btn').disabled", timeout=15_000)
    page.evaluate(_WATCH_STATUS)
    page.click("#cf-run-btn")
    page.wait_for_function(
        """() => { const t = document.getElementById('cf-status').innerText;
                   return t && !t.includes('adding the result') && !t.startsWith('Cost field...'); }""",
        timeout=60_000)

    seen = page.evaluate("window.__seen.cf")
    assert any("adding the result to the project" in text for text in seen), seen
    assert "failed" not in seen[-1] and "stopped" not in seen[-1], seen
    assert page.evaluate("document.getElementById('cf-best-info').innerText").strip()
    assert page.evaluate("getComputedStyle(document.getElementById('cf-stop-btn')).display") == "none"
    _assert_poll_stopped(page, polls)
    assert len(calls) == 1 and not calls[0].startswith("pyorps-")
    assert state.jobs == {} and len(state.layers_of_kind("raster")) == before + 1
    page.locator("#sidebar-tabs a", has_text="Layers").click()
    newest = state.layers_of_kind("raster")[-1]
    page.locator("#layers-grid .ag-row", has_text=newest.name).first.wait_for(timeout=15_000)
