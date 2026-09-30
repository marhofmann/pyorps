"""
Playwright E2E for the URL-served GeoJSON perf path (2026-07-15): a large vector
layer is rendered as ``dl.GeoJSON(url=…)`` so its data is fetched once from the
Flask ``/_gj/<id>`` route instead of riding in the callback response. This test
proves that in a real browser the layer still (a) paints and (b) stays clickable
(clickData → attribute inspection), which is the only behaviour the URL path
could plausibly change.
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
def big_server():
    import geopandas as gpd
    from shapely.geometry import box

    from pyorps.gui import ids
    from pyorps.gui.app import build_app
    from pyorps.gui.callbacks.layers import URL_GEOJSON_THRESHOLD
    from pyorps.gui.services import geo
    from pyorps.gui.state import ProjectState

    state = ProjectState()
    ox, oy = 475000.0, 5550000.0
    n = URL_GEOJSON_THRESHOLD + 60           # over the URL-serving threshold
    polys = [box(ox + (i % 30) * 12, oy + (i // 30) * 12,
                 ox + (i % 30) * 12 + 9, oy + (i // 30) * 12 + 9)
             for i in range(n)]
    gdf = gpd.GeoDataFrame(
        {"kind": [f"k{i % 5}" for i in range(n)], "idx": list(range(n))},
        geometry=polys, crs="EPSG:25832")
    layer = state.add_layer("BigLayer", "vector", gdf=gdf, crs=gdf.crs,
                            geojson=geo.gdf_to_wgs84_geojson(gdf),
                            style={"color": "#ff8800"})

    app = build_app(state)
    bounds = geo.wgs84_bounds(layer.geojson)          # [[s,w],[n,e]]
    for component in app.layout._traverse():
        if getattr(component, "id", None) == ids.LAYERS_VIEW:
            component.data = state.layers_view()
        elif getattr(component, "id", None) == ids.MAP:
            component.center = [(bounds[0][0] + bounds[1][0]) / 2,
                                (bounds[0][1] + bounds[1][1]) / 2]
            component.zoom = 16

    port = _free_port()
    server = make_server("127.0.0.1", port, app.server)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}", state
    finally:
        server.shutdown()
        state.shutdown()


@pytest.fixture(scope="module")
def page(big_server):
    url = big_server[0]
    try:
        pw = playwright_sync.sync_playwright().start()
        browser = pw.chromium.launch()
    except Exception as exc:  # pragma: no cover - no browser installed
        pytest.skip(f"chromium not launchable: {exc}")
    page = browser.new_page()
    requests = []
    page.on("request", lambda r: requests.append(r.url))
    page.goto(url)
    page.wait_for_selector(".leaflet-container", timeout=15_000)
    yield page, requests
    browser.close()
    pw.stop()


def test_url_served_layer_fetches_paints_and_clicks(page):
    pg, requests = page
    # (a) the big layer paints — which only happens if leaflet fetched the
    # layer from its URL route (geobuf /_gb/ preferred, GeoJSON /_gj/ as
    # fallback when the encoder is missing)
    pg.wait_for_selector("path.leaflet-interactive[stroke='#ff8800']",
                         timeout=20_000)
    # the data came from the Flask route, not the callback response
    assert any("/_gb/" in u or "/_gj/" in u for u in requests)
    try:
        import geobuf  # noqa: F401
        assert any("/_gb/" in u for u in requests)   # binary wire format won
    except ImportError:
        pass
    # (b) a URL-loaded feature still emits clickData -> Attrs panel populates
    path = pg.locator("path.leaflet-interactive[stroke='#ff8800']").first
    path.click()
    pg.locator("#sidebar-tabs a", has_text="Attrs").click()
    panel = pg.wait_for_selector("#attr-panel table", timeout=15_000)
    assert "kind" in panel.inner_text() or "idx" in panel.inner_text()
