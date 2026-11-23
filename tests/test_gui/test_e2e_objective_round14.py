"""Playwright E2E, round 14, P5: the "Objective & terrain" card in a real browser.

The headless tests (``test_gui_objective_ui_round14.py``) drive the callbacks; this file checks what only a browser
shows: the card is closed and opens, the slope block and the 32-bit choice are disabled for real (a disabled
fieldset), a conflict between a terrain model and delta-stepping reaches the status line before any run, the layer
table adds and removes a row, a saved card comes back on a reload, and a "Go fix" focus on a control inside a closed
card opens that card (and the "Algorithm & search" card for the Algorithm field).
"""
from __future__ import annotations

import socket
import threading

import numpy as np
import pytest

pytest.importorskip("dash")
playwright_sync = pytest.importorskip("playwright.sync_api")

import rasterio
from rasterio.transform import from_origin
from werkzeug.serving import make_server

from pyorps.gui import ids

CARD = f"#{ids.OBJ_CARD}"
SIZE = 100


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _write_tif(path, array):
    height, width = array.shape
    with rasterio.open(path, "w", driver="GTiff", height=height, width=width, count=1, dtype=str(array.dtype),
                       crs="EPSG:32632", transform=from_origin(500000.0, 5600000.0, 1, 1)) as dst:
        dst.write(array, 1)
    return str(path)


def _serve(folder, saved=None):
    """A live app with a cost raster, a DEM and a stripe layer; ``saved`` (a function of the layers) sets the
    objective settings the server holds before the page is built (the first paint of a reload)."""
    from pyorps.gui.app import build_app
    from pyorps.gui.callbacks.raster import add_raster_layer
    from pyorps.gui.state import ProjectState
    from pyorps.raster.handler import create_test_tiff

    cost = folder / "cost.tif"
    create_test_tiff(str(cost), width=SIZE, height=SIZE, pattern="gradient", crs="EPSG:32632")
    dem = _write_tif(folder / "dem.tif", np.linspace(0, 20, SIZE * SIZE, dtype="float32").reshape(SIZE, SIZE))
    stripe = _write_tif(folder / "stripe.tif", np.full((SIZE, SIZE), 3.0, dtype="float32"))
    state = ProjectState()
    layers = {}
    for key, path, name in (("cost", str(cost), "Cost"), ("dem", dem, "DGM"), ("stripe", stripe, "landscape")):
        layers[key], _ = add_raster_layer(state, path, [], name=name)
    if saved is not None:
        state.gui_settings["objective"] = saved(layers)
    app = build_app(state)
    port = _free_port()
    server = make_server("127.0.0.1", port, app.server)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return state, server, f"http://127.0.0.1:{port}"


@pytest.fixture(scope="module")
def live_app(tmp_path_factory):
    state, server, url = _serve(tmp_path_factory.mktemp("e2e_objective"))
    try:
        yield state, url
    finally:
        server.shutdown()
        state.shutdown()


@pytest.fixture()
def gui_url(live_app):
    """The URL of the live app. The server keeps the card's settings (that is how a reload restores them), so
    every test starts from a blank card."""
    state, url = live_app
    state.gui_settings.clear()
    return url


@pytest.fixture(scope="module")
def saved_gui_url(tmp_path_factory):
    def saved(layers):
        return {"weights": {"cost": 1.0, "length": 0.4, "gradient": 0.0}, "dem_layer_id": layers["dem"].id,
                "layers": [{"name": "landscape", "source": layers["stripe"].id, "weight": 0.5, "hard_max": 30.0,
                            "hard_min": None}],
                "gradient": {"multiplier": "energy", "mult_scale": 5.0}}

    state, server, url = _serve(tmp_path_factory.mktemp("e2e_objective_saved"), saved)
    try:
        yield url
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


def _open_page(browser, url):
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto(url)
    page.wait_for_selector(".leaflet-container", timeout=30_000)
    page.locator("#sidebar-tabs .nav-link", has_text="Routes").first.click()
    page.wait_for_selector(CARD, state="attached")
    return page


def _is_open(page, selector) -> bool:
    return bool(page.eval_on_selector(selector, "e => e.open"))


def _open_card(page) -> None:
    if not _is_open(page, CARD):
        page.locator(f"{CARD} > summary").click()
    page.wait_for_selector(f"#{ids.OBJ_STATUS}", state="visible")


def _disabled(page, component_id) -> bool:
    return bool(page.eval_on_selector(f"#{component_id}", "e => e.matches(':disabled')"))


def _pick(page, component_id, option_text) -> None:
    page.locator(f"#{component_id}").click()
    page.locator("[role=option]", has_text=option_text).first.click()
    page.wait_for_timeout(400)


def _status(page) -> str:
    return page.locator(f"#{ids.OBJ_STATUS}").inner_text()


def _wait_status(page, fragment: str) -> str:
    page.wait_for_function(
        "([id, fragment]) => document.getElementById(id).innerText.includes(fragment)",
        arg=[ids.OBJ_STATUS, fragment], timeout=10_000)
    return _status(page)


def test_the_card_is_closed_by_default_and_opens(browser, gui_url):
    page = _open_page(browser, gui_url)
    try:
        assert _is_open(page, CARD) is False
        assert page.locator(f"{CARD} > summary").text_content().strip().startswith("Objective & terrain")
        _open_card(page)
        assert _is_open(page, CARD) is True
        assert _wait_status(page, "Classic cost routing")
    finally:
        page.close()


def test_the_slope_block_the_slope_weight_and_32_bit_are_disabled_until_they_apply(browser, gui_url):
    page = _open_page(browser, gui_url)
    try:
        _open_card(page)
        assert _disabled(page, ids.OBJ_W_GRADIENT) and _disabled(page, ids.OBJ_GRAD_MAX_PCT)
        assert page.get_attribute(f"#{ids.OBJ_SLOPE_WRAP}", "title") == "Needs a terrain model (DEM)"
        assert page.get_attribute(f"#{ids.OBJ_W_GRADIENT_WRAP}", "title") == "Needs a terrain model (DEM)"
        assert _disabled(page, ids.OBJ_W_LENGTH) is False
        float_input = f"#{ids.OBJ_PRECISION} input[value='float32']"
        assert page.eval_on_selector(float_input, "e => e.disabled") is True
        assert page.get_attribute(f"#{ids.OBJ_PRECISION_WRAP}", "title").startswith("Only with an objective")
        # a terrain model switches the slope controls on
        _pick(page, ids.OBJ_DEM_LAYER, "DGM")
        page.wait_for_function("id => !document.getElementById(id).matches(':disabled')", arg=ids.OBJ_W_GRADIENT)
        assert _disabled(page, ids.OBJ_GRAD_MAX_PCT) is False
        # ... and the conflict with the default algorithm is named before any run
        text = _wait_status(page, "delta-stepping")
        assert text.startswith("Algorithm is delta-stepping") and "choose Dijkstra" in text
    finally:
        page.close()


def test_changing_the_algorithm_clears_the_conflict_in_the_status_line(browser, gui_url):
    page = _open_page(browser, gui_url)
    try:
        _open_card(page)
        _pick(page, ids.OBJ_DEM_LAYER, "DGM")
        _wait_status(page, "delta-stepping")
        page.evaluate("document.querySelectorAll('details.gui-card-collapsible').forEach(d => { d.open = true; })")
        _pick(page, ids.ALGORITHM, "dijkstra")
        page.wait_for_function(
            "id => !document.getElementById(id).innerText.includes('delta-stepping')", arg=ids.OBJ_STATUS)
        assert _status(page).startswith("Objective: cost 1; DEM 'DGM'")
        assert "cython runs Dijkstra only" in _status(page)
    finally:
        page.close()


def test_the_presets_write_the_weights_and_the_status_follows(browser, gui_url):
    page = _open_page(browser, gui_url)
    try:
        _open_card(page)
        page.locator(f"#{ids.OBJ_PRESET_SHORTEST_BTN}").click()
        _wait_status(page, "Objective: length 1")
        assert page.locator(f"#{ids.OBJ_W_COST}").input_value() == "0"
        assert page.locator(f"#{ids.OBJ_W_LENGTH}").input_value() == "1"
        page.locator(f"#{ids.OBJ_PRESET_CLASSIC_BTN}").click()
        _wait_status(page, "Classic cost routing")
        assert page.locator(f"#{ids.OBJ_W_COST}").input_value() == "1"
    finally:
        page.close()


def test_a_layer_is_added_to_the_table_and_removed_again(browser, gui_url):
    page = _open_page(browser, gui_url)
    try:
        _open_card(page)
        rows = f"#{ids.OBJ_LAYERS_GRID} .ag-center-cols-container .ag-row"
        # a reserved name is refused with a notice that names the field
        page.locator(f"#{ids.OBJ_LAYER_NAME}").fill("cost")
        _pick(page, ids.OBJ_LAYER_SOURCE, "landscape")
        page.locator(f"#{ids.OBJ_LAYER_ADD_BTN}").click()
        page.wait_for_selector("text=which is reserved", timeout=10_000)
        assert page.locator(rows).count() == 0
        # a good one is added and counted in the status line
        page.locator(f"#{ids.OBJ_LAYER_NAME}").fill("scenic")
        page.locator(f"#{ids.OBJ_LAYER_ADD_BTN}").click()
        page.wait_for_selector(rows, timeout=10_000)
        assert page.locator(rows).count() == 1
        assert "scenic" in page.locator(rows).first.inner_text()
        assert "landscape" in page.locator(rows).first.inner_text()          # the label of the source layer
        assert "scenic 1" in _wait_status(page, "scenic 1")
        assert page.locator(f"#{ids.OBJ_LAYER_NAME}").input_value() == ""
        # remove it
        page.locator(rows).first.click()
        page.locator(f"#{ids.OBJ_LAYER_DEL_BTN}").click()
        page.wait_for_function("sel => document.querySelectorAll(sel).length === 0", arg=rows, timeout=10_000)
        _wait_status(page, "Classic cost routing")
    finally:
        page.close()


def test_a_saved_card_comes_back_on_a_reload(browser, saved_gui_url):
    page = _open_page(browser, saved_gui_url)
    try:
        _open_card(page)
        assert page.locator(f"#{ids.OBJ_W_LENGTH}").input_value() == "0.4"
        assert page.locator(f"#{ids.OBJ_GRAD_MULT_SCALE}").input_value() == "5"
        assert page.locator(f"#{ids.OBJ_DEM_LAYER}").inner_text().strip().startswith("DGM")
        rows = f"#{ids.OBJ_LAYERS_GRID} .ag-center-cols-container .ag-row"
        page.wait_for_selector(rows, timeout=10_000)
        assert "landscape" in page.locator(rows).first.inner_text()
        # the saved terrain model meets the default algorithm: the conflict is named at once, before any run
        text = _wait_status(page, "delta-stepping")
        assert text.startswith("Algorithm is delta-stepping")
        assert _disabled(page, ids.OBJ_W_GRADIENT) is False                  # the saved DEM enables the slope block
    finally:
        page.close()


def _focus(page, control: str, seq: int) -> None:
    """What a notice's "Go fix" button writes into the focus store."""
    page.evaluate("([control, seq]) => window.dash_clientside.set_props('focus-flash', "
                  "{data: {control: control, seq: seq}})", [control, seq])
    page.wait_for_timeout(700)


def test_a_go_fix_focus_opens_the_closed_card_and_flashes_the_control(browser, gui_url):
    page = _open_page(browser, gui_url)
    try:
        assert _is_open(page, CARD) is False
        _focus(page, ids.OBJ_W_LENGTH, 1)
        assert _is_open(page, CARD) is True
        assert page.eval_on_selector(f"#{ids.OBJ_W_LENGTH}", "e => e.classList.contains('pyorps-flash')") is True
        # the user closes the card again: the next focus opens it again (the page state is not the DOM state)
        page.locator(f"{CARD} > summary").click()
        assert _is_open(page, CARD) is False
        _focus(page, ids.OBJ_DEM_LAYER, 2)
        assert _is_open(page, CARD) is True
    finally:
        page.close()


def test_a_go_fix_focus_on_the_algorithm_field_opens_its_closed_card_too(browser, gui_url):
    page = _open_page(browser, gui_url)
    try:
        selector = "details.gui-card-collapsible:has(#route-algorithm)"
        assert _is_open(page, selector) is False
        _focus(page, ids.ALGORITHM, 1)
        assert _is_open(page, selector) is True
        assert page.locator(f"#{ids.ALGORITHM}").is_visible()
    finally:
        page.close()
