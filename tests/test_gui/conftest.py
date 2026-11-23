"""
Shared fixtures + the headless callback harness for pyorps.gui tests.

The harness invokes callbacks exactly as Dash would (through
``app.callback_map``), so a test failure means the *wiring* is broken, not
just the logic — this is the layer that was missing in webviz v1.
"""
from __future__ import annotations

import contextvars
import importlib.util
import json

import pytest

_GUI_MODULES = ("dash", "dash_leaflet", "dash_ag_grid", "dash_bootstrap_components", "localtileserver")
if any(importlib.util.find_spec(_m) is None for _m in _GUI_MODULES):
    # The [gui] extra is not installed: skip this directory. pytest.importorskip at conftest level
    # would abort the whole session (seen in the wheel tests with the [full] extra, which has no GUI).
    collect_ignore_glob = ["*"]
    AttributeDict = None
else:
    from dash._utils import AttributeDict

from pyorps.raster.handler import create_test_tiff

SOURCE = (500020.0, 5599980.0)
TARGET = (500180.0, 5599820.0)
TEST_CRS = "EPSG:32632"


# ------------------------------------------------------------------- harness
def find_callback(app, *fragments: str) -> tuple[str, dict]:
    """Locate a callback entry by output key + input fragments.

    The first fragment must appear in the callback-map key (the output spec,
    e.g. ``"layers-view.data"``); any further fragments must appear in the
    callback's registered inputs/state (disambiguates the several
    ``allow_duplicate`` writers of one store).
    """
    output_frag, input_frags = fragments[0], fragments[1:]
    matches = []
    for key, entry in app.callback_map.items():
        if output_frag not in key:
            continue
        wiring = json.dumps(entry.get("inputs", [])) + json.dumps(
            entry.get("state", []))
        if all(frag in wiring for frag in input_frags):
            matches.append(key)
    if len(matches) != 1:
        raise AssertionError(
            f"Expected exactly one callback matching {fragments}, found "
            f"{len(matches)}: {matches}\nAll keys: {list(app.callback_map)}")
    return matches[0], app.callback_map[matches[0]]


def invoke(app, fragments, *args, outputs_list=None, triggered=None):
    """Invoke a registered callback headlessly and return its response dict.

    ``fragments``: str or tuple of str identifying the callback by its output
    key. ``outputs_list``: dict (single output) or list of dicts; when None it
    is derived from the callback's registered outputs. ``triggered``: list of
    ``prop_id`` strings (e.g. ``"btn.n_clicks"``) or pattern dicts to set the
    dash context, so ``ctx.triggered_id`` works.
    """
    if isinstance(fragments, str):
        fragments = (fragments,)
    key, entry = find_callback(app, *fragments)
    fn = entry["callback"]

    if outputs_list is None:
        output = entry["output"]
        outputs = output if isinstance(output, (list, tuple)) else [output]
        specs = [{"id": out.component_id, "property": out.component_property}
                 for out in outputs]
        # multi-output iff the callback key uses the '..a...b..' form
        outputs_list = specs if key.startswith("..") else specs[0]

    kwargs = {"outputs_list": outputs_list}
    if triggered is not None:
        trg = []
        for t in triggered:
            if isinstance(t, dict):
                prop_id = (json.dumps(t["id"], sort_keys=True,
                                      separators=(",", ":"))
                           + "." + t["property"])
                trg.append({"prop_id": prop_id, "value": t.get("value")})
            else:
                trg.append({"prop_id": t, "value": None})
        # dash 4: add_context reads the context from this kwarg
        kwargs["callback_context"] = AttributeDict(
            triggered_inputs=trg, updated_props={})

    def run():
        return fn(*args, **kwargs)

    ctx = contextvars.copy_context()
    raw = ctx.run(run)
    return json.loads(raw)["response"]


def _says_finalizing(resp: dict) -> bool:
    """A poll answer that says the ended job's result is being added to the project."""
    from pyorps.gui.services import jobs

    return any(isinstance(value, dict) and jobs.FINALIZING_NOTE in str(value.get("children", ""))
               for value in resp.values())


def poll_job(app, ack_id: str, notices=None, *, max_ticks: int = 10) -> dict:
    """Tick a background job's poll as the page does and return what the page ends up with (round 15).

    The poll is the callback that writes the one-shot store ``ack_id`` (``callbacks/job_poll.py``). A running
    job (or no job) answers on the first tick, and that answer is returned. An ended job is handed over in
    steps: a tick claims it and says "adding the result", the next one runs the finalize step and answers
    with the outputs and a token; the page writes the token into the store, and the tick after that sends it
    back, which switches the poll off. The returned dict is the outputs' answer merged with that last tick's
    (so it holds the result outputs, the notices and the final poll / Stop values).
    """
    from pyorps.gui import ids

    key = f"{ack_id}.data"
    _key, entry = find_callback(app, key)
    poll_id = entry["inputs"][0]["id"]
    triggered = [f"{poll_id}.n_intervals"]
    notices = list(notices or [])
    for tick in range(1, max_ticks + 1):
        resp = invoke(app, key, tick, notices, None, triggered=triggered)
        if ack_id in resp:                   # the outcome: the page keeps it, the next tick confirms it
            if ids.NOTICES in resp:
                notices = resp[ids.NOTICES]["data"]
            confirm = invoke(app, key, tick + 1, notices, resp[ack_id]["data"], triggered=triggered)
            return {**resp, **confirm}
        if not _says_finalizing(resp):
            return resp
    raise AssertionError(f"the poll writing {ack_id!r} never handed the outcome over in {max_ticks} ticks")


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="session", autouse=True)
def _isolate_logbook(tmp_path_factory):
    """Redirect the persistent notice log into a temp dir for the test run."""
    import os

    from pyorps.gui.services import logbook

    os.environ["PYORPS_LOG_DIR"] = str(tmp_path_factory.mktemp("pyorps_logs"))
    logbook.reset()
    yield
    logbook.reset()


@pytest.fixture(scope="session", autouse=True)
def _stop_the_shared_tile_server():
    """Stop localtileserver's shared server when the session ends (round 15).

    The GUI does this at exit (``app._exit_report``); the tests never did. A browser test that left a large
    raster on the map ended with tile renders still running in the server's worker threads (GDAL reads), and
    the pytest process then stayed alive after its summary, with one thread and no CPU: the interpreter had
    shut down and the process exit was blocked. ``stop_server`` lets the requests in flight finish first (and,
    since round 15, never waits for ever for idle keep-alive connections), so no render is cut off at exit.
    """
    yield
    from pyorps.gui.state import TileManager

    TileManager.stop_server()


@pytest.fixture(scope="session")
def raster_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("gui_raster") / "cost.tif"
    create_test_tiff(str(path), width=200, height=200, pattern="gradient",
                     crs=TEST_CRS)
    return str(path)


@pytest.fixture()
def state(tmp_path):
    from pyorps.gui.state import ProjectState

    st = ProjectState(work_dir=tmp_path / "work")
    yield st
    st.shutdown()


@pytest.fixture()
def app(state):
    from pyorps.gui.app import build_app

    return build_app(state)


@pytest.fixture(scope="module")
def finder(raster_path):
    from pyorps import PathFinder

    pf = PathFinder(raster_path, source_coords=SOURCE, target_coords=TARGET,
                    search_space_buffer_m=80, graph_api="cython")
    pf.find_route()
    return pf
