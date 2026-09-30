"""
Shared fixtures + the headless callback harness for pyorps.gui tests.

The harness invokes callbacks exactly as Dash would (through
``app.callback_map``), so a test failure means the *wiring* is broken, not
just the logic — this is the layer that was missing in webviz v1.
"""
from __future__ import annotations

import contextvars
import json

import pytest

pytest.importorskip("dash")
pytest.importorskip("dash_leaflet")
pytest.importorskip("dash_ag_grid")
pytest.importorskip("dash_bootstrap_components")
pytest.importorskip("localtileserver")

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
