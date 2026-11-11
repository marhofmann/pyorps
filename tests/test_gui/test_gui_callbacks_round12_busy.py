"""Round 12 (T5 #11, follow-up) - busy state of the three gated buttons and the
overlay / DEM status line.

``RASTERIZE_BTN``, ``RUN_ROUTING_BTN`` and ``RASTER_COMBINE_BTN`` have their
``disabled`` written by the workflow gating callback, so their callbacks flip a
small busy ``dcc.Store`` through ``running=`` and the gating callback ORs that
flag into the gated state (the gate is recomputed each time, so it always wins).
"""
import pytest
from conftest import invoke

from pyorps.gui import ids
from pyorps.gui.layout import OVERLAY_STATUS
from pyorps.gui.services import catalog

# (button, busy flag, the input of the callback that sets the flag): "Build
# cost raster" asks first (confirm gate), so its action listens on a GO store
BUSY = [(ids.RASTERIZE_BTN, ids.RASTERIZE_BUSY, (ids.GO_RASTERIZE, "data")),
        (ids.RASTER_COMBINE_BTN, ids.RASTER_COMBINE_BUSY,
         (ids.RASTER_COMBINE_BTN, "n_clicks"))]
GATING = ("workflow-guide.children",)
WMS = "wms|https://sgx.geodatenzentrum.de/wms_topplus_open|web"


@pytest.mark.parametrize("button, busy, trigger", BUSY)
def test_gated_button_has_a_busy_mechanism(app, button, busy, trigger):
    specs = [s for s in app._callback_list
             if ((s.get("running") or {}).get("running") or {}).get(
                 f"{busy}.data") is True]
    assert len(specs) == 1, f"{busy}: {len(specs)} callbacks with running="
    assert specs[0]["running"]["runningOff"][f"{busy}.data"] is False
    assert {"id": trigger[0], "property": trigger[1]} in specs[0]["inputs"]
    # the gating callback reads the busy flag and still owns the button
    gating = [s for s in app._callback_list
              if f"{button}.disabled" in str(s["output"])]
    assert len(gating) == 1
    assert {"id": busy, "property": "data"} in gating[0]["inputs"]
    # nobody else writes the button's disabled property through running=
    assert not [s for s in app._callback_list
                if f"{button}.disabled" in str(s.get("running") or {})]


def test_run_button_busy_flag_is_the_or_of_both_runners(app):
    """Two callbacks listen to the Run button (plain and constrained). Each
    flips its OWN flag through ``running=`` (one shared flag would be cleared
    by the runner that ends at once); the flag the gating callback reads is
    their OR."""
    for flag in (ids.RUN_PLAIN_BUSY, ids.RUN_OHL_BUSY):
        specs = [s for s in app._callback_list
                 if ((s.get("running") or {}).get("running") or {}).get(
                     f"{flag}.data") is True]
        assert len(specs) == 1, f"{flag}: {len(specs)} callbacks with running="
        assert specs[0]["running"]["runningOff"][f"{flag}.data"] is False
        assert {"id": ids.RUN_ROUTING_BTN, "property": "n_clicks"} in specs[0]["inputs"]
    # nobody writes the shared flag through running=
    assert not [s for s in app._callback_list
                if f"{ids.RUN_ROUTING_BUSY}.data" in str(s.get("running") or {})]
    # the gating callback still reads the shared flag and owns the button
    gating = [s for s in app._callback_list
              if f"{ids.RUN_ROUTING_BTN}.disabled" in str(s["output"])]
    assert len(gating) == 1
    assert {"id": ids.RUN_ROUTING_BUSY, "property": "data"} in gating[0]["inputs"]
    for plain, constrained, expected in [(False, False, False),
                                         (True, False, True),
                                         (False, True, True),
                                         (True, True, True)]:
        resp = invoke(app, (f"{ids.RUN_ROUTING_BUSY}.data", ids.RUN_PLAIN_BUSY),
                      plain, constrained)
        assert resp[ids.RUN_ROUTING_BUSY]["data"] is expected


def _gate(app, state, busy=(False, False, False), grid_state=None):
    resp = invoke(app, GATING, state.layers_view(), grid_state or {}, None,
                  None, None, *busy,
                  triggered=[f"{ids.LAYERS_VIEW}.data"])
    return (resp[ids.RASTERIZE_BTN]["disabled"],
            resp[ids.RUN_ROUTING_BTN]["disabled"],
            resp[ids.RASTER_COMBINE_BTN]["disabled"])


def test_gating_unchanged_without_busy(app, state, raster_path):
    from pyorps.gui.callbacks.raster import add_raster_layer

    assert _gate(app, state) == (True, True, True)
    add_raster_layer(state, raster_path, [])
    seeded = {"dataset_id": "d1", "feature_keys": ["nutzart"]}
    assert _gate(app, state, grid_state=seeded) == (False, False, True)


def test_busy_disables_and_gated_state_wins_after_the_run(app, state,
                                                          raster_path):
    from pyorps.gui.callbacks.raster import add_raster_layer

    seeded = {"dataset_id": "d1", "feature_keys": ["nutzart"]}
    # gated-disabled (nothing ready): stays disabled while busy and after
    assert _gate(app, state, (True, True, True)) == (True, True, True)
    assert _gate(app, state, (False, False, False)) == (True, True, True)
    # ready buttons: disabled while busy, enabled again afterwards
    add_raster_layer(state, raster_path, [])
    assert _gate(app, state, (True, True, False),
                 seeded) == (True, True, True)
    assert _gate(app, state, (False, False, False),
                 seeded) == (False, False, True)       # combine still gated


def _status(resp):
    return resp[OVERLAY_STATUS]["children"]


def test_overlay_status_text_on_success(app, state):
    resp = invoke(app, ("layers-view.data", ids.OVERLAY_LOAD_BTN), 1, WMS, [],
                  triggered=[f"{ids.OVERLAY_LOAD_BTN}.n_clicks"])
    name = state.layers_of_kind("wms")[0].name
    assert _status(resp) == f"Added {name} to the map"


def test_overlay_status_text_on_failing_load(app, state, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("service down")

    monkeypatch.setattr(catalog, "wms_tile_url", boom)
    resp = invoke(app, ("layers-view.data", ids.OVERLAY_LOAD_BTN), 1, WMS, [],
                  triggered=[f"{ids.OVERLAY_LOAD_BTN}.n_clicks"])
    assert not state.layers_of_kind("wms")
    assert _status(resp).startswith("Nothing was added")
    assert resp[ids.NOTICES]["data"]            # the error box is still raised


def test_overlay_status_without_choice_and_dem_without_area(app, state):
    resp = invoke(app, ("layers-view.data", ids.OVERLAY_LOAD_BTN), 1, None, [],
                  triggered=[f"{ids.OVERLAY_LOAD_BTN}.n_clicks"])
    assert "Pick a service" in _status(resp)
    state.study_area = None
    resp = invoke(app, ("layers-view.data", ids.DEM_LOAD_BTN), 1, None, [],
                  triggered=[f"{ids.DEM_LOAD_BTN}.n_clicks"])
    assert "Draw a study area first" in _status(resp)
