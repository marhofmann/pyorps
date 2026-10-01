"""Phase 0 headless-callback tests: app builds, notifications flow works."""
import json

from pyorps.gui import ids
from pyorps.gui.services.errors import Notice

from conftest import invoke


def test_build_app_has_layout_and_state(app, state):
    assert app._pyorps_state is state
    found = {getattr(comp, "id", None) for comp in app.layout._traverse()}
    for component_id in (ids.MAP, ids.LAYER_HOST, ids.NOTICE_STACK,
                         ids.TABS, ids.NOTICES, ids.UI_STATE,
                         ids.LAYERS_VIEW, ids.DRAW_CONTROL):
        assert component_id in found, f"missing {component_id}"


def test_render_notices_builds_toasts(app):
    notices = [
        Notice(severity="error", title="Can't reach the WFS server",
               meaning="m", impact="i", fix="f", details="d",
               focus_id="tab-data", id="n1").to_dict(),
        Notice(severity="success", title="Loaded", id="n2").to_dict(),
    ]
    resp = invoke(app, ("notice-stack.children",), notices)
    rendered = json.dumps(resp)
    assert "Can't reach the WFS server" in rendered
    assert "Loaded" in rendered
    assert "Go fix" in rendered            # focus_id renders a jump button
    assert "Details" in rendered           # collapsible details present


def test_render_notices_empty(app):
    resp = invoke(app, ("notice-stack.children",), [])
    assert resp["notice-stack"]["children"] == []


def test_dismiss_notice_prunes_store(app):
    notices = [{"id": "n1", "severity": "error", "title": "a"},
               {"id": "n2", "severity": "info", "title": "b"}]
    resp = invoke(
        app, ("notices.data", "n_dismiss"), [1], notices,
        triggered=[{"id": {"type": ids.TYPE_NOTICE, "index": "n1"},
                    "property": "n_dismiss", "value": 1}])
    remaining = resp["notices"]["data"]
    assert [n["id"] for n in remaining] == ["n2"]


def test_go_fix_switches_tab_and_flashes_control(app):
    resp = invoke(
        app, (f"{ids.TABS}.active_tab", "notice-fix"), [1],
        triggered=[{"id": {"type": ids.TYPE_NOTICE_FIX, "tab": ids.TAB_COST,
                           "control": ids.COST_DATASET, "index": "n1"},
                    "property": "n_clicks", "value": 1}])
    assert resp[ids.TABS]["active_tab"] == ids.TAB_COST
    # the flash store carries the control for the clientside scroll+flash
    assert resp[ids.FOCUS_FLASH]["data"]["control"] == ids.COST_DATASET
