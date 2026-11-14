"""Round 15, tidy-up package: nine small clean-ups that the settings and job packages reported.

Items 1 to 3 are covered where the code lives (``test_gui_jobs_round14``, ``test_errors``,
``test_gui_objective_service_round14``); this file holds the rest: the cost-field layer in the dropdowns (4),
the constrained card's intro text (5), the wrapper ids (6), the opened project's path (7), the status text of a
running job after a page reload (8) and the corridor save gate (9).
"""
from __future__ import annotations

from conftest import invoke

from pyorps.gui import ids
from pyorps.gui.callbacks import costmodel as _costmodel  # noqa: F401  (the callbacks under test)

ROUTE_OPTIONS = (f"{ids.ROUTE_RASTER}.options",)


def _plain_raster(state, raster_path, name="cost"):
    return state.add_layer(name, "raster", crs="EPSG:32632", meta={"source_path": raster_path})


def _cost_field_raster(state, raster_path, name="Cost field"):
    return state.add_layer(name, "raster", crs="EPSG:32632",
                           meta={"source_path": raster_path, "cost_field": {"origins": 1, "combine": "sum"}})


# ============================================================ 4: cost-field layers are not cost rasters
def test_a_cost_field_layer_is_not_offered_as_the_raster_to_route_on(app, state, raster_path):
    cost = _plain_raster(state, raster_path)
    field = _cost_field_raster(state, raster_path)
    resp = invoke(app, ROUTE_OPTIONS, state.layers_view())
    assert [o["value"] for o in resp[ids.ROUTE_RASTER]["options"]] == [cost.id]
    assert field.id not in [o["value"] for o in resp[ids.ROUTE_RASTER]["options"]]


def test_a_cost_field_layer_stays_available_where_any_raster_is_fine(app, state, raster_path):
    from pyorps.gui.callbacks import objective as objective_cb
    cost = _plain_raster(state, raster_path)
    field = _cost_field_raster(state, raster_path)
    resp = invoke(app, ROUTE_OPTIONS, state.layers_view())
    # raster algebra: the same callback, the other raster list
    assert [o["value"] for o in resp[ids.RASTER_COMBINE_SELECT]["options"]] == [cost.id, field.id]
    # the objective card's metric-layer source and terrain pickers
    assert [o["value"] for o in objective_cb.raster_options(state)] == [cost.id, field.id]
    source = invoke(app, (f"{ids.OBJ_LAYER_SOURCE}.options",), state.layers_view(), None, None)
    assert field.id in [o["value"] for o in source[ids.OBJ_LAYER_SOURCE]["options"]]
    # the layer list shows it
    assert field.id in [layer.id for layer in state.layers_of_kind("raster")]


def test_is_cost_field_layer_reads_the_meta_flag_only(state, raster_path):
    from pyorps.gui.services import cost_field
    assert cost_field.is_cost_field_layer(_cost_field_raster(state, raster_path)) is True
    assert cost_field.is_cost_field_layer(_plain_raster(state, raster_path)) is False
    assert cost_field.is_cost_field_layer(state.add_layer("no meta", "raster", crs="EPSG:32632")) is False


# ============================================================ shared page helpers
def _children(component) -> list:
    kids = getattr(component, "children", None)
    if kids is None:
        return []
    if isinstance(kids, (list, tuple)):
        return [k for k in kids if hasattr(k, "_traverse")]
    return [kids] if hasattr(kids, "_traverse") else []


def _walk(component):
    yield component
    for child in _children(component):
        yield from _walk(child)


def _find(root, component_id):
    found = [c for c in _walk(root) if getattr(c, "id", None) == component_id]
    assert len(found) == 1, f"{component_id!r} is in the layout {len(found)} times"
    return found[0]


# ============================================================ 5: the constrained card's intro text
def test_the_constrained_intro_says_the_profile_is_a_form_with_raw_yaml_below():
    from dash import html

    from pyorps.gui import layout
    first = layout._ohl_options()[0]
    assert isinstance(first, html.Small)
    text = first.children
    assert "Edit the profile (YAML)" not in text
    assert "profile form" in text and "Raw YAML (advanced)" in text


# ============================================================ 6: wrapper ids live in ids.py
#: the eight wrapper ids that sections used to define themselves: ids.py name -> (value, section alias)
MOVED_WRAPPERS = {
    "OHLF_ANGLE_SCALE_WRAP": ("ohlf-angle-scale-wrap", ("profile_form", "ANGLE_SCALE_WRAP")),
    "OHLF_ANGLE_TABLE_WRAP": ("ohlf-angle-table-wrap", ("profile_form", "ANGLE_TABLE_WRAP")),
    "COR_RUN_WRAP": ("cor-run-wrap", ("corridor", "RUN_WRAP")),
    "CF_RUN_WRAP": ("cf-run-wrap", ("cost_field", "RUN_WRAP")),
    "OBJ_DEM_WRAP": ("obj-dem-box", ("objective", "DEM_WRAP")),
    "OBJ_ADDITIVE_SCALE_WRAP": ("obj-shape-scale-box", ("objective", "ADDITIVE_SCALE_WRAP")),
    "OBJ_MULT_SCALE_WRAP": ("obj-mult-scale-box", ("objective", "MULT_SCALE_WRAP")),
    "OBJ_SIGMOID_WRAP": ("obj-sigmoid-box", ("objective", "SIGMOID_WRAP")),
}


def test_the_wrapper_ids_are_ids_constants_with_their_old_values():
    import importlib
    for name, (value, (module, alias)) in MOVED_WRAPPERS.items():
        assert getattr(ids, name) == value, name
        section = importlib.import_module(f"pyorps.gui.sections.{module}")
        assert getattr(section, alias) == getattr(ids, name), (module, alias)


def test_no_section_defines_a_component_id_as_a_string_literal_any_more():
    import re
    from pathlib import Path

    from pyorps.gui import sections
    for path in Path(sections.__file__).parent.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r'^[A-Z_]*WRAP\s*=\s*"', text, flags=re.M), path.name


def test_the_moved_ids_are_unique_and_swallow_no_other_id():
    values = {k: v for k, v in vars(ids).items() if k.isupper() and isinstance(v, str)}
    assert len(values) == len(set(values.values())), "ids.py holds one value under two names"
    for name, (value, _alias) in MOVED_WRAPPERS.items():
        assert [other for other, v in values.items() if v != value and v in value] == [], name
        assert [other for other, v in values.items() if v != value and value in v] == [], name


def test_every_moved_wrapper_is_in_the_page_exactly_once(state):
    from pyorps.gui.layout import build_layout
    page = build_layout(state)
    page_ids = [c.id for c in _walk(page) if isinstance(getattr(c, "id", None), str)]
    for name, (value, _alias) in MOVED_WRAPPERS.items():
        assert page_ids.count(value) == 1, name


# ============================================================ 8: a reload shows each job's own status wording
def _put_recorder():
    puts: list[tuple] = []
    return puts, lambda component_id, prop, value: puts.append((component_id, prop, value))


def _running_job(kind, monkeypatch, *, stop_requested=False, can_stop=True):
    from pyorps.gui.services import jobs
    monkeypatch.setattr(jobs, "_clock", lambda: 1000.0)               # a fixed elapsed time: the text is comparable
    job = jobs.BackgroundJob(kind=kind, label="Working", can_stop=can_stop)
    job.started = 1000.0
    job.report(2, 5, "stage")
    if stop_requested:
        job.cancel.set()
    return job


def _status_put(state, kind):
    from pyorps.gui import layout
    puts, put = _put_recorder()
    layout._prefill_jobs(state, put)  # noqa: SLF001
    status_id = layout.JOB_CONTROLS[kind][2]
    (text,) = [value for component_id, prop, value in puts if component_id == status_id and prop == "children"]
    return text


def test_the_table_has_one_status_function_per_job_kind():
    from pyorps.gui import layout
    assert set(layout.JOB_STATUS_TEXT) == set(layout.JOB_CONTROLS)


def test_a_reloaded_constrained_job_says_the_stop_acts_after_the_segment_in_flight(state, monkeypatch):
    from pyorps.gui.services import constrained
    job = state.jobs["constrained"] = _running_job("constrained", monkeypatch)
    text = _status_put(state, "constrained")
    assert text == constrained.status_text(job)
    assert "after the segment in flight" in text and "after the current step" not in text
    assert text != job.progress_text()


def test_a_reloaded_constrained_job_with_a_stop_request_says_so(state, monkeypatch):
    state.jobs["constrained"] = _running_job("constrained", monkeypatch, stop_requested=True)
    assert "stop requested" in _status_put(state, "constrained")


def test_a_reloaded_rasterize_job_with_a_stop_request_names_the_stage_boundary(state, monkeypatch):
    from pyorps.gui.callbacks import raster
    job = state.jobs["rasterize"] = _running_job("rasterize", monkeypatch, stop_requested=True)
    text = _status_put(state, "rasterize")
    assert text == raster.rasterize_status_text(job) and "stage boundary" in text
    assert text != job.progress_text()


def test_a_reloaded_cost_field_job_with_a_stop_request_names_the_fixed_point(state, monkeypatch):
    from pyorps.gui.callbacks import cost_field as cost_field_cb
    job = state.jobs["cost_field"] = _running_job("cost_field", monkeypatch, stop_requested=True)
    text = _status_put(state, "cost_field")
    assert text == cost_field_cb.status_text(job) and "fixed point" in text
    assert text != job.progress_text()


def test_a_reloaded_corridor_job_says_it_cannot_be_stopped(state, monkeypatch):
    from pyorps.gui.services import corridor
    job = state.jobs["corridor"] = _running_job("corridor", monkeypatch, can_stop=False)
    text = _status_put(state, "corridor")
    assert text == corridor.status_text(job) and corridor.CANNOT_STOP_NOTE in text
    assert text != job.progress_text()


def test_the_page_prefill_uses_the_function_the_table_names_for_every_kind(state, monkeypatch):
    import importlib

    from pyorps.gui import layout
    for kind in ("rasterize", "constrained", "cost_field", "corridor"):
        state.jobs.clear()
        state.jobs[kind] = _running_job(kind, monkeypatch, stop_requested=(kind != "corridor"),
                                        can_stop=(kind != "corridor"))
        module, name = layout.JOB_STATUS_TEXT[kind]
        function = getattr(importlib.import_module(f"pyorps.gui.{module}"), name)
        assert _status_put(state, kind) == function(state.jobs[kind]), kind


def test_a_status_function_that_fails_falls_back_to_the_generic_line(state, monkeypatch):
    from pyorps.gui import layout
    monkeypatch.setitem(layout.JOB_STATUS_TEXT, "constrained", ("services.constrained", "no_such_function"))
    job = state.jobs["constrained"] = _running_job("constrained", monkeypatch)
    assert _status_put(state, "constrained") == job.progress_text()
