"""Round 14, P5: the "Objective & terrain" card of the Routes tab (gaps 1, 2 and 6).

Headless tests: the harness of ``conftest.py`` drives the callbacks as Dash would, plus layout checks. Components
are addressed as "<id>.<prop>": several ``obj-*`` ids contain each other. The logic lives in
``services/objective`` (tested by ``test_gui_objective_service_round14.py``); here the card must show it: every
label and limit from the specs, the disabled states with their reasons, the status line with the facts and wording
a run refuses with, the mirror into ``state.gui_settings["objective"]``, the restore after Open and New, and no
loop between the controls and the stores.
"""
from __future__ import annotations

import json

import dash_bootstrap_components as dbc
import numpy as np
import pytest
import rasterio
from conftest import SOURCE, TARGET, TEST_CRS, find_callback, invoke
from dash import dcc, html
from dash.exceptions import PreventUpdate
from rasterio.transform import from_origin

from pyorps import PathFinder
from pyorps.gui import ids, layout
from pyorps.gui.callbacks import objective as cb
from pyorps.gui.layout import build_layout
from pyorps.gui.sections import objective as form_section
from pyorps.gui.services import objective as service
from pyorps.gui.services import project_io, routing
from pyorps.gui.state import ProjectState

GRID = ids.OBJ_LAYERS_GRID
OPTIONS_DEM = (f"{ids.OBJ_DEM_LAYER}.options",)
OPTIONS_SOURCE = (f"{ids.OBJ_LAYER_SOURCE}.options",)
PRESET = (f"{ids.OBJ_W_COST}.value", ids.OBJ_PRESET_CLASSIC_BTN)
ADD = (f"{GRID}.rowData", ids.OBJ_LAYER_ADD_BTN)
DELETE = (f"{GRID}.rowData", ids.OBJ_LAYER_DEL_BTN)
REFRESH = (f"{GRID}.rowData", ids.LAYERS_VIEW)
MIRROR = (f"{ids.OBJ_SETTINGS}.data", "cellValueChanged")
RESTORE = (f"{ids.OBJ_SETTINGS}.data", ids.PROJECT_STATUS)
STATUS = (f"{ids.OBJ_STATUS}.children",)
STATES = (f"{ids.OBJ_SLOPE_WRAP}.disabled",)
FIELDS = (f"{form_section.ADDITIVE_SCALE_WRAP}.style",)
RESET_SCALE = (f"{ids.OBJ_GRAD_MULT_SCALE}.value", ids.OBJ_GRAD_MULTIPLIER)
CLEAR_SLOPE = (f"{ids.OBJ_W_GRADIENT}.value", ids.OBJ_DEM_LAYER)
RUN_WIRING = (f"{ids.ROUTING_STATUS}.children", ids.RUN_ROUTING_BTN)

DEFAULT_PROPS = form_section.form_props(None)
DEFAULT_GRADIENT = dict(zip(form_section.GRADIENT_IDS, DEFAULT_PROPS[3:13]))
ORIGIN_X, ORIGIN_Y = 500000.0, 5600000.0
SIZE = 200


# ------------------------------------------------------------------ helpers
def _components(page):
    return list(page._traverse())  # noqa: SLF001


def _by_id(page, component_id):
    found = [c for c in _components(page) if getattr(c, "id", None) == component_id]
    assert len(found) == 1, f"{component_id}: {len(found)} components"
    return found[0]


def _text(component) -> str:
    """All plain text inside a component."""
    pieces = []
    for part in component._traverse():  # noqa: SLF001
        child = getattr(part, "children", None)
        if isinstance(child, str):
            pieces.append(child)
    return " ".join(pieces)


def _title(details) -> str:
    return details.children[0].children[0].children


def _write_tif(path, array, crs=TEST_CRS):
    height, width = array.shape
    with rasterio.open(path, "w", driver="GTiff", height=height, width=width, count=1, dtype=str(array.dtype),
                       crs=crs, transform=from_origin(ORIGIN_X, ORIGIN_Y, 1, 1)) as dst:
        dst.write(array, 1)
    return str(path)


@pytest.fixture(scope="module")
def grid_files(tmp_path_factory):
    """A DEM with a hill half way between SOURCE and TARGET and a stripe layer across the diagonal."""
    folder = tmp_path_factory.mktemp("objective_ui_grid")
    rows, cols = np.mgrid[0:SIZE, 0:SIZE]
    x, y = ORIGIN_X + cols + 0.5, ORIGIN_Y - rows - 0.5
    dem = (60.0 * np.exp(-(((x - 500100.0) ** 2 + (y - 5599900.0) ** 2) / (2 * 12.0 ** 2)))).astype("float32")
    stripe = np.zeros((SIZE, SIZE), dtype="float32")
    stripe[:, 90:110] = 50.0
    return {"dem": _write_tif(folder / "dem.tif", dem), "stripe": _write_tif(folder / "stripe.tif", stripe)}


@pytest.fixture()
def layers(state, raster_path, grid_files):
    """The cost raster, a DEM and a stripe layer as loaded raster layers."""
    def add(name, path):
        return state.add_layer(name, "raster", crs=TEST_CRS, meta={"source_path": path})

    return {"cost": add("cost", raster_path), "dem": add("DGM", grid_files["dem"]),
            "stripe": add("landscape", grid_files["stripe"])}


@pytest.fixture()
def page(state):
    return build_layout(state)


def _row(name, source, weight=1.0, hard_max=None, hard_min=None, *, row_id=None, label=None):
    return {"id": row_id or f"id-{name}", "name": name, "source": source,
            "source_label": label or service.source_label(source, {}), "weight": weight,
            "hard_max": hard_max, "hard_min": hard_min}


def _mirror_args(*, weights=(1.0, 0.0, 0.0), gradient=None, dem=None, precision="uint16", rows=None, edit=None,
                 virtual=None):
    """The positional arguments of ``remember_objective_settings``."""
    values = {**DEFAULT_GRADIENT, **(gradient or {})}
    return (*weights, *[values[key] for key in form_section.GRADIENT_IDS], dem, precision, rows or [], edit, virtual)


def _mirror(app, **kwargs):
    triggered = kwargs.pop("triggered", None)
    return invoke(app, MIRROR, *_mirror_args(**kwargs), triggered=triggered)


def _notices(resp) -> list[dict]:
    return resp.get(ids.NOTICES, {}).get("data", [])


def _status(app, state, settings, graph_api="cython", algorithm="dijkstra", raster_layer_id=None):
    resp = invoke(app, STATUS, settings, graph_api, algorithm, state.layers_view(), raster_layer_id)
    return resp[ids.OBJ_STATUS]["children"], resp[ids.OBJ_STATUS]["className"]


def _states(app, settings, graph_api="cython", algorithm="dijkstra"):
    resp = invoke(app, STATES, settings, graph_api, algorithm)
    return {
        "weight_disabled": resp[ids.OBJ_W_GRADIENT_WRAP]["disabled"],
        "weight_title": resp[ids.OBJ_W_GRADIENT_WRAP]["title"],
        "slope_disabled": resp[ids.OBJ_SLOPE_WRAP]["disabled"],
        "slope_title": resp[ids.OBJ_SLOPE_WRAP]["title"],
        "dem_title": resp[form_section.DEM_WRAP]["title"],
        "precision_title": resp[ids.OBJ_PRECISION_WRAP]["title"],
        "precision_options": resp[ids.OBJ_PRECISION]["options"],
    }


# ================================================================== the layout
def test_the_card_is_a_closed_collapsible_card_between_algorithm_and_constrained_routing(page):
    card = _by_id(page, ids.OBJ_CARD)
    assert isinstance(card, html.Details) and card.open is False
    assert _title(card) == "Objective & terrain"
    parent = next(c for c in _components(page) if isinstance(getattr(c, "children", None), list)
                  and card in c.children)
    titles = [_title(c) if isinstance(c, html.Details) else None for c in parent.children]
    position = parent.children.index(card)
    assert titles[position - 1] == "Algorithm & search" and titles[position + 1] == "Constrained routing"


def test_every_obj_id_of_the_contract_is_in_the_page_exactly_once(page):
    wanted = [value for name, value in vars(ids).items() if name.startswith("OBJ_") and isinstance(value, str)]
    assert len(wanted) >= 30
    for component_id in (*wanted, form_section.DEM_WRAP, form_section.ADDITIVE_SCALE_WRAP,
                         form_section.MULT_SCALE_WRAP, form_section.SIGMOID_WRAP):
        _by_id(page, component_id)


def test_the_wrapper_ids_the_section_adds_do_not_contain_a_control_id():
    wrappers = (form_section.DEM_WRAP, form_section.ADDITIVE_SCALE_WRAP, form_section.MULT_SCALE_WRAP,
                form_section.SIGMOID_WRAP)
    old = [value for name, value in vars(ids).items() if name.isupper() and isinstance(value, str)]
    for wrapper in wrappers:
        others = [value for value in old if value != wrapper]          # the wrapper itself is in ids.py now
        assert [value for value in others if value in wrapper] == [], wrapper
        assert wrapper.startswith("obj-")


def test_every_control_sits_inside_the_card(page):
    card = _by_id(page, ids.OBJ_CARD)
    inside = {getattr(c, "id", None) for c in card._traverse()}  # noqa: SLF001
    wanted = {*form_section.WEIGHT_IDS.values(), *form_section.GRADIENT_IDS.values(), *form_section.LAYER_FORM_IDS,
              ids.OBJ_DEM_LAYER, ids.OBJ_PRECISION, ids.OBJ_LAYERS_GRID, ids.OBJ_STATUS, ids.OBJ_LAYER_ADD_BTN,
              ids.OBJ_LAYER_DEL_BTN, *form_section.PRESET_BUTTONS}
    assert wanted <= inside
    assert ids.OBJ_SETTINGS not in inside               # the store is a direct child of the page container


@pytest.mark.parametrize("spec_key", sorted(form_section.NUMBER_IDS))
def test_each_number_input_comes_from_its_spec(page, spec_key):
    spec = service.FIELD_SPECS[spec_key]
    control = _by_id(page, form_section.NUMBER_IDS[spec_key])
    label = next(c for c in _components(page) if isinstance(c, dbc.Label)
                 and getattr(c, "html_for", None) == control.id)
    assert label.children == service.label_with_unit(spec)
    if spec.unit and spec.unit != "-":
        assert f"({spec.unit})" in label.children
    wrapper = next(c for c in _components(page) if isinstance(c, html.Div)
                   and isinstance(c.children, list) and label in c.children and control in c.children)
    if spec_key == "weights.gradient":
        # its fieldset carries the hover text (the reason while off, this help once a DEM is chosen), so that the
        # reason is not shadowed by a title of the field itself
        assert getattr(wrapper, "title", None) is None
        title = form_section.control_states({"dem_layer_id": "x"}, "cython", "dijkstra")["weight_title"]
    else:
        title = wrapper.title
    assert spec.help in title and service.describe_range(spec) in title
    assert isinstance(control, dbc.Input) and control.type == "number" and control.debounce is True
    minimum = (service.FIELD_SPECS["gradient.bin_width_pct"].minimum if spec_key == "gradient.s_max_pct"
               else spec.minimum)
    assert control.min == minimum and control.max == spec.maximum and control.step == "any"
    assert control.value == spec.default


def test_a_fresh_card_shows_the_spec_defaults(page):
    defaults = service.default_settings()
    for key, component_id in form_section.WEIGHT_IDS.items():
        assert _by_id(page, component_id).value == defaults["weights"][key]
    for key, component_id in form_section.GRADIENT_IDS.items():
        expected = defaults["gradient"][key]
        assert _by_id(page, component_id).value == ((expected or "none") if key in form_section.CHOICE_KEYS
                                                    else expected)
    assert _by_id(page, ids.OBJ_DEM_LAYER).value is None
    assert _by_id(page, ids.OBJ_PRECISION).value == "uint16"
    assert _by_id(page, ids.OBJ_LAYER_WEIGHT).value == service.FIELD_SPECS["layers.weight"].default
    assert _by_id(page, ids.OBJ_STATUS).children == "Classic cost routing (no objective)."


def test_the_two_lists_offer_the_service_choices_and_start_on_none(page):
    shape = _by_id(page, ids.OBJ_GRAD_ADDITIVE)
    model = _by_id(page, ids.OBJ_GRAD_MULTIPLIER)
    assert isinstance(shape, dbc.Select) and isinstance(model, dbc.Select)
    assert [o["value"] for o in shape.options] == ["none", *service.ADDITIVE_SHAPES]
    assert [o["value"] for o in model.options] == ["none", *service.MULTIPLIER_MODELS]
    assert shape.value == "none" and model.value == "none"


def test_the_three_preset_buttons_are_built_from_the_service_presets(page):
    assert tuple(form_section.PRESET_BUTTONS.values()) == tuple(service.PRESETS)
    labels = {"classic": "Cost only", "shortest": "Shortest", "gentlest": "Gentlest (needs DEM)"}
    for button_id, name in form_section.PRESET_BUTTONS.items():
        button = _by_id(page, button_id)
        assert button.children == labels[name] and button.title


def test_the_precision_choice_lists_the_service_precisions_and_starts_with_32_bit_off(page):
    radios = _by_id(page, ids.OBJ_PRECISION)
    assert [o["value"] for o in radios.options] == list(service.PRECISIONS)
    for option in radios.options:
        assert service.PRECISION_LABELS[option["value"]] in option["label"]
    assert [o["disabled"] for o in radios.options] == [False, True]
    wrapper = _by_id(page, ids.OBJ_PRECISION_WRAP)
    assert wrapper.title.startswith("Only with an objective")


def test_the_slope_weight_and_the_slope_block_start_disabled_with_the_reason(page):
    for wrapper_id, control_ids in ((ids.OBJ_W_GRADIENT_WRAP, [ids.OBJ_W_GRADIENT]),
                                    (ids.OBJ_SLOPE_WRAP, list(form_section.GRADIENT_IDS.values()))):
        wrapper = _by_id(page, wrapper_id)
        assert isinstance(wrapper, html.Fieldset) and wrapper.disabled is True
        assert wrapper.title == service.SLOPE_NEEDS_DEM
        inside = {getattr(c, "id", None) for c in wrapper._traverse()}  # noqa: SLF001
        assert set(control_ids) <= inside
    accordion = next(c for c in _by_id(page, ids.OBJ_SLOPE_WRAP)._traverse()  # noqa: SLF001
                     if isinstance(c, dbc.Accordion))
    assert accordion.start_collapsed is True
    assert accordion.children[0].title == "Slope response"


def test_the_conditional_slope_fields_start_hidden_and_the_terrain_model_is_never_disabled(page):
    for wrapper_id in (form_section.ADDITIVE_SCALE_WRAP, form_section.MULT_SCALE_WRAP, form_section.SIGMOID_WRAP):
        assert _by_id(page, wrapper_id).style == {"display": "none"}
    dem = _by_id(page, ids.OBJ_DEM_LAYER)
    assert isinstance(dem, dcc.Dropdown) and dem.clearable is True and not getattr(dem, "disabled", False)
    source = _by_id(page, ids.OBJ_LAYER_SOURCE)
    assert isinstance(source, dcc.Dropdown) and source.value is None


def test_the_layers_table_is_a_section_grid_with_editable_weight_and_bounds(page):
    grid = _by_id(page, GRID)
    assert [c["field"] for c in grid.columnDefs] == ["name", "source_label", "weight", "hard_max", "hard_min"]
    assert [c["editable"] for c in grid.columnDefs] == [False, False, True, True, True]
    assert [c["cellDataType"] for c in grid.columnDefs] == ["text", "text", "number", "number", "number"]
    assert grid.rowData == [] and grid.getRowId == "params.data.id"
    # fitted to its width whenever that changes, by assets/grid-fit.js (round 15: dash-ag-grid's columnSize fitted at
    # once, in the closed card with no width, and AG Grid warned #29)
    assert grid.className == layout.FIT_COLUMNS_RESPONSIVE and getattr(grid, "columnSize", None) is None
    assert grid.dashGridOptions["rowSelection"]["mode"] == "multiRow"
    text = json.dumps([grid.columnDefs, grid.defaultColDef])
    assert "rowGroup" not in text and "cellStyle" not in text      # the combination that freezes AG Grid
    for column, key in zip(grid.columnDefs[2:], ("layers.weight", "layers.hard_max", "layers.hard_min")):
        assert service.FIELD_SPECS[key].help in column["headerTooltip"]
    assert _by_id(page, ids.OBJ_LAYER_ADD_BTN).children == "+ Add layer"
    assert _by_id(page, ids.OBJ_LAYER_DEL_BTN).children == "− Remove selected"


def test_the_layer_name_field_states_its_length(page):
    name = _by_id(page, ids.OBJ_LAYER_NAME)
    assert name.type == "text" and name.maxLength == 32


def test_no_callback_output_is_a_duplicate(app):
    """Two callbacks with one (output, inputs) pair make Dash's renderer fail: a clientside helper must not
    reuse the pair of the notices' scroll-and-flash handler."""
    outputs = [entry["output"] for entry in app._callback_list]  # noqa: SLF001
    assert len(outputs) == len(set(outputs))


# ================================================================== options
def test_the_dem_options_are_the_loaded_rasters(app, state, layers):
    resp = invoke(app, OPTIONS_DEM, state.layers_view(), None)
    assert resp[ids.OBJ_DEM_LAYER]["options"] == [{"label": layer.name, "value": layer.id}
                                                  for layer in layers.values()]


def test_a_vector_layer_is_never_offered(app, state, layers):
    state.add_layer("Square", "vector", geojson={"type": "FeatureCollection", "features": []})
    resp = invoke(app, OPTIONS_DEM, state.layers_view(), None)
    assert "Square" not in [o["label"] for o in resp[ids.OBJ_DEM_LAYER]["options"]]


def test_a_chosen_dem_that_is_gone_stays_listed_as_not_loaded(app, state, layers):
    gone = layers["dem"].id
    state.remove_layer(gone)
    resp = invoke(app, OPTIONS_DEM, state.layers_view(), gone)
    assert {"label": cb.NOT_LOADED_LABEL, "value": gone} in resp[ids.OBJ_DEM_LAYER]["options"]
    resp = invoke(app, OPTIONS_DEM, state.layers_view(), None)
    assert cb.NOT_LOADED_LABEL not in [o["label"] for o in resp[ids.OBJ_DEM_LAYER]["options"]]


def test_the_source_options_offer_the_slope_from_the_dem_only_with_a_dem(app, state, layers):
    without = invoke(app, OPTIONS_SOURCE, state.layers_view(), None, None)
    with_dem = invoke(app, OPTIONS_SOURCE, state.layers_view(), layers["dem"].id, None)
    derive = {"label": service.DERIVE_SLOPE_LABEL, "value": service.SOURCE_DERIVE_SLOPE}
    assert derive not in without[ids.OBJ_LAYER_SOURCE]["options"]
    assert derive in with_dem[ids.OBJ_LAYER_SOURCE]["options"]
    assert [o["value"] for o in without[ids.OBJ_LAYER_SOURCE]["options"]] == [layer.id for layer in layers.values()]


def test_a_source_pick_that_is_no_longer_offered_is_cleared(app, state, layers):
    resp = invoke(app, OPTIONS_SOURCE, state.layers_view(), None, service.SOURCE_DERIVE_SLOPE)
    assert resp[ids.OBJ_LAYER_SOURCE]["value"] is None
    resp = invoke(app, OPTIONS_SOURCE, state.layers_view(), layers["dem"].id, service.SOURCE_DERIVE_SLOPE)
    assert "value" not in resp[ids.OBJ_LAYER_SOURCE]
    resp = invoke(app, OPTIONS_SOURCE, state.layers_view(), None, layers["stripe"].id)
    assert "value" not in resp[ids.OBJ_LAYER_SOURCE]


# ================================================================== presets
def _preset(app, button, settings=None, **clicks):
    counts = {"classic": 0, "shortest": 0, "gentlest": 0, **clicks}
    return invoke(app, PRESET, counts["classic"], counts["shortest"], counts["gentlest"], settings,
                  triggered=[f"{button}.n_clicks"])


@pytest.mark.parametrize("button,name", list(form_section.PRESET_BUTTONS.items()))
def test_a_preset_button_writes_the_weights_of_apply_preset(app, button, name):
    settings = {"weights": {"cost": 4.0, "length": 2.0, "gradient": 3.0},
                "layers": [service.layer_row("landscape", "layer-1", 0.5)], "dem_layer_id": "layer-2",
                "gradient": {"multiplier": "exponential"}}
    resp = _preset(app, button, settings, **{name: 1})
    expected = service.apply_preset(settings, name)["weights"]
    assert (resp[ids.OBJ_W_COST]["value"], resp[ids.OBJ_W_LENGTH]["value"],
            resp[ids.OBJ_W_GRADIENT]["value"]) == (expected["cost"], expected["length"], expected["gradient"])
    assert set(resp) == {ids.OBJ_W_COST, ids.OBJ_W_LENGTH, ids.OBJ_W_GRADIENT}      # layers, DEM, slope untouched


def test_the_presets_write_the_documented_numbers(app):
    shortest = _preset(app, ids.OBJ_PRESET_SHORTEST_BTN, {}, shortest=1)
    assert [shortest[i]["value"] for i in (ids.OBJ_W_COST, ids.OBJ_W_LENGTH, ids.OBJ_W_GRADIENT)] == [0.0, 1.0, 0.0]
    gentlest = _preset(app, ids.OBJ_PRESET_GENTLEST_BTN, {}, gentlest=1)
    assert [gentlest[i]["value"] for i in (ids.OBJ_W_COST, ids.OBJ_W_LENGTH, ids.OBJ_W_GRADIENT)] == [0.0, 0.01, 1.0]


def test_a_preset_with_no_click_or_no_known_button_changes_nothing(app):
    with pytest.raises(PreventUpdate):
        _preset(app, ids.OBJ_PRESET_CLASSIC_BTN, {}, classic=0)
    with pytest.raises(PreventUpdate):
        invoke(app, PRESET, 1, 1, 1, {}, triggered=[f"{ids.OBJ_W_COST}.value"])


def test_a_preset_works_on_an_empty_store(app):
    resp = _preset(app, ids.OBJ_PRESET_CLASSIC_BTN, None, classic=1)
    assert resp[ids.OBJ_W_COST]["value"] == 1.0


# ================================================================== the layers table
def _add(app, state, *, name="landscape", source=None, weight=1.0, dem=None, rows=None, virtual=None, notices=None):
    return invoke(app, ADD, 1, name, source, weight, dem, rows or [], virtual, notices or [])


def test_a_layer_is_added_after_its_name_is_checked_and_the_name_field_is_cleared(app, state, layers):
    resp = _add(app, state, source=layers["stripe"].id, weight=0.5)
    (row,) = resp[GRID]["rowData"]
    assert row["name"] == "landscape" and row["source"] == layers["stripe"].id and row["weight"] == 0.5
    assert row["hard_max"] is None and row["hard_min"] is None
    assert row["source_label"] == "landscape" and row["id"]          # the label of the raster layer
    assert resp[ids.OBJ_LAYER_NAME]["value"] == ""
    assert ids.NOTICES not in resp


def test_a_blank_weight_takes_the_default_of_a_new_layer(app, state, layers):
    resp = _add(app, state, source=layers["stripe"].id, weight=None)
    assert resp[GRID]["rowData"][0]["weight"] == 1.0


def test_a_new_layer_is_appended_to_the_rows_the_grid_holds_now(app, state, layers):
    held = [_row("first", layers["stripe"].id, 2.0)]
    edited = [_row("first", layers["stripe"].id, 7.0)]                 # the user edited a cell, rowData is stale
    resp = _add(app, state, name="second", source=layers["dem"].id, rows=held, virtual=edited)
    rows = resp[GRID]["rowData"]
    assert [r["name"] for r in rows] == ["first", "second"] and rows[0]["weight"] == 7.0
    assert rows[0]["id"] == "id-first"                                 # the old row keeps its id


def test_the_slope_from_the_dem_is_a_valid_source_once_a_dem_is_chosen(app, state, layers):
    resp = _add(app, state, name="steep", source=service.SOURCE_DERIVE_SLOPE, dem=layers["dem"].id)
    row = resp[GRID]["rowData"][0]
    assert row["source"] == service.SOURCE_DERIVE_SLOPE and row["source_label"] == service.DERIVE_SLOPE_LABEL


@pytest.mark.parametrize("name,fragment", [
    ("", "must not be empty"), ("   ", "must not be empty"), ("1abc", "not allowed"), ("has space", "not allowed"),
    ("a" * 33, "at most 32"), ("cost", "reserved"), ("gradient", "reserved"), ("tower", "reserved")])
def test_a_bad_layer_name_is_a_notice_that_focuses_the_name_field(app, state, layers, name, fragment):
    resp = _add(app, state, name=name, source=layers["stripe"].id)
    assert GRID not in resp and ids.OBJ_LAYER_NAME not in resp          # nothing added, the field keeps the text
    notice = _notices(resp)[-1]
    assert notice["focus_control"] == ids.OBJ_LAYER_NAME and notice["focus_id"] == ids.TAB_ROUTES
    assert notice["title"].startswith("Layer name") and fragment in notice["meaning"]
    assert notice["severity"] == "warning" and "No layer was added" in notice["impact"] and notice["fix"]


def test_a_name_that_another_layer_uses_is_refused(app, state, layers):
    rows = [_row("landscape", layers["stripe"].id)]
    resp = _add(app, state, name="landscape", source=layers["dem"].id, rows=rows)
    notice = _notices(resp)[-1]
    assert "another layer already uses" in notice["meaning"] and GRID not in resp


def test_an_empty_source_is_a_notice_that_focuses_the_source_field(app, state, layers):
    resp = _add(app, state, source=None)
    notice = _notices(resp)[-1]
    assert notice["focus_control"] == ids.OBJ_LAYER_SOURCE and "Layer source" in notice["title"]
    assert "is empty" in notice["meaning"] and GRID not in resp


def test_a_source_that_is_not_loaded_is_a_notice(app, state, layers):
    resp = _add(app, state, source="no-such-layer")
    notice = _notices(resp)[-1]
    assert notice["focus_control"] == ids.OBJ_LAYER_SOURCE and "not loaded any more" in notice["meaning"]


def test_the_slope_from_the_dem_without_a_dem_points_at_the_dem_field(app, state, layers):
    resp = _add(app, state, name="steep", source=service.SOURCE_DERIVE_SLOPE, dem=None)
    notice = _notices(resp)[-1]
    assert notice["focus_control"] == ids.OBJ_DEM_LAYER and "needs a terrain model" in notice["meaning"]
    assert GRID not in resp


@pytest.mark.parametrize("weight", [-1, 2e6, float("inf")])
def test_a_weight_outside_its_range_is_a_notice_that_focuses_the_weight_field(app, state, layers, weight):
    resp = _add(app, state, source=layers["stripe"].id, weight=weight)
    notice = _notices(resp)[-1]
    assert notice["focus_control"] == ids.OBJ_LAYER_WEIGHT and "Layer weight" in notice["title"]
    assert "from 0 to 1,000,000" in notice["meaning"] or "finite" in notice["meaning"]
    assert GRID not in resp


def test_a_stale_row_in_the_table_does_not_block_a_new_layer(app, state, layers):
    stale = [_row("old", "layer-that-was-removed", 1.0)]
    resp = _add(app, state, name="fresh", source=layers["stripe"].id, rows=stale)
    assert [r["name"] for r in resp[GRID]["rowData"]] == ["old", "fresh"]


def test_notices_are_appended_to_the_stored_ones_and_capped(app, state, layers):
    old = [{"id": f"n{i}", "severity": "info", "title": "x"} for i in range(3)]
    resp = _add(app, state, name="", source=None, notices=old)
    assert [n["id"] for n in _notices(resp)[:3]] == ["n0", "n1", "n2"] and len(_notices(resp)) == 4


def test_the_add_button_without_a_click_changes_nothing(app, state, layers):
    with pytest.raises(PreventUpdate):
        invoke(app, ADD, None, "x", layers["stripe"].id, 1.0, None, [], None, [])


def test_remove_selected_deletes_the_selected_rows_and_keeps_edits(app, state, layers):
    held = [_row("a", layers["stripe"].id, 1.0), _row("b", layers["dem"].id, 2.0), _row("c", layers["dem"].id, 3.0)]
    live = [dict(held[0], weight=9.0), held[1], held[2]]               # a was edited since rowData was set
    resp = invoke(app, DELETE, 1, held, live, [held[1]])
    assert [(r["name"], r["weight"]) for r in resp[GRID]["rowData"]] == [("a", 9.0), ("c", 3.0)]


def test_remove_selected_with_nothing_selected_changes_nothing(app, state, layers):
    held = [_row("a", layers["stripe"].id)]
    with pytest.raises(PreventUpdate):
        invoke(app, DELETE, 1, held, held, [])
    with pytest.raises(PreventUpdate):
        invoke(app, DELETE, None, held, held, [held[0]])


def test_two_rows_with_one_name_are_removed_one_at_a_time(app, state, layers):
    held = [_row("a", layers["stripe"].id, row_id="r1"), _row("a", layers["stripe"].id, row_id="r2")]
    resp = invoke(app, DELETE, 1, held, held, [held[0]])
    assert [r["id"] for r in resp[GRID]["rowData"]] == ["r2"]


def test_a_layer_that_was_removed_shows_as_not_loaded_in_the_table_and_a_rename_follows(app, state, layers):
    rows = [_row("landscape", layers["stripe"].id, label="landscape"),
            _row("steep", service.SOURCE_DERIVE_SLOPE, label=service.DERIVE_SLOPE_LABEL)]
    state.remove_layer(layers["stripe"].id)
    resp = invoke(app, REFRESH, state.layers_view(), rows, None)
    assert [r["source_label"] for r in resp[GRID]["rowData"]] == ["(layer no longer loaded)",
                                                                 service.DERIVE_SLOPE_LABEL]
    assert [r["id"] for r in resp[GRID]["rowData"]] == ["id-landscape", "id-steep"]
    renamed = [_row("dem", layers["dem"].id, label="old name")]
    layers["dem"].name = "DGM 1 m"
    resp = invoke(app, REFRESH, state.layers_view(), renamed, None)
    assert resp[GRID]["rowData"][0]["source_label"] == "DGM 1 m"


def test_refreshing_labels_that_are_current_changes_nothing(app, state, layers):
    rows = [_row("landscape", layers["stripe"].id, label="landscape")]
    with pytest.raises(PreventUpdate):
        invoke(app, REFRESH, state.layers_view(), rows, None)
    with pytest.raises(PreventUpdate):
        invoke(app, REFRESH, state.layers_view(), [], None)


# ================================================================== the mirror
def test_the_settings_are_mirrored_into_the_state_and_the_store(app, state, layers):
    rows = [_row("landscape", layers["stripe"].id, 0.5, 30.0, 1.0)]
    resp = _mirror(app, weights=(1.0, 0.2, 0.5), dem=layers["dem"].id, precision="float32", rows=rows,
                   gradient={"max_gradient_pct": 30.0, "additive": "exponential", "additive_scale": 3.0,
                             "multiplier": "sigmoid", "sig_center_deg": 12.0})
    saved = state.gui_settings["objective"]
    assert saved["weights"] == {"cost": 1.0, "length": 0.2, "gradient": 0.5}
    assert saved["layers"] == [{"name": "landscape", "source": layers["stripe"].id, "weight": 0.5,
                                "hard_max": 30.0, "hard_min": 1.0}]
    assert saved["dem_layer_id"] == layers["dem"].id and saved["weight_precision"] == "float32"
    assert saved["gradient"]["additive"] == "exponential" and saved["gradient"]["multiplier"] == "sigmoid"
    assert saved["gradient"]["max_gradient_pct"] == 30.0 and saved["gradient"]["sig_center_deg"] == 12.0
    assert resp[ids.OBJ_SETTINGS]["data"] == saved
    assert saved == service.normalise_objective_settings(saved)        # written through the normaliser
    json.dumps(saved)


def test_the_mirror_of_an_untouched_card_is_the_classic_default(app, state):
    resp = _mirror(app)
    assert state.gui_settings["objective"] == service.default_settings() == resp[ids.OBJ_SETTINGS]["data"]
    assert service.is_classic(state.gui_settings["objective"])


def test_blank_fields_take_their_defaults_and_none_lists_are_none(app, state):
    _mirror(app, weights=(None, None, None), gradient={"additive_scale": None, "multiplier": "none",
                                                       "additive": "none", "sig_steepness": None})
    saved = state.gui_settings["objective"]
    assert saved["weights"] == {"cost": 1.0, "length": 0.0, "gradient": 0.0}
    assert saved["gradient"]["additive"] is None and saved["gradient"]["multiplier"] is None
    assert saved["gradient"]["additive_scale"] == 2.0 and saved["gradient"]["sig_steepness"] == 0.2


def test_an_out_of_range_or_unusable_value_is_kept_not_clamped(app, state):
    _mirror(app, weights=(-5, "abc", 0), gradient={"max_gradient_pct": 5000})
    saved = state.gui_settings["objective"]
    assert saved["weights"]["cost"] == -5.0 and saved["weights"]["length"] == "abc"
    assert saved["gradient"]["max_gradient_pct"] == 5000.0


def test_the_mirror_changes_the_dict_in_place(app, state):
    held = state.gui_settings
    _mirror(app)
    assert state.gui_settings is held and "objective" in held


def test_a_cell_edit_is_read_from_the_rows_the_grid_holds_now(app, state, layers):
    stale = [_row("landscape", layers["stripe"].id, 1.0)]
    live = [_row("landscape", layers["stripe"].id, 4.0, hard_max=20.0)]
    _mirror(app, rows=stale, virtual=live, edit={"rowId": "x"}, triggered=[f"{GRID}.cellValueChanged"])
    assert state.gui_settings["objective"]["layers"][0]["weight"] == 4.0
    assert state.gui_settings["objective"]["layers"][0]["hard_max"] == 20.0


def test_a_fill_of_the_table_is_read_from_its_row_data_not_from_older_virtual_rows(app, state, layers):
    new = [_row("landscape", layers["stripe"].id, 2.0)]
    old = [_row("other", layers["dem"].id, 9.0)]
    _mirror(app, rows=new, virtual=old, triggered=[f"{GRID}.rowData"])
    assert [r["name"] for r in state.gui_settings["objective"]["layers"]] == ["landscape"]


def test_the_grid_columns_that_are_display_only_never_reach_the_settings(app, state, layers):
    _mirror(app, rows=[_row("landscape", layers["stripe"].id, 1.0)])
    (saved,) = state.gui_settings["objective"]["layers"]
    assert set(saved) == {"name", "source", "weight", "hard_max", "hard_min"}


# ================================================================== the status line
def test_the_status_of_classic_settings(app, state):
    text, css = _status(app, state, {})
    assert text == "Classic cost routing (no objective)." and css == form_section.STATUS_CLASS_OK


def test_the_status_summarises_a_valid_objective(app, state, layers):
    settings = {"weights": {"cost": 1, "length": 0.2}, "dem_layer_id": layers["dem"].id,
                "layers": [service.layer_row("landscape", layers["stripe"].id, 0.5)],
                "gradient": {"max_gradient_pct": 30}}
    text, css = _status(app, state, settings)
    assert text == ("Objective: cost 1 + length 0.2 + landscape 0.5; DEM 'DGM'; slope limit 30 %; "
                    "cython runs Dijkstra only.")
    assert css == form_section.STATUS_CLASS_OK


def test_a_dem_with_delta_stepping_is_named_before_the_run_in_the_words_the_run_uses(app, state, layers):
    settings = {"dem_layer_id": layers["dem"].id}
    text, css = _status(app, state, settings, algorithm="delta-stepping")
    problem = service.problems(settings, graph_api="cython", algorithm="delta-stepping",
                               raster_ids={layer.id for layer in layers.values()})[0]
    assert text == f"{problem.label} {problem.message}." and css == form_section.STATUS_CLASS_PROBLEM
    assert "delta-stepping" in text and "choose Dijkstra" in text and text.startswith("Algorithm is")


def test_the_status_text_is_the_meaning_of_the_notice_a_run_stops_with(app, state, layers, monkeypatch):
    monkeypatch.setattr(routing, "run_routing", lambda *_a, **_k: (None, [], []))
    settings = {"dem_layer_id": layers["dem"].id}
    state.gui_settings["objective"] = settings
    resp = _run(app, layers["cost"].id, algorithm="delta-stepping")
    notice = resp[ids.NOTICES]["data"][-1]
    text, _css = _status(app, state, settings, algorithm="delta-stepping")
    assert text.splitlines()[0] == notice["meaning"] and notice["focus_control"] == ids.ALGORITHM


def test_a_dem_with_a_backend_without_slope_terms_is_named(app, state, layers):
    text, _ = _status(app, state, {"dem_layer_id": layers["dem"].id}, graph_api="raster_fim", algorithm="fim")
    assert "raster_fim" in text and "cannot route over a terrain model" in text


def test_the_status_follows_the_backend_and_the_algorithm(app, state, layers):
    settings = {"dem_layer_id": layers["dem"].id}
    bad, _ = _status(app, state, settings, algorithm="delta-stepping")
    good, _ = _status(app, state, settings, algorithm="dijkstra")
    other, _ = _status(app, state, settings, graph_api="networkx", algorithm="dijkstra")
    assert bad != good and good.startswith("Objective: cost 1; DEM 'DGM'")
    assert "cython runs Dijkstra only" in good and "cython runs Dijkstra only" not in other


def test_a_layer_that_was_removed_while_referenced_is_a_problem_line_not_a_crash(app, state, layers):
    settings = {"dem_layer_id": layers["dem"].id, "layers": [service.layer_row("landscape", layers["stripe"].id)]}
    state.remove_layer(layers["stripe"].id)
    state.remove_layer(layers["dem"].id)
    text, css = _status(app, state, settings)
    assert css == form_section.STATUS_CLASS_PROBLEM
    assert text.count("not loaded any more") == 2 and "Terrain model (DEM)" in text
    assert "landscape" in text


def test_a_layer_file_that_is_gone_is_a_problem_line(app, state, layers, tmp_path):
    ghost = state.add_layer("ghost", "raster", crs=TEST_CRS, meta={"source_path": str(tmp_path / "ghost.tif")})
    text, css = _status(app, state, {"layers": [service.layer_row("g", ghost.id)]})
    assert "no longer exists" in text and css == form_section.STATUS_CLASS_PROBLEM


def test_a_dem_in_another_coordinate_system_is_a_problem_once_the_cost_raster_is_picked(app, state, layers):
    other = state.add_layer("DEM wgs84", "raster", crs="EPSG:4326",
                            meta={"source_path": layers["dem"].meta["source_path"]})
    settings = {"dem_layer_id": other.id}
    plain, _ = _status(app, state, settings)
    picked, css = _status(app, state, settings, raster_layer_id=layers["cost"].id)
    assert "EPSG:4326" not in plain                                      # no cost raster picked: nothing to compare
    assert "EPSG:4326" in picked and css == form_section.STATUS_CLASS_PROBLEM


def test_at_most_three_problems_are_listed_and_the_rest_is_counted(app, state, layers):
    settings = {"weights": {"cost": -1, "length": -2},
                "layers": [service.layer_row(name, None) for name in ("a", "b", "c")]}
    text, _ = _status(app, state, settings)
    lines = text.splitlines()
    assert len(lines) == cb.STATUS_PROBLEM_LIMIT + 1 and lines[-1].startswith("... and ")


def test_float32_on_cython_is_a_precision_problem(app, state, layers):
    settings = {"weights": {"cost": 1, "length": 1}, "weight_precision": "float32"}
    text, _ = _status(app, state, settings)
    assert text.startswith("Weight precision cannot be 32-bit float here") and "16-bit kernels" in text


def test_an_all_zero_objective_and_a_negative_weight_are_named(app, state):
    zero, _ = _status(app, state, {"weights": {"cost": 0, "length": 0}})
    assert zero.startswith("Weights are all 0")
    negative, _ = _status(app, state, {"weights": {"cost": 1, "length": -3}})
    assert "Length weight is -3" in negative and "from 0 to 1,000,000" in negative


# ================================================================== disabled states
def test_without_a_dem_the_slope_weight_and_response_are_disabled_with_the_reason(app):
    states = _states(app, {})
    assert states["weight_disabled"] is True and states["slope_disabled"] is True
    assert states["weight_title"] == service.SLOPE_NEEDS_DEM == states["slope_title"]


def test_with_a_dem_the_slope_weight_and_response_are_enabled(app, layers):
    states = _states(app, {"dem_layer_id": layers["dem"].id})
    assert states["weight_disabled"] is False and states["slope_disabled"] is False
    assert states["weight_title"] != service.SLOPE_NEEDS_DEM
    assert service.FIELD_SPECS["weights.gradient"].help in states["weight_title"]
    assert states["slope_title"] == form_section.SLOPE_HELP


def test_precision_is_off_for_classic_settings_with_the_reason(app):
    states = _states(app, {}, graph_api="networkx")
    float_option = states["precision_options"][1]
    assert float_option["value"] == "float32" and float_option["disabled"] is True
    assert states["precision_title"] == service.precision_support({}, "networkx").reason
    assert states["precision_title"].startswith("Only with an objective")


def test_precision_is_off_on_cython_even_with_an_objective_and_on_with_a_float_backend(app):
    settings = {"weights": {"cost": 1, "length": 0.5}}
    on_cython = _states(app, settings, graph_api="cython")
    assert on_cython["precision_options"][1]["disabled"] is True
    assert on_cython["precision_title"] == service.precision_support(settings, "cython").reason
    assert "16-bit kernels" in on_cython["precision_title"]
    for graph_api in ("networkx", "networkit", "igraph", "rustworkx", "raster_gpu"):
        states = _states(app, settings, graph_api=graph_api)
        assert states["precision_options"][1]["disabled"] is False, graph_api
        assert states["precision_title"] == form_section.PRECISION_HELP


def test_the_16_bit_choice_is_never_disabled_so_a_stale_32_bit_choice_can_be_left(app):
    for settings, graph_api in (({}, "cython"), ({"weights": {"cost": 1, "length": 1}}, "cython"),
                                ({"weights": {"cost": 1, "length": 1}, "weight_precision": "float32"}, "cython")):
        options = _states(app, settings, graph_api)["precision_options"]
        assert options[0]["value"] == "uint16" and options[0]["disabled"] is False


def test_the_terrain_model_hover_text_names_why_a_dem_would_be_refused(app, layers):
    refused = _states(app, {}, graph_api="cython", algorithm="delta-stepping")["dem_title"]
    assert "would be refused" in refused and "Choose Dijkstra in the Algorithm field" in refused
    fim = _states(app, {}, graph_api="raster_fim", algorithm="fim")["dem_title"]
    assert "Backend field" in fim
    fine = _states(app, {}, graph_api="cython", algorithm="dijkstra")["dem_title"]
    assert fine == form_section.DEM_HELP


def test_the_states_come_from_the_service_helpers_for_every_backend(app):
    settings = {"weights": {"cost": 1, "length": 0.5}}
    for graph_api in ("cython", "networkx", "raster_fim", "raster_gpu"):
        states = _states(app, settings, graph_api=graph_api)
        assert states == form_section.control_states(settings, graph_api, "dijkstra"), graph_api


def test_the_first_paint_states_are_what_the_callback_would_write(page):
    states = form_section.control_states(service.default_settings(), routing.DEFAULT_BACKEND,
                                         routing.default_algorithm(routing.DEFAULT_BACKEND))
    assert _by_id(page, ids.OBJ_SLOPE_WRAP).disabled is states["slope_disabled"]
    assert _by_id(page, ids.OBJ_PRECISION_WRAP).title == states["precision_title"]
    assert _by_id(page, form_section.DEM_WRAP).title == states["dem_title"]
    assert _by_id(page, ids.OBJ_PRECISION).options == states["precision_options"]


# ================================================================== shown and reset fields
@pytest.mark.parametrize("additive,multiplier", [
    ("none", "none"), ("linear", "none"), ("exponential", "none"), ("none", "exponential"), ("none", "energy"),
    ("none", "squared"), ("none", "power"), ("none", "sigmoid"), ("exponential", "sigmoid"), (None, None)])
def test_only_the_slope_fields_that_are_read_are_shown(app, additive, multiplier):
    resp = invoke(app, FIELDS, additive, multiplier)
    keys = service.applicable_gradient_keys({"additive": None if additive in (None, "none") else additive,
                                             "multiplier": None if multiplier in (None, "none") else multiplier})
    shown = {key: resp[wrapper]["style"] == {} for key, wrapper in (
        ("additive_scale", form_section.ADDITIVE_SCALE_WRAP), ("mult_scale", form_section.MULT_SCALE_WRAP),
        ("sig_center_deg", form_section.SIGMOID_WRAP))}
    assert shown == {key: key in keys for key in shown}


def test_the_scale_placeholder_names_the_default_of_the_chosen_model(app):
    for model, expected in (("exponential", "model default: 2"), ("energy", "model default: 3"),
                            ("squared", "model default: 400"), ("sigmoid", "model default"), ("none", "model default")):
        resp = invoke(app, FIELDS, "none", model)
        assert resp[ids.OBJ_GRAD_MULT_SCALE]["placeholder"] == expected


def test_a_model_change_resets_the_scale_to_the_default_of_the_new_model(app):
    previous = {"gradient": {"multiplier": "exponential", "mult_scale": 5.0}}
    for model, expected in (("energy", 3.0), ("squared", 400.0), ("sigmoid", None), ("none", None)):
        resp = invoke(app, RESET_SCALE, model, previous)
        assert resp[ids.OBJ_GRAD_MULT_SCALE]["value"] == expected


def test_a_restored_model_is_not_a_change_and_keeps_its_scale(app):
    held = {"gradient": {"multiplier": "energy", "mult_scale": 5.0}}
    with pytest.raises(PreventUpdate):
        invoke(app, RESET_SCALE, "energy", held)
    with pytest.raises(PreventUpdate):
        invoke(app, RESET_SCALE, "none", {})
    with pytest.raises(PreventUpdate):
        invoke(app, RESET_SCALE, None, None)


def test_taking_the_dem_away_switches_the_slope_settings_off(app, layers):
    held = {"dem_layer_id": layers["dem"].id, "weights": {"cost": 1, "gradient": 2.0},
            "gradient": {"max_gradient_pct": 30, "multiplier": "energy", "additive": "linear"}}
    resp = invoke(app, CLEAR_SLOPE, None, held)
    assert resp[ids.OBJ_W_GRADIENT]["value"] == 0.0 and resp[ids.OBJ_GRAD_MAX_PCT]["value"] is None
    assert resp[ids.OBJ_GRAD_ADDITIVE]["value"] == "none" and resp[ids.OBJ_GRAD_MULTIPLIER]["value"] == "none"


def test_choosing_a_dem_or_starting_without_one_resets_nothing(app, layers):
    held = {"dem_layer_id": layers["dem"].id}
    with pytest.raises(PreventUpdate):
        invoke(app, CLEAR_SLOPE, layers["dem"].id, held)               # another DEM
    with pytest.raises(PreventUpdate):
        invoke(app, CLEAR_SLOPE, layers["stripe"].id, {})              # first choice
    with pytest.raises(PreventUpdate):
        invoke(app, CLEAR_SLOPE, None, {"weights": {"gradient": 3.0}})   # no DEM before: a restore, not a change
    with pytest.raises(PreventUpdate):
        invoke(app, CLEAR_SLOPE, "", None)


# ================================================================== restore after Open and New
def _restore(app, status="opened: p"):
    return invoke(app, RESTORE, status, triggered=[f"{ids.PROJECT_STATUS}.children"])


def _saved(layers):
    return {"weights": {"cost": 1.0, "length": 0.3, "gradient": 0.5},
            "layers": [service.layer_row("landscape", layers["stripe"].id, 0.5, 30.0, 1.0),
                       service.layer_row("steep", service.SOURCE_DERIVE_SLOPE, 0.1)],
            "dem_layer_id": layers["dem"].id,
            "gradient": {"max_gradient_pct": 40.0, "additive": "exponential", "additive_scale": 3.0,
                         "multiplier": "energy", "mult_scale": 5.0, "bin_width_pct": 0.5, "s_max_pct": 150.0},
            "weight_precision": "float32"}


def test_open_project_puts_the_saved_settings_back_into_every_control(app, state, layers):
    state.gui_settings["objective"] = _saved(layers)
    resp = _restore(app)
    saved = service.normalise_objective_settings(_saved(layers))
    assert [resp[ids.OBJ_W_COST]["value"], resp[ids.OBJ_W_LENGTH]["value"], resp[ids.OBJ_W_GRADIENT]["value"]] == [
        1.0, 0.3, 0.5]
    assert resp[ids.OBJ_DEM_LAYER]["value"] == layers["dem"].id and resp[ids.OBJ_PRECISION]["value"] == "float32"
    for key, component_id in form_section.GRADIENT_IDS.items():
        shown = saved["gradient"][key]
        assert resp[component_id]["value"] == ((shown or "none") if key in form_section.CHOICE_KEYS else shown), key
    rows = resp[GRID]["rowData"]
    assert [(r["name"], r["source_label"], r["weight"], r["hard_max"], r["hard_min"]) for r in rows] == [
        ("landscape", "landscape", 0.5, 30.0, 1.0), ("steep", service.DERIVE_SLOPE_LABEL, 0.1, None, None)]
    assert all(r["id"] for r in rows) and len({r["id"] for r in rows}) == 2
    assert resp[ids.OBJ_SETTINGS]["data"] == saved                      # the store comes back with the controls


def test_open_resets_the_new_layer_form(app, state, layers):
    state.gui_settings["objective"] = _saved(layers)
    resp = _restore(app)
    assert resp[ids.OBJ_LAYER_NAME]["value"] == "" and resp[ids.OBJ_LAYER_SOURCE]["value"] is None
    assert resp[ids.OBJ_LAYER_WEIGHT]["value"] == service.FIELD_SPECS["layers.weight"].default


def test_new_project_and_a_project_without_settings_put_the_defaults_back(app, state, layers):
    state.gui_settings = {}
    resp = _restore(app, status="new project")
    assert [resp[i]["value"] for i in (ids.OBJ_W_COST, ids.OBJ_W_LENGTH, ids.OBJ_W_GRADIENT)] == [1.0, 0.0, 0.0]
    assert resp[ids.OBJ_DEM_LAYER]["value"] is None and resp[ids.OBJ_PRECISION]["value"] == "uint16"
    assert resp[GRID]["rowData"] == [] and resp[ids.OBJ_SETTINGS]["data"] == service.default_settings()
    state.gui_settings = {"constrained": {"backend": "cython"}}               # an older file: no objective entry
    assert _restore(app)[ids.OBJ_SETTINGS]["data"] == service.default_settings()


def test_saving_a_project_does_not_rewrite_the_card(app, state, layers):
    state.gui_settings["objective"] = _saved(layers)
    with pytest.raises(PreventUpdate):
        _restore(app, status="saved: C:/p/project.json")


def test_a_restore_followed_by_the_mirror_changes_nothing(app, state, layers):
    state.gui_settings["objective"] = _saved(layers)
    resp = _restore(app)
    before = json.loads(json.dumps(state.gui_settings["objective"]))
    args = (*[resp[i]["value"] for i in form_section.WEIGHT_IDS.values()],
            *[resp[i]["value"] for i in form_section.GRADIENT_IDS.values()],
            resp[ids.OBJ_DEM_LAYER]["value"], resp[ids.OBJ_PRECISION]["value"], resp[GRID]["rowData"], None, None)
    invoke(app, MIRROR, *args)
    assert state.gui_settings["objective"] == service.normalise_objective_settings(before)
    assert resp[ids.OBJ_SETTINGS]["data"] == state.gui_settings["objective"]


def test_the_settings_survive_save_and_open(app, state, layers, tmp_path):
    _mirror(app, weights=(1.0, 0.4, 0.0), dem=layers["dem"].id,
            rows=[_row("landscape", layers["stripe"].id, 0.5)],
            gradient={"multiplier": "squared", "mult_scale": 100.0})
    path = project_io.save_project(state, tmp_path / "proj")
    other = ProjectState(work_dir=tmp_path / "other")
    try:
        project_io.load_project(other, path)
        assert other.gui_settings["objective"] == state.gui_settings["objective"]
        page = build_layout(other)
        assert _by_id(page, ids.OBJ_W_LENGTH).value == 0.4
        assert _by_id(page, ids.OBJ_GRAD_MULTIPLIER).value == "squared"
        assert _by_id(page, GRID).rowData[0]["name"] == "landscape"
    finally:
        other.shutdown()


# ================================================================== first paint
def test_a_blank_project_prefills_nothing(state):
    puts: list = []
    form_section.prefill(state, lambda *args: puts.append(args))
    assert puts == []
    state.gui_settings = {"objective": {}}
    form_section.prefill(state, lambda *args: puts.append(args))
    assert puts == []
    assert set(layout._state_prefill(state)) & {cid for cid, _ in form_section.FORM_TARGETS} == set()  # noqa: SLF001


def test_a_reloaded_page_shows_the_saved_card(state, layers):
    state.gui_settings["objective"] = _saved(layers)
    page = build_layout(state)
    saved = service.normalise_objective_settings(_saved(layers))
    assert (_by_id(page, ids.OBJ_W_COST).value, _by_id(page, ids.OBJ_W_LENGTH).value,
            _by_id(page, ids.OBJ_W_GRADIENT).value) == (1.0, 0.3, 0.5)
    assert _by_id(page, ids.OBJ_DEM_LAYER).value == layers["dem"].id
    assert _by_id(page, ids.OBJ_PRECISION).value == "float32"
    assert _by_id(page, ids.OBJ_GRAD_MULTIPLIER).value == "energy"
    assert _by_id(page, ids.OBJ_GRAD_MULT_SCALE).value == 5.0
    assert _by_id(page, ids.OBJ_GRAD_ADDITIVE_SCALE).value == 3.0
    rows = _by_id(page, GRID).rowData
    assert [(r["name"], r["source_label"]) for r in rows] == [("landscape", "landscape"),
                                                              ("steep", service.DERIVE_SLOPE_LABEL)]
    assert _by_id(page, ids.OBJ_SETTINGS).data == saved


def test_a_failing_prefill_does_not_take_the_page_down(state, monkeypatch):
    state.gui_settings["objective"] = {"weights": {"cost": 2}}
    monkeypatch.setattr(form_section, "form_props", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))
    page = build_layout(state)
    assert _by_id(page, ids.OBJ_W_COST).value == 1.0


# ================================================================== no callback loop
def _prop(component_id, prop) -> tuple[str, str]:
    return str(component_id).split("@", maxsplit=1)[0], prop


def _graph(app):
    """Callback key -> (outputs, inputs) as ``(id, prop)`` pairs; States are not triggers."""
    graph = {}
    for key, entry in app.callback_map.items():
        outputs = entry["output"] if isinstance(entry["output"], (list, tuple)) else [entry["output"]]
        graph[key] = ({_prop(o.component_id, o.component_property) for o in outputs},
                      {_prop(i["id"], i["property"]) for i in entry.get("inputs", []) if isinstance(i["id"], str)})
    return graph


def _reachable(graph, start):
    seen, todo = set(), [start]
    while todo:
        key = todo.pop()
        for other, (_, inputs) in graph.items():
            if graph[key][0] & inputs and other not in seen:
                seen.add(other)
                todo.append(other)
    return seen


def test_no_card_callback_can_reach_itself_through_other_callbacks(app):
    graph = _graph(app)
    mine = [key for key, (outputs, inputs) in graph.items()
            if any(cid.startswith("obj-") for cid, _ in outputs | inputs)]
    assert len(mine) >= 12
    for key in mine:
        assert key not in _reachable(graph, key), key


def test_the_store_and_the_status_never_feed_a_control(app):
    graph = _graph(app)
    control_props = {(cid, "value") for cid in (*form_section.WEIGHT_IDS.values(), *form_section.GRADIENT_IDS.values(),
                                                ids.OBJ_DEM_LAYER, ids.OBJ_PRECISION)} | {(GRID, "rowData")}
    store = (ids.OBJ_SETTINGS, "data")
    for key, (outputs, inputs) in graph.items():
        if store in inputs:                                            # status, disabled states
            assert not outputs & control_props, key
    mirror = next(k for k, (o, i) in graph.items() if o == {store} and (GRID, "cellValueChanged") in i)
    assert not graph[mirror][0] & graph[mirror][1]


def test_the_two_reset_callbacks_read_the_store_only_as_a_state(app):
    store = (ids.OBJ_SETTINGS, "data")
    graph = _graph(app)
    for fragments in (RESET_SCALE, CLEAR_SLOPE):
        key, _entry = find_callback(app, *fragments)
        assert store not in graph[key][1]


def test_the_run_callback_got_no_new_input_or_state(app):
    _key, entry = find_callback(app, *RUN_WIRING)
    wiring = json.dumps(entry.get("inputs", [])) + json.dumps(entry.get("state", []))
    assert "obj-" not in wiring


def test_the_go_fix_helper_opens_closed_details_and_does_not_own_the_flash_store(app):
    entries = [entry for entry in app._callback_list  # noqa: SLF001
               if "clientside_function" in entry and ids.OBJ_CARD in str(entry["output"])]
    assert len(entries) == 1
    assert [i["id"] + "." + i["property"] for i in entries[0]["inputs"]] == [f"{ids.FOCUS_FLASH}.data"]
    source = cb.OPEN_CARD_JS
    assert "DETAILS" in source and ".open = true" in source and "parentElement" in source
    assert "no_update" in source                                       # the helper writes nothing itself
    assert [s for s in app._inline_scripts if "DETAILS" in s]          # noqa: SLF001


def test_every_control_a_problem_can_focus_exists_and_is_inside_a_card_the_helper_can_open(page):
    """A "Go fix" button flashes ``FIELD_CONTROLS[key]``: the control must exist, and the closed card around it
    (this card, or "Algorithm & search" for the Backend and Algorithm fields) is a ``<details>`` the helper opens."""
    for key, control in cb.FIELD_CONTROLS.items():
        component = _by_id(page, control)
        parents = [c for c in _components(page) if isinstance(c, html.Details)
                   and component in list(c._traverse())]  # noqa: SLF001
        assert parents, f"{key} -> {control}: no <details> around it"
    assert cb.control_for("layers.3.weight") == ids.OBJ_LAYERS_GRID


# ================================================================== the card -> Run -> PathFinder
def _draft(source=SOURCE, target=TARGET):
    return {"sources": [{"lat": 0, "lng": 0, "x": source[0], "y": source[1]}],
            "targets": [{"lat": 0, "lng": 0, "x": target[0], "y": target[1]}], "waypoints": []}


def _run(app, raster_id, *, algorithm="dijkstra", graph_api="cython"):
    return invoke(app, RUN_WIRING, 1, False, _draft(), raster_id, algorithm, graph_api, 1, 80, False, True, False,
                  False, 1.0, 100, 0, None, [], [], triggered=[f"{ids.RUN_ROUTING_BTN}.n_clicks"])


def test_settings_typed_into_the_card_reach_pathfinder_when_the_route_runs(app, state, layers, monkeypatch):
    constructed: list[dict] = []
    original = PathFinder.__init__

    def spy(self, *args, **kwargs):
        constructed.append(kwargs)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PathFinder, "__init__", spy)
    _mirror(app, weights=(1.0, 0.3, 0.0), dem=layers["dem"].id,
            rows=[_row("landscape", layers["stripe"].id, 0.5)],
            gradient={"multiplier": "exponential", "mult_scale": 2.5, "max_gradient_pct": 90.0})
    resp = _run(app, layers["cost"].id)
    assert "1 route(s) built" in resp[ids.ROUTING_STATUS]["children"], _notices(resp)
    first = constructed[0]
    assert first["objective"] == {"cost": 1.0, "length": 0.3, "landscape": 0.5}
    assert first["dem"] == layers["dem"].meta["source_path"]
    assert first["metric_layers"]["landscape"] == {"source": layers["stripe"].meta["source_path"]}
    assert first["gradient_options"]["multiplier"] == "exponential"
    assert first["gradient_options"]["multiplier_params"] == {"scale": 2.5}
    assert first["gradient_options"]["max_gradient_pct"] == 90.0
    route = state.layers_of_kind("route")[0]
    assert route.meta["params"]["objective_spec"] == state.gui_settings["objective"]
    session = state.search_sessions.get(route.id)                       # metric layers: no retained session
    if session is not None:
        session.close()


def test_a_card_left_on_cost_only_runs_the_plain_route(app, state, layers, monkeypatch):
    constructed: list[dict] = []
    original = PathFinder.__init__

    def spy(self, *args, **kwargs):
        constructed.append(kwargs)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PathFinder, "__init__", spy)
    _mirror(app, weights=(3.0, 0.0, 0.0))                                # a cost weight alone is still classic
    resp = _run(app, layers["cost"].id)
    assert "1 route(s) built" in resp[ids.ROUTING_STATUS]["children"], _notices(resp)
    for name in ("objective", "dem", "metric_layers", "gradient_options", "weight_precision"):
        assert name not in constructed[0]
    state.search_sessions[state.layers_of_kind("route")[0].id].close()


def test_a_dem_with_delta_stepping_in_the_card_is_refused_by_the_run_and_named_by_the_status(
        app, state, layers, monkeypatch):
    monkeypatch.setattr(routing, "run_routing", lambda *_a, **_k: pytest.fail("the run must not start"))
    _mirror(app, dem=layers["dem"].id)
    resp = _run(app, layers["cost"].id, algorithm="delta-stepping")
    notice = _notices(resp)[-1]
    status, _ = _status(app, state, state.gui_settings["objective"], algorithm="delta-stepping")
    assert notice["focus_control"] == ids.ALGORITHM and status.startswith("Algorithm is delta-stepping")
