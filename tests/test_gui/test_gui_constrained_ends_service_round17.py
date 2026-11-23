"""Round 17, P4: the line ends of constrained (overhead-line) routing in the GUI - the pure service and the plumbing.

Headless, tiny synthetic rasters, the cython constrained backend. Two layers are tested:

* the pure service ``services/constrained_ends`` (specs, normalising, problems, support, notes, what reaches the
  library, the exact-bin estimate), and that the library itself agrees with what the service says: the same
  keywords, the same refusals, the same window and the same state-space numbers;
* the run (``services/constrained``): a classic run passes nothing new and its snapshot has no new key, the heights
  reach every segment as the finder's ``find_route`` keywords (source on the first segment, target on the last, an
  explicit ``None`` at waypoints), the background snapshot carries them, and the result summary reports what the
  library says about the line ends, only where it says something.

No test depends on the number of cores or on a GPU: the GPU refusals are raised by the finder before any GPU code is
imported.
"""
from __future__ import annotations

import inspect
import json
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from rasterio.transform import from_bounds

from pyorps.core.exceptions import (
    AlgorithmNotImplementedError,
    InvalidInputError,
    InvalidTypeError,
)
from pyorps.core.infrastructure_profile import TERMINAL_SPAN_RULES, InfrastructureProfile
from pyorps.core.warning_types import PyorpsWarning
from pyorps.graph.constrained_path_finder import ConstrainedPathFinder
from pyorps.gui.services import constrained, errors, jobs
from pyorps.gui.services import constrained_ends as ce
from pyorps.gui.services.routing import default_search_buffer

CRS = "EPSG:32632"
SRC, TGT = "source_connection_height_m", "target_connection_height_m"

BASE = {
    "soft_angle_limit_deg": 20.0, "hard_angle_limit_deg": 60.0, "angle_cost_function": "linear",
    "angle_cost_params": {"scale": 10.0}, "min_span_m": 6, "max_span_m": 20, "span_bin_size_m": 1.0,
    "tower_cost_function": "terrain_plus_angle",
    "tower_cost_params": {"terrain_cost_map": {"0": 100, "500": 200}, "terrain_interpolation": "linear",
                          "angle_types": {"suspension": {"max_angle_deg": 20.0, "base_cost": 100},
                                          "strain": {"max_angle_deg": 60.0, "base_cost": 300}}}}
CLEARANCE = {"tower_height_m": 30.0, "conductor_weight_per_m": 15.0, "conductor_tension_n": 25000.0,
             "min_clearance_m": 6.0}
WITH_CLEARANCE = {**BASE, **CLEARANCE}
POINTS = [(8.0, 8.0), (52.0, 52.0)]


# ------------------------------------------------------------------ helpers
@pytest.fixture(autouse=True)
def _one_thread(monkeypatch):
    # the delta-stepping kernels read OMP_NUM_THREADS per call: one thread keeps ties deterministic (and the test
    # independent of the number of cores)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")


def _write_raster(path, array, *, cell=1.0, dtype="uint16"):
    import rasterio

    rows, cols = array.shape
    transform = from_bounds(0, 0, cols * cell, rows * cell, cols, rows)
    with rasterio.open(str(path), "w", driver="GTiff", height=rows, width=cols, count=1, dtype=dtype, crs=CRS,
                       transform=transform) as dst:
        dst.write(array.astype(dtype), 1)
    return str(path)


@pytest.fixture(scope="module")
def raster(tmp_path_factory):
    """A 60 x 60 m uniform cost raster at 1 m (a run takes a fraction of a second)."""
    return _write_raster(tmp_path_factory.mktemp("p4") / "uniform.tif", np.full((60, 60), 100, np.uint16))


@pytest.fixture(scope="module")
def dem(tmp_path_factory):
    """Flat terrain on the same grid as :func:`raster`."""
    return _write_raster(tmp_path_factory.mktemp("p4dem") / "dem.tif", np.zeros((60, 60), np.float32),
                         dtype="float32")


@pytest.fixture()
def route_calls(monkeypatch):
    """The keywords of every ``find_route`` call of the finders the GUI builds (the real class still runs)."""
    import pyorps.graph.constrained_path_finder as library

    calls: list[dict] = []
    real = library.ConstrainedPathFinder

    class Spy(real):
        def find_route(self, *args, **kwargs):
            calls.append(dict(kwargs))
            return super().find_route(*args, **kwargs)

    monkeypatch.setattr(library, "ConstrainedPathFinder", Spy)
    return calls


def _path(raster_path, *, points=POINTS, profile=BASE, ends=None, dem_path=None, backend="cython", buffer=100,
          **kwargs):
    return constrained.run_constrained_path(
        raster_path, points=points, profile=profile, backend=backend, neighborhood="r1", search_buffer_m=buffer,
        dem=dem_path, ends=ends, **kwargs)


def _quiet(function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


# ====================================================================== the spec table
def test_the_height_keys_are_the_finders_own_keywords():
    assert ce.HEIGHT_KEYS == (SRC, TGT) == tuple(ce.FIELD_SPECS) == tuple(ce.DEFAULTS)
    init = inspect.signature(ConstrainedPathFinder.__init__).parameters
    route = inspect.signature(ConstrainedPathFinder.find_route).parameters
    assert all(key in init and key in route for key in ce.HEIGHT_KEYS)


def test_the_specs_say_metres_above_zero_up_to_300_and_blank_is_not_set():
    for spec in ce.FIELD_SPECS.values():
        assert (spec.unit, spec.minimum, spec.maximum, spec.allow_equal, spec.default) == ("m", 0.0, 300.0, False,
                                                                                            None)
        assert spec.label and spec.help and spec.key in ce.LABELS[spec.key]
    assert ce.FIELD_SPECS[SRC].label != ce.FIELD_SPECS[TGT].label
    assert "first" in ce.FIELD_SPECS[SRC].help and "last" in ce.FIELD_SPECS[TGT].help


def test_the_problem_keys_cover_every_key_the_service_can_return():
    assert ce.PROBLEM_KEYS == (SRC, TGT, "rule", "backend", "bins")
    assert set(ce.PROBLEM_KEYS) <= set(ce.LABELS)


# ====================================================================== the rule table
def test_the_rule_table_is_the_librarys_four_values_in_its_order():
    assert tuple(ce.RULE_LABELS) == TERMINAL_SPAN_RULES == ("remove", "remove_if_valid", "keep", "enforce")
    assert [o["value"] for o in ce.rule_options()] == list(TERMINAL_SPAN_RULES)
    labels = [o["label"] for o in ce.rule_options()]
    assert len(set(labels)) == 4 and all(labels)
    assert all(ce.RULE_LABELS[v].help for v in TERMINAL_SPAN_RULES)
    assert ce.DEFAULT_RULE == InfrastructureProfile(
        name="x", description="x", soft_angle_limit_deg=1, hard_angle_limit_deg=2,
        angle_cost_function="linear").terminal_span_rule


def test_a_rule_label_is_the_option_text_and_an_unknown_value_stays_as_it_is():
    assert ce.rule_label("keep") == ce.RULE_LABELS["keep"].label
    assert ce.rule_label("sideways") == "sideways" and ce.rule_label(None) == "None"


# ====================================================================== normalising
def test_blank_numeric_text_junk_and_non_finite_values_normalise_without_judging():
    assert ce.normalise(None) == ce.normalise({}) == ce.normalise("junk") == {SRC: None, TGT: None}
    got = ce.normalise({SRC: "  25.5 ", TGT: ""})
    assert got == {SRC: 25.5, TGT: None} and isinstance(got[SRC], float)
    assert ce.normalise({SRC: 7})[SRC] == 7.0 and isinstance(ce.normalise({SRC: 7})[SRC], float)
    assert ce.normalise({SRC: "tall"})[SRC] == "tall"
    assert ce.normalise({SRC: True})[SRC] is True
    assert ce.normalise({SRC: float("nan")})[SRC] == "nan" and ce.normalise({TGT: "inf"})[TGT] == "inf"
    assert ce.normalise({SRC: 3, "bogus": 1}).keys() == {SRC, TGT}


def test_normalising_is_idempotent_and_json_safe():
    for raw in ({SRC: "12", TGT: float("inf")}, {SRC: "x"}, {}, None, {SRC: 0, TGT: -5}):
        once = ce.normalise(raw)
        assert ce.normalise(once) == once
        assert json.loads(json.dumps(once, allow_nan=False)) == once


def test_the_inputs_show_a_finite_number_even_outside_the_range_and_anything_else_blank():
    assert ce.input_values({SRC: 999, TGT: "tall"}) == {SRC: 999.0, TGT: None}
    assert ce.input_values(None) == {SRC: None, TGT: None}


def test_default_means_no_height_set_and_nothing_else():
    assert ce.is_default(None) and ce.is_default({}) and ce.is_default({SRC: "", TGT: None})
    assert not ce.is_default({SRC: 10}) and not ce.is_default({TGT: 0}) and not ce.is_default({SRC: "x"})


# ====================================================================== problems: the numbers
@pytest.mark.parametrize("value,word,shown", [
    (0, "too low", "0 m"), (-3, "too low", "-3 m"), (300.5, "too high", "300.5 m"),
    (1e9, "too high", "1,000,000,000 m"), ("tall", "not a finite number", "'tall'"),
    (True, "not a finite number", "True"), (float("nan"), "not a finite number", "'nan'"),
    ("inf", "not a finite number", "'inf'")])
@pytest.mark.parametrize("key", [SRC, TGT])
def test_a_bad_height_is_a_problem_that_names_the_field_the_value_the_range_and_the_unit(key, value, word, shown):
    found = ce.problems({key: value})
    assert [k for k, _ in found] == [key]
    message = found[0][1]
    assert ce.FIELD_SPECS[key].label in message and key in message and word in message and shown in message
    assert "> 0 and <= 300 (m)" in message and "blank" in message


@pytest.mark.parametrize("value", [0.001, 1, 25.5, 299.99, 300, "12", None, ""])
def test_an_allowed_or_blank_height_is_no_problem(value):
    assert ce.problems({SRC: value, TGT: value}) == []


def test_both_bad_heights_are_named_source_first_and_the_first_one_stops_a_run():
    found = ce.problems({SRC: 0, TGT: 500})
    assert [k for k, _ in found] == [SRC, TGT]
    notice = ce.problem_notice(found, controls={SRC: "src-input"})
    assert notice["severity"] == "error" and notice["focus_control"] == "src-input"
    assert "Source connection height" in notice["title"] and "0 m" in notice["meaning"]
    assert ce.problem_notice([]) is None


@pytest.mark.parametrize("value", [0, -3.0, float("nan"), float("inf"), True, 1e40, "tall"])
def test_what_the_service_refuses_the_library_refuses_too(value):
    """The service may refuse more (its 300 m limit is its own), never less."""
    assert ce.problems({SRC: value}), value
    with pytest.raises((InvalidInputError, InvalidTypeError)):
        ConstrainedPathFinder._validate_connection_height(value, SRC)  # noqa: SLF001


@pytest.mark.parametrize("value", [0.5, 12, 300, np.float32(7.5)])
def test_what_the_service_allows_the_library_accepts(value):
    assert ce.problems({SRC: float(value)}) == []
    assert ConstrainedPathFinder._validate_connection_height(value, SRC) == float(value)  # noqa: SLF001


def test_a_bad_rule_value_in_the_profile_is_a_problem_of_its_own():
    found = ce.problems({}, profile={**BASE, "terminal_span_rule": "sideways"})
    assert [k for k, _ in found] == ["rule"] and "'sideways'" in found[0][1]
    assert all(rule in found[0][1] for rule in TERMINAL_SPAN_RULES)
    assert ce.problems({}, profile={**BASE, "terminal_span_rule": None}) == []        # blank = the default
    assert ce.problem_notice(found)["focus_id"] == "tab-routes"


# ====================================================================== profile facts, effect
def test_the_profile_facts_follow_the_librarys_rules_for_a_dict_and_a_profile_object():
    facts = ce.profile_facts({**WITH_CLEARANCE, "tower_heights_m": [20.0, 25.0, 30.0]})
    assert (facts.spans_on, facts.clearance_on, facts.rule, facts.n_heights) == (True, True, "remove", 3)
    assert ce.profile_facts(None) == ce.ProfileFacts(False, False, "remove", None, None, 1)
    assert not ce.profile_facts({"min_span_m": 5}).spans_on
    assert ce.profile_facts({**BASE, "tower_heights_m": [30.0]}).n_heights == 1          # one class is not variable
    obj = InfrastructureProfile.from_dict({"name": "p", "description": "d", **WITH_CLEARANCE,
                                           "terminal_span_rule": "enforce"})
    assert ce.profile_facts(obj) == ce.profile_facts({**WITH_CLEARANCE, "terminal_span_rule": "enforce"})


def test_a_height_acts_only_with_a_dem_the_clearance_data_and_span_limits():
    both = {SRC: 20, TGT: 10}
    assert ce.effect(both, profile=WITH_CLEARANCE, dem="dem.tif") == ce.Effect(False, True, True, True)
    assert not ce.effect(both, profile=WITH_CLEARANCE, dem=None).heights_act
    assert not ce.effect(both, profile=WITH_CLEARANCE, dem="  ").heights_act
    assert not ce.effect(both, profile=BASE, dem="dem.tif").heights_act                 # no clearance data
    assert not ce.effect(both, profile=CLEARANCE, dem="dem.tif").heights_act            # no span limits
    assert ce.effect({SRC: 20}, profile=WITH_CLEARANCE, dem="d").target_acts is False
    enforce = {**BASE, "terminal_span_rule": "enforce"}
    assert ce.effect({}, profile=enforce).enforce and ce.effect({}, profile=enforce).exact_bins
    assert not ce.effect({}, profile={**CLEARANCE, "terminal_span_rule": "enforce"}).enforce
    assert not ce.effect({}, profile=WITH_CLEARANCE, dem="d").exact_bins


# ====================================================================== support
def test_the_heights_are_disabled_without_span_limits_and_on_a_gpu_backend():
    assert ce.support(profile=BASE, backend="cython").enabled
    assert ce.support(profile=BASE, backend="cython_parallel").enabled
    no_towers = ce.support(profile={}, backend="cython")
    assert not no_towers.enabled and "profile places no towers" in no_towers.reason
    for backend in ("raster_gpu", "raster_gpu_v4"):
        gpu = ce.support(profile=BASE, backend=backend)
        assert not gpu.enabled and "no connection heights" in gpu.reason and backend in gpu.reason
    assert ce.support(profile=WITH_CLEARANCE).enabled                                   # defaults: cython


def test_the_gpu_backends_are_the_librarys_list():
    assert ce.gpu_backends() == ConstrainedPathFinder.GPU_BACKENDS


def test_the_rule_select_is_disabled_without_towers_and_enforce_is_refused_on_the_gpu():
    plain = ce.rule_support(BASE, "cython")
    assert plain == ce.RuleSupport(True, "", (), "")
    assert ce.rule_support(BASE, "cython_parallel").refused == ()
    off = ce.rule_support({}, "cython")
    assert not off.enabled and "places no towers" in off.reason
    gpu = ce.rule_support(BASE, "raster_gpu_v4")
    assert gpu.enabled and gpu.refused == ("enforce",) and "raster_gpu_v4" in gpu.refused_reason
    assert not ce.rule_support({"max_span_m": 10}, "raster_gpu").enabled


# ====================================================================== problems: the combinations
ENFORCE = {**BASE, "terminal_span_rule": "enforce"}


@pytest.mark.parametrize("backend", ["raster_gpu", "raster_gpu_v4"])
def test_a_gpu_backend_with_enforce_is_a_problem_naming_the_rule(backend):
    found = ce.problems({}, backend=backend, profile=ENFORCE)
    assert [k for k, _ in found] == ["backend"] and "'enforce'" in found[0][1] and backend in found[0][1]


@pytest.mark.parametrize("backend", ["raster_gpu", "raster_gpu_v4"])
def test_a_gpu_backend_with_a_height_the_search_would_use_is_a_problem_naming_the_height(backend):
    found = ce.problems({SRC: 20}, backend=backend, profile=WITH_CLEARANCE, dem="dem.tif")
    assert [k for k, _ in found] == ["backend"] and "source height 20 m" in found[0][1]
    assert "target height" not in found[0][1]
    both = ce.problems({SRC: 20, TGT: 9}, backend=backend, profile={**WITH_CLEARANCE, "terminal_span_rule": "enforce"},
                       dem="dem.tif")
    assert len(both) == 1 and "'enforce'" in both[0][1] and "target height 9 m" in both[0][1]
    assert "cython" in ce.problem_notice(both)["fix"]


def test_a_combination_the_library_would_take_is_never_a_problem():
    gpu = "raster_gpu"
    assert ce.problems({SRC: 20}, backend=gpu, profile=WITH_CLEARANCE, dem=None) == []     # reported only
    assert ce.problems({SRC: 20}, backend=gpu, profile=BASE, dem="d") == []                # no clearance data
    assert ce.problems({SRC: 20}, backend=gpu, profile=CLEARANCE, dem="d") == []           # no span limits
    assert ce.problems({}, backend=gpu, profile={**CLEARANCE, "terminal_span_rule": "enforce"}) == []
    assert ce.problems({}, backend=gpu, profile={**BASE, "terminal_span_rule": "keep"}) == []
    for backend in ("cython", "cython_parallel"):
        assert ce.problems({SRC: 20, TGT: 9}, backend=backend, profile={**WITH_CLEARANCE, "terminal_span_rule":
                                                                        "enforce"}, dem="d") == []
    assert ce.problems({SRC: 20}, backend=None, profile=WITH_CLEARANCE, dem="d") == []     # blank = cython


def test_a_bad_value_is_reported_before_a_backend_refusal():
    found = ce.problems({SRC: 0}, backend="raster_gpu", profile=ENFORCE)
    assert [k for k, _ in found] == [SRC]


def test_the_library_refuses_what_the_service_calls_a_gpu_problem(raster, dem):
    """The finder raises before any GPU code is imported, so this needs no GPU."""
    cases = [({}, ENFORCE, None), ({SRC: 20}, WITH_CLEARANCE, dem)]
    for ends, profile, dem_path in cases:
        assert ce.problems(ends, backend="raster_gpu", profile=profile, dem=dem_path)
        with pytest.raises(AlgorithmNotImplementedError):
            _quiet(_path, raster, profile=profile, ends=ends, dem_path=dem_path, backend="raster_gpu")


# ====================================================================== notes and the status line
def test_a_height_without_effect_is_a_note_that_says_the_route_does_not_change():
    out = ce.notes({SRC: 20}, profile=BASE, dem=None)
    assert len(out) == 1 and "reported only, the route does not change" in out[0]
    assert "no terrain model" in out[0] and "lacks tower height" in out[0] and "source height 20 m" in out[0]
    assert "no span limits" in ce.notes({TGT: 9}, profile=CLEARANCE, dem="d")[0]
    only_dem = ce.notes({SRC: 20}, profile=BASE, dem="d")[0]
    assert "terrain model" not in only_dem and "lacks tower height" in only_dem


def test_notes_are_empty_when_nothing_is_set_or_the_height_acts():
    assert ce.notes({}, profile=BASE) == []
    assert ce.notes({SRC: 20}, profile=WITH_CLEARANCE, dem="d")[0].startswith("This run searches with exact span bins")
    assert not any("reported only" in n for n in ce.notes({SRC: 20}, profile=WITH_CLEARANCE, dem="d"))


def test_a_rule_without_towers_and_an_exact_search_are_notes_too():
    assert "changes nothing" in ce.notes({}, profile={"terminal_span_rule": "keep"})[0]
    assert ce.notes({}, profile={"terminal_span_rule": "remove"}) == []
    exact = ce.notes({}, profile=ENFORCE)
    assert len(exact) == 1 and "exact span bins" in exact[0] and "more memory" in exact[0]


def test_the_status_line_of_the_view():
    plain = ce.view({}, profile=BASE)
    assert plain.enabled and not plain.problem and plain.reason == ""
    assert plain.text.startswith("No connection heights are set")
    acting = ce.view({SRC: 20, TGT: 9.5}, profile=WITH_CLEARANCE, dem="d")
    assert acting.enabled and "source height 20 m, target height 9.5 m" in acting.text
    assert "reported only" not in acting.text
    quiet = ce.view({SRC: 20}, profile=BASE, dem=None)
    assert quiet.enabled and "reported only, the route does not change" in quiet.text
    bad = ce.view({SRC: -1}, profile=BASE)
    assert bad.problem and "too low" in bad.text and bad.enabled
    towers = ce.view({SRC: 20}, profile={}, dem="d")
    assert not towers.enabled and towers.reason in towers.text and "reported only" in towers.text
    gpu = ce.view({}, backend="raster_gpu", profile=BASE)
    assert not gpu.enabled and gpu.text == gpu.reason
    refused = ce.view({SRC: 20}, backend="raster_gpu_v4", profile=WITH_CLEARANCE, dem="d")
    assert refused.problem and not refused.enabled and "source height 20 m" in refused.text


def test_the_summary_line_names_the_heights_that_are_set():
    assert "No connection heights" in ce.summary({})
    assert ce.summary({TGT: 12}) == ("Connection heights: target height 12 m (metres above the ground at the end "
                                     "point).")


# ====================================================================== what reaches the library
def test_without_a_height_nothing_reaches_the_library_for_any_segment():
    for raw in (None, {}, {SRC: None, TGT: ""}):
        assert ce.library_kwargs(raw) == {} == ce.library_kwargs(raw, 2, 5)
    assert ce.used_ends(None) == {}


def test_one_segment_gets_both_heights_as_floats():
    got = ce.library_kwargs({SRC: "20", TGT: 9})
    assert got == {SRC: 20.0, TGT: 9.0} and all(isinstance(v, float) for v in got.values())
    assert ce.library_kwargs({SRC: 20}) == {SRC: 20.0, TGT: None}                      # an explicit None, not a gap


def test_the_source_height_goes_to_the_first_segment_and_the_target_height_to_the_last():
    ends = {SRC: 20.0, TGT: 9.0}
    assert ce.library_kwargs(ends, 0, 3) == {SRC: 20.0, TGT: None}
    assert ce.library_kwargs(ends, 1, 3) == {SRC: None, TGT: None}                     # a waypoint on both sides
    assert ce.library_kwargs(ends, 2, 3) == {SRC: None, TGT: 9.0}
    assert ce.library_kwargs(ends, 0, 2) == {SRC: 20.0, TGT: None}
    assert ce.library_kwargs(ends, 1, 2) == {SRC: None, TGT: 9.0}
    only_target = {TGT: 9.0}
    assert ce.library_kwargs(only_target, 0, 2) == {SRC: None, TGT: None}              # still explicit: a height IS set
    assert ce.library_kwargs(only_target, 1, 2) == {SRC: None, TGT: 9.0}


def test_a_segment_index_outside_the_run_and_a_bad_height_raise_the_projects_error():
    for index, n in ((-1, 2), (2, 2), (0, 0), (1.5, 3), (0, 2.0)):
        with pytest.raises(InvalidInputError, match="segment index"):
            ce.library_kwargs({SRC: 5}, index, n)
    with pytest.raises(InvalidInputError, match="too low"):
        ce.library_kwargs({SRC: 0})
    with pytest.raises(InvalidInputError, match="not a finite number"):
        ce.library_kwargs({TGT: "tall"}, 0, 1)
    with pytest.raises(InvalidInputError, match="too high"):
        ce.used_ends({SRC: 999})


def test_the_used_ends_are_the_set_heights_as_floats():
    assert ce.used_ends({SRC: "20", TGT: ""}) == {SRC: 20.0}
    assert ce.used_ends({SRC: 20, TGT: 9}) == {SRC: 20.0, TGT: 9.0}


# ====================================================================== the exact-bin estimate: the librarys numbers
def _finder(raster_path, *, cell_profile=BASE, backend="cython", neighborhood="r2", buffer=100,
            source=(8.0, 8.0), target=(52.0, 52.0), dem_path=None):
    return _quiet(ConstrainedPathFinder, dataset_source=raster_path, source_coords=source, target_coords=target,
                  profile={"name": "p", "description": "d", **cell_profile}, graph_api=backend,
                  neighborhood_str=neighborhood, search_space_buffer_m=buffer, dem=dem_path)


@pytest.fixture(scope="module")
def coarse_raster(tmp_path_factory):
    """40 x 40 cells of 2.5 m (a 100 m square)."""
    return _write_raster(tmp_path_factory.mktemp("p4coarse") / "c.tif", np.full((40, 40), 100, np.uint16),
                         cell=2.5)


@pytest.mark.parametrize("neighborhood", ["r1", "r2", "r3"])
@pytest.mark.parametrize("backend", ["cython", "cython_parallel", "raster_gpu"])
def test_the_estimate_equals_the_finders_bins_directions_and_states(raster, neighborhood, backend):
    finder = _finder(raster, backend=backend, neighborhood=neighborhood)
    rows, cols = 33, 47
    est = ce.estimate_exact_bins(rows, cols, max_span_m=20, cell_size_m=1.0, neighborhood=neighborhood,
                                 backend=backend)
    n_bins, bin_size = finder._exact_span_bins(np.zeros((rows, cols)), None)  # noqa: SLF001
    assert (est.n_bins, est.bin_size_m) == (n_bins, bin_size)
    assert est.n_directions == len(finder.steps)
    assert est.states == rows * cols * len(finder.steps) * n_bins
    assert est.shortest_step_m == float(np.min(finder._step_distances))  # noqa: SLF001


def test_the_estimate_uses_the_cell_size_and_the_height_classes(coarse_raster):
    finder = _finder(coarse_raster, source=(10.0, 10.0), target=(90.0, 90.0))
    est = ce.estimate_exact_bins(20, 30, max_span_m=20, cell_size_m=2.5, neighborhood="r2", n_heights=3)
    n_bins, bin_size = finder._exact_span_bins(np.zeros((20, 30)), None)  # noqa: SLF001
    assert est.shortest_step_m == 2.5 and n_bins == 9
    assert (est.n_bins, est.bin_size_m) == (n_bins, bin_size)
    assert est.n_heights == 3 and est.states == 20 * 30 * est.n_directions * 9 * 3


@pytest.mark.parametrize("neighborhood", ["r1", "r2"])
def test_the_library_refuses_with_the_numbers_of_the_estimate_and_the_estimate_says_exceeds(
        raster, monkeypatch, neighborhood):
    finder = _finder(raster, neighborhood=neighborhood)
    rows, cols = 40, 50
    monkeypatch.setattr(ConstrainedPathFinder, "EXACT_BINS_MAX_STATES", 100_000)
    est = ce.estimate_exact_bins(rows, cols, max_span_m=20, neighborhood=neighborhood)
    assert est.max_states == 100_000 and est.exceeds
    with pytest.raises(InvalidInputError) as caught:
        finder._exact_span_bins(np.zeros((rows, cols)), None)  # noqa: SLF001
    text = str(caught.value)
    assert f"{est.states:,}" in text and f"{est.n_bins} span bins" in text and f"{est.n_directions} directions" in text
    smaller = ce.estimate_exact_bins(10, 10, max_span_m=20, neighborhood=neighborhood)
    assert not smaller.exceeds


def test_too_many_bins_for_the_kernels_16_bits_is_the_librarys_second_limit(raster, monkeypatch):
    monkeypatch.setattr(ConstrainedPathFinder, "EXACT_BINS_MAX_COUNT", 5)
    est = ce.estimate_exact_bins(10, 10, max_span_m=20)
    assert est.max_bins == 5 and est.n_bins > 5 and est.exceeds
    assert est.states < est.max_states
    message = ce.bins_message(est, "a connection height")
    assert f"{est.n_bins:,} span bins" in message and "at most 5" in message
    finder = _finder(raster)
    with pytest.raises(InvalidInputError, match=f"{est.n_bins} span bins"):
        finder._exact_span_bins(np.zeros((10, 10)), None)  # noqa: SLF001


def test_the_limits_are_read_from_the_finder_and_not_copied(monkeypatch):
    monkeypatch.setattr(ConstrainedPathFinder, "EXACT_BINS_MAX_STATES", 7)
    monkeypatch.setattr(ConstrainedPathFinder, "EXACT_BINS_MAX_COUNT", 3)
    est = ce.estimate_exact_bins(1, 1, max_span_m=5)
    assert (est.max_states, est.max_bins) == (7, 3)


# ====================================================================== the four bounds: parity with the finder
# The finder refuses an exact-bin run on four bounds (bins, steps per longest span, states, memory of the kernel it
# would call). Each test below sets ONE limit exactly at the estimate's value (the run fits) and one below it (the run
# is refused) and compares the service's verdict with what the finder does when its kernel calls are stubbed out.
LONG = {**BASE, "min_span_m": 150.0, "max_span_m": 400.0, "terminal_span_rule": "enforce"}
LONG_CLEARANCE = {**LONG, **CLEARANCE}
LONG_THREE_CLASSES = {**LONG_CLEARANCE, "tower_heights_m": [20.0, 25.0, 30.0], "tower_height_cost_per_increment": 10.0}
#: (backend, profile, terrain model given, height classes): the kernels the memory estimate has to tell apart
KERNEL_CASES = [
    pytest.param("cython", LONG, False, 1, id="dijkstra"),
    pytest.param("cython", LONG, True, 1, id="dijkstra-with-dem"),
    pytest.param("cython_parallel", LONG, False, 1, id="delta-stepping"),
    pytest.param("cython", LONG_CLEARANCE, True, 1, id="clearance"),
    pytest.param("cython_parallel", LONG_THREE_CLASSES, True, 3, id="clearance-three-classes"),
]
#: bound name -> (finder class attribute holding its limit, the estimate's value for it)
BOUNDS = {
    "bins": ("EXACT_BINS_MAX_COUNT", lambda est: est.n_bins),
    "steps": ("EXACT_BINS_MAX_STEPS", lambda est: est.steps_per_span),
    "states": ("EXACT_BINS_MAX_STATES", lambda est: est.states),
    "memory": ("EXACT_BINS_MAX_BYTES", lambda est: est.nbytes),
}
_KERNELS = ("constrained_dijkstra_2d", "constrained_delta_stepping_2d", "constrained_delta_stepping_height_2d",
            "constrained_delta_stepping_lazy")


def _window_finder(n, profile, backend, with_dem):
    """A finder whose search window is the whole ``n`` x ``n`` raster of 1 m cells (the buffer is huge)."""
    import rasterio.transform

    extra = {"dem": np.zeros((n, n), np.float32)} if with_dem else {}
    return _quiet(
        ConstrainedPathFinder, dataset_source=np.full((n, n), 100, np.uint16), crs=CRS,
        transform=rasterio.transform.from_origin(0.0, float(n), 1.0, 1.0), source_coords=(5.5, n / 2),
        target_coords=(n - 5.5, n / 2), search_space_buffer_m=1000,
        profile={"name": "p", "description": "d", **profile}, graph_api=backend, neighborhood_str="r2", **extra)


def _stub_kernels(monkeypatch):
    """Replace every search kernel by a stub that notes the call; the list it returns is empty when none was reached."""
    import pyorps.utils.constrained_path_algorithms as algorithms

    called = []

    def make(name):
        def stub(**kwargs):
            called.append(name)
            route = (np.array([0, 1], np.int64), np.array([], np.int64))
            return route + (np.array([], np.float32),) if "height" in name or "lazy" in name else route
        return stub

    for name in _KERNELS:
        monkeypatch.setattr(algorithms, name, make(name))
    return called


def _finder_refuses(finder, called):
    """True when the finder refuses the exact-bin run before any kernel call, False when it reaches a kernel."""
    called.clear()
    try:
        _quiet(finder.find_route)
    except InvalidInputError as err:
        assert not called, "a refused run must not reach a kernel"
        assert "exact span bins" in str(err)
        return True
    except Exception:  # noqa: BLE001  (the stub route is not a line: the run got past the bounds)
        pass
    assert called, "the run neither refused nor reached a kernel"
    return False


def _estimate_of(finder, profile, backend, with_dem, n_heights):
    window = finder.raster_handler.window
    facts = ce.profile_facts(profile)
    return ce.estimate_exact_bins(int(window.height), int(window.width), max_span_m=profile["max_span_m"],
                                  cell_size_m=1.0, neighborhood="r2", backend=backend, n_heights=n_heights,
                                  clearance=with_dem and facts.clearance_on, has_dem=with_dem)


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
@pytest.mark.parametrize("bound", list(BOUNDS))
@pytest.mark.parametrize("n", [14, 31])
@pytest.mark.parametrize("backend,profile,with_dem,n_heights", KERNEL_CASES)
def test_the_services_verdict_equals_the_finders_just_below_and_just_above_every_bound(
        monkeypatch, backend, profile, with_dem, n_heights, n, bound, sparse):
    import pyorps.graph.constrained_path_finder as library

    if sparse:
        monkeypatch.setattr(library, "_KERNEL_DENSE_MAX_STATES", 1000)    # every window is above the dense limit
    called = _stub_kernels(monkeypatch)
    finder = _window_finder(n, profile, backend, with_dem)
    est = _estimate_of(finder, profile, backend, with_dem, n_heights)
    assert est.n_heights == n_heights and not est.exceeds, "the real limits hold these small windows"
    attribute, value_of = BOUNDS[bound]
    value = int(value_of(est))
    assert value > 0
    for limit, refused in ((value, False), (value - 1, True), (value + 1, False)):
        monkeypatch.setattr(ConstrainedPathFinder, attribute, limit)
        est = _estimate_of(finder, profile, backend, with_dem, n_heights)
        assert est.exceeds is refused, (bound, limit)
        assert est.exceeded == ((bound,) if refused else ()), (bound, limit)
        assert _finder_refuses(finder, called) is refused, (bound, limit)


@pytest.mark.parametrize("n_cells,backend,expected_form", [(270, "cython", "dense"), (550, "cython", "sparse"),
                                                          (550, "cython_parallel", "dense")])
def test_the_reviewers_large_windows_are_over_the_memory_bound_by_the_finders_own_numbers(n_cells, backend,
                                                                                         expected_form):
    est = ce.estimate_exact_bins(n_cells, n_cells, max_span_m=400, neighborhood="r2", backend=backend)
    memory = ConstrainedPathFinder.exact_bins_memory(n_cells * n_cells, est.n_directions, est.n_bins,
                                                     backend=backend)
    assert est.memory == memory and memory.form == expected_form
    assert est.exceeded == ("memory",) and est.nbytes == memory.nbytes > est.max_bytes
    message = ce.bins_message(est, "the end-span rule 'enforce'")
    assert memory.kernel in message and f"{memory.nbytes / 1024 ** 3:.1f} GiB" in message
    assert "4.0 GiB" in message and f"{est.states:,} states" in message


def test_a_window_the_finder_takes_is_not_over_any_bound_and_its_estimate_is_the_finders(monkeypatch):
    called = _stub_kernels(monkeypatch)
    finder = _window_finder(60, LONG, "cython", False)
    est = _estimate_of(finder, LONG, "cython", False, 1)
    assert est.exceeded == () and not est.exceeds
    assert _finder_refuses(finder, called) is False
    big = _window_finder(270, LONG, "cython", False)
    assert _estimate_of(big, LONG, "cython", False, 1).exceeded == ("memory",)
    assert _finder_refuses(big, called) is True


@pytest.mark.parametrize("max_span,refused", [(2047.0, False), (2048.0, False), (2049.0, True), (2100.0, True)])
def test_the_step_bound_at_its_real_limit_agrees_with_the_finder(monkeypatch, max_span, refused):
    called = _stub_kernels(monkeypatch)
    profile = {**LONG, "max_span_m": max_span}
    finder = _window_finder(14, profile, "cython", False)
    est = _estimate_of(finder, profile, "cython", False, 1)
    assert est.max_steps == ConstrainedPathFinder.EXACT_BINS_MAX_STEPS == 2048
    assert est.steps_per_span == int(max_span) and est.exceeds is refused
    assert est.exceeded == (("steps",) if refused else ())
    assert _finder_refuses(finder, called) is refused


def test_the_clearance_flag_and_the_dem_flag_change_the_memory_the_way_the_finder_does():
    kwargs = {"max_span_m": 400, "neighborhood": "r2", "backend": "cython"}
    plain = ce.estimate_exact_bins(100, 100, **kwargs)
    with_dem = ce.estimate_exact_bins(100, 100, has_dem=True, **kwargs)
    clearance = ce.estimate_exact_bins(100, 100, clearance=True, has_dem=True, **kwargs)
    assert plain.memory.kernel == with_dem.memory.kernel == "constrained_dijkstra_2d"
    assert clearance.memory.kernel == "constrained_delta_stepping_height_2d"
    assert plain.nbytes < with_dem.nbytes                       # the DEM adds a gradient cache per cell and direction
    assert with_dem.memory == ConstrainedPathFinder.exact_bins_memory(
        100 * 100, with_dem.n_directions, with_dem.n_bins, 1, backend="cython", has_dem=True)


def test_the_limits_of_all_four_bounds_are_read_from_the_finder_and_not_copied(monkeypatch):
    for name, value in (("EXACT_BINS_MAX_BYTES", 11), ("EXACT_BINS_MAX_STEPS", 5), ("EXACT_BINS_MAX_STATES", 7),
                        ("EXACT_BINS_MAX_COUNT", 3)):
        monkeypatch.setattr(ConstrainedPathFinder, name, value)
    est = ce.estimate_exact_bins(1, 1, max_span_m=6)
    assert (est.max_bytes, est.max_steps, est.max_states, est.max_bins) == (11, 5, 7, 3)
    assert est.exceeded == ("bins", "steps", "states", "memory")     # the finder's order of checks


def _over(monkeypatch, bound, rows=12, cols=12, **kwargs):
    attribute, value_of = BOUNDS[bound]
    est = ce.estimate_exact_bins(rows, cols, max_span_m=20, neighborhood="r1", **kwargs)
    with monkeypatch.context() as local:                       # the limit is lowered for this one estimate only
        local.setattr(ConstrainedPathFinder, attribute, int(value_of(est)) - 1)
        return ce.estimate_exact_bins(rows, cols, max_span_m=20, neighborhood="r1", **kwargs)


def test_the_message_says_which_bound_with_its_numbers_and_the_advice_that_fits(monkeypatch):
    gib = 1024 ** 3
    rules = '"Remove it", "Remove it only if every span stays valid" or "Keep it"'
    est = _over(monkeypatch, "memory")
    text = ce.bins_message(est, "a connection height", buffer_m=80)
    assert est.exceeded == ("memory",)
    assert f"{est.memory.kernel} ({est.memory.form} form)" in text and f"{est.nbytes / gib:.1f} GiB" in text
    assert f"limit of {est.max_bytes / gib:.1f} GiB" in text and f"{est.states:,} states" in text
    assert "coarser raster" in text and "smaller search buffer (now 80 m)" in text and "max_span_m" in text
    assert rules in text and "also above" not in text
    est = _over(monkeypatch, "steps")
    text = ce.bins_message(est, "the end-span rule 'enforce'", buffer_m=80)
    assert est.exceeded == ("steps",)
    assert f"{est.steps_per_span:,} steps of the shortest step" in text
    assert f"limit of {est.max_steps:,} steps per longest span" in text
    assert "coarser raster" in text and "max_span_m" in text and rules in text
    assert "search buffer" not in text                            # a smaller window does not shorten a span
    est = _over(monkeypatch, "states")
    text = ce.bins_message(est, "a connection height", buffer_m=80, segment="Segment 2 of 3")
    assert text.startswith("Segment 2 of 3: Exact span bins: ")
    assert f"{est.states:,} states" in text and f"limit of {est.max_states:,} states" in text
    assert "smaller search buffer (now 80 m)" in text and rules in text
    est = _over(monkeypatch, "bins")
    text = ce.bins_message(est, "a connection height")
    assert f"{est.n_bins:,} span bins" in text and f"at most {est.max_bins:,}" in text and rules in text


def test_a_message_over_several_bounds_names_the_first_and_lists_the_others(monkeypatch):
    for name, value in (("EXACT_BINS_MAX_BYTES", 11), ("EXACT_BINS_MAX_STATES", 7)):
        monkeypatch.setattr(ConstrainedPathFinder, name, value)
    est = ce.estimate_exact_bins(12, 12, max_span_m=20, neighborhood="r1")
    assert est.exceeded == ("states", "memory")
    text = ce.bins_message(est, "a connection height")
    assert "limit of 7 states" in text and "also above the limit of: " in text and "GiB (limit 0.0 GiB)" in text


def test_bins_problem_refuses_on_the_memory_bound_alone_and_names_the_kernel(raster, dem, monkeypatch):
    est = ce.estimate_exact_bins(60, 60, max_span_m=20, neighborhood="r1")
    assert not est.exceeds
    monkeypatch.setattr(ConstrainedPathFinder, "EXACT_BINS_MAX_BYTES", est.nbytes - 1)
    key, message = _bins(raster)
    assert key == "bins" and est.memory.kernel in message and "GiB" in message and "'enforce'" in message
    assert "limit of" in message and "states" in message
    # a DEM adds to the memory (the same window needs more bytes), so the limit that just fits the plain run refuses it
    monkeypatch.setattr(ConstrainedPathFinder, "EXACT_BINS_MAX_BYTES", est.nbytes)
    assert _bins(raster) is None
    key, message = _bins(raster, profile=ENFORCE, dem_path=dem)
    assert key == "bins" and "constrained_dijkstra_2d" in message
    # the finder agrees: the same window, with and without the DEM, on the real class
    with pytest.raises(InvalidInputError, match="GiB of memory"):
        _quiet(_path, raster, profile=ENFORCE, dem_path=dem)
    assert _quiet(_path, raster, profile=ENFORCE) is not None


def test_bins_problem_refuses_a_run_over_the_step_bound_before_anything_else(raster):
    profile = {**ENFORCE, "min_span_m": 100.0, "max_span_m": 5000.0}
    key, message = _bins(raster, profile=profile)
    assert key == "bins" and "steps per longest span" in message and "2,048" in message
    with pytest.raises(InvalidInputError, match="EXACT_BINS_MAX_STEPS"):
        _quiet(_path, raster, profile=profile)


# ====================================================================== the window
@pytest.mark.parametrize("buffer", [3, 12, 40, 100, 5000, None])
@pytest.mark.parametrize("source,target", [((8.0, 8.0), (52.0, 52.0)), ((2.0, 55.0), (30.0, 3.0)),
                                           ((30.5, 30.5), (31.5, 29.5))])
def test_the_window_is_the_one_the_finder_reads(raster, buffer, source, target):
    # a blank buffer is what run_constrained turns into its automatic one before the finder sees it
    effective = buffer or default_search_buffer([source], [target])
    finder = _finder(raster, buffer=effective, source=source, target=target)
    handler = finder.raster_handler
    shape = ce.window_shape(raster, source, target, buffer)
    assert shape == (int(handler.window.height), int(handler.window.width), 1.0)


def test_the_window_on_a_coarse_raster_and_for_a_blank_or_zero_buffer(coarse_raster):
    finder = _finder(coarse_raster, buffer=40, source=(10.0, 10.0), target=(70.0, 90.0))
    handler = finder.raster_handler
    assert ce.window_shape(coarse_raster, (10.0, 10.0), (70.0, 90.0), 40) == (
        int(handler.window.height), int(handler.window.width), 2.5)
    assert ce.window_shape(coarse_raster, (10.0, 10.0), (70.0, 90.0), 0) == ce.window_shape(
        coarse_raster, (10.0, 10.0), (70.0, 90.0), None)


def test_a_window_that_cannot_be_known_is_none():
    assert ce.window_shape(np.zeros((5, 5)), (1, 1), (3, 3), 10) is None
    assert ce.window_shape("does/not/exist.tif", (1, 1), (3, 3), 10) is None


# ====================================================================== the exact-bin problem before a run
def _small_limit(monkeypatch, states):
    monkeypatch.setattr(ConstrainedPathFinder, "EXACT_BINS_MAX_STATES", states)


def _bins(raster_path, ends=None, *, profile=ENFORCE, dem_path=None, points=POINTS, buffer=100, backend="cython"):
    return ce.bins_problem(ends, raster_source=raster_path, points=points, profile=profile, backend=backend,
                           neighborhood="r1", dem=dem_path, search_buffer_m=buffer)


def test_a_default_run_has_no_bin_problem_however_small_the_limit(raster, monkeypatch):
    _small_limit(monkeypatch, 10)
    for profile in (BASE, {**BASE, "terminal_span_rule": "keep"}, {**BASE, "terminal_span_rule": "remove_if_valid"}):
        assert _bins(raster, None, profile=profile) is None
    assert _bins(raster, {SRC: 20}, profile=BASE) is None                       # a height without effect
    assert _bins(raster, {SRC: 20}, profile=WITH_CLEARANCE, dem_path=None) is None


def test_enforce_over_the_limit_is_a_problem_with_the_numbers_and_the_advice(raster, monkeypatch):
    _small_limit(monkeypatch, 200_000)
    key, message = _bins(raster)
    assert key == "bins"
    est = ce.estimate_exact_bins(60, 60, max_span_m=20, cell_size_m=1.0, neighborhood="r1")
    assert f"{est.states:,}" in message and "200,000" in message and "60 x 60 cells" in message
    assert "'enforce'" in message and "coarser raster" in message and "smaller search buffer" in message
    assert "(now 100 m)" in message


def test_the_problem_is_the_libraries_refusal_with_the_same_numbers(raster, monkeypatch):
    """What the service says before the run is what the library says in the run."""
    _small_limit(monkeypatch, 200_000)
    _, message = _bins(raster)
    with pytest.raises(InvalidInputError) as caught:
        _quiet(_path, raster, profile=ENFORCE)
    library = str(caught.value)
    est = ce.estimate_exact_bins(60, 60, max_span_m=20, neighborhood="r1")
    for fact in (f"{est.states:,}", f"{est.n_bins} span bins", f"{est.n_directions} directions", "60 x 60 cells",
                 "200,000"):
        assert fact in message and fact in library, fact


def test_a_height_the_search_uses_over_the_limit_is_a_problem_and_without_a_dem_it_is_not(raster, dem, monkeypatch):
    _small_limit(monkeypatch, 200_000)
    key, message = _bins(raster, {SRC: 20}, profile=WITH_CLEARANCE, dem_path=dem)
    assert key == "bins" and "a connection height" in message and "'enforce'" not in message
    assert _bins(raster, {SRC: 20}, profile=WITH_CLEARANCE, dem_path=None) is None
    with pytest.raises(InvalidInputError, match="exact span bins"):
        _quiet(_path, raster, profile=WITH_CLEARANCE, ends={SRC: 20}, dem_path=dem)
    # without a DEM the library does not use exact bins, so it takes the same run (the height is reported only)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert _path(raster, profile=WITH_CLEARANCE, ends={SRC: 20}) is not None


def test_the_height_classes_count_only_with_the_clearance_check_on(raster, dem, monkeypatch):
    several = {**WITH_CLEARANCE, "tower_heights_m": [20.0, 25.0, 30.0], "tower_height_cost_per_increment": 10.0}
    single_class = ce.estimate_exact_bins(60, 60, max_span_m=20, neighborhood="r1", n_heights=1)
    triple = ce.estimate_exact_bins(60, 60, max_span_m=20, neighborhood="r1", n_heights=3)
    limit = (single_class.states + triple.states) // 2
    _small_limit(monkeypatch, limit)
    assert single_class.states < limit < triple.states
    assert _bins(raster, {SRC: 20}, profile=WITH_CLEARANCE, dem_path=dem) is None
    key, message = _bins(raster, {SRC: 20}, profile=several, dem_path=dem)
    assert key == "bins" and "x 3 height classes" in message


def test_the_waypoint_segments_are_judged_one_by_one(raster, monkeypatch):
    points = [(10.0, 10.0), (20.0, 20.0), (50.0, 50.0)]
    small = ce.window_shape(raster, points[0], points[1], 5)
    big = ce.window_shape(raster, points[1], points[2], 5)
    est_small = ce.estimate_exact_bins(small[0], small[1], max_span_m=20, neighborhood="r1")
    est_big = ce.estimate_exact_bins(big[0], big[1], max_span_m=20, neighborhood="r1")
    assert est_small.states < est_big.states
    _small_limit(monkeypatch, (est_small.states + est_big.states) // 2)
    key, message = _bins(raster, points=points, buffer=5)
    assert key == "bins" and message.startswith("Segment 2 of 2: ")        # enforce applies to every segment
    assert f"{est_big.states:,}" in message


def test_a_source_height_is_judged_on_the_first_segment_and_a_target_height_on_the_last(raster, dem, monkeypatch):
    points = [(10.0, 10.0), (20.0, 20.0), (50.0, 50.0)]
    small = ce.window_shape(raster, points[0], points[1], 5)
    big = ce.window_shape(raster, points[1], points[2], 5)
    states_small = ce.estimate_exact_bins(small[0], small[1], max_span_m=20, neighborhood="r1").states
    states_big = ce.estimate_exact_bins(big[0], big[1], max_span_m=20, neighborhood="r1").states
    _small_limit(monkeypatch, (states_small + states_big) // 2)
    source_only = _bins(raster, {SRC: 20}, profile=WITH_CLEARANCE, dem_path=dem, points=points, buffer=5)
    target_only = _bins(raster, {TGT: 20}, profile=WITH_CLEARANCE, dem_path=dem, points=points, buffer=5)
    assert source_only is None
    assert target_only is not None and target_only[1].startswith("Segment 2 of 2: ")


def test_an_estimate_that_cannot_be_made_leaves_the_decision_to_the_library(raster, monkeypatch):
    _small_limit(monkeypatch, 200_000)
    assert _bins(np.zeros((60, 60)), None) is None                       # not a file: the window is unknown
    assert _bins(raster, None, profile={**ENFORCE, "max_span_m": "far"}) is None
    assert ce.bins_problem({SRC: -1}, raster_source=raster, points=POINTS, profile=ENFORCE) is None
    assert ce.bins_problem(None, raster_source=raster, points=[POINTS[0]], profile=ENFORCE) is None
    # ... and the library's own refusal then reaches the notice unchanged
    with pytest.raises(InvalidInputError) as caught:
        _quiet(_path, raster, profile=ENFORCE)
    notice = errors.translate_exception(caught.value)
    assert str(caught.value) in notice.details and "exact span bins" in notice.details


def test_run_problems_joins_the_two_and_the_bin_problem_comes_last(raster, monkeypatch):
    _small_limit(monkeypatch, 200_000)
    kwargs = {"raster_source": raster, "points": POINTS, "neighborhood": "r1", "search_buffer_m": 100}
    only_bins = ce.run_problems({}, profile=ENFORCE, **kwargs)
    assert [k for k, _ in only_bins] == ["bins"]
    gpu = ce.run_problems({}, profile=ENFORCE, backend="raster_gpu", **kwargs)
    assert [k for k, _ in gpu] == ["backend"]                             # a refused combination first, no noise
    assert [k for k, _ in ce.run_problems({SRC: 0}, profile=ENFORCE, **kwargs)] == [SRC]
    assert ce.run_problems({}, profile=ENFORCE) == []                    # no raster given: only the cheap rules
    assert ce.run_problems({}, profile=BASE, **kwargs) == []
    notice = ce.problem_notice(only_bins)
    assert "exact search is too large" in notice["title"] and "coarser raster" in notice["fix"]


# ====================================================================== the run: a classic run is unchanged
END_KEYS = set(constrained.END_SUMMARY_KEYS)
#: what the library reports about a default "remove" run on its own (a removed tower and the note that says why)
LIBRARY_ONLY = {"removed_towers", "span_notes"}
OLD_KEYS = {"n_towers", "total_tower_cost", "total_angle_penalty_cost", "max_turn_angle_deg", "spans_min_max_avg"}


def test_a_classic_run_calls_find_route_without_a_keyword_and_adds_no_end_setting_key(raster, route_calls):
    out = _path(raster)
    assert route_calls == [{}]
    assert not (END_KEYS - LIBRARY_ONLY) & set(out["summary"])        # no rule, height, bin or search note
    assert set(out["summary"]) - LIBRARY_ONLY == OLD_KEYS             # every old key is still there


@pytest.mark.parametrize("ends", [None, {}, {SRC: None, TGT: None}, {SRC: "", TGT: ""}])
def test_an_empty_ends_dict_is_a_classic_run_and_its_snapshot_has_no_new_key(raster, route_calls, ends):
    _path(raster, ends=ends)
    assert route_calls == [{}]
    snap = constrained.snapshot_run_inputs(
        raster, points=POINTS, profile=BASE, backend="cython", neighborhood="r1", search_buffer_m=100, dem=None,
        dsm=None, ends=ends)
    assert "ends" not in snap and "objective" not in snap
    plain = constrained.snapshot_run_inputs(
        raster, points=POINTS, profile=BASE, backend="cython", neighborhood="r1", search_buffer_m=100, dem=None,
        dsm=None)
    assert snap == plain and set(plain) == {"raster_source", "points", "profile", "backend", "neighborhood",
                                            "search_buffer_m", "dem", "dsm"}


def test_the_service_run_without_ends_is_the_same_route_as_a_run_with_default_ends(raster):
    plain = _path(raster)
    default = _path(raster, ends=ce.default_settings())
    assert plain["line"].equals(default["line"]) and plain["summary"] == default["summary"]
    assert plain["total_cost"] == default["total_cost"]


def test_run_constrained_calls_find_route_with_exactly_the_keywords_it_is_given(raster, route_calls):
    kwargs = {"source": POINTS[0], "target": POINTS[1], "profile": BASE, "neighborhood": "r1", "search_buffer_m": 100}
    constrained.run_constrained(raster, **kwargs)
    constrained.run_constrained(raster, ends=None, **kwargs)
    constrained.run_constrained(raster, ends={}, **kwargs)
    assert route_calls == [{}, {}, {}]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        constrained.run_constrained(raster, ends={SRC: 20.0, TGT: None}, **kwargs)
    assert route_calls[-1] == {SRC: 20.0, TGT: None}
    with pytest.raises(InvalidInputError, match="unknown line-end keyword"):
        constrained.run_constrained(raster, ends={"terminal_span_rule": "keep"}, **kwargs)
    assert len(route_calls) == 4                                          # the refusal came before any finder


def test_the_snapshot_copies_the_heights_and_keeps_only_a_normalised_dict(raster):
    ends = {SRC: "20", TGT: ""}
    snap = constrained.snapshot_run_inputs(
        raster, points=POINTS, profile=BASE, backend="cython", neighborhood="r1", search_buffer_m=None, dem=None,
        dsm=None, ends=ends)
    ends[SRC] = 99.0
    assert snap["ends"] == {SRC: 20.0, TGT: None}
    assert "objective" not in snap


# ====================================================================== the run: heights reach the segments
def test_both_heights_reach_a_one_segment_run_and_the_summary_reports_them(raster, dem, route_calls):
    out = _quiet(_path, raster, profile=WITH_CLEARANCE, dem_path=dem, ends={SRC: 20, TGT: "12"})
    assert route_calls == [{SRC: 20.0, TGT: 12.0}]
    summary = out["summary"]
    assert summary[SRC] == 20.0 and summary[TGT] == 12.0
    assert summary["span_bin_size_m"] < BASE["min_span_m"]                   # exact bins, said so
    assert any("exact span bins" in note for note in summary["search_notes"])


def test_a_waypoint_run_gives_the_source_height_to_the_first_segment_and_the_target_height_to_the_last(
        raster, dem, route_calls):
    points = [(8.0, 8.0), (20.0, 25.0), (40.0, 35.0), (52.0, 52.0)]
    out = _quiet(_path, raster, profile=WITH_CLEARANCE, dem_path=dem, points=points, ends={SRC: 20, TGT: 12})
    assert route_calls == [{SRC: 20.0, TGT: None}, {SRC: None, TGT: None}, {SRC: None, TGT: 12.0}]
    assert out["summary"][SRC] == 20.0 and out["summary"][TGT] == 12.0


def test_a_source_height_alone_still_passes_an_explicit_none_for_every_other_end(raster, dem, route_calls):
    points = [(8.0, 8.0), (30.0, 30.0), (52.0, 52.0)]
    _quiet(_path, raster, profile=WITH_CLEARANCE, dem_path=dem, points=points, ends={SRC: 20})
    assert route_calls == [{SRC: 20.0, TGT: None}, {SRC: None, TGT: None}]


def test_a_bad_height_stops_the_run_before_the_first_search(raster, route_calls):
    with pytest.raises(InvalidInputError, match="too low"):
        _path(raster, ends={SRC: 0})
    assert route_calls == []


def test_a_height_without_a_terrain_model_is_reported_and_the_library_warns_once(raster, route_calls):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _path(raster, profile=WITH_CLEARANCE, ends={SRC: 20})
    assert [str(w.message) for w in caught if issubclass(w.category, PyorpsWarning) and "no effect" in str(w.message)]
    assert out["summary"][SRC] == 20.0 and TGT not in out["summary"]
    assert "span_bin_size_m" not in out["summary"] and "search_notes" not in out["summary"]
    assert ce.notes({SRC: 20}, profile=WITH_CLEARANCE, dem=None)                # the page said so before the run


def test_a_height_changes_the_search_and_the_default_run_is_the_same_as_before(raster, dem):
    plain = _quiet(_path, raster, profile=WITH_CLEARANCE, dem_path=dem)
    again = _quiet(_path, raster, profile=WITH_CLEARANCE, dem_path=dem, ends=None)
    assert plain["line"].equals(again["line"]) and plain["summary"] == again["summary"]
    assert not (END_KEYS - LIBRARY_ONLY) & set(plain["summary"])


def test_the_enforce_rule_is_a_profile_field_only_and_the_summary_reports_it(raster, route_calls):
    out = _path(raster, profile=ENFORCE)
    assert route_calls == [{}]                                              # the rule travels in the profile
    summary = out["summary"]
    assert summary["terminal_span_rule"] == "enforce"
    assert summary["span_bin_size_m"] < BASE["min_span_m"] and summary["search_notes"]
    assert summary["spans_min_max_avg"][0] >= BASE["min_span_m"] - 1e-6        # every span, the end spans too


@pytest.mark.parametrize("rule", ["remove", "remove_if_valid", "keep"])
def test_the_coarse_rules_add_no_bin_size_and_no_search_note(raster, rule):
    summary = _path(raster, profile={**BASE, "terminal_span_rule": rule})["summary"]
    assert "span_bin_size_m" not in summary and "search_notes" not in summary
    assert ("terminal_span_rule" in summary) == (rule != "remove")


# ====================================================================== the result summary
def _result(**fields):
    base = dict(n_towers=3, total_tower_cost=10.0, total_terrain_cost=5.0, total_angle_penalty_cost=1.0,
                min_span_actual_m=7.0, max_span_actual_m=9.0, avg_span_m=8.0, max_turn_angle_deg=2.0,
                tower_type_counts={"suspension": 3})
    return SimpleNamespace(**{**base, **fields})


def test_a_result_without_line_end_facts_has_exactly_the_old_keys():
    assert set(constrained.result_summary(_result())) == {
        "n_towers", "total_tower_cost", "total_terrain_cost_2d", "total_angle_penalty_cost", "spans_min_max_avg",
        "max_turn_angle_deg", "tower_type_counts", "caveat"}
    clean = _result(terminal_span_rule="remove", span_notes=[], removed_towers=[], search_notes=[],
                    source_connection_height_m=None, target_connection_height_m=None, span_bin_size_m=6.0,
                    cost_breakdown={"terrain": 1.0})
    assert not END_KEYS & set(constrained.result_summary(clean))          # the coarse default bin is not reported


def test_a_result_with_line_end_facts_reports_each_of_them():
    removed = [SimpleNamespace(total_cost=40.0), SimpleNamespace(total_cost=60.0)]
    out = constrained.result_summary(_result(
        terminal_span_rule="enforce", span_notes=["a short end span"], removed_towers=removed,
        search_notes=["exact span bins of 5.9 m"], source_connection_height_m=20.0, target_connection_height_m=None,
        span_bin_size_m=5.9, cost_breakdown={"removed_towers": 99.0, "height_premiums": 12.5}))
    assert out["terminal_span_rule"] == "enforce" and out["span_notes"] == ["a short end span"]
    assert out["removed_towers"] == {"count": 2, "cost": 99.0}                # the library's own figure first
    assert out["search_notes"] == ["exact span bins of 5.9 m"] and out["source_connection_height_m"] == 20.0
    assert TGT not in out and out["span_bin_size_m"] == 5.9 and out["height_premiums"] == 12.5
    fallback = constrained.result_summary(_result(removed_towers=removed))
    assert fallback["removed_towers"] == {"count": 2, "cost": 100.0}


def test_the_bin_size_is_reported_only_with_the_search_note_that_says_exact_bins_were_used():
    plain = constrained.result_summary(_result(target_connection_height_m=9.0, span_bin_size_m=4.0,
                                               cost_breakdown={"height_premiums": 0.0}))
    assert "span_bin_size_m" not in plain                                   # a height the search could not use
    assert plain["height_premiums"] == 0.0 and plain[TGT] == 9.0             # a premium of zero is still reported
    exact = constrained.result_summary(_result(span_bin_size_m=4.0, search_notes=["exact bins"]))
    assert exact["span_bin_size_m"] == 4.0


def test_the_segment_summaries_merge_into_one_without_changing_the_old_keys():
    first = {"span_notes": ["short"], "search_notes": ["exact bins"], "removed_towers": {"count": 1, "cost": 10.0},
             SRC: 20.0, "span_bin_size_m": 5.0, "height_premiums": 1.0, "terminal_span_rule": "keep"}
    middle = {"search_notes": ["exact bins"], "span_bin_size_m": 4.0}
    last = {"span_notes": ["long"], "removed_towers": {"count": 2, "cost": 5.0}, TGT: 9.0, "span_bin_size_m": 6.0,
            "height_premiums": 2.5, "terminal_span_rule": "keep", "search_notes": ["other note"]}
    merged = constrained.merge_end_summaries([first, middle, last])
    assert merged == {"terminal_span_rule": "keep", SRC: 20.0, TGT: 9.0,
                      "span_notes": ["Segment 1 of 3: short", "Segment 3 of 3: long"],
                      "search_notes": ["exact bins", "other note"],
                      "removed_towers": {"count": 3, "cost": 15.0}, "span_bin_size_m": 4.0, "height_premiums": 3.5}
    assert constrained.merge_end_summaries([{"span_notes": ["x"]}]) == {"span_notes": ["x"]}
    assert constrained.merge_end_summaries([{}, {}]) == {}


# ====================================================================== the background path
def test_the_worker_hands_the_ends_of_the_snapshot_to_the_run(monkeypatch):
    seen: dict = {}

    def fake(source, **kwargs):
        seen.update(source=source, **kwargs)
        return {"line": "a route"}

    monkeypatch.setattr(constrained, "run_constrained_path", fake)
    snapshot = constrained.snapshot_run_inputs(
        "r.tif", points=[(0, 0), (9, 9)], profile={"name": "p"}, backend="cython", neighborhood="r1",
        search_buffer_m=None, dem=None, dsm=None, ends={SRC: 20, TGT: ""})
    job = jobs.BackgroundJob(kind=constrained.JOB_KIND, label=constrained.JOB_LABEL, params={"run": snapshot})
    constrained.background_run(job)
    assert seen["ends"] == {SRC: 20.0, TGT: None} and seen["cancel"] is job.cancel


def test_a_classic_worker_passes_no_ends_keyword(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(constrained, "run_constrained_path", lambda source, **kw: seen.update(kw) or {"line": 1})
    snapshot = constrained.snapshot_run_inputs(
        "r.tif", points=[(0, 0), (9, 9)], profile={"name": "p"}, backend="cython", neighborhood="r1",
        search_buffer_m=None, dem=None, dsm=None)
    constrained.background_run(jobs.BackgroundJob(kind=constrained.JOB_KIND, label=constrained.JOB_LABEL,
                                                  params={"run": snapshot}))
    assert "ends" not in seen and "objective" not in seen


def test_the_background_run_gives_the_same_route_and_summary_as_the_synchronous_one(raster, dem):
    ends = {SRC: 20, TGT: 12}
    sync = _quiet(_path, raster, profile=WITH_CLEARANCE, dem_path=dem, ends=ends)
    snapshot = constrained.snapshot_run_inputs(
        raster, points=POINTS, profile=WITH_CLEARANCE, backend="cython", neighborhood="r1", search_buffer_m=100,
        dem=dem, dsm=None, ends=ends)
    job = jobs.BackgroundJob(kind=constrained.JOB_KIND, label=constrained.JOB_LABEL, params={"run": snapshot})
    out = constrained.background_run(job)["result"]
    assert out["line"].equals(sync["line"]) and out["summary"] == sync["summary"]
