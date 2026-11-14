"""Round 14, package P2 (objective-core): objective weights, metric layers, a DEM and ``weight_precision`` for
plain routing in the GUI, as services and run-path plumbing (no card yet).

Headless tests. Section A tests ``services/objective`` as pure logic (every problem rule has a case of its own),
section B the routing service (what reaches ``PathFinder``, what a route is priced on), section C the run
callback and the route edit, section D the checks of the plan's [A] assumptions that were made first
(a DEM in another CRS, raster_gpu with a DEM, the cost readout under an objective, the library warnings of a
background run).
"""
from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest
import rasterio
from conftest import SOURCE, TARGET, TEST_CRS, invoke
from rasterio.transform import from_origin
from shapely.geometry import LineString

from pyorps import PathFinder
from pyorps.core.exceptions import InvalidInputError
from pyorps.gui import ids
from pyorps.gui.callbacks import edit as edit_cb
from pyorps.gui.callbacks import objective as objective_cb
from pyorps.gui.callbacks import routes as routes_cb
from pyorps.gui.services import cost, errors, jobs, project_io, routing, routing_job
from pyorps.gui.services import objective as obj

RUN_WIRING = (f"{ids.ROUTING_STATUS}.children", ids.RUN_ROUTING_BTN)
ORIGIN_X, ORIGIN_Y = 500000.0, 5600000.0
SIZE = 200                                     # the raster of the fixtures: 200 x 200 cells of 1 m


# =============================================================== files and layers
def _write_tif(path, array, crs=TEST_CRS, transform=None):
    transform = transform or from_origin(ORIGIN_X, ORIGIN_Y, 1, 1)
    height, width = array.shape
    with rasterio.open(path, "w", driver="GTiff", height=height, width=width, count=1, dtype=str(array.dtype),
                       crs=crs, transform=transform) as dst:
        dst.write(array, 1)
    return str(path)


@pytest.fixture(scope="module")
def grid_files(tmp_path_factory):
    """A DEM with a hill half way between SOURCE and TARGET, a stripe layer across the diagonal, and the
    same DEM written in another coordinate system."""
    folder = tmp_path_factory.mktemp("objective_grid")
    rows, cols = np.mgrid[0:SIZE, 0:SIZE]
    x = ORIGIN_X + cols + 0.5
    y = ORIGIN_Y - rows - 0.5
    dem = (60.0 * np.exp(-(((x - 500100.0) ** 2 + (y - 5599900.0) ** 2) / (2 * 12.0 ** 2)))).astype("float32")
    stripe = np.zeros((SIZE, SIZE), dtype="float32")
    stripe[:, 90:110] = 50.0
    files = {"dem": _write_tif(folder / "dem.tif", dem), "stripe": _write_tif(folder / "stripe.tif", stripe)}

    from rasterio.warp import Resampling, calculate_default_transform, reproject
    transform = from_origin(ORIGIN_X, ORIGIN_Y, 1, 1)
    target, width, height = calculate_default_transform(
        TEST_CRS, "EPSG:4326", SIZE, SIZE, left=ORIGIN_X, bottom=ORIGIN_Y - SIZE, right=ORIGIN_X + SIZE,
        top=ORIGIN_Y)
    warped = np.full((height, width), np.nan, dtype="float32")
    reproject(dem, warped, src_transform=transform, src_crs=TEST_CRS, dst_transform=target, dst_crs="EPSG:4326",
              resampling=Resampling.bilinear, dst_nodata=np.nan)
    files["dem_wgs84"] = _write_tif(folder / "dem_wgs84.tif", warped, crs="EPSG:4326", transform=target)
    files["dem_array"] = dem
    return files


@pytest.fixture()
def layers(state, raster_path, grid_files):
    """The cost raster, the DEM and the stripe layer as loaded raster layers (ids by role)."""
    def add(name, path):
        return state.add_layer(name, "raster", crs=TEST_CRS, meta={"source_path": path})

    return {"cost": add("cost", raster_path), "dem": add("DGM", grid_files["dem"]),
            "stripe": add("landscape", grid_files["stripe"])}


def _paths(layers):
    return {layer.id: layer.meta["source_path"] for layer in layers.values()}


def _settings(layers=None, *, weights=None, extra=None, dem=False, gradient=None, precision=None):
    """An objective settings dict; ``extra`` is a list of ``(name, role_or_source, weight)``."""
    raw: dict = {"weights": {"cost": 1.0, "length": 0.0, "gradient": 0.0, **(weights or {})}}
    if dem:
        raw["dem_layer_id"] = layers["dem"].id
    if extra:
        raw["layers"] = [obj.layer_row(name, layers[src].id if src in (layers or {}) else src, weight)
                         for name, src, weight in extra]
    if gradient:
        raw["gradient"] = gradient
    if precision:
        raw["weight_precision"] = precision
    return raw


def _draft(source=SOURCE, target=TARGET):
    return {"sources": [{"lat": 0, "lng": 0, "x": source[0], "y": source[1]}],
            "targets": [{"lat": 0, "lng": 0, "x": target[0], "y": target[1]}], "waypoints": []}


def _run_args(raster_id, *, algorithm="dijkstra", graph_api="cython", k=1, buffer=80, draft=None):
    return (1, False, draft or _draft(), raster_id, algorithm, graph_api, k, buffer, False, True, False, False,
            1.0, 100, 0, None, [], [])


def _run(app, raster_id, **kwargs):
    return invoke(app, RUN_WIRING, *_run_args(raster_id, **kwargs), triggered=[f"{ids.RUN_ROUTING_BTN}.n_clicks"])


def _spy(monkeypatch, owner, name):
    calls: list[tuple] = []
    original = getattr(owner, name)

    def wrapper(self, *args, **kwargs):
        calls.append((args, kwargs))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(owner, name, wrapper)
    return calls


def _installed(name):
    return importlib.util.find_spec(name) is not None


def _gpu_works() -> bool:
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001  # pylint: disable=broad-exception-caught  # no CUDA here: the GPU test skips
        return False


#: what the checks below treat as the loaded raster layers
LOADED = {"R1", "DEM", "M1"}


def _problems(raw, **kwargs):
    kwargs.setdefault("graph_api", "cython")
    kwargs.setdefault("algorithm", "dijkstra")
    kwargs.setdefault("raster_ids", LOADED)
    return obj.problems(raw, **kwargs)


def _dem_settings(**changes):
    raw = {"weights": {"cost": 1.0}, "dem_layer_id": "DEM"}
    raw.update(changes)
    return raw


# ============================================================ A: normalise, classic, weights
def test_defaults_are_classic_and_normalise_is_idempotent():
    assert obj.normalise_objective_settings({}) == obj.OBJECTIVE_DEFAULTS
    assert obj.normalise_objective_settings(None) == obj.OBJECTIVE_DEFAULTS
    assert obj.normalise_objective_settings("not a dict") == obj.OBJECTIVE_DEFAULTS
    assert obj.is_classic({})
    once = obj.normalise_objective_settings({"weights": {"cost": "2", "length": ""}, "bogus": 1,
                                             "layers": [{"name": " a ", "source": "R1", "weight": None}, 7]})
    assert obj.normalise_objective_settings(once) == once
    assert "bogus" not in once
    assert once["weights"] == {"cost": 2.0, "length": 0.0, "gradient": 0.0}      # text -> float, blank -> default
    assert once["layers"] == [{"name": "a", "source": "R1", "weight": 1.0, "hard_max": None, "hard_min": None}]
    assert json.loads(json.dumps(once)) == once                                  # JSON safe


def test_a_blank_gradient_choice_means_none_and_a_blank_number_its_default():
    s = obj.normalise_objective_settings({"gradient": {"additive": "None", "multiplier": "", "mult_scale": "",
                                                       "bin_width_pct": None, "max_gradient_pct": " 30 "}})
    assert s["gradient"]["additive"] is None and s["gradient"]["multiplier"] is None
    assert s["gradient"]["mult_scale"] is None and s["gradient"]["bin_width_pct"] == 0.25
    assert s["gradient"]["max_gradient_pct"] == 30.0


def test_values_that_are_not_numbers_survive_normalising_so_problems_can_name_them():
    s = obj.normalise_objective_settings({"weights": {"cost": "abc"}})
    assert s["weights"]["cost"] == "abc"
    assert not obj.is_classic(s)
    assert _problems(s)[0].key == "weights.cost"


@pytest.mark.parametrize("raw,classic", [
    ({}, True),
    ({"weights": {"cost": 2.5}}, True),                                    # a cost weight alone changes no route
    ({"weights": {"cost": 2e6}}, False),                                   # above its limit: named, not ignored
    ({"weights": {"cost": 1, "length": 0.2}}, False),
    ({"weights": {"cost": 1, "gradient": 1}}, False),
    ({"weights": {"cost": 0, "length": 1}}, False),
    ({"weights": {"cost": 0}}, False),                                     # nothing to minimise: a problem
    ({"dem_layer_id": "DEM"}, False),
    ({"layers": [{"name": "a", "source": "M1"}]}, False),
    ({"weight_precision": "float32"}, False),
    ({"gradient": {"multiplier": "exponential"}}, False),
    ({"gradient": {"max_gradient_pct": 30}}, False),
    ({"gradient": {"additive": "quadratic"}}, True),                       # a shape without weight or DEM is unused
])
def test_is_classic(raw, classic):
    assert obj.is_classic(raw) is classic
    assert (obj.run_kwargs(raw, layer_path={"M1": "m.tif", "DEM": "d.tif"}) == {}) is classic


def test_weights_drop_zero_weights_and_keep_layer_weights_in_order():
    raw = {"weights": {"cost": 0, "length": 0.2, "gradient": 3}, "layers": [
        obj.layer_row("landscape", "M1", 0.5), obj.layer_row("barrier", "M1", 0)]}
    assert obj.weights(raw) == {"length": 0.2, "gradient": 3.0, "landscape": 0.5}
    assert list(obj.weights(raw)) == ["length", "gradient", "landscape"]


def test_presets_write_the_planned_weights():
    assert obj.preset_weights("classic") == {"cost": 1.0, "length": 0.0, "gradient": 0.0}
    assert obj.preset_weights("shortest") == {"cost": 0.0, "length": 1.0, "gradient": 0.0}
    assert obj.preset_weights("gentlest") == {"cost": 0.0, "length": 0.01, "gradient": 1.0}
    assert obj.weights({"weights": obj.preset_weights("gentlest")}) == {"length": 0.01, "gradient": 1.0}
    kept = obj.apply_preset({"dem_layer_id": "DEM", "weights": {"cost": 5}}, "shortest")
    assert kept["dem_layer_id"] == "DEM" and kept["weights"]["length"] == 1.0
    with pytest.raises(KeyError):
        obj.preset_weights("nope")


# ============================================================ A: the slope response
def test_gradient_options_carry_the_parameters_of_the_chosen_shapes_only():
    plain = obj.gradient_options({})
    assert plain["additive"] is None and plain["multiplier"] is None
    assert plain["additive_params"] == {} and plain["multiplier_params"] == {}
    assert (plain["bin_width_pct"], plain["s_max_pct"]) == (0.25, 200.0)
    exp = obj.gradient_options({"gradient": {"additive": "exponential", "additive_scale": 3,
                                             "multiplier": "exponential"}})
    assert exp["additive_params"] == {"scale": 3.0}
    assert exp["multiplier_params"] == {"scale": 2.0}                       # the model default, written out
    quad = obj.gradient_options({"gradient": {"additive": "quadratic", "additive_scale": 9}})
    assert quad["additive_params"] == {}                                    # only exponential reads a scale
    assert obj.gradient_options({"gradient": {"multiplier": "energy"}})["multiplier_params"] == {"scale": 3.0}
    assert obj.gradient_options({"gradient": {"multiplier": "squared", "mult_scale": 100}})[
        "multiplier_params"] == {"scale": 100.0}
    assert obj.gradient_options({"gradient": {"multiplier": "power", "mult_scale": 9}})["multiplier_params"] == {}
    sig = obj.gradient_options({"gradient": {"multiplier": "sigmoid"}})["multiplier_params"]
    assert sig == {"center_deg": 15.0, "steepness": 0.2, "max_penalty": 1000.0}


def test_gradient_options_build_the_library_options_for_every_shape():
    from pyorps.core.objective import GradientOptions, Objective
    for additive in (None, *obj.ADDITIVE_SHAPES):
        for multiplier in (None, *obj.MULTIPLIER_MODELS):
            raw = {"weights": {"cost": 1, "gradient": 0.5}, "dem_layer_id": "DEM",
                   "gradient": {"additive": additive, "multiplier": multiplier, "max_gradient_pct": 40}}
            options = GradientOptions.from_dict(obj.gradient_options(raw))
            objective = Objective(obj.weights(raw), options)
            assert options.max_gradient_pct == 40.0 and options.additive == (additive or "linear")
            assert objective["gradient"] == 0.5


def test_applicable_gradient_keys_follow_the_selected_shape():
    base = {"max_gradient_pct", "additive", "multiplier", "bin_width_pct", "s_max_pct"}
    assert set(obj.applicable_gradient_keys({})) == base
    assert "additive_scale" in obj.applicable_gradient_keys({"additive": "exponential"})
    assert "additive_scale" not in obj.applicable_gradient_keys({"additive": "linear"})
    assert "mult_scale" in obj.applicable_gradient_keys({"multiplier": "squared"})
    assert "mult_scale" not in obj.applicable_gradient_keys({"multiplier": "power"})
    sigmoid = obj.applicable_gradient_keys({"multiplier": "sigmoid"})
    assert {"sig_center_deg", "sig_steepness", "sig_max_penalty"} <= set(sigmoid)
    assert obj.mult_scale_default("exponential") == 2.0 and obj.mult_scale_default("power") is None


def test_the_tables_agree_with_the_library():
    from pyorps.core.objective import ADDITIVE_MODELS, MULTIPLIER_MODELS, RESERVED_LAYERS, GradientOptions
    from pyorps.graph.path_finder import PathFinder as Finder
    assert obj.GRADIENT_BACKENDS == Finder.GRADIENT_BACKENDS
    assert obj.FLOAT_BACKENDS == Finder.FLOAT_BACKENDS
    assert obj.DEM_FALLBACK_BACKENDS == Finder.DEM_FALLBACK_BACKENDS
    assert set(obj.ADDITIVE_SHAPES) == set(ADDITIVE_MODELS)
    assert set(obj.MULTIPLIER_MODELS) == set(MULTIPLIER_MODELS)
    assert obj.RESERVED_LAYER_NAMES == ("cost", *RESERVED_LAYERS)
    # the default scales the GUI writes out are the ones the library uses without a parameter
    slopes = np.array([0.0, 20.0, 60.0, 120.0])
    for model, scale in obj.MULT_SCALE_DEFAULTS.items():
        function = MULTIPLIER_MODELS[model]
        assert np.allclose(function(slopes, {}), function(slopes, {"scale": scale}))
    defaults = GradientOptions()
    assert (defaults.bin_width_pct, defaults.s_max_pct) == (0.25, 200.0)
    gradient_defaults = obj.OBJECTIVE_DEFAULTS["gradient"]
    assert (gradient_defaults["bin_width_pct"], gradient_defaults["s_max_pct"]) == (0.25, 200.0)


def test_every_numeric_setting_has_a_spec_with_the_defaults_of_the_dict():
    for key in obj.OBJECTIVE_DEFAULTS["gradient"]:
        if key in ("additive", "multiplier"):
            continue
        assert f"gradient.{key}" in obj.FIELD_SPECS, key
        assert obj.FIELD_SPECS[f"gradient.{key}"].default == obj.OBJECTIVE_DEFAULTS["gradient"][key]
    for key in obj.WEIGHT_KEYS:
        assert obj.FIELD_SPECS[f"weights.{key}"].default == obj.OBJECTIVE_DEFAULTS["weights"][key]
    assert obj.label_with_unit(obj.FIELD_SPECS["weights.cost"]) == "Cost weight (x raster value)"


# ============================================================ A: what reaches the library
def test_run_kwargs_for_weights_layers_derived_slope_and_dem():
    raw = {"weights": {"cost": 1, "length": 0.2}, "dem_layer_id": "DEM",
           "layers": [{"name": "landscape", "source": "M1", "weight": 0.5, "hard_max": 40},
                      {"name": "steep", "source": obj.SOURCE_DERIVE_SLOPE, "weight": 0.1, "hard_max": 30,
                       "hard_min": 1}],
           "gradient": {"max_gradient_pct": 35, "multiplier": "exponential"}}
    out = obj.run_kwargs(raw, layer_path={"M1": "m.tif", "DEM": "d.tif"}, graph_api="cython")
    kw = out["objective_kwargs"]
    assert kw["objective"] == {"cost": 1.0, "length": 0.2, "landscape": 0.5, "steep": 0.1}
    assert kw["dem"] == "d.tif" and kw["weight_precision"] == "uint16"
    assert kw["metric_layers"] == {"landscape": {"source": "m.tif", "hard_max": 40.0},
                                   "steep": {"derive": "slope_from_dem", "hard_max": 30.0, "hard_min": 1.0}}
    assert kw["gradient_options"]["max_gradient_pct"] == 35.0
    assert kw["gradient_options"]["multiplier"] == "exponential"
    assert out["objective_spec"] == obj.normalise_objective_settings(raw)
    assert out["objective_key"] == obj.objective_key(raw) and len(out["objective_key"]) == 64
    assert out["force_direct"] is False                # extra layers on cython: a retained session (round 16)
    assert set(out) == {"objective_kwargs", "objective_spec", "objective_key", "force_direct"}


def test_run_kwargs_without_dem_or_layers_passes_neither_dem_nor_gradient_options():
    kw = obj.run_kwargs({"weights": {"cost": 1, "length": 0.5}}, layer_path={})["objective_kwargs"]
    assert set(kw) == {"objective", "weight_precision"}


def test_run_kwargs_takes_a_callable_and_names_a_missing_layer():
    raw = {"weights": {"cost": 1}, "layers": [obj.layer_row("a", "M1")]}
    assert obj.run_kwargs(raw, layer_path=lambda layer_id: f"{layer_id}.tif")["objective_kwargs"][
        "metric_layers"] == {"a": {"source": "M1.tif"}}
    with pytest.raises(ValueError, match="metric layer 'a'"):
        obj.run_kwargs(raw, layer_path={})
    with pytest.raises(ValueError, match="DEM"):
        obj.run_kwargs({"dem_layer_id": "DEM"}, layer_path={})


def test_the_key_ignores_settings_that_are_not_read_and_follows_the_ones_that_are():
    base = {"weights": {"cost": 1, "length": 0.2}, "dem_layer_id": "DEM"}
    key = obj.objective_key(base)
    assert key == obj.objective_key(obj.normalise_objective_settings(base))
    unused = {**base, "gradient": {"additive_scale": 7, "sig_center_deg": 40, "mult_scale": 5}}
    assert obj.objective_key(unused) == key                                  # no exponential shape, no sigmoid
    assert obj.objective_key({**base, "weights": {"cost": 1, "length": 0.3}}) != key
    assert obj.objective_key({**base, "gradient": {"max_gradient_pct": 30}}) != key
    assert obj.objective_key({**base, "dem_layer_id": "OTHER"}) != key
    assert obj.objective_key({**base, "weight_precision": "float32"}) != key
    assert obj.objective_key({**base, "layers": [obj.layer_row("a", "M1")]}) != key
    exponential = {**base, "gradient": {"additive": "exponential", "additive_scale": 3}}
    assert obj.objective_key(exponential) != obj.objective_key(
        {**base, "gradient": {"additive": "exponential", "additive_scale": 4}})
    # without a DEM the slope response is not read at all
    no_dem = {"weights": {"cost": 1, "length": 0.2}}
    assert obj.objective_key(no_dem) == obj.objective_key({**no_dem, "gradient": {"max_gradient_pct": None,
                                                                                    "bin_width_pct": 1.0}})


@pytest.mark.parametrize("raw,graph_api,expected", [
    ({"weights": {"cost": 1, "length": 1}}, "cython", False),
    ({"weights": {"cost": 1, "length": 1}, "weight_precision": "float32"}, "networkx", True),
    ({"weights": {"cost": 1}, "layers": [{"name": "a", "source": "M1"}]}, "cython", False),     # round 16: proven
    ({"weights": {"cost": 1}, "layers": [{"name": "a", "source": "M1"}]}, "raster_gpu", True),    # not run (no GPU)
    ({"weights": {"cost": 1}, "layers": [{"name": "a", "source": "M1"}], "dem_layer_id": "DEM"}, "cython", False),
    ({"weights": {"cost": 1}, "dem_layer_id": "DEM"}, "cython", False),
    ({"weights": {"cost": 1}, "dem_layer_id": "DEM"}, "raster_gpu", True),
    ({"weights": {"cost": 1}, "dem_layer_id": "DEM"}, "networkx", False),
    ({}, "raster_gpu", False),
])
def test_force_direct(raw, graph_api, expected):
    assert obj.force_direct(raw, graph_api) is expected


def test_stack_bands_count_the_full_raster_bands_beyond_the_cost_raster():
    assert obj.stack_bands({}) == 0
    assert obj.stack_bands({"weights": {"cost": 1}, "dem_layer_id": "DEM"}) == 0           # zero-copy pass-through
    assert obj.stack_bands({"weights": {"cost": 1, "length": 1}}) == 1
    assert obj.stack_bands({"weights": {"cost": 1}, "layers": [obj.layer_row("a", "M1")]}) == 2
    derived = {"weights": {"cost": 1}, "dem_layer_id": "DEM",
               "layers": [obj.layer_row("s", obj.SOURCE_DERIVE_SLOPE)]}
    assert obj.stack_bands(derived) == 4


def test_summary_lines():
    assert obj.summary({}) == "Classic cost routing (no objective)."
    text = obj.summary(
        {"weights": {"cost": 1, "length": 0.2}, "dem_layer_id": "DEM",
         "layers": [obj.layer_row("landscape", "M1", 0.5)],
         "gradient": {"max_gradient_pct": 30}, "weight_precision": "float32"},
        "cython", "dijkstra", layer_names={"DEM": "DGM"})
    assert text == ("Objective: cost 1 + length 0.2 + landscape 0.5; DEM 'DGM'; slope limit 30 %; "
                    "32-bit float weights; cython runs Dijkstra only.")
    assert "cython runs" not in obj.summary(_dem_settings(), "networkx")
    assert obj.summary({"weights": {"cost": 0, "gradient": 2}, "dem_layer_id": "DEM"}, "raster_gpu").startswith(
        "Objective: slope 2;")


# ============================================================ A: texts for the controls
def test_precision_support_says_why_it_cannot_apply():
    classic = obj.precision_support({}, "networkx")
    assert not classic.enabled and "Only with an objective" in classic.reason
    cython = obj.precision_support({"weights": {"cost": 1, "length": 1}}, "cython")
    assert not cython.enabled and "16-bit kernels" in cython.reason and "networkx" in cython.reason
    assert obj.precision_support({"weights": {"cost": 1, "length": 1}}, "networkx") == obj.Support(True, "")
    assert obj.precision_support({"weights": {"cost": 1, "length": 1}}, "raster_gpu").enabled
    fim_dem = obj.precision_support(_dem_settings(), "raster_fim")
    assert not fim_dem.enabled and "raster_fim" in fim_dem.reason
    # judged without the precision itself: choosing 32-bit does not switch the choice off
    assert obj.precision_support({"weights": {"cost": 1, "length": 1}, "weight_precision": "float32"},
                                 "networkx").enabled


def test_terrain_and_slope_support_texts():
    assert obj.terrain_support("cython", "dijkstra") == obj.Support(True, "")
    assert obj.terrain_support("networkx", "dijkstra").enabled
    assert obj.terrain_support("raster_gpu", "delta-stepping").enabled
    delta = obj.terrain_support("cython", "delta-stepping")
    assert not delta.enabled and "Algorithm field" in delta.reason and "Dijkstra" in delta.reason
    assert not obj.terrain_support("cython", "delta-stepping-circular").enabled
    fim = obj.terrain_support("raster_fim", "fim")
    assert not fim.enabled and "Backend field" in fim.reason
    assert not obj.slope_support({}).enabled and obj.slope_support({}).reason == "Needs a terrain model (DEM)"
    assert obj.slope_support({"dem_layer_id": "DEM"}).enabled


def test_layer_name_check_and_row_helpers():
    assert obj.check_layer_name("landscape") is None
    assert obj.check_layer_name("a_1") is None
    assert "must not be empty" in obj.check_layer_name("  ")
    assert "not allowed" in obj.check_layer_name("1abc")
    assert "not allowed" in obj.check_layer_name("a b")
    assert "not allowed" in obj.check_layer_name("a" * 33)
    for name in ("cost", "length", "gradient", "smoothness", "tower"):
        assert "reserved" in obj.check_layer_name(name)
    assert "already uses" in obj.check_layer_name("a", ["b", "a"])
    assert obj.layer_row(" a ", "", "2")["weight"] == 2.0
    assert obj.layer_row("a", "M1")["source"] == "M1"
    assert obj.source_label(obj.SOURCE_DERIVE_SLOPE) == "Slope from the DEM (per cell, %)"
    assert obj.source_label("M1", {"M1": "landscape"}) == "landscape"
    assert obj.source_label("gone", {}) == "(layer no longer loaded)"
    assert obj.source_label(None) == "(no source)"


# ============================================================ A: every problem rule
def test_classic_settings_have_no_problem_whatever_the_backend():
    for graph_api, algorithm in (("cython", "delta-stepping"), ("raster_fim", "fim"), ("networkx", "astar")):
        assert _problems({}, graph_api=graph_api, algorithm=algorithm) == []


@pytest.mark.parametrize("raw,key,fragments", [
    # ----- weights
    ({"weights": {"cost": -1}}, "weights.cost", ("Cost weight", "-1", "from 0 to 1,000,000", "x raster value")),
    ({"weights": {"cost": 2e6}}, "weights.cost", ("2,000,000", "from 0 to 1,000,000")),
    ({"weights": {"cost": "abc"}}, "weights.cost", ("'abc'", "number")),
    ({"weights": {"cost": float("nan")}}, "weights.cost", ("finite number",)),
    ({"weights": {"cost": float("inf")}}, "weights.cost", ("finite number",)),
    ({"weights": {"cost": 1, "length": -0.1}}, "weights.length", ("Length weight", "per m")),
    ({"weights": {"cost": 1, "length": 1e7}}, "weights.length", ("from 0 to 1,000,000",)),
    ({"weights": {"cost": 0, "length": 0, "gradient": 0}}, "weights", ("all 0", "at least one")),
    ({"weights": {"cost": 0}, "layers": [{"name": "a", "source": "M1", "weight": 0}]}, "weights", ("all 0",)),
    # ----- slope settings need a DEM
    ({"weights": {"cost": 1, "gradient": 2}}, "dem_layer_id", ("slope weight is 2", "choose a terrain model")),
    ({"weights": {"cost": 1}, "gradient": {"multiplier": "exponential"}}, "dem_layer_id", ("multiplier",)),
    ({"weights": {"cost": 1}, "gradient": {"max_gradient_pct": 30}}, "dem_layer_id", ("steepest allowed slope",)),
    # ----- layers
    ({"layers": [obj.layer_row("", "M1")]}, "layers.0.name", ("must not be empty",)),
    ({"layers": [obj.layer_row("a b", "M1")]}, "layers.0.name", ("not allowed",)),
    ({"layers": [obj.layer_row("cost", "M1")]}, "layers.0.name", ("reserved",)),
    ({"layers": [obj.layer_row("length", "M1")]}, "layers.0.name", ("reserved",)),
    ({"layers": [obj.layer_row("a", "M1"), obj.layer_row("a", "M1")]}, "layers.1.name", ("already uses",)),
    ({"layers": [{"name": "a", "source": None, "weight": 1}]}, "layers.0.source", ("is empty",)),
    ({"layers": [obj.layer_row("a", "GONE")]}, "layers.0.source", ("not loaded any more",)),
    ({"layers": [obj.layer_row("a", obj.SOURCE_DERIVE_SLOPE)]}, "layers.0.source", ("needs a terrain model",)),
    ({"layers": [obj.layer_row("a", "M1", -1)]}, "layers.0.weight", ("Layer 'a'", "weight", "-1")),
    ({"layers": [obj.layer_row("a", "M1", 2e6)]}, "layers.0.weight", ("2,000,000",)),
    ({"layers": [obj.layer_row("a", "M1", 1, hard_max="x")]}, "layers.0.hard_max", ("'x'",)),
    ({"layers": [obj.layer_row("a", "M1", 1, hard_max=5, hard_min=10)]}, "layers.0.hard_max", ("every cell",)),
    ({"layers": [obj.layer_row("a", "M1", 1, hard_min=float("inf"))]}, "layers.0.hard_min", ("finite",)),
    # ----- the DEM layer
    ({"weights": {"cost": 1}, "dem_layer_id": "GONE"}, "dem_layer_id", ("not loaded any more",)),
    # ----- slope response (only with a DEM)
    (_dem_settings(gradient={"additive": "cubic"}), "gradient.additive", ("'cubic'", "linear, quadratic")),
    (_dem_settings(gradient={"multiplier": "huge"}), "gradient.multiplier", ("'huge'", "exponential")),
    (_dem_settings(gradient={"max_gradient_pct": 0}), "gradient.max_gradient_pct", ("above 0 up to 1,000", "%")),
    (_dem_settings(gradient={"max_gradient_pct": 2000}), "gradient.max_gradient_pct", ("2,000",)),
    (_dem_settings(gradient={"max_gradient_pct": "steep"}), "gradient.max_gradient_pct", ("'steep'",)),
    (_dem_settings(gradient={"additive": "exponential", "additive_scale": 0}), "gradient.additive_scale",
     ("above 0 up to 100",)),
    (_dem_settings(gradient={"multiplier": "squared", "mult_scale": -4}), "gradient.mult_scale", ("-4",)),
    (_dem_settings(gradient={"multiplier": "energy", "mult_scale": 1e5}), "gradient.mult_scale", ("100,000",)),
    (_dem_settings(gradient={"multiplier": "sigmoid", "sig_center_deg": 120}), "gradient.sig_center_deg",
     ("from 0 to 90", "deg")),
    (_dem_settings(gradient={"multiplier": "sigmoid", "sig_steepness": 0}), "gradient.sig_steepness", ("1/deg",)),
    (_dem_settings(gradient={"multiplier": "sigmoid", "sig_max_penalty": 0.5}), "gradient.sig_max_penalty",
     ("from 1 to 1,000,000",)),
    (_dem_settings(gradient={"bin_width_pct": 0.001}), "gradient.bin_width_pct", ("from 0.01 to 10",)),
    (_dem_settings(gradient={"bin_width_pct": 20}), "gradient.bin_width_pct", ("20",)),
    (_dem_settings(gradient={"bin_width_pct": 1, "s_max_pct": 0.5}), "gradient.s_max_pct", ("from 1 to 1,000",)),
    (_dem_settings(gradient={"s_max_pct": 5000}), "gradient.s_max_pct", ("5,000",)),
    # ----- weight precision
    ({"weights": {"cost": 1, "length": 1}, "weight_precision": "float64"}, "weight_precision", ("'float64'",)),
    ({"weight_precision": "float32"}, "weight_precision", ("Only with an objective",)),
])
def test_problem_rules(raw, key, fragments):
    found = _problems(raw, graph_api="networkx")
    assert found, f"no problem for {raw}"
    problem = found[0]
    assert problem.key == key, found
    sentence = f"{problem.label} {problem.message}"
    for fragment in fragments:
        assert fragment in sentence, (fragment, sentence)
    assert isinstance(problem, obj.ObjectiveProblem)


def test_a_problem_names_the_setting_the_value_the_range_and_the_unit():
    problem = _problems({"weights": {"cost": 2e6}})[0]
    assert problem == obj.ObjectiveProblem(
        "weights.cost", "Cost weight", problem.message, 2e6, "from 0 to 1,000,000", "x raster value")
    assert problem.message == "is 2,000,000, but must be from 0 to 1,000,000 (x raster value)"
    assert obj.describe_range(obj.FIELD_SPECS["gradient.max_gradient_pct"]) == "above 0 up to 1,000"
    assert obj.describe_range(obj.FIELD_SPECS["gradient.sig_max_penalty"]) == "from 1 to 1,000,000"
    assert obj.format_number(0.25) == "0.25" and obj.format_number(1e6) == "1,000,000"


def test_weights_and_gradient_are_checked_in_the_order_of_the_card():
    raw = {"weights": {"cost": -1, "length": -1}, "layers": [obj.layer_row("a b", "GONE", -2)],
           "dem_layer_id": "GONE"}
    keys = [p.key for p in _problems(raw)]
    assert keys.index("weights.cost") < keys.index("weights.length") < keys.index("layers.0.name")
    assert keys.index("layers.0.name") < keys.index("layers.0.source") < keys.index("dem_layer_id")


def test_a_control_that_is_off_is_never_checked():
    # no DEM: the slope weight and every slope setting are off, whatever they hold
    raw = {"weights": {"cost": 1, "length": 0.5, "gradient": "junk"},
           "gradient": {"additive": "cubic", "bin_width_pct": -3, "s_max_pct": "x", "sig_center_deg": 999}}
    assert _problems(raw) == []
    # a shape that is not chosen: its parameters are off
    raw = _dem_settings(gradient={"additive_scale": -5, "mult_scale": -5, "sig_center_deg": 999,
                                  "sig_steepness": -1, "sig_max_penalty": -1})
    assert _problems(raw) == []
    # a layer weight of exactly 0 is fine (the layer still forbids by its bounds)
    assert _problems({"weights": {"cost": 1}, "layers": [obj.layer_row("a", "M1", 0, hard_max=5)]}) == []


def test_every_valid_combination_passes():
    full = {"weights": {"cost": 1, "length": 0.2, "gradient": 0.5},
            "layers": [obj.layer_row("landscape", "M1", 0.5, hard_max=40, hard_min=0),
                       obj.layer_row("steep", obj.SOURCE_DERIVE_SLOPE, 0.1, hard_max=30)],
            "dem_layer_id": "DEM",
            "gradient": {"max_gradient_pct": 35, "additive": "exponential", "additive_scale": 2,
                         "multiplier": "sigmoid", "bin_width_pct": 0.5, "s_max_pct": 300}}
    assert _problems(full, raster_crs="EPSG:32632", dem_crs="EPSG:32632") == []
    assert _problems(full, graph_api="networkx", algorithm="dijkstra") == []
    assert _problems(full, graph_api="raster_gpu", algorithm="delta-stepping") == []


@pytest.mark.parametrize("graph_api,algorithm,key,fragment", [
    ("raster_fim", "fim", "graph_api", "raster_fim"),
    ("cugraph", "dijkstra", "graph_api", "cugraph"),
    ("cython", "delta-stepping", "algorithm", "delta-stepping has no slope terms"),
    ("cython", "delta-stepping-circular", "algorithm", "Dijkstra"),
])
def test_a_dem_is_refused_where_the_backend_or_algorithm_cannot_take_it(graph_api, algorithm, key, fragment):
    found = _problems(_dem_settings(), graph_api=graph_api, algorithm=algorithm)
    assert [p.key for p in found] == [key]
    assert fragment in f"{found[0].label} {found[0].message}"


@pytest.mark.parametrize("graph_api,algorithm", [
    ("cython", "dijkstra"), ("networkx", "bidirectional_dijkstra"), ("networkit", "dijkstra"),
    ("igraph", "bellman_ford"), ("rustworkx", "astar"), ("raster_gpu", "delta-stepping"),
    ("raster_gpu", "dijkstra")])
def test_a_dem_is_accepted_on_every_backend_that_applies_slope_terms(graph_api, algorithm):
    assert _problems(_dem_settings(), graph_api=graph_api, algorithm=algorithm) == []


@pytest.mark.parametrize("graph_api,ok", [
    ("networkx", True), ("networkit", True), ("igraph", True), ("rustworkx", True), ("raster_gpu", True),
    ("raster_fim", True), ("cython", False)])
def test_float32_is_offered_only_on_float_backends(graph_api, ok):
    raw = {"weights": {"cost": 1, "length": 0.5}, "weight_precision": "float32"}
    found = _problems(raw, graph_api=graph_api, algorithm="dijkstra")
    assert (found == []) is ok
    if not ok:
        assert found[0].key == "weight_precision" and "16-bit kernels" in found[0].message


def test_float32_with_a_dem_on_raster_fim_names_the_backend_not_the_precision():
    raw = _dem_settings(weight_precision="float32")
    assert [p.key for p in _problems(raw, graph_api="raster_fim", algorithm="fim")] == ["graph_api"]


def test_a_dem_in_another_coordinate_system_is_refused():
    found = _problems(_dem_settings(), raster_crs="EPSG:32632", dem_crs="EPSG:4326")
    assert [p.key for p in found] == ["dem_layer_id"]
    assert "EPSG:4326" in found[0].message and "reproject" in found[0].message
    assert _problems(_dem_settings(), raster_crs="EPSG:32632", dem_crs="epsg:32632") == []
    assert _problems(_dem_settings(), raster_crs="EPSG:32632", dem_crs=None) == []         # unknown: not judged
    assert _problems(_dem_settings(), raster_crs=None, dem_crs="EPSG:4326") == []
    assert obj.same_crs("EPSG:32632", "EPSG:32632") is True
    assert obj.same_crs("EPSG:32632", "EPSG:25832") is False
    assert obj.same_crs(None, "EPSG:32632") is None


def test_the_library_is_the_last_safety_net(monkeypatch):
    # a value the rules above let through but the library rejects is named, it does not crash the run
    import pyorps.core.objective as library
    from pyorps.core.exceptions import ObjectiveError
    plain = obj.normalise_objective_settings({"weights": {"cost": 1, "length": 1}})
    assert obj._library_problems(plain) == []

    def refuse(*_args, **_kwargs):
        raise ObjectiveError("max_gradient_pct must be > 0, got -1")

    monkeypatch.setattr(library, "Objective", refuse)
    found = obj._library_problems(plain)
    assert [p.key for p in found] == ["gradient.max_gradient_pct"]
    assert "not usable" in found[0].message and "max_gradient_pct" in found[0].message
    monkeypatch.setattr(library, "Objective", lambda *_a, **_k: (_ for _ in ()).throw(ObjectiveError("odd")))
    assert [p.key for p in obj._library_problems(plain)] == ["weights"]


def test_dem_coverage():
    bounds = (500000.0, 5599800.0, 500200.0, 5600000.0)
    inside = [(500020.0, 5599980.0), (500180.0, 5599820.0)]
    assert obj.dem_coverage(bounds, inside, buffer_m=0.0) == (None, None)
    problem, warning = obj.dem_coverage(bounds, [*inside, (500300.0, 5599900.0)])
    assert warning is None and problem.key == "dem_layer_id" and "1 of 3 routing point(s)" in problem.message
    assert "(500,300.0, 5,599,900.0)" in problem.message
    problem, warning = obj.dem_coverage(bounds, inside, buffer_m=50.0)               # window leaves the DEM
    assert problem is None and "not the whole search window" in warning
    # the window is cut to the raster first: a DEM as large as the raster covers it
    assert obj.dem_coverage(bounds, inside, raster_bounds=bounds, buffer_m=50.0) == (None, None)
    assert obj.dem_coverage(bounds, [], buffer_m=5.0) == (None, None)


def test_layer_problems_and_missing_files(tmp_path):
    spec = {"weights": {"cost": 1}, "dem_layer_id": "DEM", "layers": [obj.layer_row("a", "M1")]}
    assert obj.layer_problems(spec, {"DEM", "M1"}) == []
    assert [p.key for p in obj.layer_problems(spec, {"M1"})] == ["dem_layer_id"]
    assert [p.key for p in obj.layer_problems(spec, {"DEM"})] == ["layers.0.source"]
    here = tmp_path / "here.tif"
    here.write_bytes(b"x")
    assert obj.missing_files(spec, {"DEM": str(here), "M1": str(here)}) == []
    gone = obj.missing_files(spec, {"DEM": str(here), "M1": str(tmp_path / "gone.tif")})
    assert [p.key for p in gone] == ["layers.0.source"] and "no longer exists" in gone[0].message
    assert [p.key for p in obj.missing_files(spec, lambda layer_id: None)] == ["dem_layer_id", "layers.0.source"]


def test_the_service_imports_nothing_of_dash_ids_or_the_layout():
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(obj))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "a relative import (ids, layout, a sibling service) is not allowed here"
            modules.add(node.module or "")
    roots = {name.split(".")[0] for name in modules}
    assert not {"dash", "dash_bootstrap_components", "dash_leaflet", "dash_ag_grid"} & roots
    assert not [name for name in modules if name.startswith("pyorps.gui")]


# ============================================================ B: the routing service
def _route(raster_path, layers, settings, *, graph_api="cython", algorithm="dijkstra", buffer=80, **extra):
    """``run_routing`` with the keywords the run callback would add for ``settings``."""
    kwargs = obj.run_kwargs(settings, layer_path=_paths(layers), graph_api=graph_api)
    return routing.run_routing(
        raster_path, sources=[SOURCE], targets=[TARGET], algorithm=algorithm, graph_api=graph_api,
        neighborhood="r1", search_buffer_m=buffer, **kwargs, **extra)


def _plain_cost(raster_path, line, buffer=80):
    finder = PathFinder(raster_path, source_coords=SOURCE, target_coords=TARGET, search_space_buffer_m=buffer,
                        graph_api="cython", neighborhood_str="r1")
    return cost.evaluate_route_cost(line, finder.raster_handler)


def test_classic_settings_pass_no_objective_and_the_route_is_the_plain_one(raster_path, layers, monkeypatch):
    constructed = _spy(monkeypatch, PathFinder, "__init__")
    finder, built, failed = _route(raster_path, layers, {})
    assert failed == [] and len(built) == 1
    assert len(constructed) == 1                                           # no second (plain) finder either
    kwargs = constructed[0][1]
    for name in ("objective", "gradient_options", "metric_layers", "dem", "weight_precision"):
        assert name not in kwargs
    assert not hasattr(finder, "gui_objective_key")
    params = built[0].params
    assert "objective_spec" not in params and "objective_key" not in params
    assert built[0].session is not None                                    # still a retained session
    # the same route as a finder built by hand
    reference = PathFinder(raster_path, source_coords=SOURCE, target_coords=TARGET, search_space_buffer_m=80,
                           neighborhood_str="r1").find_route(algorithm="dijkstra")
    assert list(built[0].line.coords) == [tuple(c) for c in reference.path_coords]
    built[0].session.close()


def test_weights_metric_layer_derived_slope_and_dem_reach_pathfinder(raster_path, layers, monkeypatch):
    constructed = _spy(monkeypatch, PathFinder, "__init__")
    settings = _settings(layers, weights={"cost": 1.0, "length": 0.2, "gradient": 0.5}, dem=True,
                         extra=[("landscape", "stripe", 0.5), ("steep", obj.SOURCE_DERIVE_SLOPE, 0.1)],
                         gradient={"max_gradient_pct": 80, "multiplier": "exponential"})
    finder, built, failed = _route(raster_path, layers, settings)
    assert failed == [] and len(built) == 1
    first = constructed[0][1]                                              # the objective finder
    assert first["objective"] == {"cost": 1.0, "length": 0.2, "gradient": 0.5, "landscape": 0.5, "steep": 0.1}
    assert first["dem"] == layers["dem"].meta["source_path"]
    assert first["metric_layers"]["landscape"] == {"source": layers["stripe"].meta["source_path"]}
    assert first["metric_layers"]["steep"] == {"derive": "slope_from_dem"}
    assert first["gradient_options"]["max_gradient_pct"] == 80.0
    assert first["gradient_options"]["multiplier"] == "exponential"
    assert first["weight_precision"] == "uint16"
    second = constructed[1][1]                                             # the plain finder that prices it
    for name in ("objective", "dem", "metric_layers", "gradient_options"):
        assert name not in second
    assert finder.objective.weights == first["objective"]
    assert built[0].session is not None                                    # metric layers: a retained session
    built[0].session.close()
    assert built[0].params["objective_spec"] == obj.normalise_objective_settings(settings)
    assert built[0].params["objective_key"] == finder.gui_objective_key == obj.objective_key(settings)
    json.dumps(built[0].params)                                            # the params can be saved


def test_a_dem_changes_the_route_and_stays_on_a_retained_session_on_cython(raster_path, layers):
    _finder, classic, _ = _route(raster_path, layers, {})
    settings = _settings(layers, dem=True, gradient={"multiplier": "exponential"})
    finder, built, failed = _route(raster_path, layers, settings)
    assert failed == []
    assert list(built[0].line.coords) != list(classic[0].line.coords)         # the hill is avoided
    assert built[0].session is not None and built[0].session.has_route       # cython + Dijkstra: a session
    assert finder.gui_objective_key == obj.objective_key(settings)
    classic[0].session.close()
    built[0].session.close()


def test_readout_cost_under_an_objective_is_the_plain_raster_cost_of_the_same_line(raster_path, layers):
    settings = _settings(layers, weights={"cost": 1.0, "length": 0.5}, dem=True)
    finder, built, _ = _route(raster_path, layers, settings)
    route = built[0]
    expected = _plain_cost(raster_path, route.line)
    assert route.cost.total_cost == pytest.approx(expected.total_cost)
    assert route.cost.total_cell_cost == pytest.approx(expected.total_cell_cost)
    assert route.cost.total_cost > 0
    # what the readout used to do: price on the search raster of the objective finder
    on_search_raster = cost.evaluate_route_cost(route.line, finder.raster_handler)
    assert on_search_raster.total_cost != pytest.approx(expected.total_cost)
    route.session.close()


def test_the_priced_window_is_the_search_window(raster_path, layers):
    settings = _settings(layers, weights={"cost": 1.0, "length": 0.5})
    finder, built, _ = _route(raster_path, layers, settings, buffer=30)
    plain = routing.plain_cost_handler(finder)
    assert plain.data.shape == finder.raster_handler.data.shape
    assert plain.window_transform == finder.raster_handler.window_transform
    assert plain.data.dtype == np.uint16
    built[0].session.close()


@pytest.mark.parametrize("graph_api", ["networkx", "networkit", "igraph", "rustworkx"])
def test_float32_runs_directly_on_a_float_backend_and_is_priced_on_the_plain_raster(raster_path, layers, graph_api):
    if not _installed(graph_api):
        pytest.skip(f"{graph_api} is not installed")
    settings = _settings(layers, weights={"cost": 1.0, "length": 0.5}, precision="float32")
    finder, built, failed = _route(raster_path, layers, settings, graph_api=graph_api)
    assert failed == [] and built[0].session is None
    assert finder.weight_precision == "float32" and finder.raster_handler.data.dtype == np.float32
    expected = _plain_cost(raster_path, built[0].line)
    assert built[0].cost.total_cost == pytest.approx(expected.total_cost) and expected.total_cost > 0
    # the search raster itself cannot be priced: this was the readout of a float32 run
    with pytest.raises(ValueError, match="dtype"):
        cost.evaluate_route_cost(built[0].line, finder.raster_handler)


def test_a_metric_layer_forbids_cells_even_with_weight_zero(raster_path, layers):
    # the stripe spans every row, so every route from SOURCE to TARGET crosses it: forbidding it leaves none
    settings = _settings(layers, extra=[("wall", "stripe", 0.0)])
    settings["layers"][0]["hard_max"] = 10.0
    _finder, built, failed = _route(raster_path, layers, settings)
    assert built == [] and len(failed) == 1
    open_settings = _settings(layers, extra=[("wall", "stripe", 0.0)])
    _finder, built, failed = _route(raster_path, layers, open_settings)
    assert failed == [] and len(built) == 1


def test_estimate_counts_the_full_raster_bands_of_an_objective_run(raster_path):
    base = routing.estimate_routing(raster_path, sources=[SOURCE], targets=[TARGET], graph_api="cython",
                                    algorithm="dijkstra", search_buffer_m=80)
    more = routing.estimate_routing(raster_path, sources=[SOURCE], targets=[TARGET], graph_api="cython",
                                    algorithm="dijkstra", search_buffer_m=80, stack_bands=3)
    assert more["est_memory_mb"] - base["est_memory_mb"] == pytest.approx(SIZE * SIZE * 4 * 3 / 1e6)
    assert more["est_seconds"] == base["est_seconds"]                      # no timing claim for the slope kernels


# ============================================================ C: the run callback
@pytest.fixture()
def raster(layers):
    return layers["cost"]


@pytest.fixture()
def run_calls(monkeypatch):
    calls: list[dict] = []

    def fake_run(_path, **kwargs):
        calls.append(kwargs)
        return None, [], []

    monkeypatch.setattr(routing, "run_routing", fake_run)
    return calls


def _new_notices(resp):
    return resp[ids.NOTICES]["data"]


def test_a_classic_run_hands_run_routing_exactly_the_plain_keywords(app, state, raster, run_calls):
    state.gui_settings["objective"] = {"weights": {"cost": 3}}                # a cost weight alone is classic
    _run(app, raster.id)
    assert run_calls and set(run_calls[0]) == {
        "sources", "targets", "waypoints", "algorithm", "graph_api", "neighborhood", "pairwise",
        "search_buffer_m", "full_window", "ignore_max_cost", "delta", "num_threads", "margin"}


def test_an_objective_run_hands_run_routing_the_objective_keywords(app, state, raster, layers, run_calls):
    state.gui_settings["objective"] = _settings(layers, weights={"cost": 1, "length": 0.3}, dem=True)
    _run(app, raster.id)
    assert run_calls, "the run was stopped"
    call = run_calls[0]
    assert call["objective_kwargs"]["objective"] == {"cost": 1.0, "length": 0.3}
    assert call["objective_kwargs"]["dem"] == layers["dem"].meta["source_path"]
    assert call["objective_spec"] == obj.normalise_objective_settings(state.gui_settings["objective"])
    assert call["objective_key"] == obj.objective_key(state.gui_settings["objective"])
    assert call["force_direct"] is False


def test_a_run_with_a_dem_and_delta_stepping_is_refused_and_names_the_algorithm_field(
        app, state, raster, layers, run_calls):
    state.gui_settings["objective"] = _settings(layers, dem=True)
    resp = _run(app, raster.id, algorithm="delta-stepping")
    assert run_calls == []                                                  # nothing ran
    assert resp[ids.ROUTING_STATUS]["children"] == "check the highlighted field"
    notice = _new_notices(resp)[-1]
    assert notice["focus_control"] == ids.ALGORITHM and notice["focus_id"] == ids.TAB_ROUTES
    assert "delta-stepping" in notice["meaning"] and "Dijkstra" in notice["meaning"]
    assert "Algorithm" in notice["fix"] and notice["severity"] == "warning"
    assert state.layers_of_kind("route") == []


def test_a_run_with_a_dem_on_raster_fim_is_refused_and_names_the_backend_field(
        app, state, raster, layers, run_calls, monkeypatch):
    monkeypatch.setattr(routing, "backend_available", lambda name: True)
    state.gui_settings["objective"] = _settings(layers, dem=True)
    resp = _run(app, raster.id, algorithm="fim", graph_api="raster_fim")
    assert run_calls == []
    notice = _new_notices(resp)[-1]
    assert notice["focus_control"] == ids.GRAPH_API and "raster_fim" in notice["meaning"]


def test_float32_on_cython_is_refused_and_names_the_precision_control(app, state, raster, layers, run_calls):
    state.gui_settings["objective"] = _settings(layers, weights={"cost": 1, "length": 1}, precision="float32")
    resp = _run(app, raster.id)
    assert run_calls == []
    notice = _new_notices(resp)[-1]
    assert notice["focus_control"] == ids.OBJ_PRECISION and "16-bit kernels" in notice["meaning"]


def test_a_slope_weight_without_a_dem_is_refused_and_names_the_dem_field(app, state, raster, layers, run_calls):
    state.gui_settings["objective"] = _settings(layers, weights={"gradient": 2})
    resp = _run(app, raster.id)
    assert run_calls == []
    notice = _new_notices(resp)[-1]
    assert notice["focus_control"] == ids.OBJ_DEM_LAYER and "slope weight is 2" in notice["meaning"]


def test_a_bad_weight_names_the_field_value_range_and_unit(app, state, raster, layers, run_calls):
    state.gui_settings["objective"] = _settings(layers, weights={"cost": 1, "length": 5e6})
    notice = _new_notices(_run(app, raster.id))[-1]
    assert notice["focus_control"] == ids.OBJ_W_LENGTH
    assert "Length weight" in notice["title"]
    for fragment in ("5,000,000", "from 0 to 1,000,000", "per m"):
        assert fragment in notice["meaning"]
    assert run_calls == []


def test_a_dem_layer_that_is_gone_or_has_another_crs_is_refused(app, state, raster, layers, run_calls):
    state.gui_settings["objective"] = _settings(layers, dem=True)
    state.remove_layer(layers["dem"].id)
    notice = _new_notices(_run(app, raster.id))[-1]
    assert notice["focus_control"] == ids.OBJ_DEM_LAYER and "not loaded any more" in notice["meaning"]
    other = state.add_layer("DEM wgs84", "raster", crs="EPSG:4326",
                            meta={"source_path": layers["dem"].meta["source_path"]})
    state.gui_settings["objective"] = {"weights": {"cost": 1}, "dem_layer_id": other.id}
    notice = _new_notices(_run(app, raster.id))[-1]
    assert notice["focus_control"] == ids.OBJ_DEM_LAYER and "EPSG:4326" in notice["meaning"]
    assert run_calls == []


def test_a_layer_file_that_no_longer_exists_is_refused(app, state, raster, layers, run_calls, tmp_path):
    ghost = state.add_layer("ghost", "raster", crs=TEST_CRS, meta={"source_path": str(tmp_path / "ghost.tif")})
    state.gui_settings["objective"] = {"weights": {"cost": 1}, "layers": [obj.layer_row("g", ghost.id)]}
    notice = _new_notices(_run(app, raster.id))[-1]
    assert "no longer exists" in notice["meaning"] and notice["focus_control"] == ids.OBJ_LAYERS_GRID
    assert run_calls == []


def test_a_dem_that_does_not_cover_the_routing_points_is_refused(
        app, state, raster, layers, run_calls, grid_files, tmp_path):
    small = _write_tif(tmp_path / "small.tif", grid_files["dem_array"][:60, :60])     # the top-left corner only
    corner = state.add_layer("corner", "raster", crs=TEST_CRS, meta={"source_path": small})
    state.gui_settings["objective"] = {"weights": {"cost": 1}, "dem_layer_id": corner.id}
    resp = _run(app, raster.id)
    notice = _new_notices(resp)[-1]
    assert notice["focus_control"] == ids.OBJ_DEM_LAYER and "routing point(s)" in notice["meaning"]
    assert run_calls == []


def test_a_dem_that_covers_part_of_the_window_gives_a_warning_notice_and_runs(
        app, state, raster, layers, run_calls, grid_files, tmp_path):
    half = _write_tif(tmp_path / "half.tif", grid_files["dem_array"][:, :150])      # x 500000..500150
    layer = state.add_layer("half", "raster", crs=TEST_CRS, meta={"source_path": half})
    state.gui_settings["objective"] = {"weights": {"cost": 1}, "dem_layer_id": layer.id}
    resp = _run(app, raster.id, draft=_draft((500020.0, 5599980.0), (500140.0, 5599900.0)), buffer=80)
    assert run_calls, _new_notices(resp)
    warning = [n for n in _new_notices(resp) if "does not cover the whole search window" in n["title"]]
    assert len(warning) == 1 and warning[0]["severity"] == "warning"
    assert warning[0]["focus_control"] == ids.OBJ_DEM_LAYER


def test_an_objective_run_end_to_end_stores_the_spec_and_reports_the_objective(app, state, raster, layers):
    settings = _settings(layers, weights={"cost": 1, "length": 0.2}, dem=True, gradient={"multiplier": "exponential"})
    state.gui_settings["objective"] = settings
    resp = _run(app, raster.id)
    assert "1 route(s) built" in resp[ids.ROUTING_STATUS]["children"]
    route = state.layers_of_kind("route")[0]
    params = route.meta["params"]
    assert params["objective_spec"] == obj.normalise_objective_settings(settings)
    assert params["objective_key"] == obj.objective_key(settings)
    assert state.finders[raster.id].gui_objective_key == params["objective_key"]
    assert route.meta["metrics"]["total_cost"] > 0
    expected = _plain_cost(raster.meta["source_path"], route.gdf.geometry.iloc[0])
    assert route.meta["metrics"]["total_cost"] == pytest.approx(expected.total_cost)
    success = [n for n in _new_notices(resp) if n["severity"] == "success"][-1]
    assert success["meaning"].startswith("dijkstra on cython")
    assert ("Objective: cost 1 + length 0.2; DEM 'DGM'; exponential slope multiplier; cython runs Dijkstra only."
            in success["meaning"])
    state.search_sessions[route.id].close()


def test_a_heavy_objective_run_keeps_the_objective_keywords_and_counts_the_stack(app, state, raster, layers,
                                                                              monkeypatch):
    seen: list[dict] = []

    def heavy(_path, **kwargs):
        seen.append(kwargs)
        return {"est_seconds": 1e6, "est_memory_mb": 1.0, "n_pairs": 1, "total_cells": 1, "n_segments": 1,
                "algorithm": "dijkstra", "graph_api": "cython", "resolution_m": 1.0, "neighborhood": "r1",
                "buffer_m": 80.0, "est_storage_mb": 1.0}

    monkeypatch.setattr(routing, "estimate_routing", heavy)
    state.gui_settings["objective"] = _settings(layers, weights={"cost": 1, "length": 0.2},
                                                extra=[("landscape", "stripe", 0.5)])
    resp = _run(app, raster.id)
    assert resp[ids.ROUTING_CONFIRM_MODAL]["is_open"] is True
    assert seen[0]["stack_bands"] == 2
    pending = state.pending_routing
    assert pending["kwargs"]["objective_kwargs"]["objective"] == {"cost": 1.0, "length": 0.2, "landscape": 0.5}
    assert pending["kwargs"]["force_direct"] is False                      # extra layers on cython: a session
    assert pending["meta"]["objective_summary"].startswith("Objective: cost 1 + length 0.2 + landscape 0.5")
    # a plain heavy run asks for nothing extra
    state.gui_settings["objective"] = {}
    seen.clear()
    _run(app, raster.id)
    assert "stack_bands" not in seen[0]


# ============================================================ C: the heavy job and the library's warnings
#: the box the GUI shows for the library's AlgorithmFallbackWarning (it used to be the generic warning box)
FALLBACK_TITLE = "A different search ran than the one you chose"


def _fallback_kwargs(raster_path, layers):
    """run_routing keywords that make the library warn: delta-stepping with a DEM (the GUI refuses this
    combination before it gets here, so it is built by hand)."""
    settings = _settings(layers, dem=True)
    return {"sources": [SOURCE], "targets": [TARGET], "algorithm": "delta-stepping", "graph_api": "cython",
            "neighborhood": "r1", "search_buffer_m": 80.0,
            **obj.run_kwargs(settings, layer_path=_paths(layers), graph_api="cython")}


def test_a_library_fallback_warning_of_a_synchronous_run_becomes_a_notice(raster_path, layers):
    result, notices = errors.guard(routing.run_routing, raster_path, **_fallback_kwargs(raster_path, layers))
    assert result is not None
    titles = [n for n in notices if n["title"] == FALLBACK_TITLE]
    assert len(titles) == 1
    assert "Dijkstra runs instead" in titles[0]["details"]
    result[1][0].session.close()


def _collect_routing_job(state):
    """The ended routing job through the job poll's steps (round 15): claim, finalize, confirm."""
    def finalize(job):
        return routes_cb.routing_outcome(state, job)

    assert jobs.poll_tick(state, jobs.ROUTING_SLOT, None, finalize).step == jobs.TICK_FINALIZING
    tick = jobs.poll_tick(state, jobs.ROUTING_SLOT, None, finalize)
    assert tick.step == jobs.TICK_DELIVER
    assert jobs.poll_tick(state, jobs.ROUTING_SLOT, tick.handover.token, finalize).step == jobs.TICK_IDLE
    return tick.handover.results, tick.handover.notices


def test_a_library_fallback_warning_of_a_background_run_is_not_lost(state, raster, raster_path, layers):
    kwargs = _fallback_kwargs(raster_path, layers)
    meta = {"raster_layer_id": raster.id, "algorithm": "delta-stepping", "n_sources": 1, "n_targets": 1,
            "n_waypoints": 0, "simplify_tol": None, "wp_names": []}
    job = routing_job.start_routing_job(state, {"raster_path": raster_path, "kwargs": kwargs, "meta": meta})
    job.thread.join(120)
    assert job.status == "done", job.error
    assert [n for n in job.notices if n["title"] == FALLBACK_TITLE]
    (view, status, _options, _newest, _draft_out), notices = _collect_routing_job(state)
    assert "1 route(s) built" in status
    assert [n for n in notices if n["title"] == FALLBACK_TITLE]
    assert state.routing_job is None
    for session in list(state.search_sessions.values()):
        session.close()


def test_a_failed_background_run_keeps_the_warnings_it_raised_and_its_error(state, monkeypatch):
    import warnings

    def boom(_path, **_kwargs):
        warnings.warn("something odd happened first", UserWarning)
        raise RuntimeError("the run broke")

    monkeypatch.setattr(routing, "run_routing", boom)
    job = routing_job.start_routing_job(state, {"raster_path": "x", "kwargs": {}, "meta": {}})
    job.thread.join(30)
    assert job.status == "failed" and "the run broke" in str(job.error)
    assert any("something odd happened first" in n["details"] for n in job.notices)
    (_view, status, *_rest), notices = _collect_routing_job(state)
    assert status == "routing failed"
    assert any("something odd happened first" in n["details"] for n in notices)
    assert any("the run broke" in json.dumps(n) for n in notices)


# ============================================================ C: route edits
def _close_all_sessions(state):
    for session in list(state.search_sessions.values()):
        session.close()


def _build_route(app, state, raster, settings, **run_kwargs):
    state.gui_settings["objective"] = settings
    resp = _run(app, raster.id, **run_kwargs)
    routes = state.layers_of_kind("route")
    assert routes, _new_notices(resp)
    return routes[-1]


def _objective_finders(constructed):
    return [kwargs for _args, kwargs in constructed if "objective" in kwargs]


def test_an_edit_of_an_objective_route_reuses_the_finder_with_the_same_key(app, state, raster, layers, monkeypatch):
    route = _build_route(app, state, raster, _settings(layers, dem=True, gradient={"multiplier": "exponential"}))
    key = route.meta["params"]["objective_key"]
    constructed = _spy(monkeypatch, PathFinder, "__init__")
    moved = [list(SOURCE), [TARGET[0] - 10.0, TARGET[1] + 10.0]]
    new_layer, notices = edit_cb.recompute_variant(state, route, moved, "moved target", [])
    assert new_layer is not None, notices
    assert _objective_finders(constructed) == []                            # the cached finder fitted
    assert new_layer.meta["params"]["objective_key"] == key
    assert new_layer.meta["params"]["objective_spec"] == route.meta["params"]["objective_spec"]
    # priced on the plain raster of the same window
    expected = _plain_cost(raster.meta["source_path"], new_layer.gdf.geometry.iloc[0])
    assert new_layer.meta["metrics"]["total_cost"] == pytest.approx(expected.total_cost)
    state.search_sessions[new_layer.id].close()


def test_an_edit_rebuilds_the_finder_when_the_key_differs_and_when_the_cached_finder_is_plain(
        app, state, raster, layers, monkeypatch):
    route = _build_route(app, state, raster, _settings(layers, weights={"cost": 1, "length": 0.4}, dem=True))
    key = route.meta["params"]["objective_key"]
    state.finders[raster.id].gui_objective_key = "some other objective"
    constructed = _spy(monkeypatch, PathFinder, "__init__")
    moved = [list(SOURCE), [TARGET[0] - 10.0, TARGET[1] + 10.0]]
    new_layer, notices = edit_cb.recompute_variant(state, route, moved, "moved target", [])
    assert new_layer is not None, notices
    rebuilt = _objective_finders(constructed)
    assert len(rebuilt) == 1 and rebuilt[0]["objective"] == {"cost": 1.0, "length": 0.4}
    assert rebuilt[0]["dem"] == layers["dem"].meta["source_path"]
    assert state.finders[raster.id].gui_objective_key == key
    # a plain finder in the cache does not fit an objective route either
    state.finders[raster.id] = PathFinder(raster.meta["source_path"], source_coords=SOURCE, target_coords=TARGET,
                                          search_space_buffer_m=80, neighborhood_str="r1")
    constructed.clear()
    again, notices = edit_cb.recompute_variant(state, new_layer, moved, "moved target again", [])
    assert again is not None, notices
    assert len(_objective_finders(constructed)) == 1
    for session in list(state.search_sessions.values()):
        session.close()


def test_a_plain_route_does_not_fit_an_objective_finder(app, state, raster, layers, monkeypatch):
    plain_route = _build_route(app, state, raster, {})
    state.finders[raster.id].gui_objective_key = "left over from an objective run"
    constructed = _spy(monkeypatch, PathFinder, "__init__")
    new_layer, notices = edit_cb.recompute_variant(
        state, plain_route, [list(SOURCE), [TARGET[0] - 10.0, TARGET[1] + 10.0]], "moved target", [])
    assert new_layer is not None, notices
    assert len(constructed) == 1 and "objective" not in constructed[0][1]    # a new, plain finder
    assert "objective_key" not in new_layer.meta["params"]
    for session in list(state.search_sessions.values()):
        session.close()


def test_an_edit_needs_the_layers_the_route_used(app, state, raster, layers):
    route = _build_route(app, state, raster, _settings(layers, dem=True))
    state.finders.clear()                                                   # a finder that must be rebuilt
    state.remove_layer(layers["dem"].id)
    moved = [list(SOURCE), [TARGET[0] - 10.0, TARGET[1] + 10.0]]
    new_layer, notices = edit_cb.recompute_variant(state, route, moved, "moved target", [])
    assert new_layer is None
    notice = notices[-1]
    assert notice["title"] == "The terrain or metric layer this route used is not loaded"
    assert "Terrain model (DEM)" in notice["meaning"] and "edit was not applied" in notice["impact"]
    assert notice["focus_control"] == ids.OBJ_DEM_LAYER
    assert state.get(route.id) is route                                     # nothing was replaced


def test_a_metric_layer_route_keeps_its_session_and_survives_a_move_out_of_the_window(
        app, state, raster, layers, monkeypatch):
    """Round 16: a route with a metric layer is retained in a session again (``force_direct`` is False on cython);
    the session rebuilds its window itself and the clone carries the layers (before, the edit rebuilt a direct
    finder)."""
    settings = _settings(layers, extra=[("landscape", "stripe", 0.5)])
    route = _build_route(app, state, raster, settings, buffer=5)
    assert route.id in state.search_sessions                                # no force_direct: a retained session
    constructed = _spy(monkeypatch, PathFinder, "__init__")
    outside = [list(SOURCE), [500005.0, 5599900.0]]                         # left of the 5 m window
    new_layer, notices = edit_cb.recompute_variant(state, route, outside, "moved target", [])
    assert new_layer is not None, notices
    rebuilt = _objective_finders(constructed)
    assert len(rebuilt) == 1 and "landscape" in rebuilt[0]["metric_layers"]  # the session's clone
    assert new_layer.id in state.search_sessions
    assert new_layer.meta["params"]["objective_key"] == route.meta["params"]["objective_key"]
    assert new_layer.meta["metrics"]["total_cost"] > 0
    _close_all_sessions(state)


def test_a_metric_layer_route_on_a_gpu_backend_is_still_edited_directly():
    """No device here, so the retained GPU/FIM session with metric layers was not run: those backends keep the
    old safeguard (``force_direct``)."""
    spec = {"weights": {"cost": 1}, "layers": [obj.layer_row("a", "M1")]}
    assert edit_cb._route_plan({"graph_api": "raster_gpu", "algorithm": "dijkstra", "objective_spec": spec})[3] is True
    assert edit_cb._route_plan({"graph_api": "raster_fim", "algorithm": "fim", "objective_spec": spec})[3] is True


def test_a_float32_route_is_edited_directly_on_its_backend(app, state, raster, layers, monkeypatch):
    pytest.importorskip("networkx")
    settings = _settings(layers, weights={"cost": 1, "length": 0.4}, precision="float32")
    route = _build_route(app, state, raster, settings, graph_api="networkx", algorithm="dijkstra")
    assert route.meta["params"]["graph_api"] == "networkx" and route.id not in state.search_sessions
    new_layer, notices = edit_cb.recompute_variant(
        state, route, [list(SOURCE), [TARGET[0] - 10.0, TARGET[1] + 10.0]], "moved target", [])
    assert new_layer is not None, notices
    assert new_layer.meta["metrics"]["total_cost"] > 0
    assert new_layer.meta["params"]["objective_spec"]["weight_precision"] == "float32"


def test_route_plan_is_direct_when_the_stored_objective_forces_it():
    plan = edit_cb._route_plan({"graph_api": "cython", "algorithm": "dijkstra"})
    assert plan[3] is False
    with_layers = edit_cb._route_plan({"graph_api": "cython", "algorithm": "dijkstra",
                                       "objective_spec": {"weights": {"cost": 1},
                                                          "layers": [obj.layer_row("a", "M1")]}})
    assert with_layers[3] is False                      # round 16: extra layers on cython use the session
    float32 = edit_cb._route_plan({"graph_api": "cython", "algorithm": "dijkstra",
                                   "objective_spec": {"weights": {"cost": 1, "length": 1},
                                                      "weight_precision": "float32"}})
    assert float32[3] is True
    assert edit_cb._route_plan({"graph_api": "raster_gpu", "algorithm": "dijkstra",
                                "objective_spec": {"dem_layer_id": "DEM"}})[3] is True


# ============================================================ C: the stored spec is saved with the project
def test_the_stored_spec_and_the_settings_round_trip_through_save_and_load(state, tmp_path):
    from pyorps.gui.state import ProjectState
    spec = obj.normalise_objective_settings({
        "weights": {"cost": 1, "length": 0.2}, "dem_layer_id": "abc123", "weight_precision": "float32",
        "layers": [obj.layer_row("landscape", "def456", 0.5, hard_max=40)],
        "gradient": {"max_gradient_pct": 30, "multiplier": "sigmoid"}})
    params = {"graph_api": "networkx", "algorithm": "dijkstra", "objective_spec": spec,
              "objective_key": obj.objective_key(spec)}
    line = LineString([SOURCE, TARGET])
    routes_cb.add_route_layer(
        state, line=line, route_cost=cost.RouteCost(1.0, 2.0, 3.0, 1.0), params=params,
        control_points=[SOURCE, TARGET], crs=TEST_CRS)
    state.gui_settings["objective"] = spec
    path = project_io.save_project(state, tmp_path / "proj" / "project.json")
    manifest = json.loads((tmp_path / "proj" / "project.json").read_text(encoding="utf-8"))
    assert manifest["gui_settings"]["objective"] == spec
    other = ProjectState(work_dir=tmp_path / "other")
    try:
        project_io.load_project(other, path)
        loaded = other.layers_of_kind("route")[0].meta["params"]
        assert loaded["objective_spec"] == spec and loaded["objective_key"] == params["objective_key"]
        assert other.gui_settings["objective"] == spec
        assert objective_cb.read_settings(other) == spec
        # the loaded spec rebuilds the very same library arguments
        assert obj.objective_key(loaded["objective_spec"]) == loaded["objective_key"]
    finally:
        other.shutdown()


# ============================================================ C: callback helpers
def test_every_problem_key_has_a_control_and_a_notice():
    for key in obj.PROBLEM_KEYS:
        control = objective_cb.control_for(key.replace("<i>", "0"))
        assert control, key
    for control in objective_cb.FIELD_CONTROLS.values():
        assert control in vars(ids).values(), control
    assert objective_cb.control_for("layers.3.hard_max") == ids.OBJ_LAYERS_GRID
    assert objective_cb.control_for("something.else") is None
    notice = objective_cb.problem_notice(_problems({"weights": {"cost": -1}})[0])
    assert notice["focus_control"] == ids.OBJ_W_COST and notice["focus_id"] == ids.TAB_ROUTES
    assert notice["title"] == "Cost weight: value not usable"
    assert notice["meaning"].startswith("Cost weight is -1, but must be from 0 to 1,000,000")
    assert notice["impact"] == "Nothing was run."


def test_prepare_run_classic_and_problem_paths(state, raster, layers):
    kwargs = {"raster_layer": raster, "graph_api": "cython", "algorithm": "dijkstra",
              "raster_bounds": (ORIGIN_X, ORIGIN_Y - SIZE, ORIGIN_X + SIZE, ORIGIN_Y),
              "points": [SOURCE, TARGET], "buffer_m": 80.0}
    assert objective_cb.prepare_run(state, notices=[], **kwargs) == ({}, {}, [])
    state.gui_settings["objective"] = _settings(layers, weights={"cost": 1, "length": 0.2})
    run, meta, notices = objective_cb.prepare_run(state, notices=[], **kwargs)
    assert run["objective_kwargs"]["objective"] == {"cost": 1.0, "length": 0.2} and notices == []
    assert meta["objective_summary"].startswith("Objective: cost 1 + length 0.2") and meta["stack_bands"] == 1
    state.gui_settings["objective"] = {"weights": {"cost": 0}}
    run, meta, notices = objective_cb.prepare_run(state, notices=["kept"], **kwargs)
    assert run is None and meta == {} and notices[0] == "kept" and "all 0" in notices[-1]["meaning"]


# ============================================================ D: the plan's [A] assumptions
def test_the_library_refuses_a_dem_in_another_crs_and_names_both_systems(raster_path, grid_files):
    """The library does not reproject a DEM file. A DEM in EPSG:4326 next to a cost raster in EPSG:32632 stops the
    finder with an ``InvalidInputError`` that names both systems (it used to be rasterio's "non-negative" error).
    The GUI still refuses the DEM first (``problems``), before any run starts."""
    with pytest.raises(InvalidInputError, match=r"DEM is in EPSG:4326 but the cost raster is in EPSG:32632"):
        PathFinder(raster_path, source_coords=SOURCE, target_coords=TARGET, search_space_buffer_m=80,
                   neighborhood_str="r1", dem=grid_files["dem_wgs84"], objective={"cost": 1})
    # the same DEM in the CRS of the cost raster routes
    PathFinder(raster_path, source_coords=SOURCE, target_coords=TARGET, search_space_buffer_m=80,
               neighborhood_str="r1", dem=grid_files["dem"], objective={"cost": 1})
    assert [p.key for p in _problems(_dem_settings(), raster_crs=TEST_CRS, dem_crs="EPSG:4326")] == ["dem_layer_id"]


def test_the_library_refuses_a_dem_that_misses_the_search_window_and_names_both_extents(
        raster_path, grid_files, tmp_path):
    """A DEM far from the points is an ``InvalidInputError`` of the library naming the extent of the DEM and the
    one of the search window. The GUI catches the same case earlier with ``dem_coverage``."""
    elsewhere = _write_tif(tmp_path / "elsewhere.tif", grid_files["dem_array"],
                           transform=from_origin(ORIGIN_X + 10000, ORIGIN_Y, 1, 1))
    with pytest.raises(InvalidInputError) as caught:
        PathFinder(raster_path, source_coords=SOURCE, target_coords=TARGET, search_space_buffer_m=80,
                   neighborhood_str="r1", dem=elsewhere, objective={"cost": 1})
    message = str(caught.value)
    assert "does not overlap the search window" in message
    assert "the DEM covers x 510000.00 to 510200.00" in message          # the DEM's own extent
    assert "the search window covers x 500000.00 to 500200.00" in message    # the whole raster: the buffer is 80 m
    problem, _warning = obj.dem_coverage((ORIGIN_X + 10000, ORIGIN_Y - SIZE, ORIGIN_X + 10000 + SIZE, ORIGIN_Y),
                                         [SOURCE, TARGET])
    assert problem is not None


@pytest.mark.skipif(not _gpu_works(), reason="no CUDA device: raster_gpu cannot run here")
def test_assumption_raster_gpu_with_a_dem_routes_through_find_route_and_matches_the_cpu(raster_path, layers):
    settings = _settings(layers, dem=True, gradient={"multiplier": "exponential"})
    _f, cpu, _ = _route(raster_path, layers, settings, graph_api="cython", algorithm="dijkstra")
    finder, gpu, failed = _route(raster_path, layers, settings, graph_api="raster_gpu", algorithm="dijkstra")
    assert failed == [] and gpu[0].session is None                          # direct, no retained GPU field
    assert finder.graph_api_name == "raster_gpu"
    assert gpu[0].params["objective_key"] == cpu[0].params["objective_key"]
    cpu_cost = _plain_cost(raster_path, cpu[0].line).total_length
    gpu_cost = _plain_cost(raster_path, gpu[0].line).total_length
    assert gpu_cost == pytest.approx(cpu_cost, rel=0.05)
    cpu[0].session.close()
    finder.release_device_resources()


@pytest.mark.skipif(not _gpu_works(), reason="no CUDA device: raster_gpu cannot run here")
def test_assumption_raster_gpu_float32_runs_directly(raster_path, layers):
    settings = _settings(layers, weights={"cost": 1, "length": 0.3}, precision="float32")
    finder, built, failed = _route(raster_path, layers, settings, graph_api="raster_gpu", algorithm="dijkstra")
    assert failed == [] and built[0].session is None and built[0].cost.total_cost > 0
    finder.release_device_resources()


def test_a_session_that_rebuilds_its_window_keeps_the_metric_layers(raster_path, layers):
    """The library carries ``metric_layers`` over when a session rebuilds its search window (it used to stop with
    "unknown metric layer"). Since round 16 the GUI keeps such a route in a retained session on cython
    (``force_direct`` is False for extra metric layers there): this test pins the library side, the proof that a
    session route equals the direct route is in ``test_gui_terrain_length_round16.py``. The buffer is several
    cells, so the window rebuild is not mixed up with the window-snapping of a one-cell buffer."""
    settings = _settings(layers, extra=[("landscape", "stripe", 0.5)])
    kwargs = obj.run_kwargs(settings, layer_path=_paths(layers))["objective_kwargs"]
    finder = PathFinder(raster_path, source_coords=[SOURCE], target_coords=[TARGET], search_space_buffer_m=8,
                        neighborhood_str="r1", **kwargs)
    session = finder.search_session(algorithm="dijkstra")
    session.route([SOURCE, TARGET])
    outside = (500005.0, 5599900.0)                                         # left of the window (x from 500012)
    path = session.update([SOURCE, outside, TARGET])                        # rebuilds the window
    assert session.finder is not finder                                      # it really was rebuilt
    assert session.finder.metric_layers == kwargs["metric_layers"]           # ... and kept the layers
    assert path is not None
    assert obj.force_direct(settings, "cython") is False                     # the GUI uses the session (round 16)
    assert obj.force_direct(settings, "raster_gpu") is True                  # not run without a device: guard stays
    session.close()


