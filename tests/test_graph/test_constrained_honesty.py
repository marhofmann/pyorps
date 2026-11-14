"""Constrained routing and the GPU paths say what they do not do, and why something failed.

Three contracts (audit T4, findings #5, #25 and #26):

* #5  A constrained GPU solver refuses a constraint setting it does not enforce instead of returning a route
  that ignores it (``pyorps.utils.constrained_sssp_gpu.check_constrained_gpu_settings``, called at the dispatch
  point of ``ConstrainedPathFinder`` and in the solver entry functions).
* #25 ``ConstrainedPathFinder`` rejects unknown keyword arguments with the closest valid name, takes a profile
  path as ``os.PathLike``, and refuses a source/target outside the search window.
* #26 A GPU failure keeps its cause: "unavailable" is only said for import/CUDA errors and names the error,
  out-of-memory is reported as out-of-memory, and a programming error propagates.

Everything except the last section runs without a GPU: the failing call is replaced by a fake.
"""
import contextlib
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from pyorps.core.exceptions import AlgorithmNotImplementedError, PyorpsError
from pyorps.graph.api import raster_gpu_api
from pyorps.graph.constrained_path_finder import ConstrainedPathFinder
from pyorps.utils import constrained_sssp_gpu as gpu_v1
from pyorps.utils import constrained_sssp_gpu_v2 as gpu_v2
from pyorps.utils import constrained_sssp_gpu_v4 as gpu_v4
from pyorps.utils import sssp_gpu, traversal_gpu

try:
    import cupy
    HAS_CUPY = True
except ImportError:
    cupy = None
    HAS_CUPY = False

gpu_only = pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")
needs_cupy_class = pytest.mark.skipif(not HAS_CUPY, reason="CuPy not importable: its OutOfMemoryError cannot be raised")

# Same opt-in as tests/test_graph/test_constrained_gpu_v2.py / _v4.py: the constrained GPU planners are
# unvalidated and can pin the card, so nothing here starts one unless the owner asks for it.
RUN_CONSTRAINED_GPU = os.environ.get("PYORPS_RUN_CONSTRAINED_GPU", "").strip().lower() not in ("", "0", "false")
constrained_gpu_only = pytest.mark.skipif(
    not RUN_CONSTRAINED_GPU,
    reason="constrained GPU planners are unvalidated and can wedge the GPU; set PYORPS_RUN_CONSTRAINED_GPU=1 to run")

SOURCE = (5.5, 5.5)
TARGET = (34.5, 34.5)
SHAPE = (40, 40)


def _profile(**overrides) -> dict:
    profile = {
        "name": "honesty", "description": "tiny test profile",
        "soft_angle_limit_deg": 5.0, "hard_angle_limit_deg": 60.0,
        "angle_cost_function": "linear", "angle_cost_params": {"scale": 100},
        "min_span_m": 10.0, "max_span_m": 20.0, "span_bin_size_m": 1.0,
        "tower_cost_function": "terrain_plus_angle",
        "tower_cost_params": {
            "terrain_cost_map": {"0": 1000, "500": 5000}, "terrain_interpolation": "linear",
            "angle_types": {"suspension": {"max_angle_deg": 10.0, "base_cost": 500},
                            "angle_tower": {"max_angle_deg": 60.0, "base_cost": 2000}},
        },
    }
    profile.update(overrides)
    return profile


@pytest.fixture(scope="module")
def raster_path(tmp_path_factory) -> str:
    """A 40 x 40 m raster of 1 m cells with random costs 50..149 (no forbidden cell)."""
    path = tmp_path_factory.mktemp("honesty") / "cost.tif"
    data = np.random.default_rng(7).integers(50, 150, size=SHAPE).astype(np.uint16)
    with rasterio.open(path, "w", driver="GTiff", height=SHAPE[0], width=SHAPE[1], count=1, dtype="uint16",
                       crs="EPSG:32632", transform=from_bounds(0, 0, SHAPE[1], SHAPE[0], SHAPE[1], SHAPE[0])) as dst:
        dst.write(data, 1)
    return str(path)


def _finder(raster_path, profile=None, **kwargs) -> ConstrainedPathFinder:
    return ConstrainedPathFinder(raster_path, SOURCE, TARGET, profile=profile or _profile(),
                                 neighborhood_str="r1", **kwargs)


@pytest.fixture
def finder(raster_path) -> ConstrainedPathFinder:
    return _finder(raster_path)


@pytest.fixture(scope="module")
def cython_cost(raster_path) -> float:
    return float(_finder(raster_path).find_route().total_cost)


@contextlib.contextmanager
def _no_gpu_warning():
    """Fail on any of the warnings this module is about (other warnings of a routing call are not its business)."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", message=r".*(unavailable|falling back|does not enforce|no device session)")
        yield


class _Entry:
    """Stands in for a GPU solver entry function: records the call, then raises ``error`` (if any)."""

    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        raise AssertionError("the fake GPU solver has no result to return")


def _force_gpu_backend(monkeypatch, finder, backend, error=None) -> _Entry:
    """Send ``finder`` to the GPU solver of ``backend`` with the solver replaced by a fake (no CUDA needed)."""
    module, name = {"raster_gpu": (gpu_v2, "constrained_sssp_raster_gpu_v2"),
                    "raster_gpu_v4": (gpu_v4, "constrained_sssp_raster_gpu_v4")}[backend]
    entry = _Entry(error)
    monkeypatch.setattr(module, name, entry)
    monkeypatch.setattr(finder, "_constrained_backend", backend)
    return entry


# ---------------------------------------------------------------------------
# #25  keyword arguments, profile path, bounds check  (CPU only)
# ---------------------------------------------------------------------------

class TestKeywordsProfileAndBounds:

    def test_find_route_rejects_a_misspelt_keyword_with_the_closest_name(self, finder):
        with pytest.raises(TypeError, match=r"find_route\(\).*'sorce' \(did you mean 'source'\?\)"):
            finder.find_route(sorce=SOURCE)

    def test_find_route_rejects_pathfinder_options_it_never_read(self, finder):
        """``algorithm=`` was accepted and dropped; the message now lists what is accepted."""
        with pytest.raises(TypeError, match=r"'algorithm'.*Accepted here: 'source', 'source_connection_height_m', "
                                            r"'target', 'target_connection_height_m'"):
            finder.find_route(algorithm="delta-stepping")

    def test_find_route_still_takes_source_and_target(self, finder, cython_cost):
        result = finder.find_route(source=SOURCE, target=TARGET)
        assert float(result.total_cost) == pytest.approx(cython_cost)

    def test_constructor_rejects_a_misspelt_keyword_with_the_closest_name(self, raster_path):
        """The constructor's **kwargs go to PathFinder, which names the typo."""
        with pytest.raises(TypeError, match=r"'neighbourhood_str' \(did you mean 'neighborhood_str'\?\)"):
            ConstrainedPathFinder(raster_path, SOURCE, TARGET, profile=_profile(), neighbourhood_str="r1")

    def test_gui_keywords_stay_accepted(self, raster_path):
        """The exact call of pyorps.gui.services.constrained.run_constrained (dem/dsm left empty)."""
        from pyorps.core.infrastructure_profile import InfrastructureProfile
        gui_finder = ConstrainedPathFinder(
            dataset_source=raster_path, source_coords=SOURCE, target_coords=TARGET,
            profile=InfrastructureProfile.from_dict(_profile()), graph_api="cython", neighborhood_str="r1",
            search_space_buffer_m=30.0, dem=None, dsm=None)
        assert gui_finder.find_route().path_geometry is not None

    @pytest.mark.parametrize("suffix", [".json", ".yaml"])
    def test_profile_given_as_pathlib_path(self, raster_path, tmp_path, suffix):
        path = tmp_path / f"profile{suffix}"
        if suffix == ".json":
            path.write_text(json.dumps(_profile()), encoding="utf-8")
        else:
            yaml = pytest.importorskip("yaml")
            path.write_text(yaml.safe_dump(_profile()), encoding="utf-8")
        assert isinstance(path, Path)
        loaded = _finder(raster_path, profile=path)
        assert loaded.profile.max_span_m == 20.0
        assert _finder(raster_path, profile=str(path)).profile == loaded.profile

    def test_profile_given_as_any_os_pathlike(self, raster_path, tmp_path):
        path = tmp_path / "profile.json"
        path.write_text(json.dumps(_profile()), encoding="utf-8")

        class Wrapped:
            def __fspath__(self):
                return str(path)

        assert _finder(raster_path, profile=Wrapped()).profile.min_span_m == 10.0

    def test_profile_of_a_wrong_type_is_refused_at_construction(self, raster_path):
        with pytest.raises(TypeError, match="profile must be an InfrastructureProfile, a dict or a path.*got int"):
            _finder(raster_path, profile=3)

    @pytest.mark.parametrize("which", ["source", "target"])
    def test_point_outside_the_raster_raises_before_any_kernel(self, finder, monkeypatch, which):
        def no_kernel(*args, **kwargs):
            raise AssertionError("the solver was reached with an out-of-raster point")

        monkeypatch.setattr(finder, "_find_route_coupled", no_kernel)
        with pytest.raises(PyorpsError, match=r"Point \(500\.0, 5\.5\) lies outside the search window"):
            finder.find_route(**{which: (500.0, 5.5)})

    def test_non_finite_point_raises(self, finder):
        with pytest.raises(PyorpsError, match="not finite"):
            finder.find_route(source=(float("nan"), 5.5))


# ---------------------------------------------------------------------------
# #5  settings a GPU solver does not enforce  (CPU only)
# ---------------------------------------------------------------------------

_DEM = np.zeros(SHAPE, dtype=np.float32)
_AREA = np.zeros(2, dtype=np.int32)

#: (keyword handed to the V4 entry function, a non-default value, text the error must contain)
V4_IGNORED = [
    ("dem", _DEM, "dem (terrain gradient penalty"),
    ("obstacle_heights", _DEM, "obstacle_heights"),
    ("min_clearance", 7.0, "min_clearance=7"),
    ("max_gradient_pct", 30.0, "max_gradient_pct=30"),
    ("gradient_scale", 3.0, "gradient_scale=3"),
    ("area_offsets", _AREA, "area_offsets (exact tower footprint"),
    ("area_offset_starts", _AREA, "area_offsets (exact tower footprint"),
    ("area_offset_counts", _AREA, "area_offsets (exact tower footprint"),
]


def _solver_arguments(**overrides) -> dict:
    """Arguments every constrained GPU entry function takes (8-neighbourhood, 10 m cells, spans 20..40 m)."""
    steps = np.array([[0, 1], [1, 0], [0, -1], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=np.int8)
    n_dirs = len(steps)
    arguments = dict(
        raster=np.full((8, 8), 10, dtype=np.uint16), source_row=1, source_col=1, target_row=6, target_col=6,
        steps=steps,
        angle_cost_lut=np.zeros((n_dirs, n_dirs), dtype=np.float32),
        angle_valid_lut=np.ones((n_dirs, n_dirs), dtype=np.uint8),
        step_distances=(np.linalg.norm(steps.astype(np.float32), axis=1) * 10.0).astype(np.float32),
        tower_terrain_costs=np.ones(65536, dtype=np.float32),
        tower_angle_costs=np.zeros((n_dirs, n_dirs), dtype=np.float32),
        n_span_bins=2, span_bin_size=20.0, min_span=20.0, max_span=40.0)
    arguments.update(overrides)
    return arguments


class TestIgnoredSettingsHelper:

    @pytest.mark.parametrize("keyword,value,expected", V4_IGNORED, ids=[row[0] for row in V4_IGNORED])
    def test_v4_refuses_each_setting_it_does_not_implement(self, keyword, value, expected):
        with pytest.raises(AlgorithmNotImplementedError) as caught:
            gpu_v1.check_constrained_gpu_settings("raster_gpu_v4", **{keyword: value})
        message = str(caught.value)
        assert expected in message                                   # the setting
        assert "'raster_gpu_v4' (constrained_sssp_raster_gpu_v4)" in message   # the solver
        assert "graph_api='cython'" in message and "'raster_gpu' (GPU v2)" in message   # what to use instead
        assert isinstance(caught.value, PyorpsError)

    def test_v4_names_every_ignored_setting_at_once(self):
        with pytest.raises(AlgorithmNotImplementedError, match=r"dem \(.*min_clearance=7.*area_offsets"):
            gpu_v1.check_constrained_gpu_settings("raster_gpu_v4", dem=_DEM, min_clearance=7.0, area_offsets=_AREA)

    @pytest.mark.parametrize("solver", ["raster_gpu_v1", "raster_gpu"])
    def test_v1_and_v2_report_a_max_span_their_bin_grid_exceeds(self, solver):
        """7 bins of 50 m stop a span at 350 m: a 300 m limit is not enforced."""
        with pytest.raises(AlgorithmNotImplementedError) as caught:
            gpu_v1.check_constrained_gpu_settings(solver, n_span_bins=7, span_bin_size=50.0, max_span=300.0)
        message = str(caught.value)
        assert "max_span=300" in message and "7 * 50 = 350 m" in message
        assert "'raster_gpu_v4'" in message and "graph_api='cython'" in message

    def test_v4_enforces_max_span_itself(self):
        gpu_v1.check_constrained_gpu_settings("raster_gpu_v4", n_span_bins=7, span_bin_size=50.0, max_span=300.0)

    @pytest.mark.parametrize("solver", sorted(gpu_v1.CONSTRAINED_GPU_SOLVERS))
    def test_neutral_values_do_not_raise(self, solver):
        gpu_v1.check_constrained_gpu_settings(solver)
        gpu_v1.check_constrained_gpu_settings(
            solver, dem=None, obstacle_heights=None, min_clearance=0.0, max_gradient_pct=100.0,
            gradient_scale=2.0, area_offsets=None, area_offset_starts=None, area_offset_counts=None,
            n_span_bins=6, span_bin_size=50.0, max_span=300.0)
        # no span constraint at all, as ConstrainedPathFinder passes it
        gpu_v1.check_constrained_gpu_settings(solver, n_span_bins=1, span_bin_size=1e6, max_span=1e6)

    def test_unknown_solver_name_is_a_value_error(self):
        with pytest.raises(ValueError, match="Unknown constrained GPU solver 'raster_gpu_v9'"):
            gpu_v1.check_constrained_gpu_settings("raster_gpu_v9")

    def test_checked_settings_are_parameters_of_the_check(self):
        import inspect
        parameters = set(inspect.signature(gpu_v1.constrained_gpu_ignored_settings).parameters) - {"solver"}
        assert parameters == set(gpu_v1.CONSTRAINED_GPU_CHECKED_SETTINGS)


class TestSolverEntryFunctions:
    """The entry functions check before they touch the GPU, so this needs none."""

    @pytest.mark.parametrize("keyword,value,expected", V4_IGNORED, ids=[row[0] for row in V4_IGNORED])
    def test_v4_entry_raises_for_each_ignored_setting(self, keyword, value, expected):
        with pytest.raises(AlgorithmNotImplementedError, match="does not enforce") as caught:
            gpu_v4.constrained_sssp_raster_gpu_v4(**_solver_arguments(**{keyword: value}))
        assert expected in str(caught.value)

    @pytest.mark.parametrize("module,name", [(gpu_v1, "constrained_sssp_raster_gpu"),
                                             (gpu_v2, "constrained_sssp_raster_gpu_v2")])
    def test_v1_and_v2_entries_warn_about_a_max_span_beyond_the_bin_grid(self, monkeypatch, module, name):
        """They warn (their callers choose the bin grid on purpose) and then run; here: stop at "no GPU"."""
        monkeypatch.setattr(module, "GPU_AVAILABLE", False)
        with pytest.warns(UserWarning, match=rf"{name} does not enforce max_span=30 .*2 \* 20 = 40 m"):
            with pytest.raises(RuntimeError, match="CUDA GPU not available"):
                getattr(module, name)(**_solver_arguments(max_span=30.0))

    @pytest.mark.parametrize("module,name", [(gpu_v1, "constrained_sssp_raster_gpu"),
                                             (gpu_v2, "constrained_sssp_raster_gpu_v2"),
                                             (gpu_v4, "constrained_sssp_raster_gpu_v4")])
    def test_default_settings_pass_the_check_and_no_gpu_names_the_import_error(self, monkeypatch, module, name):
        """#26: "CUDA GPU not available" carries the reason the CuPy import failed."""
        cause = ImportError("No module named 'cupy'")
        monkeypatch.setattr(module, "GPU_AVAILABLE", False)
        if module is gpu_v1:
            monkeypatch.setattr(traversal_gpu, "GPU_IMPORT_ERROR", cause)
        else:
            monkeypatch.setattr(module, "_GPU_IMPORT_ERROR", cause)
        with warnings.catch_warnings():
            warnings.simplefilter("error")          # default settings: nothing to warn about
            with pytest.raises(RuntimeError,
                               match=r"CUDA GPU not available.*\(cause: ImportError: No module") as caught:
                getattr(module, name)(**_solver_arguments())
        assert caught.value.__cause__ is cause


class TestDispatch:
    """ConstrainedPathFinder decides in one place, before the GPU solver is imported or called."""

    def test_v4_with_a_dem_is_refused(self, monkeypatch, finder):
        entry = _force_gpu_backend(monkeypatch, finder, "raster_gpu_v4")
        dem_kwargs = {"dem_data": _DEM, "cell_size": 1.0, "max_gradient_pct": 100.0, "gradient_scale": 2.0}
        monkeypatch.setattr(finder, "_prepare_dem_data", lambda raster: (_DEM, dem_kwargs))
        with pytest.raises(AlgorithmNotImplementedError, match=r"'raster_gpu_v4'.*does not enforce: dem \("):
            finder.find_route()
        assert entry.calls == []

    def test_v4_with_clearance_and_gradient_limit_names_them(self, monkeypatch, raster_path):
        finder = _finder(raster_path, profile=_profile(
            min_clearance_m=7.0, max_gradient_percent=30.0, conductor_weight_per_m=15.0,
            conductor_tension_n=30000.0, tower_height_m=30.0))
        entry = _force_gpu_backend(monkeypatch, finder, "raster_gpu_v4")
        dem_kwargs = {"dem_data": _DEM, "cell_size": 1.0, "max_gradient_pct": 30.0, "gradient_scale": 2.0}
        monkeypatch.setattr(finder, "_prepare_dem_data", lambda raster: (_DEM, dem_kwargs))
        with pytest.raises(AlgorithmNotImplementedError, match=r"min_clearance=7; max_gradient_pct=30"):
            finder.find_route()
        assert entry.calls == []

    def test_v4_with_exact_tower_area_is_refused(self, monkeypatch, raster_path):
        finder = _finder(raster_path, profile=_profile(tower_ground_area_m2=9.0, tower_area_cost_mode="exact"))
        assert finder._area_offsets is not None                     # the profile asked for the exact footprint
        entry = _force_gpu_backend(monkeypatch, finder, "raster_gpu_v4")
        with pytest.raises(AlgorithmNotImplementedError, match=r"area_offsets \(exact tower footprint"):
            finder.find_route()
        assert entry.calls == []

    def test_v2_with_a_max_span_off_its_bin_grid_is_refused(self, monkeypatch, raster_path):
        """min 10 m, max 25 m -> 3 bins of 10 m: v2 would accept spans up to 30 m."""
        finder = _finder(raster_path, profile=_profile(max_span_m=25.0))
        entry = _force_gpu_backend(monkeypatch, finder, "raster_gpu")
        with pytest.raises(AlgorithmNotImplementedError, match=r"'raster_gpu'.*max_span=25 .*3 \* 10 = 30 m"):
            finder.find_route()
        assert entry.calls == []

    def test_cython_enforces_what_the_gpu_solvers_refuse(self, raster_path):
        """The same off-grid max_span is fine on the CPU backend (it tracks the span exactly)."""
        result = _finder(raster_path, profile=_profile(max_span_m=25.0)).find_route()
        assert result.max_span_actual_m <= 25.0 + 1e-6

    @pytest.mark.parametrize("backend,label", [("raster_gpu", "GPU v2"), ("raster_gpu_v4", "GPU v4")])
    def test_default_settings_reach_the_solver_and_an_import_error_falls_back_with_its_cause(
            self, monkeypatch, finder, cython_cost, backend, label):
        entry = _force_gpu_backend(monkeypatch, finder, backend, ImportError("fake: no CuPy here"))
        with pytest.warns(UserWarning) as caught:
            result = finder.find_route()
        assert len(entry.calls) == 1                                # the settings check let the solver run
        fallback = [str(w.message) for w in caught if "falling back to Cython" in str(w.message)]
        assert fallback == [f"{label} unavailable (ImportError: fake: no CuPy here), falling back to Cython"]
        assert float(result.total_cost) == pytest.approx(cython_cost)

    def test_runtime_error_falls_back_and_names_its_type(self, monkeypatch, finder, cython_cost):
        _force_gpu_backend(monkeypatch, finder, "raster_gpu_v4", RuntimeError("CUDA is not found."))
        with pytest.warns(UserWarning, match=r"GPU v4 unavailable \(RuntimeError: CUDA is not found\.\), falling"):
            result = finder.find_route()
        assert float(result.total_cost) == pytest.approx(cython_cost)

    @pytest.mark.parametrize("backend", ["raster_gpu", "raster_gpu_v4"])
    def test_programming_error_propagates(self, monkeypatch, finder, backend):
        _force_gpu_backend(monkeypatch, finder, backend, ZeroDivisionError("division by zero"))
        with _no_gpu_warning():                                     # and no "unavailable" warning either
            with pytest.raises(ZeroDivisionError):
                finder.find_route()

    def test_cuda_runtime_out_of_memory_is_reported_as_out_of_memory(self, monkeypatch, finder):
        cause = RuntimeError("cudaErrorMemoryAllocation: out of memory")
        _force_gpu_backend(monkeypatch, finder, "raster_gpu", cause)
        with _no_gpu_warning():                                     # no "unavailable", no silent CPU run
            with pytest.raises(MemoryError, match=r"GPU out of memory while running the constrained GPU v2 "
                                                  r"solver \(array shape \(40, 40\), 3,200 bytes\)") as caught:
                finder.find_route()
        assert caught.value.__cause__ is cause

    @needs_cupy_class
    def test_cupy_out_of_memory_propagates_as_memory_error(self, monkeypatch, finder):
        cause = cupy.cuda.memory.OutOfMemoryError(1024, 0)
        _force_gpu_backend(monkeypatch, finder, "raster_gpu_v4", cause)
        with _no_gpu_warning():
            with pytest.raises(MemoryError, match="Out of memory allocating 1,024 bytes") as caught:
                finder.find_route()
        assert caught.value is cause


# ---------------------------------------------------------------------------
# #26  GPU failures keep their cause  (CPU only: the failing call is a fake)
# ---------------------------------------------------------------------------

def _api_without_device(monkeypatch, session_factory) -> raster_gpu_api.RasterGPUAPI:
    """A RasterGPUAPI whose device session is built by ``session_factory`` (bypasses __init__: needs no CuPy)."""
    api = object.__new__(raster_gpu_api.RasterGPUAPI)
    api.raster_data = np.full((30, 50), 7, dtype=np.uint16)
    api.steps = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=np.int8)
    api.ignore_max = True
    api.dem_data = None
    api.gradient_luts = None
    api._delta = "auto"
    api._arena_factor = 1.0
    api._session = None
    api._session_unavailable = False
    api._session_error = None
    monkeypatch.setattr(raster_gpu_api, "RASTER_GPU_AVAILABLE", True)
    monkeypatch.setattr(raster_gpu_api, "GpuSsspSession", session_factory)
    return api


def _raising(error):
    def factory(*args, **kwargs):
        raise error
    return factory


class TestRasterGpuSession:

    @pytest.mark.parametrize("error", [
        ImportError("No module named 'nvidia'"),
        RuntimeError("GpuSsspSession requires the V5 async kernel"),
    ], ids=["ImportError", "RuntimeError"])
    def test_unavailable_session_warns_once_with_the_cause(self, monkeypatch, error):
        api = _api_without_device(monkeypatch, _raising(error))
        with pytest.warns(UserWarning) as caught:
            assert api._get_session() is None
        assert [str(w.message) for w in caught] == [
            f"raster_gpu: no device session ({type(error).__name__}: {error}); this API object solves with the "
            "stateless GPU kernel (V4/V3) instead"]
        assert api._session_error is error and api._session_unavailable
        with warnings.catch_warnings():
            warnings.simplefilter("error")                          # asked again: no second attempt, no warning
            assert api._get_session() is None

    def test_programming_error_propagates(self, monkeypatch):
        api = _api_without_device(monkeypatch, _raising(ZeroDivisionError("division by zero")))
        with pytest.raises(ZeroDivisionError):
            api._get_session()
        assert not api._session_unavailable and api._session_error is None

    def test_cuda_driver_out_of_memory_is_reported_as_out_of_memory(self, monkeypatch):
        cause = RuntimeError("CUDA_ERROR_OUT_OF_MEMORY: out of memory")
        api = _api_without_device(monkeypatch, _raising(cause))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(MemoryError, match=r"GPU out of memory while allocating the raster_gpu device "
                                                  r"session \(array shape \(30, 50\), 3,000 bytes\)") as caught:
                api._get_session()
        assert caught.value.__cause__ is cause
        assert not api._session_unavailable                         # not "unavailable": the next query retries

    @needs_cupy_class
    def test_cupy_out_of_memory_is_reported_as_out_of_memory(self, monkeypatch):
        cause = cupy.cuda.memory.OutOfMemoryError(4096, 0)
        api = _api_without_device(monkeypatch, _raising(cause))
        with pytest.raises(MemoryError, match=r"GPU out of memory .*\(30, 50\).*OutOfMemoryError: Out of memory "
                                              r"allocating 4,096 bytes") as caught:
            api._get_session()
        assert caught.value.__cause__ is cause
        assert not api._session_unavailable

    def test_constructor_without_cupy_names_the_import_error(self, monkeypatch):
        cause = ImportError("No module named 'cupy'")
        monkeypatch.setattr(raster_gpu_api, "RASTER_GPU_AVAILABLE", False)
        monkeypatch.setattr(raster_gpu_api, "_RASTER_GPU_IMPORT_ERROR", cause)
        with pytest.raises(ImportError,
                           match=r"requires CuPy.*\(cause: ImportError: No module named 'cupy'\)") as caught:
            raster_gpu_api.RasterGPUAPI(np.zeros((4, 4), dtype=np.uint16), np.array([[0, 1]], dtype=np.int8))
        assert caught.value.__cause__ is cause


def _probe_cases():
    """(module, probe, flag, cause attribute, name of the first call inside the probe's try block)."""
    return [
        pytest.param(sssp_gpu, "_check_v5_available", "_v5_available", "_v5_unavailable_cause",
                     "_ensure_cuda_path", id="sssp-v5"),
        pytest.param(sssp_gpu, "_check_v4_available", "_v4_available", "_v4_unavailable_cause",
                     "_ensure_cuda_path", id="sssp-v4"),
        pytest.param(gpu_v2, "_check_v2_available", "_v2_available", "_v2_unavailable_cause",
                     "_ensure_cuda_path", id="constrained-v2"),
        pytest.param(gpu_v4, "_check_v4_available", "_v4_available", "_v4_unavailable_cause",
                     "_load_kernel_source", id="constrained-v4"),
    ]


@pytest.mark.parametrize("module,probe,flag,cause_name,first_call", _probe_cases())
class TestKernelProbes:
    """The compile probes: only import/CUDA errors mean "unavailable", and the cause is kept."""

    @staticmethod
    def _arm(monkeypatch, module, flag, cause_name, first_call, error):
        monkeypatch.setattr(module, flag, None)                     # forget the cached answer (restored afterwards)
        monkeypatch.setattr(module, cause_name, None)
        monkeypatch.setattr(module, "GPU_AVAILABLE", True)
        monkeypatch.setattr(module, first_call, _raising(error))

    @pytest.mark.parametrize("error", [
        ImportError("No module named 'cupy.cuda.compiler'"),
        RuntimeError("Relocatable PTX code is requested, but cudadevrt is not found"),
        FileNotFoundError("sssp_async_v5.cu"),
    ], ids=["ImportError", "RuntimeError", "OSError"])
    def test_unavailable_keeps_and_reports_the_cause(self, monkeypatch, module, probe, flag, cause_name,
                                                     first_call, error):
        self._arm(monkeypatch, module, flag, cause_name, first_call, error)
        with pytest.warns(UserWarning, match="kernel unavailable") as caught:
            assert getattr(module, probe)() is False
        assert f"({type(error).__name__}: {error})" in str(caught[0].message)
        assert getattr(module, cause_name) is error
        with warnings.catch_warnings():
            warnings.simplefilter("error")                          # cached: reported once per process
            assert getattr(module, probe)() is False

    def test_programming_error_propagates(self, monkeypatch, module, probe, flag, cause_name, first_call):
        self._arm(monkeypatch, module, flag, cause_name, first_call, ZeroDivisionError("division by zero"))
        with pytest.raises(ZeroDivisionError):
            getattr(module, probe)()
        assert getattr(module, flag) is None                        # no cached "unavailable"

    def test_cuda_out_of_memory_is_not_unavailable(self, monkeypatch, module, probe, flag, cause_name, first_call):
        cause = RuntimeError("cudaErrorMemoryAllocation: out of memory")
        self._arm(monkeypatch, module, flag, cause_name, first_call, cause)
        with pytest.raises(MemoryError, match="GPU out of memory while compiling the .* kernel") as caught:
            getattr(module, probe)()
        assert caught.value.__cause__ is cause
        assert getattr(module, flag) is None

    @needs_cupy_class
    def test_cupy_out_of_memory_propagates(self, monkeypatch, module, probe, flag, cause_name, first_call):
        cause = cupy.cuda.memory.OutOfMemoryError(2048, 0)
        self._arm(monkeypatch, module, flag, cause_name, first_call, cause)
        with pytest.raises(MemoryError, match="Out of memory allocating 2,048 bytes") as caught:
            getattr(module, probe)()
        assert caught.value is cause
        assert getattr(module, flag) is None


class TestGpuErrorHelpers:

    def test_out_of_memory_is_recognised_in_all_three_spellings(self):
        assert traversal_gpu.is_gpu_out_of_memory(MemoryError())
        assert traversal_gpu.is_gpu_out_of_memory(RuntimeError("cudaErrorMemoryAllocation: out of memory"))
        assert traversal_gpu.is_gpu_out_of_memory(RuntimeError("CUDA_ERROR_OUT_OF_MEMORY: out of memory"))
        assert not traversal_gpu.is_gpu_out_of_memory(RuntimeError("CUDA is not found."))
        assert not traversal_gpu.is_gpu_out_of_memory(ValueError("out of memory"))

    def test_probe_errors_exclude_programming_errors(self):
        errors = traversal_gpu.gpu_probe_errors()
        for unavailable in (ImportError, ModuleNotFoundError, OSError, FileNotFoundError, RuntimeError):
            assert issubclass(unavailable, errors)
        for bug in (ZeroDivisionError, TypeError, ValueError, AttributeError, KeyError, NameError, MemoryError):
            assert not issubclass(bug, errors)

    @needs_cupy_class
    def test_probe_errors_cover_the_cupy_cuda_errors(self):
        from cupy.cuda.compiler import CompileException
        from cupy_backends.cuda.api.driver import CUDADriverError
        from cupy_backends.cuda.api.runtime import CUDARuntimeError
        errors = traversal_gpu.gpu_probe_errors()
        for cuda_error in (CompileException, CUDADriverError, CUDARuntimeError):
            assert issubclass(cuda_error, errors)
        assert not issubclass(cupy.cuda.memory.OutOfMemoryError, errors)

    def test_not_available_error_keeps_the_cause(self):
        cause = ImportError("DLL load failed")
        error = traversal_gpu.gpu_not_available_error("CUDA GPU not available", cause)
        assert isinstance(error, RuntimeError) and error.__cause__ is cause
        assert str(error) == "CUDA GPU not available (cause: ImportError: DLL load failed)"


# ---------------------------------------------------------------------------
# Real GPU: the paths above still run when nothing fails
# ---------------------------------------------------------------------------

@gpu_only
def test_real_session_is_built_without_a_fallback_warning():
    """A 32 x 32 raster (a few kB on the device): the session path is taken, silently."""
    if not sssp_gpu._check_v5_available():
        pytest.skip(f"V5 kernel unavailable here: {sssp_gpu._v5_unavailable_cause!r}")
    steps = np.array([[0, 1], [1, 0], [0, -1], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=np.int8)
    with raster_gpu_api.RasterGPUAPI(np.full((32, 32), 10, dtype=np.uint16), steps) as api:
        with _no_gpu_warning():
            path = api.shortest_path(0, 32 * 32 - 1)
        assert api._session is not None and api._session_error is None and not api._session_unavailable
        assert path[0] == 0 and path[-1] == 32 * 32 - 1


@gpu_only
@constrained_gpu_only
def test_real_constrained_gpu_run_with_default_settings_is_unchanged(raster_path, cython_cost):
    """Default settings: the GPU v2 solver runs (no refusal, no fallback) and lands where the CPU solver does.

    The tolerance is the one of ``test_gpu_backend_matches_cython_basic`` (test_constrained_gpu_v2.py): the v2
    kernel tracks spans in float16 and on a bin grid, so it is close to the Cython optimum, not identical.
    """
    if not gpu_v2._check_v2_available():
        pytest.skip(f"constrained GPU v2 kernel unavailable here: {gpu_v2._v2_unavailable_cause!r}")
    gpu_finder = ConstrainedPathFinder(raster_path, SOURCE, TARGET, profile=_profile(),
                                       graph_api="raster_gpu", neighborhood_str="r1")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = gpu_finder.find_route()
    messages = [str(w.message) for w in caught]
    assert not [m for m in messages if "falling back to Cython" in m or "does not enforce" in m], messages
    assert len(result.path_indices) > 0
    assert 0.8 < float(result.total_cost) / cython_cost < 1.2


# ---------------------------------------------------------------------------
# Line-end settings (round 17, P3): no GPU solver has them, so every GPU solver refuses them  (CPU only)
# ---------------------------------------------------------------------------

#: (keyword of the settings check, a non-neutral value, text the error must contain)
LINE_END_SETTINGS = [
    ("target_min_span", 3.0, "target_min_span=3 (terminal_span_rule='enforce'"),
    ("source_attach_height", 12.0, "source_attach_height=12 (source_connection_height_m"),
    ("target_attach_height", 15.5, "target_attach_height=15.5 (target_connection_height_m"),
]


class TestLineEndSettings:

    @pytest.mark.parametrize("solver", sorted(gpu_v1.CONSTRAINED_GPU_SOLVERS))
    @pytest.mark.parametrize("keyword,value,expected", LINE_END_SETTINGS, ids=[row[0] for row in LINE_END_SETTINGS])
    def test_every_solver_refuses_each_line_end_setting(self, solver, keyword, value, expected):
        with pytest.raises(AlgorithmNotImplementedError) as caught:
            gpu_v1.check_constrained_gpu_settings(solver, **{keyword: value})
        message = str(caught.value)
        assert expected in message
        assert f"'{solver}'" in message and "graph_api='cython'" in message
        assert isinstance(caught.value, PyorpsError)

    @pytest.mark.parametrize("solver", sorted(gpu_v1.CONSTRAINED_GPU_SOLVERS))
    def test_the_neutral_values_pass(self, solver):
        gpu_v1.check_constrained_gpu_settings(
            solver, target_min_span=0.0, source_attach_height=None, target_attach_height=None)
        gpu_v1.check_constrained_gpu_settings(solver, target_min_span=None)

    def test_all_three_are_named_at_once(self):
        with pytest.raises(AlgorithmNotImplementedError,
                           match=r"target_min_span=3 .*source_attach_height=12 .*target_attach_height=15"):
            gpu_v1.check_constrained_gpu_settings(
                "raster_gpu", target_min_span=3.0, source_attach_height=12.0, target_attach_height=15.0)

    def test_the_three_settings_are_in_the_table_of_checked_settings(self):
        assert {"target_min_span", "source_attach_height", "target_attach_height"} <= set(
            gpu_v1.CONSTRAINED_GPU_CHECKED_SETTINGS)

    @pytest.mark.parametrize("backend", ["raster_gpu", "raster_gpu_v4"])
    def test_enforce_is_refused_before_the_gpu_solver_is_reached(self, monkeypatch, raster_path, backend):
        finder = _finder(raster_path, profile=_profile(terminal_span_rule="enforce"))
        entry = _force_gpu_backend(monkeypatch, finder, backend)
        with pytest.raises(AlgorithmNotImplementedError, match=r"does not enforce: target_min_span=10 "):
            finder.find_route()
        assert entry.calls == []

    @pytest.mark.parametrize("backend", ["raster_gpu", "raster_gpu_v4"])
    def test_a_connection_height_the_search_would_use_is_refused(self, monkeypatch, raster_path, backend):
        finder = _finder(raster_path, profile=_profile(
            conductor_weight_per_m=15.0, conductor_tension_n=30000.0, min_clearance_m=7.0, tower_height_m=30.0))
        entry = _force_gpu_backend(monkeypatch, finder, backend)
        dem_kwargs = {"dem_data": _DEM, "cell_size": 1.0, "max_gradient_pct": 100.0, "gradient_scale": 2.0}
        monkeypatch.setattr(finder, "_prepare_dem_data", lambda raster: (_DEM, dem_kwargs))
        # the DEM and the clearance data alone would be fine for v2 (it enforces them), not for v4
        with pytest.raises(AlgorithmNotImplementedError, match=r"target_attach_height=14 \(target_connection_height_m"):
            finder.find_route(target_connection_height_m=14.0)
        assert entry.calls == []

    def test_a_connection_height_that_cannot_act_is_no_reason_to_refuse_a_gpu_solver(self, monkeypatch, finder):
        """Without a DEM the height changes nothing in any backend: the solver is reached (and warns once that the
        height has no effect)."""
        entry = _force_gpu_backend(monkeypatch, finder, "raster_gpu_v4", ImportError("fake: no CuPy here"))
        with pytest.warns(UserWarning, match="has no effect"):
            finder.find_route(target_connection_height_m=14.0)
        assert len(entry.calls) == 1

    def test_the_gpu_solver_prices_per_cell_warning_is_given_once_per_finder_and_backend(self, monkeypatch, finder):
        finder._cell_size = 5.0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            finder._warn_gpu_cell_size("raster_gpu")
            finder._warn_gpu_cell_size("raster_gpu")
            finder._warn_gpu_cell_size("raster_gpu_v4")
        messages = [str(w.message) for w in caught]
        assert len(messages) == 2 and "'raster_gpu'" in messages[0] and "'raster_gpu_v4'" in messages[1]
        assert all("per cell" in m and "per metre" in m for m in messages)
        finder._cell_size = 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            finder._warn_gpu_cell_size("raster_gpu")      # cell size 1: nothing to say (and already said)
