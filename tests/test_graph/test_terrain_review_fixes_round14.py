"""Fixes after the independent review of the terrain-on-every-backend change (round 14).

One section per finding of the review:

1. what a bare DEM means (metres, ``exp(2 s^2)``, 0.25 % bins up to 200 %) at a cell size other than 1 m, and the
   texts that say it;
2. ``CostField`` / ``CostFieldSet`` on ``raster_gpu`` with a DEM run as the Dijkstra tree (one warning); an explicit
   delta-stepping or GPU algorithm is refused with advice that works;
3. ``dem_kwargs["gradient_function"]`` is a slope option, never a raster read option;
4. float32 on a backend without float32 kernels raises ``FeatureNotImplementedError`` (see also
   ``test_float_precision.py`` and ``test_dem_backend_agreement_round14.py``);
5. a fallback is recorded as announced only after its warning went out;
6. a session's fallback warning is attributed to the caller's line, however the session was opened;
7. what happens to DEM voids, and what the texts say about it;
8. a session on a graph-library finder costs what ``find_route`` costs, with and without a DEM.

The GPU cases follow the guard of ``test_dem_backend_agreement_round14.py`` and use rasters of a few hundred cells.
"""

from __future__ import annotations

import importlib.util
import inspect
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pytest
import rasterio
from exactness_referee import EdgeModel, reprice_path
from rasterio.transform import from_origin

from pyorps import AlgorithmFallbackWarning, GradientOptions, Objective, PathFinder
from pyorps.core.exceptions import FeatureNotImplementedError, InvalidInputError
from pyorps.graph import path_finder as path_finder_module
from pyorps.graph.search_session import SearchSession
from pyorps.utils.neighborhood import get_neighborhood_steps

CRS = "EPSG:25832"
THIS_FILE = Path(__file__).resolve()
DOCS_PAGE = THIS_FILE.parents[2] / "docs" / "source" / "api" / "metric_stack_objective.md"


def _installed(package: str) -> bool:
    try:
        return importlib.util.find_spec(package) is not None
    except (ImportError, ValueError):
        return False


def _gpu_usable() -> bool:
    """CuPy is importable and sees a CUDA device (the guard of test_backend_parity.py)."""
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # any failure to reach a device means "no GPU"
        return False


GPU = _gpu_usable()
needs_gpu = pytest.mark.skipif(not GPU, reason="no usable CUDA GPU (CuPy missing or no device)")

LIBRARIES = [pytest.param(name, id=name, marks=pytest.mark.skipif(not _installed(name), reason=f"{name} not installed"))
             for name in ("networkx", "networkit", "igraph", "rustworkx")]


def _fallbacks(caught) -> list:
    return [w for w in caught if issubclass(w.category, AlgorithmFallbackWarning)]


def _same_file(filename: str) -> bool:
    return os.path.samefile(filename, THIS_FILE)


# ====================================================================================================== 1. 5 m cells

CELL_5 = 5.0
RIDGE_ROWS, RIDGE_COLS = 21, 41
RIDGE_SLOPE_RISE = 0.50625          # m per 5 m cell: s = 10.125 %, the middle of the 0.25 % bin 40 (never on an edge)
RIDGE_TRANSFORM = from_origin(0.0, RIDGE_ROWS * CELL_5, CELL_5, CELL_5)


def _xy5(row: int, col: int) -> tuple[float, float]:
    return ((col + 0.5) * CELL_5, RIDGE_ROWS * CELL_5 - (row + 0.5) * CELL_5)


def ridge_dem(rise_per_cell: float) -> np.ndarray:
    """A ridge across columns 10..30 of rows 0..14 (up 10 cells, down 10 cells); rows 15..20 are a flat gap."""
    dem = np.zeros((RIDGE_ROWS, RIDGE_COLS), dtype=np.float32)
    for col in range(RIDGE_COLS):
        dem[:15, col] = rise_per_cell * max(0, min(col - 10, 30 - col, 10))
    return dem


def ridge_finder(graph_api: str, dem: np.ndarray, **kwargs) -> PathFinder:
    source, target = _xy5(5, 2), _xy5(5, 38)
    return PathFinder(
        dataset_source=np.full((RIDGE_ROWS, RIDGE_COLS), 100, dtype=np.uint16), crs=CRS,
        transform=RIDGE_TRANSFORM, source_coords=source, target_coords=target, search_space_buffer_m=500,
        graph_api=graph_api, dem=dem, **kwargs)


def _peak(dem: np.ndarray, cells) -> float:
    cells = np.asarray(cells)
    return float(dem[cells // RIDGE_COLS, cells % RIDGE_COLS].max())


BACKENDS_5M = [pytest.param("cython", id="cython")] + LIBRARIES


@pytest.mark.parametrize("graph_api", BACKENDS_5M)
def test_a_bare_dem_is_priced_in_metres_at_5_m_cells(graph_api):
    """A 10 % slope on 5 m cells is a 10 % slope (penalty ~1.03), and a 100 % slope is one (penalty ~7.4)."""
    gentle, steep = ridge_dem(RIDGE_SLOPE_RISE), ridge_dem(10 * RIDGE_SLOPE_RISE)   # 10.125 % and 101.25 %
    source, target = _xy5(5, 2), _xy5(5, 38)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        over_gentle = ridge_finder(graph_api, gentle).find_route(source, target)
        around_steep = ridge_finder(graph_api, steep).find_route(source, target)
    # the gentle ridge costs ~0.5 cell lengths to cross, the detour through the gap 3 or more: the route crosses it
    assert _peak(gentle, over_gentle.path_indices) == pytest.approx(10 * RIDGE_SLOPE_RISE, rel=1e-6)
    # the steep ridge costs ~20 x 7.4 cell lengths: the route leaves the straight line and stays on flat ground
    assert _peak(steep, around_steep.path_indices) == 0.0
    # and the reported length is the 3-D length in metres
    run2 = [math.hypot((b % RIDGE_COLS - a % RIDGE_COLS), (b // RIDGE_COLS - a // RIDGE_COLS)) * CELL_5
            for a, b in zip(over_gentle.path_indices[:-1], over_gentle.path_indices[1:])]
    rise = [abs(float(gentle[b // RIDGE_COLS, b % RIDGE_COLS]) - float(gentle[a // RIDGE_COLS, a % RIDGE_COLS]))
            for a, b in zip(over_gentle.path_indices[:-1], over_gentle.path_indices[1:])]
    assert over_gentle.total_length == pytest.approx(sum(math.hypot(r, z) for r, z in zip(run2, rise)), rel=1e-6)


@pytest.mark.parametrize("graph_api", BACKENDS_5M)
def test_the_slope_table_of_a_bare_dem_at_5_m_cells(graph_api):
    """The tables every backend reads: run in metres, factor exp(2 s^2) on top of the 3-D stretch, 0.25 % bins."""
    finder = ridge_finder(graph_api, ridge_dem(RIDGE_SLOPE_RISE))
    luts = finder.graph_api.gradient_luts
    steps = np.asarray(finder.steps)
    orthogonal = int(np.flatnonzero((steps[:, 0] == 0) & (steps[:, 1] == 1))[0])
    # run of an orthogonal step: 1 / (5 m); one bin is 0.25 %, so 100 / (5 m) / 0.25 = 80 bins per metre of rise
    assert float(luts.inv_horiz_m[orthogonal]) == pytest.approx(1.0 / CELL_5, rel=1e-6)
    assert float(luts.bin_factor[orthogonal]) == pytest.approx(80.0, rel=1e-6)
    s = 0.10125
    slope_bin = int(RIDGE_SLOPE_RISE * float(luts.bin_factor[orthogonal]))
    assert slope_bin == 40
    assert float(luts.mult[slope_bin]) == pytest.approx(math.sqrt(1 + s * s) * math.exp(2 * s * s), rel=1e-6)
    # the table ends at 200 %: the largest penalty is exp(8) (the stretch comes on top), not a cap of 10 000
    s_last = 1.99875
    penalty_last = float(luts.mult[-1]) / math.sqrt(1 + s_last * s_last)
    assert penalty_last == pytest.approx(math.exp(2 * s_last * s_last), rel=1e-5)
    assert math.exp(2 * 1.9 ** 2) < penalty_last < math.exp(8) < 3000


@needs_gpu
def test_the_5_m_gentle_ridge_is_crossed_on_the_gpu_too():
    gentle = ridge_dem(RIDGE_SLOPE_RISE)
    finder = ridge_finder("raster_gpu", gentle)
    try:
        path = finder.find_route(_xy5(5, 2), _xy5(5, 38))
        assert _peak(gentle, path.path_indices) == pytest.approx(10 * RIDGE_SLOPE_RISE, rel=1e-6)
    finally:
        finder.release_device_resources()


def _unwrapped_doc(obj) -> str:
    return " ".join(inspect.getdoc(obj).split())


def test_the_texts_do_not_claim_the_old_graph_library_behaviour():
    """The new formula FIXES the old one at cell sizes other than 1 m; no text may say it equals the past."""
    docs = {
        "_bare_dem_objective": _unwrapped_doc(path_finder_module._bare_dem_objective),
        "PathFinder": _unwrapped_doc(PathFinder),
        "PathFinder.__init__": _unwrapped_doc(PathFinder.__init__),
    }
    if DOCS_PAGE.exists():
        docs["metric_stack_objective.md"] = " ".join(DOCS_PAGE.read_text(encoding="utf-8").split())
    for name, text in docs.items():
        assert "always did" not in text and "always built" not in text, name
        assert "capped at 10 000" not in text.replace("not a cap of 10 000", ""), name
    for name in ("_bare_dem_objective", "PathFinder.__init__"):
        text = docs[name]
        assert "FIXES" in text and "other than 1 m" in text, name
        assert "exp(8)" in text and "2981" in text, name
    page = docs.get("metric_stack_objective.md")
    if page is not None:
        assert "fixes the old graph-library behaviour at cell sizes other than 1 m" in page
        assert "exp(8)" in page and "2981" in page
        assert "capped at 10 000" not in page.replace("not a cap of 10 000", "")


# ====================================================================================================== common scene

N = 15
SCENE_TRANSFORM = from_origin(0.0, float(N), 1.0, 1.0)
SOURCE = (1.5, N - 0.5 - 7)           # row 7, column 1
TARGET = (N - 1.5, N - 0.5 - 7)       # row 7, column 13


def scene_dem() -> np.ndarray:
    """A cone, 200 % slope, on the line between SOURCE and TARGET."""
    rows, cols = np.mgrid[0:N, 0:N]
    return np.maximum(0.0, 2.0 * (4.0 - np.hypot(rows - 7, cols - 7))).astype(np.float32)


def scene_finder(graph_api: str, dem=None, objective=None, **kwargs) -> PathFinder:
    return PathFinder(
        dataset_source=np.full((N, N), 100, dtype=np.uint16), crs=CRS, transform=SCENE_TRANSFORM,
        source_coords=kwargs.pop("source_coords", SOURCE), target_coords=kwargs.pop("target_coords", TARGET),
        search_space_buffer_m=kwargs.pop("search_space_buffer_m", 100), graph_api=graph_api, dem=dem,
        objective=objective, **kwargs)


# ====================================================================================================== 2. cost fields

def _field_costs(field) -> np.ndarray:
    return np.asarray(field.costs_to([TARGET, (N - 1.5, N - 0.5 - 3)]))


def test_a_cython_cost_field_with_a_dem_is_dijkstra_and_silent():
    """"auto" and "dijkstra" settle the field by the Dijkstra tree; nothing needs announcing on cython."""
    reference = scene_finder("cython", scene_dem())
    expected = None
    for algorithm in ("auto", "dijkstra"):
        finder = scene_finder("cython", scene_dem())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with finder.cost_field(SOURCE, algorithm=algorithm) as field:
                costs = _field_costs(field)
                assert field.algorithm == "dijkstra"
        assert not _fallbacks(caught)
        if expected is None:
            expected = costs
        np.testing.assert_allclose(costs, expected, rtol=1e-12)
    reference.release_device_resources()


def test_an_explicit_delta_stepping_field_with_a_dem_is_refused_with_advice_that_works():
    finder = scene_finder("cython", scene_dem())
    with pytest.raises(FeatureNotImplementedError, match=r"algorithm='dijkstra' or algorithm='auto'"):
        finder.cost_field(SOURCE, algorithm="delta-stepping")
    # the advice works
    for algorithm in ("auto", "dijkstra"):
        with finder.cost_field(SOURCE, algorithm=algorithm) as field:
            assert np.isfinite(_field_costs(field)).all()


def test_cost_fields_of_a_cython_finder_with_a_dem_settle_as_dijkstra():
    finder = scene_finder("cython", scene_dem())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with finder.cost_fields([SOURCE, TARGET]) as fields:
            assert fields.algorithm == "dijkstra"
            assert np.isfinite(fields.costs_to([TARGET])).all()
    assert not _fallbacks(caught)


@needs_gpu
@pytest.mark.parametrize("algorithm", ["auto", "dijkstra"])
def test_a_gpu_cost_field_with_a_dem_is_settled_by_dijkstra_with_one_warning(algorithm):
    reference = scene_finder("cython", scene_dem())
    finder = scene_finder("raster_gpu", scene_dem())
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            field = finder.cost_field(SOURCE, algorithm=algorithm)         # raised FeatureNotImplementedError before
            other = finder.cost_field(TARGET, algorithm=algorithm)         # the same finder announces once
        assert len(_fallbacks(caught)) == 1, [str(w.message) for w in _fallbacks(caught)]
        assert _same_file(_fallbacks(caught)[0].filename), _fallbacks(caught)[0].filename
        assert field.algorithm == "dijkstra" and other.algorithm == "dijkstra"
        with reference.cost_field(SOURCE, algorithm="dijkstra") as expected:
            np.testing.assert_allclose(_field_costs(field), _field_costs(expected), rtol=1e-12)
        field.close()
        other.close()
    finally:
        finder.release_device_resources()
        reference.release_device_resources()


@needs_gpu
def test_a_gpu_cost_field_set_with_a_dem_is_settled_by_dijkstra_with_one_warning():
    finder = scene_finder("raster_gpu", scene_dem())
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fields = finder.cost_fields([SOURCE, TARGET, (N - 1.5, N - 0.5 - 3)], spill=False)
        assert len(_fallbacks(caught)) == 1, [str(w.message) for w in _fallbacks(caught)]
        assert fields.algorithm == "dijkstra"
        assert np.isfinite(fields.costs_to([TARGET])).all()
        fields.close()
    finally:
        finder.release_device_resources()


@needs_gpu
@pytest.mark.parametrize("algorithm", ["delta-stepping", "delta-stepping-circular"])
def test_an_explicit_gpu_field_with_a_dem_is_refused_with_advice_that_works(algorithm):
    finder = scene_finder("raster_gpu", scene_dem())
    try:
        with pytest.raises(FeatureNotImplementedError, match=r"algorithm='dijkstra' or algorithm='auto'"):
            finder.cost_field(SOURCE, algorithm=algorithm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AlgorithmFallbackWarning)
            with finder.cost_field(SOURCE, algorithm="auto") as field:      # the advice works
                assert np.isfinite(_field_costs(field)).all()
    finally:
        finder.release_device_resources()


@needs_gpu
def test_a_gpu_cost_field_without_a_dem_still_uses_the_gpu_kernel():
    finder = scene_finder("raster_gpu", None)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with finder.cost_field(SOURCE) as field:
                assert field._kind == "gpu"
        assert not _fallbacks(caught)
    finally:
        finder.release_device_resources()


# ====================================================================================================== 3. dem_kwargs

def _write_dem(path: Path, dem: np.ndarray) -> str:
    with rasterio.open(path, "w", driver="GTiff", height=N, width=N, count=1, dtype="float32", crs=CRS,
                       transform=SCENE_TRANSFORM) as dataset:
        dataset.write(dem, 1)
    return str(path)


def test_dem_read_options_leave_out_the_slope_options():
    keep = path_finder_module._dem_read_options({"gradient_function": "power", "indexes": 1})
    assert keep == {"indexes": 1}
    assert path_finder_module._dem_read_options(None) == {}
    assert path_finder_module._dem_read_options({"gradient_function": "power"}) == {}


@pytest.mark.parametrize("corridor_first", [True, False])
@pytest.mark.parametrize("function", ["power", "exponential"])
def test_gradient_function_is_a_slope_option_not_a_read_option_for_a_dem_file(tmp_path, corridor_first, function):
    """It raised ``TypeError: read() got an unexpected keyword argument 'gradient_function'`` before."""
    dem_file = _write_dem(tmp_path / "dem.tif", scene_dem())
    bare = scene_finder("cython", dem_file, dem_kwargs={"gradient_function": function}, search_space_buffer_m=4,
                        corridor_first=corridor_first)
    explicit = scene_finder("cython", dem_file, Objective({"cost": 1.0}, GradientOptions(multiplier=function)),
                            search_space_buffer_m=4, corridor_first=corridor_first)
    assert list(bare.find_route().path_indices) == list(explicit.find_route().path_indices)


def test_gradient_function_still_picks_the_slope_model_for_a_dem_file(tmp_path):
    """Taking the key out of the read options must not take it out of the model choice."""
    dem_file = _write_dem(tmp_path / "dem.tif", scene_dem())
    flat = scene_finder("cython", None).find_route()
    for function in ("power", "energy", "sigmoid"):
        route = scene_finder("cython", dem_file, dem_kwargs={"gradient_function": function}).find_route()
        assert max(float(scene_dem()[c // N, c % N]) for c in route.path_indices) \
            < max(float(scene_dem()[c // N, c % N]) for c in flat.path_indices), function


@pytest.mark.parametrize("corridor_first", [True, False])
def test_gradient_function_survives_a_session_window_rebuild_with_a_dem_file(tmp_path, corridor_first):
    dem_file = _write_dem(tmp_path / "dem.tif", scene_dem())
    near = (6.5, N - 0.5 - 7)
    finder = scene_finder("cython", dem_file, dem_kwargs={"gradient_function": "power"}, target_coords=near,
                          search_space_buffer_m=2, corridor_first=corridor_first)
    session = finder.search_session()
    try:
        session.route([SOURCE, near])
        updated = session.update([SOURCE, TARGET])           # outside the 2 m window: the window is rebuilt
        assert session.finder is not finder
        assert len(updated.path_indices) > 1
        fresh = scene_finder("cython", dem_file, dem_kwargs={"gradient_function": "power"},
                             search_space_buffer_m=2, corridor_first=corridor_first,
                             source_coords=[SOURCE, TARGET], target_coords=[TARGET]).find_route(SOURCE, TARGET)
        assert list(updated.path_indices) == list(fresh.path_indices)
    finally:
        session.close()


# ====================================================================================================== 4. float32

def test_float32_on_cython_is_the_documented_feature_not_implemented_error():
    with pytest.raises(FeatureNotImplementedError, match="cython") as raised:
        scene_finder("cython", None, {"cost": 1.0}, weight_precision="float32")
    assert isinstance(raised.value, NotImplementedError)
    assert "uint16" in str(raised.value)


def test_the_other_float32_refusals_stay_invalid_input_errors():
    with pytest.raises(InvalidInputError, match="objective"):
        scene_finder("cython", None, None, weight_precision="float32")
    with pytest.raises(InvalidInputError, match="weight_precision"):
        scene_finder("cython", None, {"cost": 1.0}, weight_precision="float64")


# ====================================================================================================== 5. warn first

def _delta_dem_finder() -> PathFinder:
    return scene_finder("cython", scene_dem())


def test_a_fallback_that_raised_as_a_warning_is_announced_again():
    """Recorded before the warning went out, the second call under the error filter used to run silently."""
    finder = _delta_dem_finder()
    with warnings.catch_warnings():
        warnings.simplefilter("error", AlgorithmFallbackWarning)
        for _ in range(2):
            with pytest.raises(AlgorithmFallbackWarning):
                finder.find_route(algorithm="delta-stepping")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        finder.find_route(algorithm="delta-stepping")
        finder.find_route(algorithm="delta-stepping")
    assert len(_fallbacks(caught)) == 1


def test_warn_fallback_records_the_kind_only_after_the_warning():
    finder = _delta_dem_finder()
    with warnings.catch_warnings():
        warnings.simplefilter("error", AlgorithmFallbackWarning)
        for _ in range(2):
            with pytest.raises(AlgorithmFallbackWarning):
                finder._warn_fallback("probe", "first")
    assert "probe" not in finder._fallback_warned
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        finder._warn_fallback("probe", "second")
        finder._warn_fallback("probe", "third")
    assert [str(w.message) for w in _fallbacks(caught)] == ["second"]
    assert "probe" in finder._fallback_warned


def test_a_session_opened_under_the_error_filter_raises_every_time():
    finder = _delta_dem_finder()
    with warnings.catch_warnings():
        warnings.simplefilter("error", AlgorithmFallbackWarning)
        for _ in range(2):
            with pytest.raises(AlgorithmFallbackWarning):
                finder.search_session(algorithm="delta-stepping")


# ====================================================================================================== 6. attribution

def test_the_session_warning_points_at_the_line_that_opened_the_session():
    finder = _delta_dem_finder()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        via_finder = finder.search_session(algorithm="delta-stepping")          # line A
        direct = SearchSession(finder, algorithm="delta-stepping")              # line B
    try:
        warned = _fallbacks(caught)
        assert len(warned) == 2
        for w in warned:
            assert _same_file(w.filename), f"attributed to {w.filename}:{w.lineno}"
        assert warned[1].lineno == warned[0].lineno + 1
    finally:
        via_finder.close()
        direct.close()


def test_the_find_route_warning_points_at_the_calling_line():
    finder = _delta_dem_finder()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        finder.find_route(algorithm="delta-stepping")
    assert len(_fallbacks(caught)) == 1
    assert _same_file(_fallbacks(caught)[0].filename), _fallbacks(caught)[0].filename


# ====================================================================================================== 7. voids

def _void_wall_finder(**kwargs) -> PathFinder:
    """A flat DEM with a void wall across the whole raster: the only way from source to target is through it."""
    dem = np.zeros((N, N), dtype=np.float32)
    dem[:, 6:9] = np.nan
    return scene_finder("cython", dem, **kwargs)


def test_dem_voids_are_hard_barriers_with_ignore_max_cost_true():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(Exception, match="No path found"):
            _void_wall_finder(ignore_max_cost=True).find_route()


def test_dem_voids_are_traversable_at_the_maximum_cost_with_ignore_max_cost_false():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        path = _void_wall_finder(ignore_max_cost=False).find_route()
    columns = np.asarray(path.path_indices) % N
    assert ((columns >= 6) & (columns < 9)).any(), "the route does not cross the void wall"


def test_the_texts_state_both_sides_of_the_void_rule():
    docs = {"PathFinder.__init__": _unwrapped_doc(PathFinder.__init__)}
    if DOCS_PAGE.exists():
        docs["metric_stack_objective.md"] = " ".join(DOCS_PAGE.read_text(encoding="utf-8").split())
    for name, text in docs.items():
        assert "ignore_max_cost=True" in text and "ignore_max_cost=False" in text, name
        assert "hard barriers" in text and "traversable at the maximum cost" in text, name
        assert "float32" in text, name


def test_the_weight_precision_text_says_the_eikonal_refusal_is_only_for_float32():
    text = _unwrapped_doc(PathFinder.__init__)
    assert "raster_fim with a DEM is refused" not in text
    assert "with a DEM is refused only for float32" in text
    assert "falls back to the cython Dijkstra with an AlgorithmFallbackWarning" in text


# ====================================================================================================== 8. sessions

SESSION_N = 28
SESSION_TRANSFORM = from_origin(0.0, float(SESSION_N), 1.0, 1.0)
SESSION_SOURCE = (1.5, SESSION_N - 0.5 - 14)
SESSION_TARGET = (SESSION_N - 1.5, SESSION_N - 0.5 - 9)
_BARE_MEANING = Objective({"cost": 1.0}, GradientOptions(multiplier="exponential"))


def _varied_raster() -> np.ndarray:
    return np.random.default_rng(5).integers(50, 120, size=(SESSION_N, SESSION_N)).astype(np.uint16)


def _bumps() -> np.ndarray:
    rows, cols = np.mgrid[0:SESSION_N, 0:SESSION_N]
    return (6.0 * np.exp(-((cols - 14) / 3.0) ** 2) * (np.abs(rows - 12) < 8)).astype(np.float32)


def _edge_cost(cells, raster: np.ndarray, dem) -> float:
    steps = get_neighborhood_steps("r2", directed=True)
    luts = None if dem is None else _BARE_MEANING.build_gradient_luts(steps, cell_size=1.0, quant_scale=1.0)
    model = EdgeModel(steps, dem=dem, gradient_luts=luts)
    priced = reprice_path(cells, raster, model)
    assert priced.ok, priced.violations
    return priced.cost


@pytest.mark.parametrize("with_dem", [False, True], ids=["no-dem", "dem"])
@pytest.mark.parametrize("graph_api", LIBRARIES)
def test_a_library_session_route_costs_what_find_route_costs(graph_api, with_dem):
    """The session searched half the neighbourhood steps without a DEM: 63 cells against 48 on a flat raster."""
    raster = _varied_raster()
    dem = _bumps() if with_dem else None
    finder = PathFinder(
        dataset_source=raster.copy(), crs=CRS, transform=SESSION_TRANSFORM, source_coords=SESSION_SOURCE,
        target_coords=SESSION_TARGET, search_space_buffer_m=100, graph_api=graph_api, dem=dem)
    session = finder.search_session()
    try:
        direct = finder.find_route(SESSION_SOURCE, SESSION_TARGET)
        routed = session.route([SESSION_SOURCE, SESSION_TARGET])
        cost_direct = _edge_cost(direct.path_indices, raster, dem)
        cost_session = _edge_cost(routed.path_indices, raster, dem)
        cells = (len(routed.path_indices), len(direct.path_indices))
        assert cost_session == pytest.approx(cost_direct, rel=1e-7), cells
    finally:
        session.close()


def test_a_cython_session_still_costs_what_find_route_costs():
    raster = _varied_raster()
    finder = PathFinder(
        dataset_source=raster.copy(), crs=CRS, transform=SESSION_TRANSFORM, source_coords=SESSION_SOURCE,
        target_coords=SESSION_TARGET, search_space_buffer_m=100)
    session = finder.search_session()
    try:
        direct = finder.find_route(SESSION_SOURCE, SESSION_TARGET)
        routed = session.route([SESSION_SOURCE, SESSION_TARGET])
        assert _edge_cost(routed.path_indices, raster, None) == pytest.approx(
            _edge_cost(direct.path_indices, raster, None), rel=1e-12)
    finally:
        session.close()
