"""Review fixes of the line-end feature, Python side (round 17, finder fix package).

Each block is the test of one finding of the independent review (and of the three kernel-review points the finder has
to follow):

1. With several tower heights a run that has a line-end setting in the search ("enforce", a connection height the
   search uses) runs the FULL variable-height search, not the lowest-height shortcut; checked against an independent
   exact programme over tower positions and height classes. A default run keeps the shortcut and says so.
2. An accepted exact-bin run could allocate tens of GB: the finder estimates the memory of the kernel it will call
   (``ConstrainedPathFinder.exact_bins_memory``) and refuses above ``EXACT_BINS_MAX_BYTES`` (stub kernels: nothing is
   allocated).
3. The report checks EVERY span of a returned route strictly (the search does not test the first cell after a tower,
   nor a span of one step) and says so in one ``SpanRuleWarning`` and ``span_notes``; a connection height below
   ``min_clearance_m`` is refused.
4. The height the source is reported with is the height class the search charged; the warning words what the search did.
5. Exact bins stay exact: more steps per longest span than ``EXACT_BINS_MAX_STEPS`` is refused, and a kernel refusal of
   the bins becomes an ``InvalidInputError``.
6. Small ones: ``span_bin_size_m`` is None without span limits, the line-end ``NoPathFoundError`` has its own message.
7. The documentation says that "enforce" can be cheaper than "keep" or "remove".
8. The report works on positions along the route, not on cells (a route that crosses itself).
"""

import math
import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import from_origin

import pyorps.core.infrastructure_profile as profile_module
import pyorps.utils.constrained_path_algorithms as algorithms
from pyorps.core.exceptions import InvalidInputError, NoPathFoundError
from pyorps.core.warning_types import PyorpsWarning, SpanRuleWarning
from pyorps.graph.constrained_path_finder import ConstrainedPathFinder, ExactBinsMemory
from pyorps.graph.span_clearance import path_span_clearance, span_clearance
from pyorps.utils.neighborhood import get_neighborhood_steps
from test_graph.test_terminal_enforce_round17 import l_shaped_finder

CRS = "EPSG:25832"
FORBIDDEN = 65535


@pytest.fixture(autouse=True)
def _one_thread(monkeypatch):
    # the delta-stepping kernels read OMP_NUM_THREADS per call: one thread keeps ties deterministic
    monkeypatch.setenv("OMP_NUM_THREADS", "1")


def caught_warnings(callable_, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = callable_(*args, **kwargs)
    return result, caught


def span_warnings(caught):
    return [w for w in caught if issubclass(w.category, SpanRuleWarning)]


# ============================================================================================== 1. the shortcut
W, T, MC = 20.0, 200.0, 5.0          # conductor weight (N/m), tension (N), minimum clearance (m)
MIN, MAX = 3.0, 8.0
SUSP = 50.0


def corridor_profile(heights, per_increment, rule):
    profile = dict(
        name="corridor", description="one open row", soft_angle_limit_deg=5.0, hard_angle_limit_deg=90.0,
        angle_cost_function="linear", angle_cost_params={"scale": 10.0},
        min_span_m=MIN, max_span_m=MAX, span_bin_size_m=1.0, terminal_span_rule=rule,
        tower_cost_params={"terrain_cost_map": {0: 40, 200: 300},
                           "angle_types": {"suspension": {"max_angle_deg": 5.0, "base_cost": SUSP},
                                           "angle": {"max_angle_deg": 90.0, "base_cost": 200.0}}},
        conductor_weight_per_m=W, conductor_tension_n=T, min_clearance_m=MC,
        max_gradient_percent=1.0e6, gradient_cost_function="linear", gradient_cost_params={"scale": 0.0})
    if len(heights) == 1:
        profile["tower_height_m"] = float(heights[0])
    else:
        profile.update(tower_heights_m=[float(h) for h in heights], tower_height_cost_per_increment=per_increment)
    return profile


class Corridor:
    """One open row of ``n`` cells (everything else forbidden) with random tower costs and obstacles near both ends and
    in the middle. The route is fixed, so only the towers (positions and height classes) differ between candidate
    lines, and ``optimum`` is an exact dynamic programme over them: it prices a line tower as foundation (the profile's
    table at the cell's raster value) + suspension cost + height premium, charges the source its class premium unless
    a connection height is given, and checks every span with its own parabola formula in float64 and the SEARCH's
    sampling. Between two line towers (and from a source without a connection height) a span of one step is not
    checked, the first cell after tower A is not sampled and tower B is. A span at a connection point (the first
    span from a source connection height, the last span to a target connection height) is sampled in full, the
    first cell after the tower and the end cell included, a span of one step too, and for the source the start cell
    itself: the kernels do so since the kernel review."""

    def __init__(self, seed, n=26, heights=(10.0, 14.0, 18.0), per_increment=25.0):
        rng = np.random.default_rng(seed)
        self.n, self.heights, self.per_increment = n, list(heights), per_increment
        self.rows, self.row = 5, 2
        self.raster = np.full((self.rows, n), FORBIDDEN, dtype=np.uint16)
        self.raster[self.row] = rng.integers(20, 400, n)
        self.dem = np.zeros((self.rows, n), np.float32)
        self.obs = np.zeros((self.rows, n), np.float32)
        near_source = int(rng.integers(2, 6))
        near_target = n - 1 - int(rng.integers(2, 6))
        self.obs[self.row, near_source] = float(rng.uniform(5.5, 13.0))
        self.obs[self.row, near_target] = float(rng.uniform(5.5, 13.0))
        self.obs[self.row, n // 2] = float(rng.uniform(5.5, 13.0))
        self.lut = None

    def finder(self, rule="keep"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            finder = ConstrainedPathFinder(
                dataset_source=self.raster.copy(), crs=CRS, transform=from_origin(0.0, float(self.rows), 1.0, 1.0),
                source_coords=(0.5, self.rows - self.row - 0.5),
                target_coords=(self.n - 0.5, self.rows - self.row - 0.5),
                search_space_buffer_m=100, profile=corridor_profile(self.heights, self.per_increment, rule),
                graph_api="cython", neighborhood_str="r1", dem=self.dem.copy())
        finder._obstacle_data = self.obs.copy()               # DSM minus DEM, as the finder would have loaded it
        self.lut = finder._profile_tower_terrain_costs
        return finder

    def clear(self, a, b, height_a, height_b, full=False, start_cell=False):
        """Does the span a -> b clear? ``full``: sample every cell after a, one-step spans included (a span at a
        connection point); ``start_cell``: also the conductor's own cell at a (a source connection structure)."""
        steps = b - a
        if steps <= 1 and not full:
            return True                                         # the search does not check a span of one step
        length = steps * 1.0
        z_a, z_b = float(self.dem[self.row, a]) + height_a, float(self.dem[self.row, b]) + height_b
        if start_cell and height_a - float(self.obs[self.row, a]) < MC - 1e-9:
            return False
        for j in range(1 if full else 2, steps + 1):            # the first cell after tower A (j = 1): full only
            x = float(j)
            z = z_a + (z_b - z_a) * x / length - W * x * (length - x) / (2 * T)
            cell = a + j
            if z - float(self.dem[self.row, cell]) - float(self.obs[self.row, cell]) < MC - 1e-9:
                return False
        return True

    def premium(self, height):
        base = min(self.heights)
        return (height - base) / 3.0 * self.per_increment if self.per_increment > 0 else 0.0

    def optimum(self, source_height=None, target_height=None, enforce=False, only_base=False):
        """Cost of the cheapest valid line (line towers + the source's class premium), or None if there is none."""
        classes = [min(self.heights)] if only_base else sorted(set(self.heights))
        source = [(source_height, 0.0)] if source_height is not None else [(h, self.premium(h)) for h in classes]
        best = {}
        for height, premium in source:
            best[(0, height, "source")] = min(best.get((0, height, "source"), math.inf), premium)
        finished = []
        for pos in range(self.n):
            for (p, height, kind), cost in list(best.items()):
                if p != pos:
                    continue
                span = (self.n - 1 - p) * 1.0                    # finish at the target
                if 0 < span < MAX - 1e-9:
                    ok = not (enforce and kind != "source" and span < MIN - 1e-9)
                    from_connection = kind == "source" and source_height is not None
                    if ok and target_height is not None and not self.clear(
                            p, self.n - 1, height, target_height, full=True, start_cell=from_connection):
                        ok = False
                    if ok:
                        finished.append(cost)
                for q in range(p + 1, self.n - 1):               # place the next tower
                    span = (q - p) * 1.0
                    if span >= MAX - 1e-9:
                        break
                    if span < MIN - 1e-9:
                        continue
                    from_connection = kind == "source" and source_height is not None
                    for height_q in classes:
                        if not self.clear(p, q, height, height_q, full=from_connection,
                                          start_cell=from_connection):
                            continue
                        cost_q = cost + float(self.lut[self.raster[self.row, q]]) + SUSP + self.premium(height_q)
                        if cost_q < best.get((q, height_q, "line"), math.inf):
                            best[(q, height_q, "line")] = cost_q
        return min(finished) if finished else None


def line_cost(path):
    """What the exact programme prices: the line towers' costs and the source's premium (the terminals' foundation and
    type costs, the terrain and the turn penalties are the same for every line on this fixed route)."""
    terminals = sum(t.terrain_cost + t.angle_cost for t in path.towers if t.tower_type == "terminal")
    return (path.total_cost - terminals - path.total_terrain_cost - path.total_angle_penalty_cost
            + path.cost_breakdown.get("removed_towers", 0.0))


SEEDS = range(14)
END_CASES = [("keep", 25.0, 25.0), ("keep", None, 25.0), ("keep", 25.0, None), ("enforce", None, None),
             ("enforce", 25.0, 25.0)]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("rule, source_height, target_height", END_CASES)
def test_a_line_end_setting_in_the_search_gets_the_full_variable_height_search(seed, rule, source_height,
                                                                              target_height):
    case = Corridor(seed)
    finder = case.finder(rule)
    expected = case.optimum(source_height, target_height, enforce=rule == "enforce")
    try:
        path, _ = caught_warnings(finder.find_route, source_connection_height_m=source_height,
                                  target_connection_height_m=target_height)
    except NoPathFoundError:
        assert expected is None, "the finder found no route although the exact programme has one"
        return
    assert expected is not None, "the finder returned a line the exact programme says is infeasible"
    assert line_cost(path) == pytest.approx(expected, abs=1e-3)
    assert not any("lowest height" in note for note in path.search_notes)          # no shortcut was taken


def test_the_lowest_height_shortcut_would_have_been_dearer_on_these_corridors():
    """The test above is meaningful: on several of its corridors the optimum with all heights is below the best
    line that uses only the lowest height, which is what the shortcut would have returned."""
    dearer = 0
    for seed in SEEDS:
        case = Corridor(seed)
        case.finder()
        full, base = case.optimum(25.0, 25.0), case.optimum(25.0, 25.0, only_base=True)
        if full is not None and base is not None and base > full + 1e-6:
            dearer += 1
    assert dearer >= 3


def test_the_reviewers_case_taller_towers_are_cheaper_and_the_finder_finds_them():
    case = Corridor(10)
    finder = case.finder("keep")
    full, base = case.optimum(25.0, 25.0), case.optimum(25.0, 25.0, only_base=True)
    assert base is None or base > full + 1.0                    # the lowest-height-only line is dearer
    path, _ = caught_warnings(finder.find_route, source_connection_height_m=25.0, target_connection_height_m=25.0)
    assert line_cost(path) == pytest.approx(full, abs=1e-3)
    assert any(t.height_m > 10.0 for t in path.towers[1:-1])    # taller towers are what makes it cheaper


def test_a_default_run_keeps_the_shortcut_and_the_result_says_so():
    """A default run (no line-end setting) still tries the lowest height alone first. When that finds a route it is
    returned with a search note, uses no taller tower, and is never cheaper than the exact optimum over all heights
    (it can be dearer: the known limitation). With the default (coarse) bins the lowest-height search can find no
    route although one exists; the full search then runs and there is no note."""
    shortcut = 0
    for seed in SEEDS:
        case = Corridor(seed)
        finder = case.finder("keep")
        path, _ = caught_warnings(finder.find_route)
        if len(path.path_indices) == 0:
            continue
        notes = [note for note in path.search_notes if "lowest height" in note]
        assert path.span_bin_size_m == 3.0, seed                 # default bins: a default run is unchanged
        if not notes:
            continue
        shortcut += 1
        assert len(notes) == 1 and "10 m" in notes[0] and "taller towers" in notes[0], seed
        assert "known limitation" in notes[0]
        assert all(t.height_m == 10.0 and t.height_premium == 0.0 for t in path.towers), seed
        assert line_cost(path) >= case.optimum() - 1e-6, seed
    assert shortcut >= 3


def test_a_default_run_with_one_tower_height_has_no_shortcut_note():
    case = Corridor(10, heights=(10.0,), per_increment=0.0)
    path, _ = caught_warnings(case.finder("keep").find_route)
    assert path.search_notes == []


# ====================================================================================== 2. the memory of the kernel
def stub_kernel(monkeypatch, name, height=False):
    """Replace a search kernel by a stub that records its arguments and returns a two-cell route."""
    seen = {}

    def stub(**kwargs):
        seen.update(kwargs)
        route = (np.array([0, 1], np.int64), np.array([], np.int64))
        return route + (np.array([], np.float32),) if height else route

    monkeypatch.setattr(algorithms, name, stub)
    return seen


def big_finder(n_cells, backend, rule="enforce", dem=False, **profile_extra):
    """An ``n_cells`` x ``n_cells`` window at 1 m cells with spans of 150 to 400 m (401 exact bins)."""
    profile = {
        "name": "long", "description": "long spans", "soft_angle_limit_deg": 5.0, "hard_angle_limit_deg": 90.0,
        "angle_cost_function": "linear", "angle_cost_params": {"scale": 10.0},
        "min_span_m": 150.0, "max_span_m": 400.0, "span_bin_size_m": 1.0, "terminal_span_rule": rule,
        "tower_cost_params": {"terrain_cost_map": {0: 100, 500: 100},
                              "angle_types": {"suspension": {"max_angle_deg": 5.0, "base_cost": 50.0},
                                              "angle": {"max_angle_deg": 90.0, "base_cost": 200.0}}},
        **profile_extra}
    extra = {"dem": np.zeros((n_cells, n_cells), np.float32)} if dem else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ConstrainedPathFinder(
            dataset_source=np.full((n_cells, n_cells), 100, np.uint16), crs=CRS,
            transform=from_origin(0.0, float(n_cells), 1.0, 1.0), source_coords=(5.5, n_cells / 2),
            target_coords=(n_cells - 5.5, n_cells / 2), search_space_buffer_m=1000, profile=profile,
            graph_api=backend, neighborhood_str="r2", **extra)


def test_cython_parallel_is_always_dense_and_a_550_cell_window_is_refused(monkeypatch):
    """The reviewer's case: 970 million states of 24 bytes (23.3 GB, 21.7 GiB) through the always-dense delta-stepping
    kernel, where default bins would need 0.17 GB."""
    seen = stub_kernel(monkeypatch, "constrained_delta_stepping_2d")
    finder = big_finder(550, "cython_parallel")
    with pytest.raises(InvalidInputError) as caught:
        finder.find_route()
    text = str(caught.value)
    assert not seen, "the kernel must not be reached"
    assert "constrained_delta_stepping_2d" in text and "970,420,000" in text and "21.7 GiB" in text
    assert "EXACT_BINS_MAX_BYTES" in text and "4.0 GiB" in text
    assert "coarser raster" in text and "search_space_buffer_m" in text and "max_span_m" in text
    assert "'keep' or 'remove_if_valid'" in text
    assert finder.EXACT_BINS_MAX_BYTES == 4 * 1024 ** 3


def test_the_dense_dijkstra_window_of_the_reviewer_is_refused_by_its_bytes_not_its_states(monkeypatch):
    seen = stub_kernel(monkeypatch, "constrained_dijkstra_2d")
    finder = big_finder(270, "cython")
    with pytest.raises(InvalidInputError) as caught:
        finder.find_route()
    text = str(caught.value)
    assert not seen and "467,726,400" in text                    # under 5e8 states: dense, 24 bytes each
    assert "constrained_dijkstra_2d" in text and "10.5 GiB" in text and "dense state array" in text
    assert "above ConstrainedPathFinder.EXACT_BINS_MAX_STATES" not in text      # the state bound did not trigger


def test_a_sparse_kernel_is_estimated_at_its_worst_case_and_refused_above_the_limit(monkeypatch):
    seen = stub_kernel(monkeypatch, "constrained_dijkstra_2d")
    finder = big_finder(550, "cython")                           # 1.94e9 states: above 5e8 the kernel is sparse
    with pytest.raises(InvalidInputError) as caught:
        finder.find_route()
    text = str(caught.value)
    assert not seen and "1,940,840,000" in text and "hash maps" in text
    assert "every state as reached" in text and "worst case" in text and "about 200 bytes" in text


def test_the_clearance_kernels_have_their_own_estimate(monkeypatch):
    seen = stub_kernel(monkeypatch, "constrained_delta_stepping_height_2d", height=True)
    clearance = dict(tower_heights_m=[10.0, 14.0, 18.0], tower_height_cost_per_increment=10.0,
                     conductor_weight_per_m=5.0, conductor_tension_n=2000.0, min_clearance_m=5.0)
    finder = big_finder(100, "cython", dem=True, **clearance)    # 100 x 100 x 16 x 401 x 3 classes = 192 million states
    with pytest.raises(InvalidInputError) as caught:
        finder.find_route()
    text = str(caught.value)
    assert not seen and "constrained_delta_stepping_height_2d" in text and "192,480,000" in text
    assert "3 height classes" in text and "4.3 GiB" in text


def test_a_small_exact_bin_run_passes_and_reaches_the_kernel(monkeypatch):
    seen = stub_kernel(monkeypatch, "constrained_dijkstra_2d")
    finder = big_finder(60, "cython")                            # 60 x 60 x 16 x 401 = 23 million states, 0.55 GB
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            finder.find_route()
        except Exception:                                        # noqa: BLE001  (the two-cell stub route is not a line)
            pass
    assert seen["n_span_bins"] == 401 and seen["target_min_span"] == 150.0


def test_raising_the_byte_limit_lets_the_run_through(monkeypatch):
    seen = stub_kernel(monkeypatch, "constrained_delta_stepping_2d")
    monkeypatch.setattr(ConstrainedPathFinder, "EXACT_BINS_MAX_BYTES", 40 * 1024 ** 3)
    finder = big_finder(550, "cython_parallel")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            finder.find_route()
        except Exception:                                        # noqa: BLE001
            pass
    assert seen["n_span_bins"] == 401


def test_the_state_bound_stays_and_is_checked_before_the_bytes(monkeypatch):
    stub_kernel(monkeypatch, "constrained_dijkstra_2d")
    monkeypatch.setattr(ConstrainedPathFinder, "EXACT_BINS_MAX_STATES", 1000)
    with pytest.raises(InvalidInputError) as caught:
        big_finder(60, "cython").find_route()
    text = str(caught.value)
    assert "EXACT_BINS_MAX_STATES" in text and "coarser raster" in text and "GiB" not in text


def test_exact_bins_memory_follows_the_kernel_the_finder_calls():
    dense = ConstrainedPathFinder.exact_bins_memory(1000, 8, 10, backend="cython")
    assert isinstance(dense, ExactBinsMemory) and dense.kernel == "constrained_dijkstra_2d"
    assert dense.form == "dense" and dense.states == 80_000
    assert dense.nbytes == 80_000 * 24 + 1000 * 8 * 5            # state array + intermediate-cell caches
    with_dem = ConstrainedPathFinder.exact_bins_memory(1000, 8, 10, backend="cython", has_dem=True)
    assert with_dem.nbytes == dense.nbytes + 1000 * 8 * 4        # + the gradient cache
    parallel = ConstrainedPathFinder.exact_bins_memory(10 ** 6, 8, 100, backend="cython_parallel")
    assert parallel.kernel == "constrained_delta_stepping_2d" and parallel.form == "dense"     # always dense
    assert parallel.nbytes == 8 * 10 ** 8 * 24 + 10 ** 6 * 8 * 5                               # although 8e8 > 5e8
    sparse = ConstrainedPathFinder.exact_bins_memory(10 ** 6, 8, 100, backend="cython")
    assert sparse.kernel == "constrained_dijkstra_2d" and sparse.form == "sparse"
    assert sparse.nbytes == 8 * 10 ** 8 * 200 + 10 ** 6 * 8 * 5
    height = ConstrainedPathFinder.exact_bins_memory(1000, 8, 10, 3, backend="cython_parallel", clearance=True)
    assert height.kernel == "constrained_delta_stepping_height_2d" and height.states == 240_000
    assert height.nbytes == 240_000 * 24 + 1000 * 8 * 9          # caches: 5 + 4 bytes with the DEM
    lazy = ConstrainedPathFinder.exact_bins_memory(10 ** 6, 8, 100, 1, backend="cython", clearance=True)
    assert lazy.kernel == "constrained_delta_stepping_lazy" and lazy.form == "sparse"
    assert lazy.nbytes == 8 * 10 ** 8 * 64 + 10 ** 6 * 8 * 9
    assert "worst case" in lazy.detail and "dense state array" in dense.detail


# ================================================================================= 3. the strict clearance report
COMMON = dict(conductor_weight_per_m=1.0, conductor_tension_n=1000.0, min_clearance_m=5.0)


def flat_check(obstacle_at=None, obstacle=8.0, *, a=2, steps=6, height_a=10.0, height_b=10.0, strict=False, dem=None):
    ground = np.zeros((3, 24), np.float32) if dem is None else dem
    obstacles = np.zeros_like(ground)
    for col in ([] if obstacle_at is None else np.atleast_1d(obstacle_at)):
        obstacles[1, col] = obstacle
    return span_clearance(ground, obstacles, (1, a), (0, 1), steps, 1.0, height_a, height_b, strict=strict, **COMMON)


def test_the_strict_check_samples_the_first_cell_after_the_tower_and_the_default_does_not():
    blind = flat_check(obstacle_at=3)                            # cell a + 1: the one the search never samples
    assert blind.ok and blind.checked                            # the kernel mirror passes
    strict = flat_check(obstacle_at=3, strict=True)
    assert not strict.ok and strict.worst_cell == (1, 3) and strict.worst_clearance_m < 5.0
    assert flat_check(obstacle_at=4).ok is False                 # one cell further: both see it
    assert flat_check(obstacle_at=4, strict=True).ok is False


def test_the_strict_check_checks_a_span_of_one_step_and_the_default_does_not():
    mirror = flat_check(obstacle_at=3, steps=1)
    assert mirror.ok and not mirror.checked                      # one step: clear without a look
    strict = flat_check(obstacle_at=3, steps=1, strict=True)
    assert not strict.ok and strict.checked and strict.worst_cell == (1, 3)
    assert flat_check(steps=1, strict=True).ok                   # a clear one-step span stays clear
    assert flat_check(steps=0, strict=True).checked is False     # no span at all


def test_the_strict_check_does_not_take_the_tall_tower_shortcut():
    hill = np.zeros((3, 24), np.float32)
    hill[1, 5] = 90.0                                            # ground rising far above the 80 m towers
    default = flat_check(height_a=80.0, height_b=80.0, dem=hill)
    assert default.ok and not default.checked                    # min height - sag - clearance > 50: not looked at
    strict = flat_check(height_a=80.0, height_b=80.0, dem=hill, strict=True)
    assert not strict.ok and strict.checked and strict.worst_cell == (1, 5)


def test_a_span_that_passes_the_strict_check_passes_the_default_check():
    rng = np.random.default_rng(5)
    disagreements = 0
    for _ in range(200):
        ground = rng.uniform(0.0, 3.0, (3, 24)).astype(np.float32)
        obstacles = np.zeros_like(ground)
        obstacles[1, rng.integers(2, 12, 3)] = rng.uniform(0.0, 9.0, 3).astype(np.float32)
        steps = int(rng.integers(1, 10))
        height_a, height_b = (float(h) for h in rng.uniform(6.0, 14.0, 2))
        args = (ground, obstacles, (1, 2), (0, 1), steps, 1.0, height_a, height_b)
        default, strict = span_clearance(*args, **COMMON), span_clearance(*args, strict=True, **COMMON)
        assert default.ok or not strict.ok                       # strict ok  =>  default ok
        disagreements += default.ok and not strict.ok
    assert disagreements > 0                                     # and the blind spots do show up


def test_path_span_clearance_passes_strict_on():
    cells = np.arange(2, 8)
    rows, cols = np.ones(6, dtype=np.int64), cells
    obstacles = np.zeros((3, 24), np.float32)
    obstacles[1, 3] = 8.0
    args = (np.zeros((3, 24), np.float32), obstacles, rows, cols, 0, 5, 1.0, 10.0, 10.0)
    assert path_span_clearance(*args, **COMMON).ok
    assert not path_span_clearance(*args, strict=True, **COMMON).ok


def test_start_cell_samples_the_cell_the_conductor_leaves_the_structure_on():
    """The first span from a source connection height is also sampled on the start cell (the kernels do so): the
    conductor is at the structure's height there, so an obstacle under the structure counts."""
    ground = np.zeros((3, 24), np.float32)
    obstacles = np.zeros_like(ground)
    obstacles[1, 2] = 8.0                                        # on the start cell (tower A at column 2)
    args = (ground, obstacles, (1, 2), (0, 1), 6, 1.0, 10.0, 10.0)
    assert span_clearance(*args, **COMMON).ok and span_clearance(*args, strict=True, **COMMON).ok
    start = span_clearance(*args, start_cell=True, **COMMON)
    assert not start.ok and start.worst_cell == (1, 2) and start.worst_offset_m == 0.0
    assert start.worst_clearance_m == pytest.approx(2.0)
    assert span_clearance(ground, obstacles, (1, 2), (0, 1), 6, 1.0, 14.0, 10.0, start_cell=True, **COMMON).ok
    one_step = span_clearance(ground, obstacles, (1, 2), (0, 1), 1, 1.0, 10.0, 10.0, start_cell=True, **COMMON)
    assert not one_step.ok and one_step.checked                  # start_cell implies the full sampling


def test_the_strict_check_agrees_with_the_search_on_the_two_end_spans_of_a_run_with_connection_heights():
    """With connection heights the search samples the first span from the source (start cell included) and the last
    span to the target in full, as ``strict`` does: a line the search returns passes the strict check on both end
    spans, whatever the heights (low ones make the clearance bind)."""
    checked = 0
    for seed in SEEDS:
        for source_height, target_height in ((7.0, 9.0), (9.0, 7.0), (12.0, 12.0), (6.0, 25.0)):
            case = Corridor(seed)
            finder = case.finder("keep")
            try:
                path, _ = caught_warnings(finder.find_route, source_connection_height_m=source_height,
                                          target_connection_height_m=target_height)
            except NoPathFoundError:
                continue
            cols = [t.cell_index % case.n for t in path.towers]
            heights = [t.height_m for t in path.towers]
            assert heights[0] == source_height and heights[-1] == target_height
            for index, start_cell in ((0, True), (len(cols) - 2, False)):
                result = span_clearance(
                    case.dem, case.obs, (case.row, cols[index]), (0, 1), cols[index + 1] - cols[index], 1.0,
                    heights[index], heights[index + 1], conductor_weight_per_m=W, conductor_tension_n=T,
                    min_clearance_m=MC, strict=True, start_cell=start_cell)
                assert result.ok, (seed, source_height, target_height, index, result)
            end_notes = [n for n in path.span_notes if n.startswith(("the first span", "the last span", "the only"))]
            assert end_notes == [], (seed, end_notes)           # the report agrees: no end span is flagged
            checked += 1
    assert checked >= 20


ROWS, COLS, ROW = 5, 64, 2
SYN_CLEARANCE = dict(tower_height_m=10.0, conductor_weight_per_m=1.0, conductor_tension_n=1000.0, min_clearance_m=5.0)
SYN_PROFILE = dict(
    name="synthetic", description="hand-made kernel output", soft_angle_limit_deg=5.0, hard_angle_limit_deg=90.0,
    angle_cost_function="linear", angle_cost_params={"scale": 10.0}, min_span_m=3.0, max_span_m=8.0,
    span_bin_size_m=1.0,
    tower_cost_params={"terrain_cost_map": {0: 100, 500: 100},
                       "angle_types": {"suspension": {"max_angle_deg": 5.0, "base_cost": 50.0},
                                       "angle": {"max_angle_deg": 90.0, "base_cost": 200.0}}})


def synthetic(cells, towers, *, rows=ROWS, cols=COLS, rule="keep", heights=None, obstacle=None, dem=True, **extra):
    """Report hand-made kernel output. ``cells`` are the route's (row, col) cells, ``towers`` the kernel's tower
    cells in route order (a cell may appear twice), ``heights`` the kernel's height per tower entry. Returns the path
    and the ``SpanRuleWarning``s."""
    xy = lambda row, col: (col + 0.5, rows - row - 0.5)  # noqa: E731
    profile = {**SYN_PROFILE, **(SYN_CLEARANCE if dem else {}), "terminal_span_rule": rule, **extra}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        finder = ConstrainedPathFinder(
            dataset_source=np.full((rows, cols), 100, np.uint16), crs=CRS,
            transform=from_origin(0.0, float(rows), 1.0, 1.0),
            source_coords=xy(*cells[0]), target_coords=xy(*cells[-1]), search_space_buffer_m=100, profile=profile,
            graph_api="cython", neighborhood_str="r1", dem=np.zeros((rows, cols), np.float32) if dem else None)
    finder._obstacle_data = obstacle
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        path = finder._build_constrained_path(
            np.array([r * cols + c for r, c in cells]), np.array([r * cols + c for r, c in towers]),
            finder.raster_handler.data[0], xy(*cells[0]), xy(*cells[-1]), 0.0,
            tower_heights_arr=None if heights is None else np.array(heights, np.float32))
    return path, span_warnings(caught)


def east(first, last, row=ROW):
    return [(row, c) for c in range(first, last + 1)]


def cols_of(path):
    return [t.cell_index % COLS for t in path.towers]


def test_a_blind_first_cell_on_an_interior_span_is_reported_though_the_search_accepts_it():
    cells, towers = east(0, 21), [(ROW, 7), (ROW, 14)]
    obstacle = np.zeros((ROWS, COLS), np.float32)
    obstacle[ROW, 8] = 8.0                                       # the first cell after the tower at column 7
    path, warned = synthetic(cells, towers, obstacle=obstacle)
    assert cols_of(path) == [0, 7, 14, 21]                       # the route and its towers are unchanged
    (warning,) = warned
    text = str(warning.message)
    assert "span 2 of 3 from (7.5, 2.5) to (14.5, 2.5)" in text and "fails the clearance check" in text
    assert "the search accepted it because it does not test the first cell after a tower" in text
    assert "The search does not test the first cell after a tower, nor a span of one step" in text
    assert "1 of these spans" in text and "1 case)" in text
    assert [n for n in path.span_notes if "span 2 of 3" in n] and len(path.span_notes) == 1       # the same entry


def test_a_one_step_end_span_below_the_clearance_is_reported():
    cells, towers = east(0, 15), [(ROW, 7), (ROW, 14)]
    obstacle = np.zeros((ROWS, COLS), np.float32)
    obstacle[ROW, 15] = 8.0                                      # under the target cell, the last span is one step
    path, warned = synthetic(cells, towers, obstacle=obstacle)
    (warning,) = warned
    text = str(warning.message)
    assert "the last span from (14.5, 2.5) to (15.5, 2.5)" in text and "does not test a span of one step" in text


def test_a_clean_route_gives_no_warning_and_no_note():
    path, warned = synthetic(east(0, 21), [(ROW, 7), (ROW, 14)])
    assert warned == [] and path.span_notes == []


def test_the_warning_lists_a_few_spans_and_gives_the_total():
    cells, towers = east(0, 60), [(ROW, 6 * k) for k in range(1, 10)]
    obstacle = np.zeros((ROWS, COLS), np.float32)
    for k in range(10):
        obstacle[ROW, 6 * k + 1] = 8.0                           # the first cell after every tower and the source
    path, warned = synthetic(cells, towers, obstacle=obstacle)
    (warning,) = warned
    text = str(warning.message)
    assert "10 cases, the first 6 listed here, ConstrainedPath.span_notes has all of them" in text
    assert "(6) " in text and "(7) " not in text
    assert len(path.span_notes) == 10 and "10 of these spans" in text


def test_the_strict_check_runs_on_every_span_also_under_the_removal_rules():
    cells, towers = east(0, 21), [(ROW, 7), (ROW, 14)]
    obstacle = np.zeros((ROWS, COLS), np.float32)
    obstacle[ROW, 8] = 8.0
    for rule in ("remove", "remove_if_valid", "keep", "enforce"):
        path, warned = synthetic(cells, towers, rule=rule, obstacle=obstacle)
        assert len(warned) == 1 and "span 2 of 3" in str(warned[0].message), rule
        assert cols_of(path) == [0, 7, 14, 21], rule


def test_a_connection_height_below_the_minimum_clearance_is_refused():
    """A 4 m gantry under a 5 m minimum clearance can never clear at its own end cell."""
    def finder(**kw):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ConstrainedPathFinder(
                dataset_source=np.full((ROWS, 40), 100, np.uint16), crs=CRS,
                transform=from_origin(0.0, float(ROWS), 1.0, 1.0), source_coords=(0.5, 2.5), target_coords=(39.5, 2.5),
                search_space_buffer_m=100, profile={**SYN_PROFILE, **SYN_CLEARANCE}, graph_api="cython",
                neighborhood_str="r1", dem=np.zeros((ROWS, 40), np.float32), **kw)

    for name in ("source_connection_height_m", "target_connection_height_m"):
        with pytest.raises(InvalidInputError) as caught:
            finder().find_route(**{name: 4.0})
        text = str(caught.value)
        assert name in text and "4 m" in text and "min_clearance_m 5 m" in text
    path, _ = caught_warnings(finder().find_route, target_connection_height_m=5.0)       # equal: allowed
    assert path.target_connection_height_m == 5.0
    path, _ = caught_warnings(finder().find_route, source_connection_height_m=12.0, target_connection_height_m=9.0)
    assert (path.source_connection_height_m, path.target_connection_height_m) == (12.0, 9.0)


def test_a_connection_height_that_has_no_effect_is_only_reported_even_when_low():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        finder = ConstrainedPathFinder(
            dataset_source=np.full((ROWS, 40), 100, np.uint16), crs=CRS,
            transform=from_origin(0.0, float(ROWS), 1.0, 1.0),
            source_coords=(0.5, 2.5), target_coords=(39.5, 2.5), search_space_buffer_m=100,
            profile={**SYN_PROFILE, **SYN_CLEARANCE}, graph_api="cython", neighborhood_str="r1")     # no DEM
    path, caught = caught_warnings(finder.find_route, target_connection_height_m=2.0)
    assert path.target_connection_height_m == 2.0 and path.towers[-1].height_m == 2.0
    assert [w for w in caught if issubclass(w.category, PyorpsWarning) and "has no effect" in str(w.message)]


# ====================================================================================== 4. the source's height class
HEIGHT_PROFILE = dict(
    name="classes", description="three height classes", soft_angle_limit_deg=5.0, hard_angle_limit_deg=90.0,
    angle_cost_function="linear", angle_cost_params={"scale": 10.0}, min_span_m=3.0, max_span_m=8.0,
    span_bin_size_m=1.0,
    tower_cost_params={"terrain_cost_map": {0: 100, 100: 500, 500: 2000},
                       "angle_types": {"suspension": {"max_angle_deg": 5.0, "base_cost": 50.0},
                                       "angle": {"max_angle_deg": 90.0, "base_cost": 200.0}}},
    conductor_weight_per_m=20.0, conductor_tension_n=2000.0, min_clearance_m=5.0, max_gradient_percent=1.0e6,
    gradient_cost_function="linear", gradient_cost_params={"scale": 0.0},
    tower_heights_m=[10.0, 20.0, 30.0], tower_height_cost_per_increment=30.0)


def classes_run(terminal_height=None, obstacles=True, **kwargs):
    """The reviewer's case: uniform 30 x 30 raster, 14 m obstacles near both ends, three height classes."""
    n = 30
    profile = dict(HEIGHT_PROFILE)
    if terminal_height is not None:
        profile["terminal_tower_height_m"] = terminal_height
    obstacle = np.zeros((n, n), np.float32)
    if obstacles:
        obstacle[:, 5:7] = 14.0
        obstacle[:, 23:25] = 14.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        finder = ConstrainedPathFinder(
            dataset_source=np.full((n, n), 100, np.uint16), crs=CRS, transform=from_origin(0.0, float(n), 1.0, 1.0),
            source_coords=(3.5, 15.5), target_coords=(26.5, 15.5), search_space_buffer_m=1000, profile=profile,
            graph_api="cython", neighborhood_str="r2", dem=np.zeros((n, n), np.float32))
    finder._obstacle_data = obstacle
    return caught_warnings(finder.find_route, **kwargs)


def test_the_source_is_reported_with_the_height_class_the_search_charged_not_the_terminal_height():
    path, caught = classes_run(terminal_height=15.0)
    source = path.towers[0]
    assert source.height_premium > 0.0                            # the first span needs a taller class
    assert source.height_m in (20.0, 30.0) and source.height_m != 15.0
    notes = [note for note in path.search_notes if "height class" in note]
    assert len(notes) == 1 and f"the {source.height_m:g} m height class" in notes[0]
    for warning in span_warnings(caught):
        assert "heights 15 m" not in str(warning.message)         # the check used the class height


def test_without_a_charged_premium_the_terminal_height_stays_the_reported_height():
    path, _ = classes_run(terminal_height=15.0, obstacles=False)
    source = path.towers[0]
    assert source.height_premium == 0.0 and source.height_m == 15.0
    assert not [n for n in path.search_notes if "height class" in n]


def test_a_source_connection_height_always_wins():
    path, _ = classes_run(terminal_height=15.0, source_connection_height_m=25.0)
    source = path.towers[0]
    assert source.height_m == 25.0 and source.height_premium == 0.0


def test_the_warning_words_what_the_search_did_at_the_source():
    cells, towers = east(0, 14), [(ROW, 7)]
    obstacle = np.zeros((ROWS, COLS), np.float32)
    obstacle[ROW, 3] = 14.0                                       # the first span needs the 20 m class at the source
    obstacle[ROW, 12] = 50.0                                      # the last span fails; the search never checks it
    path, warned = synthetic(cells, towers, heights=[20.0], obstacle=obstacle, rule="keep",
                             tower_height_m=None, tower_heights_m=[10.0, 20.0], tower_height_cost_per_increment=30.0,
                             terminal_tower_height_m=15.0)
    source = path.towers[0]
    assert source.height_premium == pytest.approx(100.0) and source.height_m == 20.0
    (warning,) = warned
    text = str(warning.message)
    assert "the first span with the source at the 20 m height class it chose" in text
    assert "line height at the source" not in text and "heights 15 m" not in text
    assert "attachment heights 20 m and 15 m" in text             # the last span: class tower to the reported terminal


def test_a_fixed_height_names_the_line_height_in_the_warning():
    cells, towers = east(0, 14), [(ROW, 7)]
    obstacle = np.zeros((ROWS, COLS), np.float32)
    obstacle[ROW, 12] = 50.0
    _, warned = synthetic(cells, towers, obstacle=obstacle)
    assert "the first span with the line height 10 m at the source" in str(warned[0].message)


# ======================================================================================= 5. the bins stay exact
def float32_bins_are_distinct(cell, max_span, neighborhood):
    """The kernel's own arithmetic in float32: do the spans of 1 to K steps of every direction fall into different bins
    (and the shortest step's k-th span into bin k)? ``new = double(span32) + step32; bin = int(new / bin32);
    span32 = float32(new)``."""
    f32 = np.float32
    steps = get_neighborhood_steps(neighborhood)
    lengths = {float(f32(math.sqrt(dr * dr + dc * dc) * cell)) for dr, dc in steps}
    shortest = float(f32(cell))
    k_steps = max(1, math.ceil(max_span / shortest))
    bin32 = float(f32(shortest * k_steps / (k_steps + 0.25)))
    limit = float(f32(max_span))
    for step in sorted(lengths):
        span32, bins = f32(0.0), []
        while float(span32) + step < limit:
            new = float(span32) + step
            bins.append(int(new / bin32))
            span32 = f32(new)
        if len(set(bins)) != len(bins) or (bins and max(bins) >= k_steps + 1):
            return False
        if step == shortest and bins != list(range(1, len(bins) + 1)):
            return False
    return True


def bins_finder(cell, max_span, rule="enforce", neighborhood="r2"):
    profile = {**SYN_PROFILE, "min_span_m": 1.0, "max_span_m": float(max_span), "terminal_span_rule": rule}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ConstrainedPathFinder(
            dataset_source=np.full((20, 20), 100, np.uint16), crs=CRS,
            transform=from_origin(0.0, 20.0 * cell, cell, cell), source_coords=(1.5 * cell, 10.5 * cell),
            target_coords=(18.5 * cell, 10.5 * cell), search_space_buffer_m=1000, profile=profile, graph_api="cython",
            neighborhood_str=neighborhood)


@pytest.mark.parametrize("neighborhood", ["r1", "r2", "r3"])
@pytest.mark.parametrize("cell", [0.01, 0.1, 0.3, 0.5, 0.9, 1.0, 1.9542, 2.5])
def test_the_accepted_step_limit_keeps_every_bin_distinct_in_the_kernels_arithmetic(cell, neighborhood):
    limit = ConstrainedPathFinder.EXACT_BINS_MAX_STEPS
    finder = bins_finder(cell, (limit - 0.5) * cell, neighborhood=neighborhood)
    finder._ends.target_min_span = 1.0
    n_bins, bin_size = finder._exact_span_bins(np.zeros((20, 20)), None)                  # accepted
    assert n_bins == limit + 1
    assert float32_bins_are_distinct(cell, (limit - 0.5) * cell, neighborhood)


def test_the_step_limit_is_well_below_where_the_simulation_first_finds_merged_bins():
    """The limit (2048) is the worst-case bound (K * K * 2**-24 < 1/4); the simulation finds merged bins only far above
    (around 4500 steps for some cell sizes, none up to 4096 for the cell sizes tried here), so the problem is real
    and the limit keeps a margin."""
    assert ConstrainedPathFinder.EXACT_BINS_MAX_STEPS <= 2048
    for cell in (0.01, 0.1, 0.3, 0.9, 1.9542, 2.5):
        assert float32_bins_are_distinct(cell, (4096 - 0.5) * cell, "r2")
    assert any(not float32_bins_are_distinct(0.9, (k - 0.5) * 0.9, "r1") for k in range(4500, 6001, 250))


def test_too_many_steps_per_longest_span_is_refused_with_the_numbers():
    """The reviewer's case: 0.01 m cells with 400 m spans (40 000 steps)."""
    finder = bins_finder(0.01, 400.0)
    with pytest.raises(InvalidInputError) as caught:
        finder.find_route()
    text = str(caught.value)
    assert "terminal_span_rule='enforce'" in text and "40000 steps" in text and "0.01 m" in text
    assert "EXACT_BINS_MAX_STEPS = 2048" in text and "single precision" in text
    assert "coarser raster" in text and "max_span_m" in text and "'keep' or 'remove_if_valid'" in text


def test_one_step_over_the_limit_is_refused_and_a_default_run_is_not():
    limit = ConstrainedPathFinder.EXACT_BINS_MAX_STEPS
    finder = bins_finder(0.5, (limit + 0.5) * 0.5)
    finder._ends.target_min_span = 1.0
    with pytest.raises(InvalidInputError, match="EXACT_BINS_MAX_STEPS"):
        finder._exact_span_bins(np.zeros((20, 20)), None)
    plain = bins_finder(0.5, (limit + 0.5) * 0.5, rule="keep")                           # default bins: no such bound
    assert plain._effective_n_span_bins == math.ceil((limit + 0.5) * 0.5 / 1.0)


def test_a_kernel_refusal_of_the_bins_becomes_an_input_error_with_the_numbers(monkeypatch):
    def refusing(**kwargs):
        raise ValueError("span bins of direction 3 are not injective: spans of 2048 and 2049 steps share bin 2047")

    monkeypatch.setattr(algorithms, "constrained_dijkstra_2d", refusing)
    finder = bins_finder(1.0, 8.0)
    with pytest.raises(InvalidInputError) as caught:
        finder.find_route()
    text = str(caught.value)
    assert "refused the span bins" in text and "not injective" in text and "9 span bins" in text
    assert "max_span_m 8 m, is 8 steps of the shortest step 1 m" in text and "EXACT_BINS_MAX_STEPS = 2048" in text


def test_another_value_error_of_a_kernel_is_not_turned_into_a_bins_error(monkeypatch):
    def refusing(**kwargs):
        raise ValueError("target_min_span must be finite")

    monkeypatch.setattr(algorithms, "constrained_dijkstra_2d", refusing)
    with pytest.raises(ValueError, match="target_min_span") as caught:
        bins_finder(1.0, 8.0).find_route()
    assert not isinstance(caught.value, InvalidInputError)


def test_the_kernels_own_bin_check_runs_in_the_exact_bin_setup_before_any_kernel_call(monkeypatch):
    """The finder calls ``check_exact_span_bins`` (the check the kernels run at entry) itself, so a refusal reaches the
    caller as the finder's ``InvalidInputError`` with the numbers, and no kernel is called."""
    import pyorps.utils._constrained_context as context

    seen = stub_kernel(monkeypatch, "constrained_dijkstra_2d")
    calls = []

    def refusing(step_distances, span_bin_size, max_span, n_span_bins):
        calls.append((len(step_distances), span_bin_size, max_span, n_span_bins))
        raise ValueError("span_bin_size is too coarse: the spans of 1 and 2 steps share span bin 0")

    monkeypatch.setattr(context, "check_exact_span_bins", refusing)
    with pytest.raises(InvalidInputError) as caught:
        bins_finder(1.0, 8.0).find_route()
    text = str(caught.value)
    assert not seen, "the kernel must not be reached"
    assert calls and calls[0][2:] == (8.0, 9)
    assert "would refuse the 9 span bins of 0.9697 m for max_span_m 8 m (8 steps of the shortest step 1 m)" in text
    assert "too coarse" in text and "terminal_span_rule='enforce'" in text and "coarser raster" in text


def test_the_real_bin_check_accepts_the_finders_exact_bins_and_refuses_coarse_ones():
    from pyorps.utils._constrained_context import check_exact_span_bins

    finder = bins_finder(1.0, 8.0)
    finder._ends.target_min_span = 1.0
    n_bins, bin_size = finder._exact_span_bins(np.zeros((20, 20)), None)
    steps = finder._step_distances.astype(np.float32)
    check_exact_span_bins(steps, bin_size, 8.0, n_bins)                                   # does not raise
    with pytest.raises(ValueError, match="too coarse"):
        check_exact_span_bins(steps, 3.0, 8.0, 3)


def test_a_kernel_type_error_about_a_line_end_argument_becomes_an_invalid_type_error(monkeypatch):
    from pyorps.core.exceptions import InvalidTypeError

    def refusing(**kwargs):
        raise TypeError("source_attach_height must be a number, not bool")

    monkeypatch.setattr(algorithms, "constrained_dijkstra_2d", refusing)
    with pytest.raises(InvalidTypeError, match="source_attach_height") as caught:
        bins_finder(1.0, 8.0).find_route()
    assert isinstance(caught.value, TypeError) and "connection heights" in str(caught.value)

    def other(**kwargs):
        raise TypeError("an unrelated type error")

    monkeypatch.setattr(algorithms, "constrained_dijkstra_2d", other)
    with pytest.raises(TypeError, match="unrelated") as caught:
        bins_finder(1.0, 8.0).find_route()
    assert not isinstance(caught.value, InvalidTypeError)


def test_only_validated_numbers_reach_the_kernel(monkeypatch):
    seen = stub_kernel(monkeypatch, "constrained_dijkstra_2d")
    finder = bins_finder(1.0, 8.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            finder.find_route()
        except Exception:                                        # noqa: BLE001  (the stub route is not a line)
            pass
    assert type(seen["target_min_span"]) is float and seen["target_min_span"] == 1.0
    assert "source_attach_height" not in seen and "target_attach_height" not in seen    # no DEM: nothing to attach


# ============================================================================================== 6. small ones
def test_the_stale_comment_about_a_refused_enforce_is_gone():
    source = Path(profile_module.__file__).read_text(encoding="utf-8")
    assert "refuses it until then" not in source and "needs the search update" not in source
    assert "CHEAPER" in source


def test_span_bin_size_is_none_without_span_limits_and_a_number_with_them():
    profile = dict(name="road", description="no towers", soft_angle_limit_deg=15.0, hard_angle_limit_deg=90.0,
                   angle_cost_function="linear", angle_cost_params={"scale": 50})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        finder = ConstrainedPathFinder(
            dataset_source=np.full((20, 40), 100, np.uint16), crs=CRS, transform=from_origin(0.0, 20.0, 1.0, 1.0),
            source_coords=(1.5, 10.5), target_coords=(38.5, 10.5), search_space_buffer_m=100, profile=profile,
            graph_api="cython", neighborhood_str="r1")
    path, _ = caught_warnings(finder.find_route)
    assert path.span_bin_size_m is None and path.search_notes == []
    spans, _ = caught_warnings(bins_finder(1.0, 8.0, rule="keep").find_route)
    assert spans.span_bin_size_m == 1.0                                                   # min_span_m: the default bins


def test_the_line_end_no_path_error_has_its_own_complete_message():
    with pytest.raises(NoPathFoundError) as caught:
        l_shaped_finder("enforce").find_route()
    error = caught.value
    text = str(error)
    assert "increase buffer" not in text and "Choose different" not in text
    assert text.startswith("No route from ") and "terminal_span_rule='enforce'" in text and "min_span_m 3" in text
    assert "'keep'" in text and "'remove_if_valid'" in text and "no route exists" in text
    assert type(error) is NoPathFoundError and error.complete_message == text            # the existing class
    assert "satisfies the line-end settings" in text                                      # the words the GUI matches on
    again = pickle.loads(pickle.dumps(error))                                             # survives a round trip
    assert type(again) is NoPathFoundError and str(again) == text and str(again.source) == str(error.source)


def empty_result():
    """The empty ConstrainedPath a default run returns when the search finds no route (a wall of forbidden cells)."""
    raster = np.full((20, 20), 100, np.uint16)
    raster[:, 10] = FORBIDDEN
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        finder = ConstrainedPathFinder(
            dataset_source=raster, crs=CRS, transform=from_origin(0.0, 20.0, 1.0, 1.0), source_coords=(1.5, 10.5),
            target_coords=(18.5, 10.5), search_space_buffer_m=100, profile=dict(SYN_PROFILE), graph_api="cython",
            neighborhood_str="r1")
    path, _ = caught_warnings(finder.find_route)
    assert len(path.path_indices) == 0 and path.towers == []
    return path


def test_the_towers_frame_of_an_empty_result_is_empty_with_a_geometry_column_and_the_crs():
    path = empty_result()
    frame = path.towers_to_geodataframe(crs=CRS)
    assert len(frame) == 0 and frame.geometry.name == "geometry" and frame.crs.to_string() == CRS
    assert {"tower_id", "geometry", "tower_type", "span_to_previous_m", "span_to_next_m", "total_cost"} <= set(
        frame.columns)
    assert path.towers_to_geodataframe().crs is None and len(path.towers_to_geodataframe()) == 0
    full, _ = synthetic(east(0, 14), [(ROW, 7)])                                  # a route with towers: as before
    frame = full.towers_to_geodataframe(crs=CRS)
    assert len(frame) == 3 and frame.crs is not None


def test_other_no_path_errors_keep_their_text():
    text = str(NoPathFoundError(3, 4))
    assert text == "No path found from 3 to 4! Choose different source and target or increase buffer!"
    assert str(NoPathFoundError(3, 4, " extra")).endswith("increase buffer! extra")


# ================================================================================================== 7. the docs
def test_the_docs_say_that_enforce_can_be_cheaper_than_keep_or_remove():
    doc = ConstrainedPathFinder.__init__.__doc__
    assert "CHEAPER" in doc and '"keep"' in doc and '"remove"' in doc and "coarse" in doc
    assert "skips" in doc and "shortcut" in doc and "limitation" in doc      # the shortcut and its limitation


# ======================================================================================== 8. positions, not cells
def loop_cells():
    """East along row 5, south along column 10, west along row 9, north along column 3: the route crosses itself at
    (5, 3), passed at positions 3 and 25 of 29."""
    cells = [(5, c) for c in range(0, 11)]
    cells += [(r, 10) for r in range(6, 10)]
    cells += [(9, c) for c in range(9, 2, -1)]
    cells += [(r, 3) for r in range(8, 1, -1)]
    return cells


LOOP = dict(rows=12, cols=16)
THREE_HEIGHTS = dict(tower_height_m=None, tower_heights_m=[10.0, 14.0, 18.0], tower_height_cost_per_increment=30.0)


def at(row, col, cols=LOOP["cols"]):
    return row * cols + col


def test_the_hand_made_loop_really_crosses_itself():
    cells = loop_cells()
    assert len(cells) == 29 and cells.index((5, 3)) == 3 and cells[25] == (5, 3)
    assert len(set(cells)) == 28 and cells[0] == (5, 0) and cells[-1] == (2, 3)


def test_two_towers_on_one_cell_of_a_route_that_crosses_itself_are_both_reported_with_their_own_data():
    cells = loop_cells()
    entries = [(5, 3), (5, 10), (9, 10), (9, 3), (5, 3)]          # the kernel names the cell (5, 3) twice
    path, warned = synthetic(cells, entries, heights=[10.0, 14.0, 10.0, 18.0, 14.0], **LOOP, **THREE_HEIGHTS)
    assert warned == []
    assert path.n_towers == 7 and [t.tower_id for t in path.towers] == list(range(7))
    assert [t.cell_index for t in path.towers] == [at(*c) for c in
                                                   [(5, 0), (5, 3), (5, 10), (9, 10), (9, 3), (5, 3), (2, 3)]]
    assert path.spans == pytest.approx([3.0, 7.0, 4.0, 7.0, 4.0, 3.0])
    assert [t.height_m for t in path.towers] == [10.0, 10.0, 14.0, 10.0, 18.0, 14.0, 10.0]
    assert [t.height_premium for t in path.towers] == pytest.approx([0.0, 0.0, 40.0, 0.0, 80.0, 40.0, 0.0])
    assert path.cost_breakdown["height_premiums"] == pytest.approx(160.0)
    first, second = path.towers[1], path.towers[5]               # the same cell, two towers, two heights
    assert first.cell_index == second.cell_index and first.height_m != second.height_m
    assert [t.tower_type for t in path.towers].count("terminal") == 2
    assert second.turn_angle_deg == 0.0 and first.turn_angle_deg == 0.0


def test_a_tower_at_only_one_passage_of_a_crossing_cell_is_one_tower_at_that_passage():
    cells = loop_cells()
    entries = [(5, 10), (9, 10), (9, 3), (5, 3)]                  # no tower at the first passage (position 3)
    path, _ = synthetic(cells, entries, **LOOP)
    assert [t.cell_index for t in path.towers] == [at(*c) for c in
                                                   [(5, 0), (5, 10), (9, 10), (9, 3), (5, 3), (2, 3)]]
    assert path.spans == pytest.approx([10.0, 4.0, 7.0, 4.0, 3.0])   # 0 -> 10 directly, as the kernel output says


def test_tower_entries_are_matched_to_passages_in_route_order():
    path = np.array([at(*c) for c in loop_cells()])
    entries = [at(5, 3), at(5, 10), at(9, 10), at(9, 3), at(5, 3)]
    match = ConstrainedPathFinder._tower_positions(path, entries, LOOP["cols"])
    assert match == [3, 10, 14, 21, 25]
    assert ConstrainedPathFinder._tower_positions(path, entries[1:], LOOP["cols"]) == [10, 14, 21, 25]
    assert ConstrainedPathFinder._tower_positions(path, [at(0, 0), at(5, 10)], LOOP["cols"]) == [None, 10]
    assert ConstrainedPathFinder._tower_positions(path, [at(5, 10), at(5, 3)], LOOP["cols"]) == [10, 25]
    assert ConstrainedPathFinder._tower_positions(path, [at(2, 3)], LOOP["cols"]) == [None]    # the last cell: no tower
    assert ConstrainedPathFinder._tower_positions(path[:1], [at(5, 0)], LOOP["cols"]) == [None]
    assert ConstrainedPathFinder._tower_positions(path, [], LOOP["cols"]) == []


def revisit_cells(last_row=2):
    """East along row 5, south along column 6, west along row 9, north along column 0 through the START cell (5, 0)."""
    cells = [(5, c) for c in range(0, 7)]
    cells += [(r, 6) for r in range(6, 10)]
    cells += [(9, c) for c in range(5, -1, -1)]
    cells += [(r, 0) for r in range(8, last_row - 1, -1)]
    return cells


def test_a_tower_at_a_later_passage_of_the_start_cell_is_a_line_tower_not_a_terminal():
    cells = revisit_cells()
    assert cells[20] == (5, 0) and cells[0] == (5, 0)
    path, _ = synthetic(cells, [(5, 6), (9, 6), (9, 0), (5, 0)], **LOOP)
    assert [t.cell_index for t in path.towers] == [at(*c) for c in
                                                   [(5, 0), (5, 6), (9, 6), (9, 0), (5, 0), (2, 0)]]
    assert [t.tower_type for t in path.towers][0] == "terminal" and path.towers[4].tower_type == "suspension"
    assert path.spans == pytest.approx([6.0, 4.0, 6.0, 4.0, 3.0])


def test_remove_takes_out_the_tower_at_the_second_passage_only():
    """The last search tower stands at the second passage of the start cell, 2 m from the target; the first passage is
    the source terminal and stays."""
    cells = revisit_cells(last_row=3)
    path, warned = synthetic(cells, [(5, 6), (9, 6), (9, 0), (5, 0)], rule="remove", **LOOP)
    assert warned == []
    assert [t.cell_index for t in path.towers] == [at(*c) for c in [(5, 0), (5, 6), (9, 6), (9, 0), (3, 0)]]
    (gone,) = path.removed_towers
    assert gone.cell_index == at(5, 0) and gone.tower_id == -1 and "removed by terminal_span_rule" in gone.note
    assert path.towers[0].tower_type == "terminal" and path.towers[0].note is None
    kept, _ = synthetic(cells, [(5, 6), (9, 6), (9, 0), (5, 0)], rule="keep", **LOOP)
    assert [t.cell_index for t in kept.towers].count(at(5, 0)) == 2 and kept.removed_towers == []
    assert kept.towers[4].note and "shorter than min_span_m" in kept.towers[4].note
    assert kept.towers[0].note is None


def test_the_strict_check_works_on_positions_of_a_crossing_route():
    cells = loop_cells()
    obstacle = np.zeros((12, 16), np.float32)
    obstacle[5, 4] = 8.0              # first cell after the tower at position 3 (passage 1 of (5, 3)): east leg
    path, warned = synthetic(cells, [(5, 3), (5, 10), (9, 10), (9, 3), (5, 3)], obstacle=obstacle,
                             heights=[10.0] * 5, **LOOP, **THREE_HEIGHTS)
    (warning,) = warned
    text = str(warning.message)
    assert "span 2 of 6 from (3.5, 6.5) to (10.5, 6.5)" in text                  # positions 3 -> 10 only
    assert len(path.span_notes) == 1 and path.n_towers == 7


# ------------------------------------------------------------------------- self-crossing routes from the real kernel
GRID_FORBIDDEN = 65535


def grid_case(seed):
    """A random small r1 grid like the kernel review's: tower and turn costs make a route cross itself now and then."""
    rng = np.random.default_rng(seed)
    rows, cols = int(rng.integers(4, 8)), int(rng.integers(8, 15))
    raster = rng.integers(10, 60, size=(rows, cols)).astype(np.uint16)
    raster[rng.random((rows, cols)) < 0.1] = GRID_FORBIDDEN
    free = [(r, c) for r in range(rows) for c in range(cols) if raster[r, c] != GRID_FORBIDDEN]
    while True:
        source, target = free[int(rng.integers(0, len(free)))], free[int(rng.integers(0, len(free)))]
        if abs(source[1] - target[1]) + abs(source[0] - target[0]) >= 3:
            break
    return raster, source, target, float(rng.choice([2.5, 3.0])), float(rng.choice([5.5, 6.5, 7.5]))


def grid_finder(seed, rule="enforce"):
    raster, source, target, min_span, max_span = grid_case(seed)
    rows, _ = raster.shape
    profile = dict(
        name="grid", description="random grid", soft_angle_limit_deg=10.0, hard_angle_limit_deg=90.0,
        angle_cost_function="linear", angle_cost_params={"scale": 1.0}, min_span_m=min_span, max_span_m=max_span,
        span_bin_size_m=1.0, terminal_span_rule=rule,
        tower_cost_params={"terrain_cost_map": {0: 40, 200: 300},
                           "angle_types": {"suspension": {"max_angle_deg": 10.0, "base_cost": 20.0},
                                           "angle": {"max_angle_deg": 50.0, "base_cost": 60.0},
                                           "dead_end": {"max_angle_deg": 95.0, "base_cost": 150.0}}})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        finder = ConstrainedPathFinder(
            dataset_source=raster, crs=CRS, transform=from_origin(0.0, float(rows), 1.0, 1.0),
            source_coords=(source[1] + 0.5, rows - source[0] - 0.5),
            target_coords=(target[1] + 0.5, rows - target[0] - 0.5), search_space_buffer_m=100, profile=profile,
            graph_api="cython", neighborhood_str="r1")
    return finder, min_span, max_span


@pytest.fixture
def dijkstra_spy(monkeypatch):
    """Record the route, the towers and the optimal distance the Dijkstra kernel returned."""
    original = algorithms.constrained_dijkstra_2d
    calls = []

    def spy(**kwargs):
        path, towers, dist = original(return_dist=1, **kwargs)
        calls.append({"path": [int(c) for c in path], "towers": [int(t) for t in towers], "dist": float(dist)})
        return path, towers

    monkeypatch.setattr(algorithms, "constrained_dijkstra_2d", spy)
    return calls


#: (seed, a cell carries two towers in the kernel's list): routes of the random grids that cross themselves
CROSSING_SEEDS = [(7, False), (33, False), (62, False), (398, False), (145, True), (481, True)]


@pytest.mark.parametrize("seed, two_towers_on_a_cell", CROSSING_SEEDS)
def test_a_route_that_crosses_itself_is_reported_tower_by_tower(seed, two_towers_on_a_cell, dijkstra_spy):
    finder, min_span, max_span = grid_finder(seed)
    path, _ = caught_warnings(finder.find_route)
    (call,) = dijkstra_spy
    cells = [int(c) for c in path.path_indices]
    assert len(set(cells)) < len(cells)                                       # the route really crosses itself
    assert (len(set(call["towers"])) < len(call["towers"])) == two_towers_on_a_cell
    # every tower the kernel named is a line tower of the report, at one passage of its cell; none lost, none added
    assert path.n_towers == len(call["towers"]) + 2
    assert [t.cell_index for t in path.towers[1:-1]] == call["towers"]
    # the cost identity: reported total minus the terminals = the kernel's optimum
    terminals = sum(t.terrain_cost + t.angle_cost for t in path.towers if t.tower_type == "terminal")
    assert path.total_cost - terminals == pytest.approx(call["dist"], rel=1e-6)           # float32 sums in the kernel
    # a tower stands at every direction change, and only there (or on a straight stretch by the span rule) is a
    # turn angle reported: the turning towers are exactly the direction changes of the route
    ncols = grid_case(seed)[0].shape[1]
    steps = [(b // ncols - a // ncols, b % ncols - a % ncols) for a, b in zip(cells[:-1], cells[1:])]
    turns = [k for k in range(1, len(steps)) if steps[k] != steps[k - 1]]
    assert sum(1 for t in path.towers[1:-1] if t.turn_angle_deg > 0.0) == len(turns)
    # enforce: every span, the end spans included, is in [min_span_m, max_span_m)
    assert all(min_span * (1 - 1e-6) <= s < max_span * (1 + 1e-6) for s in path.spans), path.spans
    if two_towers_on_a_cell:
        doubled = next(c for c in call["towers"] if call["towers"].count(c) > 1)
        on_cell = [t for t in path.towers if t.cell_index == doubled]
        assert len(on_cell) == 2 and on_cell[0].tower_id != on_cell[1].tower_id     # two towers on one cell
