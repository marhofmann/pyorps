"""The Python clearance check of one straight span (``pyorps.graph.span_clearance``), round 17.

The check repeats ``_check_span_clearance_vh`` of the constrained search so that a route can be verified without the
kernels. Three kinds of tests:

1. A differential test against the fixed-height Cython kernel: on a one-row corridor whose tower positions are forced
   (every span has exactly the same length), the kernel finds a route if and only if every span clears the ground, so
   the Python answer for each span must give the same verdict. Random ground and obstacle heights, both verdicts occur.
2. The closed form on flat terrain: ``z(x) = z_A + (z_B - z_A) x / L - w x (L - x) / (2 T)``, minimised over the cells
   the search samples (written here independently of the module, in double precision).
3. The documented quirks of the kernel's walk: the first cell after tower A is not sampled, a span of one step is not
   checked, tall towers skip the walk, intermediate cells of a diagonal or knight step are sampled.
"""

import math

import numpy as np
import pytest

import pyorps.utils.constrained_path_algorithms as constrained_algorithms
from pyorps.core.exceptions import InvalidInputError
from pyorps.core.infrastructure_profile import InfrastructureProfile
from pyorps.graph.span_clearance import SpanClearance, path_span_clearance, span_clearance, straight_step
from pyorps.utils.neighborhood import get_neighborhood_steps

W, T, MIN_CLEARANCE = 10.0, 100.0, 5.0
HEIGHT = 10.0


@pytest.fixture(autouse=True)
def one_thread(monkeypatch):
    """The delta-stepping kernel reads its thread count from OMP_NUM_THREADS: the tests do not need more than one."""
    monkeypatch.setenv("OMP_NUM_THREADS", "1")


def ok(dem, obstacle, a, step, n, height_a=HEIGHT, height_b=HEIGHT, *, step_distance=1.0, w=W, t=T,
       min_clearance=MIN_CLEARANCE):
    return span_clearance(dem, obstacle, a, step, n, step_distance, height_a, height_b, conductor_weight_per_m=w,
                          conductor_tension_n=t, min_clearance_m=min_clearance)


# ------------------------------------------------------------------------------------------ the kernel as the oracle
ROW = 1
SPAN = 7           # the corridor's towers stand every SPAN cells (min_span = SPAN, max_span just above it)
N_SPANS = 3
COLS = N_SPANS * SPAN + 2   # the target is one cell behind the last tower, so the last span is a single step


def kernel_profile():
    return InfrastructureProfile.from_dict(dict(
        name="p", description="p", soft_angle_limit_deg=5.0, hard_angle_limit_deg=90.0,
        angle_cost_function="linear", angle_cost_params={"scale": 1.0},
        min_span_m=float(SPAN), max_span_m=SPAN + 0.5, span_bin_size_m=1.0,
        tower_cost_params={"terrain_cost_map": {0: 100, 500: 100},
                           "angle_types": {"s": {"max_angle_deg": 90.0, "base_cost": 50.0}}},
        tower_height_m=HEIGHT, conductor_weight_per_m=W, conductor_tension_n=T, min_clearance_m=MIN_CLEARANCE,
        max_gradient_percent=100000.0, gradient_cost_params={"scale": 0.0}))


def run_kernel(dem, obstacle):
    """Towers (columns) of the fixed-height kernel on the forced corridor; [] when it finds no route."""
    steps = get_neighborhood_steps("r1")
    prof = kernel_profile()
    angle_lut, angle_valid = prof.precompute_angle_lut(steps)
    raster = np.full((3, COLS), 65535, np.uint16)
    raster[ROW, :] = 100
    path, towers, _ = constrained_algorithms.constrained_delta_stepping_height_2d(
        raster=raster, source_row=ROW, source_col=0, target_row=ROW, target_col=COLS - 1, steps=steps,
        angle_cost_lut=angle_lut.astype(np.float32), angle_valid_lut=angle_valid.astype(np.uint8),
        step_distances=prof.compute_step_distances(steps, 1.0).astype(np.float32),
        tower_terrain_costs=prof.precompute_tower_terrain_costs().astype(np.float32),
        tower_angle_costs=prof.precompute_tower_angle_costs(steps).astype(np.float32),
        n_span_bins=2, span_bin_size=float(SPAN), min_span=float(SPAN), max_span=SPAN + 0.5,
        dem_data=dem, cell_size=1.0, tower_heights=np.array([HEIGHT], np.float32),
        height_premiums=np.array([0.0], np.float32), conductor_weight_per_m=W, conductor_tension=T,
        min_clearance_val=MIN_CLEARANCE, obstacle_heights=obstacle, max_gradient_pct=100000.0, gradient_scale=0.0)
    return [int(t) % COLS for t in towers] if len(path) else []


def python_spans_clear(dem, obstacle):
    chain = [0] + [SPAN * k for k in range(1, N_SPANS + 1)]
    return [ok(dem, obstacle, (ROW, a), (0, 1), b - a).ok for a, b in zip(chain[:-1], chain[1:])]


def test_the_forced_corridor_gives_the_expected_towers_on_flat_ground():
    flat = np.zeros((3, COLS), np.float32)
    assert run_kernel(flat, flat.copy()) == [SPAN * k for k in range(1, N_SPANS + 1)]
    assert python_spans_clear(flat, flat.copy()) == [True] * N_SPANS


def test_python_check_gives_the_kernels_verdict_on_random_corridors():
    verdicts = []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        dem = rng.uniform(0.0, 2.0, (3, COLS)).astype(np.float32)
        obstacle = np.where(rng.random((3, COLS)) < 0.12, rng.uniform(0.0, 9.0, (3, COLS)), 0.0).astype(np.float32)
        kernel_clear = run_kernel(dem, obstacle) != []
        python_clear = all(python_spans_clear(dem, obstacle))
        assert kernel_clear == python_clear, f"seed {seed}: kernel {kernel_clear}, python {python_clear}"
        verdicts.append(kernel_clear)
    # the test only means something when both verdicts occur
    assert 5 <= sum(verdicts) <= len(verdicts) - 5, sum(verdicts)


def test_python_check_fails_the_last_span_that_the_kernel_never_looks_at():
    """The kernel accepts the target at any span: an obstacle under the last span is invisible to it."""
    rows, cols = 5, 61
    dem = np.zeros((rows, cols), np.float32)
    obstacle = np.zeros((rows, cols), np.float32)
    obstacle[2, 55:58] = 20.0
    prof = InfrastructureProfile.from_dict(dict(
        name="p", description="p", soft_angle_limit_deg=5.0, hard_angle_limit_deg=90.0,
        angle_cost_function="linear", angle_cost_params={"scale": 1.0}, min_span_m=5.0, max_span_m=12.0,
        span_bin_size_m=1.0, tower_cost_params={"terrain_cost_map": {0: 100, 500: 100},
                                                "angle_types": {"s": {"max_angle_deg": 90.0, "base_cost": 50.0}}},
        tower_height_m=HEIGHT, conductor_weight_per_m=0.001, conductor_tension_n=1.0e6, min_clearance_m=5.0,
        max_gradient_percent=100000.0, gradient_cost_params={"scale": 0.0}))
    steps = get_neighborhood_steps("r1")
    angle_lut, angle_valid = prof.precompute_angle_lut(steps)
    raster = np.full((rows, cols), 65535, np.uint16)
    raster[2, :] = 100
    path, towers, _ = constrained_algorithms.constrained_delta_stepping_height_2d(
        raster=raster, source_row=2, source_col=0, target_row=2, target_col=cols - 1, steps=steps,
        angle_cost_lut=angle_lut.astype(np.float32), angle_valid_lut=angle_valid.astype(np.uint8),
        step_distances=prof.compute_step_distances(steps, 1.0).astype(np.float32),
        tower_terrain_costs=prof.precompute_tower_terrain_costs().astype(np.float32),
        tower_angle_costs=prof.precompute_tower_angle_costs(steps).astype(np.float32),
        n_span_bins=3, span_bin_size=5.0, min_span=5.0, max_span=12.0, dem_data=dem, cell_size=1.0,
        tower_heights=np.array([HEIGHT], np.float32), height_premiums=np.array([0.0], np.float32),
        conductor_weight_per_m=0.001, conductor_tension=1.0e6, min_clearance_val=5.0,
        obstacle_heights=obstacle, max_gradient_pct=100000.0, gradient_scale=0.0)
    chain = [0] + [int(t) % cols for t in towers] + [cols - 1]
    assert len(path) and len(chain) > 2
    kwargs = dict(conductor_weight_per_m=0.001, conductor_tension_n=1.0e6, min_clearance_m=5.0)
    verdicts = [span_clearance(dem, obstacle, (2, a), (0, 1), b - a, 1.0, HEIGHT, HEIGHT, **kwargs).ok
                for a, b in zip(chain[:-1], chain[1:])]
    assert verdicts[:-1] == [True] * (len(verdicts) - 1)     # every span the kernel placed a tower for clears
    assert verdicts[-1] is False                              # the last span crosses the obstacle


# ----------------------------------------------------------------------------------------------------- closed form
def closed_form_worst(n, step_distance, height_a, height_b, w, t):
    """Smallest conductor height over the cells the search samples (x = 2 .. n steps), ground at 0, in float64."""
    length = n * step_distance
    return min(height_a + (height_b - height_a) * x / length - w * x * (length - x) / (2.0 * t)
               for x in (j * step_distance for j in range(2, n + 1)))


@pytest.mark.parametrize("cell", [1.0, 2.5])
@pytest.mark.parametrize("heights", [(10.0, 10.0), (10.0, 11.0), (14.0, 9.0)])
def test_worst_clearance_follows_the_closed_form_on_flat_ground(cell, heights):
    n, w, t = 10, 100.0, 1000.0
    dem = np.zeros((3, 40), np.float32)
    expected = closed_form_worst(n, cell, *heights, w, t)
    res = ok(dem, None, (1, 0), (0, 1), n, *heights, step_distance=cell, w=w, t=t, min_clearance=expected - 0.5)
    assert isinstance(res, SpanClearance) and res.ok and res.checked
    assert res.worst_clearance_m == pytest.approx(expected, rel=1e-5)
    # the sample that is lowest: x on the lattice, and its cell lies on the span
    assert res.worst_cell[0] == 1 and 2 <= res.worst_cell[1] <= n
    assert res.worst_offset_m == pytest.approx(res.worst_cell[1] * cell, rel=1e-6)
    # a limit 0.5 m above the lowest point fails, 0.5 m below passes (margin far above the float32 rounding)
    assert not ok(dem, None, (1, 0), (0, 1), n, *heights, step_distance=cell, w=w, t=t,
                  min_clearance=expected + 0.5).ok
    assert ok(dem, None, (1, 0), (0, 1), n, *heights, step_distance=cell, w=w, t=t,
              min_clearance=expected - 0.5).ok


def test_a_higher_end_gantry_lets_the_span_clear_and_a_lower_one_does_not():
    dem = np.zeros((3, 40), np.float32)
    kwargs = dict(w=100.0, t=1000.0, min_clearance=8.0)
    n = 10
    assert ok(dem, None, (1, 0), (0, 1), n, 14.0, 10.0, **kwargs).ok
    assert not ok(dem, None, (1, 0), (0, 1), n, 3.0, 10.0, **kwargs).ok


def test_ground_and_obstacle_heights_lower_the_clearance_one_for_one():
    n, w, t = 8, 100.0, 1000.0
    flat = np.zeros((3, 40), np.float32)
    lowest = closed_form_worst(n, 1.0, HEIGHT, HEIGHT, w, t)        # at x = 4: 10 - 100 * 4 * 4 / 2000 = 9.2
    assert lowest == pytest.approx(9.2)
    assert ok(flat, None, (1, 0), (0, 1), n, w=w, t=t, min_clearance=0.0).worst_cell == (1, 4)
    # a 1.5 m rise of the ground and a 2 m obstacle on that cell: the conductor stays where it was (the attachment
    # points are on other cells), so the clearance there drops by 3.5 m
    dem, obstacle = flat.copy(), flat.copy()
    dem[1, 4] = 1.5
    obstacle[1, 4] = 2.0
    res = ok(dem, obstacle, (1, 0), (0, 1), n, w=w, t=t, min_clearance=0.0)
    assert res.worst_cell == (1, 4)
    assert res.worst_clearance_m == pytest.approx(lowest - 3.5, rel=1e-5)


# ------------------------------------------------------------------------------------- the kernel's walk, as it is
def test_the_first_cell_after_tower_a_is_not_sampled_but_the_second_is():
    dem = np.zeros((3, 40), np.float32)
    first = np.zeros((3, 40), np.float32)
    first[1, 1] = 30.0
    second = np.zeros((3, 40), np.float32)
    second[1, 2] = 30.0
    assert ok(dem, first, (1, 0), (0, 1), 6).ok
    assert not ok(dem, second, (1, 0), (0, 1), 6).ok


def test_the_cell_of_tower_b_is_sampled():
    dem = np.zeros((3, 40), np.float32)
    obstacle = np.zeros((3, 40), np.float32)
    obstacle[1, 6] = 7.0          # height 10 - 7 = 3 m above the obstacle at tower B
    res = ok(dem, obstacle, (1, 0), (0, 1), 6)
    assert not res.ok and res.worst_cell == (1, 6)


def test_a_span_of_one_step_or_less_is_not_checked():
    dem = np.zeros((3, 40), np.float32)
    obstacle = np.full((3, 40), 50.0, np.float32)
    for n in (0, 1):
        res = ok(dem, obstacle, (1, 0), (0, 1), n)
        assert res.ok and not res.checked and res.worst_clearance_m is None


def test_towers_tall_enough_skip_the_walk_like_the_kernel():
    dem = np.zeros((3, 40), np.float32)
    obstacle = np.zeros((3, 40), np.float32)
    obstacle[1, 4] = 100.0
    tall = ok(dem, obstacle, (1, 0), (0, 1), 6, 80.0, 80.0, w=1.0, t=1.0e5)
    assert tall.ok and not tall.checked                  # 80 - sag - 5 > 50: not looked at, as in the kernel
    walked = ok(dem, obstacle, (1, 0), (0, 1), 6, 50.0, 50.0, w=1.0, t=1.0e5)
    assert walked.checked and not walked.ok              # 50 - sag - 5 is not above 50: the walk sees the obstacle


def test_a_span_that_leaves_the_raster_is_not_clear():
    dem = np.zeros((3, 10), np.float32)
    beyond = ok(dem, None, (1, 5), (0, 1), 6)
    assert not beyond.ok and "outside" in beyond.reason
    before = ok(dem, None, (1, 12), (0, 1), 3)
    assert not before.ok and "outside" in before.reason


def test_intermediate_cells_of_a_diagonal_step_are_sampled():
    """A (1, 1) step passes over the two orthogonal cells too; the cell the step ends on is the first one skipped."""
    dem = np.zeros((10, 10), np.float32)

    def blocked(row, col):
        obstacle = np.zeros((10, 10), np.float32)
        obstacle[row, col] = 30.0
        return not ok(dem, obstacle, (0, 0), (1, 1), 4, step_distance=math.sqrt(2.0)).ok

    assert blocked(0, 1) and blocked(1, 0)      # intermediates of the first step, sampled
    assert not blocked(1, 1)                    # the first lattice cell after A: skipped
    assert blocked(2, 1) and blocked(1, 2)      # intermediates of the second step
    assert blocked(2, 2) and blocked(4, 4)      # lattice cells from the second step on, tower B included
    assert not blocked(0, 3) and not blocked(3, 0)   # cells no step passes over


def test_intermediate_cells_of_a_knight_step_are_sampled():
    dem = np.zeros((10, 10), np.float32)

    def blocked(row, col):
        obstacle = np.zeros((10, 10), np.float32)
        obstacle[row, col] = 30.0
        return not ok(dem, obstacle, (0, 0), (1, 2), 3, step_distance=math.sqrt(5.0)).ok

    assert blocked(0, 1) and blocked(1, 1)      # the floor and the ceil cell of the first step
    assert not blocked(1, 2)                    # first lattice cell after A: skipped
    assert not blocked(0, 2) and not blocked(2, 0)   # not passed over
    assert blocked(3, 6)                        # tower B


def test_obstacle_and_dem_shapes_and_the_step_are_checked():
    dem = np.zeros((3, 10), np.float32)
    with pytest.raises(InvalidInputError):
        ok(dem, None, (1, 0), (0, 0), 4)
    with pytest.raises(InvalidInputError):
        ok(np.zeros(10, np.float32), None, (1, 0), (0, 1), 4)
    with pytest.raises(InvalidInputError):
        ok(dem, np.zeros((3, 9), np.float32), (1, 0), (0, 1), 4)


# ------------------------------------------------------------------------------------------------ along a route
def test_straight_step_is_the_repeated_lattice_step_or_none_at_a_bend():
    rows = [1, 1, 1, 1, 1, 2, 3]
    cols = [0, 1, 2, 3, 4, 4, 4]
    assert straight_step(rows, cols, 0, 4) == (0, 1)
    assert straight_step(rows, cols, 4, 6) == (1, 0)
    assert straight_step(rows, cols, 0, 5) is None          # the bend at position 4
    assert straight_step(rows, cols, 3, 3) is None
    assert straight_step(rows, cols, 4, 2) is None


def test_path_span_clearance_equals_span_clearance_and_skips_a_bend():
    dem = np.zeros((6, 12), np.float32)
    obstacle = np.zeros((6, 12), np.float32)
    obstacle[1, 4] = 8.0
    rows = np.array([1, 1, 1, 1, 1, 1, 1, 2, 3])
    cols = np.array([0, 1, 2, 3, 4, 5, 6, 6, 6])
    kwargs = dict(conductor_weight_per_m=W, conductor_tension_n=T, min_clearance_m=MIN_CLEARANCE)
    along = path_span_clearance(dem, obstacle, rows, cols, 0, 6, 2.0, HEIGHT, HEIGHT, **kwargs)
    direct = span_clearance(dem, obstacle, (1, 0), (0, 1), 6, 2.0, HEIGHT, HEIGHT, **kwargs)
    assert along == direct and not along.ok
    assert path_span_clearance(dem, obstacle, rows, cols, 0, 8, 2.0, HEIGHT, HEIGHT, **kwargs) is None
    diagonal = path_span_clearance(dem, obstacle, np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3]), 0, 3, 3.0,
                                   HEIGHT, HEIGHT, **kwargs)
    assert diagonal == span_clearance(dem, obstacle, (0, 0), (1, 1), 3, 3.0 * math.sqrt(2.0), HEIGHT, HEIGHT,
                                      **kwargs)
