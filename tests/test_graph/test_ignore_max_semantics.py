"""Config matrix pinning what ``ignore_max`` means for the cython backend.

``ignore_max=True``  -> 65535 (IMPASSABLE_CELL_COST) cells are obstacles.
``ignore_max=False`` -> no cell is an obstacle; 65535 is just a very
expensive cell.

Regression: ``CythonAPI`` used to map ``ignore_max=False`` to
``max_value=0``. Because the kernels build their exclude mask from
``raster == max_value``, that made every *0-cost* cell impassable - the
exact inverse of the intent, and the CPU twin of the GPU 0-sentinel bug
fixed in 40c0f3a. The wall-with-gap geometry is exercised under both
settings because a mis-configured run there is the most plausible source
of the recorded "cython Dijkstra traverses impassable cells" report.
"""

import numpy as np
import pytest

from pyorps.core.types import IMPASSABLE_CELL_COST
from pyorps.graph.api.cython_api import CythonAPI
from pyorps.utils._raster_context import (
    NO_EXCLUSION_VALUE,
    RasterContext,
    py_create_exclude_mask,
)

STEPS4 = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int8)

# The eight `_delta_stepping.pyx` wrappers were widened to `int64_t max_value`
# alongside the four in `_dijkstra.pyx`, so the out-of-domain no-exclusion
# sentinel now reaches their `(raster_arr != max_value)` masks. These cases pass
# on both kernels; the marker they used to carry is gone deliberately.


# ==================== FIXTURES ====================

@pytest.fixture
def zero_corridor():
    """5x7 raster, cost 100, with a free (0-cost) corridor along row 2.

    Routing (0,0) -> (0,6) costs 600 along row 0 but only 300 via the
    corridor, so an optimal route must use row 2.
    """
    r = np.full((5, 7), 100, dtype=np.uint16)
    r[2, :] = 0
    return r


@pytest.fixture
def full_wall():
    """5x7 raster, cost 100, with a full-height 65535 wall in column 3."""
    r = np.full((5, 7), 100, dtype=np.uint16)
    r[:, 3] = IMPASSABLE_CELL_COST
    return r


@pytest.fixture
def corridor_and_wall():
    """5x7 raster with both a 0-cost corridor and a full-height wall."""
    r = np.full((5, 7), 100, dtype=np.uint16)
    r[2, :] = 0
    r[:, 3] = IMPASSABLE_CELL_COST
    return r


@pytest.fixture
def wall_with_gap():
    """5x5 raster, cost 10, wall in column 2 with a gap at (4,2).

    Same geometry as tests/test_graph/test_dijkstra_solver.py.
    """
    r = np.full((5, 5), 10, dtype=np.uint16)
    r[0, 2] = IMPASSABLE_CELL_COST
    r[1, 2] = IMPASSABLE_CELL_COST
    r[2, 2] = IMPASSABLE_CELL_COST
    r[3, 2] = IMPASSABLE_CELL_COST
    return r


# ==================== HELPERS ====================

def _idx(row, col, cols):
    return row * cols + col


def _route(raster, source, target, ignore_max, algorithm="dijkstra", **kwargs):
    api = CythonAPI(raster, STEPS4, ignore_max=ignore_max)
    return api.shortest_path(source, target, algorithm=algorithm, **kwargs)


def _rows_of(path, cols):
    return {int(i) // cols for i in path}


def _cols_of(path, cols):
    return {int(i) % cols for i in path}


def _values_of(path, raster):
    return {int(raster.flat[int(i)]) for i in path}


# ==================== EXCLUDE MASK SEMANTICS ====================

def test_impassable_sentinel_marks_only_max_cells(corridor_and_wall):
    mask = py_create_exclude_mask(corridor_and_wall, IMPASSABLE_CELL_COST)
    expected = (corridor_and_wall != IMPASSABLE_CELL_COST).astype(np.uint8)
    np.testing.assert_array_equal(mask, expected)


def test_no_exclusion_sentinel_marks_nothing(corridor_and_wall):
    """0-cost and 65535 cells alike stay traversable."""
    mask = py_create_exclude_mask(corridor_and_wall, NO_EXCLUSION_VALUE)
    assert mask.shape == corridor_and_wall.shape
    assert (mask == 1).all()


def test_no_exclusion_sentinel_is_outside_the_uint16_domain():
    assert NO_EXCLUSION_VALUE > np.iinfo(np.uint16).max


def test_in_domain_value_still_excludes(corridor_and_wall):
    """The mask builder stays a generic utility: 0 means "exclude 0-cost"."""
    mask = py_create_exclude_mask(corridor_and_wall, 0)
    np.testing.assert_array_equal(mask, (corridor_and_wall != 0).astype(np.uint8))


def test_raster_context_no_exclusion_makes_wall_traversable(full_wall):
    blocked = RasterContext(full_wall, STEPS4, IMPASSABLE_CELL_COST)
    assert not blocked.is_traversable(0, 3)
    assert blocked.is_traversable(0, 0)

    open_ctx = RasterContext(full_wall, STEPS4, NO_EXCLUSION_VALUE)
    assert open_ctx.is_traversable(0, 3)


# ==================== API MAPPING ====================

@pytest.mark.parametrize("ignore_max, expected", [
    (True, IMPASSABLE_CELL_COST),
    (False, NO_EXCLUSION_VALUE),
])
def test_cython_api_max_value_mapping(ignore_max, expected):
    api = CythonAPI(np.ones((3, 3), dtype=np.uint16), STEPS4,
                    ignore_max=ignore_max)
    assert api.max_value == expected


# ==================== 0-COST CELLS ====================

@pytest.mark.parametrize("ignore_max", [True, False])
def test_zero_cost_corridor_is_used(zero_corridor, ignore_max):
    """0-cost cells are cheap, never obstacles - under either setting."""
    cols = zero_corridor.shape[1]
    path = _route(zero_corridor, _idx(0, 0, cols), _idx(0, 6, cols),
                  ignore_max)
    assert path, "no route found across a fully connected raster"
    assert 2 in _rows_of(path, cols)


@pytest.mark.parametrize("ignore_max", [True, False])
def test_route_between_zero_cost_endpoints(zero_corridor, ignore_max):
    """Source and target sitting on 0-cost cells must be reachable."""
    cols = zero_corridor.shape[1]
    source, target = _idx(2, 0, cols), _idx(2, 6, cols)
    path = _route(zero_corridor, source, target, ignore_max)
    assert path
    assert path[0] == source
    assert path[-1] == target
    assert _rows_of(path, cols) == {2}


# ==================== 65535 WALLS ====================

def test_full_wall_blocks_when_ignore_max_true(full_wall):
    cols = full_wall.shape[1]
    path = _route(full_wall, _idx(2, 0, cols), _idx(2, 6, cols),
                  ignore_max=True)
    assert path == []


def test_full_wall_is_crossed_when_ignore_max_false(full_wall):
    cols = full_wall.shape[1]
    source, target = _idx(2, 0, cols), _idx(2, 6, cols)
    path = _route(full_wall, source, target, ignore_max=False)
    assert path
    assert path[0] == source
    assert path[-1] == target
    assert 3 in _cols_of(path, cols)


# ==================== BOTH ====================

def test_corridor_and_wall_blocks_when_ignore_max_true(corridor_and_wall):
    cols = corridor_and_wall.shape[1]
    path = _route(corridor_and_wall, _idx(2, 0, cols), _idx(2, 6, cols),
                  ignore_max=True)
    assert path == []


def test_corridor_and_wall_when_ignore_max_false(corridor_and_wall):
    """The route runs along the 0-cost corridor and crosses the wall once."""
    cols = corridor_and_wall.shape[1]
    source, target = _idx(2, 0, cols), _idx(2, 6, cols)
    path = _route(corridor_and_wall, source, target, ignore_max=False)
    assert path
    assert _rows_of(path, cols) == {2}
    assert 3 in _cols_of(path, cols)


# ==================== WALL WITH GAP ====================

@pytest.mark.parametrize("ignore_max", [True, False])
def test_wall_with_gap_detours_through_the_gap(wall_with_gap, ignore_max):
    """With ignore_max=False the wall is legal but costs 65535 per cell,
    so the optimal route still detours through the gap at (4,2)."""
    cols = wall_with_gap.shape[1]
    source, target = _idx(0, 0, cols), _idx(0, 4, cols)
    path = _route(wall_with_gap, source, target, ignore_max)
    assert path
    assert path[0] == source
    assert path[-1] == target
    assert _idx(4, 2, cols) in path
    assert IMPASSABLE_CELL_COST not in _values_of(path, wall_with_gap)


# ==================== DELTA-STEPPING ====================

def test_delta_stepping_zero_corridor_ignore_max_true(zero_corridor):
    cols = zero_corridor.shape[1]
    path = _route(zero_corridor, _idx(0, 0, cols), _idx(0, 6, cols),
                  ignore_max=True, algorithm="delta-stepping", num_threads=1)
    assert path
    assert 2 in _rows_of(path, cols)


def test_delta_stepping_zero_corridor_ignore_max_false(zero_corridor):
    cols = zero_corridor.shape[1]
    path = _route(zero_corridor, _idx(0, 0, cols), _idx(0, 6, cols),
                  ignore_max=False, algorithm="delta-stepping", num_threads=1)
    assert path
    assert 2 in _rows_of(path, cols)


def test_delta_stepping_full_wall_is_crossed_when_ignore_max_false(full_wall):
    cols = full_wall.shape[1]
    path = _route(full_wall, _idx(2, 0, cols), _idx(2, 6, cols),
                  ignore_max=False, algorithm="delta-stepping", num_threads=1)
    assert path
    assert 3 in _cols_of(path, cols)
