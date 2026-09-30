"""Unit tests: pyorps.gui.services.cost — the metric must equal pyorps' own."""
import numpy as np
import pytest
from shapely.geometry import LineString

from pyorps.gui.services import cost


def test_cost_matches_pyorps_metric(finder):
    path = finder.paths.all[0]
    rc = cost.evaluate_route_cost(path.path_geometry, finder.raster_handler)
    # Same numba kernel on the same 8-connected chain -> identical totals (C12).
    assert rc.total_length == pytest.approx(path.total_length, rel=1e-6)
    assert rc.total_cost == pytest.approx(path.total_cost, rel=1e-6)
    assert rc.total_cell_cost == pytest.approx(path.total_cell_cost, rel=1e-6)
    assert not rc.crosses_forbidden


def test_cost_flags_forbidden(finder):
    handler = finder.raster_handler
    forbidden = np.iinfo(handler.data.dtype).max
    handler.data[0, 5, 5] = forbidden
    handler.data[0, 5, 6] = forbidden
    xs = handler.indices_to_coords([(5, 4), (5, 7)])
    line = LineString([tuple(xs[0]), tuple(xs[1])])
    rc = cost.evaluate_route_cost(line, handler)
    assert rc.crosses_forbidden
    assert rc.n_forbidden_cells >= 1


def test_bresenham_is_8_connected():
    cells = cost._bresenham(0, 0, 3, 7)
    assert cells[0] == (0, 0) and cells[-1] == (3, 7)
    for (r0, c0), (r1, c1) in zip(cells[:-1], cells[1:]):
        assert max(abs(r1 - r0), abs(c1 - c0)) == 1
