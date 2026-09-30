"""Tests for pyorps.utils.linestring_simplify."""
import math
import pytest
from shapely.geometry import LineString

from pyorps.utils.linestring_simplify import simplify_linestring


def test_douglas_peucker_removes_collinear_midpoints():
    line = LineString([(0, 0), (1, 0), (2, 0), (3, 0)])
    simplified = simplify_linestring(line, method="douglas_peucker",
                                     tolerance=0.001)
    # All midpoints are collinear; DP must collapse to endpoints only.
    assert list(simplified.coords) == [(0.0, 0.0), (3.0, 0.0)]


def test_douglas_peucker_preserves_endpoints_for_zigzag():
    coords = [(0, 0), (1, 5), (2, 0), (3, 5), (4, 0)]
    line = LineString(coords)
    # tolerance well below the zigzag amplitude → no simplification possible
    simplified = simplify_linestring(line, method="douglas_peucker",
                                     tolerance=0.1)
    assert simplified.coords[0] == (0.0, 0.0)
    assert simplified.coords[-1] == (4.0, 0.0)


def test_unknown_method_raises():
    line = LineString([(0, 0), (1, 1)])
    with pytest.raises(ValueError, match="Unknown simplification method"):
        simplify_linestring(line, method="nope", tolerance=1.0)


def test_negative_tolerance_raises():
    line = LineString([(0, 0), (1, 1)])
    with pytest.raises(ValueError, match="tolerance"):
        simplify_linestring(line, method="douglas_peucker", tolerance=-0.1)


def test_two_vertex_line_returns_self():
    line = LineString([(0, 0), (10, 10)])
    simplified = simplify_linestring(line, method="douglas_peucker",
                                     tolerance=1.0)
    assert list(simplified.coords) == [(0.0, 0.0), (10.0, 10.0)]


def test_visvalingam_drops_low_area_vertex():
    # Three nearly-collinear points: the middle triangle has area ≈ 0.
    line = LineString([(0, 0), (5, 0.001), (10, 0)])
    simplified = simplify_linestring(line, method="visvalingam", tolerance=1.0)
    assert list(simplified.coords) == [(0.0, 0.0), (10.0, 0.0)]


def test_visvalingam_keeps_high_area_vertex():
    # Sharp triangle with area = 0.5 * base * height = 0.5 * 10 * 5 = 25.
    line = LineString([(0, 0), (5, 5), (10, 0)])
    simplified = simplify_linestring(line, method="visvalingam", tolerance=1.0)
    # Mid-point survives; area (25) > tolerance (1).
    assert (5.0, 5.0) in list(simplified.coords)


def test_visvalingam_preserves_endpoints():
    coords = [(0, 0), (1, 0.0001), (2, 0.0001), (3, 0)]
    line = LineString(coords)
    simplified = simplify_linestring(line, method="visvalingam",
                                     tolerance=100.0)
    assert simplified.coords[0] == (0.0, 0.0)
    assert simplified.coords[-1] == (3.0, 0.0)


def test_grid_snaps_to_integer_grid():
    line = LineString([(0.1, 0.2), (1.4, 0.8), (2.6, 1.1), (3.9, 1.9)])
    simplified = simplify_linestring(line, method="grid", tolerance=1.0)
    coords = list(simplified.coords)
    # All vertices land on integer multiples of 1.0.
    for x, y in coords:
        assert abs(x - round(x)) < 1e-9
        assert abs(y - round(y)) < 1e-9


def test_grid_collapses_consecutive_duplicates():
    # Three points all snap to (0, 0); should collapse to (0,0) -> (1,1).
    line = LineString([(0.1, 0.1), (0.2, 0.2), (1.0, 1.0)])
    simplified = simplify_linestring(line, method="grid", tolerance=1.0)
    coords = list(simplified.coords)
    assert coords[0] != coords[-1]
    # No consecutive duplicates.
    for a, b in zip(coords, coords[1:]):
        assert a != b


def test_grid_tolerance_zero_is_identity():
    line = LineString([(0.1, 0.2), (1.4, 0.8)])
    simplified = simplify_linestring(line, method="grid", tolerance=0.0)
    assert list(simplified.coords) == [(0.1, 0.2), (1.4, 0.8)]
