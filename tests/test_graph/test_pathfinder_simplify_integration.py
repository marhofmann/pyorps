"""End-to-end tests for find_route(simplify=...)."""
import math
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from pyorps.graph.path_finder import PathFinder


@pytest.fixture
def simple_raster(tmp_path):
    """A 20x20 raster with two categories split vertically (cols < 10 -> 1,
    else 2). Returns the written .tif path."""
    data = np.ones((20, 20), dtype=np.uint16)
    data[:, 10:] = 2
    transform = from_origin(0, 20, 1.0, 1.0)  # origin top-left, 1m pixels
    path = tmp_path / "split.tif"
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=20, width=20,
        count=1, dtype="uint16",
        crs="EPSG:31467",  # arbitrary projected CRS in metres
        transform=transform,
    ) as dst:
        dst.write(data, 1)
    return str(path)


def _build_finder(raster_path):
    return PathFinder(
        dataset_source=raster_path,
        source_coords=(0.5, 19.5),
        target_coords=(19.5, 0.5),
        search_space_buffer_m=20,
    )


def test_simplify_kwarg_passes_metadata(simple_raster):
    finder = _build_finder(simple_raster)
    with pytest.warns(UserWarning, match="simplified geometry may traverse"):
        path = finder.find_route(
            simplify={"method": "douglas_peucker", "tolerance": 0.5}
        )
    assert path.simplification_method == "douglas_peucker"
    assert path.simplification_tolerance == 0.5
    assert path.original_path_geometry is not None
    assert len(path.path_geometry.coords) < len(
        path.original_path_geometry.coords
    )


def test_simplify_none_is_no_op(simple_raster):
    finder = _build_finder(simple_raster)
    path = finder.find_route()
    assert path.simplification_method is None
    assert path.original_path_geometry is None


def test_simplified_total_length_matches_unsimplified_geometry(simple_raster):
    """Cost / length metrics are computed from the un-simplified line; the
    simplified geometry is only used for export. So total_length must match
    the original (un-simplified) line length, not the simplified one."""
    finder = _build_finder(simple_raster)
    with pytest.warns(UserWarning):
        path = finder.find_route(
            simplify={"method": "douglas_peucker", "tolerance": 0.5}
        )
    assert math.isclose(
        path.total_length, path.original_path_geometry.length, rel_tol=1e-6
    )


def test_simplify_preserves_unsimplified_metrics(simple_raster):
    """Per-category lengths, total cost and cell cost must be byte-identical
    to the un-simplified routing because simplification only touches the
    exported geometry, never the metrics."""
    raw = _build_finder(simple_raster).find_route()
    with pytest.warns(UserWarning):
        simp = _build_finder(simple_raster).find_route(
            simplify={"method": "douglas_peucker", "tolerance": 0.5}
        )
    assert simp.total_length == raw.total_length
    assert simp.total_cost == raw.total_cost
    assert simp.total_cell_cost == raw.total_cell_cost
    assert simp.length_by_category == raw.length_by_category


def test_per_category_lengths_sum_to_total(simple_raster):
    finder = _build_finder(simple_raster)
    with pytest.warns(UserWarning):
        path = finder.find_route(
            simplify={"method": "douglas_peucker", "tolerance": 0.5}
        )
    assert math.isclose(
        sum(path.length_by_category.values()),
        path.total_length,
        rel_tol=1e-9,
    )
    assert math.isclose(
        sum(path.length_by_category_percent.values()),
        100.0,
        abs_tol=1e-6,
    )


def test_tolerance_zero_preserves_geometry(simple_raster):
    """tolerance=0 should not drop inflection points; per-category totals
    should agree with the un-simplified case to high precision."""
    raw = _build_finder(simple_raster).find_route()
    with pytest.warns(UserWarning):
        simp = _build_finder(simple_raster).find_route(
            simplify={"method": "douglas_peucker", "tolerance": 0.0}
        )
    for cat, length in raw.length_by_category.items():
        assert math.isclose(
            simp.length_by_category[cat], length, rel_tol=1e-3, abs_tol=1e-3
        )


def test_invalid_method_raises(simple_raster):
    finder = _build_finder(simple_raster)
    with pytest.raises(ValueError, match="Unknown simplification method"):
        finder.find_route(simplify={"method": "bogus", "tolerance": 1.0})


def test_missing_tolerance_raises(simple_raster):
    finder = _build_finder(simple_raster)
    with pytest.raises(KeyError, match="tolerance"):
        finder.find_route(simplify={"method": "douglas_peucker"})


@pytest.mark.parametrize(
    "method,tolerance",
    [
        ("douglas_peucker", 0.5),
        ("douglas_peucker_topology", 0.5),
        ("visvalingam", 1.0),
        ("grid", 1.0),
    ],
)
def test_all_methods_produce_consistent_metrics(simple_raster, method,
                                                tolerance):
    finder = _build_finder(simple_raster)
    with pytest.warns(UserWarning):
        path = finder.find_route(
            simplify={"method": method, "tolerance": tolerance}
        )
    # Endpoint behaviour: DP/Visvalingam keep endpoints exactly; the grid
    # method snaps endpoints to the grid (so they shift by ≤ tolerance).
    orig_start = path.original_path_geometry.coords[0]
    orig_end = path.original_path_geometry.coords[-1]
    new_start = path.path_geometry.coords[0]
    new_end = path.path_geometry.coords[-1]
    if method == "grid":
        assert math.hypot(new_start[0] - orig_start[0],
                          new_start[1] - orig_start[1]) <= tolerance
        assert math.hypot(new_end[0] - orig_end[0],
                          new_end[1] - orig_end[1]) <= tolerance
    else:
        assert new_start == orig_start
        assert new_end == orig_end
    # Per-category lengths sum to total.
    assert math.isclose(
        sum(path.length_by_category.values()),
        path.total_length,
        rel_tol=1e-9,
    )
    # Non-negative.
    for v in path.length_by_category.values():
        assert v >= 0
