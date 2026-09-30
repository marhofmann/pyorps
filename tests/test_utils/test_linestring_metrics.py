"""Tests for the DDA-based linestring metric kernel."""
import math
import numpy as np
import pytest

from pyorps.utils._traversal_numba import calculate_linestring_metrics_numba


# Helper: convert (x, y) -> fractional (row, col) for a unit raster whose
# top-left corner sits at (0, 0), pixel size = 1, north-up.
#   row_frac = (origin_y - y) / pixel_h
#   col_frac = (x - origin_x) / pixel_w
# With origin_y=raster_rows, origin_x=0, pixel_w=pixel_h=1 we get:
#   row_frac = rows - y
#   col_frac = x


def _to_rowcol(coords_xy, rows):
    arr = np.asarray(coords_xy, dtype=np.float64)
    rc = np.empty_like(arr)
    rc[:, 0] = rows - arr[:, 1]  # row_frac
    rc[:, 1] = arr[:, 0]         # col_frac
    return np.ascontiguousarray(rc, dtype=np.float64)


def test_single_horizontal_segment_crossing_three_cells():
    raster = np.full((3, 3), 7, dtype=np.uint16)
    rows = 3
    # Line from (x=0.0, y=0.5) to (x=3.0, y=0.5) - bottom row of raster.
    coords_rc = _to_rowcol([(0.0, 0.5), (3.0, 0.5)], rows=rows)
    total, cats, lens = calculate_linestring_metrics_numba(raster, coords_rc)
    assert math.isclose(total, 3.0, abs_tol=1e-9)
    assert list(cats) == [7]
    assert math.isclose(lens[0], 3.0, abs_tol=1e-9)


def test_single_diagonal_segment_exact_sqrt2_per_cell():
    raster = np.full((3, 3), 4, dtype=np.uint16)
    coords_rc = _to_rowcol([(0.0, 0.0), (3.0, 3.0)], rows=3)
    total, cats, lens = calculate_linestring_metrics_numba(raster, coords_rc)
    assert math.isclose(total, 3.0 * math.sqrt(2.0), abs_tol=1e-9)
    assert math.isclose(lens[0], 3.0 * math.sqrt(2.0), abs_tol=1e-9)


def test_horizontal_segment_two_categories_split():
    raster = np.array([[10, 10, 20, 20]], dtype=np.uint16)
    coords_rc = _to_rowcol([(0.0, 0.5), (4.0, 0.5)], rows=1)
    total, cats, lens = calculate_linestring_metrics_numba(raster, coords_rc)
    assert math.isclose(total, 4.0, abs_tol=1e-9)
    cat_to_len = dict(zip(cats.tolist(), lens.tolist()))
    assert math.isclose(cat_to_len[10], 2.0, abs_tol=1e-9)
    assert math.isclose(cat_to_len[20], 2.0, abs_tol=1e-9)


def test_segment_entirely_inside_one_cell():
    raster = np.full((3, 3), 5, dtype=np.uint16)
    # From (0.2, 0.3) to (0.8, 0.6): both inside cell (row=2, col=0) in
    # row-col space (because of the row flip).
    coords_rc = _to_rowcol([(0.2, 0.3), (0.8, 0.6)], rows=3)
    seg_len = math.hypot(0.6, 0.3)
    total, cats, lens = calculate_linestring_metrics_numba(raster, coords_rc)
    assert math.isclose(total, seg_len, abs_tol=1e-9)
    assert math.isclose(lens[0], seg_len, abs_tol=1e-9)


def test_two_segment_polyline_lengths_sum():
    raster = np.full((4, 4), 1, dtype=np.uint16)
    coords_rc = _to_rowcol([(0.0, 0.0), (4.0, 0.0), (4.0, 4.0)], rows=4)
    total, cats, lens = calculate_linestring_metrics_numba(raster, coords_rc)
    assert math.isclose(total, 8.0, abs_tol=1e-9)
    assert math.isclose(lens[0], 8.0, abs_tol=1e-9)


def test_sum_of_lengths_equals_total_length_random():
    rng = np.random.default_rng(seed=42)
    raster = rng.integers(1, 5, size=(20, 20), dtype=np.uint16)
    pts = [(0.5, 0.5), (10.7, 4.3), (15.2, 12.8), (19.5, 19.5)]
    coords_rc = _to_rowcol(pts, rows=20)
    total, cats, lens = calculate_linestring_metrics_numba(raster, coords_rc)
    assert math.isclose(total, float(lens.sum()), rel_tol=1e-9)


def test_segment_clipped_at_raster_boundary():
    raster = np.full((2, 2), 9, dtype=np.uint16)
    # From (x=-0.5, y=0.5) to (x=2.5, y=0.5): horizontal, 1.0 unit outside on
    # each side. Only x in [0,2] (length 2.0) is inside.
    coords_rc = _to_rowcol([(-0.5, 0.5), (2.5, 0.5)], rows=2)
    total, cats, lens = calculate_linestring_metrics_numba(raster, coords_rc)
    assert math.isclose(lens[0], 2.0, abs_tol=1e-9)


def test_dda_matches_existing_kernel_for_8connected_path():
    """For an 8-connected centre-to-centre path the new DDA kernel must
    agree with the legacy ``calculate_path_metrics_numba`` total length to
    within floating-point noise."""
    from pyorps.utils._traversal_numba import calculate_path_metrics_numba

    raster = np.array(
        [[1, 1, 2, 2, 3],
         [1, 2, 2, 3, 3],
         [4, 4, 2, 3, 3],
         [4, 4, 4, 3, 3]],
        dtype=np.uint16,
    )
    cells = [(0, 0), (1, 1), (2, 2), (3, 3)]
    rows, cols = raster.shape
    path_indices = np.array(
        [r * cols + c for (r, c) in cells], dtype=np.uint32
    )
    legacy_total, _, _ = calculate_path_metrics_numba(raster, path_indices)

    coords_rc = np.array(
        [[r + 0.5, c + 0.5] for (r, c) in cells], dtype=np.float64
    )
    coords_rc = np.ascontiguousarray(coords_rc, dtype=np.float64)
    dda_total, _, _ = calculate_linestring_metrics_numba(raster, coords_rc)

    assert math.isclose(legacy_total, dda_total, rel_tol=1e-9, abs_tol=1e-9)
