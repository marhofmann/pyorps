"""Phase 2 windowing equivalence tests (plan items 2.1 and 2.3).

The contract these tests defend is exactness, not speed:

* item 2.1 — the handler must never write the buffer sentinel back through
  into the dataset it was handed, while the cells it exposes stay
  bit-identical to the historical view-plus-write-through result;
* item 2.3 — reading only the search window from a GeoTIFF must return
  exactly the cells a full read followed by slicing would, at the raster's
  edges and for windows clipped by the raster boundary as well.

Every "new way" result is compared against a verbatim re-implementation of
the pre-Phase-2 code path (:func:`legacy_window_data`), so a regression in
either direction fails here.
"""

import numpy as np
import pytest
from rasterio import open as rio_open
from rasterio.features import rasterize
from rasterio.transform import from_origin, rowcol
from rasterio.windows import Window
from rasterio.windows import transform as transform_window
from shapely.geometry import LineString, MultiPoint

from pyorps.io.geo_dataset import InMemoryRasterDataset, LocalRasterDataset
from pyorps.raster.handler import RasterHandler

CRS = "EPSG:32632"
RES = 1.0
ORIGIN_X = 500000.0
ORIGIN_Y = 5600000.0
HEIGHT = 200
WIDTH = 160


def make_transform():
    return from_origin(ORIGIN_X, ORIGIN_Y, RES, RES)


def make_array(bands: int = 1) -> np.ndarray:
    """Deterministic uint16 raster with a forbidden stripe and a 0-cost patch.

    Deliberately NOT square (200 x 160): a transposed row/col index anywhere in
    the windowing code then raises instead of silently reading the wrong cell.
    """
    rng = np.random.default_rng(20260807)
    data = rng.integers(1, 250, size=(bands, HEIGHT, WIDTH), dtype=np.uint16)
    data[:, 90:95, :] = 65535          # a wall
    data[:, 90:95, 70:76] = 7          # with a gap
    data[:, 10:14, 10:14] = 0          # 0-cost patch
    return data


SQUARE = 128


def make_square_array() -> np.ndarray:
    """Square raster for the one test that exercises estimate_buffer_width().

    That estimator indexes the raster with (col, row) swapped, so it only
    survives on a square raster — see the Phase 2a report. Nothing here
    depends on the estimator's value, only on it being reproducible.
    """
    rng = np.random.default_rng(99)
    data = rng.integers(1, 250, size=(1, SQUARE, SQUARE), dtype=np.uint16)
    data[:, 40:44, 20:100] = 65535
    return data


def write_tiff(path, data, transform=None, nodata=None):
    transform = make_transform() if transform is None else transform
    with rio_open(path, "w", driver="GTiff", height=data.shape[1],
                  width=data.shape[2], count=data.shape[0], dtype=data.dtype,
                  crs=CRS, transform=transform, nodata=nodata) as dst:
        dst.write(data)
    return str(path)


# ---------------------------------------------------------------------------
# Verbatim re-implementation of the pre-Phase-2 handler data path
# ---------------------------------------------------------------------------

def legacy_buffer_geometry(source_coords, target_coords, buffer_m):
    """Buffer polygon exactly as RasterHandler builds it."""
    single_src = isinstance(source_coords, tuple)
    single_tgt = isinstance(target_coords, tuple)
    if single_src and single_tgt:
        geom = LineString([source_coords, target_coords])
    else:
        points = []
        points.append(source_coords) if single_src else points.extend(source_coords)
        points.append(target_coords) if single_tgt else points.extend(target_coords)
        geom = MultiPoint(points).convex_hull
    return geom.buffer(distance=buffer_m, quad_segs=32)


def legacy_window(buffer_geometry, transform, shape):
    """The window arithmetic as it stood before the refactor."""
    bounds = buffer_geometry.bounds
    min_row, min_col = rowcol(transform, bounds[0], bounds[3])
    max_row, max_col = rowcol(transform, bounds[2], bounds[1])
    min_row = max(0, min_row)
    min_col = max(0, min_col)
    max_row = min(shape[0], max_row)
    max_col = min(shape[1], max_col)
    return Window(min_col, min_row, max_col - min_col, max_row - min_row)


def legacy_window_data(array, transform, source_coords, target_coords, buffer_m,
                       apply_mask=True, outside_value=None):
    """Old behaviour: slice a VIEW, then write the sentinel through it.

    Returns (data, window, window_transform, parent_array_after_masking).
    The parent is returned so a test can show what the old code did to the
    dataset it was given.
    """
    parent = array.copy()
    shape = parent.shape[-2:]
    buffer_geometry = legacy_buffer_geometry(source_coords, target_coords, buffer_m)
    window = legacy_window(buffer_geometry, transform, shape)
    wt = transform_window(window, transform)

    r0, c0 = int(window.row_off), int(window.col_off)
    r1, c1 = r0 + int(window.height), c0 + int(window.width)
    if parent.ndim == 3:
        data = parent[:, r0:r1, c0:c1]
    else:
        data = np.expand_dims(parent[r0:r1, c0:c1], axis=0)

    if apply_mask:
        if outside_value is None:
            outside_value = np.iinfo(data.dtype).max
        mask = rasterize([(buffer_geometry, 1)],
                         out_shape=(int(window.height), int(window.width)),
                         transform=wt, fill=0, dtype=np.uint8)
        for b in range(data.shape[0]):
            data[b][mask == 0] = outside_value
    return data, window, wt, parent


# ---------------------------------------------------------------------------
# Item 2.1 — copy-based masking
# ---------------------------------------------------------------------------

SRC = (ORIGIN_X + 30.5, ORIGIN_Y - 60.5)
TGT = (ORIGIN_X + 120.5, ORIGIN_Y - 170.5)
BUF = 15.0


@pytest.mark.parametrize("bands", [1, 3])
def test_window_contents_bit_identical_to_legacy(bands):
    """New handler.data == old view-and-write-through data, cell for cell."""
    array = make_array(bands)
    transform = make_transform()

    expected, window, wt, _ = legacy_window_data(array, transform, SRC, TGT, BUF)

    ds = InMemoryRasterDataset(array.copy(), CRS, transform)
    handler = RasterHandler(ds, SRC, TGT, search_space_buffer_m=BUF)

    assert (handler.window.col_off, handler.window.row_off,
            handler.window.width, handler.window.height) == (
        window.col_off, window.row_off, window.width, window.height)
    assert handler.window_transform == wt
    assert handler.data.dtype == expected.dtype
    assert handler.data.shape == expected.shape
    np.testing.assert_array_equal(handler.data, expected)

    # The mask really did something, otherwise the test proves nothing: more
    # cells carry the sentinel after masking than the raster itself contains.
    unmasked, _, _, _ = legacy_window_data(array, transform, SRC, TGT, BUF,
                                           apply_mask=False)
    sentinel = np.iinfo(handler.data.dtype).max
    assert (int((handler.data == sentinel).sum())
            > int((unmasked == sentinel).sum()) > 0)


@pytest.mark.parametrize("bands", [1, 3])
def test_source_dataset_unchanged_after_handler_construction(bands):
    """Item 2.1: building a handler must not mutate its input dataset."""
    array = make_array(bands)
    pristine = array.copy()
    ds = InMemoryRasterDataset(array, CRS, make_transform())

    RasterHandler(ds, SRC, TGT, search_space_buffer_m=BUF)

    np.testing.assert_array_equal(ds.data, pristine)
    np.testing.assert_array_equal(array, pristine)


def test_source_dataset_unchanged_for_2d_input():
    """The 2D branch goes through expand_dims, which is also a view."""
    array = make_array(1)[0]
    pristine = array.copy()
    ds = InMemoryRasterDataset(array, CRS, make_transform())

    handler = RasterHandler(ds, SRC, TGT, search_space_buffer_m=BUF)

    assert handler.data.ndim == 3
    np.testing.assert_array_equal(ds.data, pristine)


def test_handler_data_owns_its_memory():
    ds = InMemoryRasterDataset(make_array(1), CRS, make_transform())
    handler = RasterHandler(ds, SRC, TGT, search_space_buffer_m=BUF)
    assert handler.data.flags.owndata
    assert not np.shares_memory(handler.data, ds.data)


def test_second_handler_sees_a_pristine_dataset():
    """The bug item 2.1 fixes: handler #2 inherited handler #1's buffer."""
    array = make_array(1)
    transform = make_transform()
    ds = InMemoryRasterDataset(array, CRS, transform)

    RasterHandler(ds, SRC, TGT, search_space_buffer_m=BUF)

    other_src = (ORIGIN_X + 20.5, ORIGIN_Y - 150.5)
    other_tgt = (ORIGIN_X + 140.5, ORIGIN_Y - 30.5)
    second = RasterHandler(ds, other_src, other_tgt, search_space_buffer_m=BUF)

    expected, _, _, _ = legacy_window_data(make_array(1), transform,
                                           other_src, other_tgt, BUF)
    np.testing.assert_array_equal(second.data, expected)


def test_masking_writes_only_into_handler_memory():
    """A later apply_geometry_mask() call must also stay local."""
    array = make_array(1)
    pristine = array.copy()
    ds = InMemoryRasterDataset(array, CRS, make_transform())
    handler = RasterHandler(ds, SRC, TGT, search_space_buffer_m=BUF,
                            apply_mask=False, copy_window=False)

    # copy_window=False deliberately keeps the historical view ...
    assert np.shares_memory(handler.data, ds.data)
    # ... but masking detaches before writing.
    handler.apply_geometry_mask(handler.buffer_geometry)

    np.testing.assert_array_equal(ds.data, pristine)
    assert handler.data.flags.owndata


def test_copy_window_false_with_masking_is_rejected():
    ds = InMemoryRasterDataset(make_array(1), CRS, make_transform())
    with pytest.raises(ValueError, match="copy_window"):
        RasterHandler(ds, SRC, TGT, search_space_buffer_m=BUF,
                      apply_mask=True, copy_window=False)


def test_unmasked_handler_matches_legacy_slice():
    """apply_mask=False (the DEM/DSM path) must be unchanged as well."""
    array = make_array(1)
    transform = make_transform()
    expected, _, _, _ = legacy_window_data(array, transform, SRC, TGT, BUF,
                                           apply_mask=False)
    ds = InMemoryRasterDataset(array.copy(), CRS, transform)
    handler = RasterHandler(ds, SRC, TGT, search_space_buffer_m=BUF,
                            apply_mask=False)
    np.testing.assert_array_equal(handler.data, expected)


def test_float_outside_value_still_applies():
    """The float32 lossless path uses +inf as the outside sentinel."""
    array = make_array(1).astype(np.float32)
    ds = InMemoryRasterDataset(array.copy(), CRS, make_transform())
    handler = RasterHandler(ds, SRC, TGT, search_space_buffer_m=BUF,
                            outside_value=np.float32(np.inf))
    assert np.isinf(handler.data).any()
    assert not np.isinf(ds.data).any()


# ---------------------------------------------------------------------------
# Item 2.3 — windowed file reads
# ---------------------------------------------------------------------------

WINDOWS = [
    Window(0, 0, WIDTH, HEIGHT),        # everything
    Window(0, 0, 1, 1),                 # top-left corner
    Window(WIDTH - 1, HEIGHT - 1, 1, 1),  # bottom-right corner
    Window(0, 0, WIDTH, 3),             # full-width strip at the top edge
    Window(WIDTH - 5, 0, 5, HEIGHT),    # full-height strip at the right edge
    Window(37, 91, 43, 17),             # interior, crosses the forbidden wall
    Window(0, HEIGHT - 4, 9, 4),        # bottom-left corner block
]


@pytest.mark.parametrize("bands", [1, 3])
@pytest.mark.parametrize("window", WINDOWS)
def test_read_window_equals_full_read_then_slice(tmp_path, bands, window):
    array = make_array(bands)
    path = write_tiff(tmp_path / f"r{bands}.tif", array)

    full = LocalRasterDataset(path)
    full.load_data()

    ds = LocalRasterDataset(path)
    ds.load_metadata()
    windowed = ds.read_window(window)

    r0, c0 = int(window.row_off), int(window.col_off)
    r1, c1 = r0 + int(window.height), c0 + int(window.width)
    expected = full.data[:, r0:r1, c0:c1]

    assert windowed.dtype == expected.dtype
    assert windowed.shape == expected.shape
    np.testing.assert_array_equal(windowed, expected)
    # read_window is a pure read: it must not populate .data
    assert ds.data is None


def test_read_window_degenerate_window(tmp_path):
    """A zero-sized window mirrors numpy's empty slice, it does not raise."""
    path = write_tiff(tmp_path / "r.tif", make_array(1))
    ds = LocalRasterDataset(path)
    ds.load_metadata()
    out = ds.read_window(Window(5, 5, 0, 0))
    assert out.shape == (1, 0, 0)
    assert out.dtype == np.uint16


def test_load_metadata_matches_full_load(tmp_path):
    array = make_array(2)
    path = write_tiff(tmp_path / "r.tif", array, nodata=65535)

    full = LocalRasterDataset(path)
    full.load_data()

    meta = LocalRasterDataset(path)
    meta.load_metadata()

    assert meta.data is None
    assert meta.shape == full.shape == (HEIGHT, WIDTH)
    assert meta.transform == full.transform
    assert meta.crs == full.crs
    assert meta.count == full.count == 2
    assert meta.dtype == full.dtype == np.dtype(np.uint16)
    assert meta.nodata == 65535


def test_full_load_data_is_unchanged(tmp_path):
    """The explicit full-load path keeps meaning "the whole raster"."""
    array = make_array(1)
    path = write_tiff(tmp_path / "r.tif", array)

    ds = LocalRasterDataset(path)
    ds.load_data()

    np.testing.assert_array_equal(ds.data, array)
    assert ds.shape == (HEIGHT, WIDTH)
    assert ds.transform == make_transform()
    assert ds.window is None
    assert ds.file_shape == (HEIGHT, WIDTH)


def test_load_data_with_window_is_self_consistent(tmp_path):
    """data / transform / shape must describe the same thing after windowing."""
    array = make_array(1)
    path = write_tiff(tmp_path / "r.tif", array)
    window = Window(37, 91, 43, 17)

    ds = LocalRasterDataset(path)
    ds.load_data(window=window)

    np.testing.assert_array_equal(ds.data, array[:, 91:108, 37:80])
    assert ds.shape == (17, 43)
    assert ds.count == 1
    assert ds.window == window
    assert ds.file_shape == (HEIGHT, WIDTH)
    assert ds.transform == transform_window(window, make_transform())
    # transform still maps data[0, 0]: its origin is that pixel's corner.
    assert ds.transform.c == pytest.approx(ORIGIN_X + 37 * RES)
    assert ds.transform.f == pytest.approx(ORIGIN_Y - 91 * RES)


# ---------------------------------------------------------------------------
# Item 2.3 through the handler: window-read == full-read
# ---------------------------------------------------------------------------

HANDLER_CASES = [
    # interior corridor
    (SRC, TGT, BUF),
    # corridor pushed against the top-left corner: window clips at 0
    ((ORIGIN_X + 3.5, ORIGIN_Y - 2.5), (ORIGIN_X + 40.5, ORIGIN_Y - 45.5), 12.0),
    # corridor pushed against the bottom-right corner: window clips at shape
    ((ORIGIN_X + 130.5, ORIGIN_Y - 160.5),
     (ORIGIN_X + 158.5, ORIGIN_Y - 198.5), 12.0),
    # buffer wider than the raster: window is the whole file
    (SRC, TGT, 900.0),
    # multi-point endpoints -> convex-hull buffer
    ([(ORIGIN_X + 20.5, ORIGIN_Y - 30.5), (ORIGIN_X + 25.5, ORIGIN_Y - 60.5)],
     [(ORIGIN_X + 110.5, ORIGIN_Y - 150.5),
      (ORIGIN_X + 130.5, ORIGIN_Y - 120.5)], 18.0),
]


@pytest.mark.parametrize("source_coords,target_coords,buffer_m", HANDLER_CASES)
@pytest.mark.parametrize("apply_mask", [True, False])
def test_windowed_handler_matches_full_load_handler(
        tmp_path, source_coords, target_coords, buffer_m, apply_mask):
    array = make_array(1)
    path = write_tiff(tmp_path / "r.tif", array)

    # Old way: load the whole file, then window by slicing.
    full_ds = LocalRasterDataset(path)
    full_ds.load_data()
    full_handler = RasterHandler(full_ds, source_coords, target_coords,
                                 search_space_buffer_m=buffer_m,
                                 apply_mask=apply_mask)

    # New way: header first, read only the window.
    win_ds = LocalRasterDataset(path)
    win_handler = RasterHandler(win_ds, source_coords, target_coords,
                                search_space_buffer_m=buffer_m,
                                apply_mask=apply_mask)

    assert win_handler.windowed_source_read is True
    assert full_handler.windowed_source_read is False
    assert win_handler.window == full_handler.window
    assert win_handler.window_transform == full_handler.window_transform
    assert win_handler.data.dtype == full_handler.data.dtype
    assert win_handler.data.shape == full_handler.data.shape
    np.testing.assert_array_equal(win_handler.data, full_handler.data)

    # dataset.data still means "the whole raster" for the full-load caller,
    # and stays unread for the windowed one.
    np.testing.assert_array_equal(full_ds.data, array)
    assert win_ds.data is None
    assert win_ds.shape == (HEIGHT, WIDTH)


def test_windowed_handler_matches_legacy_path(tmp_path):
    """Window-read handler == the verbatim pre-Phase-2 result."""
    array = make_array(1)
    path = write_tiff(tmp_path / "r.tif", array)
    expected, window, wt, _ = legacy_window_data(array, make_transform(),
                                                 SRC, TGT, BUF)

    ds = LocalRasterDataset(path)
    handler = RasterHandler(ds, SRC, TGT, search_space_buffer_m=BUF)

    assert handler.window == window
    assert handler.window_transform == wt
    np.testing.assert_array_equal(handler.data, expected)


def test_windowed_read_reads_fewer_cells_than_the_file(tmp_path):
    """Guard the point of the exercise: the window really is smaller."""
    path = write_tiff(tmp_path / "r.tif", make_array(1))
    ds = LocalRasterDataset(path)
    handler = RasterHandler(ds, SRC, TGT, search_space_buffer_m=BUF)
    assert handler.data[0].size < HEIGHT * WIDTH


def test_windowed_read_opt_out(tmp_path):
    """windowed_read=False keeps the caller-loads-everything contract."""
    path = write_tiff(tmp_path / "r.tif", make_array(1))
    ds = LocalRasterDataset(path)
    ds.load_data()
    handler = RasterHandler(ds, SRC, TGT, search_space_buffer_m=BUF,
                            windowed_read=False)
    assert handler.windowed_source_read is False
    assert ds.data is not None


def test_windowed_read_without_buffer_falls_back_to_full_load(tmp_path):
    """The buffer estimator needs pixels, so no buffer means no window."""
    array = make_square_array()
    path = write_tiff(tmp_path / "square.tif", array)
    src = (ORIGIN_X + 10.5, ORIGIN_Y - 20.5)
    tgt = (ORIGIN_X + 100.5, ORIGIN_Y - 90.5)

    ds = LocalRasterDataset(path)
    with pytest.warns(UserWarning, match="search_space_buffer_m"):
        handler = RasterHandler(ds, src, tgt)

    assert handler.windowed_source_read is False
    np.testing.assert_array_equal(ds.data, array)

    # And the fallback lands on exactly the historical result.
    reference = LocalRasterDataset(path)
    reference.load_data()
    ref_handler = RasterHandler(reference, src, tgt, windowed_read=False)
    assert handler.search_space_buffer_m == ref_handler.search_space_buffer_m
    assert handler.window == ref_handler.window
    np.testing.assert_array_equal(handler.data, ref_handler.data)


def test_windowed_handler_leaves_no_mask_behind(tmp_path):
    """Items 2.1 and 2.3 together: nothing the handler does is visible in the
    dataset, so a second handler on the same file is unaffected."""
    array = make_array(1)
    path = write_tiff(tmp_path / "r.tif", array)

    ds = LocalRasterDataset(path)
    RasterHandler(ds, SRC, TGT, search_space_buffer_m=BUF)
    assert ds.data is None

    second = RasterHandler(ds, SRC, TGT, search_space_buffer_m=BUF)
    expected, _, _, _ = legacy_window_data(array, make_transform(),
                                           SRC, TGT, BUF)
    np.testing.assert_array_equal(second.data, expected)


def test_coords_round_trip_through_a_windowed_handler(tmp_path):
    """Window offsets stay relative to the FILE, not to the window."""
    path = write_tiff(tmp_path / "r.tif", make_array(1))

    full_ds = LocalRasterDataset(path)
    full_ds.load_data()
    full_handler = RasterHandler(full_ds, SRC, TGT, search_space_buffer_m=BUF)

    win_ds = LocalRasterDataset(path)
    win_handler = RasterHandler(win_ds, SRC, TGT, search_space_buffer_m=BUF)

    probes = [SRC, TGT, (ORIGIN_X + 60.5, ORIGIN_Y - 100.5)]
    np.testing.assert_array_equal(win_handler.coords_to_indices(probes),
                                  full_handler.coords_to_indices(probes))
    idx = full_handler.coords_to_indices(probes)
    np.testing.assert_allclose(win_handler.indices_to_coords(idx),
                               full_handler.indices_to_coords(idx))


def test_window_from_bounds_matches_legacy_arithmetic():
    transform = make_transform()
    shape = (HEIGHT, WIDTH)
    for source_coords, target_coords, buffer_m in HANDLER_CASES:
        geom = legacy_buffer_geometry(source_coords, target_coords, buffer_m)
        assert RasterHandler.window_from_bounds(geom.bounds, transform,
                                                shape) == legacy_window(
            geom, transform, shape)


def test_covering_window_covers_the_bounds_it_was_given():
    """Floor the near corner, ceil the far one -- cover, never clip.

    The pitfall it replaces is ``rasterio.windows.from_bounds``, whose
    fractional offsets put ``window_transform``'s origin half a pixel off
    the raster grid, so a floor-indexed lookup reads the neighbouring
    cell. Anisotropic pixels on purpose: the two sizes must not be mixed
    up between the row and column axes.
    """
    from affine import Affine

    sx, sy = 2.0, 3.0
    transform = Affine(sx, 0.0, ORIGIN_X, 0.0, -sy, ORIGIN_Y)
    bounds = (ORIGIN_X + 10.5, ORIGIN_Y - 40.5, ORIGIN_X + 33.5,
              ORIGIN_Y - 12.5)
    win, win_transform = RasterHandler.covering_window(bounds, transform)

    # the window's own transform sits exactly on the raster grid
    assert (win_transform.c - transform.c) % sx == pytest.approx(0.0)
    assert (transform.f - win_transform.f) % sy == pytest.approx(0.0)
    assert (win_transform.a, win_transform.e) == (transform.a, transform.e)

    # and it covers every requested corner, with room to spare from ceil
    left, top = win_transform.c, win_transform.f
    right = left + win.width * sx
    bottom = top - win.height * sy
    assert left <= bounds[0] and bottom <= bounds[1]
    assert right >= bounds[2] and top >= bounds[3]

    padded, _ = RasterHandler.covering_window(bounds, transform, pad=10.0)
    assert padded.width > win.width and padded.height > win.height

    # shape clips; omitting it leaves the window for a boundless read
    off_grid = (ORIGIN_X - 50.0, ORIGIN_Y - 40.5, ORIGIN_X + 33.5,
                ORIGIN_Y + 50.0)
    loose, _ = RasterHandler.covering_window(off_grid, transform)
    clipped, _ = RasterHandler.covering_window(off_grid, transform,
                                               shape=(HEIGHT, WIDTH))
    assert loose.col_off < 0 and loose.row_off < 0
    assert (clipped.col_off, clipped.row_off) == (0, 0)
