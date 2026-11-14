"""Regression tests for the input validation of GeoRasterizer and CostAssumptions.

Five audited defects (T4 audit, findings #3, #4, #17, #18 and #19), each with the
failing input and with the boundary values that must keep working:

* #3  a cost below 0 or above 65535 wrapped silently in the uint16 raster
* #4  a class missing from the cost table became forbidden without a word
* #17 ``resolution_in_m`` and a geographic CRS were not validated
* #18 the constructor stored ``bbox`` and ``default_crs`` without using them and
      failed on the documented input types
* #19 ``CostAssumptions({})`` leaked a ``StopIteration``; further top-level keys
      were dropped silently

Everything is built from tiny in-memory GeoDataFrames; the two path tests write
one small GeoJSON file into pytest's ``tmp_path``.
"""
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from geopandas.testing import assert_geodataframe_equal
from rasterio.transform import from_origin
from shapely.geometry import box

from pyorps.core.cost_assumptions import CostAssumptions
from pyorps.core.exceptions import FormatError, InvalidInputError
from pyorps.core.types import IMPASSABLE_CELL_COST
from pyorps.core.warning_types import CrsReprojectionWarning
from pyorps.io.geo_dataset import InMemoryRasterDataset, InMemoryVectorDataset
from pyorps.raster.rasterizer import GeoRasterizer

CRS = "EPSG:32632"
OTHER_PROJECTED_CRS = "EPSG:25832"
GEOGRAPHIC_CRS = "EPSG:4326"
#: Edge length of one feature square in CRS units (metres).
SIZE = 4.0
MISSING = "missing from the cost table"


def _gdf(categories, crs=CRS, **columns):
    """One 4 m x 4 m square per category, side by side along x."""
    geometry = [box(i * SIZE, 0.0, (i + 1) * SIZE, SIZE)
                for i in range(len(categories))]
    return gpd.GeoDataFrame({"category": list(categories), **columns},
                            geometry=geometry, crs=crs)


def _rasterizer(categories, costs, crs=CRS, **columns):
    dataset = InMemoryVectorDataset(_gdf(categories, crs, **columns), crs=crs)
    return GeoRasterizer(dataset, costs)


def _burn(categories, costs, method="rasterize", **kwargs):
    """Rasterize and return ``(rasterizer, pyorps user warnings)``.

    The thin-feature audit is switched off: it reports on every forbidden
    feature and is not what these tests are about.
    """
    rasterizer = _rasterizer(categories, costs)
    kwargs.setdefault("on_thin_features", "ignore")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        getattr(rasterizer, method)(**kwargs)
    return rasterizer, [w for w in caught if issubclass(w.category, UserWarning)]


def _values(raster):
    return sorted(np.unique(raster).tolist())


# ---------------------------------------------------------------------------
# #3  cost values that do not fit the raster dtype
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cost, shown", [
    (-5, "-5"),
    (-1, "-1"),
    (65536, "65536"),
    (70000, "70000"),
    (65535.5, "65535.5"),       # rounds to 65536, which used to wrap to 0
    (float("inf"), "inf"),
    (float("-inf"), "-inf"),
])
def test_cost_outside_uint16_is_rejected_and_named(cost, shown):
    rasterizer = _rasterizer(["a", "b"], {"category": {"a": cost, "b": 2}})
    with pytest.raises(FormatError) as excinfo:
        rasterizer.rasterize()
    message = str(excinfo.value)
    assert f"'a': {shown}" in message
    assert "'b'" not in message
    assert "uint16" in message
    assert "0 to 65535" in message
    assert rasterizer.raster is None


def test_out_of_range_error_names_only_the_first_classes():
    categories = [f"c{i}" for i in range(8)]
    costs = {"category": {name: 70000 + i for i, name in enumerate(categories)}}
    with pytest.raises(FormatError) as excinfo:
        _rasterizer(categories, costs).rasterize()
    message = str(excinfo.value)
    assert "8 class(es)" in message
    assert "'c0': 70000" in message and "'c4': 70004" in message
    assert "'c5'" not in message
    assert "and 3 more" in message


def test_uint16_boundary_costs_still_work():
    costs = {"category": {"a": 0, "b": 65534, "c": IMPASSABLE_CELL_COST}}
    rasterizer, caught = _burn(["a", "b", "c"], costs)
    assert rasterizer.raster.dtype == np.uint16
    expected = np.repeat(np.array([[0, 65534, 65535]], dtype=np.uint16), 4, axis=1)
    np.testing.assert_array_equal(rasterizer.raster, np.repeat(expected, 4, axis=0))
    assert caught == []


def test_costs_that_round_into_range_still_work():
    """Values are burned rounded, so the check is on the rounded value."""
    costs = {"category": {"a": -0.4, "b": 65535.4, "c": 1.5}}
    rasterizer, _ = _burn(["a", "b", "c"], costs)
    assert _values(rasterizer.raster) == [0, 2, 65535]


def test_valid_costs_burn_the_same_raster_as_before():
    rasterizer, caught = _burn(["a", "b"], {"category": {"a": 1, "b": 2}})
    expected = np.array([[1] * 4 + [2] * 4] * 4, dtype=np.uint16)
    np.testing.assert_array_equal(rasterizer.raster, expected)
    assert caught == []


def test_float_raster_accepts_negative_and_infinite_costs():
    """No false alarm: a float raster holds any sign, and inf."""
    costs = {"category": {"a": -5.5, "b": float("inf"), "c": 70000.0}}
    rasterizer, _ = _burn(["a", "b", "c"], costs, dtype="float32")
    assert rasterizer.raster.dtype == np.float32
    assert _values(rasterizer.raster) == [-6.0, 70000.0, float("inf")]


def test_float_raster_rejects_a_value_beyond_the_float_range():
    costs = {"category": {"a": 1e39, "b": 2}}
    with pytest.raises(FormatError, match="float32"):
        _rasterizer(["a", "b"], costs).rasterize(dtype="float32")


def test_range_follows_the_integer_dtype_in_use():
    with pytest.raises(FormatError) as excinfo:
        _rasterizer(["a", "b"], {"category": {"a": 300, "b": 2}}).rasterize(
            dtype="uint8", fill_value=255)
    assert "'a': 300" in str(excinfo.value)
    assert "0 to 255" in str(excinfo.value)

    rasterizer, _ = _burn(["a", "b"], {"category": {"a": -5, "b": 2}},
                          dtype="int16", fill_value=0)
    assert _values(rasterizer.raster) == [-5, 2]


def test_custom_field_outside_the_dtype_is_rejected_by_row():
    rasterizer = _rasterizer(["a", "b"], {"category": {"a": 1, "b": 2}},
                             height=[10, -3])
    with pytest.raises(FormatError) as excinfo:
        rasterizer.rasterize(field_name="height")
    message = str(excinfo.value)
    assert "Field 'height'" in message
    assert "1: -3" in message          # row label 1, value -3


@pytest.mark.parametrize("method", ["rasterize", "rasterize_metrics"])
@pytest.mark.parametrize("empty", [float("nan"), None])
def test_cost_table_entry_without_a_number_is_rejected(method, empty):
    """NaN is not a documented way to forbid a class; 65535 is."""
    rasterizer = _rasterizer(["a", "b"], {"category": {"a": empty, "b": 2}})
    with pytest.raises(FormatError) as excinfo:
        getattr(rasterizer, method)()
    message = str(excinfo.value)
    assert "'a'" in message
    assert "'b'" not in message
    assert "65535" in message


def test_nested_cost_table_entry_without_a_number_is_rejected():
    costs = {("category", "kind"): {"a": {"x": 1, "y": float("nan")}}}
    rasterizer = _rasterizer(["a", "a"], costs, kind=["x", "y"])
    with pytest.raises(FormatError, match=r"\('a', 'y'\)"):
        rasterizer.rasterize()


# ---------------------------------------------------------------------------
# #4  classes missing from the cost table
# ---------------------------------------------------------------------------

def test_missing_class_warns_once_and_stays_forbidden():
    rasterizer, caught = _burn(["a", "b"], {"category": {"a": 1, "zzz": 2}})
    missing = [w for w in caught if MISSING in str(w.message)]
    assert len(missing) == 1
    message = str(missing[0].message)
    assert "1 class(es) of 'category'" in message
    assert "'b'" in message
    assert "'a'" not in message
    assert "forbidden" in message
    assert missing[0].category is UserWarning
    # The result itself is unchanged: the missing class is forbidden.
    expected = np.array([[1] * 4 + [IMPASSABLE_CELL_COST] * 4] * 4,
                        dtype=np.uint16)
    np.testing.assert_array_equal(rasterizer.raster, expected)


def test_missing_class_warning_points_at_the_caller():
    rasterizer = _rasterizer(["a", "b"], {"category": {"a": 1}})
    with pytest.warns(UserWarning, match=MISSING) as record:
        rasterizer.rasterize(on_thin_features="ignore")
    assert Path(record[0].filename).name == Path(__file__).name


def test_missing_class_warning_truncates_a_long_list():
    categories = ["a"] + [f"m{i}" for i in range(8)]
    _, caught = _burn(categories, {"category": {"a": 1}})
    missing = [str(w.message) for w in caught if MISSING in str(w.message)]
    assert len(missing) == 1
    assert "8 class(es)" in missing[0]
    assert "(8 feature(s))" in missing[0]
    assert "'m0'" in missing[0] and "'m4'" in missing[0]
    assert "'m5'" not in missing[0]
    assert "and 3 more" in missing[0]


def test_missing_class_is_counted_once_for_many_features():
    _, caught = _burn(["a", "b", "b", "b"], {"category": {"a": 1}})
    missing = [str(w.message) for w in caught if MISSING in str(w.message)]
    assert len(missing) == 1
    assert "1 class(es)" in missing[0]
    assert "(3 feature(s))" in missing[0]


def test_missing_side_feature_combination_is_named():
    costs = {("category", "kind"): {"a": {"x": 1}}}
    rasterizer = _rasterizer(["a", "a"], costs, kind=["x", "y"])
    with pytest.warns(UserWarning, match=MISSING) as record:
        rasterizer.rasterize(on_thin_features="ignore")
    assert "('a', 'y')" in str(record[0].message)
    assert _values(rasterizer.raster) == [1, IMPASSABLE_CELL_COST]


def test_missing_class_warning_reports_another_fill_value():
    rasterizer, caught = _burn(["a", "b"], {"category": {"a": 1}}, fill_value=0)
    missing = [str(w.message) for w in caught if MISSING in str(w.message)]
    assert len(missing) == 1
    assert "received fill_value=0" in missing[0]
    assert _values(rasterizer.raster) == [0, 1]


def test_complete_cost_table_does_not_warn():
    """A class that is forbidden ON PURPOSE (65535) is not a missing class."""
    costs = {"category": {"a": 1, "b": IMPASSABLE_CELL_COST}}
    rasterizer, caught = _burn(["a", "b"], costs)
    assert caught == []
    assert _values(rasterizer.raster) == [1, IMPASSABLE_CELL_COST]


def test_missing_class_warns_once_in_rasterize_metrics():
    rasterizer, caught = _burn(["a", "b"], {"category": {"a": 1}},
                               method="rasterize_metrics")
    missing = [str(w.message) for w in caught if MISSING in str(w.message)]
    assert len(missing) == 1
    assert "'b'" in missing[0]
    # Unchanged result: the missing class is forbidden in the whole stack.
    expected = np.array([[False] * 4 + [True] * 4] * 4)
    np.testing.assert_array_equal(rasterizer.metric_stack.forbidden_mask, expected)


def test_custom_field_without_values_keeps_its_silent_fill():
    """Only the cost table has classes; a custom field is filled as before."""
    rasterizer = _rasterizer(["a", "b"], {"category": {"a": 1, "b": 2}},
                             height=[10.0, float("nan")])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rasterizer.rasterize(field_name="height", fill_value=7,
                             on_thin_features="ignore")
    assert not [w for w in caught if MISSING in str(w.message)]
    assert _values(rasterizer.raster) == [7, 10]


# ---------------------------------------------------------------------------
# #17  resolution_in_m and geographic CRS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["rasterize", "rasterize_metrics"])
@pytest.mark.parametrize("resolution", [0, 0.0, -2, -0.5, float("nan"),
                                        float("inf"), float("-inf")])
def test_resolution_must_be_finite_and_positive(method, resolution):
    rasterizer = _rasterizer(["a", "b"], {"category": {"a": 1, "b": 2}})
    with pytest.raises(ValueError, match="resolution_in_m") as excinfo:
        getattr(rasterizer, method)(resolution_in_m=resolution)
    assert "greater than 0" in str(excinfo.value)
    # Rejected before the cost column is written to the caller's data.
    assert "cost" not in rasterizer.base_data.columns


@pytest.mark.parametrize("method", ["rasterize", "rasterize_metrics"])
@pytest.mark.parametrize("resolution", ["1", "fine", None, [1.0]])
def test_resolution_must_be_a_number(method, resolution):
    rasterizer = _rasterizer(["a", "b"], {"category": {"a": 1, "b": 2}})
    with pytest.raises(TypeError, match="resolution_in_m must be a number"):
        getattr(rasterizer, method)(resolution_in_m=resolution)


@pytest.mark.parametrize("resolution, shape", [
    (0.5, (8, 16)),
    (1, (4, 8)),
    (1.0, (4, 8)),
    (np.float32(2.0), (2, 4)),
    (np.int64(4), (1, 2)),
])
def test_valid_resolutions_still_work(resolution, shape):
    rasterizer, caught = _burn(["a", "b"], {"category": {"a": 1, "b": 2}},
                               resolution_in_m=resolution)
    assert rasterizer.raster.shape == shape
    assert _values(rasterizer.raster) == [1, 2]
    assert caught == []


def test_shape_helper_rejects_a_bad_resolution():
    with pytest.raises(ValueError, match="resolution_in_m"):
        GeoRasterizer._get_rows_and_columns(8.0, 4.0, 0, 32.0)
    assert GeoRasterizer._get_rows_and_columns(8.0, 4.0, 0.5, 32.0) == (8, 16)


def _geographic_rasterizer(width_deg=0.002, crs=GEOGRAPHIC_CRS):
    geometry = [box(9.0, 50.0, 9.0 + width_deg / 2, 50.001),
                box(9.0 + width_deg / 2, 50.0, 9.0 + width_deg, 50.001)]
    gdf = gpd.GeoDataFrame({"category": ["a", "b"]}, geometry=geometry, crs=crs)
    return gdf, GeoRasterizer(InMemoryVectorDataset(gdf),
                              {"category": {"a": 1, "b": 2}})


@pytest.mark.parametrize("method", ["rasterize", "rasterize_metrics"])
def test_geographic_crs_is_reprojected_to_utm_with_one_warning(method):
    gdf, rasterizer = _geographic_rasterizer()
    before = gdf.copy()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if method == "rasterize":
            rasterizer.rasterize(resolution_in_m=10, on_thin_features="ignore")
        else:
            rasterizer.rasterize_metrics(resolution_in_m=10)
    reprojections = [w for w in caught
                     if issubclass(w.category, CrsReprojectionWarning)]
    assert len(reprojections) == 1
    message = str(reprojections[0].message)
    assert "EPSG:4326 (degrees)" in message
    assert "reprojected to EPSG:32632 (UTM, metres)" in message
    assert "source and target coordinates" in message
    assert "to_crs('EPSG:32632')" in message
    assert Path(reprojections[0].filename).name == Path(__file__).name
    assert rasterizer.crs == "EPSG:32632"
    assert rasterizer.crs.is_projected
    # The caller's GeoDataFrame is untouched.
    assert gdf.crs == GEOGRAPHIC_CRS
    assert "cost" not in gdf.columns
    assert_geodataframe_equal(gdf, before)


def test_reprojected_raster_has_the_metric_size():
    gdf, rasterizer = _geographic_rasterizer()
    with pytest.warns(CrsReprojectionWarning):
        rasterizer.rasterize(resolution_in_m=10, on_thin_features="ignore")
    utm = gdf.to_crs(gdf.estimate_utm_crs())
    minx, miny, maxx, maxy = utm.total_bounds
    rows, columns = rasterizer.raster.shape
    assert abs(columns - (maxx - minx) / 10) <= 1
    assert abs(rows - (maxy - miny) / 10) <= 1
    assert _values(rasterizer.raster) == [1, 2]


def test_second_call_does_not_warn_again():
    _, rasterizer = _geographic_rasterizer()
    with pytest.warns(CrsReprojectionWarning):
        rasterizer.rasterize(resolution_in_m=10, on_thin_features="ignore")
    _, caught = _user_warnings(lambda: rasterizer.rasterize(
        resolution_in_m=10, on_thin_features="ignore"))
    assert not [w for w in caught if issubclass(w.category, CrsReprojectionWarning)]


def test_geographic_crs_is_reprojected_with_a_bounding_box_too():
    gdf, rasterizer = _geographic_rasterizer()
    # Round 14: the box is read in the CRS the data was given in (degrees)
    # and reprojected with it; a box already in UTM metres is no longer read.
    box_degrees = box(9.0, 50.0, 9.001, 50.001)
    with pytest.warns(CrsReprojectionWarning):
        rasterizer.rasterize(resolution_in_m=10, bounding_box=box_degrees,
                             on_thin_features="ignore")
    assert rasterizer.crs == "EPSG:32632"
    assert rasterizer.raster.size > 0


def test_wide_geographic_extent_says_so_in_the_warning():
    geometry = [box(0.0, 50.0, 0.001, 50.001), box(9.0, 50.0, 9.001, 50.001)]
    gdf = gpd.GeoDataFrame({"category": ["a", "b"]}, geometry=geometry,
                           crs=GEOGRAPHIC_CRS)
    rasterizer = GeoRasterizer(InMemoryVectorDataset(gdf),
                               {"category": {"a": 1, "b": 2}})
    with pytest.warns(CrsReprojectionWarning, match="more than one UTM zone") as rec:
        rasterizer.rasterize(resolution_in_m=1000, on_thin_features="ignore")
    assert "zone of its centre" in str(rec[0].message)


def test_geographic_data_without_an_estimable_zone_is_a_clear_error():
    gdf, rasterizer = _geographic_rasterizer()
    rasterizer.base_dataset.data = gdf.assign(
        geometry=gpd.GeoSeries([None, None], crs=GEOGRAPHIC_CRS))
    with pytest.raises(InvalidInputError, match="no UTM zone"):
        rasterizer.rasterize()


def test_empty_geographic_data_is_still_rejected_as_empty():
    gdf = gpd.GeoDataFrame({"category": []}, geometry=[], crs=GEOGRAPHIC_CRS)
    rasterizer = GeoRasterizer(InMemoryVectorDataset(gdf), {"category": {"a": 1}})
    with pytest.raises(ValueError, match="empty"):
        rasterizer.rasterize()


def test_dataset_added_later_is_reprojected_to_the_raster_crs():
    _, rasterizer = _geographic_rasterizer()
    with pytest.warns(CrsReprojectionWarning):
        rasterizer.rasterize(resolution_in_m=10, on_thin_features="ignore")
    overlay = gpd.GeoDataFrame(
        {"category": ["a"]}, geometry=[box(9.0, 50.0, 9.0005, 50.0005)],
        crs=GEOGRAPHIC_CRS)
    raster = rasterizer.modify_raster_from_dataset(overlay, 5, ignore_value=None)
    assert rasterizer.crs == "EPSG:32632"
    assert 5 in raster


# ---------------------------------------------------------------------------
# #18  constructor: documented input types, bbox, mask and default_crs
# ---------------------------------------------------------------------------

COSTS = {"category": {"a": 1, "b": 2}}


def _user_warnings(function):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = function()
    return result, [w for w in caught if issubclass(w.category, UserWarning)]


def test_constructor_accepts_a_geodataframe():
    rasterizer, caught = _user_warnings(
        lambda: GeoRasterizer(_gdf(["a", "b"]), COSTS))
    assert caught == []
    assert isinstance(rasterizer.base_dataset, InMemoryVectorDataset)
    rasterizer.rasterize()
    reference = _rasterizer(["a", "b"], COSTS)
    reference.rasterize()
    np.testing.assert_array_equal(rasterizer.raster, reference.raster)
    assert rasterizer.transform == reference.transform


@pytest.mark.parametrize("as_path", [str, Path], ids=["str", "Path"])
def test_constructor_accepts_a_file_path(tmp_path, as_path):
    source = tmp_path / "landuse.geojson"
    _gdf(["a", "b"]).to_file(source, driver="GeoJSON")
    rasterizer = GeoRasterizer(as_path(source), COSTS)
    assert list(rasterizer.base_data["category"]) == ["a", "b"]
    rasterizer.rasterize()
    assert _values(rasterizer.raster) == [1, 2]


def test_constructor_reports_an_unsupported_source_as_a_value_error():
    """Not the AttributeError ("no attribute 'data'") it used to be."""
    with pytest.raises(ValueError):
        GeoRasterizer(42, COSTS)


def test_bbox_limits_a_geodataframe_source():
    rasterizer, caught = _user_warnings(
        lambda: GeoRasterizer(_gdf(["a", "b"]), COSTS, bbox=(0.0, 0.0, 4.0, 4.0)))
    assert caught == []
    assert list(rasterizer.base_data["category"]) == ["a"]
    assert tuple(rasterizer.base_data.total_bounds) == (0.0, 0.0, 4.0, 4.0)
    rasterizer.rasterize()
    assert rasterizer.raster.shape == (4, 4)


def test_mask_limits_a_geodataframe_source():
    mask = gpd.GeoDataFrame(geometry=[box(4.0, 0.0, 8.0, 4.0)], crs=CRS)
    rasterizer = GeoRasterizer(_gdf(["a", "b"]), COSTS, mask=mask)
    assert list(rasterizer.base_data["category"]) == ["b"]
    assert rasterizer.mask is mask


def test_default_crs_reprojects_a_geodataframe_source():
    rasterizer, caught = _user_warnings(
        lambda: GeoRasterizer(_gdf(["a", "b"]), COSTS,
                              default_crs=OTHER_PROJECTED_CRS))
    assert caught == []
    assert rasterizer.crs == OTHER_PROJECTED_CRS
    assert rasterizer.base_data.crs == OTHER_PROJECTED_CRS


def test_unloaded_dataset_adopts_bbox_and_default_crs():
    dataset = InMemoryVectorDataset(_gdf(["a", "b"]))
    assert dataset.data is None
    rasterizer, caught = _user_warnings(
        lambda: GeoRasterizer(dataset, COSTS, bbox=(0.0, 0.0, 4.0, 4.0),
                              default_crs=CRS))
    assert caught == []
    assert rasterizer.base_dataset is dataset
    assert list(rasterizer.base_data["category"]) == ["a"]
    assert rasterizer.crs == CRS


def test_loaded_dataset_with_another_bbox_warns_instead_of_dropping_it():
    dataset = InMemoryVectorDataset(_gdf(["a", "b"]), crs=CRS)
    dataset.load_data()
    with pytest.warns(UserWarning, match="ignores bbox") as record:
        rasterizer = GeoRasterizer(dataset, COSTS, bbox=(0.0, 0.0, 4.0, 4.0))
    assert "already loaded" in str(record[0].message)
    assert "clip_to_area" in str(record[0].message)
    assert Path(record[0].filename).name == Path(__file__).name
    # Nothing was cut: the loaded data is used as it is, as before.
    assert list(rasterizer.base_data["category"]) == ["a", "b"]


def test_loaded_dataset_with_its_own_bbox_does_not_warn():
    """The pattern of the examples: load with a bbox, pass the same one again."""
    bbox = gpd.GeoDataFrame(geometry=[box(0.0, 0.0, 8.0, 4.0)], crs=CRS)
    dataset = InMemoryVectorDataset(_gdf(["a", "b"]), crs=CRS, bbox=bbox)
    dataset.load_data()
    rasterizer, caught = _user_warnings(
        lambda: GeoRasterizer(dataset, COSTS, bbox))
    assert caught == []
    assert rasterizer.bbox is bbox
    assert list(rasterizer.base_data["category"]) == ["a", "b"]


def test_loaded_dataset_default_crs_warns_only_when_it_differs():
    dataset = InMemoryVectorDataset(_gdf(["a", "b"]), crs=CRS)
    dataset.load_data()
    _, caught = _user_warnings(
        lambda: GeoRasterizer(dataset, COSTS, default_crs="epsg:32632"))
    assert caught == []
    with pytest.warns(UserWarning, match="ignores default_crs"):
        rasterizer = GeoRasterizer(dataset, COSTS,
                                   default_crs=OTHER_PROJECTED_CRS)
    assert rasterizer.crs == CRS


def test_raster_dataset_with_bbox_warns():
    raster = InMemoryRasterDataset(np.ones((4, 4), dtype=np.uint16), CRS,
                                   from_origin(0.0, 4.0, 1.0, 1.0))
    with pytest.warns(UserWarning, match="ignores bbox") as record:
        rasterizer = GeoRasterizer(raster, COSTS, bbox=(0.0, 0.0, 2.0, 2.0))
    assert "raster data" in str(record[0].message)
    assert rasterizer.raster.shape == (4, 4)


def test_dataset_without_options_is_taken_as_it_is():
    dataset = InMemoryVectorDataset(_gdf(["a", "b"]), crs=CRS)
    rasterizer, caught = _user_warnings(lambda: GeoRasterizer(dataset, COSTS))
    assert caught == []
    assert rasterizer.base_dataset is dataset
    assert rasterizer.bbox is None and rasterizer.default_crs is None


# ---------------------------------------------------------------------------
# #19  CostAssumptions: empty table, extra top-level keys
# ---------------------------------------------------------------------------

def test_empty_cost_dictionary_raises_format_error():
    with pytest.raises(FormatError, match="empty"):
        CostAssumptions({})


def test_empty_cost_dictionary_raises_format_error_in_load():
    with pytest.raises(FormatError, match="empty"):
        CostAssumptions().load({})


def test_empty_cost_dictionary_raises_format_error_in_the_rasterizer():
    with pytest.raises(FormatError, match="empty"):
        GeoRasterizer(InMemoryVectorDataset(_gdf(["a"]), crs=CRS), {})


def test_empty_cost_table_under_a_key_still_raises_format_error():
    with pytest.raises(FormatError):
        CostAssumptions({"category": {}})


def test_extra_top_level_keys_warn_and_are_named():
    source = {"category": {"a": 1}, "other": {"b": 2}, 7: {"c": 3}}
    with pytest.warns(UserWarning, match="top-level") as record:
        assumptions = CostAssumptions(source)
    message = str(record[0].message)
    assert "'other'" in message and "7" in message
    assert "'category'" in message
    assert Path(record[0].filename).name == Path(__file__).name
    # The first entry is used, as before.
    assert assumptions.main_feature == "category"
    assert assumptions.cost_assumptions == {"a": 1}


def test_extra_top_level_keys_warn_in_load():
    assumptions = CostAssumptions()
    with pytest.warns(UserWarning, match="Ignored: 'other'") as record:
        assumptions.load({"category": {"a": 1}, "other": {"b": 2}})
    assert Path(record[0].filename).name == Path(__file__).name
    assert assumptions.main_feature == "category"


def test_many_extra_top_level_keys_are_truncated():
    source = {"category": {"a": 1}, **{f"k{i}": {"b": 2} for i in range(8)}}
    with pytest.warns(UserWarning, match="top-level") as record:
        CostAssumptions(source)
    message = str(record[0].message)
    assert "9 top-level keys" in message
    assert "'k4'" in message and "'k5'" not in message
    assert "and 3 more" in message


@pytest.mark.parametrize("source, main, side", [
    ({"category": {"a": 1, "b": 2}}, "category", []),
    ({("category", "kind"): {("a", "x"): 1}}, "category", ["kind"]),
])
def test_single_top_level_key_loads_without_a_warning(source, main, side):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assumptions = CostAssumptions(source)
    assert caught == []
    assert assumptions.main_feature == main
    assert assumptions.side_features == side
