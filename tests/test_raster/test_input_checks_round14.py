"""Input checks of GeoRasterizer (round 14).

Three gaps, each with the failing input and the boundary that must keep working:

* ``modify_raster_from_dataset`` / ``modify_raster_with_geodataframe`` wrote a
  value that does not fit the uint16 raster (negative, above 65535, fractional,
  NaN) without a word; they now raise ``InvalidInputError`` before any cell
  changes. A multiplier saturates at 65535 (documented) and is kept.
* ``rasterize(fill_value=...)`` is checked against the output dtype.
* A ``bounding_box`` given with geographic input is reprojected together with
  the data instead of being read as UTM metres.

Everything is built from tiny in-memory GeoDataFrames.
"""
import math
import warnings

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from pyorps.core.exceptions import InvalidInputError
from pyorps.core.types import IMPASSABLE_CELL_COST
from pyorps.core.warning_types import CrsReprojectionWarning
from pyorps.io.geo_dataset import InMemoryVectorDataset
from pyorps.raster.rasterizer import GeoRasterizer

CRS = "EPSG:32632"
GEOGRAPHIC_CRS = "EPSG:4326"
COSTS = {"category": {"a": 10, "b": 20}}


def _rasterizer(crs=CRS, scale=1.0, origin=(500000.0, 5500000.0)):
    """Two touching 4 x 4 squares (scaled), categories a and b."""
    x0, y0 = origin
    size = 4.0 * scale
    geometry = [box(x0, y0, x0 + size, y0 + size),
                box(x0 + size, y0, x0 + 2 * size, y0 + size)]
    gdf = gpd.GeoDataFrame({"category": ["a", "b"]}, geometry=geometry, crs=crs)
    return GeoRasterizer(InMemoryVectorDataset(gdf, crs=crs), COSTS)


def _burned():
    rasterizer = _rasterizer()
    rasterizer.rasterize(resolution_in_m=1.0, on_thin_features="ignore")
    return rasterizer


def _overlay(rasterizer):
    """A polygon over the first square, in the raster CRS."""
    return gpd.GeoDataFrame(geometry=[box(500000.0, 5500000.0, 500004.0,
                                          5500004.0)], crs=rasterizer.crs)


# ----------------------------------------------------------------------------
# Gap 1: values written by the modify methods
# ----------------------------------------------------------------------------

BAD_REPLACE_VALUES = [-1, -5, 65536, 70000, 1.5, float("nan"), float("inf"),
                      "20", None, True]


class TestModifyValueRange:
    @pytest.mark.parametrize("value", BAD_REPLACE_VALUES)
    def test_bad_value_is_rejected_and_the_raster_is_untouched(self, value):
        rasterizer = _burned()
        before = rasterizer.raster.copy()
        with pytest.raises(InvalidInputError, match="value"):
            rasterizer.modify_raster_with_geodataframe(_overlay(rasterizer), value)
        np.testing.assert_array_equal(rasterizer.raster, before)

    def test_message_names_value_range_and_parameter(self):
        rasterizer = _burned()
        with pytest.raises(InvalidInputError) as info:
            rasterizer.modify_raster_with_geodataframe(_overlay(rasterizer), 70000)
        text = str(info.value)
        assert "value=70000" in text
        assert "0 to 65535" in text
        assert str(IMPASSABLE_CELL_COST) in text

    @pytest.mark.parametrize("value", [0, 1, 20, 20.0, 65534,
                                       IMPASSABLE_CELL_COST, np.uint16(7),
                                       np.float64(9.0)])
    def test_legal_values_are_written_as_before(self, value):
        rasterizer = _burned()
        rasterizer.modify_raster_with_geodataframe(_overlay(rasterizer), value,
                                                   ignore_value=None)
        assert rasterizer.raster[0, 0] == int(value)
        assert rasterizer.raster.dtype == np.uint16

    @pytest.mark.parametrize("factor", [-1.0, -0.001, float("nan"),
                                        float("inf"), "2"])
    def test_bad_multiplier_is_rejected(self, factor):
        rasterizer = _burned()
        before = rasterizer.raster.copy()
        with pytest.raises(InvalidInputError, match="value"):
            rasterizer.modify_raster_with_geodataframe(_overlay(rasterizer),
                                                       factor, multiply=True)
        np.testing.assert_array_equal(rasterizer.raster, before)

    def test_multiplier_saturates_at_the_maximum_and_never_wraps(self):
        """Documented: the product is rounded and clipped to 0..65535."""
        rasterizer = _burned()
        assert rasterizer.raster[0, 0] == 10
        rasterizer.modify_raster_with_geodataframe(_overlay(rasterizer), 1e9,
                                                   multiply=True)
        assert rasterizer.raster[0, 0] == IMPASSABLE_CELL_COST

    def test_fractional_and_zero_multipliers_keep_working(self):
        rasterizer = _burned()
        rasterizer.modify_raster_with_geodataframe(_overlay(rasterizer), 1.5,
                                                   multiply=True)
        assert rasterizer.raster[0, 0] == 15
        rasterizer.modify_raster_with_geodataframe(_overlay(rasterizer), 0,
                                                   multiply=True)
        assert rasterizer.raster[0, 0] == 0

    def test_bad_ignore_value_is_rejected(self):
        rasterizer = _burned()
        with pytest.raises(InvalidInputError, match="ignore_value"):
            rasterizer.modify_raster_with_geodataframe(
                _overlay(rasterizer), 5, ignore_value="x")

    @pytest.fixture
    def from_dataset(self):
        rasterizer = _burned()
        overlay = _overlay(rasterizer)
        return rasterizer, overlay, rasterizer.raster.copy()

    @pytest.mark.parametrize("cost", [-1, 65536, 2.5, float("nan"), True])
    def test_modify_from_dataset_rejects_a_bad_cost_naming_the_parameter(
            self, from_dataset, cost):
        rasterizer, overlay, before = from_dataset
        with pytest.raises(InvalidInputError, match="cost_assumptions"):
            rasterizer.modify_raster_from_dataset(overlay, cost)
        np.testing.assert_array_equal(rasterizer.raster, before)

    @pytest.mark.parametrize("factor", [-2, float("nan")])
    def test_modify_from_dataset_rejects_a_bad_multiplier(self, from_dataset,
                                                          factor):
        rasterizer, overlay, before = from_dataset
        with pytest.raises(InvalidInputError, match="cost_assumptions"):
            rasterizer.modify_raster_from_dataset(overlay, factor, multiply=True)
        np.testing.assert_array_equal(rasterizer.raster, before)

    @pytest.mark.parametrize("forbidden", [-1, 65536, 0.5, float("nan")])
    def test_modify_from_dataset_rejects_a_bad_forbidden_value(self, from_dataset,
                                                               forbidden):
        rasterizer, overlay, before = from_dataset
        overlay["zone"] = "no_go"
        with pytest.raises(InvalidInputError, match="forbidden_value"):
            rasterizer.modify_raster_from_dataset(
                overlay, 5, zone_field="zone", forbidden_zone="no_go",
                forbidden_value=forbidden)
        np.testing.assert_array_equal(rasterizer.raster, before)

    @pytest.mark.parametrize("cost", [-3, 70000, 1.5])
    def test_a_bad_group_cost_from_a_cost_table_is_rejected(self, from_dataset,
                                                            cost):
        rasterizer, overlay, before = from_dataset
        overlay["category"] = "z"
        with pytest.raises(InvalidInputError, match="cost"):
            rasterizer.modify_raster_from_dataset(overlay, {"category": {"z": cost}})
        np.testing.assert_array_equal(rasterizer.raster, before)

    def test_group_cost_multiplier_saturates(self, from_dataset):
        rasterizer, overlay, _ = from_dataset
        overlay["category"] = "z"
        rasterizer.modify_raster_from_dataset(overlay, {"category": {"z": 1e7}},
                                              multiply=True)
        assert rasterizer.raster[0, 0] == IMPASSABLE_CELL_COST

    def test_legal_costs_from_dataset_keep_working(self, from_dataset):
        rasterizer, overlay, _ = from_dataset
        rasterizer.modify_raster_from_dataset(overlay, 33, ignore_value=None)
        assert rasterizer.raster[0, 0] == 33
        overlay["zone"] = "no_go"
        rasterizer.modify_raster_from_dataset(
            overlay, 5, ignore_value=None, zone_field="zone",
            forbidden_zone="no_go", forbidden_value=IMPASSABLE_CELL_COST)
        assert rasterizer.raster[0, 0] == IMPASSABLE_CELL_COST


# ----------------------------------------------------------------------------
# Gap 2: fill_value against the output dtype
# ----------------------------------------------------------------------------

class TestFillValue:
    @pytest.mark.parametrize("fill", [70000, -1, 65536, 1.5, float("nan"),
                                      float("inf"), "5", None])
    def test_uint16_rejects_an_unrepresentable_fill_value(self, fill):
        rasterizer = _rasterizer()
        with pytest.raises(InvalidInputError, match="fill_value"):
            rasterizer.rasterize(fill_value=fill, on_thin_features="ignore")
        assert rasterizer.raster is None

    def test_message_names_value_range_and_parameter(self):
        with pytest.raises(InvalidInputError) as info:
            _rasterizer().rasterize(fill_value=70000)
        text = str(info.value)
        assert "fill_value=70000" in text and "0 to 65535" in text

    def test_check_runs_before_any_work(self):
        """Before the cost table is applied, so the base data is untouched."""
        rasterizer = _rasterizer()
        with pytest.raises(InvalidInputError, match="fill_value"):
            rasterizer.rasterize(fill_value=-1)
        assert "cost" not in rasterizer.base_data.columns

    @pytest.mark.parametrize("dtype,fill", [("uint8", 256), ("uint8", -1),
                                            ("int16", 40000), ("int16", 2.5),
                                            ("int32", float("nan"))])
    def test_other_integer_dtypes(self, dtype, fill):
        rasterizer = _rasterizer()
        with pytest.raises(InvalidInputError, match="fill_value"):
            rasterizer.rasterize(fill_value=fill, dtype=dtype)

    def test_float_dtype_takes_nan_and_fractions(self):
        rasterizer = _rasterizer()
        rasterizer.rasterize(fill_value=float("nan"), dtype="float32",
                             bounding_box=box(499998, 5499998, 500012, 5500006),
                             on_thin_features="ignore")
        assert np.isnan(rasterizer.raster).any()
        rasterizer = _rasterizer()
        rasterizer.rasterize(fill_value=1.5, dtype="float32",
                             on_thin_features="ignore")
        assert rasterizer.raster.dtype == np.float32

    @pytest.mark.parametrize("fill", [0, 1, 65534, IMPASSABLE_CELL_COST, 5.0,
                                      np.uint16(3)])
    def test_representable_fill_values_keep_working(self, fill):
        rasterizer = _rasterizer()
        rasterizer.rasterize(
            fill_value=fill, bounding_box=box(499996, 5499996, 500012, 5500006),
            on_thin_features="ignore")
        assert (rasterizer.raster == int(fill)).any()

    def test_the_default_is_unchanged(self):
        rasterizer = _rasterizer()
        rasterizer.rasterize(
            bounding_box=box(499996, 5499996, 500012, 5500006),
            on_thin_features="ignore")
        assert rasterizer.raster[0, 0] == IMPASSABLE_CELL_COST


# ----------------------------------------------------------------------------
# Gap 3: bounding box with geographic input
# ----------------------------------------------------------------------------

def _geographic():
    # About 0.0005 degrees (roughly 35 m x 55 m) per square.
    return _rasterizer(crs=GEOGRAPHIC_CRS, scale=0.000125, origin=(9.0, 50.0))


class TestBoundingBoxCrs:
    def test_projected_input_reads_the_box_as_given(self):
        """Already correct: a projected CRS is not reprojected, nor the box."""
        rasterizer = _rasterizer()
        rasterizer.rasterize(bounding_box=box(499996, 5499996, 500012, 5500006),
                             on_thin_features="ignore")
        assert rasterizer.raster.shape == (10, 16)
        left, _, _, top = (rasterizer.transform.c, 0, 0, rasterizer.transform.f)
        assert (left, top) == (499996, 5500006)

    def test_geographic_box_is_reprojected_with_the_data(self):
        rasterizer = _geographic()
        degrees = box(8.99990, 49.99990, 9.00110, 50.00060)
        with pytest.warns(CrsReprojectionWarning):
            rasterizer.rasterize(bounding_box=degrees, on_thin_features="ignore")
        expected = gpd.GeoSeries([degrees], crs=GEOGRAPHIC_CRS).to_crs(
            rasterizer.crs).total_bounds
        transform = rasterizer.transform
        height, width = rasterizer.raster.shape
        got = (transform.c, transform.f + transform.e * height,
               transform.c + transform.a * width, transform.f)
        np.testing.assert_allclose(got, expected, atol=1e-6)
        # The box is a few tens of metres, not read as degrees-as-metres.
        assert 10 < width < 200 and 10 < height < 200

    def test_box_around_the_data_covers_all_of_it(self):
        rasterizer = _geographic()
        degrees = box(8.99990, 49.99990, 9.00110, 50.00060)
        with pytest.warns(CrsReprojectionWarning):
            rasterizer.rasterize(bounding_box=degrees, on_thin_features="ignore")
        cost = rasterizer.raster
        assert (cost == 10).any() and (cost == 20).any()

    def test_no_box_still_works_with_geographic_input(self):
        rasterizer = _geographic()
        with pytest.warns(CrsReprojectionWarning):
            rasterizer.rasterize(on_thin_features="ignore")
        assert (rasterizer.raster == 10).any()

    def test_docstring_states_the_crs_of_the_box(self):
        doc = GeoRasterizer.rasterize.__doc__
        assert "CRS of the base data" in doc and "geographic" in doc


def test_no_unexpected_warning_for_a_valid_call():
    rasterizer = _rasterizer()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rasterizer.rasterize(on_thin_features="ignore")
    assert math.isfinite(float(rasterizer.raster.max()))
