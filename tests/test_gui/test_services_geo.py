"""Unit tests: pyorps.gui.services.geo (carried over from webviz v1)."""
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from pyorps.gui.services import geo

from conftest import SOURCE, TEST_CRS


def test_gdf_wgs84_roundtrip(finder):
    gdf = gpd.GeoDataFrame(finder.paths.to_geodataframe_records(),
                           geometry="geometry", crs=finder.dataset.crs)
    gj = geo.gdf_to_wgs84_geojson(gdf)
    assert gj["type"] == "FeatureCollection"
    coords = gj["features"][0]["geometry"]["coordinates"]
    assert 8 < coords[0][0] < 10 and 50 < coords[0][1] < 51

    line_crs = geo.wgs84_linestring_to_crs(coords, finder.dataset.crs)
    assert abs(line_crs.coords[0][0] - SOURCE[0]) < 5


def test_gdf_wgs84_requires_crs():
    gdf = gpd.GeoDataFrame({"a": [1]}, geometry=[Point(0, 0)])
    with pytest.raises(ValueError, match="no CRS"):
        geo.gdf_to_wgs84_geojson(gdf)


def test_gdf_wgs84_handles_timestamp_columns():
    """Real WFS/ALKIS layers carry datetime columns; they must serialize (C11)."""
    gdf = gpd.GeoDataFrame(
        {"name": ["a"], "checked": [pd.Timestamp("2024-01-02")]},
        geometry=[Point(500000, 5600000)], crs=TEST_CRS)
    gj = geo.gdf_to_wgs84_geojson(gdf)
    props = gj["features"][0]["properties"]
    assert props["name"] == "a"
    assert isinstance(props["checked"], str) and "2024-01-02" in props["checked"]


def test_point_wgs84_to_crs_roundtrip():
    x, y = geo.point_wgs84_to_crs(50.5, 9.0, TEST_CRS)
    assert 400_000 < x < 600_000 and 5_500_000 < y < 5_700_000


def test_wgs84_bounds_and_feature_collection():
    gdf = gpd.GeoDataFrame({"v": [1]},
                           geometry=[LineString([(9.0, 50.0), (9.1, 50.1)])],
                           crs="EPSG:4326")
    gj = geo.gdf_to_wgs84_geojson(gdf)
    bounds = geo.wgs84_bounds(gj)
    (s, w), (n, e) = bounds
    assert s == pytest.approx(50.0) and n == pytest.approx(50.1)
    assert w == pytest.approx(9.0) and e == pytest.approx(9.1)

    fc = geo.as_feature_collection({"type": "Point", "coordinates": [1, 2]})
    assert fc["type"] == "FeatureCollection"
    assert fc["features"][0]["geometry"]["type"] == "Point"


def test_wgs84_bounds_empty_returns_none():
    assert geo.wgs84_bounds({"type": "FeatureCollection",
                             "features": []}) is None
