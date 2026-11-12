"""The cached GeoDataFrame of a ConstrainedPathFinder never lags behind its path collection."""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from pyorps.graph.constrained_path_finder import ConstrainedPathFinder

SHAPE = (40, 40)
SOURCE = (5.5, 5.5)
TARGET = (34.5, 34.5)
OTHER_TARGET = (34.5, 10.5)


def _profile() -> dict:
    return {
        "name": "gdf", "description": "tiny test profile",
        "soft_angle_limit_deg": 5.0, "hard_angle_limit_deg": 60.0,
        "angle_cost_function": "linear", "angle_cost_params": {"scale": 100},
        "min_span_m": 10.0, "max_span_m": 20.0, "span_bin_size_m": 1.0,
        "tower_cost_function": "terrain_plus_angle",
        "tower_cost_params": {
            "terrain_cost_map": {"0": 1000, "500": 5000}, "terrain_interpolation": "linear",
            "angle_types": {"suspension": {"max_angle_deg": 10.0, "base_cost": 500},
                            "angle_tower": {"max_angle_deg": 60.0, "base_cost": 2000}},
        },
    }


@pytest.fixture
def finder(tmp_path) -> ConstrainedPathFinder:
    path = tmp_path / "cost.tif"
    data = np.random.default_rng(7).integers(50, 150, size=SHAPE).astype(np.uint16)
    with rasterio.open(path, "w", driver="GTiff", height=SHAPE[0], width=SHAPE[1], count=1, dtype="uint16",
                       crs="EPSG:32632", transform=from_bounds(0, 0, SHAPE[1], SHAPE[0], SHAPE[1], SHAPE[0])) as dst:
        dst.write(data, 1)
    return ConstrainedPathFinder(str(path), SOURCE, TARGET, profile=_profile(), neighborhood_str="r1")


def test_second_route_is_in_the_cached_geodataframe(finder):
    finder.find_route()
    assert len(finder.create_path_geodataframe()) == 1
    finder.find_route(target=OTHER_TARGET)
    # The attribute is the cache: it must not keep the one-route frame.
    cached = finder.path_gdf
    assert cached is None or len(cached) == len(finder.paths) == 2
    assert len(finder.create_path_geodataframe()) == len(finder.paths) == 2


def test_export_after_two_routes_holds_both(finder, tmp_path):
    gpd = pytest.importorskip("geopandas")
    finder.find_route()
    finder.create_path_geodataframe()
    finder.find_route(target=OTHER_TARGET)
    out = tmp_path / "paths.gpkg"
    finder.save_paths(str(out))
    assert len(gpd.read_file(out)) == len(finder.paths) == 2
