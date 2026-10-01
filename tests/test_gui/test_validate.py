"""Unit tests: pre-flight validators (Section 21.5) neutralize F1/F3/F4/F6."""
import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from pyorps.gui.services import validate


def test_project_crs_blocks_geographic():
    notice = validate.validate_project_crs("EPSG:4326")
    assert notice is not None and notice.severity == "error"
    assert "metric" in notice.title.lower()


def test_project_crs_accepts_projected():
    assert validate.validate_project_crs("EPSG:25832") is None


def test_project_crs_unknown():
    notice = validate.validate_project_crs("EPSG:0")
    assert notice is not None and notice.title == "Unknown CRS"


def test_search_buffer_f1():
    assert validate.validate_search_buffer(500.0) is None
    for empty in (None, 0, -5):
        notice = validate.validate_search_buffer(empty)
        assert notice is not None
        assert "search buffer" in notice.title.lower()


def test_raster_size_estimate_and_thresholds():
    est = validate.estimate_raster_size((0, 0, 1000, 500), 1.0)
    assert est["n_cells"] == 500_000
    assert "500,000 cells" in est["label"]
    # 30 km x 20 km at 1 m -> 600M cells, uint16 = 1.2 GB -> warn (not block)
    warn = validate.validate_raster_size((0, 0, 30_000, 20_000), 1.0)
    assert warn is not None and warn.severity == "warning"
    # 60 km x 30 km at 1 m -> 1.8B cells -> 3.6 GB -> block
    block = validate.validate_raster_size((0, 0, 60_000, 30_000), 1.0)
    assert block is not None and block.severity == "error"
    assert validate.validate_raster_size((0, 0, 1000, 1000), 1.0) is None
    assert validate.validate_raster_size(None, 1.0) is None


def _gdf(records):
    return gpd.GeoDataFrame(
        records,
        geometry=[Polygon([(0, 0), (1, 0), (1, 1)])] * len(records),
        crs="EPSG:25832")


def test_cost_coverage_single_key():
    gdf = _gdf([{"use": "forest"}, {"use": "road"}, {"use": "water"}])
    missing = validate.uncovered_categories(gdf, ("use",),
                                            {"forest": 10, "road": 20})
    assert missing == ["water"]
    # a "" catch-all covers everything
    assert validate.uncovered_categories(
        gdf, ("use",), {"forest": 10, "": 5}) == []


def test_cost_coverage_combination_key():
    gdf = _gdf([{"use": "forest", "sub": "oak"},
                {"use": "forest", "sub": "pine"},
                {"use": "road", "sub": "highway"}])
    assumptions = {"forest": {"oak": 10}, "road": {"": 20}}
    missing = validate.uncovered_categories(gdf, ("use", "sub"), assumptions)
    assert missing == ["forest / pine"]
    notice = validate.validate_cost_coverage(gdf, ("use", "sub"), assumptions)
    assert notice is not None and "forbidden" in notice.title
    # full coverage -> no notice
    assumptions["forest"][""] = 10
    assert validate.validate_cost_coverage(gdf, ("use", "sub"),
                                           assumptions) is None


def test_points_in_bounds():
    bounds = (0, 0, 100, 100)
    assert validate.validate_points_in_bounds([(10, 10)], bounds) is None
    notice = validate.validate_points_in_bounds([(10, 10), (500, 5)], bounds)
    assert notice is not None and "outside" in notice.title


def test_validate_file(tmp_path):
    assert validate.validate_file("", "vector") is not None
    assert validate.validate_file("x.xyz", "vector").title == \
        "Unsupported file type"
    missing = tmp_path / "gone.geojson"
    assert validate.validate_file(str(missing), "vector").title == \
        "File not found"
    present = tmp_path / "ok.geojson"
    present.write_text("{}")
    assert validate.validate_file(str(present), "vector") is None


def test_validate_wfs_url():
    assert validate.validate_wfs_url("https://example.com/wfs") is None
    for bad in ("", "ftp://x", "not a url", "https://"):
        assert validate.validate_wfs_url(bad) is not None
