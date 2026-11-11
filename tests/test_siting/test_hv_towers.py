"""Verified 110 kV towers from OSM lines (owner rule 2026-10-06)."""
import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point

from pyorps.siting.hv_towers import verified_110kv_towers


def _lines():
    return gpd.GeoDataFrame({
        "osm_id": ["a", "b", "c", "d", "e"],
        "power": ["line", "line", "line", "substation", "line"],
        "voltage": ["110000", "380000;110000", "20000", "110000", "110000"],
        "operator": ["Netz", None, "Netz", None, "DB Energie"],
        "frequency": ["50", None, "50", None, "16.7"],
    }, geometry=[LineString([(0, 0), (100, 0)]), LineString([(0, 50), (100, 50)]), LineString([(0, 100), (100, 100)]),
                 LineString([(0, 150), (100, 150)]), LineString([(0, 200), (100, 200)])], crs="EPSG:25832")


def _towers():
    pts = [(10, 0), (50, 50), (10, 100), (10, 150), (10, 200), (10, 0.5), (10, 30), (60, 0)]
    power = ["tower", "tower", "tower", "tower", "tower", "pole", "tower", "portal"]
    return gpd.GeoDataFrame({"power": power}, geometry=[Point(p) for p in pts], crs="EPSG:25832")


def test_a_tower_on_a_110000_line_is_kept_everything_else_is_not():
    out = verified_110kv_towers(_lines(), _towers())
    # lines a, b (mixed voltage) and the pole 0.5 m off a; the traction line e is dropped (D-e)
    assert sorted(out.index) == [0, 1, 5]
    assert out.loc[1, "line_voltage"] == "380000;110000"            # the multi-circuit line counts
    assert 2 not in out.index and 3 not in out.index                 # 20 kV line, substation outline
    # 30 m off any line; a portal is not a support point
    assert 6 not in out.index and 7 not in out.index


def test_tolerance_and_traction_option():
    assert 5 not in verified_110kv_towers(_lines(), _towers(), tol_m=0.1).index
    out = verified_110kv_towers(_lines(), _towers(), exclude_traction=False)
    assert bool(out.loc[4, "traction"]) and not bool(out.loc[0, "traction"])
    assert 4 not in verified_110kv_towers(_lines(), _towers()).index


def test_empty_and_crs_errors():
    assert verified_110kv_towers(_lines(), _towers().iloc[0:0]).empty
    with pytest.raises(ValueError, match="projected CRS"):
        verified_110kv_towers(_lines(), _towers().to_crs("EPSG:4326"))


def test_traction_markers_are_parameters():
    # a different railway operator name and frequency: the German-marked line e stays, line a is dropped instead
    out = verified_110kv_towers(_lines(), _towers(), traction_operator="Netz", traction_frequency="60")
    assert 4 in out.index and 0 not in out.index and bool(out.loc[1, "traction"]) is False
