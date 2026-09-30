"""Terminals are named by their id attribute, never by row (plan A2).

The retired pipeline keyed PCC fields by ``enumerate(pccs.geometry)``; the
case-study file's rows hold ids 1, 2, 0, so every PCC name was wrong.
"""
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point

from pyorps.siting.terminals import load_terminals, terminals_from_frame

REPO = Path(__file__).resolve().parents[2]
PCCS = REPO / "case_studies" / "cired2026" / "data" / "input" / "PCCs.shp"


def _frame(ids, crs="EPSG:25832"):
    pts = [Point(100.0 * i + 0.5, 10.0 * i) for i in range(len(ids))]
    return gpd.GeoDataFrame({"id": ids, "geometry": pts}, crs=crs)


def test_labels_follow_the_id_not_the_row():
    frame = _frame([1, 2, 0])
    got = terminals_from_frame(frame, prefix="PCC")
    assert [t.label for t in got] == ["PCC0", "PCC1", "PCC2"]
    by = {t.label: t.xy for t in got}
    assert by["PCC1"] == (0.5, 0.0)          # row 0 holds id 1
    assert by["PCC0"] == (200.5, 20.0)       # row 2 holds id 0


def test_float_ids_from_a_shapefile_read_as_integers():
    got = terminals_from_frame(_frame([1.0, 0.0]), prefix="T")
    assert [t.label for t in got] == ["T0", "T1"]
    assert all(isinstance(t.id, int) for t in got)


@pytest.mark.parametrize("ids,match", [
    ([1, 1], "twice"),
    ([1, None], "no 'id'"),
    ([1, float("nan")], "no 'id'"),
    ([1, 2.5], "not an integer"),
    (["a", " "], "empty"),
])
def test_bad_ids_are_refused(ids, match):
    with pytest.raises(ValueError, match=match):
        terminals_from_frame(_frame(ids))


def test_a_missing_attribute_or_a_non_point_is_refused():
    with pytest.raises(ValueError, match="no 'name' attribute"):
        terminals_from_frame(_frame([0]), id_field="name")
    frame = gpd.GeoDataFrame(
        {"id": [0], "geometry": [LineString([(0, 0), (1, 1)])]},
        crs="EPSG:25832")
    with pytest.raises(ValueError, match="not a point"):
        terminals_from_frame(frame)


def test_reprojection():
    frame = gpd.GeoDataFrame({"id": [0], "geometry": [Point(9.0, 50.0)]},
                             crs="EPSG:4326")
    (t,) = terminals_from_frame(frame, crs="EPSG:25832")
    assert 400_000 < t.xy[0] < 600_000 and 5_500_000 < t.xy[1] < 5_600_000


@pytest.mark.skipif(not PCCS.exists(), reason="case-study input not present")
def test_the_case_study_pccs():
    got = {t.label: t.xy for t in load_terminals(PCCS, prefix="PCC")}
    assert got["PCC0"] == pytest.approx((433748.807, 5584129.062), abs=1e-3)
    assert got["PCC1"] == pytest.approx((437840.502, 5590960.116), abs=1e-3)
    assert got["PCC2"] == pytest.approx((451907.625, 5589719.224), abs=1e-3)
