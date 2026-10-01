"""End-to-end test for examples/preprocess_mv_connections.py.

Builds a synthetic 3-element manifest + 6-station layer in a metric CRS
(EPSG:25832), runs the preprocessing with the uniform-cost fallback (no cost
raster/vector), and asserts the candidate artifact is correct.
"""

import importlib.util
import json
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point

# Load the example script as a module from its file path (examples/ is not a
# package, so a normal import is not available).
_SCRIPT = Path(__file__).resolve().parents[2] / "examples" / "preprocess_mv_connections.py"
_spec = importlib.util.spec_from_file_location("preprocess_mv_connections", _SCRIPT)
pmc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pmc)

CRS = "EPSG:25832"
K = 4

# Six stations on a coarse grid (~1 km spacing -> distinct raster cells).
_STATIONS = {
    10: (470000.0, 5600000.0),
    11: (471000.0, 5600000.0),
    12: (472000.0, 5600000.0),
    13: (470000.0, 5601000.0),
    14: (471000.0, 5601000.0),
    15: (472000.0, 5601000.0),
}
# Three elements, each placed next to a distinct station whose bus is the default PCC.
_ELEMENTS = [
    ("PV-1", "sgen", "openfield_pv", 3.5, 10, (470100.0, 5600100.0)),
    ("LOAD-1", "load", "large_load", 6.0, 11, (471100.0, 5600050.0)),
    ("HEAT-1", "sgen", "heat_plant", 2.0, 14, (471050.0, 5600950.0)),
]


def _write_fixture(tmp_dir: Path):
    stations = gpd.GeoDataFrame(
        {
            "bus_id": list(_STATIONS),
            "name": [f"ST-{b}" for b in _STATIONS],
            "vn_kv": [20.0] * len(_STATIONS),
        },
        geometry=[Point(xy) for xy in _STATIONS.values()],
        crs=CRS,
    )
    manifest = gpd.GeoDataFrame(
        {
            "element_id": [e[0] for e in _ELEMENTS],
            "kind": [e[1] for e in _ELEMENTS],
            "tech": [e[2] for e in _ELEMENTS],
            "s_mva": [e[3] for e in _ELEMENTS],
            "default_pcc_bus": [e[4] for e in _ELEMENTS],
            "year": [2025] * len(_ELEMENTS),
            "scenario_id": ["base"] * len(_ELEMENTS),
            "seed": [42] * len(_ELEMENTS),
        },
        geometry=[Point(e[5]) for e in _ELEMENTS],
        crs=CRS,
    )
    manifest_path = tmp_dir / "manifest.gpkg"
    stations_path = tmp_dir / "stations.gpkg"
    manifest.to_file(manifest_path, layer=pmc.DEFAULT_MANIFEST_LAYER, driver="GPKG")
    stations.to_file(stations_path, layer=pmc.DEFAULT_STATIONS_LAYER, driver="GPKG")
    return manifest_path, stations_path


def test_preprocess_uniform_fallback(tmp_path):
    manifest_path, stations_path = _write_fixture(tmp_path)
    out_dir = tmp_path / "out"

    gpkg = pmc.preprocess_mv_connections(
        manifest_path=manifest_path,
        stations_path=stations_path,
        k=K,
        out_dir=out_dir,
    )

    # Artifact exists and has rows.
    assert Path(gpkg).exists()
    cand = gpd.read_file(gpkg, layer=pmc.CANDIDATES_LAYER)
    assert len(cand) > 0
    assert cand.crs.to_epsg() == 25832

    # Companion CSV + failures CSV exist; no element should fail (all reachable).
    assert gpkg.with_suffix(".csv").exists()
    failures = gpkg.parent / f"{gpkg.stem}_failures.csv"
    assert failures.exists()
    import pandas as pd
    assert len(pd.read_csv(failures)) == 0

    station_xy = {b: Point(xy) for b, xy in _STATIONS.items()}

    for element_id, _, _, _, default_bus, site in _ELEMENTS:
        rows = cand[cand["element_id"] == element_id].sort_values("rank")
        # Between 1 and k candidates per element.
        assert 1 <= len(rows) <= K

        # Geometry: LineString starting near the element and ending near its station.
        site_pt = Point(site)
        for _, r in rows.iterrows():
            geom = r.geometry
            assert isinstance(geom, LineString)
            start = Point(geom.coords[0])
            end = Point(geom.coords[-1])
            assert start.distance(site_pt) < 30.0
            assert end.distance(station_xy[int(r["target_bus_id"])]) < 30.0

        # Ranking: route_cost non-decreasing with rank, rank is 1..n contiguous.
        costs = rows["route_cost"].tolist()
        assert costs == sorted(costs)
        assert rows["rank"].tolist() == list(range(1, len(rows) + 1))

        # is_default_pcc True exactly for the row whose bus == default_pcc_bus
        # (the default station is among the k nearest by construction).
        assert default_bus in rows["target_bus_id"].tolist()
        dflt = rows[rows["target_bus_id"] == default_bus]
        assert dflt["is_default_pcc"].all()
        assert not rows[rows["target_bus_id"] != default_bus]["is_default_pcc"].any()

        # length_by_category JSON is parseable.
        for blob in rows["length_by_category_json"]:
            assert isinstance(json.loads(blob), dict)


def test_idempotency_skips_recompute(tmp_path):
    manifest_path, stations_path = _write_fixture(tmp_path)
    out_dir = tmp_path / "out"

    gpkg = pmc.preprocess_mv_connections(
        manifest_path=manifest_path, stations_path=stations_path, k=K, out_dir=out_dir,
    )
    mtime = Path(gpkg).stat().st_mtime_ns

    # Second call without force must not rewrite the file.
    gpkg2 = pmc.preprocess_mv_connections(
        manifest_path=manifest_path, stations_path=stations_path, k=K, out_dir=out_dir,
    )
    assert Path(gpkg2) == Path(gpkg)
    assert Path(gpkg2).stat().st_mtime_ns == mtime
