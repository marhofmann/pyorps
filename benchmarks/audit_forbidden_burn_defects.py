"""Audit ``report_forbidden_burn_defects`` on pyorps example / tutorial windows.

Run::
    .venv/Scripts/python.exe benchmarks/audit_forbidden_burn_defects.py

Prints structured ``ForbiddenBurnReport`` summaries for each scenario that
could be fetched (WFS needs network).
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandapower as pp
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyorps import CostAssumptions, GeoRasterizer, initialize_geo_dataset  # noqa: E402
from pyorps import detect_feature_columns  # noqa: E402
from pyorps.core.types import IMPASSABLE_CELL_COST  # noqa: E402
from pyorps.raster.rasterizer import GeoRasterizer as GR  # noqa: E402
from pyorps.raster.thinness import (  # noqa: E402
    ForbiddenBurnRoutingAssessment,
    ThinForbiddenFeatureWarning,
    recommended_geometry_buffer_m,
)

CRS = "EPSG:25832"

# --- cost tables copied from tutorial / mv_oberrhein / batch examples --------

BW_ALKIS_COSTS = {
    "objektname": {
        "Wohnbaufläche": 65535,
        "Industrie- und Gewerbefläche": 65535,
        "Fläche besonderer funktionaler Prägung": 65535,
        "Tagebau/Grube/Steinbruch": 65535,
        "Friedhof": 65535,
        "Halde": 65535,
        "Sumpf": 65535,
        "Flugverkehr": 65535,
        "Straßenverkehr": 178,
        "Sport-, Freizeit- und Erholungsfläche": 107,
        "Weg": 97,
        "Landwirtschaft": 285,
        "Wald": 365,
        "Fließgewässer": 155,
        "Gehölz": 365,
        "Fläche gemischter Nutzung": 107,
        "Platz": 152,
        "Unland/Vegetationslose Fläche": 92,
        "Stehendes Gewässer": 155,
        "Bahnverkehr": 415,
    }
}

LUBW = "https://geodienste-umwelt.hessen.de/arcgis/services/inspire"
HESSEN_ALKIS_WFS = {
    "url": "https://www.gds.hessen.de/wfs2/aaa-suite/cgi-bin/alkis/vereinf/wfs",
    "layer": "ave_Nutzung",
}


@dataclass
class Scenario:
    name: str
    wfs: dict
    bbox: tuple[float, float, float, float]
    costs: dict
    resolution_m: float = 1.0
    geometry_buffer_m: float = 0.0
    feature_column: str | None = None


def _bbox_polygon(bbox):
    return box(*bbox)


def _forbidden_geometries(rasterizer: GeoRasterizer, field_name: str = "cost"):
    data = rasterizer.base_data.copy()
    if field_name == "cost":
        rasterizer.cost_manager.apply_to_geodataframe(data)
    data[field_name] = data[field_name].fillna(IMPASSABLE_CELL_COST).round().astype("uint16")
    buffered = data.sort_values(by=field_name, ascending=True, kind="stable")
    forbidden = GR._forbidden_rows(buffered[field_name].to_numpy())
    return buffered["geometry"].to_numpy()[forbidden]


def audit(scenario: Scenario) -> dict:
    bbox_poly = _bbox_polygon(scenario.bbox)
    ds = initialize_geo_dataset(scenario.wfs, bbox=scenario.bbox)
    ds.load_data()
    n_features = len(ds.data)
    costs = CostAssumptions(scenario.costs)
    rz = GeoRasterizer(ds, costs, bbox_poly)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ThinForbiddenFeatureWarning)
        warnings.simplefilter("always", ForbiddenBurnRoutingAssessment)
        rz.rasterize(
            resolution_in_m=scenario.resolution_m,
            bounding_box=bbox_poly,
            geometry_buffer_m=scenario.geometry_buffer_m,
            on_thin_features="warn",
        )
    warn_msgs = [str(w.message) for w in caught
                 if issubclass(w.category, ThinForbiddenFeatureWarning)]
    assessment_msgs = [str(w.message) for w in caught
                       if issubclass(w.category, ForbiddenBurnRoutingAssessment)]

    forbidden_geoms = _forbidden_geometries(rz)
    passable = rz.raster != IMPASSABLE_CELL_COST
    report = rz.last_forbidden_burn_report
    if report is None:
        from pyorps.raster.thinness import detect_forbidden_burn_defects  # noqa: WPS433
        report = detect_forbidden_burn_defects(
            forbidden_geoms,
            rz.raster.shape,
            rz.transform,
            resolution_in_m=scenario.resolution_m,
            passable=passable,
        )
    return {
        "scenario": scenario.name,
        "features_total": n_features,
        "forbidden_features": int(forbidden_geoms.size),
        "raster_shape": tuple(rz.raster.shape),
        "forbidden_cells": int((rz.raster == IMPASSABLE_CELL_COST).sum()),
        "passable_cells": int(passable.sum()),
        "report": report,
        "warned": bool(warn_msgs),
        "warning_preview": warn_msgs[0][:240] + "..." if warn_msgs else "",
        "assessment_preview": (
            assessment_msgs[0] if assessment_msgs
            else (rz.last_forbidden_burn_report.assessment_message()
                  if rz.last_forbidden_burn_report else "")),
    }


def _mv_oberrhein_net_bbox(buffer_m: float = 1000.0):
    net = pp.networks.mv_oberrhein(
        scenario="generation", separation_by_sub=True, include_substations=True)[1]
    gs = net.bus.geo.geojson.as_geoseries.to_crs(CRS)
    minx, miny, maxx, maxy = gs.total_bounds
    return (minx - buffer_m, miny - buffer_m, maxx + buffer_m, maxy + buffer_m)


def _hessen_batch_costs():
    sys.path.insert(0, str(ROOT / "examples"))
    from batch_route_planning_pandapower import LAND_USE_COSTS  # noqa: WPS433
    return LAND_USE_COSTS


def _small_raster_bbox():
    import rasterio
    path = ROOT / "examples/data/raster/small_raster.tiff"
    with rasterio.open(path) as src:
        b = src.bounds
    return (b.left, b.bottom, b.right, b.top)


def _print_result(res: dict) -> None:
    r = res["report"]
    print(f"\n{'=' * 72}")
    print(res["scenario"])
    print(f"{'=' * 72}")
    print(f"  vector features (all classes): {res['features_total']:,}")
    print(f"  forbidden-class features:      {res['forbidden_features']:,}")
    print(f"  raster shape:                  {res['raster_shape']}")
    print(f"  forbidden cells:               {res['forbidden_cells']:,}")
    print(f"  passable cells:                {res['passable_cells']:,}")
    print(f"  report.ok:                     {r.ok}")
    print(f"  vanished (burned 0 cells):       {r.vanished.size}")
    print(f"  partially_vanished (holed):      {r.partially_vanished.size}")
    print(f"  fragmented (corner-only):        {r.fragmented.size}")
    print(f"  at_risk (thin but OK now):       {r.at_risk.size}")
    if r.advice is not None and r.advice.summary():
        print(f"  resolution advice:             {r.advice.summary()}")
    if r.summary():
        print(f"  summary: {r.summary()}")
    assessment = r.assessment_message()
    if assessment:
        print(f"\n  routing assessment:\n{assessment}")
    verdict = r.assess()
    print(f"  routing severity:              {verdict.severity.value.upper()}")
    print(f"  routing suitable:              {verdict.routing_suitable}")
    if res["warned"]:
        print(f"  WARNING emitted: yes")
        print(f"  warning preview: {res['warning_preview']}")
    else:
        print(f"  WARNING emitted: no")


def main():
    hessen_costs = _hessen_batch_costs()
    scenarios: list[Scenario] = [
        Scenario(
            name="Tutorial — MV Oberrhein 5×5 km (BW ALKIS, 1 m)",
            wfs={
                "url": "https://owsproxy.lgl-bw.de/owsproxy/wfs/WFS_LGL-BW_ALKIS?version=2.0.0",
                "layer": "Tatsächliche Nutzung",
            },
            bbox=(410000, 5358000, 415000, 5363000),
            costs=BW_ALKIS_COSTS,
        ),
    ]
    try:
        scenarios.append(Scenario(
            name="Case study — MV Oberrhein net bbox + 1 km buffer (BW ALKIS, 1 m)",
            wfs={
                "url": "https://owsproxy.lgl-bw.de/owsproxy/wfs/WFS_LGL-BW_ALKIS?version=2.0.0",
                "layer": "Tatsächliche Nutzung",
            },
            bbox=_mv_oberrhein_net_bbox(),
            costs=BW_ALKIS_COSTS,
        ))
    except Exception as exc:
        print(f"Skipping MV-Oberrhein net bbox scenario: {exc}")
    try:
        scenarios.append(Scenario(
            name="Examples — small_raster.tiff window (Hessen ALKIS, 1 m)",
            wfs=HESSEN_ALKIS_WFS,
            bbox=_small_raster_bbox(),
            costs=hessen_costs,
            geometry_buffer_m=recommended_geometry_buffer_m(1.0),
        ))
    except Exception as exc:
        print(f"Skipping small_raster scenario: {exc}")
    scenarios.append(Scenario(
        name="Batch example-style — 5×5 km + 1 m buffer (Hessen ALKIS)",
        wfs=HESSEN_ALKIS_WFS,
        bbox=(476000, 5568000, 481000, 5573000),
        costs=hessen_costs,
        geometry_buffer_m=1.0,
    ))

    results = []
    for sc in scenarios:
        try:
            results.append(audit(sc))
        except Exception as exc:
            import traceback
            print(f"\n{'=' * 72}")
            print(f"{sc.name} — FAILED: {exc}")
            traceback.print_exc()
    for res in results:
        _print_result(res)


if __name__ == "__main__":
    main()
