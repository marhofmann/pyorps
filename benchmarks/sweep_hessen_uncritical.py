"""Sweep resolution / repair options until Hessen audits are uncritical."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import rasterio
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pyorps import CostAssumptions, GeoRasterizer, initialize_geo_dataset  # noqa: E402
from pyorps.core.types import IMPASSABLE_CELL_COST  # noqa: E402
from pyorps.raster.thinness import (  # noqa: E402
    ForbiddenBurnRoutingAssessment,
    ForbiddenBurnSeverity,
    ThinForbiddenFeatureWarning,
)

HESSEN_ALKIS_WFS = {
    "url": "https://www.gds.hessen.de/wfs2/aaa-suite/cgi-bin/alkis/vereinf/wfs",
    "layer": "ave_Nutzung",
}


def _hessen_costs():
    sys.path.insert(0, str(ROOT / "examples"))
    from batch_route_planning_pandapower import LAND_USE_COSTS  # noqa: WPS433
    return LAND_USE_COSTS


def _small_raster_bbox():
    path = ROOT / "examples/data/raster/small_raster.tiff"
    with rasterio.open(path) as src:
        b = src.bounds
    return (b.left, b.bottom, b.right, b.top)


def _run(name, bbox, *, resolution_m, geometry_buffer_m=0.0,
         widen_thin_forbidden=False, all_touched=False):
    ds = initialize_geo_dataset(HESSEN_ALKIS_WFS, bbox=bbox)
    ds.load_data()
    costs = CostAssumptions(_hessen_costs())
    bbox_poly = box(*bbox)
    rz = GeoRasterizer(ds, costs, bbox_poly)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ThinForbiddenFeatureWarning)
        warnings.simplefilter("ignore", ForbiddenBurnRoutingAssessment)
        rz.rasterize(
            resolution_in_m=resolution_m,
            bounding_box=bbox_poly,
            geometry_buffer_m=geometry_buffer_m,
            widen_thin_forbidden=widen_thin_forbidden,
            all_touched=all_touched,
            on_thin_features="warn",
        )
    rep = rz.last_forbidden_burn_report
    if rep is None:
        return {"name": name, "severity": "n/a", "cells": 0}
    sev = rep.assess().severity
    return {
        "name": name,
        "resolution_m": resolution_m,
        "buffer_m": geometry_buffer_m,
        "widen": widen_thin_forbidden,
        "all_touched": all_touched,
        "severity": sev.value,
        "vanished": int(rep.vanished.size),
        "partial": int(rep.partially_vanished.size),
        "frag": int(rep.fragmented.size),
        "at_risk": int(rep.at_risk.size),
        "cells": int(rz.raster.size),
        "shape": tuple(rz.raster.shape),
    }


def main():
    scenarios = [
        ("small_raster window", _small_raster_bbox(), 0.0),
        ("example 5x5 + 1m buffer", (476000, 5568000, 481000, 5573000), 1.0),
    ]
    configs = [
        dict(resolution_m=1.0),
        dict(resolution_m=1.0, geometry_buffer_m=1.0),
        dict(resolution_m=1.0, widen_thin_forbidden=True),
        dict(resolution_m=0.5),
        dict(resolution_m=0.5, geometry_buffer_m=1.0),
        dict(resolution_m=0.25),
    ]
    print(f"{'scenario':<28} {'res':>5} {'buf':>4} {'widen':>5} | "
          f"{'sev':<10} v/p/f/r | shape")
    print("-" * 90)
    for sname, bbox, buf in scenarios:
        for cfg in configs:
            cfg = dict(cfg)
            if "geometry_buffer_m" not in cfg:
                cfg["geometry_buffer_m"] = buf
            try:
                r = _run(sname, bbox, **cfg)
            except Exception as exc:
                print(f"{sname:<28} {cfg.get('resolution_m', '?'):>5} "
                      f"ERR {exc}")
                continue
            uncritical = r["severity"] in ("none", "low", "moderate")
            mark = "OK" if uncritical else "!!"
            print(
                f"{sname:<28} {r['resolution_m']:>5} {r['buffer_m']:>4.0f} "
                f"{str(r['widen']):>5} | {r['severity']:<10} "
                f"{r['vanished']}/{r['partial']}/{r['frag']}/{r['at_risk']} "
                f"| {r['shape']} {mark}"
            )


if __name__ == "__main__":
    main()
