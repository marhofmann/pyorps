"""
Evaluate ArcGIS Distance Accumulation results against the analytic
references, next to the pyorps block-FIM column (eikonal plan §6.4).

Reads (produced by generate_cases.py + the manual ArcGIS runs per
PROCEDURE.md):
    cases/cases.json, fim/<name>_fim.tif,
    results_arcgis/<name>_arcgis.tif   (any subset; missing = pending)

Writes comparison.json and prints a markdown table. Error metrics:
    - uniform / radial: relative error field-wide for r >= 5 cells
      from the source (L-inf and mean)
    - snell: relative error on a probe grid (every 4th cell) plus the
      benchmark target point
    - barrier: signed relative error at the hand-derived probe points
    - smooth_random / real_raster (no closed form): the ArcGIS field is
      cross-compared against our FIM field (relative difference stats +
      coverage mismatches); the FIM row is the reference by definition

Usage:
    .venv/Scripts/python.exe benchmarks/arcgis_comparison/compare_results.py
"""

import json
from pathlib import Path

import numpy as np
import rasterio

import sys
sys.path.insert(0, str(Path(__file__).parent))
from cases import ALL_CASES  # noqa: E402

HERE = Path(__file__).parent
NODATA = -9999.0


def load_field(path):
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float64)
        nodata = src.nodata
    if nodata is not None:
        data[data == nodata] = np.nan
    return data


def field_metrics(field, case):
    """Relative-error metrics for a solver field vs the analytic form."""
    name = case["name"]
    rows, cols = case["raster"].shape
    sr, sc = case["source"]

    if case.get("analytic") is None:      # barrier: probes only
        out = {}
        for pname, probe in case["probes"].items():
            got = field[probe["row"], probe["col"]]
            ref = probe["analytic"]
            out[f"probe_{pname}_err_pct"] = (
                float((got - ref) / ref * 100) if np.isfinite(got)
                else None)
        return out

    rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    if name == "snell":
        probe = np.zeros((rows, cols), dtype=bool)
        probe[::4, ::4] = True
    else:
        probe = np.ones((rows, cols), dtype=bool)
    r_cells = np.hypot(rr - sr, cc - sc)
    probe &= (r_cells >= 5.0) & np.isfinite(field)

    ref = np.asarray(case["analytic"](rr[probe], cc[probe]),
                     dtype=np.float64)
    rel = (field[probe] - ref) / ref
    return dict(
        n_cells=int(probe.sum()),
        linf_rel_pct=float(np.max(np.abs(rel)) * 100),
        mean_rel_pct=float(np.mean(np.abs(rel)) * 100),
        bias_pct=float(np.mean(rel) * 100),
    )


def cross_metrics(field, reference, case):
    """ArcGIS-vs-FIM cross-comparison for cases without a closed form.

    Relative difference over cells >= 5 cells from the source where
    both fields are finite, plus coverage mismatches (finite in exactly
    one field — differing barrier/NoData semantics would show here).
    """
    rows, cols = case["raster"].shape
    sr, sc = case["source"]
    rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    far = np.hypot(rr - sr, cc - sc) >= 5.0
    both = far & np.isfinite(field) & np.isfinite(reference)
    only_one = far & (np.isfinite(field) ^ np.isfinite(reference))
    rel = (field[both] - reference[both]) / reference[both]
    return dict(
        n_cells=int(both.sum()),
        coverage_mismatch_cells=int(only_one.sum()),
        linf_rel_pct=float(np.max(np.abs(rel)) * 100),
        mean_rel_pct=float(np.mean(np.abs(rel)) * 100),
        bias_pct=float(np.mean(rel) * 100),
    )


def main():
    meta_path = HERE / "cases" / "cases.json"
    if not meta_path.exists():
        raise SystemExit("cases/cases.json missing — run "
                         "generate_cases.py first")

    results = {}
    lines = ["| case | solver | L-inf rel | mean rel | bias |",
             "|---|---|---|---|---|"]
    for factory in ALL_CASES:
        try:
            case = factory()
        except FileNotFoundError as exc:
            print(f"SKIPPED {factory.__name__}: missing input {exc}")
            continue
        name = case["name"]
        results[name] = {}
        fim_field = load_field(HERE / "fim" / f"{name}_fim.tif")
        cross_ref = case.get("reference") == "fim"
        for solver, path in [
            ("fim", HERE / "fim" / f"{name}_fim.tif"),
            ("arcgis", HERE / "results_arcgis" / f"{name}_arcgis.tif"),
        ]:
            field = load_field(path)
            if field is None:
                results[name][solver] = "pending"
                lines.append(f"| {name} | {solver} | *pending* | | |")
                continue
            if field.shape != case["raster"].shape:
                results[name][solver] = (
                    f"SHAPE MISMATCH {field.shape} vs "
                    f"{case['raster'].shape} — re-export without "
                    f"resampling")
                lines.append(f"| {name} | {solver} | shape mismatch | | |")
                continue
            if cross_ref:
                if solver == "fim":
                    results[name][solver] = "reference"
                    lines.append(f"| {name} | fim | *reference* | | |")
                    continue
                if fim_field is None:
                    results[name][solver] = (
                        "FIM reference field missing — re-run "
                        "generate_cases.py on a GPU machine")
                    lines.append(f"| {name} | {solver} | "
                                 f"no FIM reference | | |")
                    continue
                m = cross_metrics(field, fim_field, case)
                m["referee"] = "fim field (no closed form)"
            else:
                m = field_metrics(field, case)
            results[name][solver] = m
            if "linf_rel_pct" in m:
                lines.append(
                    f"| {name} | {solver} | {m['linf_rel_pct']:.3f}% "
                    f"| {m['mean_rel_pct']:.3f}% | {m['bias_pct']:+.3f}% |")
            else:
                probes = ", ".join(
                    f"{k.replace('probe_', '').replace('_err_pct', '')}: "
                    + (f"{v:+.2f}%" if v is not None else "unreached")
                    for k, v in m.items())
                lines.append(f"| {name} | {solver} | {probes} | | |")

    table = "\n".join(lines)
    print(table)
    (HERE / "comparison.json").write_text(json.dumps(results, indent=1))
    print(f"\n-> {HERE / 'comparison.json'}")


if __name__ == "__main__":
    main()
