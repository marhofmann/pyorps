"""Score a benchmark run: canonical cost, accuracy, runtime summaries.

Every version reports ``total_cost`` in its own units -- the metric changed
over the project's life (cells vs metres, uint16 vs float weights), and the
scikit-image prototype uses a different edge model entirely. Comparing those
numbers directly would be comparing yardsticks, not routes.

So this module ignores what each engine reported and re-evaluates every route
geometry under ONE model, the current pyorps one:

    step cost = (c[from] + c[to] + sum c[intermediates]) * ||step|| / (2 + n_inter)

That is the only number in the report that is comparable across engines. Each
engine's self-reported cost is kept alongside it, so the divergence itself is
visible.

    python benchmarks/history/score.py --run <run_dir>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402

FORBIDDEN = 65535


# --------------------------------------------------------------------------
def load_window():
    meta = json.loads(C.WINDOW_META.read_text(encoding="utf-8"))
    cost = np.load(C.WINDOW_COST)
    from affine import Affine
    return cost, Affine(*meta["transform"]), meta


def make_intermediates():
    """Return step -> intermediate offsets, using pyorps' own definition."""
    sys.path.insert(0, str(C.REPO_ROOT))
    from pyorps.utils.traversal import intermediate_steps_numba
    cache: dict[tuple[int, int], np.ndarray] = {}

    def inter(dr: int, dc: int) -> np.ndarray:
        key = (dr, dc)
        if key not in cache:
            arr = np.asarray(intermediate_steps_numba(np.int8(dr), np.int8(dc)))
            cache[key] = arr.reshape(-1, 2)
        return cache[key]

    return inter


def evaluate(wkt: str, cost: np.ndarray, transform, inter) -> dict:
    """Re-evaluate one route under the canonical model."""
    from shapely import wkt as shapely_wkt

    geom = shapely_wkt.loads(wkt)
    xs, ys = np.asarray(geom.coords.xy[0]), np.asarray(geom.coords.xy[1])
    inv = ~transform
    cols, rows = inv * (xs, ys)
    cols = np.asarray(cols, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.float64)

    # Which point of the pixel does this version write?
    #   v0.3.0 and later: the CENTRE  -> fractional part ~0.5
    #   v0.1.0 .. v0.2.3: the CORNER  -> fractional part ~0.0
    # Flooring a corner-convention coordinate puts every vertex exactly on a
    # cell boundary, where floating-point noise picks the cell arbitrarily.
    # That alone fabricated ~30 % extra steps and hundreds of phantom
    # forbidden-cell crossings for the older versions. Re-centre each vertex
    # inside its own cell first, then floor safely away from the boundary.
    fx = float(np.median(np.mod(cols, 1.0)))
    fy = float(np.median(np.mod(rows, 1.0)))
    shift_x, shift_y = 0.5 - fx, 0.5 - fy
    rows_i = np.floor(rows + shift_y).astype(np.int64)
    cols_i = np.floor(cols + shift_x).astype(np.int64)
    rows, cols = rows_i, cols_i
    pixel_anchor = ("centre" if abs(fx - 0.5) < 0.25 and abs(fy - 0.5) < 0.25
                    else "corner")

    nrow, ncol = cost.shape
    if (rows < 0).any() or (rows >= nrow).any() or \
       (cols < 0).any() or (cols >= ncol).any():
        return dict(canonical_status="outside_window")

    total = 0.0
    length = 0.0
    n_steps = 0
    forbidden_cells = 0
    tunnels = 0          # steps crossing a forbidden cell between clean ends
    max_cell = 0
    cell_size = abs(transform.a)

    r, c = int(rows[0]), int(cols[0])
    max_cell = max(max_cell, int(cost[r, c]))
    if cost[r, c] == FORBIDDEN:
        forbidden_cells += 1

    for i in range(1, len(rows)):
        dr = int(rows[i]) - r
        dc = int(cols[i]) - c
        if dr == 0 and dc == 0:
            continue
        g = math.gcd(abs(dr), abs(dc))
        ur, uc = dr // g, dc // g
        if max(abs(ur), abs(uc)) > 3:
            # Not decomposable into a neighbourhood step: the geometry was
            # simplified past the point where the traversed cells can be
            # recovered. Reported, never silently approximated.
            return dict(canonical_status="unresolvable_step",
                        canonical_bad_step=[ur, uc])
        dist = math.hypot(ur, uc) * cell_size
        offs = inter(ur, uc)
        for _ in range(g):
            nr, nc = r + ur, c + uc
            acc = float(cost[r, c]) + float(cost[nr, nc])
            hit_forbidden = False
            for k in range(offs.shape[0]):
                ir, ic = r + int(offs[k, 0]), c + int(offs[k, 1])
                v = float(cost[ir, ic])
                acc += v
                if v == FORBIDDEN:
                    hit_forbidden = True
            total += acc * dist / (2.0 + offs.shape[0])
            length += dist
            n_steps += 1
            if (hit_forbidden and cost[r, c] != FORBIDDEN
                    and cost[nr, nc] != FORBIDDEN):
                tunnels += 1
            r, c = nr, nc
            v = int(cost[r, c])
            max_cell = max(max_cell, v)
            if v == FORBIDDEN:
                forbidden_cells += 1

    return dict(canonical_status="ok", canonical_cost=total,
                canonical_length_m=length, canonical_steps=n_steps,
                canonical_max_cell=max_cell,
                canonical_forbidden_cells=forbidden_cells,
                canonical_barrier_tunnels=tunnels,
                pixel_anchor=pixel_anchor,
                pixel_frac_x=round(fx, 4), pixel_frac_y=round(fy, 4))


# --------------------------------------------------------------------------
def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    run_dir = args.run.resolve()

    records = [json.loads(ln) for ln in
               (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
               if ln.strip()]
    print(f"{len(records)} records from {run_dir}")

    cost, transform, meta = load_window()
    inter = make_intermediates()
    print(f"canonical window {cost.shape}, cell {abs(transform.a):.2f} m")

    # ---------------------------------------------------------------- paths
    path_rows = []
    cache: dict[str, dict] = {}
    for rec in records:
        if rec.get("status") != "ok" or rec.get("kind") != "route":
            continue
        # each execution mode writes its own route file
        res_path = (run_dir / "routes"
                    / f"{rec['job_id']}__{rec['exec_mode']}.json")
        if not res_path.exists():
            # runs made before route files were split by exec_mode
            res_path = run_dir / "routes" / f"{rec['job_id']}.json"
        if not res_path.exists():
            continue
        res = json.loads(res_path.read_text(encoding="utf-8"))
        for p in res.get("paths", []) or []:
            wkt = p.get("wkt")
            row = dict(
                job_id=rec["job_id"], track=rec["track"], engine=rec["engine"],
                version=rec["version"], neighborhood=rec["neighborhood"],
                graph_api=rec.get("graph_api") or p.get("graph_api"),
                exec_mode=rec["exec_mode"], repeat=rec["repeat"],
                bus=p.get("bus"),
                reported_cost=p.get("total_cost"),
                reported_length=p.get("total_length"),
                geom_length_m=p.get("geom_length_m"),
                n_vertices=p.get("n_vertices"),
            )
            if wkt:
                key = f"{hash(wkt)}"
                if key not in cache:
                    cache[key] = evaluate(wkt, cost, transform, inter)
                row.update(cache[key])
            else:
                row["canonical_status"] = "no_geometry"
            path_rows.append(row)

    print(f"{len(path_rows)} routes scored "
          f"({len(cache)} distinct geometries)")

    # reference optimum per bus: the cheapest canonical cost anyone achieved
    best = defaultdict(lambda: math.inf)
    for r in path_rows:
        if r.get("canonical_status") == "ok":
            best[r["bus"]] = min(best[r["bus"]], r["canonical_cost"])
    for r in path_rows:
        if r.get("canonical_status") == "ok" and best[r["bus"]] < math.inf:
            r["excess_pct"] = 100.0 * (r["canonical_cost"] / best[r["bus"]] - 1.0)

    # ------------------------------------------------------------- write out
    import csv

    if path_rows:
        cols = sorted({k for r in path_rows for k in r})
        with open(run_dir / "summary_paths.csv", "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(path_rows)

    job_cols = sorted({k for r in records for k in r
                       if k not in ("groups", "raster_histogram")})
    with open(run_dir / "summary_jobs.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=job_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)

    # ---------------------------------------------------------------- report
    L = []
    A = L.append
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    A(f"# MV-Oberrhein history benchmark -- {manifest['run_id']}\n")
    A(f"- host: {manifest['host']['node']}, "
      f"{manifest['host']['cpus']} logical CPUs, {manifest['host']['platform']}")
    A(f"- profile `{manifest['profile']}`, repeats {manifest['repeats']}, "
      f"exec {manifest['exec_modes']}")
    A(f"- search window {manifest['window'][0]} x {manifest['window'][1]} "
      f"cells, buffer {manifest['search_space_buffer_m']} m")
    A(f"- elapsed {manifest.get('elapsed_s', 0) / 3600:.2f} h")
    A(f"- backends available: {manifest['backends_available']}")
    A("")
    A("All costs below are **re-evaluated under one canonical model** "
      "(current pyorps edge semantics). Each engine's self-reported cost is "
      "kept in `summary_paths.csv` next to it.\n")

    failed = [r for r in records if r.get("status") != "ok"]
    if failed:
        A(f"## Jobs that did not complete ({len(failed)})\n")
        A("| job | status | detail |")
        A("|---|---|---|")
        for r in failed:
            det = (r.get("error") or "").strip().splitlines()
            A(f"| `{r['job_id']}` | {r.get('status')} | "
              f"{det[-1][:110] if det else ''} |")
        A("")

    # runtime, serial only
    A("## Runtime -- serial (comparable)\n")
    A("| track | version | nbhd | backend | median wall s | min | max | "
      "peak GiB |")
    A("|---|---|---|---|---:|---:|---:|---:|")
    groups = defaultdict(list)
    for r in records:
        if r.get("status") == "ok" and r["exec_mode"] == "serial":
            groups[(r["track"], r["version"], r["neighborhood"],
                    r.get("graph_api") or "default")].append(r)
    for k in sorted(groups):
        rs = groups[k]
        walls = [x["wall_s"] for x in rs]
        peaks = [x.get("peak_rss_gb") for x in rs if x.get("peak_rss_gb")]
        A(f"| {k[0]} | {k[1]} | {k[2] or '-'} | {k[3]} | "
          f"{median(walls):.1f} | {min(walls):.1f} | {max(walls):.1f} | "
          f"{max(peaks) if peaks else float('nan'):.1f} |")
    A("")

    # serial vs parallel throughput
    par = [r for r in records if r["exec_mode"] == "parallel"
           and r.get("status") == "ok"]
    ser = [r for r in records if r["exec_mode"] == "serial"
           and r.get("status") == "ok"]
    if par and ser:
        A("## Serial vs parallel\n")
        phases = manifest.get("phase_elapsed_s") or {}
        if phases.get("serial") and phases.get("parallel"):
            A(f"- **wall clock, serial pass: "
              f"{phases['serial'] / 3600:.2f} h**")
            A(f"- **wall clock, parallel pass: "
              f"{phases['parallel'] / 3600:.2f} h "
              f"({manifest['workers']} workers, "
              f"{manifest['mem_budget_gb']} GiB budget)**")
            A(f"- **throughput speedup: "
              f"{phases['serial'] / phases['parallel']:.2f}x**")
            A("")
        # only jobs that completed in BOTH modes -- otherwise the ratio
        # silently compares different work
        common = ({r["job_id"] for r in ser} & {r["job_id"] for r in par})
        s_wall = sum(r["wall_s"] for r in ser if r["job_id"] in common)
        p_wall = sum(r["wall_s"] for r in par if r["job_id"] in common)
        A(f"- summed per-job time over the {len(common)} jobs that completed "
          f"in both modes: {s_wall / 3600:.2f} h serial, "
          f"{p_wall / 3600:.2f} h parallel")
        if s_wall:
            A(f"- per-job slowdown under contention: {p_wall / s_wall:.2f}x "
              f"(jobs share cores, memory bandwidth and cache)")
        dropped = ({r["job_id"] for r in ser} | {r["job_id"] for r in par}) - common
        if dropped:
            A(f"- {len(dropped)} job(s) succeeded in only one mode and are "
              f"excluded from that ratio")
        A("")
        A("| track | version | nbhd | backend | serial s | parallel s | ratio |")
        A("|---|---|---|---|---:|---:|---:|")
        pg = defaultdict(list)
        for r in par:
            pg[(r["track"], r["version"], r["neighborhood"],
                r.get("graph_api") or "default")].append(r["wall_s"])
        for k in sorted(groups):
            if k not in pg:
                continue
            s = median([x["wall_s"] for x in groups[k]])
            p = median(pg[k])
            A(f"| {k[0]} | {k[1]} | {k[2] or '-'} | {k[3]} | {s:.1f} | "
              f"{p:.1f} | {p / s:.2f} |")
        A("")

    # accuracy
    ok_paths = [r for r in path_rows if r.get("canonical_status") == "ok"]
    if ok_paths:
        A("## Route quality -- canonical cost vs the best route found\n")
        A("| version | nbhd | backend | mean excess % | max excess % | "
          "median length m | forbidden cells | barrier tunnels |")
        A("|---|---|---|---:|---:|---:|---:|---:|")
        ag = defaultdict(list)
        for r in ok_paths:
            ag[(r["version"], r["neighborhood"], r["graph_api"])].append(r)
        for k in sorted(ag, key=lambda x: (str(x[0]), str(x[1]), str(x[2]))):
            rs = ag[k]
            ex = [r["excess_pct"] for r in rs if "excess_pct" in r]
            A(f"| {k[0]} | {k[1]} | {k[2]} | "
              f"{sum(ex) / len(ex):.2f} | {max(ex):.2f} | "
              f"{median([r['canonical_length_m'] for r in rs]):.0f} | "
              f"{sum(r['canonical_forbidden_cells'] for r in rs)} | "
              f"{sum(r['canonical_barrier_tunnels'] for r in rs)} |")
        A("")
        A("`barrier tunnels` counts steps that pass through a maximum-cost "
          "(65535) cell while both endpoints avoid it -- a route crossing a "
          "barrier it never pays full price for. It is the concrete signature "
          "of an edge model that ignores intermediate cells.\n")

        A("### Which point of the pixel each version writes\n")
        A("Detected from the geometry itself, per route. Versions up to "
          "v0.2.3 anchor path vertices at the pixel **corner**; v0.3.0 "
          "onwards at the pixel **centre** -- a half-cell (0.5 m) shift on "
          "this raster. The route is the same either way; only where its "
          "vertices are written moves. The scorer re-centres each vertex in "
          "its own cell before mapping to cells, because flooring a "
          "corner-anchored coordinate lands exactly on a cell boundary and "
          "lets floating-point noise choose the cell.\n")
        A("| version | anchor | median frac x | median frac y |")
        A("|---|---|---:|---:|")
        anch = {}
        for r in ok_paths:
            if r.get("pixel_anchor"):
                anch.setdefault(r["version"], []).append(
                    (r["pixel_anchor"], r["pixel_frac_x"], r["pixel_frac_y"]))
        for v in sorted(anch):
            kinds = {a[0] for a in anch[v]}
            fx = median([a[1] for a in anch[v]])
            fy = median([a[2] for a in anch[v]])
            A(f"| {v} | {'/'.join(sorted(kinds))} | {fx:.3f} | {fy:.3f} |")
        A("")

        A("## Self-reported vs canonical cost\n")
        A("| version | nbhd | backend | median reported | median canonical | "
          "ratio |")
        A("|---|---|---|---:|---:|---:|")
        for k in sorted(ag, key=lambda x: (str(x[0]), str(x[1]), str(x[2]))):
            rs = [r for r in ag[k] if r.get("reported_cost")]
            if not rs:
                continue
            mr = median([r["reported_cost"] for r in rs])
            mc = median([r["canonical_cost"] for r in rs])
            A(f"| {k[0]} | {k[1]} | {k[2]} | {mr:.0f} | {mc:.0f} | "
              f"{mr / mc:.3f} |")
        A("")

    bad = [r for r in path_rows if r.get("canonical_status") not in ("ok", None)]
    if bad:
        A(f"## Routes that could not be scored ({len(bad)})\n")
        seen = defaultdict(int)
        for r in bad:
            seen[(r["version"], r["neighborhood"], r["canonical_status"])] += 1
        for k, n in sorted(seen.items(), key=lambda kv: str(kv[0])):
            A(f"- {k[0]} / {k[1]}: {k[2]} ({n})")
        A("")

    # rasterisation
    ras = [r for r in records if r.get("kind") == "rasterize"
           and r.get("status") == "ok" and r["exec_mode"] == "serial"]
    if ras:
        A("## Rasterisation (vector -> 1 m cost raster)\n")
        A("| version | median total s | vector load s | rasterize s | "
          "checksum |")
        A("|---|---:|---:|---:|---|")
        rg = defaultdict(list)
        for r in ras:
            rg[r["version"]].append(r)
        for v in sorted(rg):
            rs = rg[v]
            A(f"| {v} | {median([x['total_s'] for x in rs]):.2f} | "
              f"{median([x['vector_load_s'] for x in rs]):.2f} | "
              f"{median([x['rasterize_s'] for x in rs]):.2f} | "
              f"`{rs[0]['raster_checksum']}` |")
        A("")
        sums = {r["raster_checksum"] for r in ras}
        A(f"Distinct raster checksums across versions: **{len(sums)}** "
          + ("-- every version rasterises to a bit-identical surface."
             if len(sums) == 1 else
             "-- the cost surface itself changed between versions, so route "
             "costs across those versions are NOT directly comparable.") + "\n")

    (run_dir / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\nwrote:\n  {run_dir / 'REPORT.md'}\n"
          f"  {run_dir / 'summary_paths.csv'}\n"
          f"  {run_dir / 'summary_jobs.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
