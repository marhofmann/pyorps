"""R1: a measured runtime model for the trench-sharing collector (plan 2026-10-05-runtime-and-pruning-study.md).

Runs the dense raster collector (``pyorps.collector.raster.RasterCollector``, one core) on crops of the
real priced 1 m cost raster with stand-in turbine positions, and records per job: state counts (labels,
merge pair-ops, junction arcs, drains), seconds in drains, seconds in node operations, peak memory.
Then it fits ``node_ops_s = a*ops + b*ops*N + c*labels*N`` and ``drain_s`` per call, and checks the fit
on held-out jobs (a window size and an n not used in the fit).

Safety, in this order:
* ``--plan`` lists the jobs and exits; it builds nothing.
* Timings need an idle machine (system CPU < 15 % over ``--idle-seconds``). Otherwise the script refuses,
  or with ``--allow-busy`` runs and keeps COUNTS only (no seconds).
* One core; BLAS pinned to one thread; one job at a time; resident-memory guard ``--max-gb`` (default 4)
  and a per-job limit ``--limit-s`` (default 1800); an aborted job keeps its partial counts.
* Writes one small JSON per job (a few KB) into ``results/runtime_study/``; finished jobs are skipped on
  a re-run, so the session can be stopped and resumed. No raster or binary is written.

Usage::

    .venv/Scripts/python.exe case_studies/runkel_free_siting/scripts/runtime_study/r1_measure.py --plan
    .venv/Scripts/python.exe case_studies/runkel_free_siting/scripts/runtime_study/r1_measure.py [--groups cells,n,mp,level,engineA,holdout]
    .venv/Scripts/python.exe case_studies/runkel_free_siting/scripts/runtime_study/r1_measure.py --fit-only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402  (pins the BLAS threads before numpy)
import numpy as np  # noqa: E402

# (id, group, engine, cells, n, m_max, p_max, u_kv)
JOBS: list[tuple] = []


def _add(group, engine, cells, n, m, p, kv=33.0):
    JOBS.append((f"{group}_{engine}_n{n}_N{cells}_m{m}_p{p}_{int(kv)}kV", group, engine, cells, n, m, p, kv))


for N in (10_000, 100_000, 1_000_000):
    _add("cells", "B", N, 3, 4, 2)
for N in (10_000, 100_000):
    _add("cells", "B", N, 4, 4, 2)
for n in (3, 4, 5, 6):
    _add("n", "B", 10_000, n, 4, 2)
_add("n", "B", 2_000, 7, 4, 2)                       # the headline n; 84 862 labels x 2e3 cells x 8 B = 1.4 GB
for m, p in ((2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2)):
    _add("mp", "B", 10_000, 5, m, p)
for kv in (20.0, 30.0, 33.0):
    _add("level", "B", 10_000, 5, 4, 2, kv)
for n in (4, 5, 6):
    _add("engineA", "A", 10_000, n, 4, 2)
_add("holdout", "B", 30_000, 5, 4, 2)               # not in the fit
_add("holdout", "B", 5_000, 6, 4, 2)                # not in the fit


def run_jobs(args) -> None:
    jobs = [j for j in JOBS if j[1] in args.groups and (not args.only or j[0] in args.only)]
    if args.plan:
        for j in jobs:
            print(*j, sep="  ")
        print(f"{len(jobs)} jobs; each writes one JSON of a few KB; nothing is built or run.")
        return
    head = C.header(args.allow_busy, args.idle_seconds)
    if not head["machine_idle"] and not args.allow_busy:
        print(f"machine busy (CPU load {head['cpu_load_pct_before']} %): timings refused. "
              "Wait for a quiet window, or use --allow-busy to keep counts only.")
        sys.exit(3)
    base = {}
    with C.LoadSampler() as sampler_all:
        for jid, group, engine, cells, n, m, p, kv in jobs:
            out = C.RESULTS / f"r1_{jid}.json"
            if out.exists():
                old = json.loads(out.read_text(encoding="utf-8"))
                if old.get("status") == "ok" and "wall_s" in old and not (old.get("load") or {}).get("contaminated"):
                    print("skip (done)", jid)
                    continue
            if kv not in base:
                base[kv] = C.build_base(kv)
            vals = C.load_window(cells)
            turb = C.pick_turbines(vals, n)
            model = C.model_for(base[kv], n, m_max=m, p_max=p)
            meter = C.Meter(limit_s=args.limit_s, max_gb=args.max_gb, time_on=head["timings_kept"])
            print("run", jid, flush=True)
            with C.LoadSampler() as ls:
                res = C.run_collector(vals, turb, model, engine=engine, k=args.k, meter=meter)
            res.update(id=jid, group=group, u_kv=kv, window_shape=list(vals.shape), turbines=turb,
                       load=ls.summary(), header=head)
            C.save(f"r1_{jid}.json", res)
            print(" ", res["status"], {k_: v for k_, v in res.items() if k_ in ("wall_s", "peak_rss_gb")}, flush=True)
    print("session load:", sampler_all.summary())
    fit()


def _load_results() -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(C.RESULTS.glob("r1_*_*.json"))
            if "fit" not in p.name]


def fit() -> dict | None:
    """Least squares on the finished, uncontaminated jobs of groups cells/n/mp/level/engineA; prediction on holdout."""
    rows = [r for r in _load_results() if r["status"] == "ok" and "wall_s" in r
            and not (r["load"]["contaminated"])]
    train = [r for r in rows if r["group"] != "holdout" and not (r["n"] == 7)]
    hold = [r for r in rows if r["group"] == "holdout" or r["n"] == 7]
    if len(train) < 6:
        print("fit: fewer than 6 usable jobs; nothing fitted")
        return None

    def feats(r):
        ops = r["stats"]["merge_pairs"] + r["junction_arcs"]
        return [ops, ops * r["cells"], r["stats"]["labels"] * r["cells"]]

    X = np.array([feats(r) for r in train], dtype=float)
    y = np.array([r["node_ops_s"] for r in train])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    res = {"node_ops_model": "node_ops_s = a*ops + b*ops*N + c*labels*N", "a_s_per_op": coef[0],
           "b_s_per_op_cell": coef[1], "c_s_per_label_cell": coef[2],
           "fit_max_rel_err": float(np.max(np.abs(pred - y) / y)), "n_train": len(train),
           "coeff_negative": bool((coef < 0).any())}
    d = [(r["cells"], r["drain_s"] / max(1, r["drain_calls"])) for r in train if r["drain_calls"]]
    Nd = np.array([c for c, _ in d], dtype=float)
    td = np.array([t for _, t in d])
    A = np.vstack([np.ones_like(Nd), Nd]).T
    dc, *_ = np.linalg.lstsq(A, td, rcond=None)
    res.update(drain_s_per_call_fixed=dc[0], drain_s_per_call_per_cell=dc[1])
    res["holdout"] = []
    for r in hold:
        ops, opsN, lN = feats(r)
        node = float(np.array([ops, opsN, lN]) @ coef)
        drain = r["drain_calls"] * (dc[0] + dc[1] * r["cells"])
        res["holdout"].append({
            "id": r["id"], "measured_s": r["wall_s"], "predicted_s": node + drain,
            "rel_err": (node + drain - r["wall_s"]) / r["wall_s"]})
    C.save("r1_fit.json", res)
    print(json.dumps(res, indent=1, default=float))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", default="cells,n,mp,level,engineA,holdout")
    ap.add_argument("--only", default="", help="comma-separated job ids")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--fit-only", action="store_true")
    ap.add_argument("--allow-busy", action="store_true", help="run on a busy machine, keep counts, drop seconds")
    ap.add_argument("--idle-seconds", type=float, default=10.0)
    ap.add_argument("--max-gb", type=float, default=4.0)
    ap.add_argument("--limit-s", type=float, default=1800.0)
    ap.add_argument("--k", type=int, default=3, help="neighbourhood k (the case study uses 3)")
    args = ap.parse_args()
    args.groups = set(args.groups.split(","))
    args.only = set(filter(None, args.only.split(",")))
    if args.fit_only:
        fit()
    else:
        run_jobs(args)


if __name__ == "__main__":
    main()
