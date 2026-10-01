"""MV-Oberrhein history benchmark: every relevant pyorps version, one case.

Reproduces the CIRED 2025 case study -- connect one PV plant to the
MV-Oberrhein grid, 8 candidate points of common coupling, neighbourhoods
R0..R3 on a 1 m ALKIS cost raster -- against every relevant pyorps version and
against the pre-pyorps scikit-image prototype from the NEIS 2021 paper.

    python benchmarks/history/run_history_benchmark.py --profile standard

Everything is written under ``~/Documents/pyorps_benchmarks`` (override with
PYORPS_BENCH_ROOT), never into the repository.

Tracks
  raster     rasterisation of the ALKIS vector data, per version
  evolution  every version on ITS OWN default backend, R0..R3
             -- "what a user actually got at the time"
  control    every version pinned to one common backend (networkit), R2
             -- isolates algorithmic change from backend change
  backends   HEAD across every installed backend, R2
  baseline   the NEIS 2021 scikit-image prototype

Execution modes
  serial     one job at a time -- the timings that are comparable
  parallel   several jobs at once under a memory budget -- measures throughput
             on this machine, NOT per-job latency; recorded separately and
             never mixed with serial numbers
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config as C          # noqa: E402
import prepare as P         # noqa: E402


# --------------------------------------------------------------------------
def human(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


def probe_backends(python: str) -> dict[str, bool]:
    """Ask the runner interpreter which graph backends can actually import."""
    code = (
        "import json,importlib;"
        "mods={'cython':'pyorps.graph.api.cython_api',"
        "'networkit':'networkit','networkx':'networkx','igraph':'igraph',"
        "'rustworkx':'rustworkx','raster_gpu':'cupy','raster_fim':'cupy'};"
        "out={}\n"
        "for k,m in mods.items():\n"
        "    try:\n"
        "        importlib.import_module(m); out[k]=True\n"
        "    except Exception: out[k]=False\n"
        "print(json.dumps(out))"
    )
    try:
        r = subprocess.run([python, "-c", code], cwd=C.REPO_ROOT,
                           capture_output=True, text=True, timeout=180)
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return {}


def probe_prototype(python: str) -> bool:
    r = subprocess.run([python, "-c", "import skimage"], capture_output=True)
    return r.returncode == 0


# --------------------------------------------------------------------------
PROFILES = {
    "quick": dict(
        versions=["neis2021", "v0.1.0", "v0.2.1", "v0.3.2", "HEAD"],
        neighborhoods=["r1", "r2"], repeats=1,
        tracks=["raster", "evolution", "baseline"]),
    "standard": dict(
        versions=None, neighborhoods=["r0", "r1", "r2"], repeats=2,
        tracks=["raster", "evolution", "control", "backends", "baseline"]),
    "full": dict(
        versions="all", neighborhoods=["r0", "r1", "r2", "r3"], repeats=3,
        tracks=["raster", "evolution", "control", "backends", "baseline"]),
}


def select_versions(spec) -> list[dict]:
    if spec is None:
        return list(C.VERSIONS)
    if spec == "all":
        allv = C.VERSIONS + C.EXTRA_VERSIONS
        order = {v["label"]: i for i, v in enumerate(
            sorted(allv, key=lambda v: (v["date"], v["label"])))}
        return sorted(allv, key=lambda v: order[v["label"]])
    by_label = {v["label"]: v for v in C.VERSIONS + C.EXTRA_VERSIONS}
    missing = [s for s in spec if s not in by_label]
    if missing:
        raise SystemExit(f"unknown version(s): {missing}\n"
                         f"known: {sorted(by_label)}")
    return [by_label[s] for s in spec]


def build_jobs(versions, neighborhoods, repeats, tracks, checkouts,
               backends_available, cells, run_dir, prototype_ok) -> list[dict]:
    jobs: list[dict] = []
    inputs = dict(bbox=str(C.BBOX), alkis=str(C.ALKIS), source=str(C.SOURCE),
                  targets=str(C.TARGETS), ref_raster=str(C.REF_RASTER),
                  window_cost=str(C.WINDOW_COST), window_meta=str(C.WINDOW_META))
    meta = json.loads(C.WINDOW_META.read_text(encoding="utf-8"))

    def add(**kw):
        base = dict(engine="pyorps", kind="route", neighborhood=None,
                    graph_api=None, algorithm="dijkstra", use_gpu=False,
                    inputs=inputs, ignore_max_cost=C.IGNORE_MAX_COST,
                    bus_order=C.CASE_BUS_ORDER, r3_clusters=C.R3_CLUSTERS,
                    cost_assumptions=C.COST_ASSUMPTIONS,
                    search_space_buffer_m=None)
        base.update(kw)
        # Graph-library backends hit HEAD's large-raster guard on this window;
        # raster-direct backends never build an edge list and are unaffected.
        base["allow_large_raster"] = (
            base["graph_api"] in ("networkit", "networkx", "igraph",
                                  "rustworkx"))
        api = base["graph_api"] or "default"
        # an explicit estimate from the caller wins; only routing jobs get the
        # model-derived one
        if "est_mem_gb" not in kw:
            base["est_mem_gb"] = round(C.estimate_mem_gb(
                base["graph_api"] or "networkit", base["neighborhood"] or "r1",
                cells), 2)
        jid = (f"{base['track']}__{base['version']}__{base['kind']}"
               f"__{base['neighborhood'] or '-'}__{api}"
               f"{'_gpu' if base['use_gpu'] else ''}__rep{base['repeat']}")
        base["job_id"] = jid
        base["out_json"] = str(run_dir / "routes" / f"{jid}.json")
        jobs.append(base)

    pyorps_versions = [v for v in versions if v["kind"] != "prototype"]

    for rep in range(repeats):
        # ---- rasterisation
        if "raster" in tracks:
            for v in pyorps_versions:
                add(track="raster", version=v["label"], kind="rasterize",
                    checkout=checkouts[v["label"]]["checkout"], repeat=rep,
                    raster_out=str(run_dir / "tmp_rasters"
                                   / f"{v['label']}_rep{rep}.tiff"),
                    # measured 6.9-7.9 GiB peak: the whole 580 Mcell surface is
                    # held, written, and read back for the checksum
                    est_mem_gb=8.5)

        # ---- each version on its own default backend
        if "evolution" in tracks:
            for v in pyorps_versions:
                for nb in neighborhoods:
                    # size the job by the backend it will ACTUALLY pick, not by
                    # the networkit fallback the estimator assumes
                    add(track="evolution", version=v["label"], neighborhood=nb,
                        checkout=checkouts[v["label"]]["checkout"], repeat=rep,
                        est_mem_gb=round(C.estimate_mem_gb(
                            v.get("default_api", "networkit"), nb, cells), 2))

        # ---- every version pinned to one common backend
        if "control" in tracks and backends_available.get(C.CONTROL_BACKEND):
            for v in pyorps_versions:
                # For v0.1.x the default IS networkit, so a control job at r2
                # would repeat the evolution job byte for byte -- ~200 s each,
                # twice per repeat, twice per exec mode. Skip it and let the
                # report read the evolution row instead.
                redundant = (v.get("default_api") == C.CONTROL_BACKEND
                             and "evolution" in tracks
                             and "r2" in neighborhoods)
                if redundant:
                    continue
                add(track="control", version=v["label"], neighborhood="r2",
                    graph_api=C.CONTROL_BACKEND, repeat=rep,
                    checkout=checkouts[v["label"]]["checkout"])

        # ---- HEAD across all installed backends
        if "backends" in tracks:
            head = next((v for v in versions if v["kind"] == "worktree"), None)
            if head is not None:
                for be in C.HEAD_BACKENDS:
                    if not backends_available.get(be["graph_api"], False):
                        continue
                    add(track="backends", version=head["label"],
                        neighborhood="r2", graph_api=be["graph_api"],
                        use_gpu=be["use_gpu"], repeat=rep,
                        checkout=checkouts[head["label"]]["checkout"])

        # ---- the 2021 prototype
        if "baseline" in tracks and prototype_ok:
            proto = next((v for v in versions if v["kind"] == "prototype"), None)
            if proto is not None:
                for nb in neighborhoods:
                    add(track="baseline", engine="neis2021",
                        version=proto["label"], neighborhood=nb, repeat=rep,
                        checkout=None, graph_api="skimage",
                        pyorps_for_steps=str(C.REPO_ROOT),
                        source_xy=meta["source_xy"],
                        target_xy=meta["target_xy"],
                        max_cost_impassable=False,
                        est_mem_gb=round(1.0 + cells * 24 / 2 ** 30, 2))
    return jobs


# --------------------------------------------------------------------------
def launch(job: dict, python: str, run_dir: Path, exec_mode: str):
    # Each execution mode gets its OWN result file. Sharing one would let the
    # parallel pass overwrite the serial pass's routes, and a failure in the
    # second pass would destroy the first pass's geometry.
    job = dict(job, exec_mode=exec_mode)
    job["out_json"] = str(run_dir / "routes"
                          / f"{job['job_id']}__{exec_mode}.json")
    spec_path = run_dir / "jobs" / f"{job['job_id']}__{exec_mode}.json"
    spec_path.write_text(json.dumps(job, indent=1, default=str), encoding="utf-8")
    worker = "_worker_neis2021.py" if job["engine"] == "neis2021" \
        else "_worker_pyorps.py"
    logf = open(run_dir / "logs" / f"{job['job_id']}__{exec_mode}.log",
                "w", encoding="utf-8")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen([python, str(HERE / worker), str(spec_path)],
                            cwd=str(C.REPO_ROOT), stdout=logf,
                            stderr=subprocess.STDOUT, env=env)
    return job, proc, logf, time.perf_counter()


def collect(job, proc, logf, t0, run_dir, exec_mode, records_fh,
            timed_out=False):
    logf.close()
    wall = time.perf_counter() - t0
    out = Path(job["out_json"])
    rec = dict(job_id=job["job_id"], track=job["track"], engine=job["engine"],
               version=job["version"], kind=job["kind"],
               neighborhood=job["neighborhood"], graph_api=job["graph_api"],
               use_gpu=job["use_gpu"], repeat=job["repeat"],
               exec_mode=exec_mode, wall_s=round(wall, 3),
               returncode=proc.returncode, est_mem_gb=job["est_mem_gb"])
    if out.exists():
        try:
            res = json.loads(out.read_text(encoding="utf-8"))
        except Exception as exc:
            res = {"status": "unreadable_result", "error": repr(exc)}
        for k, v in res.items():
            if k != "paths":
                rec[k] = v
        rec["n_paths"] = len(res.get("paths", []) or [])
    else:
        # a killed process still reports a returncode, so "did it time
        # out" has to be carried in, not inferred from the exit status
        rec["status"] = ("timeout" if timed_out
                         else f"crashed(rc={proc.returncode})")
        rec["log"] = str(run_dir / "logs"
                         / f"{job['job_id']}__{exec_mode}.log")
    records_fh.write(json.dumps(rec, default=str) + "\n")
    records_fh.flush()
    return rec


def result_ok(job, run_dir: Path, mode: str) -> float | None:
    """Peak RSS if this job already finished cleanly in this mode, else None.

    Used both by --resume and by the measured-peak re-estimation, so a resumed
    run schedules the parallel pass from real numbers rather than the model.
    """
    p = run_dir / "routes" / f"{job['job_id']}__{mode}.json"
    if not p.exists():
        return None
    try:
        res = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if res.get("status") != "ok":
        return None
    return res.get("peak_rss_gb") or 0.0


def status_line(rec) -> str:
    bits = [f"{rec['status']:>12s}", f"{rec['wall_s']:8.1f}s",
            f"{rec['track']:<9s}", f"{rec['version']:<9s}",
            f"{rec['kind']:<9s}", f"{str(rec['neighborhood']):<4s}",
            f"{str(rec['graph_api'] or 'default'):<22s}",
            f"rep{rec['repeat']}"]
    if rec.get("peak_rss_gb"):
        bits.append(f"peak {rec['peak_rss_gb']:.1f}G")
    return "  ".join(bits)


def run_serial(jobs, python, run_dir, records_fh, timeout):
    done = []
    for i, job in enumerate(jobs, 1):
        print(f"\n[{i}/{len(jobs)}] {job['job_id']}", flush=True)
        job, proc, logf, t0 = launch(job, python, run_dir, "serial")
        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            timed_out = True
            print(f"  TIMEOUT after {timeout}s -- killed", flush=True)
        rec = collect(job, proc, logf, t0, run_dir, "serial", records_fh,
                      timed_out=timed_out)
        print("  " + status_line(rec), flush=True)
        done.append(rec)
    return done


def run_parallel(jobs, python, run_dir, records_fh, timeout, workers,
                 mem_budget_gb):
    """Admit jobs while both the worker slot and the memory budget allow it.

    A job whose own estimate exceeds the budget still runs -- alone. That keeps
    the heavy R2/R3 graph-library jobs from being silently dropped.
    """
    pending = list(jobs)
    running = []          # (job, proc, logf, t0)
    done = []
    total = len(jobs)

    while pending or running:
        used = sum(j["est_mem_gb"] for j, _, _, _ in running)
        while pending and len(running) < workers:
            nxt = pending[0]
            fits = (used + nxt["est_mem_gb"] <= mem_budget_gb) or not running
            if not fits:
                break
            pending.pop(0)
            nxt, proc, logf, t0 = launch(nxt, python, run_dir, "parallel")
            running.append((nxt, proc, logf, t0))
            used += nxt["est_mem_gb"]
            print(f"  -> start {nxt['job_id']} "
                  f"(slots {len(running)}/{workers}, "
                  f"mem ~{used:.1f}/{mem_budget_gb:.1f} GiB)", flush=True)

        time.sleep(1.0)

        still = []
        for job, proc, logf, t0 in running:
            timed_out = False
            if proc.poll() is None:
                if time.perf_counter() - t0 > timeout:
                    proc.kill()
                    proc.wait()
                    timed_out = True
                    print(f"  TIMEOUT {job['job_id']}", flush=True)
                else:
                    still.append((job, proc, logf, t0))
                    continue
            rec = collect(job, proc, logf, t0, run_dir, "parallel", records_fh,
                          timed_out=timed_out)
            done.append(rec)
            print(f"[{len(done)}/{total}] " + status_line(rec), flush=True)
        running = still
    return done


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=sorted(PROFILES), default="standard")
    ap.add_argument("--exec", dest="exec_mode",
                    choices=["serial", "parallel", "both"], default="both")
    ap.add_argument("--versions", nargs="+", default=None,
                    help="explicit version labels (default: profile's list)")
    ap.add_argument("--neighborhoods", nargs="+", default=None)
    ap.add_argument("--tracks", nargs="+", default=None,
                    choices=["raster", "evolution", "control", "backends",
                             "baseline"])
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel workers (default: CPUs//4, min 2)")
    ap.add_argument("--mem-budget-gb", type=float, default=None,
                    help="memory ceiling for the parallel scheduler "
                         "(default: 55%% of total RAM)")
    ap.add_argument("--timeout", type=int, default=5400,
                    help="per-job timeout in seconds (default 5400)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--seed-from", type=Path, default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the job matrix and exit")
    ap.add_argument("--rebuild", action="store_true",
                    help="re-export and rebuild every version checkout")
    ap.add_argument("--no-score", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="skip jobs that already completed cleanly in "
                         "this run-id (use with the same --run-id)")
    args = ap.parse_args()

    prof = PROFILES[args.profile]
    versions = select_versions(args.versions or prof["versions"])
    neighborhoods = args.neighborhoods or prof["neighborhoods"]
    repeats = args.repeats if args.repeats is not None else prof["repeats"]
    tracks = args.tracks or prof["tracks"]

    cpus = os.cpu_count() or 4
    workers = args.workers or max(2, cpus // 4)
    if args.mem_budget_gb is not None:
        mem_budget = args.mem_budget_gb
    else:
        try:
            import psutil
            mem_budget = psutil.virtual_memory().total / 2 ** 30 * 0.55
        except Exception:
            mem_budget = 8.0

    print("=" * 78)
    print("pyorps MV-Oberrhein history benchmark")
    print("=" * 78)
    print(f"profile      {args.profile}")
    print(f"versions     {[v['label'] for v in versions]}")
    print(f"neighborhood {neighborhoods}   repeats {repeats}")
    print(f"tracks       {tracks}")
    print(f"exec         {args.exec_mode} "
          f"(parallel: {workers} workers, {mem_budget:.1f} GiB budget)")
    print(f"data root    {C.DATA_ROOT}")
    try:
        import psutil
        avail = psutil.virtual_memory().available / 2 ** 30
        print(f"free RAM     {avail:.1f} GiB now")
        if args.exec_mode in ("parallel", "both") and avail < mem_budget:
            print(f"!! the parallel budget ({mem_budget:.1f} GiB) exceeds free "
                  f"RAM right now ({avail:.1f} GiB).\n"
                  f"   Close other work before the parallel pass, or lower it "
                  f"with --mem-budget-gb.")
    except Exception:
        pass
    print()

    checkouts = P.prepare_all(versions, args.python, args.seed_from,
                              rebuild=args.rebuild)
    if args.prepare_only:
        print(json.dumps(checkouts, indent=1))
        return 0

    backends = probe_backends(args.python)
    prototype_ok = probe_prototype(args.python)
    print(f"\nbackends available: "
          f"{ {k: v for k, v in backends.items()} }")
    if not prototype_ok:
        print("!! scikit-image is NOT installed -- the NEIS 2021 baseline "
              "track cannot run.\n"
              "   install it with:  "
              f"{args.python} -m pip install scikit-image")
    missing = [k for k, v in backends.items() if not v]
    if missing:
        print(f"!! not installed, so NOT measured: {missing}")

    meta = json.loads(C.WINDOW_META.read_text(encoding="utf-8"))
    cells = float(meta["shape"][0] * meta["shape"][1])
    print(f"search window: {meta['shape'][0]} x {meta['shape'][1]} "
          f"= {cells / 1e6:.1f} Mcells, buffer "
          f"{meta['search_space_buffer_m']} m")

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = C.RUNS / run_id
    for sub in ("jobs", "routes", "logs", "tmp_rasters"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(versions, neighborhoods, repeats, tracks, checkouts,
                      backends, cells, run_dir, prototype_ok)

    # A job whose estimate exceeds what the machine physically has would not
    # fail fast -- it would swap for hours and take the desktop with it. Refuse
    # it up front and RECORD the refusal, so it shows up in the report as a
    # measured limit rather than a silent gap.
    try:
        import psutil
        total_ram = psutil.virtual_memory().total / 2 ** 30
    except Exception:
        total_ram = 32.0
    ceiling = total_ram * 0.9
    runnable, refused = [], []
    for j in jobs:
        (refused if j["est_mem_gb"] > ceiling else runnable).append(j)
    if refused:
        print(f"\n!! {len(refused)} job(s) exceed this machine's memory "
              f"({total_ram:.0f} GiB total, ceiling {ceiling:.0f} GiB) and "
              f"will NOT be run:")
        for j in sorted({(x["version"], x["neighborhood"], x["graph_api"],
                          x["est_mem_gb"]) for x in refused}):
            print(f"     {j[0]} {j[1]} {j[2]}: needs ~{j[3]:.0f} GiB")
        print("   Override with --mem-budget-gb only if you know better; the "
              "estimate is a model, not a measurement.")
    jobs = runnable

    modes = (["serial", "parallel"] if args.exec_mode == "both"
             else [args.exec_mode])
    print(f"\n{len(jobs)} jobs x {len(modes)} execution mode(s) "
          f"= {len(jobs) * len(modes)} runs")

    if args.dry_run:
        for j in jobs:
            print(f"  {j['job_id']:<70s} est {j['est_mem_gb']:5.1f} GiB")
        return 0

    manifest = dict(
        run_id=run_id, started=datetime.now().isoformat(timespec="seconds"),
        profile=args.profile, versions=versions, neighborhoods=neighborhoods,
        repeats=repeats, tracks=tracks, exec_modes=modes, workers=workers,
        mem_budget_gb=round(mem_budget, 1), timeout_s=args.timeout,
        checkouts=checkouts, backends_available=backends,
        prototype_available=prototype_ok, window=meta["shape"],
        search_space_buffer_m=meta["search_space_buffer_m"],
        n_jobs=len(jobs), python=args.python,
        host=dict(node=platform.node(), platform=platform.platform(),
                  processor=platform.processor(), cpus=cpus),
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=1, default=str), encoding="utf-8")

    t_start = time.perf_counter()
    phase_elapsed = {}
    records_fh = open(run_dir / "records.jsonl", "a", encoding="utf-8")
    for j in refused:
        records_fh.write(json.dumps(dict(
            job_id=j["job_id"], track=j["track"], engine=j["engine"],
            version=j["version"], kind=j["kind"],
            neighborhood=j["neighborhood"], graph_api=j["graph_api"],
            use_gpu=j["use_gpu"], repeat=j["repeat"], exec_mode="none",
            wall_s=0.0, est_mem_gb=j["est_mem_gb"],
            status="skipped_insufficient_memory",
            error=f"estimated {j['est_mem_gb']:.0f} GiB > ceiling "
                  f"{ceiling:.0f} GiB on a {total_ram:.0f} GiB machine")) + "\n")
    records_fh.flush()
    try:
        for mode in modes:
            print("\n" + "=" * 78)
            print(f"EXECUTION MODE: {mode}")
            if mode == "parallel":
                print("timings below measure THROUGHPUT under contention; "
                      "per-job latency is only comparable within serial mode")
            print("=" * 78)
            t_mode = time.perf_counter()

            todo = jobs
            if args.resume:
                todo = [j for j in jobs if result_ok(j, run_dir, mode) is None]
                if len(todo) < len(jobs):
                    print(f"resume: {len(jobs) - len(todo)} job(s) already "
                          f"completed in {mode} mode, {len(todo)} to go")
                if not todo:
                    print("nothing left to do in this mode")

            if mode == "serial":
                done = run_serial(todo, args.python, run_dir, records_fh,
                                  args.timeout)
                # Replace the modelled footprint with what each job actually
                # used. The model was measured wrong by 40 % on networkit R2;
                # a measurement is strictly better input for the scheduler
                # that decides what may run alongside what.
                measured = {r["job_id"]: r["peak_rss_gb"] for r in done
                            if r.get("peak_rss_gb")}
                # jobs skipped by --resume never appear in `done`, so read
                # their peaks off disk or the scheduler falls back to the
                # model for exactly the heavy jobs it most needs to know
                for j in jobs:
                    if j["job_id"] not in measured:
                        peak = result_ok(j, run_dir, "serial")
                        if peak:
                            measured[j["job_id"]] = peak
                adjusted = 0
                for j in jobs:
                    peak = measured.get(j["job_id"])
                    if peak and abs(peak - j["est_mem_gb"]) > 0.5:
                        j["est_mem_gb"] = round(peak * 1.1, 2)  # 10 % headroom
                        adjusted += 1
                if adjusted:
                    print(f"\nre-estimated {adjusted} job footprint(s) from "
                          f"measured serial peaks")
            else:
                run_parallel(todo, args.python, run_dir, records_fh,
                             args.timeout, workers, mem_budget)
            phase_elapsed[mode] = round(time.perf_counter() - t_mode, 1)
            print(f"\n{mode} pass finished in {human(phase_elapsed[mode])}")
    finally:
        records_fh.close()

    elapsed = time.perf_counter() - t_start
    print(f"\nfinished in {human(elapsed)}")
    print(f"records: {run_dir / 'records.jsonl'}")

    manifest["finished"] = datetime.now().isoformat(timespec="seconds")
    manifest["elapsed_s"] = round(elapsed, 1)
    manifest["phase_elapsed_s"] = phase_elapsed
    if len(phase_elapsed) == 2 and phase_elapsed.get("parallel"):
        speedup = phase_elapsed["serial"] / phase_elapsed["parallel"]
        manifest["parallel_speedup"] = round(speedup, 2)
        print(f"wall-clock speedup from running in parallel: {speedup:.2f}x "
              f"({human(phase_elapsed['serial'])} -> "
              f"{human(phase_elapsed['parallel'])})")
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=1, default=str), encoding="utf-8")

    if not args.no_score:
        print("\nscoring ...")
        rc = subprocess.run([args.python, str(HERE / "score.py"),
                             "--run", str(run_dir)], cwd=str(C.REPO_ROOT))
        if rc.returncode != 0:
            print("!! scoring failed; the raw records are intact and can be "
                  f"re-scored with:\n   {args.python} "
                  f"{HERE / 'score.py'} --run {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
