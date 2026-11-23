"""Run the independent GPU stages of the case study on all cards of one machine (two RTX 5060 Ti on the box).

The stages share no output file and no input that another stage writes, so each one is a separate process pinned to one
card with ``CUDA_VISIBLE_DEVICES``. Each card is a worker that takes the next job from a common queue, longest job first, so
the two cards stay busy to the end. One solve is never split across cards (``GpuSsspSession`` is single-device; a window
needs about 22 bytes per cell on the card, so a 20 x 10 km window at 1 m is about 5 GB and fits one card).

Stages (script, what it writes, sidecar):
  mv_hv_bound  compute_mv_hv_bound_fields.py  mv_lb_5m_lattice, hv_field_taps_5m_lattice   mv_hv_bound_sidecar.json
  access       compute_access_field.py        access_field_5m_lattice                     access_field_sidecar.json
  hv_fields    compute_hv_fields.py           hv_field_min / winning connection point      hv_field_sidecar.json
  window       compute_window_bound.py        mv_window_dtr_5m_lattice (+ CPU Win values)   mv_window_sidecar.json

Registration events go to the download log by single-line appends, which two processes can do at the same time.

Usage (on the box, from the repo root, in the project's environment):

    python case_studies/runkel_free_siting/scripts/run_gpu_stages.py --dry-run
    python case_studies/runkel_free_siting/scripts/run_gpu_stages.py --gpus 0 1 --log-dir <dir>
    python case_studies/runkel_free_siting/scripts/run_gpu_stages.py --selftest        # no GPU work, checks the scheduler

The summary JSON (``gpu_stages_<date>.json`` in ``--log-dir``) holds the machine, versions, wall time of every stage and every
card. A run on a shared machine is not a performance figure: say so in the paper if it was not dedicated.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASE = HERE.parent
PY = sys.executable

# (name, script, extra arguments, relative weight for the queue order: larger runs first)
STAGES = {
    "mv_hv_bound": ("compute_mv_hv_bound_fields.py", [], 4),
    "window": ("compute_window_bound.py", ["--levels", "33", "30", "20"], 3),
    "access": ("compute_access_field.py", [], 2),
    "hv_fields": ("compute_hv_fields.py", [], 1),
}


def machine_record() -> dict:
    rec = {"platform": platform.platform(), "python": sys.version.split()[0], "cpu_count": os.cpu_count()}
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        rec["gpus"] = [ln.strip() for ln in out]
    except (OSError, subprocess.SubprocessError):
        rec["gpus"] = []
    versions = {}
    for mod in ("numpy", "numba", "cupy", "rasterio", "pyorps"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:                                             # noqa: BLE001  (a missing package is a fact to record)
            versions[mod] = None
    rec["versions"] = versions
    return rec


def run_queue(jobs: list[tuple[str, list[str]]], gpus: list[str], *, env_extra: dict, log_dir: Path,
              runner=subprocess.run) -> dict:
    """Run ``jobs`` (name, command) on the cards; returns ``{job: {gpu, seconds, returncode}}`` and the wall time per card."""
    q: queue.Queue = queue.Queue()
    for j in jobs:
        q.put(j)
    results: dict = {}
    busy: dict = {g: 0.0 for g in gpus}
    lock = threading.Lock()
    log_dir.mkdir(parents=True, exist_ok=True)

    def worker(gpu: str):
        while True:
            try:
                name, cmd = q.get_nowait()
            except queue.Empty:
                return
            env = {**os.environ, **env_extra, "CUDA_VISIBLE_DEVICES": gpu}
            t0 = time.time()
            with open(log_dir / f"{name}_gpu{gpu}.log", "w", encoding="utf-8") as fh:
                print(f"[gpu {gpu}] start {name}", flush=True)
                rc = runner(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, cwd=str(CASE.parent.parent)).returncode
            dt_s = time.time() - t0
            with lock:
                results[name] = {"gpu": gpu, "seconds": round(dt_s, 1), "returncode": rc}
                busy[gpu] += dt_s
            print(f"[gpu {gpu}] done  {name}: {dt_s / 60:.1f} min, exit {rc}", flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in gpus]
    t_all = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return {"jobs": results, "busy_seconds_per_gpu": {g: round(v, 1) for g, v in busy.items()},
            "wall_seconds": round(time.time() - t_all, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gpus", nargs="+", default=["0", "1"], help="card indices (physical, as nvidia-smi lists them)")
    ap.add_argument("--stages", nargs="+", choices=sorted(STAGES), default=sorted(STAGES, key=lambda s: -STAGES[s][2]))
    ap.add_argument("--log-dir", default=str(CASE / "results" / "gpu_stages_logs"))
    ap.add_argument("--max-temp", type=float, default=85.0)
    ap.add_argument("--cool-s", type=int, default=0, help="pause between GPU solves; 0 on a desktop card (the laptop needs 90)")
    ap.add_argument("--numba-threads", type=int, default=8, help="threads of the CPU part of a stage (the window stage)")
    ap.add_argument("--no-register", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="run trivial commands on the cards' queue; no GPU work")
    args = ap.parse_args(argv)

    if args.selftest:
        jobs = [(f"selftest_{i}", [PY, "-c", "import os,time;time.sleep(0.2);print(os.environ['CUDA_VISIBLE_DEVICES'])"]) for i in range(4)]
    else:
        jobs = []
        for name in args.stages:
            script, extra, _ = STAGES[name]
            cmd = [PY, str(HERE / script), *extra, "--max-temp", str(args.max_temp)]
            if args.no_register:
                cmd.append("--no-register")
            jobs.append((name, cmd))
    if args.dry_run:
        for name, cmd in jobs:
            print(name, "->", " ".join(cmd))
        print("cards:", args.gpus)
        return 0

    log_dir = Path(args.log_dir)
    env_extra = {"PYORPS_COOL_S": str(args.cool_s), "NUMBA_NUM_THREADS": str(args.numba_threads)}
    summary = {"started": dt.datetime.now().isoformat(timespec="seconds"), "machine": machine_record(), "cards": args.gpus,
               "cool_s": args.cool_s, "numba_threads": args.numba_threads, **run_queue(jobs, args.gpus, env_extra=env_extra, log_dir=log_dir)}
    out = log_dir / f"gpu_stages_{dt.date.today().isoformat()}{'_selftest' if args.selftest else ''}.json"
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    failed = [n for n, r in summary["jobs"].items() if r["returncode"] != 0]
    if failed:
        print("FAILED:", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
