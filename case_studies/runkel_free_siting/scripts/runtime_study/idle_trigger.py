"""Watch the machine and start the timing sessions (R1, then R2) after 3 quiet minutes in a row.

Owner instruction 2026-10-05: "add a monitor that checks the system usage every 60 seconds and if the machine is idle
for 3 minutes in a row, kick the timing sessions". This covers ONLY the two timing sessions of the runtime study
(``r1_measure.py`` on one core, ``r2_parallel.py`` on up to 8 threads for short bursts; both write only small JSON files).
Nothing else is started by it.

* Every 60 s it measures the system-wide CPU load (mean over that minute) and the free RAM and prints one status line.
* ``--quiet-minutes`` (default 3) consecutive minutes below ``--idle-pct`` (default 15 %) with at least ``--min-free-gb``
  (default 8) free RAM start the sessions as child processes, one after the other.
* While a session runs it keeps watching: if the load from OTHER processes exceeds ``--back-pct`` (default 25 %) for 3
  minutes in a row (the owner is back), it stops the session and waits for the next quiet window. The sessions skip
  jobs that are already done and clean, so they resume where they stopped.
* It gives up after ``--max-wait-h`` hours of no quiet window (default 12) and says so.
* Stop it by killing its process; it leaves nothing running except the child it started.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PY = REPO / ".venv" / "Scripts" / "python.exe"
SESSIONS = [("r1", [str(PY), str(HERE / "r1_measure.py")]),
            ("r2", [str(PY), str(HERE / "r2_parallel.py")])]


def say(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def minute_sample() -> tuple[float, float]:
    """(system CPU % over the next 60 s, free RAM in GB)."""
    load = psutil.cpu_percent(interval=60)
    return float(load), psutil.virtual_memory().available / 1e9


def kill_tree(proc: subprocess.Popen) -> None:
    try:
        parent = psutil.Process(proc.pid)
        for c in parent.children(recursive=True):
            c.kill()
        parent.kill()
    except psutil.Error:
        pass


_KNOWN: dict = {}


def tree_cpu(proc: subprocess.Popen) -> float:
    """The child tree's CPU share (percent of all logical CPUs) since the last call. The Process objects are
    kept between calls (a new object's first cpu_percent is always 0)."""
    try:
        root = _KNOWN.setdefault(proc.pid, psutil.Process(proc.pid))
        procs = [root] + [_KNOWN.setdefault(c.pid, c) for c in root.children(recursive=True)]
    except psutil.Error:
        return 0.0
    tot = 0.0
    for p in procs:
        try:
            tot += p.cpu_percent(None)
        except psutil.Error:
            pass
    return tot / psutil.cpu_count()


def run_session(name: str, cmd: list[str], args) -> str:
    """Run one session; return 'done', 'failed' or 'stopped' (owner back)."""
    say(f"START {name}: {' '.join(Path(c).name for c in cmd)}")
    proc = subprocess.Popen(cmd, cwd=str(REPO))
    tree_cpu(proc)                                    # prime the per-process counters
    busy = 0
    while proc.poll() is None:
        load = psutil.cpu_percent(interval=60)
        own = tree_cpu(proc)
        others = max(0.0, load - own)
        busy = busy + 1 if others > args.back_pct else 0
        say(f"  {name} running: system {load:.0f} %, session {own:.0f} %, others {others:.0f} % (busy streak {busy})")
        if busy >= args.quiet_minutes:
            say(f"STOP {name}: other work above {args.back_pct} % for {busy} minutes; waiting for the next quiet window")
            kill_tree(proc)
            return "stopped"
    say(f"END {name}: exit code {proc.returncode}")
    if proc.returncode == 3:                          # the session found the machine busy at its own start
        return "stopped"
    return "done" if proc.returncode == 0 else "failed"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quiet-minutes", type=int, default=3)
    ap.add_argument("--idle-pct", type=float, default=15.0)
    ap.add_argument("--back-pct", type=float, default=25.0)
    ap.add_argument("--min-free-gb", type=float, default=8.0)
    ap.add_argument("--max-wait-h", type=float, default=12.0)
    ap.add_argument("--dry-run", action="store_true", help="watch and report, but never start a session")
    args = ap.parse_args()
    deadline = time.time() + args.max_wait_h * 3600
    pending = list(SESSIONS)
    quiet, lowest = 0, 100.0
    say(f"watching: idle = system CPU < {args.idle_pct} % for {args.quiet_minutes} minutes in a row, "
        f"free RAM >= {args.min_free_gb} GB; sessions: {[n for n, _ in pending]}{' (dry run)' if args.dry_run else ''}")
    while pending:
        if time.time() > deadline:
            say(f"GIVE UP: no quiet window in {args.max_wait_h} h (lowest minute seen: {lowest:.0f} %). Sessions left: {[n for n, _ in pending]}")
            return 2
        load, free = minute_sample()
        lowest = min(lowest, load)
        ok = load < args.idle_pct and free >= args.min_free_gb
        quiet = quiet + 1 if ok else 0
        say(f"load {load:.0f} %, free RAM {free:.1f} GB -> {'quiet' if ok else 'not quiet'} ({quiet}/{args.quiet_minutes}); lowest so far {lowest:.0f} %")
        if quiet >= args.quiet_minutes and not args.dry_run:
            name, cmd = pending[0]
            result = run_session(name, cmd, args)
            if result == "done":
                pending.pop(0)
            elif result == "failed":
                say(f"{name} failed; not retrying automatically. See the session output above.")
                return 1
            quiet = 0
    say("ALL TIMING SESSIONS DONE: results in case_studies/runkel_free_siting/results/runtime_study/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
