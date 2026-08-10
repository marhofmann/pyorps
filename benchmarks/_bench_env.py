"""Shared benchmark environment, thermal and TDR instrumentation.

Performance-plan items 0.1 (machine gate, TDR incident logging) and 0.12
(thermal/variance protocol). Imported by ``benchmark_scale_5km.py``,
``benchmark_gpu_v5.py`` and ``benchmark_eikonal_gpu.py``.

Nothing in this module creates a CUDA context: every GPU query goes
through ``nvidia-smi`` or the Windows registry, so :func:`gpu_gate` can
refuse to start *before* CuPy touches the device. :func:`cuda_device_info`
is the single exception and must only be called after the gate passes.

Why the extra machinery:

* The machine is shared. A run that finds another compute process or a
  hot idle GPU must degrade to CPU-only rows, not compete for the device.
* Sustained persistent-kernel runs at 10000^2 throttle a laptop GPU, so a
  median of N reps without recorded clocks cannot distinguish a real
  1.2x lever from thermal noise. Every timed region therefore carries
  its own clock/temperature envelope, and the first (cold) run is
  reported separately from the steady-state reps.
* The Windows WDDM watchdog can kill a long single-launch kernel. The
  configured TdrDelay on this machine is unknown (2 s WDDM default vs a
  ~60 s recorded observation), so it is read from the registry and
  printed, and a killed run is recorded as data instead of aborting the
  sweep.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

#: Fields queried once for the environment snapshot.
_STATIC_FIELDS = ("name", "driver_version", "memory.total", "memory.free",
                  "memory.used", "power.limit", "clocks.max.sm",
                  "clocks.max.mem", "temperature.gpu", "utilization.gpu",
                  "pstate")

#: Fields sampled around every timed region (kept to long-supported ones —
#: nvidia-smi fails the *whole* query if a single field is unavailable).
_SAMPLE_FIELDS = ("temperature.gpu", "clocks.sm", "clocks.mem",
                  "power.draw", "utilization.gpu", "memory.used")

_TDR_KEY = r"SYSTEM\CurrentControlSet\Control\GraphicsDrivers"
_TDR_VALUES = ("TdrLevel", "TdrDelay", "TdrDdiDelay", "TdrLimitCount",
               "TdrLimitTime", "TdrDebugMode")
#: WDDM defaults when the value is absent from the registry.
_TDR_DEFAULTS = {"TdrLevel": 3, "TdrDelay": 2, "TdrDdiDelay": 5,
                 "TdrLimitCount": 6, "TdrLimitTime": 60, "TdrDebugMode": 0}

#: Exception text that marks a run as killed by the driver rather than by
#: an ordinary programming error.
_TDR_HINTS = ("launch timed out", "unspecified launch failure",
              "context is destroyed", "context was destroyed",
              "invalid device context", "device-side assert",
              "illegal memory access", "ecc uncorrectable",
              "no cuda-capable device", "os call failed")

_CREATE_NO_WINDOW = 0x08000000


# ----------------------------------------------------------------------
# subprocess / nvidia-smi plumbing
# ----------------------------------------------------------------------

def _run(cmd, timeout=30.0):
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = _CREATE_NO_WINDOW
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, **kw)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        return None, (proc.stderr or proc.stdout or "").strip()[:300]
    return proc.stdout, None


def _num(text):
    """nvidia-smi emits '[N/A]' / '[Not Supported]' for missing values."""
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _nvsmi_query(fields, timeout=30.0):
    out, err = _run(["nvidia-smi", "--query-gpu=" + ",".join(fields),
                     "--format=csv,noheader,nounits"], timeout=timeout)
    if out is None:
        return None, err
    rows = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        rows.append(dict(zip(fields, parts)))
    return rows, None


def gpu_static_snapshot(index=0):
    """One-shot nvidia-smi snapshot; never raises."""
    rows, err = _nvsmi_query(_STATIC_FIELDS)
    if rows is None:
        return {"available": False, "error": err}
    if index >= len(rows):
        return {"available": False, "error": f"no GPU at index {index}"}
    raw = rows[index]
    snap = {"available": True, "index": index, "n_gpus": len(rows),
            "name": raw["name"], "driver_version": raw["driver_version"]}
    for field in _STATIC_FIELDS[2:]:
        snap[field.replace(".", "_")] = _num(raw[field])
    snap["pstate"] = raw["pstate"]
    return snap


def compute_processes():
    """Compute processes attached to the GPU.

    On WDDM (this machine) nvidia-smi frequently cannot enumerate compute
    apps at all, so ``supported`` records whether the answer is
    trustworthy; :func:`gpu_gate` falls back to resident-memory evidence.
    """
    out, err = _run(["nvidia-smi",
                     "--query-compute-apps=pid,process_name,used_memory",
                     "--format=csv,noheader,nounits"])
    if out is None:
        return {"supported": False, "error": err, "processes": []}
    text = out.strip()
    if not text or "not supported" in text.lower():
        return {"supported": False, "error": text[:200] or "empty",
                "processes": []}
    procs = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        procs.append({"pid": int(_num(parts[0]) or -1), "name": parts[1],
                      "used_mib": _num(parts[2])})
    return {"supported": True, "processes": procs}


# ----------------------------------------------------------------------
# TDR (Windows WDDM watchdog)
# ----------------------------------------------------------------------

def tdr_settings():
    """Read HKLM\\SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers.

    Absent values mean the WDDM default applies; both the raw values and
    the effective ones are reported so a 2 s vs 60 s TdrDelay can never be
    inferred from silence.
    """
    info = {"registry_key": r"HKLM\\" + _TDR_KEY, "values": {},
            "defaults": dict(_TDR_DEFAULTS), "effective": {},
            "available": False}
    if os.name != "nt":
        info["error"] = "not Windows"
    else:
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _TDR_KEY) as key:
                for name in _TDR_VALUES:
                    try:
                        info["values"][name] = int(
                            winreg.QueryValueEx(key, name)[0])
                    except OSError:
                        info["values"][name] = None
            info["available"] = True
        except OSError as exc:
            info["error"] = f"{type(exc).__name__}: {exc}"
    for name, default in _TDR_DEFAULTS.items():
        value = info["values"].get(name)
        info["effective"][name] = default if value is None else value
    info["effective_delay_s"] = info["effective"]["TdrDelay"]
    info["watchdog_enabled"] = info["effective"]["TdrLevel"] != 0
    return info


def windows_tdr_events(max_events=400, limit=20):
    """Recent display-driver reset events from the System log.

    Event 4101 is 'Display driver stopped responding and has recovered'.
    Requires no elevation on most systems but is allowed to fail — it is
    corroborating evidence for an incident, never the primary signal.
    """
    if os.name != "nt":
        return {"available": False, "error": "not Windows", "events": []}
    script = (
        "Get-WinEvent -LogName System -MaxEvents %d -ErrorAction "
        "SilentlyContinue | Where-Object { $_.Id -eq 4101 -or "
        "$_.ProviderName -match 'nvlddmkm|Display' } | Select-Object "
        "-First %d @{n='time';e={$_.TimeCreated.ToString('o')}},Id,"
        "ProviderName,@{n='message';e={$_.Message}} | ConvertTo-Json "
        "-Compress" % (max_events, limit))
    out, err = _run(["powershell", "-NoProfile", "-NonInteractive",
                     "-Command", script], timeout=60.0)
    if out is None:
        return {"available": False, "error": err, "events": []}
    text = out.strip()
    if not text:
        return {"available": True, "events": []}
    try:
        parsed = json.loads(text)
    except ValueError as exc:
        return {"available": False, "error": str(exc), "events": []}
    if isinstance(parsed, dict):
        parsed = [parsed]
    return {"available": True, "events": parsed}


def looks_like_tdr(exc):
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(hint in text for hint in _TDR_HINTS)


class GpuFaultLog:
    """Sticky CUDA-fault recorder for a whole sweep.

    A TDR (or any fault that destroys the context) makes every later CuPy
    call fail, so ``context_dead`` latches and callers skip the remaining
    GPU rows instead of collecting a cascade of identical tracebacks.
    """

    def __init__(self):
        self.incidents = []
        self.context_dead = False

    def guard(self, label, fn):
        """Run ``fn``; on failure record an incident and return None.

        Returns ``(ok, value, incident)``.
        """
        if self.context_dead:
            incident = {"label": label, "kind": "skipped_context_dead",
                        "time": _utc_now()}
            self.incidents.append(incident)
            return False, None, incident
        t0 = time.perf_counter()
        try:
            value = fn()
        except Exception as exc:                # CuPy driver errors included
            dt = time.perf_counter() - t0
            tdr = looks_like_tdr(exc)
            incident = {
                "label": label,
                "kind": "tdr_suspected" if tdr else "cuda_error",
                "exception": type(exc).__name__,
                "message": str(exc)[:500],
                "elapsed_s": dt,
                "time": _utc_now(),
                "gpu": gpu_static_snapshot(),
            }
            healthy = cuda_context_healthy()
            incident["context_healthy_after"] = healthy
            if not healthy or tdr:
                self.context_dead = True
            self.incidents.append(incident)
            print(f"  !! {label}: {incident['kind']} after {dt:.1f} s "
                  f"({incident['exception']}) — context "
                  f"{'dead' if self.context_dead else 'alive'}")
            return False, None, incident
        return True, value, None


def cuda_context_healthy():
    """Cheap liveness probe after a suspected context death."""
    try:
        import cupy as cp
        probe = cp.zeros(16, dtype=cp.float32)
        probe += 1
        cp.cuda.Stream.null.synchronize()
        return bool(float(probe[0]) == 1.0)
    except Exception:
        return False


# ----------------------------------------------------------------------
# Machine gate
# ----------------------------------------------------------------------

def gpu_gate(max_idle_temp_c=60.0, max_idle_util_pct=20.0,
             busy_memory_mib=512.0, min_free_mib=None, allow_shared=False):
    """Decide whether this process may start GPU work.

    The machine is shared and the standing rule is to leave room for the
    user's other work, so anything that looks like a resident tenant is a
    refusal, not a warning. ``allow_shared=True`` downgrades the tenancy
    and thermal reasons to warnings (an explicit operator override) but
    never the 'nvidia-smi is missing' one.
    """
    snap = gpu_static_snapshot()
    procs = compute_processes()
    result = {"ok": False, "hard_reasons": [], "warnings": [],
              "snapshot": snap, "compute_apps": procs,
              "allow_shared": bool(allow_shared)}

    if not snap.get("available"):
        result["hard_reasons"].append(
            f"nvidia-smi unavailable ({snap.get('error')})")
        return result

    soft = []
    others = [p for p in procs["processes"] if p["pid"] != os.getpid()]
    if others:
        soft.append(f"{len(others)} compute process(es) attached: "
                    + ", ".join(f"{p['name']}#{p['pid']}" for p in others))
    used = snap.get("memory_used")
    if not procs["supported"] and used is not None and used > busy_memory_mib:
        soft.append(f"{used:.0f} MiB already resident and nvidia-smi cannot "
                    f"enumerate compute apps (WDDM) — assuming a tenant")
    util = snap.get("utilization_gpu")
    if util is not None and util > max_idle_util_pct:
        soft.append(f"idle utilization {util:.0f}% > {max_idle_util_pct:.0f}%")

    # Temperature alone is a poor tenancy signal on a laptop: the GPU shares a
    # chassis with a CPU that may be running someone else's job, so the card
    # can read 65 C while sitting at 0% utilisation, 0 MiB and its idle clock.
    # Refusing on that blocks legitimate work, and a gate that cries wolf is a
    # gate people learn to pass --allow-shared-gpu to by reflex. So temperature
    # is a hard reason only when something else also suggests the GPU itself is
    # working; otherwise it is recorded as a warning.
    temp = snap.get("temperature_gpu")
    if temp is not None and temp > max_idle_temp_c:
        looks_idle = (
            (util is None or util <= max_idle_util_pct)
            and (used is None or used <= busy_memory_mib)
            and not others
        )
        message = (f"idle temperature {temp:.0f} C > {max_idle_temp_c:.0f} C")
        if looks_idle:
            result["warnings"].append(
                message + " but the GPU is otherwise idle (0 tenants, low "
                          "utilisation, little memory resident) — treating as "
                          "chassis heat, not GPU load")
        else:
            soft.append(message)

    free = snap.get("memory_free")
    if min_free_mib is not None and free is not None and free < min_free_mib:
        result["hard_reasons"].append(
            f"only {free:.0f} MiB free, {min_free_mib:.0f} MiB required")

    if allow_shared:
        result["warnings"].extend(soft)
    else:
        result["hard_reasons"].extend(soft)
    result["ok"] = not result["hard_reasons"]
    return result


def print_gate(gate):
    snap = gate["snapshot"]
    if snap.get("available"):
        print(f"GPU: {snap['name']} driver {snap['driver_version']} | "
              f"{snap.get('memory_free', 0):.0f}/"
              f"{snap.get('memory_total', 0):.0f} MiB free | "
              f"{snap.get('temperature_gpu')} C | "
              f"util {snap.get('utilization_gpu')}% | "
              f"pstate {snap.get('pstate')}")
    for reason in gate["warnings"]:
        print(f"  WARNING (override active): {reason}")
    for reason in gate["hard_reasons"]:
        print(f"  REFUSING GPU WORK: {reason}")
    if gate["ok"]:
        print("  GPU gate: OK")
    else:
        print("  GPU gate: CLOSED — continuing with CPU-only rows")


def cuda_device_info():
    """Device attributes that nvidia-smi does not expose (SM count).

    Creates a CUDA context — call only after :func:`gpu_gate` passes.
    """
    try:
        import cupy as cp
        dev = cp.cuda.Device()
        attrs = dev.attributes
        free, total = dev.mem_info
        return {
            "available": True,
            "multiprocessor_count": int(attrs["MultiProcessorCount"]),
            "max_threads_per_multiprocessor":
                int(attrs["MaxThreadsPerMultiProcessor"]),
            "compute_capability": dev.compute_capability,
            "total_memory_bytes": int(total),
            "free_memory_bytes": int(free),
            "cupy_version": cp.__version__,
        }
    except BaseException as exc:
        return {"available": False,
                "error": f"{type(exc).__name__}: {exc}"}


# ----------------------------------------------------------------------
# Environment snapshot
# ----------------------------------------------------------------------

def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cpu_contention(interval=1.0, top=5):
    """Who else is using the CPU right now, sampled over ``interval``.

    Absolute wall-clock numbers are only comparable between runs taken under
    comparable load, and nothing else in this snapshot records that. A run
    taken while another job holds eight cores is not wrong, it is simply not
    comparable with one taken on a quiet machine - and six weeks later
    nothing in the results file would say so. A/B comparisons within one run
    survive contention (both arms pay it); absolute baselines do not.

    Returns busy-core counts, not percentages: "3.2 of 16 cores" is directly
    comparable across machines, "20%" is not.
    """
    logical = psutil.cpu_count(logical=True) or 1
    procs = {}
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            proc.cpu_percent(None)          # prime; first call always 0.0
            procs[proc.pid] = proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    total_before = psutil.cpu_percent(None)  # prime the system-wide counter
    del total_before
    time.sleep(max(0.1, float(interval)))

    me = os.getpid()
    rows = []
    for pid, proc in procs.items():
        # PID 0 is Windows' "System Idle Process": its CPU share is the
        # machine being IDLE. Counting it inverts the reading - a quiet
        # 16-core box reports ~13 busy cores.
        if pid == 0:
            continue
        try:
            cores = proc.cpu_percent(None) / 100.0
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if cores >= 0.05:
            try:
                name = proc.name()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                name = "?"
            rows.append({"pid": pid, "name": name,
                         "cores": round(cores, 2), "self": pid == me})
    rows.sort(key=lambda r: -r["cores"])
    busy = round(sum(r["cores"] for r in rows), 2)
    return {
        "interval_s": float(interval),
        "cpu_count_logical": logical,
        "busy_cores": busy,
        "busy_cores_excluding_self": round(
            sum(r["cores"] for r in rows if not r["self"]), 2),
        "system_percent": psutil.cpu_percent(None),
        "top": rows[:top],
    }


def print_contention(con, label=""):
    """One line, so a contaminated run is obvious in the console too."""
    other = con["busy_cores_excluding_self"]
    verdict = ("quiet" if other < 0.5 else
               "light" if other < 2.0 else
               "BUSY - absolute timings are not comparable")
    head = f"cpu{(' ' + label) if label else ''}: "
    names = ", ".join(f"{r['name']}({r['cores']:.1f})"
                      for r in con["top"] if not r["self"]) or "-"
    print(f"{head}{other:.2f} of {con['cpu_count_logical']} cores busy "
          f"elsewhere [{verdict}] | {names}")


def environment_snapshot(include_gpu=True, include_processes=True):
    """Everything a later reader needs to judge whether two runs are
    comparable: host, GPU, TDR configuration and current tenancy.

    ``include_gpu=False`` suppresses every nvidia-smi and registry query,
    for offline smoke tests on a machine whose GPU must not be touched at
    all.
    """
    vm = psutil.virtual_memory()
    snap = {
        "time_utc": _utc_now(),
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": sys.version.split()[0],
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "ram_total_bytes": int(vm.total),
            "ram_available_bytes": int(vm.available),
        },
    }
    snap["cpu_contention"] = cpu_contention()
    if not include_gpu:
        snap["gpu"] = {"available": False, "error": "not queried"}
        snap["tdr"] = {"available": False, "error": "not queried",
                       "values": {}, "effective": dict(_TDR_DEFAULTS),
                       "effective_delay_s": _TDR_DEFAULTS["TdrDelay"],
                       "watchdog_enabled": True}
        return snap
    snap["gpu"] = gpu_static_snapshot()
    snap["tdr"] = tdr_settings()
    if include_processes:
        snap["gpu_compute_apps"] = compute_processes()
    return snap


def print_environment(snap):
    host = snap["host"]
    print(f"host: {host['platform']} | python {host['python']} | "
          f"{host['cpu_count_logical']} logical CPUs | "
          f"{host['ram_total_bytes'] / 2**30:.1f} GiB RAM "
          f"({host['ram_available_bytes'] / 2**30:.1f} free)")
    if snap.get("cpu_contention"):
        print_contention(snap["cpu_contention"], "at start")
    tdr = snap["tdr"]
    raw = ", ".join(f"{k}={v}" for k, v in tdr["values"].items()) or "none"
    print(f"TDR registry: {raw}")
    print(f"TDR effective: delay {tdr['effective_delay_s']} s, level "
          f"{tdr['effective']['TdrLevel']}, watchdog "
          f"{'ON' if tdr['watchdog_enabled'] else 'OFF'}"
          + ("" if tdr["values"].get("TdrDelay") is not None
             else "  (TdrDelay absent -> WDDM default assumed)"))


def peak_rss_bytes():
    """Host high-water mark. ``peak_wset`` is Windows-only; elsewhere the
    current RSS is the best available answer and is labelled as such."""
    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None)
    return int(peak if peak is not None else info.rss)


def gpu_pool_bytes():
    """CuPy pool occupancy; ``None`` when CuPy never initialised."""
    try:
        import cupy as cp
        pool = cp.get_default_memory_pool()
        return {"used_bytes": int(pool.used_bytes()),
                "total_bytes": int(pool.total_bytes())}
    except Exception:
        return None


def free_gpu_pool():
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


# ----------------------------------------------------------------------
# Thermal / clock sampling around a timed region
# ----------------------------------------------------------------------

class GpuSampler:
    """Background nvidia-smi sampler for one timed region.

    Used as a context manager. One sample is taken immediately so that
    even a sub-interval run carries an envelope; the sampling thread is a
    plain subprocess loop, which is negligible next to the GPU work it
    observes.
    """

    def __init__(self, interval=0.5, enabled=True):
        self.interval = float(interval)
        self.enabled = bool(enabled)
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def _sample_once(self):
        rows, err = _nvsmi_query(_SAMPLE_FIELDS, timeout=10.0)
        if rows is None:
            return
        raw = rows[0]
        self.samples.append({
            "t": time.perf_counter(),
            "temp_c": _num(raw["temperature.gpu"]),
            "sm_mhz": _num(raw["clocks.sm"]),
            "mem_mhz": _num(raw["clocks.mem"]),
            "power_w": _num(raw["power.draw"]),
            "util_pct": _num(raw["utilization.gpu"]),
            "mem_used_mib": _num(raw["memory.used"]),
        })

    def _loop(self):
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval)

    def __enter__(self):
        if self.enabled:
            self._sample_once()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=15.0)
            self._sample_once()
        return False

    def summary(self):
        if not self.samples:
            return {"n_samples": 0}

        def agg(key):
            vals = [s[key] for s in self.samples if s[key] is not None]
            if not vals:
                return None
            return {"min": min(vals), "mean": statistics.fmean(vals),
                    "max": max(vals), "first": vals[0], "last": vals[-1]}

        out = {"n_samples": len(self.samples), "interval_s": self.interval}
        for key in ("temp_c", "sm_mhz", "mem_mhz", "power_w", "util_pct",
                    "mem_used_mib"):
            out[key] = agg(key)
        sm = out["sm_mhz"]
        if sm is not None and sm["max"]:
            # A steady region whose SM clock sags well below its own peak
            # is throttling; flagged so a 1.2x "lever" is never read off a
            # thermally degraded steady state.
            out["sm_clock_droop_pct"] = (sm["max"] - sm["min"]) / sm["max"] * 100
        return out


def timed_reps(fn, reps=1, sample=False, sample_interval=0.5, label=""):
    """Time ``fn`` ``reps`` times with first-run vs steady-state split.

    No warm-up is discarded: the first run is *reported*, because at
    target scale (kernel JIT, cold caches, cold clocks) it is the number a
    user actually experiences once. ``steady_*`` covers runs 2..N and is
    ``None`` for a single rep — stating that the steady state was not
    measured is the honest answer, an average over one cold run is not.
    """
    times = []
    result = None
    with GpuSampler(interval=sample_interval, enabled=sample) as sampler:
        for _ in range(max(1, int(reps))):
            t0 = time.perf_counter()
            result = fn()
            times.append(time.perf_counter() - t0)
    steady = times[1:]
    stats = {
        "label": label,
        "reps": len(times),
        "all_s": times,
        "first_s": times[0],
        "steady_median_s": statistics.median(steady) if steady else None,
        "steady_min_s": min(steady) if steady else None,
        "steady_max_s": max(steady) if steady else None,
        "thermal": sampler.summary(),
    }
    if steady and stats["steady_median_s"]:
        stats["steady_spread_pct"] = (
            (stats["steady_max_s"] - stats["steady_min_s"])
            / stats["steady_median_s"] * 100)
        stats["first_over_steady"] = times[0] / stats["steady_median_s"]
    else:
        stats["steady_spread_pct"] = None
        stats["first_over_steady"] = None
    #: The number to quote: steady state when it exists, else the cold run.
    stats["report_s"] = (stats["steady_median_s"] if steady else times[0])
    return stats, result


# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------

def results_dir():
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    return out


def write_results(prefix, payload):
    """Timestamped JSON next to the other benchmark results."""
    path = results_dir() / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(payload, indent=1, default=float))
    return path
