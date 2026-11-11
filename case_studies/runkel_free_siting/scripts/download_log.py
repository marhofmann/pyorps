"""Append-only log of data downloads for the free-siting case study.

One JSON object per line in ``data/download_log.jsonl``. Every fetch script calls
:func:`append_event` after a file has been saved and registered, so the record of where,
when, how and with which tool a layer came from is written at the moment it happens. The log is
never rewritten; a correction is a new line that says which earlier line it corrects.

An event written after the fact (from file timestamps) carries ``"reconstructed": true`` and the
reason, so nobody mistakes it for a live record.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata as md
import json
import platform
from pathlib import Path

CASE = Path(__file__).resolve().parents[1]
LOG = CASE / "data" / "download_log.jsonl"


def now_stamps() -> dict:
    """Local time (with offset) and UTC, to the second."""
    local = dt.datetime.now().astimezone().replace(microsecond=0)
    return {"time_local": local.isoformat(),
            "time_utc": local.astimezone(dt.UTC).isoformat()}


def stamps_from(ts: float) -> dict:
    local = dt.datetime.fromtimestamp(ts).astimezone().replace(microsecond=0)
    return {"time_local": local.isoformat(),
            "time_utc": local.astimezone(dt.UTC).isoformat()}


def tool_versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("pyorps", "geopandas", "pyogrio", "rasterio", "requests", "numpy"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def append_event(**fields) -> dict:
    """Write one event. ``layer_id``, ``service``, ``what`` and ``file`` are the minimum."""
    for need in ("layer_id", "service", "what", "file", "sha256"):
        if need not in fields:
            raise ValueError(f"download event needs {need!r}")
    event = {"schema": 1, **fields}
    event.setdefault("tool_versions", tool_versions())
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    return event


def read_events() -> list[dict]:
    if not LOG.exists():
        return []
    return [json.loads(ln) for ln in LOG.read_text(encoding="utf-8").splitlines() if ln.strip()]
