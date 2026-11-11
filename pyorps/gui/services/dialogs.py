"""
PYORPS GUI: native file/folder pickers (tkinter, in a child process).

The Dash server runs on the user's own machine, so the familiar open/save
dialogs can be raised server-side and their result written into the path
inputs. Each dialog runs in a short-lived SUBPROCESS
(:mod:`pyorps.gui.services.dialog_worker`) that prints the answer as JSON, so
Tk never lives in the server process: no Tk-off-the-main-thread crash on macOS,
and a missing display or tkinter is reported instead of swallowed.

A lock keeps it to one dialog at a time and is held until the child has really
ended; a dialog left open for :data:`DIALOG_TIMEOUT_S` is killed first.

Every ``ask_*`` returns a :class:`DialogResult`; ``status`` is ``"path"``,
``"cancelled"`` or ``"unavailable"`` (with ``reason``). The result also
unpacks as the historical ``(path_or_None, error_or_None)`` pair.
"""
from __future__ import annotations

import json
import subprocess  # nosec B404 - runs only our own helper, fixed argv, no shell
import sys
import threading
from pathlib import Path
from typing import NamedTuple

_dialog_lock = threading.Lock()

#: a dialog left open this long is killed (seconds)
DIALOG_TIMEOUT_S = 300.0
#: grace period for the killed child to be reaped (seconds)
KILL_WAIT_S = 10.0

_WORKER = Path(__file__).with_name("dialog_worker.py")
_CREATION_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # Windows only; 0 elsewhere

FILETYPES = {
    "vector": [("Vector / raster data",
                "*.shp *.geojson *.json *.gpkg *.gml *.kml *.tif *.tiff"),
               ("All files", "*.*")],
    "raster": [("GeoTIFF raster", "*.tif *.tiff"), ("All files", "*.*")],
    "routes": [("Routes", "*.geojson *.json *.gpkg *.shp *.csv"),
               ("All files", "*.*")],
    "table": [("Cost tables", "*.csv *.json *.xlsx"),
              ("All files", "*.*")],
    "profile": [("Infrastructure profiles", "*.yaml *.yml *.json"),
                ("All files", "*.*")],
    "project": [("PYORPS project", "project.json *.json"),
                ("All files", "*.*")],
}


class DialogResult(NamedTuple):
    """Outcome of one dialog: a chosen path, a cancel, or "cannot be shown"."""

    path: str | None = None
    error: str | None = None      # why no dialog could be shown (None: it was shown)

    @property
    def status(self) -> str:
        if self.path:
            return "path"
        return "unavailable" if self.error else "cancelled"

    @property
    def reason(self) -> str | None:
        return self.error

    @classmethod
    def chosen(cls, path: str) -> DialogResult:
        return cls(path=path)

    @classmethod
    def cancelled(cls) -> DialogResult:
        return cls()

    @classmethod
    def unavailable(cls, reason: str) -> DialogResult:
        return cls(error=reason or "the file dialog is not available")


def _stop(proc: subprocess.Popen) -> None:
    """Kill the child and wait until it is really gone (reaps it, closes pipes)."""
    try:
        proc.kill()
    except OSError:     # already gone
        pass
    try:
        proc.communicate(timeout=KILL_WAIT_S)
    except (subprocess.TimeoutExpired, OSError, ValueError):  # nosec B110
        pass


def _parse_reply(returncode: int | None, stdout: str, stderr: str) -> DialogResult:
    """Turn the child's output into a result; a crash is "unavailable"."""
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            reply = json.loads(line)
        except ValueError:
            continue
        status = reply.get("status") if isinstance(reply, dict) else None
        if status == "path" and reply.get("path"):
            return DialogResult.chosen(str(reply["path"]))
        if status == "cancelled":
            return DialogResult.cancelled()
        if status == "unavailable":
            return DialogResult.unavailable(str(reply.get("reason") or ""))
    tail = (stderr or "").strip().splitlines()[-1:]      # last stderr line, if any
    detail = f": {tail[0]}" if tail else ""
    return DialogResult.unavailable(
        f"the file dialog helper ended with exit code {returncode} "
        f"without an answer{detail}")


def _run(request: dict) -> DialogResult:
    with _dialog_lock:      # released only after the child has really ended
        try:
            proc = subprocess.Popen(  # nosec B603 - fixed argv: our own helper, no shell
                [sys.executable, "-I", str(_WORKER)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, encoding="utf-8",
                errors="replace", creationflags=_CREATION_FLAGS)
        except OSError as exc:
            return DialogResult.unavailable(
                f"could not start the file dialog helper ({exc})")
        try:
            try:
                stdout, stderr = proc.communicate(
                    json.dumps(request), timeout=DIALOG_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                _stop(proc)
                return DialogResult.unavailable(
                    f"the file dialog was left open for {DIALOG_TIMEOUT_S:.0f} s "
                    "and was closed")
        finally:
            if proc.poll() is None:     # any other way out: never leave a window behind
                _stop(proc)
        return _parse_reply(proc.returncode, stdout, stderr)


def ask_open(kind: str = "vector", title: str = "Open file") -> DialogResult:
    return _run({"mode": "open", "title": title,
                 "filetypes": FILETYPES.get(kind) or FILETYPES["vector"]})


def ask_save(kind: str = "routes", title: str = "Save as",
             defaultextension: str = "") -> DialogResult:
    return _run({"mode": "save", "title": title,
                 "defaultextension": defaultextension,
                 "filetypes": FILETYPES.get(kind) or FILETYPES["routes"]})


def ask_directory(title: str = "Choose folder") -> DialogResult:
    return _run({"mode": "directory", "title": title})
