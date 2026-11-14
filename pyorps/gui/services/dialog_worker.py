"""
Helper process for :mod:`pyorps.gui.services.dialogs`: shows ONE native dialog.

The GUI server never imports tkinter itself. Each dialog runs here, in a
short-lived child process (``python -I dialog_worker.py``), so Tk lives on the
main thread of its own interpreter (macOS aborts when Tk is created off the
main thread) and a missing display or a missing tkinter cannot take the server
down. Standard library only, so the child starts without importing pyorps.

Protocol: one JSON request on stdin, one JSON line on stdout::

    in : {"mode": "open"|"save"|"directory", "title": str,
          "filetypes": [[label, patterns], ...], "defaultextension": str}
    out: {"status": "path", "path": str}
         {"status": "cancelled"}
         {"status": "unavailable", "reason": str}
"""
from __future__ import annotations

import json
import sys


def _unavailable(reason: str) -> dict:
    return {"status": "unavailable", "reason": reason}


def handle(request: dict) -> dict:
    """Show the requested dialog and return the reply dict (never raises)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        return _unavailable(f"tkinter is not installed ({exc})")
    try:
        root = tk.Tk()
    except Exception as exc:  # tk.TclError: no display name / no $DISPLAY
        return _unavailable(f"no display is available ({exc})")
    try:
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:  # some window managers refuse; the dialog still works
            pass
        mode = request.get("mode")
        options = {"parent": root, "title": str(request.get("title") or "")}
        filetypes = [(str(label), str(patterns))
                     for label, patterns in request.get("filetypes") or []]
        if mode == "open":
            chosen = filedialog.askopenfilename(filetypes=filetypes, **options)
        elif mode == "save":
            chosen = filedialog.asksaveasfilename(
                defaultextension=str(request.get("defaultextension") or ""),
                filetypes=filetypes, **options)
        elif mode == "directory":
            chosen = filedialog.askdirectory(**options)
        else:
            return _unavailable(f"unknown dialog mode {mode!r}")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return _unavailable(f"the file dialog failed ({exc})")
    finally:
        try:
            root.destroy()
        except Exception:  # nosec B110  # pylint: disable=broad-exception-caught
            pass
    # Tk answers '' (or an empty tuple) when the user cancels
    return {"status": "path", "path": str(chosen)} if chosen else {"status": "cancelled"}


def main() -> int:
    try:
        request = json.loads(sys.stdin.buffer.read().decode("utf-8") or "{}")
    except ValueError as exc:
        reply = _unavailable(f"unreadable dialog request ({exc})")
    else:
        reply = handle(request if isinstance(request, dict) else {})
    # ASCII-only JSON: the pipe's code page cannot garble a path with umlauts
    sys.stdout.write(json.dumps(reply) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
