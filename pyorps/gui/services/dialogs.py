"""
PYORPS GUI: native file/folder pickers (tkinter).

The Dash server runs on the user's own machine, so the familiar Windows
open/save dialogs can be raised server-side and their result written into
the path inputs. Each dialog runs in its own short-lived thread with a fresh
Tk root (Tk objects are not reusable across threads); a lock keeps it to one
dialog at a time. Returns ``(path_or_None, error_or_None)`` — cancel is
``(None, None)``.
"""
from __future__ import annotations

import threading
from typing import Callable

_dialog_lock = threading.Lock()

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


def _run(fn: Callable) -> tuple[str | None, str | None]:
    result: dict = {}

    def worker():
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            try:
                result["path"] = fn(filedialog, root) or None
            finally:
                root.destroy()
        except Exception as exc:  # pragma: no cover - headless machines
            result["error"] = str(exc)

    with _dialog_lock:
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(timeout=300)
    if thread.is_alive():  # pragma: no cover - dialog left open forever
        return None, "file dialog timed out"
    return result.get("path"), result.get("error")


def ask_open(kind: str = "vector",
             title: str = "Open file") -> tuple[str | None, str | None]:
    return _run(lambda fd, root: fd.askopenfilename(
        parent=root, title=title, filetypes=FILETYPES.get(kind) or
        FILETYPES["vector"]))


def ask_save(kind: str = "routes", title: str = "Save as",
             defaultextension: str = "") -> tuple[str | None, str | None]:
    return _run(lambda fd, root: fd.asksaveasfilename(
        parent=root, title=title,
        defaultextension=defaultextension,
        filetypes=FILETYPES.get(kind) or FILETYPES["routes"]))


def ask_directory(title: str = "Choose folder") \
        -> tuple[str | None, str | None]:
    return _run(lambda fd, root: fd.askdirectory(parent=root, title=title))
