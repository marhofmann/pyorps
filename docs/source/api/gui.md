---
title: "Interactive GUI"
summary: "Browser or desktop workbench: draw an area, load data, edit costs, rasterise and route."
status: experimental
since: "0.4.0"
available_in: pypi
module: "pyorps.gui"
api:
  - pyorps.gui.launch
  - pyorps.gui.build_app
---
# Interactive GUI

`pyorps.gui` is a browser or desktop workbench for the whole routing workflow: draw an area, load data (files, WFS services, OpenStreetMap), edit the cost table, rasterise, then route and edit routes.

(gui-install)=
## Install and start

```bash
pip install pyorps[gui]          # dash, dash-leaflet, dash-ag-grid and friends
python -m pyorps.gui             # opens in the browser
python -m pyorps.gui --desktop   # native window (pywebview)
python -m pyorps.gui --port 8060
python -m pyorps.gui --project my_area/project.json   # open a saved project at start
```

From Python:

```python
from pyorps.gui import launch, build_app

launch(host="127.0.0.1", port=8050, debug=False, desktop=False)
app = build_app()   # a Dash app, for embedding or testing
```

`launch(...)` runs the server; every setting is also a command-line option (`python -m pyorps.gui --help`): `--desktop`, `--host` (default 127.0.0.1), `--port` (8050), `--allow-remote`, `--threads` (8), `--work-dir`, `--compress`/`--no-compress` and `--debug`. A host other than the local machine is refused unless `--allow-remote` is given, because the server has no authentication and its path fields read and write files on the machine it runs on. `build_app(state=None, *, compress=None, **dash_kwargs)` returns the Dash app; response compression is on whenever `flask-compress` is installed.

`--project FILE` (a `project.json`, or a folder that holds one; a relative path is taken from the current folder) opens a saved project at start, in the browser and with `--desktop`, through the same code as the "Open project" button. A missing or broken file prints one error line and ends with exit code 1 before a server or window starts. This flag and the cards described below are on `main`; they are not part of release 0.4.0.

(gui-routes-settings)=
## Settings in the Routes tab

Every card is collapsible; its settings are saved in the project file and restored after "Open project" and on a page reload. A value that cannot be used is named in a notice with its allowed range and unit, and the "Go fix" button of the notice opens the card and focuses the field.

- **Algorithm & search**: backend, algorithm, neighbourhood, search buffer, whole-raster switch, threads.
- **Objective & terrain** (plain routing): weights for cost, length and slope, extra metric layers (a loaded raster layer, or the slope derived from the terrain model) with optional hard limits, a terrain model (DEM) with the slope response, and 16-bit or 32-bit weights. With the default values the route is the classic least-cost route. The status line of the card names a conflict before a run. A terrain model needs the Dijkstra algorithm: with a delta-stepping algorithm, or with the `raster_fim` backend, the GUI refuses the run and points at the field to change (the library itself would fall back to Dijkstra with an `AlgorithmFallbackWarning`, see {ref}`metric-objective`). The terrain model must be in the coordinate system of the cost raster and must cover the routing points. A route computed with an objective is still priced on the plain cost raster, so its cost is comparable with other routes; an edit of such a route uses the same objective, terrain model and layers again.
- **Constrained routing**: the infrastructure profile is a form (fields and three tables with their limits and units). Form edits keep the YAML text current, and keys that the form does not show are kept; "Raw YAML (advanced)" holds the text, and "YAML → form" fills the form from it.
- **Cost field**: the cost to reach every cell from the sources or from all control points, summed or as the lowest of the points, added as a raster layer; the card shows the cheapest cell. Unreachable and forbidden cells are shown as no data. The field uses the plain cost raster (the objective and terrain settings are not applied). A run that would need more memory than the limit is refused with the numbers and with what to change.
- **Corridor graph**: the shared-trench corridor graph between the control points on the plain cost raster (`cython` backend), with the layers "Corridor segments", "Corridor shared runs" and "Corridor nodes", a summary, and an export as GeoPackage (an existing file is replaced only after confirmation). The graph is not saved in the project file; build it again after reopening a project.

(gui-jobs)=
## Background runs and Stop

One background run is active at a time; a second click gets a notice that names the running one. The result is added to the project when the run has finished.

| Run | Goes to the background | Stop |
|---|---|---|
| Plain routing | heavy runs (after confirmation) | between two point pairs |
| Rasterize | more than 20 million cells | at the next stage (before the base burn, each modifier, the crop, the save); a stopped run builds no raster |
| Constrained routing | heavy runs, and runs whose effort cannot be estimated | after the segment in flight; a run without waypoints is one segment |
| Cost field | always | between two fixed points; a stopped run adds no layer |
| Corridor graph | always | cannot be stopped |

Smaller rasterize and constrained runs stay in the foreground, as before.

(gui-notes)=
## Notes

- The GUI keeps all heavy objects on the server in one project state; the browser holds only small view state.
- An edit replaces the edited route. The replaced route is kept in memory as a one-step undo ("Undo last edit"); it is not saved with the project.
- A raster layer releases its tile server client and the raster file when the layer is removed or replaced, when a project is closed, and at exit.
- The older `pyorps.webviz` viewer is deprecated and warns on import; the `viz` extra is an alias of `gui`.
- The GUI needs Chromium only for its end-to-end tests, not for use.
