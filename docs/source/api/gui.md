---
title: "Interactive GUI"
summary: "Browser or desktop workbench: draw an area, load data, edit costs, rasterise and route."
status: unreleased
since: "unreleased"
available_in: source
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
```

From Python:

```python
from pyorps.gui import launch, build_app

launch(host="127.0.0.1", port=8050, debug=False, desktop=False)
app = build_app()   # a Dash app, for embedding or testing
```

`launch(state=None, *, host='127.0.0.1', port=8050, debug=False, desktop=False)` runs the server. `build_app(state=None, *, compress=None, **dash_kwargs)` returns the Dash app; response compression is on whenever `flask-compress` is installed.

(gui-notes)=
## Notes

- The GUI keeps all heavy objects on the server in one project state; the browser holds only small view state.
- Route edits are non-destructive: every edit creates a new route layer that remembers its parent.
- The older `pyorps.webviz` viewer is deprecated and warns on import; the `viz` extra is an alias of `gui`.
- The GUI needs Chromium only for its end-to-end tests, not for use.
