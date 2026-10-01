"""
PYORPS GUI — interactive route-planning workbench (Dash + Leaflet).

Start from a blank map: draw a study area, load vector data (local files or
WFS), calibrate the cost model in an editable table, rasterize, build routes
(with chained waypoints), and edit them non-destructively — every edit spawns
a new route with lineage back to its parent.

Usage::

    from pyorps.gui import launch
    launch()                       # browser
    launch(desktop=True)           # pywebview desktop window

or from the command line::

    python -m pyorps.gui [--desktop] [--port 8050]

Requires the ``gui`` extra: ``pip install pyorps[gui]``.
"""
from .app import build_app, launch
from .state import CostLayerConfig, Layer, ProjectState

__all__ = ["build_app", "launch", "ProjectState", "Layer", "CostLayerConfig"]
