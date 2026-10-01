"""
PYORPS webviz (DEPRECATED): use :mod:`pyorps.gui` instead.

``pyorps.webviz`` is replaced by the rebuilt ``pyorps.gui`` package (blank-map
workflow: draw a study area, load data, calibrate costs, rasterize, build and
edit routes non-destructively). This package stays importable for one release
and re-exports its still-correct services from ``pyorps.gui.services``; it
will be removed in the release after next.

Migration::

    # old                                    # new
    from pyorps.webviz import RouteViewer    from pyorps.gui import launch
    RouteViewer.from_path_finder(f).launch() launch()   # or launch(desktop=True)

Install the new extra: ``pip install "pyorps[gui]"`` (the ``[viz]`` extra is
kept as an alias for one release).
"""
import warnings as _warnings

_warnings.warn(
    "pyorps.webviz is deprecated and will be removed in a future release; "
    "use pyorps.gui instead (pip install \"pyorps[gui]\").",
    DeprecationWarning, stacklevel=2)
del _warnings

from .viewer import RouteViewer

__all__ = ["RouteViewer"]
