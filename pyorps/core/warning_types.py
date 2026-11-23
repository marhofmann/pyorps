"""
Root of the warning categories defined by pyorps.

It lives in its own module, not in ``exceptions.py``, because every warning
class is also a subclass of ``Exception``: a module that lists pyorps's
exceptions (for example the GUI's error registry) would otherwise pick it up
as a raisable error. ``pyorps.core.exceptions`` re-exports it for convenience.
"""


class PyorpsWarning(UserWarning):
    """
    Root of every warning category pyorps defines.

    ``warnings.simplefilter("error", PyorpsWarning)`` turns all pyorps warnings
    into errors; ``simplefilter("ignore", PyorpsWarning)`` silences them. The
    class derives from ``UserWarning``, so filters written for ``UserWarning``
    keep working.
    """


class CrsReprojectionWarning(PyorpsWarning):
    """Input data in a geographic CRS (degrees) was reprojected to UTM (metres).

    The raster and every route are in the new CRS: source and target
    coordinates have to be given in it.
    """


class AlgorithmFallbackWarning(PyorpsWarning):
    """A different search runs than the one that was asked for.

    The answer is still the one the request defines (terrain is never dropped
    to keep the requested search); only the route to it changed. Raised once
    per ``PathFinder`` or ``SearchSession``, not once per route. Typical case:
    delta-stepping cannot evaluate slope terms, so Dijkstra runs instead.
    """


class CoverageWarning(PyorpsWarning):
    """A layer or the terrain model does not cover part of the search area.

    Raised for a metric layer with cells that have no value (outside its
    source, nodata or NaN), for a terrain model (DEM) with cells that have no
    elevation, and when a source or target is moved because the terrain model
    has no elevation there. The result is still valid under the documented
    rule (a void layer cell counts as 0, a cell without elevation is
    forbidden); the warning says that the data has a gap.
    """


class SpanRuleWarning(PyorpsWarning):
    """The returned overhead line breaks a span or clearance rule of its profile.

    Raised once per route, by ``ConstrainedPathFinder``, with one line per
    case: ``terminal_span_rule="remove"`` took out a tower next to an end
    point and the merged span reaches ``max_span_m``, leaves a bend without a
    tower or fails the clearance check; or an end span (the last span is never
    checked by the search) fails the clearance check with the heights the
    report states. The route and all its numbers are still returned; the
    warning names the rule, the span and the cause.
    """
