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
