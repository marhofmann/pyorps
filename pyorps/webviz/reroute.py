"""DEPRECATED shim: control-point routing moved to
:mod:`pyorps.gui.services.routing` (Section 15)."""
from __future__ import annotations

from shapely.geometry import LineString

from pyorps.gui.services.routing import route_through_points  # noqa: F401


def reroute_through_waypoints(finder, base_line: LineString,
                              waypoints: list[tuple[float, float]]):
    """Route from base_line's start through waypoints to its end."""
    source = tuple(base_line.coords[0])
    target = tuple(base_line.coords[-1])
    return route_through_points(finder, [source, *waypoints, target])
