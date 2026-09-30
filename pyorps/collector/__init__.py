"""The trench-sharing MV collector (plan rev. 5, section 3.2 and Phase D4).

``D_share`` is the collector model of the proven-optimal siting study: a
trench tree in the routing raster, a radial electrical network of turbines,
optional switching stations and the UW, cables that share trenches (paid
once, plus a per-cable overhead, with group derating), one cable option
per connection, parallel cables allowed. :mod:`.model` states it,
:mod:`.reference` solves it exactly for every UW position at once with a
cut-signature Dreyfus--Wagner, :mod:`.raster` runs the same recursion as
seeded drains on a raster window, :mod:`.design` re-prices a solution
independently (on a raster through :mod:`.raster_pricer`),
:mod:`.bounds` holds the Stage-2 lower bound and the completion bounds
that prune, and :mod:`.exact` couples engines A and B into one certified
field.

This package is deliberately NOT imported by ``pyorps/__init__``.
"""

from pyorps.collector.bounds import (
    completion_bounds_raster,
    mu_star,
    rho_hat,
    rho_hat_lower_bound,
    rho_hat_lower_bound_raster,
)
from pyorps.collector.design import Design, DesignError, Junction, System, reprice
from pyorps.collector.exact import ExactCollectorField, exact_collector_field
from pyorps.collector.model import CableType, CollectorModel
from pyorps.collector.raster import RasterCollector, graph_from_raster
from pyorps.collector.raster_pricer import RasterStepPricer
from pyorps.collector.reference import (
    CollectorGraph,
    CollectorResult,
    solve_collector,
)

__all__ = [
    "CableType",
    "CollectorGraph",
    "CollectorModel",
    "CollectorResult",
    "Design",
    "DesignError",
    "ExactCollectorField",
    "Junction",
    "RasterCollector",
    "RasterStepPricer",
    "System",
    "completion_bounds_raster",
    "exact_collector_field",
    "graph_from_raster",
    "mu_star",
    "reprice",
    "rho_hat",
    "rho_hat_lower_bound",
    "rho_hat_lower_bound_raster",
    "solve_collector",
]
