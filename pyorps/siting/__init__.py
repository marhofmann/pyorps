"""Free siting: candidate lattices, footprint pricing, bounds, export.

The generalisation of what ``case_studies/substation_planning2`` and
TopoMILP's ``examples/cired2027`` each did once, bespoke and locked to a
single free facility. ``CostField`` / ``CostFieldSet`` /
``SavedCostField`` were already general and shipped; the WORKFLOW above
them was not, and this package is that workflow.

What it does::

    screen = screen_footprints(cost, blocked, footprint=fp, ...)
    cands  = candidate_lattice(screen, stride_m=5.0)
    with finder.cost_fields(terminals) as fields:
        cable = fields.costs_to(cands.points)
    lb, ub = tower_field_bounds(raster, profile=profile, ...)
    export = export_tower_fields(cands, {"PCC0": (lb, ub)})
    export.write_npz("candidates.npz")

What it deliberately does NOT do: choose. Selecting among candidates,
pairing two free facilities, enforcing separation between them -- that
is a MILP's job and it is out of PYORPS' scope by decision. Dropping it
is what retired the ``a = b`` collapse (a min-plus chain fold silently
places both facilities on the same cell, because ``d(a, a) = 0``), the
whole chain abstraction, and the seeded multi-source settle that only
existed to serve them.

Where a bound is involved, this package names the direction it errs in
and refuses when its preconditions do not hold; see
:mod:`pyorps.siting.bounds`.
"""

from pyorps.siting.bounds import (
    OverheadScreenBound,
    assert_overhead_bound_preconditions,
    collection_cost_upper_bound,
    lipschitz_stride_gap,
    local_lipschitz_gap,
    min_tower_cost_eur,
    overhead_screen_lower_bound,
)
from pyorps.siting.candidates import (
    CandidateSet,
    candidate_lattice,
    cell_xy,
    sample_field,
)
from pyorps.siting.export import LinkCosts, SitingExport, export_tower_fields
from pyorps.siting.footprint import (
    Footprint,
    ScreenResult,
    rotated_kernel,
    screen_footprints,
    verify_exact,
)
from pyorps.siting.terminals import (
    Terminal,
    load_terminals,
    terminals_from_frame,
)

__all__ = [
    "CandidateSet",
    "Footprint",
    "LinkCosts",
    "OverheadScreenBound",
    "ScreenResult",
    "SitingExport",
    "Terminal",
    "assert_overhead_bound_preconditions",
    "candidate_lattice",
    "cell_xy",
    "collection_cost_upper_bound",
    "export_tower_fields",
    "lipschitz_stride_gap",
    "load_terminals",
    "local_lipschitz_gap",
    "min_tower_cost_eur",
    "overhead_screen_lower_bound",
    "rotated_kernel",
    "sample_field",
    "screen_footprints",
    "terminals_from_frame",
    "verify_exact",
]
