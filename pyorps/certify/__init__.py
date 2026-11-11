"""Certificates for bound-and-prune siting: window, line-integral, prune and relax-and-verify.

Experimental research code. The mathematics is tested on synthetic instances;
the method is not complete and interfaces may change. Nothing here produces
a result that can be quoted.

Modules exported by this package (each name is available as
``pyorps.certify.<name>``):

* :mod:`.prune`: the bound sum, the prune rule and the bound-and-prune loop
  with its optimality verdict.
* :mod:`.windows`: where a field computed on a window equals the unwindowed
  field, and the lower bound elsewhere (also re-exports ``drain``,
  ``boundary_cells`` and ``step_table`` from :mod:`pyorps.graph.drain`).
* :mod:`.line_integral`: re-prices a route by the exact line integral of the
  cost raster.
* :mod:`.rv`: the relax-and-verify driver, the consent lattice and the epsilon
  manifest.

Two modules import :mod:`pyorps.collector` and so are NOT imported here. Import
them as submodules when needed (``import pyorps.certify.voltage``):

* :mod:`pyorps.certify.feasible_incumbent`: checks a priced design against the
  voltage rules and repairs it.
* :mod:`pyorps.certify.voltage`: relax-and-verify with the voltage limit.

Layering: ``core <- collector <- certify``. This package is not imported by
``pyorps/__init__``.
"""

from pyorps.certify.line_integral import fidelity_delta, line_integral_cost
from pyorps.certify.prune import (
    BoundAndPruneResult,
    bound_and_prune,
    branch_lower_bound,
    prune_mask,
)
from pyorps.certify.rv import (
    ConsentResult,
    EpsilonManifest,
    RVResult,
    consent_lattice,
    relax_and_verify,
)
from pyorps.certify.windows import (
    TreeCertificate,
    WindowCertificate,
    WindowGrowth,
    boundary_cells,
    certify_path_field,
    certify_tree_field,
    drain,
    ellipse_window,
    grow_window,
    step_table,
)

__all__ = [
    "BoundAndPruneResult",
    "ConsentResult",
    "EpsilonManifest",
    "RVResult",
    "TreeCertificate",
    "WindowCertificate",
    "WindowGrowth",
    "boundary_cells",
    "bound_and_prune",
    "branch_lower_bound",
    "certify_path_field",
    "certify_tree_field",
    "consent_lattice",
    "drain",
    "ellipse_window",
    "fidelity_delta",
    "grow_window",
    "line_integral_cost",
    "prune_mask",
    "relax_and_verify",
    "step_table",
]
