"""Certificates for the proven-optimal siting study (plan rev. 5, D2/D7/H).

:mod:`.windows` decides where a field computed on a window equals the
unwindowed field (plan section 3.5), and gives the lower bound elsewhere.
:mod:`.line_integral` re-prices a route by the exact line integral of the
cost raster (plan D9), for the Stage-5 fidelity gap. :mod:`.rv` holds the
relax-and-verify driver, the consent lattice and the epsilon manifest
(plan sections 3.6 and 3.7).

This package is deliberately NOT imported by ``pyorps/__init__``.
"""

from pyorps.certify.line_integral import fidelity_delta, line_integral_cost
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
    boundary_cells,
    certify_path_field,
    certify_tree_field,
    drain,
    ellipse_window,
    step_table,
)

__all__ = [
    "ConsentResult",
    "EpsilonManifest",
    "RVResult",
    "TreeCertificate",
    "WindowCertificate",
    "boundary_cells",
    "certify_path_field",
    "certify_tree_field",
    "consent_lattice",
    "drain",
    "ellipse_window",
    "fidelity_delta",
    "line_integral_cost",
    "relax_and_verify",
    "step_table",
]
