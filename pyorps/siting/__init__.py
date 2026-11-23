"""Free siting: footprint pricing, candidate lattices, bounds, site fields, export.

Experimental research code. The mathematics is tested on synthetic instances;
the method is not complete, some inputs (prices, pad size) are placeholders,
and interfaces may change. Nothing here produces a result that can be quoted.

A substation may stand on any cell. This package prices a footprint at every
cell, builds a lattice of feasible candidate positions, gives lower and upper
bounds that state their direction and refuse when their preconditions do not
hold, and exports candidate tables and fields. It does not choose among
candidates: that is the job of an optimiser outside this package.

Modules, all exported here (each name is available as
``pyorps.siting.<name>``):

* :mod:`.footprint`: rotated footprint kernels and the FFT screen of the build
  cost at every cell.
* :mod:`.candidates`: candidate positions on a lattice and field sampling.
* :mod:`.bounds`: screening bounds with their preconditions made executable.
* :mod:`.export`: the candidate table and fields handed to an optimiser.
* :mod:`.terminals`: fixed terminals (connection points, turbines) named by id.
* :mod:`.site_field`: the exact pad cost at every anchor cell and rotation, in
  integer cents. ``SiteField``, ``footprint_mask`` and ``site_values`` are
  re-exported. The function ``site_field`` would hide the module, so import it
  from :mod:`pyorps.siting.site_field`.
* :mod:`.lattice_bounds`: lower-bound pieces on a position lattice.
* :mod:`.hv_towers`: 110 kV towers from OpenStreetMap lines.
* :mod:`.data_manifest`: the data manifest of a case study (layers, hashes,
  windows). Reading it needs PyYAML when called, not when imported.

Layering: ``siting`` imports no other research package. It is not imported by
``pyorps/__init__``.
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
from pyorps.siting.data_manifest import (
    KINDS,
    DataManifest,
    LayerEntry,
    ManifestError,
    RasterWindow,
)
from pyorps.siting.export import LinkCosts, SitingExport, export_tower_fields
from pyorps.siting.footprint import (
    Footprint,
    ScreenResult,
    rotated_kernel,
    screen_footprints,
    verify_exact,
)
from pyorps.siting.hv_towers import verified_110kv_towers
from pyorps.siting.lattice_bounds import (
    ETA_FLOAT32,
    float32_lower_bound,
    offset_min_filter,
    steiner_lower_bound,
)
from pyorps.siting.site_field import SiteField, footprint_mask, site_values
from pyorps.siting.terminals import (
    Terminal,
    load_terminals,
    terminals_from_frame,
)

__all__ = [
    "ETA_FLOAT32",
    "KINDS",
    "CandidateSet",
    "DataManifest",
    "Footprint",
    "LayerEntry",
    "LinkCosts",
    "ManifestError",
    "OverheadScreenBound",
    "RasterWindow",
    "ScreenResult",
    "SiteField",
    "SitingExport",
    "Terminal",
    "assert_overhead_bound_preconditions",
    "candidate_lattice",
    "cell_xy",
    "collection_cost_upper_bound",
    "export_tower_fields",
    "float32_lower_bound",
    "footprint_mask",
    "lipschitz_stride_gap",
    "load_terminals",
    "local_lipschitz_gap",
    "min_tower_cost_eur",
    "offset_min_filter",
    "overhead_screen_lower_bound",
    "rotated_kernel",
    "sample_field",
    "screen_footprints",
    "site_values",
    "steiner_lower_bound",
    "terminals_from_frame",
    "verified_110kv_towers",
    "verify_exact",
]
