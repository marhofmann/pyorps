"""Cost inputs for the siting study: catalogue, losses, consent classes, access road, HV leg.

Experimental research code. The mathematics is tested on synthetic instances;
the method is not complete, many input values are placeholders tagged
``assumption``, and interfaces may change. Nothing here produces a result that
can be quoted.

Modules exported by this package (each name is available as
``pyorps.costmodel.<name>``):

* :mod:`.catalogue`: the sourced cost catalogue with confidence tiers.
* :mod:`.losses`: the capitalisation of load losses from joint power
  categories (annuity factor, loss price, loss weights).
* :mod:`.consent`: priced consent classes for protected areas.
* :mod:`.access_road`: constraints and array functions for the construction
  access road to the substation pad.
* :mod:`.hv_leg`: the weight per metre of the 110 kV cable leg.
* :mod:`.branch_constants`: the constants of one design branch.

``FORBIDDEN`` is defined in :mod:`.access_road`, :mod:`.consent` and
:mod:`.hv_leg` and is not re-exported here; read it from the module you use.

:mod:`pyorps.costmodel.collector_inputs` imports :mod:`pyorps.collector` and so is
NOT imported here. Import it as a submodule when needed (it builds a
``CollectorModel`` from the inputs file, the catalogue and the loss table).

The loaders read YAML files and need PyYAML. Importing the package does not.

Layering: ``core <- collector <- costmodel``. This package is not imported by
``pyorps/__init__``.
"""

from pyorps.costmodel.access_road import (
    AccessConstraints,
    band_multiplier,
    block_cost,
    class_cost,
    clearance,
    slope_multiplier,
    slope_percent,
)
from pyorps.costmodel.branch_constants import BranchConstants
from pyorps.costmodel.catalogue import (
    CONFIDENCE_TOLERANCE,
    CostCatalogue,
    CostItem,
)
from pyorps.costmodel.consent import PRICE_CAP, ConsentClass, ConsentTable
from pyorps.costmodel.hv_leg import HvLegConstraints
from pyorps.costmodel.losses import (
    LossCategories,
    annuity_factor,
    category_square_sums,
    loss_coef,
    loss_price_eur_per_mwh,
)

__all__ = [
    "CONFIDENCE_TOLERANCE",
    "PRICE_CAP",
    "AccessConstraints",
    "BranchConstants",
    "ConsentClass",
    "ConsentTable",
    "CostCatalogue",
    "CostItem",
    "HvLegConstraints",
    "LossCategories",
    "annuity_factor",
    "band_multiplier",
    "block_cost",
    "category_square_sums",
    "class_cost",
    "clearance",
    "loss_coef",
    "loss_price_eur_per_mwh",
    "slope_multiplier",
    "slope_percent",
]
