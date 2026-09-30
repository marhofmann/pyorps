"""Cost-model closure for the proven-optimal siting study (plan rev. 5, Phase C).

:mod:`.catalogue` reads the sourced cost catalogue with its confidence
tiers (C1); :mod:`.losses` turns the joint power-category table (C11) into
the loss weights and the capitalisation the collector and HV engines use.

This package is deliberately NOT imported by ``pyorps/__init__``.
"""

from pyorps.costmodel.catalogue import (
    CONFIDENCE_TOLERANCE,
    CostCatalogue,
    CostItem,
)
from pyorps.costmodel.losses import (
    LossCategories,
    annuity_factor,
    category_square_sums,
    loss_coef,
    loss_price_eur_per_mwh,
)

__all__ = [
    "CONFIDENCE_TOLERANCE",
    "CostCatalogue",
    "CostItem",
    "LossCategories",
    "annuity_factor",
    "category_square_sums",
    "loss_coef",
    "loss_price_eur_per_mwh",
]
