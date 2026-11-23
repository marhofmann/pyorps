---
title: "Cost model (research)"
summary: "Experimental cost catalogue, loss capitalisation and cost inputs for free siting."
status: unreleased
since: "unreleased"
available_in: source
module: "pyorps.costmodel"
api:
  - pyorps.costmodel.CostCatalogue
  - pyorps.costmodel.CostItem
  - pyorps.costmodel.CONFIDENCE_TOLERANCE
  - pyorps.costmodel.LossCategories
  - pyorps.costmodel.annuity_factor
  - pyorps.costmodel.category_square_sums
  - pyorps.costmodel.loss_price_eur_per_mwh
  - pyorps.costmodel.loss_coef
  - pyorps.costmodel.ConsentClass
  - pyorps.costmodel.ConsentTable
  - pyorps.costmodel.PRICE_CAP
  - pyorps.costmodel.AccessConstraints
  - pyorps.costmodel.class_cost
  - pyorps.costmodel.slope_percent
  - pyorps.costmodel.slope_multiplier
  - pyorps.costmodel.band_multiplier
  - pyorps.costmodel.clearance
  - pyorps.costmodel.block_cost
  - pyorps.costmodel.HvLegConstraints
  - pyorps.costmodel.BranchConstants
---
# Cost model: catalogue, losses and cost inputs

`pyorps.costmodel` turns sourced prices and operating assumptions into the
numbers the siting method uses: cable and plant prices with a confidence
tier, the capitalised cost of load losses, priced consent classes for
protected areas, an access-road cost model and the 110 kV leg.

**Status: experimental research code.** The mathematics is tested on synthetic
instances. The method is not complete, and many input values are placeholders
tagged `assumption` in the input files. Several model terms are still missing
or set to zero. Interfaces may change, and nothing here produces a result that
can be quoted. The package is in the development tree. A PyPI release may hold an older, partial
copy of it, so use a source checkout (see {doc}`../getting_started/release_status`). It is not imported by `pyorps`;
import it as `pyorps.costmodel`.

---

(costmodel-public-names)=
## Public names

All names below are available as `pyorps.costmodel.<name>`. The loaders read
YAML files and need PyYAML when they are called. Importing the package does not.

### `pyorps.costmodel.catalogue`

| Name | What it does |
|---|---|
| `CostCatalogue` | Loads and validates a cost catalogue (YAML) and reads items by key with their confidence tier. |
| `CostItem` | One sourced figure of the catalogue. |
| `CONFIDENCE_TOLERANCE` | The relative tolerance of each confidence tier. |

### `pyorps.costmodel.losses`

| Name | What it does |
|---|---|
| `annuity_factor` | Present value of 1 per year over a number of years. |
| `loss_price_eur_per_mwh` | Value of one MWh of loss energy. |
| `loss_coef` | Capitalised cost per unit of resistance and squared apparent power (`CollectorModel.loss_coef`). |
| `category_square_sums` | Squared-power sums for every turbine bitmask, from the power categories. |
| `LossCategories` | The power-category table and its derived quantities. |

### `pyorps.costmodel.consent`

| Name | What it does |
|---|---|
| `ConsentClass` | One protected-area class with its price and its bit in the flag raster. |
| `ConsentTable` | The classes, the water-zone multipliers and the delay price. |
| `PRICE_CAP` | The largest value a priced cell of the uint16 cost raster may take (65535 means forbidden). |

### `pyorps.costmodel.access_road`

| Name | What it does |
|---|---|
| `AccessConstraints` | The resolved access-road constraints. |
| `class_cost` | Cost of one land-use row from a class table. |
| `slope_percent` | Grade in percent from a DEM. |
| `slope_multiplier`, `band_multiplier` | Cut-and-fill multiplier by grade band, as an array and as a function. |
| `clearance` | Cells whose clear disc lies wholly in passable cells. |
| `block_cost` | Access cost of a block of raster rows. |

### Other modules

| Name | Module | What it does |
|---|---|---|
| `HvLegConstraints` | `hv_leg` | The resolved 110 kV cable leg model (weight per metre on the shared trench raster). |
| `BranchConstants` | `branch_constants` | The constants of one design branch. |

`FORBIDDEN` is defined in `access_road`, `consent` and `hv_leg`. It is not
re-exported; read it from the module you use.

### Submodule that is not imported by the package

`pyorps.costmodel.collector_inputs` imports `pyorps.collector`. Import it
directly. It builds a `CollectorModel` from the inputs file, the catalogue and
the loss table (`build_collector_model`, `CollectorInputs`, `BuiltCollector`).

---

(costmodel-example)=
## Minimal example

The loss functions need no data file. The numbers below are synthetic.

```python
from pyorps.costmodel import annuity_factor, loss_coef, loss_price_eur_per_mwh

af = annuity_factor(years=25.0, rate=0.04)              # 15.622
price = loss_price_eur_per_mwh(price=60.0, nu=0.05)     # EUR/MWh, synthetic
coef = loss_coef(price_eur_per_mwh=price, u_kv=30.0)    # for CollectorModel.loss_coef
print(round(af, 3), price, coef)
```

The catalogue, consent table, access-road and HV-leg loaders read YAML input
files from a case study. Those files are not part of the package, so there is
no example for them here.

---

(costmodel-layering)=
## Layering

`core <- collector <- costmodel`. `costmodel` may import `pyorps.collector`
(only `collector_inputs` does); the collector never imports `costmodel`. A test
(`tests/test_core/test_package_layering.py`) checks this rule. See also
{doc}`collector`, {doc}`certify` and {doc}`siting`.
