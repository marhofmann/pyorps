---
title: "Siting (research)"
summary: "Experimental footprint pricing, candidate lattices, bounds and site fields for free siting."
status: unreleased
since: "unreleased"
available_in: source
module: "pyorps.siting"
api:
  - pyorps.siting.Footprint
  - pyorps.siting.ScreenResult
  - pyorps.siting.rotated_kernel
  - pyorps.siting.screen_footprints
  - pyorps.siting.verify_exact
  - pyorps.siting.CandidateSet
  - pyorps.siting.candidate_lattice
  - pyorps.siting.cell_xy
  - pyorps.siting.sample_field
  - pyorps.siting.OverheadScreenBound
  - pyorps.siting.assert_overhead_bound_preconditions
  - pyorps.siting.collection_cost_upper_bound
  - pyorps.siting.lipschitz_stride_gap
  - pyorps.siting.local_lipschitz_gap
  - pyorps.siting.min_tower_cost_eur
  - pyorps.siting.overhead_screen_lower_bound
  - pyorps.siting.LinkCosts
  - pyorps.siting.SitingExport
  - pyorps.siting.export_tower_fields
  - pyorps.siting.Terminal
  - pyorps.siting.load_terminals
  - pyorps.siting.terminals_from_frame
  - pyorps.siting.SiteField
  - pyorps.siting.footprint_mask
  - pyorps.siting.site_values
  - pyorps.siting.site_field.site_field
  - pyorps.siting.ETA_FLOAT32
  - pyorps.siting.float32_lower_bound
  - pyorps.siting.offset_min_filter
  - pyorps.siting.steiner_lower_bound
  - pyorps.siting.verified_110kv_towers
  - pyorps.siting.DataManifest
  - pyorps.siting.LayerEntry
  - pyorps.siting.RasterWindow
  - pyorps.siting.ManifestError
  - pyorps.siting.KINDS
---
# Siting: footprints, candidates, bounds and site fields

`pyorps.siting` supports free siting: a substation may stand on any cell of a
raster, not only on a short list of candidates. The package prices the
footprint of the facility at every cell, builds a lattice of feasible
candidate positions, gives lower and upper bounds that say in which direction
they err, and exports the candidate table and fields. It does not choose among
candidates. That is the job of an optimiser outside this package.

**Status: experimental research code.** The mathematics is tested on synthetic
instances. The method is not complete, and some inputs (site prices, pad size)
are placeholders. Interfaces may change, and nothing here produces a result
that can be quoted. The package is in the development tree. A PyPI release may hold an older, partial
copy of it, so use a source checkout (see {doc}`../getting_started/release_status`). It is not imported by `pyorps`;
import it as `pyorps.siting`.

---

(siting-public-names)=
## Public names

All names below are available as `pyorps.siting.<name>`. Importing the package
needs no optional package. Reading a data manifest needs PyYAML when it is
called. The footprint screen needs SciPy.

### `pyorps.siting.footprint`

| Name | What it does |
|---|---|
| `Footprint` | A rectangular facility footprint and the rotations to try. |
| `rotated_kernel` | Fractional-coverage rotated rectangle kernel. |
| `screen_footprints`, `ScreenResult` | The cheapest feasible orientation of the footprint at every cell, by FFT, and its result. |
| `verify_exact` | Re-scores placements without the FFT. |

### `pyorps.siting.candidates`

| Name | What it does |
|---|---|
| `candidate_lattice`, `CandidateSet` | Feasible candidate centres on a stride grid, eroded for the footprint. |
| `cell_xy` | Cell centres as map coordinates for an affine transform. |
| `sample_field` | Field value at map coordinates, with an explicit cell-selection rule. |

### `pyorps.siting.bounds`

| Name | What it does |
|---|---|
| `overhead_screen_lower_bound`, `OverheadScreenBound` | A lower bound on an overhead line's cost to each candidate, with its provenance. |
| `assert_overhead_bound_preconditions` | Refuses an overhead screen whose preconditions do not hold. |
| `min_tower_cost_eur` | The cheapest any single intermediate tower can be. |
| `collection_cost_upper_bound` | An upper bound on collection cost (sum of radial connections). |
| `lipschitz_stride_gap`, `local_lipschitz_gap` | How much a stride lattice can miss, globally and per candidate. |

### `pyorps.siting.export`, `terminals`

| Name | What it does |
|---|---|
| `export_tower_fields`, `SitingExport`, `LinkCosts` | The candidate table and per-terminal link costs handed to an optimiser. |
| `Terminal`, `terminals_from_frame`, `load_terminals` | Fixed terminals (connection points, turbines) named by id. |

### `pyorps.siting.site_field`

| Name | What it does |
|---|---|
| `SiteField` | The cheapest rotation per anchor cell, in integer cents. |
| `footprint_mask` | The pad's pixel set at one rotation, as a boolean kernel. |
| `site_values` | Direct pixel sums at a few anchors, as an independent check. |

The function `site_field` is not re-exported: its name would hide the module.
Import it with `from pyorps.siting.site_field import site_field`.

### `pyorps.siting.lattice_bounds`, `hv_towers`, `data_manifest`

| Name | What it does |
|---|---|
| `float32_lower_bound`, `ETA_FLOAT32` | A float64 lower bound of the exact field from a float32 drain, and its error constant. |
| `steiner_lower_bound` | Admissible bound of the trench weight that connects all turbines to every root cell. |
| `offset_min_filter` | Minimum of a field over a square around every cell. |
| `verified_110kv_towers` | The towers on OpenStreetMap lines whose voltage matches. |
| `DataManifest`, `LayerEntry`, `RasterWindow`, `ManifestError`, `KINDS` | The data manifest of a case study: layers, hashes and raster windows. |

---

(siting-example)=
## Minimal example

Screen a 40 m by 20 m pad on a synthetic 100 by 100 cell raster with 5 m cells
and one blocked area, then build a candidate lattice.

```python
import numpy as np
from affine import Affine

from pyorps.siting import Footprint, candidate_lattice, screen_footprints

rng = np.random.default_rng(4)
cost = rng.random((100, 100)).astype(np.float32) * 50 + 10     # price per cell
blocked = np.zeros((100, 100), np.float32)                      # 1 where nothing may be built
blocked[60:80, 20:50] = 1.0
transform = Affine(5.0, 0.0, 400000.0, 0.0, -5.0, 5500000.0)   # 5 m cells

screen = screen_footprints(
    cost, blocked,
    footprint=Footprint(40.0, 20.0, rotation_step_deg=30.0),
    transform=transform, resolution_m=5.0,
)
candidates = candidate_lattice(screen, stride_m=25.0)
print(screen.shape, len(candidates))
```

The bounds and the export need cost fields from a `PathFinder` and a tower
field profile. The site field and the data manifest need case-study layers.
None of them has an example here.

---

(siting-layering)=
## Layering

`siting` imports no other research package; it uses the core only. Nothing in
the core imports it. A test (`tests/test_core/test_package_layering.py`)
checks this rule. See also {doc}`collector`, {doc}`certify` and
{doc}`costmodel`.
