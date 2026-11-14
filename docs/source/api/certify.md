---
title: "Certify (research)"
summary: "Experimental bound-and-prune, window and relax-and-verify certificates for free siting."
status: unreleased
since: "unreleased"
available_in: source
module: "pyorps.certify"
api:
  - pyorps.certify.BoundAndPruneResult
  - pyorps.certify.bound_and_prune
  - pyorps.certify.branch_lower_bound
  - pyorps.certify.prune_mask
  - pyorps.certify.WindowCertificate
  - pyorps.certify.WindowGrowth
  - pyorps.certify.TreeCertificate
  - pyorps.certify.grow_window
  - pyorps.certify.certify_path_field
  - pyorps.certify.certify_tree_field
  - pyorps.certify.ellipse_window
  - pyorps.certify.drain
  - pyorps.certify.boundary_cells
  - pyorps.certify.step_table
  - pyorps.certify.line_integral_cost
  - pyorps.certify.fidelity_delta
  - pyorps.certify.EpsilonManifest
  - pyorps.certify.RVResult
  - pyorps.certify.relax_and_verify
  - pyorps.certify.ConsentResult
  - pyorps.certify.consent_lattice
---
# Certify: bound-and-prune and certificates

`pyorps.certify` holds the pieces that prove a choice of site optimal: a lower
bound for every position, a rule that drops positions that cannot win, and
checks that a field computed on a window equals the field on the whole raster.
It belongs to the research method for free siting of a wind-farm substation.

**Status: experimental research code.** The mathematics is tested on synthetic
instances. The method is not complete: there is no certificate file, no
independent checker and no per-connection-point prune yet. Its inputs are
partly placeholders. Interfaces may change, and nothing here produces a result
that can be quoted. The package is in the development tree. A PyPI release may hold an older, partial
copy of it, so use a source checkout (see {doc}`../getting_started/release_status`). It is not imported by `pyorps`;
import it as `pyorps.certify`.

---

(certify-public-names)=
## Public names

All names below are available as `pyorps.certify.<name>`.

### `pyorps.certify.prune`

| Name | What it does |
|---|---|
| `branch_lower_bound` | Adds a constant and non-negative term bounds into one lower-bound field. |
| `prune_mask` | True where a position is kept: its bound minus the margin is not above the incumbent. |
| `bound_and_prune` | Runs the incumbent loop, the exact evaluation of the kept set and the optimality check. |
| `BoundAndPruneResult` | The optimum, the argmin, the ties, the lower end, the verdicts and the kept counts. |

### `pyorps.certify.windows`

| Name | What it does |
|---|---|
| `certify_path_field` | Seeded path certificate: where a window field equals the full-raster field. |
| `certify_tree_field` | The same certificate for a collector field. |
| `WindowCertificate`, `TreeCertificate` | The results of the two certificates above. |
| `grow_window`, `WindowGrowth` | One capped regrowth of a window, and its outcome. |
| `ellipse_window` | The window of every cell a competitive collector can use. |
| `drain`, `step_table`, `boundary_cells` | Re-exported from `pyorps.graph.drain`: the drain kernel, the step table and the window border. |

### `pyorps.certify.line_integral`

| Name | What it does |
|---|---|
| `line_integral_cost` | Exact integral of the cost raster along a polyline through cell centres. |
| `fidelity_delta` | Discrete price, integral price and relative gap of one route. |

### `pyorps.certify.rv`

| Name | What it does |
|---|---|
| `relax_and_verify` | Relax-and-verify driver: verifies candidates in order of their relaxed bound. |
| `RVResult` | What `relax_and_verify` proved: best verified upper end, lower end, verdict. |
| `consent_lattice`, `ConsentResult` | Minimum over sets of consented protection classes, and its result. |
| `EpsilonManifest` | Named, sourced numerical allowances that add up to the margins. |

### Submodules that are not imported by the package

These two modules import `pyorps.collector`. Importing `pyorps.certify` does not
import the collector. Import them directly.

| Module | What it does |
|---|---|
| `pyorps.certify.feasible_incumbent` | Checks a priced design against the voltage rules at every leg length and repairs it (`make_feasible`, `FeasibleResult`, `leg_length_interval_km`, `length_grid_km`). |
| `pyorps.certify.voltage` | Relax-and-verify with the voltage limit (`certify_voltage`, `VoltageCertificate`). |

---

(certify-example)=
## Minimal example

Bound-and-prune on synthetic data. Two branches have 200 positions each. The
lower bound is below the exact cost everywhere, as the method requires.

```python
import numpy as np

from pyorps.certify import bound_and_prune

rng = np.random.default_rng(0)
exact = {b: rng.uniform(100.0, 1000.0, 200) for b in ("A", "B")}   # true cost
lower = {b: np.maximum(v - rng.uniform(0.0, 50.0, v.size), 0.0)    # admissible bound
         for b, v in exact.items()}

result = bound_and_prune(lower, lambda b, idx: exact[b][idx],
                         z_ub=1000.0, eps_adm=0.5)
print(result.optimal, result.argmin, result.kept, result.total)
```

The exact evaluator is called only for the kept positions. A tighter bound keeps
fewer positions and gives the same answer. The window and relax-and-verify
functions need fields from the other packages and have no example here.

---

(certify-layering)=
## Layering

`core <- collector <- certify`. `certify` may import `pyorps.collector`; the
collector never imports `certify`. A test
(`tests/test_core/test_package_layering.py`) checks this rule. See also
{doc}`collector`, {doc}`costmodel` and {doc}`siting`.
