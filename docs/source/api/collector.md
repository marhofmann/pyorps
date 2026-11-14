---
title: "Collector (research)"
summary: "Experimental trench-sharing medium-voltage collector, its bounds and its voltage check."
status: unreleased
since: "unreleased"
available_in: source
module: "pyorps.collector"
api:
  - pyorps.collector.CableType
  - pyorps.collector.CollectorModel
  - pyorps.collector.INF
  - pyorps.collector.popcount
  - pyorps.collector.bits
  - pyorps.collector.all_submasks
  - pyorps.collector.set_partitions
  - pyorps.collector.CollectorGraph
  - pyorps.collector.CollectorResult
  - pyorps.collector.solve_collector
  - pyorps.collector.RasterCollector
  - pyorps.collector.graph_from_raster
  - pyorps.collector.RasterStepPricer
  - pyorps.collector.Design
  - pyorps.collector.DesignError
  - pyorps.collector.Junction
  - pyorps.collector.System
  - pyorps.collector.Layout
  - pyorps.collector.layout
  - pyorps.collector.reprice
  - pyorps.collector.rho_hat
  - pyorps.collector.rho_hat_lower_bound
  - pyorps.collector.rho_hat_lower_bound_raster
  - pyorps.collector.mu_star
  - pyorps.collector.completion_bounds_raster
  - pyorps.collector.cable_rate_floor
  - pyorps.collector.exit_cells
  - pyorps.collector.rho_hat_exit_value
  - pyorps.collector.ExactCollectorField
  - pyorps.collector.exact_collector_field
  - pyorps.collector.ConditionC
  - pyorps.collector.condition_c.condition_c
  - pyorps.collector.VoltageModel
  - pyorps.collector.CableElectrical
  - pyorps.collector.TurbineTransformer
  - pyorps.collector.Corner
  - pyorps.collector.standard_corners
  - pyorps.collector.ElectricalTree
  - pyorps.collector.Branch
  - pyorps.collector.electrical_tree
  - pyorps.collector.load_flow
  - pyorps.collector.LoadFlowResult
  - pyorps.collector.LoadFlowError
  - pyorps.collector.check_voltage
  - pyorps.collector.check_tree
  - pyorps.collector.VoltageCheck
  - pyorps.collector.CornerResult
  - pyorps.collector.evaluate_limits
  - pyorps.collector.LimitResult
  - pyorps.collector.decide_taps
  - pyorps.collector.voltage_model_from_yaml
  - pyorps.collector.PlantModel
  - pyorps.collector.UWTransformer
  - pyorps.collector.HVLeg
  - pyorps.collector.GridRules
  - pyorps.collector.NapCorner
  - pyorps.collector.plant_tree
  - pyorps.collector.check_plant
  - pyorps.collector.check_plant_tree
  - pyorps.collector.PlantCheck
  - pyorps.collector.PlantCornerResult
  - pyorps.collector.grid_rules_from_yaml
  - pyorps.collector.plant_model_from_yaml
  - pyorps.collector.repair_design
  - pyorps.collector.RepairResult
  - pyorps.collector.BACKENDS
  - pyorps.collector.DesignBatch
  - pyorps.collector.design_batch
  - pyorps.collector.slot_edges
  - pyorps.collector.BatchLoadFlow
  - pyorps.collector.batch_load_flow
  - pyorps.collector.numba_load_flow
  - pyorps.collector.inflated_nr_load_flow
  - pyorps.collector.batch_voltages
  - pyorps.collector.BatchVoltageCheck
  - pyorps.collector.check_batch
  - pyorps.collector.islands_load_flow
  - pyorps.collector.inflated_load_flow
---
# Collector: trench-sharing cable network and voltage check

`pyorps.collector` prices the medium-voltage cable network (the collector) of
a wind farm when the cables share trenches. A trench is paid once, each cable
adds a small overhead, and cables in one trench derate each other. The package
solves this model exactly on a graph, runs the same recursion on a raster
window, gives lower bounds that let other code prune, re-prices a design
independently and checks the voltage limit.

**Status: experimental research code.** The mathematics is tested on synthetic
instances. The method is not complete: the raster engine is a dense prototype
that does not fit large instances, and the lengths are planimetric (no DEM).
Interfaces may change, and nothing here produces a result that can be quoted.
The package is in the development tree. A PyPI release may hold an older, partial
copy of it, so use a source checkout (see {doc}`../getting_started/release_status`). It is not imported by `pyorps`; import it as
`pyorps.collector`.

---

(collector-public-names)=
## Public names

All names below are available as `pyorps.collector.<name>`. Importing the
package needs no optional package. The voltage functions that read YAML files
need PyYAML, and the power-grid-model backends need power-grid-model, when they
are called.

### `pyorps.collector.model`

| Name | What it does |
|---|---|
| `CableType` | One cable type: ampacity, capitalised cost per metre, resistance. |
| `CollectorModel` | One medium-voltage level of the model (cables, currents, derating, stations). |
| `INF`, `popcount`, `bits`, `all_submasks`, `set_partitions` | Infinity and the bitmask helpers on turbine sets. |

### `pyorps.collector.reference`, `raster`, `raster_pricer`

| Name | What it does |
|---|---|
| `solve_collector` | The exact cut-signature Dreyfus--Wagner engine: the collector cost for every node of a graph. |
| `CollectorGraph` | An undirected graph with trench cost and length per edge. |
| `CollectorResult` | The cost field, and the argmin design at one root. |
| `RasterCollector` | The same recursion as seeded drains on a raster window (dense prototype). |
| `graph_from_raster` | The collector graph of a raster window, for the reference engine. |
| `RasterStepPricer` | Prices one trench step between two cells of a raster window. |

### `pyorps.collector.design`

| Name | What it does |
|---|---|
| `Design`, `System`, `Junction` | An explicit design: trench tree, cables and station busbar sections. |
| `layout`, `Layout` | Validates a design and routes every cable through its trench tree. |
| `reprice` | Prices a design from scratch and raises `DesignError` if it breaks a rule. |
| `DesignError` | A design violates a rule of the model. |

### `pyorps.collector.bounds`, `bounds_window`, `exact`

| Name | What it does |
|---|---|
| `rho_hat`, `rho_hat_lower_bound`, `rho_hat_lower_bound_raster` | The relaxed subset recursion and the admissible lower bound on the collector cost, on a graph and on a raster window. |
| `mu_star` | The cheapest per-metre rate of any label that carries a given number of turbines. |
| `completion_bounds_raster` | Consistent lower bounds on the cost still to pay for a partial collector. |
| `cable_rate_floor`, `exit_cells`, `rho_hat_exit_value` | The window decomposition of the bound for roots far outside the turbine window. |
| `exact_collector_field`, `ExactCollectorField` | Couples the two engines into one certified field for a budget. |

### `pyorps.collector.condition_c`

| Name | What it does |
|---|---|
| `ConditionC` | The result of condition (C) of the model: whether merging two cables can pay. |

The function `condition_c` is not re-exported: its name would hide the module.
Import it with `from pyorps.collector.condition_c import condition_c`.

### `pyorps.collector.voltage`

| Name | What it does |
|---|---|
| `VoltageModel`, `CableElectrical`, `TurbineTransformer` | Electrical data of one collector level, per cable type and turbine transformer. |
| `Corner`, `standard_corners` | The operating corners and the standard set. |
| `ElectricalTree`, `Branch`, `electrical_tree` | The radial network of a traced design. |
| `load_flow`, `LoadFlowResult`, `LoadFlowError` | Exact balanced load flow on a radial tree, its result and its failure. |
| `check_voltage`, `check_tree`, `VoltageCheck`, `CornerResult` | The voltage check of a design or a tree, and what it found. |
| `evaluate_limits`, `LimitResult`, `decide_taps` | Taps, band and equipment rating for several designs at once. |
| `voltage_model_from_yaml` | Builds a `VoltageModel` from a data file (needs PyYAML). |

### `pyorps.collector.voltage_plant`

| Name | What it does |
|---|---|
| `PlantModel`, `UWTransformer`, `HVLeg`, `GridRules`, `NapCorner` | The plant around a collector: substation transformer, 110 kV leg, grid rules, connection-point corners. |
| `plant_tree` | The collector inside the plant at one tap. |
| `check_plant`, `check_plant_tree`, `PlantCheck`, `PlantCornerResult` | The voltage rules at the connection point, and what they found. |
| `grid_rules_from_yaml`, `plant_model_from_yaml` | Build the rules and the plant from a data file (need PyYAML). |

### `pyorps.collector.voltage_rv`, `voltage_batch`, `voltage_pgm`

| Name | What it does |
|---|---|
| `repair_design`, `RepairResult` | Thickens cables of a failing design until it meets the limit. |
| `DesignBatch`, `design_batch`, `slot_edges` | Several designs on one fully meshed slot network. |
| `batch_load_flow`, `numba_load_flow`, `inflated_nr_load_flow`, `BatchLoadFlow` | Batched load flow with the NumPy, numba and Newton--Raphson backends. |
| `batch_voltages`, `BACKENDS` | Voltages of every design at every corner by one named backend. |
| `check_batch`, `BatchVoltageCheck` | The voltage limit for a whole batch. |
| `islands_load_flow`, `inflated_load_flow` | The power-grid-model backends (need power-grid-model when called). |

---

(collector-example)=
## Minimal example

Five nodes on a line, 100 m apart, two turbines at the left end and a
synthetic cable. The exact engine returns the cost for a substation at every
node. The independent re-pricer gives the same total for the traced design.

```python
from pyorps.collector import (
    CableType, CollectorGraph, CollectorModel, reprice, solve_collector,
)

# Edges are (u, w, trench EUR, length m).
edges = [(i, i + 1, 100.0 * 20.0, 100.0) for i in range(4)]
graph = CollectorGraph(5, edges)
cable = CableType("synthetic", ampacity_a=400.0, cost_eur_per_m=30.0,
                  r_ohm_per_m=1e-4)
model = CollectorModel.identical_turbines(2, current_one_a=100.0, types=(cable,))

result = solve_collector(graph, turbines=[0, 1], model=model)
print(result.mv)                  # EUR per node; inf at the turbine nodes
design = result.design(4)         # optimal design with the substation at node 4
print(reprice(design, graph, [0, 1], model)[0])
```

The raster engine, the bounds and the voltage check need a raster window or
catalogue data and have no example here.

---

(collector-layering)=
## Layering

`core <- collector <- {certify, costmodel}`. The collector imports no other
research package. `certify` and `costmodel` import it, never the other way
round. A test (`tests/test_core/test_package_layering.py`) checks this rule.
See also {doc}`certify`, {doc}`costmodel` and {doc}`siting`.
