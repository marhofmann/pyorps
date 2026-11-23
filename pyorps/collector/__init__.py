"""The trench-sharing MV collector, its bounds and its voltage check.

Experimental research code. The mathematics is tested on synthetic instances;
the method is not complete (the raster engine is a dense prototype that does
not fit large instances) and interfaces may change. Nothing here produces a
result that can be quoted.

``D_share`` is the collector model of the siting study: a trench tree in the
routing raster, a radial electrical network of turbines, optional switching
stations and the substation, cables that share trenches (paid once, plus a
per-cable overhead, with group derating), one cable option per connection,
parallel cables allowed.

Modules, all exported here (each name is available as
``pyorps.collector.<name>``):

* :mod:`.model`: the model (``CableType``, ``CollectorModel``) and bitmask helpers.
* :mod:`.reference`: the exact cut-signature Dreyfus--Wagner engine on a graph
  (``solve_collector``).
* :mod:`.raster`: the same recursion as seeded drains on a raster window
  (``RasterCollector``).
* :mod:`.raster_pricer`: prices one trench step on a raster window.
* :mod:`.design`: an explicit design and its independent re-pricer.
* :mod:`.bounds`: the lower bound and the completion bounds used to prune.
* :mod:`.bounds_window`: the window decomposition of that bound for roots far
  outside the turbine window.
* :mod:`.exact`: couples the two engines into one certified field.
* :mod:`.condition_c`: condition (C) of the model. Only ``ConditionC`` is
  re-exported; the function ``condition_c`` would hide the module, so import it
  from :mod:`pyorps.collector.condition_c`.
* :mod:`.voltage`: the voltage limit on a collector tree (load flow, taps).
* :mod:`.voltage_plant`: the same limit for the whole plant, up to the grid
  connection point.
* :mod:`.voltage_rv`: repairs a design that fails the voltage limit.
* :mod:`.voltage_batch`: batched voltage checks on a fully meshed slot network.
* :mod:`.voltage_pgm`: power-grid-model backends for the batched check. They
  need the optional package power-grid-model when called, not when imported.

Layering: ``core <- collector``. This package imports no other research
package and is not imported by ``pyorps/__init__``.
"""

from pyorps.collector.bounds import (
    completion_bounds_raster,
    mu_star,
    rho_hat,
    rho_hat_lower_bound,
    rho_hat_lower_bound_raster,
)
from pyorps.collector.bounds_window import (
    cable_rate_floor,
    exit_cells,
    rho_hat_exit_value,
)
from pyorps.collector.condition_c import ConditionC
from pyorps.collector.design import (
    Design,
    DesignError,
    Junction,
    Layout,
    System,
    layout,
    reprice,
)
from pyorps.collector.exact import ExactCollectorField, exact_collector_field
from pyorps.collector.model import (
    INF,
    CableType,
    CollectorModel,
    all_submasks,
    bits,
    popcount,
    set_partitions,
)
from pyorps.collector.raster import RasterCollector, graph_from_raster
from pyorps.collector.raster_pricer import RasterStepPricer
from pyorps.collector.reference import CollectorGraph, CollectorResult, solve_collector
from pyorps.collector.voltage import (
    Branch,
    CableElectrical,
    Corner,
    CornerResult,
    ElectricalTree,
    LimitResult,
    LoadFlowError,
    LoadFlowResult,
    TurbineTransformer,
    VoltageCheck,
    VoltageModel,
    check_tree,
    check_voltage,
    decide_taps,
    electrical_tree,
    evaluate_limits,
    load_flow,
    standard_corners,
    voltage_model_from_yaml,
)
from pyorps.collector.voltage_batch import (
    BACKENDS,
    BatchLoadFlow,
    BatchVoltageCheck,
    DesignBatch,
    batch_load_flow,
    batch_voltages,
    check_batch,
    design_batch,
    inflated_nr_load_flow,
    numba_load_flow,
    slot_edges,
)
from pyorps.collector.voltage_pgm import inflated_load_flow, islands_load_flow
from pyorps.collector.voltage_plant import (
    GridRules,
    HVLeg,
    NapCorner,
    PlantCheck,
    PlantCornerResult,
    PlantModel,
    UWTransformer,
    check_plant,
    check_plant_tree,
    grid_rules_from_yaml,
    plant_model_from_yaml,
    plant_tree,
)
from pyorps.collector.voltage_rv import RepairResult, repair_design

__all__ = [
    "BACKENDS",
    "BatchLoadFlow",
    "BatchVoltageCheck",
    "Branch",
    "CableElectrical",
    "CableType",
    "CollectorGraph",
    "CollectorModel",
    "CollectorResult",
    "ConditionC",
    "Corner",
    "CornerResult",
    "Design",
    "DesignBatch",
    "DesignError",
    "ElectricalTree",
    "ExactCollectorField",
    "GridRules",
    "HVLeg",
    "INF",
    "Junction",
    "Layout",
    "LimitResult",
    "LoadFlowError",
    "LoadFlowResult",
    "NapCorner",
    "PlantCheck",
    "PlantCornerResult",
    "PlantModel",
    "RasterCollector",
    "RasterStepPricer",
    "RepairResult",
    "System",
    "TurbineTransformer",
    "UWTransformer",
    "VoltageCheck",
    "VoltageModel",
    "all_submasks",
    "batch_load_flow",
    "batch_voltages",
    "bits",
    "cable_rate_floor",
    "check_batch",
    "check_plant",
    "check_plant_tree",
    "check_tree",
    "check_voltage",
    "completion_bounds_raster",
    "decide_taps",
    "design_batch",
    "electrical_tree",
    "evaluate_limits",
    "exact_collector_field",
    "exit_cells",
    "graph_from_raster",
    "grid_rules_from_yaml",
    "inflated_load_flow",
    "inflated_nr_load_flow",
    "islands_load_flow",
    "layout",
    "load_flow",
    "mu_star",
    "numba_load_flow",
    "plant_model_from_yaml",
    "plant_tree",
    "popcount",
    "repair_design",
    "reprice",
    "rho_hat",
    "rho_hat_exit_value",
    "rho_hat_lower_bound",
    "rho_hat_lower_bound_raster",
    "set_partitions",
    "slot_edges",
    "solve_collector",
    "standard_corners",
    "voltage_model_from_yaml",
]
