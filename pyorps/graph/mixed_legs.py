"""Mixed cable / overhead legs by a two-layer fixpoint (plan rev. 5, D6).

A leg (UW to PCC, say) may run as cable, as overhead line, or switch
between them any number of times, each switch paying ``C_tr`` (a
cable-to-overhead transition structure). The model is a two-layer product
(plan section 2.2, MODEL-03, MATH-13):

* layer 0, the cable raster (steps of the routing kernel);
* layer 1, the tower lattice (a :class:`~pyorps.graph.tower_field.
  TowerFieldSolver`, tier 1 or 2);
* transition arcs between a lattice cell and its raster cell, both ways,
  at ``C_tr``.

It is solved by block Gauss--Seidel Bellman--Ford: drain the cable layer
from the source and the overhead arrivals (+ ``C_tr``), solve the tower
layer seeded with the cable values (+ ``C_tr``), and repeat until neither
changes. Every step only lowers values that are costs of real mixed
routes, and a route with ``k`` switches is found by round ``k + 1``, so
the fixpoint is the exact optimum over any number of switches -- D's
definition. In D_LEP the caller restricts layer 1 to the corridor mask by
building the tower solver with it.

Units: EUR throughout. The cable layer is drained in cell units and scaled
by ``cell_m``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pyorps.certify.windows import drain

__all__ = ["MixedLegResult", "mixed_leg_fixpoint"]


@dataclass
class MixedLegResult:
    """The fixpoint of one mixed leg.

    Attributes:
        cable: EUR to reach each raster cell ending in cable.
        overhead: EUR to reach each lattice cell ending overhead (the
            tower field's arrival, a line END there).
        rounds: Gauss--Seidel rounds to the fixpoint.
        tower_field: The last tower field (for its tower sequences).
        log: Per round, the number of cells that improved in each layer.
    """
    cable: np.ndarray
    overhead: np.ndarray
    rounds: int
    tower_field: object = None
    log: list = field(default_factory=list)

    def value_at(self, raster_cell: int, lattice_cell: int | None = None
                 ) -> float:
        """Cheapest way to END at a target: in cable at its raster cell,
        or overhead at its lattice cell (a line may end on a gantry)."""
        v = float(self.cable.ravel()[raster_cell])
        if lattice_cell is not None:
            v = min(v, float(self.overhead.ravel()[lattice_cell]))
        return v


def mixed_leg_fixpoint(cable_values, steps, cell_m: float, source_cell: int,
                       tower_solver, transitions, *, c_tr: float,
                       overhead_source: tuple[int, int] | None = None,
                       cable_rate: float = 0.0, cable_mult: float = 1.0,
                       max_rounds: int = 50,
                       drain_engine: str = "auto") -> MixedLegResult:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Exact mixed legs, any number of switches (see the module docstring).

    Parameters:
        cable_values: The cable cost raster (EUR per metre; 65535 excluded).
        steps: The raster step table.
        cell_m: Raster cell size in metres.
        source_cell: Flat raster index where the leg starts (in cable).
        tower_solver: A prepared :class:`TowerFieldSolver` for layer 1
            (with the corridor mask in D_LEP).
        transitions: ``(k, 2)`` int array of ``(lattice flat index, raster
            flat index)`` pairs where the line may switch layers.
        c_tr: EUR per switch, including any line-end tower the tower model
            does not charge itself.
        overhead_source: The lattice cell where the leg may START overhead
            without a switch (the UW gantry); ``None``: it must switch.
        cable_rate, cable_mult: EUR per metre of cable added to every step
            and the multiplier on the raster value (as in the drains).
        max_rounds: Guard; the fixpoint needs (switches + 1) rounds.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    if c_tr < 0:
        raise ValueError("c_tr must be >= 0")
    vals = np.asarray(cable_values)
    cell = float(cell_m)
    trans = np.asarray(transitions, dtype=np.int64).reshape(-1, 2)
    lat_idx, ras_idx = trans[:, 0], trans[:, 1]
    lat_shape = tower_solver.shape
    n_lat = int(np.prod(lat_shape))
    if (lat_idx < 0).any() or (lat_idx >= n_lat).any():
        raise ValueError("a transition lies outside the tower lattice")
    if (ras_idx < 0).any() or (ras_idx >= vals.size).any():
        raise ValueError("a transition lies outside the cable raster")
    blocked = np.asarray(tower_solver.blocked).ravel()
    keep = ~blocked[lat_idx] & (vals.ravel()[ras_idx] != 65535)
    lat_idx, ras_idx = lat_idx[keep], ras_idx[keep]

    def cable_layer(overhead):
        seeds = [int(source_cell)]
        labels = [0.0]
        if overhead is not None and lat_idx.size:
            ov = overhead.ravel()[lat_idx] + c_tr
            ok = np.isfinite(ov)
            seeds.extend(ras_idx[ok].tolist())
            labels.extend((ov[ok] / cell).tolist())
        d = drain(vals, steps, np.asarray(seeds), np.asarray(labels),
                  length_rate=cable_rate, weight_mult=cable_mult,
                  engine=drain_engine)
        return np.asarray(d, dtype=np.float64).ravel() * cell

    def overhead_layer(cable):
        seed = np.full(n_lat, np.inf)
        if lat_idx.size:
            np.minimum.at(seed, lat_idx, cable[ras_idx] + c_tr)
        seed = seed.reshape(lat_shape)
        if overhead_source is None and not np.isfinite(seed).any():
            return None, np.full(lat_shape, np.inf)
        f = tower_solver.solve(overhead_source, seed_chain=seed,
                               record_pred=True)
        return f, np.asarray(f.arrival, dtype=np.float64)

    cable = cable_layer(None)
    fld, overhead = overhead_layer(cable)
    log = []
    for rounds in range(1, max_rounds + 1):
        new_cable = cable_layer(overhead)
        fld2, new_over = overhead_layer(new_cable)
        imp_c = int(np.sum(new_cable < cable))
        imp_o = int(np.sum(new_over < overhead))
        log.append({"round": rounds, "cable_improved": imp_c,
                    "overhead_improved": imp_o})
        if np.any(new_cable > cable * (1 + 1e-12) + 1e-9) or np.any(
                new_over > overhead * (1 + 1e-12) + 1e-9):
            raise RuntimeError("a Gauss-Seidel round raised a value: the "
                               "layers are not monotone")
        cable, overhead = new_cable, new_over
        if fld2 is not None:
            fld = fld2
        if imp_c == 0 and imp_o == 0:
            return MixedLegResult(cable=cable.reshape(vals.shape),
                                  overhead=overhead, rounds=rounds,
                                  tower_field=fld, log=log)
    raise RuntimeError(f"mixed legs did not reach a fixpoint in "
                       f"{max_rounds} rounds")
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
