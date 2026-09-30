"""Two engines, one exact answer (plan rev. 5, section 3.2; D4c coupling).

Engine A (count tokens, conductor free per section) is a relaxation, so
its value at every root is a lower bound on the exact engine B. B is run
only where A cannot decide:

1. A over the window, pruned against the budget ``Z_UB``;
2. the **gap window**: the roots with ``A(g) + root_cost(g) <= Z_UB + eps``;
3. B with the roots outside the gap window closed (``root_cost = inf``
   there) and pruned against the same budget, so its completion bounds
   only have to reach the gap window.

Every root then carries one of two certified states: **exact** (B's value,
wherever ``B(g) + root_cost(g) <= Z_UB + eps``) or **above budget** (A or B
proves it exceeds ``Z_UB + eps``). The pointwise minimum over sites is
therefore exact wherever it can matter. There is no bracket: a root is
never reported as "between".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pyorps.collector.model import INF, CollectorModel
from pyorps.collector.raster import RasterCollector

__all__ = ["ExactCollectorField", "exact_collector_field"]


@dataclass
class ExactCollectorField:
    """The certified collector field of one window and budget.

    Attributes:
        mv: EUR per root: B's exact value where :attr:`exact`, ``inf``
            where the root is certified above the budget.
        total: ``mv + root_cost``, what the budget is compared with.
        exact: Roots whose value is exact.
        above_budget: Roots certified to exceed ``budget + eps``.
        lower_a: Engine A's (pruned) field, a lower bound where finite
            and exact-for-A inside its own budget.
        gap: The gap window B ran on.
        budget: The budget the certificate refers to (``Z_UB``).
        eps: The slack added to it (safe sign).
        stats: Drains, labels and pruned cells of both engines.
        engine_b: The B run (for :meth:`RasterCollector.design`).
    """
    mv: np.ndarray
    total: np.ndarray
    exact: np.ndarray
    above_budget: np.ndarray
    lower_a: np.ndarray
    gap: np.ndarray
    budget: float
    eps: float
    stats: dict = field(default_factory=dict)
    engine_b: RasterCollector | None = None

    @property
    def argmin(self) -> int:
        """Flat index of the exact root with the cheapest total (collector
        plus root cost); -1 if none is exact."""
        if not self.exact.any():
            return -1
        return int(np.argmin(np.where(self.exact, self.total, INF).ravel()))


def exact_collector_field(values, steps, cell_m: float, turbines,
                          model: CollectorModel, *, budget: float,
                          root_cost=None, eps: float = 0.0,
                          trench_mult: float = 1.0,
                          drain_engine: str = "auto",
                          keep_trace: bool = True) -> ExactCollectorField:
    """Certify every root of a window against ``budget`` (see the module).

    Parameters:
        budget: ``Z_UB``, an incumbent's re-priced total (EUR).
        root_cost: EUR per cell of placing the UW there (site, HV, ...);
            ``None`` for 0. ``inf`` closes a cell as a root.
        eps: Slack added to the budget; must be ``>= 0``.
        keep_trace: Keep B's trace codes so the exact designs can be
            traced and re-priced (plan D7).
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    if eps < 0:
        raise ValueError("eps must be >= 0 (the safe sign)")
    v = np.asarray(values)
    N = v.size
    rc = (np.zeros(N) if root_cost is None
          else np.asarray(root_cost, dtype=np.float64).ravel().copy())
    limit = float(budget) + float(eps)
    run_a = RasterCollector(v, steps, cell_m, turbines, model, engine="A",
                            trench_mult=trench_mult,
                            drain_engine=drain_engine, budget=budget,
                            root_cost=rc, prune_eps=eps)
    mv_a = run_a.run().ravel()
    gap = np.isfinite(mv_a) & (mv_a + rc <= limit)
    rc_b = np.where(gap, rc, INF)
    stats = {"A": dict(run_a.stats), "gap_roots": int(gap.sum())}
    if gap.any():
        run_b = RasterCollector(v, steps, cell_m, turbines, model,
                                engine="B", trench_mult=trench_mult,
                                drain_engine=drain_engine,
                                keep_trace=keep_trace, budget=budget,
                                root_cost=rc_b, prune_eps=eps)
        mv_b = run_b.run().ravel()
        stats["B"] = dict(run_b.stats)
    else:
        run_b = None
        mv_b = np.full(N, INF)
    exact = gap & np.isfinite(mv_b) & (mv_b + rc <= limit)
    mv = np.where(exact, mv_b, INF)
    candidate = ~(v.ravel() == 65535)
    candidate[np.asarray(turbines, dtype=np.int64)] = False
    candidate &= np.isfinite(rc)
    above = candidate & ~exact
    shape = v.shape
    total = np.where(exact, mv_b + rc, INF)
    return ExactCollectorField(
        mv=mv.reshape(shape), total=total.reshape(shape),
        exact=exact.reshape(shape),
        above_budget=above.reshape(shape), lower_a=mv_a.reshape(shape),
        gap=gap.reshape(shape), budget=float(budget), eps=float(eps),
        stats=stats, engine_b=run_b)
