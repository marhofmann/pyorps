"""Independent re-pricing of raster collector designs (plan rev. 5, D7).

A design traced out of :class:`~pyorps.collector.raster.RasterCollector`
is a tree of kernel steps between raster cells. The plan never trusts a DP
value it has not re-priced along a path that shares no code with the DP:

* the trench weight of each step comes from the Cython kernel's own
  ``MultiSourceSolver.price_route`` on a context built from the UNMUTATED
  raster (no no-transit masking, nothing written into the exclusion mask);
* the length comes from the step's offsets, computed here;
* a step whose intermediate cells include a turbine is refused, and
  :func:`~pyorps.collector.design.reprice` refuses a trench node on a
  turbine cell, so no walk passes through a turbine;
* ``(1 + omega)`` multiplies the trench and the cable rate ``R`` is applied
  to the independent length inside :func:`~pyorps.collector.design.reprice`.

Use it as the ``graph`` argument of :func:`~pyorps.collector.design.reprice`.
"""

from __future__ import annotations

import math

import numpy as np

from pyorps.collector.design import DesignError

__all__ = ["RasterStepPricer"]


class RasterStepPricer:
    """Prices one trench step between two cells of a raster window.

    Parameters:
        values: The uint16 trench-cost raster the design was traced on
            (EUR per metre of trench; 65535 excluded).
        steps: The neighbourhood step table.
        cell_m: Cell size in metres.
        turbines: Flat cell indices of the turbines.
        trench_mult: ``1 + omega`` on the trench.
    """

    def __init__(self, values, steps, cell_m: float, turbines, *,
                 trench_mult: float = 1.0):
        from pyorps.graph.tower_field import intermediate_offsets
        from pyorps.utils._dijkstra import make_multi_source_solver

        raster = np.ascontiguousarray(np.asarray(values, dtype=np.uint16))
        self._shape = raster.shape
        self._solver = make_multi_source_solver(
            raster.copy(), np.ascontiguousarray(np.asarray(steps,
                                                           dtype=np.int8)))
        self._offsets = {}
        for dr, dc in np.asarray(steps)[:, :2]:
            dr, dc = int(dr), int(dc)
            self._offsets[(dr, dc)] = intermediate_offsets(dr, dc)
        self._turbines = {int(t) for t in turbines}
        self._cell = float(cell_m)
        self._mult = float(trench_mult)

    def price_step(self, _key, a: int, b: int) -> tuple[float, float]:
        """``(trench EUR, length m)`` of the step ``a -> b``.

        Raises:
            DesignError: not a step of the table, an intermediate cell on a
                turbine, or a step the kernel refuses (excluded cells).
        """
        rows, cols = self._shape
        ra, ca = divmod(int(a), cols)
        rb, cb = divmod(int(b), cols)
        if not (0 <= ra < rows and 0 <= rb < rows):
            raise DesignError(f"step {a} -> {b} leaves the window")
        dr, dc = rb - ra, cb - ca
        inter = self._offsets.get((dr, dc))
        if inter is None:
            raise DesignError(f"{a} -> {b} is not a step of the table "
                              f"({dr}, {dc})")
        for r, c in inter:
            cell = (ra + r) * cols + (ca + c)
            if cell in self._turbines:
                raise DesignError(f"step {a} -> {b} crosses turbine cell "
                                  f"{cell}")
        try:
            w = float(self._solver.price_route(np.array([a, b],
                                                        dtype=np.int64)))
        except (ValueError, IndexError) as exc:
            raise DesignError(f"the kernel refuses step {a} -> {b}: "
                              f"{exc}") from exc
        return (self._mult * self._cell * w,
                self._cell * math.hypot(dr, dc))
