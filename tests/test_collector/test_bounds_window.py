"""The window decomposition bound never exceeds engine B at a root outside the turbine window (math.md M7.6d)."""
import random

import numpy as np
import pytest

from pyorps.collector.bounds_window import cable_rate_floor, exit_cells, rho_hat_exit_value
from pyorps.collector.raster import RasterCollector
from pyorps.graph.drain import drain
from pyorps.utils.neighborhood import get_neighborhood_steps

from .test_engine_vs_oracle_v2 import _model


@pytest.mark.parametrize("seed,count", [(31, 6), pytest.param(32, 20, marks=pytest.mark.slow)])
def test_window_decomposition_is_admissible(seed, count):
    rng = random.Random(seed)
    checked = 0
    for it in range(count):
        R, C, W = 7, 14, 7                                    # raster; window = columns 0..W-1
        nrng = np.random.default_rng(100 * seed + it)
        vals = nrng.integers(1, 6, size=(R, C)).astype(np.uint16)
        if it % 2:
            vals[nrng.random((R, C)) < 0.06] = 65535
        steps = np.asarray(get_neighborhood_steps(rng.choice([1, 2]), directed=True), dtype=np.int8)
        n = rng.choice([2, 3])
        free_w = [r * C + c for r in range(R) for c in range(W) if vals[r, c] != 65535]
        turb = [int(x) for x in nrng.choice(free_w, size=n, replace=False)]
        model = _model(rng, n)
        cell = rng.choice([1.0, 2.5])
        mult = rng.choice([1.0, 1.2])
        win = np.ascontiguousarray(vals[:, :W])
        t_win = [r * W + c for r, c in (divmod(t, C) for t in turb)]
        value = rho_hat_exit_value(win, steps, cell, t_win, model, trench_mult=mult, drain_engine="python")
        # outside part: trench distance from the boundary cells to every cell, euclid distance to the window,
        # cable floor
        ex = exit_cells(win, steps)
        ex_full = [r * C + c for r, c in (divmod(e, W) for e in ex)]
        dtr = drain(vals, steps, ex_full, np.zeros(len(ex_full)), weight_mult=mult, engine="python").ravel() * cell
        alpha = cable_rate_floor(model)
        mv = RasterCollector(vals, steps, cell, turb, model, engine="B", trench_mult=mult,
                             drain_engine="python").run().ravel()
        for g in range(R * C):
            r_, c_ = divmod(g, C)
            if c_ < W or vals[r_, c_] == 65535 or not np.isfinite(mv[g]):
                continue
            dist_w = min(np.hypot(r_ - rr, c_ - cc) for rr in range(R) for cc in range(W)
                         if vals[rr, cc] != 65535) * cell
            lb = value + dtr[g] + alpha * n * dist_w + model.bay_eur
            assert lb <= mv[g] * (1 + 1e-9) + 1e-9, (seed, it, g, lb, mv[g])
            checked += 1
    assert checked > 30


def test_exit_cells_are_the_border_band():
    steps = np.asarray(get_neighborhood_steps(1, directed=True), dtype=np.int8)
    v = np.ones((5, 6), dtype=np.uint16)
    ex = set(exit_cells(v, steps).tolist())
    assert 0 in ex and 29 in ex and 2 * 6 + 2 not in ex          # corners are exits, the centre is not
