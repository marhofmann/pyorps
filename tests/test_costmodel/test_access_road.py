"""Access-road constraints and the block cost (plan 2026-10-06-paper-run-pipeline.md, step 1)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyorps.costmodel.access_road import (
    FORBIDDEN,
    AccessConstraints,
    block_cost,
    class_cost,
    clearance,
    slope_multiplier,
    slope_percent,
)
from pyorps.costmodel.catalogue import CostCatalogue

CONFIG = Path(__file__).resolve().parents[2] / "case_studies" / "runkel_free_siting" / "config"
RECIPE = {
    "Wald": {"Nadelholz": 405, "": 405},
    "Straßenverkehr": {"Landstr.": 450, "Autobahn": 750, "": 340},
    "Weg": {"Fußweg": 220, "": 220},
    "Landwirtschaft": {"Ackerland": 437, "": 437},
    "Fließgewässer": {"Bach": 346, "": 332},
    "Wohnbaufläche": {"": FORBIDDEN},
}


@pytest.fixture(scope="module")
def cat():
    if not (CONFIG / "access_constraints_2026.yaml").exists():
        pytest.skip("case-study files not present")
    return CostCatalogue.load(CONFIG / "cost_catalogue_2026.yaml")


@pytest.fixture(scope="module")
def acc(cat):
    return AccessConstraints.load(CONFIG / "access_constraints_2026.yaml", cat)


def test_prices_equal_the_catalogue(acc, cat):
    assert acc.new_eur_m == cat.value("access_roads.base_rate_eur_per_running_m.heavy_load_reinforced") == 225
    assert acc.simple_eur_m == 70
    assert acc.forest_mult == 1.5
    assert acc.upgrade_eur_m == 155


def test_assumptions_listed_and_certified_load_refuses(cat):
    acc = AccessConstraints.load(CONFIG / "access_constraints_2026.yaml", cat)
    assert "terrain.g_max_percent" in acc.assumptions() and "classes" in acc.assumptions()
    assert "road.new_eur_per_m" not in acc.assumptions()
    with pytest.raises(ValueError, match="assumptions remain"):
        AccessConstraints.load(CONFIG / "access_constraints_2026.yaml", cat, certified=True)


def test_class_table(acc):
    t = acc.class_table(RECIPE)
    assert t["Wald"][""] == 338                      # ceil(225 x 1.5)
    assert t["Landwirtschaft"]["Ackerland"] == 225
    assert t["Weg"]["Fußweg"] == 155
    assert t["Straßenverkehr"]["Landstr."] == 1 and t["Straßenverkehr"][""] == 1
    assert t["Straßenverkehr"]["Autobahn"] == FORBIDDEN
    assert t["Fließgewässer"]["Bach"] == FORBIDDEN
    assert t["Wohnbaufläche"][""] == FORBIDDEN       # listed nowhere -> forbidden
    assert class_cost(t, "Unbekannt", "x") == FORBIDDEN
    assert class_cost(t, "Weg", "Radweg") == 155     # unlisted sub-class takes the class default


def test_public_cost_must_stay_below_all_others(cat, tmp_path):
    text = (CONFIG / "access_constraints_2026.yaml").read_text(encoding="utf-8").replace(
        "{value: 1, status: assumption", "{value: 400, status: assumption")
    p = tmp_path / "bad.yaml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="below every other"):
        AccessConstraints.load(p, cat).class_table(RECIPE)


def test_bad_status_and_bad_bands_refused(cat, tmp_path):
    src = (CONFIG / "access_constraints_2026.yaml").read_text(encoding="utf-8")
    p = tmp_path / "s.yaml"
    p.write_text(src.replace("status: decided", "status: guessed", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid status"):
        AccessConstraints.load(p, cat)
    p.write_text(src.replace("[3.0, 6.0, 1.15]", "[3.5, 6.0, 1.15]"), encoding="utf-8")
    with pytest.raises(ValueError, match="contiguous"):
        AccessConstraints.load(p, cat)


def test_slope_of_a_plane():
    y = np.arange(40, dtype=np.float32)[:, None] * 0.08 * np.ones((1, 30), np.float32)   # 8 % grade
    s = slope_percent(y, half=2)
    assert np.allclose(s[5:-5], 8.0, atol=1e-3)
    flat = slope_percent(np.full((9, 9), 100.0), half=2)
    assert np.all(flat == 0)


def test_slope_multiplier_bands(acc):
    m = slope_multiplier(np.array([0.0, 2.9, 3.0, 5.9, 6.0, 9.9]), acc.bands)
    assert m.tolist() == [1.0, 1.0, 1.15, 1.15, 1.4, 1.4]


def test_clearance_needs_a_free_disc():
    free = np.zeros((21, 41), bool)
    free[8:13, :] = True                              # a 5-cell-wide corridor
    assert clearance(free, 2)[10, 20]                 # radius 2 fits (disc diameter 5)
    assert not clearance(free, 3)[10, 20]             # radius 3 does not
    assert clearance(free, 0).sum() == free.sum()
    assert clearance(np.ones((6, 6), bool), 3).all()  # the array edge does not close cells


def _scene(h=60, w=50):
    cls = np.full((h, w), 225, np.uint16)
    z = np.zeros((h, w), np.float32)
    return cls, z


def test_block_cost_rules(acc):
    cls, z = _scene()
    cls[:, 0:3] = FORBIDDEN                            # a wall on the left
    cls[10, 20:30] = 1                                 # an existing public road
    z += (np.arange(50, dtype=np.float32) * 0.40)[None, :]   # 40 % ground slope, above the 30 % cap
    cost, entry = block_cost(cls, z, acc)
    assert entry[10, 20:30].all() and entry.sum() == 10
    assert (cost[10, 20:30] == 1).all()                # entry cells are never closed by slope
    assert (cost[30, 10:40] == FORBIDDEN).all()
    cls2, z2 = _scene()
    cls2[:, 0:3] = FORBIDDEN
    z2 += (np.arange(50, dtype=np.float32) * 0.12)[None, :]   # 12 % ground slope: allowed here, the grade
    cost2, _ = block_cost(cls2, z2, acc)                       # along the route is the kernel's rule
    assert cost2[30, 20] == 225
    assert (cost2[30, 0:3] == FORBIDDEN).all()
    assert (cost2[30, 3:3 + acc.radius_cells()] == FORBIDDEN).all()   # no clear disc next to the wall
    assert cost2[30, 3 + acc.radius_cells()] != FORBIDDEN


def test_gradient_options_give_the_kernel_table(acc):
    from pyorps.core.objective import Objective

    steps = np.array([[0, 1], [1, 1]])
    luts = Objective({"cost": 1.0}, acc.gradient_options()).build_gradient_luts(steps, 1.0)
    s = (np.arange(luts.n_bins) + 0.5) * luts.bin_width_pct
    stretch = np.sqrt(1 + (s / 100) ** 2)
    assert np.isinf(luts.mult[s > 10.0]).all() and np.isfinite(luts.mult[s < 10.0]).all()
    i = int(np.argmin(np.abs(s - 4.0)))
    assert luts.mult[i] == pytest.approx(stretch[i] * 1.15, rel=1e-6)
    j = int(np.argmin(np.abs(s - 1.0)))
    assert luts.mult[j] == pytest.approx(stretch[j] * 1.00, rel=1e-6)


def test_cost_never_reaches_the_forbidden_value(acc):
    cls = np.full((30, 30), 65534, np.uint16)
    cost, _ = block_cost(cls, np.zeros((30, 30), np.float32), acc)
    assert cost.max() == 65534


def test_blocks_with_a_halo_equal_the_whole(acc):
    rng = np.random.default_rng(3)
    cls = rng.choice(np.array([1, 155, 225, 338, FORBIDDEN], np.uint16), size=(120, 40), p=[.05, .1, .5, .25, .1])
    z = np.cumsum(rng.normal(0, 0.05, (120, 40)).astype(np.float32), axis=0)
    whole, whole_entry = block_cost(cls, z, acc)
    halo = acc.half_window + acc.radius_cells()
    parts, entries = [], []
    for r0, r1 in ((0, 45), (45, 90), (90, 120)):
        a, b = max(0, r0 - halo), min(120, r1 + halo)
        c_, e_ = block_cost(cls[a:b], z[a:b], acc, top=r0 - a, bottom=b - r1)
        parts.append(c_)
        entries.append(e_)
    assert np.array_equal(np.vstack(parts), whole)
    assert np.array_equal(np.vstack(entries), whole_entry)
