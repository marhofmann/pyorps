"""The 110 kV cable leg on the shared trench raster (plan 2026-09-23 section 3.3; pipeline plan stage S2)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("yaml")

from pyorps.costmodel.catalogue import CostCatalogue
from pyorps.costmodel.collector_inputs import CollectorInputs, build_collector_model
from pyorps.costmodel.hv_leg import FORBIDDEN, HvLegConstraints
from pyorps.costmodel.losses import LossCategories, annuity_factor

CASE = Path(__file__).resolve().parents[2] / "case_studies" / "runkel_free_siting"


@pytest.fixture(scope="module")
def parts():
    table = CASE / "data" / "loss_categories.npz"
    if not table.exists():
        # Same rule as tests/test_costmodel/test_losses.py: the table is case-study data, not always in a checkout.
        pytest.skip("case-study table not present")
    return (CostCatalogue.load(CASE / "config" / "cost_catalogue_2026.yaml"),
            CollectorInputs.load(CASE / "config" / "collector_inputs_2026.yaml"),
            LossCategories.load(CASE / "data" / "loss_categories.npz"))


@pytest.fixture(scope="module")
def hv(parts):
    cat, inp, loss = parts
    return HvLegConstraints.load(CASE / "config" / "hv_leg_constraints_2026.yaml", cat, inp, loss)


def test_omega_and_cores_come_from_the_catalogue_like_the_collector(hv, parts):
    cat, inp, _ = parts
    om = cat.value("personnel_and_om.om_annual_pct_of_capex")
    af = annuity_factor(25.0, 0.04, 0.0)
    assert hv.omega == pytest.approx(om * af)
    assert hv.cores == 3
    assert hv.trench_raster == "cost_priced_1m"


def test_section_is_the_cheapest_total_and_losses_use_the_whole_farm_weight(hv, parts):
    cat, inp, loss = parts
    totals = {m: s["total"] for m, s in hv.sections.items()}
    assert hv.section_mm2 == min(totals, key=totals.get) and hv.rho_eur_m == totals[hv.section_mm2]
    assert hv.sections[240]["material"] == pytest.approx(300.0 * (1 + hv.omega))
    assert hv.sections[630]["loss"] < hv.sections[240]["loss"] < 1e3
    # the loss of 240 mm2 equals the collector's own loss formula with the full-farm weight W(all)
    built = build_collector_model(inp, cat, loss, u_kv=33.0, rated_mw=[6.8] * 7)
    w_all = float(loss.loss_weight("exact")[(1 << 7) - 1])
    assert w_all == pytest.approx(4291697, rel=1e-3)
    assert built.model.loss_coef > 0


def test_cost_block_is_the_plan_weight_and_never_below_the_trench(hv):
    t = np.array([[437, 405, 750, FORBIDDEN, 220]], dtype=np.uint16)
    out = hv.cost_block(t)
    assert out[0, 3] == FORBIDDEN
    free = t != FORBIDDEN
    expect = np.ceil((1 + hv.omega) * t[free] + hv.rho_eur_m - 1e-9).astype(np.uint16)
    assert np.array_equal(out[free], expect)
    assert (out[free] > t[free]).all()                    # never cheaper than the shared trench rate
    assert (np.diff(np.sort(out[free])) >= 0).all() and out[0, 2] > out[0, 0] > out[0, 1]


def test_only_one_circuit_and_certified_refusal(parts, tmp_path):
    cat, inp, loss = parts
    src = (CASE / "config" / "hv_leg_constraints_2026.yaml").read_text(encoding="utf-8")
    p = tmp_path / "two.yaml"
    p.write_text(src.replace("{value: 1, status: decided, source: \"owner 2026-10-06: a single",
                             "{value: 2, status: decided, source: \"owner 2026-10-06: a single"),
                 encoding="utf-8")
    with pytest.raises(ValueError, match="one circuit"):
        HvLegConstraints.load(p, cat, inp, loss, voltage_file=CASE / "config" / "voltage_2026.yaml")
    with pytest.raises(ValueError, match="assumptions remain"):
        HvLegConstraints.load(CASE / "config" / "hv_leg_constraints_2026.yaml", cat, inp, loss, certified=True)


def test_a_capped_trench_cell_saturates_and_is_counted(hv):
    before = hv.saturated_cells
    out = hv.cost_block(np.array([[65534, 100]], dtype=np.uint16))
    assert out[0, 0] == 65534 and out[0, 1] < 65534 and hv.saturated_cells == before + 1
