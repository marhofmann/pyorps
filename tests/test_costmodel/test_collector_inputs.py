"""The collector model built from sourced inputs (plan rev. 5, Phase C closure).

Plan acceptance: costs equal the catalogue items to the cent, ``W(X)`` equals
the loss table's square sums, and condition (C) is printed with its margin.
"""
import copy
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("yaml")

from pyorps.costmodel import (
    CostCatalogue,
    LossCategories,
    annuity_factor,
    category_square_sums,
    loss_coef,
)
from pyorps.costmodel.collector_inputs import (
    CollectorInputs,
    build_collector_model,
)

CASE = Path(__file__).resolve().parents[2] / "case_studies" / "runkel_free_siting"
INPUTS = CASE / "config" / "collector_inputs_2026.yaml"
CATALOGUE = CASE / "config" / "cost_catalogue_2026.yaml"
TABLE = CASE / "data" / "loss_categories.npz"
RATED = [6.8] * 7

needs_table = pytest.mark.skipif(not TABLE.exists(),
                                 reason="loss category table not present")


@pytest.fixture(scope="module")
def inputs():
    if not INPUTS.exists():
        pytest.skip("case-study files not present")
    return CollectorInputs.load(INPUTS)


@pytest.fixture(scope="module")
def catalogue():
    if not CATALOGUE.exists():
        pytest.skip("case-study files not present")
    return CostCatalogue.load(CATALOGUE)


@pytest.fixture(scope="module")
def loss():
    if not TABLE.exists():
        pytest.skip("loss category table not present")
    return LossCategories.load(TABLE)


def test_every_tagged_value_has_a_valid_status(inputs):
    st = inputs.statuses()
    assert st and set(st.values()) <= {"sourced", "decided", "catalogue",
                                       "derived", "assumption"}
    ass = inputs.assumptions()
    assert "trench_sharing.derating" in ass
    assert "switchgear.bay_eur" in ass and "cable.price_eur_per_m" in ass


def test_an_unknown_status_is_refused(tmp_path):
    f = tmp_path / "i.yaml"
    f.write_text("loss:\n  years: {value: 25, status: guess}\n",
                 encoding="utf-8")
    with pytest.raises(ValueError, match="invalid status"):
        CollectorInputs.load(f)


@needs_table
def test_costs_equal_the_catalogue_items_to_the_cent(inputs, catalogue, loss):
    b = build_collector_model(inputs, catalogue, loss, u_kv=20, rated_mw=RATED)
    m = b.model
    assert m.station_building_eur == catalogue.value(
        "station_buildings_security.derived_building_cost_eur."
        "compact_mv_switching_station_50_150m2")
    base = catalogue.value("mv_rmu_panel_multiplier.base_2way_eur_estimate")
    assert m.turbine_panel_eur == pytest.approx(base / 2, abs=0.005)
    assert m.station_panel_eur == pytest.approx(base / 2, abs=0.005)
    assert m.bay_eur == catalogue.value(
        "mv_switchgear_by_voltage_class.usd_33kv_gis_bay")
    civil = catalogue.value(
        "underground_cable_civil.civil_base_eur_per_m.mv_10_20_33_66kv")
    assert m.sigma_eur_per_m == pytest.approx(civil * 0.10 / 0.5, abs=0.005)
    om = catalogue.value("personnel_and_om.om_annual_pct_of_capex")
    af = annuity_factor(25, 0.04, 0.0)
    t240 = {t.name: t for t in m.types}["NA2XS(F)2Y 1x240 Al 20 kV"]
    assert t240.cost_eur_per_m == pytest.approx(60.0 * (1 + om * af),
                                                abs=0.005)


@needs_table
def test_cable_data_follow_the_voltage_file(inputs, catalogue, loss):
    m20 = build_collector_model(inputs, catalogue, loss, u_kv=20,
                                rated_mw=RATED).model
    t = {x.name: x for x in m20.types}["NA2XS(F)2Y 1x240 Al 20 kV"]
    assert t.ampacity_a == 417.0                      # min of the two sources
    f = (1 + 0.00403 * 23.3) / (1 + 0.00403 * 70.0)
    assert t.r_ohm_per_m == pytest.approx(0.160 * f / 1e3, rel=1e-12)
    m33 = build_collector_model(inputs, catalogue, loss, u_kv=33,
                                rated_mw=RATED).model
    t33 = {x.name: x for x in m33.types}["NA2XS(F)2Y 1x240 Al 33 kV"]
    assert t33.ampacity_a == 385.0                    # min of the three
    assert m33.bay_a == 630.0 and m20.bay_a == 1250.0
    assert m20.switchgear_a == 630.0 and m20.m_max == 4


@needs_table
def test_the_30kv_price_is_the_interpolation_between_20_and_33(inputs,
                                                               catalogue, loss):
    om_af = 1 + catalogue.value(
        "personnel_and_om.om_annual_pct_of_capex") * annuity_factor(
            25, 0.04, 0.0)
    m30 = build_collector_model(inputs, catalogue, loss, u_kv=30,
                                rated_mw=RATED).model
    t = {x.name: x for x in m30.types}["NA2XS(F)2Y 1x240 Al 30 kV"]
    assert t.cost_eur_per_m == pytest.approx(
        (60.0 + 10.0 * 10.0 / 13.0) * om_af, abs=0.005)
    assert t.ampacity_a == 422.0                      # the trefoil column


@needs_table
def test_loss_weight_equals_the_tables_square_sums(inputs, catalogue, loss):
    b = build_collector_model(inputs, catalogue, loss, u_kv=20, rated_mw=RATED)
    # the model carries the exact (hourly) form ...
    assert b.model.loss_weight == loss.loss_weight("exact")
    # ... which is the quadratic form of M_exact over every subset
    for mask in (1, 5, 0b1111111):
        idx = [i for i in range(7) if mask >> i & 1]
        w = float(loss.M_exact[np.ix_(idx, idx)].sum())
        assert b.model.loss_weight[mask] == pytest.approx(w, rel=1e-12)
    # and the category table's own square sums reproduce the mean-only form
    cat = category_square_sums(loss.hours, loss.P, loss.Q)
    assert np.allclose(cat, loss.loss_weight("cat"), rtol=1e-12)
    assert b.model.loss_weight[-1] >= cat[-1] * (1 - 1e-12)  # cat is a lower bound


@needs_table
def test_loss_price_headline_and_sensitivity(inputs, catalogue, loss):
    h = build_collector_model(inputs, catalogue, loss, u_kv=20, rated_mw=RATED)
    s = build_collector_model(inputs, catalogue, loss, u_kv=20, rated_mw=RATED,
                              price="sensitivity")
    assert h.record["loss_price_eur_per_mwh"] == 67.37
    assert s.record["loss_price_eur_per_mwh"] == 61.86
    assert h.model.loss_coef == pytest.approx(
        loss_coef(price_eur_per_mwh=67.37, u_kv=20.0))
    assert s.model.loss_coef < h.model.loss_coef
    with pytest.raises(ValueError, match="headline"):
        build_collector_model(inputs, catalogue, loss, u_kv=20,
                              rated_mw=RATED, price="plan")


@needs_table
def test_a_certified_build_refuses_while_assumptions_remain(inputs, catalogue,
                                                            loss):
    with pytest.raises(ValueError, match="assumption"):
        build_collector_model(inputs, catalogue, loss, u_kv=20,
                              rated_mw=RATED, certified=True)
    cleared = copy.deepcopy(inputs)

    def clear(node):
        if isinstance(node, dict):
            if node.get("status") == "assumption":
                node["status"] = "sourced"
            for v in node.values():
                clear(v)

    clear(cleared.data)
    b = build_collector_model(cleared, catalogue, loss, u_kv=20,
                              rated_mw=RATED, certified=True)
    assert b.assumptions == () and b.record["certified"] is True


@needs_table
def test_wrong_turbine_count_or_level_is_refused(inputs, catalogue, loss):
    with pytest.raises(ValueError, match="turbines"):
        build_collector_model(inputs, catalogue, loss, u_kv=20,
                              rated_mw=[6.8] * 5)
    with pytest.raises(ValueError, match="cable"):
        build_collector_model(inputs, catalogue, loss, u_kv=66,
                              rated_mw=RATED)


@needs_table
def test_report_prints_the_assumptions_and_condition_c(inputs, catalogue, loss):
    b = build_collector_model(inputs, catalogue, loss, u_kv=20, rated_mw=RATED)
    assert "not evaluated" in b.report()
    b.check_condition_c(40.0)                 # the catalogue's MV civil base
    text = b.report()
    assert "condition (C)" in text and "assumptions" in text
    assert b.record["condition_c"]["min_trench_rate"] == 40.0
    assert b.record["model_sha256"] and len(b.record["model_sha256"]) == 64
