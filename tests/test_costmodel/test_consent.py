"""Priced consent classes: protected areas are priced, not forbidden (owner decision, 2026-10-05)."""
from pathlib import Path

import numpy as np
import pytest

from pyorps.certify.rv import consent_lattice
from pyorps.costmodel import CostCatalogue
from pyorps.costmodel.consent import FORBIDDEN, PRICE_CAP, ConsentTable

CASE = Path(__file__).resolve().parents[2] / "case_studies" / "runkel_free_siting"
CONSENT = CASE / "config" / "consent_classes_2026.yaml"
CATALOGUE = CASE / "config" / "cost_catalogue_2026.yaml"
MANIFEST = CASE / "data" / "data_manifest.yaml"


@pytest.fixture(scope="module")
def table():
    return ConsentTable.load(CONSENT, CostCatalogue.load(CATALOGUE))


def test_the_table_loads_with_the_catalogue_hdd_price_and_names_its_stand_ins(table):
    hdd = CostCatalogue.load(CATALOGUE).value(
        "underground_cable_civil.hdd_mandatory_crossing.multiplier_on_open_cut_total")
    assert hdd == 1.5
    assert set(table.classes) == {"nature_reserves", "landscape_protection", "natura2000_habitats",
                                  "bird_protection_areas", "biotopes_protected",
                                  "biotope_complexes_protected"}
    assert table.classes["natura2000_habitats"].multiplier == hdd
    assert table.classes["natura2000_habitats"].one_off_eur == 0.5 * (15000 + 60000)
    assert table.classes["nature_reserves"].one_off_range == (5000.0, 20000.0)
    assert table.water_multipliers == {1: 100.0, 2: 2.0, 3: 1.5, 4: 1.5, 5: 1.2}
    assert {c.confidence for c in table.classes.values()} == {"LOW"}
    ass = table.assumptions()
    assert "water_zones.per_metre_multiplier" in ass and len(ass) >= 15
    assert len(table.sha256) == 64


def test_only_priced_classes_and_valid_numbers_are_accepted(tmp_path):
    import yaml

    doc = yaml.safe_load(CONSENT.read_text(encoding="utf-8"))
    doc["classes"]["nature_reserves"]["treatment"] = "hard"
    f = tmp_path / "c.yaml"
    f.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="only priced classes"):
        ConsentTable.load(f, CostCatalogue.load(CATALOGUE))
    doc = yaml.safe_load(CONSENT.read_text(encoding="utf-8"))
    doc["classes"]["landscape_protection"]["bit"] = 0           # shares a bit
    f.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="share a flag bit"):
        ConsentTable.load(f, CostCatalogue.load(CATALOGUE))
    doc = yaml.safe_load(CONSENT.read_text(encoding="utf-8"))
    doc["classes"]["nature_reserves"]["one_off_eur"]["low"] = 99999
    f.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="low <= high"):
        ConsentTable.load(f, CostCatalogue.load(CATALOGUE))


def test_apply_prices_protected_cells_and_never_touches_forbidden_or_free_cells(table):
    base = np.array([[437, 437, 437, 800, FORBIDDEN, FORBIDDEN, 125, 65000]], dtype=np.uint16)
    flags = np.array([[0, 1 << 2, 1 << 1, (1 << 0) | (1 << 2), 1 << 4, 0, 3 << 8, 1 << 8]], dtype=np.uint16)
    out = table.apply(base, flags)
    assert out.dtype == np.uint16
    assert out[0, 0] == 437                              # free cell unchanged
    assert out[0, 1] == round(437 * 1.5)                 # Natura 2000: HDD multiplier
    assert out[0, 2] == 437                              # landscape protection: multiplier 1.0
    assert out[0, 3] == 1200                             # two classes: the largest multiplier, not the product
    assert out[0, 4] == FORBIDDEN and out[0, 5] == FORBIDDEN   # forbidden stays forbidden, flagged or not
    assert out[0, 6] == round(125 * 1.5)                 # water zone III
    assert out[0, 7] == PRICE_CAP                        # zone I (x100) would overflow: capped below forbidden
    assert (out >= np.minimum(base, PRICE_CAP)).all()
    assert ((out == FORBIDDEN) == (base == FORBIDDEN)).all()


def test_apply_keeps_the_exact_multiplier_of_the_largest_class_per_cell(table):
    rng = np.random.default_rng(5)
    base = rng.integers(50, 700, size=(64, 64)).astype(np.uint16)
    flags = rng.integers(0, 64, size=(64, 64)).astype(np.uint16) | (
        rng.integers(0, 6, size=(64, 64)).astype(np.uint16) << 8)
    mult = table.multiplier_of(flags)
    assert (mult >= 1.0).all()
    out = table.apply(base, flags)
    assert (out == np.minimum(np.rint(base * mult), PRICE_CAP).astype(np.uint16)).all()


def test_the_consent_lattice_takes_the_one_off_costs_and_delay(table):
    classes = list(table.classes)
    one_off = table.one_off()
    assert set(one_off) == set(classes) and all(v >= 0 for v in one_off.values())

    def masked_value(S):                  # a design that is cheaper the more classes it may enter
        return 1_000_000.0 - 20_000.0 * len(S) - (30_000.0 if "natura2000_habitats" in S else 0.0)

    res = consent_lattice(classes, one_off, masked_value, delay=table.delay_value)
    # entering a class pays C_k: only classes whose saving exceeds their one-off cost are used
    assert res.value <= masked_value(frozenset()) + 1e-9
    assert table.delay_months(["nature_reserves", "natura2000_habitats"]) == 9.0   # parallel: the longest
    assert table.delay_months([]) == 0.0 and table.delay_value(classes) == 0.0


@pytest.mark.skipif(not MANIFEST.exists(), reason="case-study manifest not present")
def test_the_registered_priced_raster_keeps_the_forbidden_cells_and_never_lowers_a_cost():
    import rasterio
    from rasterio.windows import Window

    from pyorps.siting.data_manifest import DataManifest

    m = DataManifest.load(MANIFEST)
    need = ("cost_base_1m_closed", "cost_priced_1m", "protection_flags_1m")
    if any(k not in m.layers or m.layer(k).status != "present" for k in need):
        pytest.skip("rasters not built")
    paths = [m.file_of(k) for k in need]
    if not all(p.is_file() for p in paths):
        pytest.skip("raster files not on disk")
    with rasterio.open(paths[0]) as b, rasterio.open(paths[1]) as p, rasterio.open(paths[2]) as f:
        for win in (Window(8000, 4000, 800, 800), Window(14000, 6000, 800, 800), Window(0, 0, 500, 500)):
            base, priced, flg = b.read(1, window=win), p.read(1, window=win), f.read(1, window=win)
            assert ((priced == FORBIDDEN) == (base == FORBIDDEN)).all()
            assert (priced >= base).all()
            assert (priced[flg == 0] == base[flg == 0]).all()
