"""The C1 cost-catalogue loader: tiers, validation, hashing."""
from pathlib import Path

import pytest

pytest.importorskip("yaml")

from pyorps.costmodel.catalogue import CONFIDENCE_TOLERANCE, CostCatalogue

CATALOGUE = Path(__file__).resolve().parents[2] / "case_studies" / \
    "runkel_free_siting" / "config" / "cost_catalogue_2026.yaml"

_REFS = """
references:
  src_a: {title: "A", url: null}
  interpolated: {title: "none", url: null}
"""


def _write(tmp_path, body):
    p = tmp_path / "cat.yaml"
    p.write_text(_REFS + body, encoding="utf-8")
    return p


class TestSynthetic:
    def test_items_are_flattened_with_dotted_keys(self, tmp_path):
        p = _write(tmp_path, """
section:
  sub:
    one: {value: 12.5, source: src_a, confidence: HIGH}
    band: {low: 1.0, high: 3.0, source: [src_a, interpolated], confidence: low}
  nested:
    source: src_a
    confidence: MEDIUM
    investment: {low: 2.0, high: 4.0}
    kind: "text only"
""")
        cat = CostCatalogue.load(p)
        assert cat.value("section.sub.one") == 12.5
        assert cat["section.sub.band"].central == 2.0
        assert cat["section.sub.band"].confidence == "LOW"
        assert cat["section.sub.band"].sources == ("src_a", "interpolated")
        child = cat["section.nested.investment"]
        assert child.band == (2.0, 4.0) and child.confidence == "MEDIUM"
        assert cat["section.nested"].qualitative
        assert cat["section.nested"].text == {"kind": "text only"}
        assert set(cat.section("section.sub")) == {"section.sub.one",
                                                   "section.sub.band"}

    @pytest.mark.parametrize("bad,match", [
        ("{value: 1, source: nowhere, confidence: HIGH}", "unknown source"),
        ("{value: 1, source: src_a, confidence: SOMEWHAT}", "confidence"),
        ("{low: 5, high: 1, source: src_a, confidence: LOW}", "low 5.0 > high"),
        ("{value: 9, low: 1, high: 3, source: src_a, confidence: LOW}",
         "outside"),
    ])
    def test_invalid_items_are_refused(self, tmp_path, bad, match):
        p = _write(tmp_path, f"section:\n  x: {bad}\n")
        with pytest.raises(ValueError, match=match):
            CostCatalogue.load(p)

    def test_derived_terms_obey_the_tier_tolerance(self, tmp_path):
        p = _write(tmp_path, """
s:
  hi: {value: 100.0, source: src_a, confidence: HIGH}
  lo: {value: 100.0, source: src_a, confidence: LOW}
""")
        cat = CostCatalogue.load(p)
        assert cat.check_derived("s.hi", 100.9) < 0.01
        with pytest.raises(ValueError, match="HIGH item allows 1%"):
            cat.check_derived("s.hi", 102.0)
        assert cat.check_derived("s.lo", 114.0) < 0.15
        assert CONFIDENCE_TOLERANCE["MEDIUM"] == 0.05

    def test_parameter_hash_follows_the_numbers(self, tmp_path):
        a = CostCatalogue.load(_write(tmp_path, "s:\n  x: {value: 1.0, "
                                                "source: src_a, confidence: HIGH}\n"))
        b = CostCatalogue.load(_write(tmp_path, "s:\n  x: {value: 1.0000001, "
                                                "source: src_a, confidence: HIGH}\n"))
        assert a.parameter_hash() != b.parameter_hash()


@pytest.mark.skipif(not CATALOGUE.exists(), reason="case-study catalogue absent")
class TestCaseStudyCatalogue:
    def test_loads_and_carries_the_anchors(self):
        cat = CostCatalogue.load(CATALOGUE)
        assert cat.value("substations_110kv.incremental_line_bay_eur.ais") \
            == 1_300_000
        tiers = cat.by_confidence()
        assert set(tiers) <= set(CONFIDENCE_TOLERANCE)
        assert sum(tiers.values()) == len(cat.items) > 100

    def test_section_43h_note_is_the_corrected_reading(self):
        cat = CostCatalogue.load(CATALOGUE)
        item = cat["cross_check_cable_vs_ohl_ratio_by_voltage."
                   "enwg_regulatory_cap_up_to_110kv"]
        assert item.central == 2.75
        assert "PLANNING rule" in item.note
        assert "reimbursed" not in item.note
