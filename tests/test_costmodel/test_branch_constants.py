"""The branch-constants ledger (math.md M4.8)."""
from pathlib import Path

import pytest
import yaml

from pyorps.costmodel import CostCatalogue
from pyorps.costmodel.branch_constants import BranchConstants

CASE = Path(__file__).resolve().parents[2] / "case_studies" / "runkel_free_siting" / "config"


@pytest.fixture(scope="module")
def cat():
    if not (CASE / "branch_constants_2026.yaml").exists():
        pytest.skip("case-study files not present")
    return CostCatalogue.load(CASE / "cost_catalogue_2026.yaml")


def test_the_shipped_ledger_resolves_from_the_catalogue(cat):
    k = BranchConstants.load(CASE / "branch_constants_2026.yaml", cat)
    bay = cat.value("substations_110kv.incremental_line_bay_eur.ais")
    assert k.station_eur == pytest.approx(2_400_000 + 2 * bay + 800_000)       # bottom-up, 63 MVA 110/33 kV
    assert k.pcc_bay_eur == bay
    # tap works are zero, so the cheapest branch has no connection constant
    assert k.k_min <= k.k_pcc and k.k_min == k.station_eur
    assert "consent_and_delay" in k.outside
    assert set(k.statuses.values()) <= {"sourced", "decided", "catalogue", "assumption"}


def test_negative_values_and_missing_status_are_refused(cat, tmp_path):
    doc = yaml.safe_load((CASE / "branch_constants_2026.yaml").read_text(encoding="utf-8"))
    bad = dict(doc, connection={"pcc_bay_eur": {"value": -1, "status": "assumption"},
                                "tap_works_eur": {"value": 0, "status": "assumption"}})
    (tmp_path / "a.yaml").write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ValueError, match=">= 0"):
        BranchConstants.load(tmp_path / "a.yaml", cat)
    bad2 = dict(doc, connection={"pcc_bay_eur": {"value": 1}, "tap_works_eur": {"value": 0, "status": "assumption"}})
    (tmp_path / "b.yaml").write_text(yaml.safe_dump(bad2), encoding="utf-8")
    with pytest.raises(ValueError, match="status"):
        BranchConstants.load(tmp_path / "b.yaml", cat)
