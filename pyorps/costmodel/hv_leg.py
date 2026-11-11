"""The 110 kV cable leg: weight per metre on the shared trench raster (plan 2026-09-23 section 3.3, term 9).

Digging (civil) cost is the same for the collector voltages and 110 kV, so the leg is routed on the SAME priced trench
raster as the collector. One circuit, no trench sharing: the weight of one metre of route in a cell is

    (1 + omega) * c(cell) + rho,   rho = cores * material(section) * (1 + omega) + loss_coef * R_L(section) * W(all),

with ``omega`` the capitalised O&M multiple of the cable class (catalogue O&M share times the annuity factor, as in the
collector), ``c`` the priced trench rate, ``W(all)`` the whole-farm square sum of the joint power-category table (built
from the hourly record with wake, wind direction and the turbine time series), and ``loss_coef`` the capitalised loss
price at 110 kV. ``rho`` is the same in every cell, so the cheapest section is route-independent and chosen once. By
construction the 110 kV weight is never below the trench rate, hence never below a collector metre.

Items the plan lists but the catalogue has no key for (joints, cross-bonding, easement, fibre, dielectric loss) are
priced at zero and reported; cable ends and the bay belong to the connection-point term.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pyorps.core.exceptions import InvalidInputError
from pyorps.costmodel.losses import annuity_factor, loss_coef

__all__ = ["FORBIDDEN", "HvLegConstraints"]

FORBIDDEN = 65535
STATUSES = ("sourced", "decided", "catalogue", "derived", "assumption")
_CAP = FORBIDDEN - 1


def _walk_status(node, prefix: str, out: dict[str, str]) -> None:
    if not isinstance(node, Mapping):
        return
    if isinstance(node.get("status"), str):
        out[prefix.rstrip(".")] = node["status"]
    for key, val in node.items():
        if isinstance(val, Mapping):
            _walk_status(val, f"{prefix}{key}.", out)


@dataclass
class HvLegConstraints:
    """The resolved 110 kV leg model.

    Attributes:
        omega: Capitalised O&M multiple of the cable class.
        cores: Single-core cables per circuit.
        sections: ``{mm2: {"material": EUR/m incl. omega, "loss": EUR/m, "total": EUR/m}}``.
        section_mm2: The section with the smallest total.
        rho_eur_m: Its total, the cell-independent part of the weight.
        trench_raster: Layer id of the shared trench raster.
        unpriced: Plan items priced at zero.
    """
    path: Path
    sha256: str
    catalogue_sha256: str
    loss_inputs_sha256: str
    omega: float
    cores: int
    sections: dict[int, dict[str, float]]
    section_mm2: int
    rho_eur_m: float
    trench_raster: str
    unpriced: tuple[str, ...]
    statuses: dict[str, str]
    saturated_cells: int = 0

    @classmethod
    def load(cls, path, catalogue, inputs, loss, *, voltage_file=None, u_kv: float = 110.0,
             certified: bool = False) -> HvLegConstraints:
        """Resolve the file against the catalogue, the collector inputs and the loss table. Refuses a bad status,
        more than one circuit, and (``certified``) any remaining assumption."""
        from pyorps.core.optional_deps import import_yaml

        yaml = import_yaml()

        raw = Path(path).read_bytes()
        d = yaml.safe_load(raw.decode("utf-8"))
        st: dict[str, str] = {}
        _walk_status(d, "", st)
        bad = [f"{k}: {s!r}" for k, s in st.items() if s not in STATUSES]
        if bad:
            raise InvalidInputError(f"{path}: invalid status in " + "; ".join(bad))
        assumed = sorted(k for k, s in st.items() if s == "assumption")
        if certified and assumed:
            raise InvalidInputError("a certified build needs sourced inputs; assumptions remain: " + ", ".join(assumed))
        if int(d["route"]["circuits"]["value"]) != 1:
            raise InvalidInputError("only one circuit is modelled: there is no trench sharing on the 110 kV side")

        def val(node):
            return catalogue.value(node["catalogue_key"]) if "catalogue_key" in node else node["value"]

        cab = d["cable"]
        cores = int(val(cab["cores_per_circuit"]))
        mat = {int(k): float(v) for k, v in cab["material_eur_per_m_per_core"]["value"].items()}
        secs = [int(s) for s in cab["sections_mm2"]["value"]]

        c = inputs.data
        lc, ci = c["loss"], c["cable"]
        years, rate, growth = (float(lc[k]["value"]) for k in ("years", "rate", "growth"))
        af = annuity_factor(years, rate, growth)
        omega = float(val(cab["om_share_of_capex"])) * af
        price = float(lc["price_eur_per_mwh"]["headline"]["value"])
        coef = loss_coef(price_eur_per_mwh=price, u_kv=u_kv, years=years, rate=rate, growth=growth)
        w_all = float(loss.loss_weight(lc["weight_kind"]["value"])[(1 << loss.n) - 1])
        vfile = Path(voltage_file) if voltage_file else Path(inputs.path).parent / cab["voltage_data_file"]
        r90 = yaml.safe_load(vfile.read_text(encoding="utf-8"))["cables"]["r_ac90_ohm_per_km"]["value"]
        alpha = float(ci["alpha_aluminium_per_k"]["value"])
        factor = (1 + alpha * (float(ci["loss_temperature_c"]["value"]) - 20.0)) / (1 + alpha * 70.0)
        sections = {}
        for m in secs:
            material = cores * mat[m] * (1.0 + omega)
            loss_eur = coef * float(r90[m]) * factor / 1e3 * w_all
            sections[m] = {"material": material, "loss": loss_eur, "total": material + loss_eur}
        best = min(sections, key=lambda m: sections[m]["total"])
        return cls(path=Path(path), sha256=hashlib.sha256(raw).hexdigest(),
                   catalogue_sha256=getattr(catalogue, "sha256", ""), loss_inputs_sha256=getattr(inputs, "sha256", ""),
                   omega=omega, cores=cores, sections=sections, section_mm2=best, rho_eur_m=sections[best]["total"],
                   trench_raster=str(d["route"]["trench_raster"]["value"]),
                   unpriced=tuple(cab["unpriced_per_metre_items"]["value"]), statuses=st)

    def assumptions(self) -> list[str]:
        return sorted(k for k, s in self.statuses.items() if s == "assumption")

    def cost_block(self, trench: np.ndarray) -> np.ndarray:
        """The 110 kV weight (uint16 EUR per metre, rounded up) for a block of the priced trench raster; forbidden
        cells stay forbidden, every other cell is at least its trench rate. A weight above 65 534 is capped there, as
        the priced trench raster itself caps (zone-I water protection, x100); ``saturated_cells`` counts them."""
        t = np.asarray(trench, dtype=np.uint16)
        w = np.ceil((1.0 + self.omega) * t.astype(np.float64) + self.rho_eur_m - 1e-9)
        self.saturated_cells += int((w[t != FORBIDDEN] > _CAP).sum())
        out = np.minimum(w, _CAP).astype(np.uint16)
        out[t == FORBIDDEN] = FORBIDDEN
        return out

    def record(self) -> dict:
        """JSON-able provenance."""
        return {"file": self.path.name, "sha256": self.sha256, "catalogue_sha256": self.catalogue_sha256,
                "loss_inputs_sha256": self.loss_inputs_sha256, "omega": self.omega, "cores": self.cores,
                "section_mm2": self.section_mm2, "rho_eur_per_m": self.rho_eur_m, "sections": self.sections,
                "trench_raster": self.trench_raster, "priced_at_zero": list(self.unpriced),
                "assumptions": self.assumptions()}
