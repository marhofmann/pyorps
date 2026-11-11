"""Build a :class:`~pyorps.collector.model.CollectorModel` from sourced inputs
(plan rev. 5, Phase C closure, G5).

Three sources meet here, so no number is typed twice:

* the inputs file (``case_studies/runkel_free_siting/config/collector_inputs_2026.yaml``),
  every value tagged ``sourced``, ``decided``, ``catalogue``, ``derived`` or
  ``assumption``;
* the cost catalogue (:class:`~pyorps.costmodel.catalogue.CostCatalogue`), read
  at build time, so a catalogue-priced item equals the catalogue to the cent;
* the loss table (:class:`~pyorps.costmodel.losses.LossCategories`) and the
  voltage data file (cable ampacity and resistance).

An ``assumption`` is a stand-in because research found no primary source. A
build with ``certified=True`` refuses to run while one remains; every build
records them (:attr:`BuiltCollector.assumptions`).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from pyorps.collector.condition_c import ConditionC, condition_c
from pyorps.collector.model import CableType, CollectorModel
from pyorps.costmodel.catalogue import CostCatalogue
from pyorps.costmodel.losses import LossCategories, annuity_factor, loss_coef

__all__ = ["BuiltCollector", "CollectorInputs", "build_collector_model"]

STATUSES = ("sourced", "decided", "catalogue", "derived", "assumption")

SCOPE_SENTENCES = (
    "the collector trenches form a tree (no loops); parallel trenches in one "
    "cell are separate trenches and do not derate each other; a system keeps "
    "one cable option between two nodes; no turbine's power crosses a trench "
    "twice in the same direction (R-once)",
)


@dataclass
class CollectorInputs:
    """The inputs file, read once."""
    data: dict
    path: Path
    sha256: str

    @classmethod
    def load(cls, path) -> CollectorInputs:
        """Read the file and check that every tagged node has a valid status."""
        import yaml

        raw = Path(path).read_bytes()
        data = yaml.safe_load(raw.decode("utf-8"))
        inp = cls(data=data, path=Path(path),
                  sha256=hashlib.sha256(raw).hexdigest())
        bad = [f"{k}: {s!r}" for k, s in inp.statuses().items()
               if s not in STATUSES]
        if bad:
            raise ValueError(f"{path}: invalid status in " + "; ".join(bad))
        return inp

    def statuses(self) -> dict[str, str]:
        """``{dotted path: status}`` for every node that carries one."""
        out: dict[str, str] = {}
        _walk_status(self.data, "", out)
        return out

    def assumptions(self) -> list[str]:
        """Dotted paths of every value that is only an assumption."""
        return sorted(k for k, s in self.statuses().items()
                      if s == "assumption")


def _walk_status(node, prefix: str, out: dict[str, str]) -> None:
    if not isinstance(node, Mapping):
        return
    if "status" in node and isinstance(node["status"], str):
        out[prefix.rstrip(".")] = node["status"]
    for key, val in node.items():
        if isinstance(val, Mapping):
            _walk_status(val, f"{prefix}{key}.", out)


@dataclass
class BuiltCollector:
    """A model plus the record that explains it.

    Attributes:
        model: The collector model.
        record: JSON-able provenance (hashes, loss price, resolved catalogue
            values, the assumptions that remain).
        assumptions: Dotted paths of the unsourced values used.
        condition: Condition (C), set by :meth:`check_condition_c`.
    """
    model: CollectorModel
    record: dict
    assumptions: tuple[str, ...]
    condition: ConditionC | None = field(default=None)

    def check_condition_c(self, min_trench_rate_eur_per_m: float
                          ) -> ConditionC:
        """Evaluate condition (C) against the cheapest trench rate and keep
        the result for :meth:`report`."""
        self.condition = condition_c(self.model, min_trench_rate_eur_per_m)
        self.record["condition_c"] = self.condition.as_record()
        return self.condition

    def report(self) -> str:
        """The text printed with every result: scope sentences, the
        assumptions that remain, and condition (C) with its margin."""
        lines = ["Collector model D_share, scope: "
                 + "; ".join(SCOPE_SENTENCES) + "."]
        if self.assumptions:
            lines.append(f"{len(self.assumptions)} input(s) are assumptions, "
                         "not sourced: " + ", ".join(self.assumptions))
        else:
            lines.append("no input is an unsourced assumption")
        lines.append(str(self.condition) if self.condition is not None
                     else "condition (C): not evaluated (no trench rate given)")
        return "\n".join(lines)


def _val(node):
    """``node["value"]`` of a tagged leaf."""
    return node["value"]


def _level_key(u_kv: float) -> str:
    return str(int(round(u_kv)))


def _price_table(spec: Mapping, u_kv: float) -> dict[int, float]:
    """Cable price per cross-section at ``u_kv``: the 20 or 33 kV column, or the
    linear interpolation in kV between them."""
    lo, hi = spec["20"], spec["33"]
    key = _level_key(u_kv)
    if key in spec:
        return {int(m): float(v) for m, v in spec[key].items()}
    if not 20.0 < u_kv < 33.0:
        raise ValueError(f"no cable price for {u_kv} kV (20 to 33 kV only)")
    w = (u_kv - 20.0) / 13.0
    return {int(m): float(lo[m]) * (1 - w) + float(hi[m]) * w for m in lo}


def _cable_electrics(voltage_file: Path, u_kv: float, sections: list[int],
                     alpha: float, temp_c: float
                     ) -> dict[int, tuple[float, float]]:
    """``{mm2: (ampacity A, r ohm/m at temp_c)}`` from the voltage data file."""
    import yaml

    cab = yaml.safe_load(voltage_file.read_text(encoding="utf-8"))["cables"]
    key = _level_key(u_kv)
    cls = {"20": "kv20", "30": "kv30", "33": "kv33"}.get(key)
    if cls is None:
        raise ValueError(f"no cable class for {u_kv} kV (20, 30 or 33)")
    table = cab[cls]
    if cls == "kv30":                    # [trefoil, flat] per cross-section
        amp = {m: float(table["ampacity_trefoil_flat_a"]["value"][m][0])
               for m in sections}
    else:                                # one value per manufacturer source
        amp = {m: float(min(table["ampacity_trefoil_a"]["value"][m]))
               for m in sections}
    factor = (1 + alpha * (temp_c - 20.0)) / (1 + alpha * (90.0 - 20.0))
    r90 = cab["r_ac90_ohm_per_km"]["value"]
    return {m: (amp[m], float(r90[m]) * factor / 1e3) for m in sections}


def build_collector_model(inputs: CollectorInputs, catalogue: CostCatalogue,
                          loss: LossCategories, *, u_kv: float,
                          rated_mw, price: str = "headline",
                          certified: bool = False,
                          voltage_file: str | Path | None = None
                          ) -> BuiltCollector:
    """One MV level of ``D_share`` from the inputs, the catalogue and the loss
    table.

    Parameters:
        inputs: The inputs file.
        catalogue: The cost catalogue (items marked ``catalogue_key`` are read
            from it).
        loss: The joint power-category table; its turbine count must equal
            ``len(rated_mw)``.
        u_kv: Collector voltage (20, 30 or 33).
        rated_mw: Rated power per turbine.
        price: ``"headline"`` or ``"sensitivity"``, the loss price of decision D5.
        certified: Refuse to build while any input is an assumption.
        voltage_file: The voltage data file (default: the file named in the
            inputs, beside them).

    Returns:
        :class:`BuiltCollector`.
    """
    d = inputs.data
    assumed = tuple(inputs.assumptions())
    if certified and assumed:
        raise ValueError(
            f"a certified build needs sourced inputs; {len(assumed)} "
            "assumption(s) remain: " + ", ".join(assumed))
    rated = [float(x) for x in rated_mw]
    if loss.n != len(rated):
        raise ValueError(f"the loss table has {loss.n} turbines, "
                         f"rated_mw has {len(rated)}")
    if price not in ("headline", "sensitivity"):
        raise ValueError("price must be 'headline' or 'sensitivity'")

    lc = d["loss"]
    years, rate, growth = (float(_val(lc[k])) for k in
                           ("years", "rate", "growth"))
    af = annuity_factor(years, rate, growth)
    price_eur = float(_val(lc["price_eur_per_mwh"][price]))
    coef = loss_coef(price_eur_per_mwh=price_eur, u_kv=u_kv, years=years,
                     rate=rate, growth=growth)

    cab = d["cable"]
    sections = [int(m) for m in _val(cab["sections_mm2"])]
    vfile = Path(voltage_file) if voltage_file else (
        inputs.path.parent / cab["voltage_data_file"])
    electrics = _cable_electrics(
        vfile, u_kv, sections, float(_val(cab["alpha_aluminium_per_k"])),
        float(_val(cab["loss_temperature_c"])))
    prices = _price_table(cab["price_eur_per_m"], u_kv)
    om = catalogue.value(cab["om_share_of_capex"]["catalogue_key"])
    extra = (float(_val(cab["end_of_life_eur_per_m"]))
             + float(_val(cab["dielectric_loss_eur_per_m"])))
    types = tuple(
        CableType(name=f"NA2XS(F)2Y 1x{m} Al {u_kv:g} kV",
                  ampacity_a=electrics[m][0],
                  cost_eur_per_m=prices[m] * (1.0 + om * af) + extra,
                  r_ohm_per_m=electrics[m][1])
        for m in sections)

    ts, sw = d["trench_sharing"], d["switchgear"]
    sg = ts["sigma_eur_per_m"]
    civil = catalogue.value(sg["civil_base_catalogue_key"])
    sigma = civil * float(_val(sg["extra_width_per_system_m"])) / float(
        _val(sg["base_trench_width_m"]))

    def cat(node):
        return catalogue.value(node["catalogue_key"]) / float(
            node.get("divisor", 1.0))

    key = _level_key(u_kv)
    model = CollectorModel.from_turbines(
        rated_mw=rated, u_kv=float(u_kv),
        gamma=float(_val(d["turbine_current"]["gamma"])),
        loss_weight=loss.loss_weight(_val(lc["weight_kind"])),
        types=types, loss_coef=coef,
        derating=tuple(float(x) for x in ts["derating"]["values"]),
        sigma_eur_per_m=sigma, m_max=int(_val(ts["m_max"])),
        p_max=int(_val(ts["p_max"])),
        switchgear_a=float(_val(sw["turbine_switchgear_a"])),
        ring_panels=int(_val(sw["ring_panels"])),
        turbine_panel_eur=cat(sw["turbine_panel_eur"]),
        station_building_eur=cat(sw["station_building_eur"]),
        station_panel_eur=cat(sw["station_panel_eur"]),
        bay_eur=cat(sw["bay_eur"]),
        bay_a=float(_val(sw["bay_a"][key])))

    record = {
        "u_kv": float(u_kv), "price": price, "loss_price_eur_per_mwh": price_eur,
        "annuity_factor": af, "loss_coef": coef, "sigma_eur_per_m": sigma,
        "om_share_of_capex": om,
        "inputs_sha256": inputs.sha256,
        "catalogue_sha256": catalogue.sha256,
        "loss_table_sha256": loss.sha256,
        "voltage_file": vfile.name,
        "assumptions": list(assumed),
        "certified": certified,
        "model_sha256": hashlib.sha256(json.dumps(
            model.describe(), sort_keys=True,
            default=lambda x: None if math.isinf(x) else x).encode()
        ).hexdigest(),
    }
    return BuiltCollector(model=model, record=record, assumptions=assumed)
