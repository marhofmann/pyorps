"""Priced consent classes (plan rev. 5, section 2.6 and 3.6): protected areas are priced, not forbidden.

Owner decision of 2026-10-05: protected areas are PRICED. A class has two parts:

* a per-metre multiplier on the trench cost while a trench lies inside the area (the price of drilling
  underneath, HDD). It is separable, so it is applied to the cost raster (:meth:`ConsentTable.apply`);
* a one-off cost ``C_k`` and a procedure time in months when a design uses the class at all. That is
  not separable and goes to the consent lattice of relax-and-verify
  (:func:`pyorps.certify.rv.consent_lattice`) through :meth:`ConsentTable.one_off`,
  :meth:`ConsentTable.delay_months` and :meth:`ConsentTable.delay_value`.

Hard exclusions remain only where the land-use recipe marks a class forbidden: :meth:`apply` never turns a
forbidden cell (65535) into a passable one, and never lowers a cost.

The values come from ``case_studies/runkel_free_siting/config/consent_classes_2026.yaml``; every value carries
a status (``sourced``, ``decided``, ``catalogue``, ``derived``, ``assumption``) and a build can refuse
assumptions, as in :mod:`pyorps.costmodel.collector_inputs`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pyorps.core.exceptions import InvalidInputError
from pyorps.costmodel.catalogue import CostCatalogue

__all__ = ["ConsentClass", "ConsentTable", "FORBIDDEN", "PRICE_CAP"]

#: the cost raster's forbidden value, and the largest value a priced cell may take
FORBIDDEN = 65535
PRICE_CAP = 65534

_STATUSES = ("sourced", "decided", "catalogue", "derived", "assumption")
WATER_ZONE_SHIFT = 8
WATER_ZONE_MASK = 0xF


@dataclass(frozen=True)
class ConsentClass:
    """One protected-area class.

    Attributes:
        name: The manifest id of the layer (also the consent-lattice class key).
        bit: Its bit in the protection flag raster.
        multiplier: Cost multiplier while inside (>= 1).
        one_off_eur: Permit and assessment cost ``C_k`` of using the class (middle of the stated range).
        one_off_range: ``(low, high)`` as stated.
        delay_months: Procedure time of the class.
        confidence: Tier of the numbers (``LOW`` for every stand-in).
    """
    name: str
    bit: int
    multiplier: float
    one_off_eur: float
    one_off_range: tuple[float, float]
    delay_months: float
    confidence: str


@dataclass
class ConsentTable:
    """The consent classes, the water-zone multipliers and the delay price."""
    classes: dict[str, ConsentClass]
    water_multipliers: dict[int, float]
    delay_cost_eur_per_month: float
    statuses: dict[str, str] = field(default_factory=dict)
    sha256: str | None = None

    # ------------------------------------------------------------ loading

    @classmethod
    def load(cls, path, catalogue: CostCatalogue) -> ConsentTable:
        """Read the table; ``use: hdd_multiplier`` entries are resolved from the catalogue."""
        from pyorps.core.optional_deps import import_yaml

        yaml = import_yaml()

        raw = Path(path).read_bytes()
        doc = yaml.safe_load(raw.decode("utf-8"))
        hdd = catalogue.value(doc["hdd_multiplier"]["catalogue_key"])
        statuses: dict[str, str] = {"hdd_multiplier": doc["hdd_multiplier"]["status"]}
        classes: dict[str, ConsentClass] = {}
        for name, body in doc["classes"].items():
            if body.get("treatment") != "priced":
                raise InvalidInputError(f"{name}: only priced classes are supported (owner decision, 2026-10-05)")
            mult_node = body["per_metre_multiplier"]
            mult = hdd if mult_node.get("use") == "hdd_multiplier" else float(mult_node["value"])
            one = body["one_off_eur"]
            lo, hi = float(one["low"]), float(one["high"])
            if not (0.0 <= lo <= hi) or mult < 1.0 or float(body["delay_months"]["value"]) < 0:
                raise InvalidInputError(f"{name}: need 0 <= low <= high, multiplier >= 1, delay >= 0")
            for key in ("per_metre_multiplier", "one_off_eur", "delay_months"):
                statuses[f"{name}.{key}"] = body[key]["status"]
            classes[name] = ConsentClass(
                name=name, bit=int(body["bit"]), multiplier=float(mult), one_off_eur=0.5 * (lo + hi),
                one_off_range=(lo, hi), delay_months=float(body["delay_months"]["value"]),
                confidence=str(body.get("confidence", "LOW")))
        if len({c.bit for c in classes.values()}) != len(classes):
            raise InvalidInputError("two classes share a flag bit")
        wz = doc["water_zones"]["per_metre_multiplier"]
        statuses["water_zones.per_metre_multiplier"] = wz["status"]
        water = {int(k): float(v) for k, v in wz.items() if str(k).isdigit()}
        if any(v < 1.0 for v in water.values()):
            raise InvalidInputError("water zone multipliers must be >= 1")
        dc = doc["delay_cost"]
        statuses["delay_cost"] = dc["status"]
        bad = {k: s for k, s in statuses.items() if s not in _STATUSES}
        if bad:
            raise InvalidInputError(f"invalid status in {bad}")
        return cls(classes=classes, water_multipliers=water,
                   delay_cost_eur_per_month=float(dc["value_eur_per_month"]), statuses=statuses,
                   sha256=hashlib.sha256(raw).hexdigest())

    def assumptions(self) -> list[str]:
        """Dotted names of every value that is only a stand-in."""
        return sorted(k for k, s in self.statuses.items() if s == "assumption")

    # ------------------------------------------------- the consent lattice

    def one_off(self) -> dict[str, float]:
        """``{class: C_k}`` for :func:`pyorps.certify.rv.consent_lattice`."""
        return {n: c.one_off_eur for n, c in self.classes.items()}

    def delay_months(self, used: Iterable[str]) -> float:
        """The longest procedure chain among the classes a design uses (parallel procedures)."""
        return max((self.classes[k].delay_months for k in used), default=0.0)

    def delay_value(self, used: Iterable[str]) -> float:
        """``V(S)``: non-decreasing in ``S``, zero while the delay price is zero (plan 2.6)."""
        return self.delay_cost_eur_per_month * self.delay_months(used)

    # --------------------------------------------------------- the raster

    def multiplier_luts(self) -> tuple[np.ndarray, np.ndarray]:
        """``(per class bit multipliers [16], water zone multipliers [16])``, 1.0 where unused."""
        bits = np.ones(16, dtype=np.float64)
        for c in self.classes.values():
            bits[c.bit] = c.multiplier
        water = np.ones(16, dtype=np.float64)
        for code, mult in self.water_multipliers.items():
            water[code] = mult
        return bits, water

    def multiplier_of(self, flags: np.ndarray) -> np.ndarray:
        """The cost multiplier of every cell: the largest of its classes' and its water zone's."""
        flags = np.asarray(flags, dtype=np.uint16)
        bits, water = self.multiplier_luts()
        out = water[(flags >> WATER_ZONE_SHIFT) & WATER_ZONE_MASK]
        for bit in range(16):
            if bits[bit] != 1.0:
                has = (flags >> bit) & 1
                out = np.where(has == 1, np.maximum(out, bits[bit]), out)
        return out

    def apply(self, base: np.ndarray, flags: np.ndarray) -> np.ndarray:
        """Priced cost: ``round(base x multiplier)``, capped just below the forbidden value.

        A forbidden cell stays forbidden; a passable cell never becomes forbidden (the cap) and never gets
        cheaper.
        """
        base = np.asarray(base, dtype=np.uint16)
        priced = np.minimum(np.rint(base * self.multiplier_of(flags)), PRICE_CAP).astype(np.uint16)
        priced = np.maximum(priced, np.minimum(base, PRICE_CAP).astype(np.uint16))
        return np.where(base == FORBIDDEN, np.uint16(FORBIDDEN), priced).astype(np.uint16)
