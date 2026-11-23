"""Branch constants K(b) of the all-cable scenario (paper math.md M4.8; plan 2026-09-23 section 2.3, terms 4, 5, 14).

``K(b)`` does not depend on the pad position, so it cannot change which position is best for a fixed branch, but it
decides between branches (a connection-point bay against a tap on an existing line) and it makes lower bounds and
incumbents comparable with the plan's objective. Only terms with a catalogue or sourced value are in the ledger; the
rest are named in ``outside_the_model`` so the paper can state its scope.

``K_min`` (cheapest branch) goes into the lower bound, ``K_PCC`` (the branch that connects at a connection point) into
an incumbent that does.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from pyorps.core.exceptions import InvalidInputError

__all__ = ["BranchConstants"]

STATUSES = ("sourced", "decided", "catalogue", "derived", "assumption")


def _value(node, catalogue) -> float:
    if not isinstance(node, Mapping) or node.get("status") not in STATUSES:
        raise InvalidInputError(f"every ledger entry needs a status from {STATUSES}: {node!r}")
    if "catalogue_key" in node:
        v = float(catalogue.value(node["catalogue_key"]))
    elif "value" in node:
        v = float(node["value"])
    else:
        raise InvalidInputError(f"entry without value or catalogue_key: {node!r}")
    if v < 0:
        raise InvalidInputError("ledger values must be >= 0 (the bounds need non-negative constants)")
    return v


@dataclass(frozen=True)
class BranchConstants:
    """The resolved ledger.

    Attributes:
        station_eur: Minimal greenfield 110 kV station, bottom-up: transformer + bays + building (protection is in the
            bay price).
        pcc_bay_eur: Line bay at a connection point.
        tap_works_eur: Works of a tap on an existing line.
        outside: Terms the model does not price.
        statuses: ``{entry: status}``.
    """
    station_eur: float
    pcc_bay_eur: float
    tap_works_eur: float
    outside: tuple[str, ...]
    statuses: Mapping[str, str]

    @classmethod
    def load(cls, path, catalogue) -> BranchConstants:
        from pyorps.core.optional_deps import import_yaml

        yaml = import_yaml()

        doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        st = doc["station"]
        cn = doc["connection"]
        station = (_value(st["transformer_eur"], catalogue)
                   + _value(st["bays_in_minimal_station"], catalogue) * _value(st["bay_eur"], catalogue)
                   + _value(st["building_eur"], catalogue))
        pcc, tap = _value(cn["pcc_bay_eur"], catalogue), _value(cn["tap_works_eur"], catalogue)
        statuses = {"station.transformer_eur": st["transformer_eur"]["status"],
                    "station.bays_in_minimal_station": st["bays_in_minimal_station"]["status"],
                    "station.bay_eur": st["bay_eur"]["status"],
                    "station.building_eur": st["building_eur"]["status"],
                    "connection.pcc_bay_eur": cn["pcc_bay_eur"]["status"],
                    "connection.tap_works_eur": cn["tap_works_eur"]["status"]}
        return cls(station_eur=station, pcc_bay_eur=pcc, tap_works_eur=tap,
                   outside=tuple(doc.get("outside_the_model", ())), statuses=statuses)

    @property
    def k_min(self) -> float:
        """Smallest K(b) over the connection branches: the constant of the lower bound."""
        return self.station_eur + min(self.pcc_bay_eur, self.tap_works_eur)

    @property
    def k_pcc(self) -> float:
        """K(b) of a branch that connects at a connection point."""
        return self.station_eur + self.pcc_bay_eur

    def record(self) -> dict:
        return {"station_eur": self.station_eur, "pcc_bay_eur": self.pcc_bay_eur, "tap_works_eur": self.tap_works_eur,
                "k_min": self.k_min, "k_pcc": self.k_pcc, "statuses": dict(self.statuses),
                "outside_the_model": list(self.outside)}
