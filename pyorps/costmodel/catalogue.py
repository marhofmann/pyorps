"""The cost catalogue with confidence tiers (plan rev. 5, Phase C1).

Plan C1: "One YAML loader with tiers (today no Python reads the
catalogue)." The catalogue
(``case_studies/runkel_free_siting/config/cost_catalogue_2026.yaml``, a
copy of TopoMILP's with the section 43h note corrected) stores every cost
figure as a mapping with a ``source`` key into its ``references`` block and
a ``confidence`` tier, plus the numbers themselves under ``value``, or
``low``/``high`` (optionally ``recommended``), or a few named variants
(``typical``, ``difficult_terrain``, ...).

:class:`CostCatalogue` reads it, checks every item (known sources, known
tiers, numeric fields, ``low <= value <= high``), gives each item a dotted
key such as ``substations_110kv.incremental_line_bay_eur.ais``, and hashes
the whole parameter vector for the plan-A1 provenance record.

The tier tolerances are plan C's acceptance rule: a derived composite must
lie within 1 % (HIGH), 5 % (MEDIUM) or 15 % (LOW, INTERPOLATED) of its
anchor.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["CONFIDENCE_TOLERANCE", "CostCatalogue", "CostItem"]

#: Plan C acceptance: relative tolerance of a derived term per tier.
CONFIDENCE_TOLERANCE = {"HIGH": 0.01, "MEDIUM": 0.05, "LOW": 0.15,
                        "INTERPOLATED": 0.15}

_NUMERIC_ORDER = ("value", "recommended", "typical")


@dataclass(frozen=True)
class CostItem:
    """One sourced figure of the catalogue."""
    key: str
    numbers: Mapping[str, float]
    sources: tuple[str, ...]
    confidence: str
    note: str | None = None
    text: Mapping[str, str] = field(default_factory=dict)

    @property
    def qualitative(self) -> bool:
        """A sourced fact without a number (e.g. a pole type or a trench
        geometry)."""
        return not self.numbers

    @property
    def central(self) -> float:
        """``value``, else ``recommended``, else ``typical``, else the
        midpoint of ``low``/``high``."""
        for k in _NUMERIC_ORDER:
            if k in self.numbers:
                return float(self.numbers[k])
        if "low" in self.numbers and "high" in self.numbers:
            return 0.5 * (float(self.numbers["low"])
                          + float(self.numbers["high"]))
        raise KeyError(f"{self.key} has no central value "
                       f"(fields {sorted(self.numbers)})")

    @property
    def band(self) -> tuple[float, float]:
        """``(low, high)`` when given, else the central value twice."""
        lo = self.numbers.get("low")
        hi = self.numbers.get("high")
        c = self.central
        return (float(lo) if lo is not None else c,
                float(hi) if hi is not None else c)

    @property
    def tolerance(self) -> float:
        """Plan C's relative acceptance tolerance for this tier."""
        return CONFIDENCE_TOLERANCE[self.confidence]


@dataclass
class CostCatalogue:
    """A validated, flattened cost catalogue."""
    items: dict[str, CostItem]
    references: dict[str, dict]
    meta: dict = field(default_factory=dict)
    sha256: str | None = None
    path: str | None = None

    @classmethod
    def load(cls, path) -> CostCatalogue:
        """Read and validate a catalogue YAML."""
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        try:
            import yaml
        except ImportError as exc:                      # pragma: no cover
            raise ImportError(
                "reading the cost catalogue needs PyYAML "
                "(pip install pyyaml)") from exc
        raw = Path(path).read_bytes()
        doc = yaml.safe_load(raw.decode("utf-8"))
        if not isinstance(doc, Mapping):
            raise ValueError(f"{path}: not a mapping at top level")
        refs = dict(doc.get("references") or {})
        items: dict[str, CostItem] = {}
        problems: list[str] = []
        for top, body in doc.items():
            if top in ("references",):
                continue
            _walk(body, top, items, problems)
        for it in items.values():
            for s in it.sources:
                if s not in refs:
                    problems.append(f"{it.key}: unknown source {s!r}")
        if problems:
            raise ValueError(f"{path}: {len(problems)} problem(s):\n  "
                             + "\n  ".join(problems[:30]))
        return cls(items=items, references=refs, meta=dict(doc.get("meta") or {}),
                   sha256=hashlib.sha256(raw).hexdigest(), path=str(path))

    def __getitem__(self, key: str) -> CostItem:
        try:
            return self.items[key]
        except KeyError:
            near = [k for k in self.items if k.startswith(key.split(".")[0])]
            raise KeyError(f"{key!r} not in the catalogue; "
                           f"same section: {near[:8]}") from None

    def value(self, key: str) -> float:
        """The central value of an item."""
        return self[key].central

    def section(self, prefix: str) -> dict[str, CostItem]:
        """Every item under a dotted prefix."""
        p = prefix.rstrip(".") + "."
        return {k: v for k, v in self.items.items() if k.startswith(p)}

    def by_confidence(self) -> dict[str, int]:
        """How many items sit in each tier."""
        out: dict[str, int] = {}
        for it in self.items.values():
            out[it.confidence] = out.get(it.confidence, 0) + 1
        return out

    def parameter_vector(self) -> dict[str, dict]:
        """``{key: {numbers, confidence}}`` -- what the A1 hash covers."""
        return {k: {"numbers": dict(v.numbers), "confidence": v.confidence}
                for k, v in sorted(self.items.items())}

    def parameter_hash(self) -> str:
        """SHA-256 of :meth:`parameter_vector` (plan A1 cost-model key)."""
        from pyorps.io.provenance import parameter_hash
        return parameter_hash(self.parameter_vector())

    def check_derived(self, key: str, derived: float) -> float:
        """Relative deviation of a derived term from its anchor; raise when
        it exceeds the tier's tolerance (plan C acceptance)."""
        it = self[key]
        anchor = it.central
        rel = abs(derived - anchor) / abs(anchor) if anchor else abs(derived)
        if rel > it.tolerance:
            raise ValueError(
                f"{key}: derived {derived:.6g} is {rel:.1%} from the anchor "
                f"{anchor:.6g}; a {it.confidence} item allows "
                f"{it.tolerance:.0%}")
        return rel


def _walk(node: Any, key: str, items: dict, problems: list) -> None:
    if not isinstance(node, Mapping):
        return
    if "source" in node and "confidence" in node:
        _item(node, key, items, problems)
        return
    for k, v in node.items():
        _walk(v, f"{key}.{k}", items, problems)


def _item(node: Mapping, key: str, items: dict, problems: list,
          inherited: tuple[tuple[str, ...], str] | None = None) -> None:
    """One sourced item. Nested mappings inside it become child items that
    inherit its source and tier; text fields are kept as ``text``."""
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    if inherited is None:
        conf = str(node["confidence"]).upper()
        src = node["source"]
        sources = tuple(src) if isinstance(src, (list, tuple)) else (str(src),)
    else:
        sources, conf = inherited
        if "confidence" in node:
            conf = str(node["confidence"]).upper()
        if "source" in node:
            src = node["source"]
            sources = (tuple(src) if isinstance(src, (list, tuple))
                       else (str(src),))
    if conf not in CONFIDENCE_TOLERANCE:
        problems.append(f"{key}: unknown confidence {conf!r}")
        return
    numbers: dict[str, float] = {}
    text: dict[str, str] = {}
    for k, v in node.items():
        if k in ("source", "confidence", "note"):
            continue
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            if not math.isfinite(float(v)):
                problems.append(f"{key}.{k}: not finite")
            numbers[k] = float(v)
        elif isinstance(v, str):
            text[k] = v
        elif isinstance(v, Mapping):
            _item(v, f"{key}.{k}", items, problems, (sources, conf))
    lo, hi = numbers.get("low"), numbers.get("high")
    if lo is not None and hi is not None and lo > hi:
        problems.append(f"{key}: low {lo} > high {hi}")
    for name in _NUMERIC_ORDER:
        v = numbers.get(name)
        if v is not None and lo is not None and hi is not None                 and not lo <= v <= hi:
            problems.append(f"{key}: {name} {v} outside [{lo}, {hi}]")
    items[key] = CostItem(key=key, numbers=numbers, sources=sources,
                          confidence=conf,
                          note=None if node.get("note") is None
                          else str(node["note"]), text=text)
