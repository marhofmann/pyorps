"""Construction access road to the substation pad: constraints and the array functions that turn them
into a 1 m cost raster (plan 2026-10-06-paper-run-pipeline.md, section 8 step 1).

The constraints live in a YAML file whose every value is tagged ``sourced``, ``decided``, ``catalogue``,
``derived`` or ``assumption`` (as ``collector_inputs_2026.yaml``). Catalogue prices are read at build time,
so a cost equals the catalogue item. A certified load refuses while an assumption remains.

Raster convention: uint16, EUR per metre of road centre line, ``FORBIDDEN`` = 65535. Cells of existing
public roads carry the cost ``public_road_eur_per_m`` (the smallest of all costs) and are the entry cells
of the road network: they are exempt from the slope and clearance rules because the road already exists.
Limits of the model (stated in the YAML meta): the curve radius of a transport vehicle is not modelled;
the slope rule is a cell rule (rise over run across the cell's neighbourhood), not a rule along the route.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["FORBIDDEN", "AccessConstraints", "band_multiplier", "block_cost", "class_cost", "clearance",
           "slope_multiplier", "slope_percent"]

FORBIDDEN = 65535
STATUSES = ("sourced", "decided", "catalogue", "derived", "assumption")
RULE_NAMES = ("public_road", "track_upgrade", "open_new", "forest_new", "forbidden")
_CAP = FORBIDDEN - 1


def _walk_status(node, prefix: str, out: dict[str, str]) -> None:
    if not isinstance(node, Mapping):
        return
    if isinstance(node.get("status"), str):
        out[prefix.rstrip(".")] = node["status"]
    for key, val in node.items():
        if isinstance(val, Mapping):
            _walk_status(val, f"{prefix}{key}.", out)


def _ceil(x: float) -> int:
    return int(math.ceil(x - 1e-9))


@dataclass
class AccessConstraints:
    """The access-road constraints, resolved.

    Attributes:
        path, sha256: The YAML file and its hash.
        width_m: Clear road width; a route cell needs a free disc of this diameter.
        new_eur_m, simple_eur_m, forest_mult, upgrade_eur_m, public_eur_m: Costs in EUR per metre.
        g_max_pct: Cells at least this steep are forbidden.
        half_window: Slope baseline, cells on each side.
        bands: ``(from_pct, to_pct, multiplier)`` for cut and fill, contiguous from 0 to ``g_max_pct``.
        rules: ``{(nutzart, bez): rule}`` with ``bez == "*"`` for every sub-class.
        statuses: ``{dotted path: status}``.
    """
    path: Path
    sha256: str
    width_m: float
    new_eur_m: float
    simple_eur_m: float
    forest_mult: float
    upgrade_eur_m: float
    public_eur_m: int
    g_max_pct: float
    cross_max_pct: float
    half_window: int
    bands: tuple[tuple[float, float, float], ...]
    rules: dict[tuple[str, str], str]
    statuses: dict[str, str]
    catalogue_sha256: str

    @classmethod
    def load(cls, path, catalogue, *, certified: bool = False) -> AccessConstraints:
        """Read the file and the catalogue prices; refuse a bad status, a bad rule or band, and (when
        ``certified``) any remaining assumption."""
        import yaml

        raw = Path(path).read_bytes()
        d = yaml.safe_load(raw.decode("utf-8"))
        st: dict[str, str] = {}
        _walk_status(d, "", st)
        bad = [f"{k}: {s!r}" for k, s in st.items() if s not in STATUSES]
        if bad:
            raise ValueError(f"{path}: invalid status in " + "; ".join(bad))
        assumed = sorted(k for k, s in st.items() if s == "assumption")
        if certified and assumed:
            raise ValueError("a certified build needs sourced inputs; assumptions remain: " + ", ".join(assumed))

        def val(node):
            return catalogue.value(node["catalogue_key"]) if "catalogue_key" in node else node["value"]

        r, t = d["road"], d["terrain"]
        new, simple = float(val(r["new_eur_per_m"])), float(val(r["simple_track_eur_per_m"]))
        public = int(val(r["public_road_eur_per_m"]))
        bands = tuple((float(a), float(b), float(m)) for a, b, m in val(t["slope_bands"]))
        g_max = float(val(t["g_max_percent"]))
        edges_ok = bands and bands[0][0] == 0.0 and bands[-1][1] == g_max and all(
            bands[i][1] == bands[i + 1][0] for i in range(len(bands) - 1))
        if not edges_ok:
            raise ValueError("slope_bands must be contiguous from 0 to g_max_percent")
        rules: dict[tuple[str, str], str] = {}
        for rule, pairs in d["classes"].items():
            if rule == "status":
                continue
            if rule not in RULE_NAMES:
                raise ValueError(f"unknown class rule {rule!r}")
            for nutzart, bez in pairs:
                if (nutzart, bez) in rules:
                    raise ValueError(f"class {(nutzart, bez)} is listed twice")
                rules[(nutzart, bez)] = rule
        return cls(path=Path(path), sha256=hashlib.sha256(raw).hexdigest(),
                   width_m=float(val(r["width_m"])), new_eur_m=new, simple_eur_m=simple,
                   forest_mult=float(val(r["forest_multiplier"])), upgrade_eur_m=new - simple,
                   public_eur_m=public, g_max_pct=g_max, cross_max_pct=float(val(t["cross_slope_max_percent"])),
                   half_window=int(val(t["slope_half_window_cells"])),
                   bands=bands, rules=rules, statuses=st, catalogue_sha256=getattr(catalogue, "sha256", ""))

    def assumptions(self) -> list[str]:
        return sorted(k for k, s in self.statuses.items() if s == "assumption")

    def radius_cells(self, cell: float = 1.0) -> int:
        """Radius of the clearance disc in cells."""
        return max(0, math.ceil(self.width_m / 2.0 / cell - 1e-9))

    def rule_of(self, nutzart: str, bez: str) -> str:
        """Road case of one land-use sub-class: an exact pair, else the class's ``*``, else forbidden."""
        return self.rules.get((nutzart, bez)) or self.rules.get((nutzart, "*")) or "forbidden"

    def rule_cost(self, rule: str) -> int:
        """Flat cost of a rule before the slope multiplier."""
        return {"public_road": self.public_eur_m, "track_upgrade": _ceil(self.upgrade_eur_m),
                "open_new": _ceil(self.new_eur_m), "forest_new": _ceil(self.new_eur_m * self.forest_mult),
                "forbidden": FORBIDDEN}[rule]

    def class_table(self, recipe: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, int]]:
        """``{nutzart: {bez: cost}}`` over the classes of the cable recipe (same keys, so the rasterizer
        sees the same class universe). The entry cost must be below every other cost."""
        table = {n: {b: self.rule_cost(self.rule_of(n, b)) for b in sub} for n, sub in recipe.items()}
        others = [c for sub in table.values() for c in sub.values() if c not in (self.public_eur_m, FORBIDDEN)]
        if others and min(others) <= self.public_eur_m:
            raise ValueError("public_road_eur_per_m must be below every other class cost")
        return table

    def gradient_options(self):
        """The kernel's per-step grade response: steps steeper than ``g_max_pct`` forbidden, the cost of
        the others times the band multiplier (the 3D length stretch is added by the objective itself)."""
        from pyorps.core.objective import GradientOptions

        return GradientOptions(multiplier=band_multiplier(self.bands), max_gradient_pct=self.g_max_pct)

    def record(self) -> dict:
        """JSON-able provenance."""
        return {"file": self.path.name, "sha256": self.sha256, "catalogue_sha256": self.catalogue_sha256,
                "width_m": self.width_m, "new_eur_per_m": self.new_eur_m, "upgrade_eur_per_m": self.upgrade_eur_m,
                "forest_multiplier": self.forest_mult, "public_road_eur_per_m": self.public_eur_m,
                "g_max_percent": self.g_max_pct, "cross_slope_max_percent": self.cross_max_pct,
                "slope_half_window_cells": self.half_window,
                "slope_bands": [list(b) for b in self.bands], "assumptions": self.assumptions()}


def class_cost(table: Mapping[str, Mapping[str, int]], nutzart, bez) -> int:
    """Cost of one land-use row from a class table; an unlisted class is forbidden, an unlisted
    sub-class takes the class default (the rasterizer's rule)."""
    sub = table.get(nutzart)
    if sub is None:
        return FORBIDDEN
    key = bez if isinstance(bez, str) and bez in sub else ""
    return sub.get(key, sub.get("", FORBIDDEN))


def slope_percent(z: np.ndarray, half: int = 2, cell: float = 1.0) -> np.ndarray:
    """Grade in percent, rise over run across ``half`` cells on each side, edges replicated."""
    zp = np.pad(np.asarray(z, dtype=np.float32), half, mode="edge")
    run = 2.0 * half * cell
    dx = (zp[half:-half, 2 * half:] - zp[half:-half, :-2 * half]) / run
    dy = (zp[2 * half:, half:-half] - zp[:-2 * half, half:-half]) / run
    return (100.0 * np.hypot(dx, dy)).astype(np.float32)


def slope_multiplier(slope: np.ndarray, bands) -> np.ndarray:
    """Cut-and-fill multiplier by grade band; 1 outside the bands."""
    mult = np.ones(np.shape(slope), dtype=np.float64)
    slope = np.asarray(slope)
    for lo, hi, m in bands:
        mult[(slope >= lo) & (slope < hi)] = m
    return mult


def band_multiplier(bands):
    """The band multiplier as a callable ``f(grade_pct) -> multiplier`` for ``GradientOptions``."""
    def response(s_pct, params=None):  # noqa: ARG001
        return slope_multiplier(np.asarray(s_pct, dtype=np.float64), bands)
    response.__qualname__ = "access_band_multiplier"
    return response


def clearance(passable: np.ndarray, radius_cells: int) -> np.ndarray:
    """Cells whose disc of ``radius_cells`` lies wholly in ``passable`` (outside the array counts as
    passable, so an array edge does not close cells)."""
    if radius_cells <= 0:
        return passable.copy()
    from scipy import ndimage

    r = radius_cells
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    return ndimage.binary_erosion(passable, structure=(yy * yy + xx * xx) <= r * r, border_value=1)


def block_cost(cls: np.ndarray, z: np.ndarray, c: AccessConstraints, *, top: int = 0, bottom: int = 0,
               cell: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """The access cost of a block of rows.

    Args:
        cls: uint16 class costs (``class_table``), with ``top`` and ``bottom`` halo rows.
        z: DEM of the same shape. Halo rows must be at least ``half_window + radius_cells`` deep except
            at the raster edge.
        c: The constraints.
        top, bottom: Halo rows to cut from the result.

    Returns:
        ``(cost uint16, entry bool)`` without the halo. Entry cells (existing public roads) are never
        forbidden by slope or clearance; every other cell keeps its class cost or is forbidden (class,
        ground slope at least ``cross_max_pct``, or no clear disc of the road width). The grade ALONG the
        route is not here: it is the kernel's per-step table (``AccessConstraints.gradient_options``).
    """
    entry = cls == c.public_eur_m
    slope = slope_percent(z, c.half_window, cell)
    hard = (cls == FORBIDDEN) | ((slope >= c.cross_max_pct) & ~entry)
    ok = clearance(~hard, c.radius_cells(cell))
    forbidden = (hard | ~ok) & ~entry
    cost = np.minimum(cls, _CAP).astype(np.uint16)
    cost[forbidden] = FORBIDDEN
    rows = slice(top, cls.shape[0] - bottom)
    return cost[rows], entry[rows]
