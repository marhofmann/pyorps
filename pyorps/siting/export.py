"""What PYORPS hands a MILP: a candidate table and a set of fields.

Section 5 of ``docs/superpowers/plans/2026-09-20-precomputed-tower-fields.md``.

PYORPS does routing and siting. It produces, for every candidate
position, the cost of connecting it -- by cable or by overhead line with
optimised tower placement -- to each fixed terminal. **Selecting among
those candidates is explicitly out of scope**: a third-party MILP does
that, under its own separation and topology constraints, and it reads
arrays rather than importing PYORPS.

That scope line is what removed the worst defects of the predecessor
plan. There is no chain fold here, no ``design:`` section and no pairing
of two free facilities, so the ``a = b`` collapse that made a min-plus
composition wrong for two or more free nodes cannot arise: nothing is
composed.

The contract::

    candidates:  id, x, y, buildable, footprint_cost_eur,
                 best_rotation_deg
    links:       per (terminal, link_type in {cable, overhead})
                   cost_lb   float64
                   cost_ub   float64
                   n_towers  uint16   (overhead only)
                   length_m  float32

Both ends of the cost are always present. A single number would force
the MILP to guess whether it may prune with it, and the plan's review
found that guess made wrongly in both directions; carrying the interval
costs one array and removes the question. :meth:`SitingExport.check`
verifies ``cost_lb <= cost_ub`` on every entry before anything is
written.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field as _field

import numpy as np

__all__ = ["LinkCosts", "SitingExport", "export_tower_fields"]

_LINK_TYPES = ("cable", "overhead")


@dataclass
class LinkCosts:
    """Per-candidate cost of connecting to ONE terminal by ONE link type."""

    terminal: str
    link_type: str
    cost_lb: np.ndarray
    cost_ub: np.ndarray
    n_towers: np.ndarray | None = None
    length_m: np.ndarray | None = None
    meta: dict = _field(default_factory=dict)

    def __post_init__(self):
        if self.link_type not in _LINK_TYPES:
            raise ValueError(
                f"link_type must be one of {_LINK_TYPES}, "
                f"got {self.link_type!r}")
        self.cost_lb = np.asarray(self.cost_lb, dtype=np.float64)
        self.cost_ub = np.asarray(self.cost_ub, dtype=np.float64)
        if self.cost_lb.shape != self.cost_ub.shape:
            raise ValueError("cost_lb and cost_ub must have the same shape")
        if self.link_type == "cable" and self.n_towers is not None:
            raise ValueError("n_towers is meaningless for a cable link")

    @property
    def key(self) -> str:
        return f"{self.terminal}::{self.link_type}"

    def violations(self, tol: float = 1e-6) -> int:
        """Entries where the lower bound exceeds the upper one."""
        both = np.isfinite(self.cost_lb) & np.isfinite(self.cost_ub)
        if not both.any():
            return 0
        return int((self.cost_lb[both] > self.cost_ub[both] + tol).sum())


class SitingExport:
    """A candidate table plus one :class:`LinkCosts` per terminal and type."""

    def __init__(self, candidates, *, crs=None, meta=None):
        self.candidates = candidates
        self.crs = crs if crs is not None else getattr(candidates, "crs",
                                                       None)
        self.links: dict[str, LinkCosts] = {}
        self.meta = dict(meta or {})

    # ----------------------------------------------------------- build

    def add_link(self, link: LinkCosts) -> SitingExport:
        """Add one terminal/link-type column block."""
        n = len(self.candidates)
        if link.cost_lb.shape != (n,):
            raise ValueError(
                f"{link.key}: expected {n} costs, got {link.cost_lb.shape}")
        if link.key in self.links:
            raise ValueError(f"{link.key} was added twice")
        self.links[link.key] = link
        return self

    def check(self, tol: float = 1e-6) -> dict:
        """``cost_lb <= cost_ub`` everywhere, and nothing reachable only above.

        Raises:
            AssertionError: on any violation, naming the block.
        """
        report = {}
        for key, link in self.links.items():
            bad = link.violations(tol)
            if bad:
                raise AssertionError(
                    f"{key}: cost_lb exceeds cost_ub on {bad} candidates -- "
                    f"one of the two fields is not the bound it claims")
            only_ub = ((~np.isfinite(link.cost_lb))
                       & np.isfinite(link.cost_ub))
            if only_ub.any():
                raise AssertionError(
                    f"{key}: {int(only_ub.sum())} candidates are reachable "
                    f"in the upper bound but not the lower one, which no "
                    f"relaxation can do")
            both = np.isfinite(link.cost_lb) & np.isfinite(link.cost_ub)
            gap = link.cost_ub[both] - link.cost_lb[both]
            report[key] = {
                "reachable": int(both.sum()),
                "gap_mean_eur": float(gap.mean()) if gap.size else float(
                    "nan"),
                "gap_max_eur": float(gap.max()) if gap.size else float("nan"),
            }
        return report

    # ----------------------------------------------------------- write

    def to_arrays(self) -> dict[str, np.ndarray]:
        """Everything as a flat ``{name: array}`` mapping."""
        c = self.candidates
        out: dict[str, np.ndarray] = {
            "candidate_id": c.ids,
            "x": np.asarray(c.xs, dtype=np.float64),
            "y": np.asarray(c.ys, dtype=np.float64),
            "buildable": np.isfinite(
                np.asarray(c.build_cost_eur, dtype=np.float64))
            if c.build_cost_eur is not None
            else np.ones(len(c), dtype=bool),
            "footprint_cost_eur": (
                np.asarray(c.build_cost_eur, dtype=np.float64)
                if c.build_cost_eur is not None
                else np.zeros(len(c), dtype=np.float64)),
            "best_rotation_deg": (
                np.asarray(c.best_theta_deg, dtype=np.float64)
                if c.best_theta_deg is not None
                else np.zeros(len(c), dtype=np.float64)),
        }
        if getattr(c, "area_id", None) is not None:
            out["area_id"] = np.asarray(c.area_id, dtype=np.int64)
        for key, link in self.links.items():
            out[f"{key}::cost_lb"] = link.cost_lb
            out[f"{key}::cost_ub"] = link.cost_ub
            if link.n_towers is not None:
                out[f"{key}::n_towers"] = np.asarray(
                    link.n_towers, dtype=np.uint16)
            if link.length_m is not None:
                out[f"{key}::length_m"] = np.asarray(
                    link.length_m, dtype=np.float32)
        return out

    def write_npz(self, path, *, compress: bool = True, check: bool = True):
        """Write the whole export as one ``.npz``.

        The MILP reads arrays; it does not need PYORPS in its process.
        """
        from pathlib import Path

        if check:
            self.meta["check"] = self.check()
        payload = self.to_arrays()
        payload["meta"] = np.array(json.dumps({
            "crs": str(self.crs) if self.crs is not None else None,
            "n_candidates": len(self.candidates),
            "links": sorted(self.links),
            "link_meta": {k: v.meta for k, v in self.links.items()},
            **self.meta,
        }))
        dest = Path(path)
        if dest.suffix != ".npz":
            dest = dest.with_suffix(".npz")
        dest.parent.mkdir(parents=True, exist_ok=True)
        writer = np.savez_compressed if compress else np.savez
        writer(dest, **payload)
        return dest

    def to_dataframe(self):
        """The same table as a pandas ``DataFrame`` (for Parquet or CSV)."""
        import pandas as pd
        arrays = self.to_arrays()
        return pd.DataFrame({k: v for k, v in arrays.items()
                             if np.ndim(v) == 1})

    def write_parquet(self, path, *, check: bool = True):
        """Write the candidate table as Parquet, metadata alongside as JSON."""
        from pathlib import Path
        if check:
            self.meta["check"] = self.check()
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(dest, index=False)
        dest.with_suffix(".meta.json").write_text(
            json.dumps({"crs": str(self.crs) if self.crs is not None
                        else None,
                        "links": sorted(self.links), **self.meta},
                       indent=2, default=str), encoding="utf-8")
        return dest

    def __repr__(self) -> str:
        return (f"SitingExport({len(self.candidates)} candidates, "
                f"{len(self.links)} link blocks)")


def export_tower_fields(candidates, fields, *, link_type: str = "overhead",
                        with_geometry: bool = False) -> SitingExport:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Build a :class:`SitingExport` from per-terminal bound pairs.

    Parameters:
        candidates: A :class:`~pyorps.siting.candidates.CandidateSet`.
        fields: ``{terminal_name: (lower_field, upper_field)}``, each a
            :class:`~pyorps.graph.tower_field.TowerField`. A single
            field in place of a pair is used for BOTH ends, which is
            correct only when it is exact -- say so in the metadata if
            you do it.
        link_type: ``"overhead"`` or ``"cable"``.
        with_geometry: Also reconstruct tower counts and line lengths
            from the upper (feasible) field, which needs it to have been
            solved with ``record_pred=True``.

    Returns:
        A :class:`SitingExport` with one block per terminal, already
        checked for ``lb <= ub``.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    exp = SitingExport(candidates)
    pts = candidates.points
    for name, pair in fields.items():
        lower, upper = pair if isinstance(pair, (tuple, list)) else (pair,
                                                                     pair)
        lb = lower.costs_to(pts)
        ub = upper.costs_to(pts)
        n_towers = length = None
        if with_geometry and upper.has_paths:
            rows, cols = upper.lattice.xy_to_cell(pts[:, 0], pts[:, 1])
            n_towers = np.zeros(len(candidates), dtype=np.uint16)
            length = np.zeros(len(candidates), dtype=np.float32)
            for i, (r, c) in enumerate(zip(rows, cols)):
                if not np.isfinite(ub[i]):
                    continue
                seq = upper.tower_sequence(int(r), int(c))
                n_towers[i] = len(seq)
                length[i] = sum(t.span_to_next_m for t in seq)
        exp.add_link(LinkCosts(
            terminal=str(name), link_type=link_type, cost_lb=lb, cost_ub=ub,
            n_towers=n_towers if link_type == "overhead" else None,
            length_m=length,
            meta={"lower_model": lower.model.describe(),
                  "upper_model": upper.model.describe(),
                  "lower_sweeps": lower.sweeps, "upper_sweeps": upper.sweeps,
                  # None rather than a collapsed scalar when the raster's
                  # two pixel sizes differ; see TowerLattice.sigma_m.
                  "sigma_m": (lower.lattice.sigma_x_m
                              if lower.lattice.is_square else None),
                  "sigma_x_m": lower.lattice.sigma_x_m,
                  "sigma_y_m": lower.lattice.sigma_y_m,
                  "n_directions": lower.lattice.n_directions}))
    exp.check()
    return exp
