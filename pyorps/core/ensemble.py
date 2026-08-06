"""
PYORPS: An Open-Source Tool for Automated Power Line Routing

Route ensembles: comparing route variants across objectives.

A :class:`RouteEnsemble` holds named (objective variant -> Path) results.
Because every path carries ALL metrics (Phase 6), the comparison table is
free, and a posteriori trade-off analysis (Pareto filtering) needs no
further routing work.

Honest caveat (plan section 9): a weight sweep traces the CONVEX part of
the Pareto front only — efficient routes in non-convex regions are
unreachable by weighted-sum scalarization.
"""
from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

from .exceptions import PyorpsError
from .path import Path


class EnsembleError(PyorpsError):
    """Exception raised for invalid route-ensemble operations."""


class RouteEnsemble:
    """Named route variants (one Path per objective), insertion-ordered."""

    def __init__(self):
        self._variants: dict[str, Path] = {}

    # ------------------------------------------------------------- content

    def add(self, name: str, path: Path) -> None:
        if not isinstance(name, str) or not name:
            raise EnsembleError(
                f"Variant name must be a non-empty string, got {name!r}")
        if name in self._variants:
            raise EnsembleError(f"Variant '{name}' already exists")
        self._variants[name] = path

    @property
    def names(self) -> list[str]:
        return list(self._variants)

    def __getitem__(self, name: str) -> Path:
        try:
            return self._variants[name]
        except KeyError:
            raise EnsembleError(
                f"No variant '{name}' in the ensemble. Available: "
                f"{list(self._variants)}") from None

    def __contains__(self, name: str) -> bool:
        return name in self._variants

    def __len__(self) -> int:
        return len(self._variants)

    def __iter__(self) -> Iterator[tuple[str, Path]]:
        return iter(self._variants.items())

    def __repr__(self) -> str:
        return f"RouteEnsemble(variants={list(self._variants)})"

    # ------------------------------------------------------------ analysis

    def _route_key(self, path: Path):
        return tuple(int(i) for i in path.path_indices)

    def to_dataframe(self) -> pd.DataFrame:
        """Comparison table: one row per variant, all metrics as columns.

        Adds ``same_route_as`` (first variant with the identical route,
        empty otherwise) and the routing runtime.
        """
        if not self._variants:
            return pd.DataFrame()

        metric_names: list[str] = []
        for path in self._variants.values():
            for name in (path.metrics or {}):
                if name not in metric_names:
                    metric_names.append(name)

        seen_routes: dict[tuple, str] = {}
        rows = []
        for name, path in self._variants.items():
            row: dict = {"feasibility": path.feasibility}
            for metric in metric_names:
                row[metric] = (path.metrics or {}).get(metric)
            row["length_2d"] = path.total_length_2d
            row["length_3d"] = path.total_length_3d
            row["max_gradient_pct"] = path.max_gradient_pct
            weights = ((path.objective_spec or {}).get("weights")
                       if path.objective_spec else None)
            row["weights"] = str(weights) if weights else ""
            row["runtime_s"] = (path.runtimes or {}).get("shortest_path")

            key = self._route_key(path)
            row["same_route_as"] = seen_routes.get(key, "")
            seen_routes.setdefault(key, name)
            rows.append(row)

        return pd.DataFrame(rows, index=list(self._variants))

    def pareto_front(self, metrics: list[str]) -> "RouteEnsemble":
        """Non-dominated variants w.r.t. the given metrics (all minimized).

        A variant is dropped when another one is at least as good in every
        listed metric and strictly better in at least one — or when it
        duplicates an earlier variant's metric values exactly.
        """
        if not metrics:
            raise EnsembleError("pareto_front needs at least one metric")

        values: dict[str, tuple] = {}
        for name, path in self._variants.items():
            missing = [m for m in metrics if m not in (path.metrics or {})]
            if missing:
                raise EnsembleError(
                    f"Variant '{name}' lacks metric(s) {missing} — "
                    f"available: {sorted(path.metrics or {})}")
            values[name] = tuple(float(path.metrics[m]) for m in metrics)

        def dominates(a: tuple, b: tuple) -> bool:
            return (all(x <= y for x, y in zip(a, b))
                    and any(x < y for x, y in zip(a, b)))

        front = RouteEnsemble()
        kept_values: list[tuple] = []
        names = list(values)
        for name in names:
            mine = values[name]
            if mine in kept_values:            # exact tie: keep first only
                continue
            if any(dominates(values[other], mine)
                   for other in names if other != name):
                continue
            front.add(name, self._variants[name])
            kept_values.append(mine)
        return front
