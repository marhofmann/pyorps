"""
PYORPS GUI: cost-model editing — feature detection, grid <-> dict mapping,
seeding, and table import/export (R3, R5).

Internal representation ("unwrapped"): the GUI holds ``feature_keys`` (an
ordered tuple of column names, 1 or 2 entries) and ``assumptions``:

- single key:      ``{value: cost}``
- combination key: ``{main_value: {side_value: cost, "": default}}``

pyorps' ``CostAssumptions`` wants the column names wrapped around that dict
(:func:`wrap_assumptions`). ``65535`` is the forbidden sentinel, surfaced in
the grid as a first-class boolean column (F11), and every category gets a
``""`` catch-all row so unmapped values can't silently become forbidden (F4).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..presets import FORBIDDEN, default_land_use_costs

NOTEBOOK_KEYS = ("nutzart", "bez")

#: operators for the conditional modifier editor (Feature 5). Equality-family
#: (==/!=/in/is-empty) compare as trimmed strings so categorical columns work;
#: the ordered operators (</<=/>/>=) coerce the column to numeric. "all" (or a
#: blank operator) means the whole modifier dataset (no condition).
MODIFIER_OPERATORS = ["all", "==", "!=", "<", "<=", ">", ">=", "in", "is-empty"]


# ------------------------------------------------------------ feature columns
def propose_features(gdf, max_features_per_column: int = 100):
    """Wrap pyorps' detector: returns (proposed_keys, candidate_columns)."""
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    from pyorps import detect_feature_columns

    candidates = [c for c in gdf.columns
                  if c != gdf.geometry.name
                  and gdf[c].dtype == object
                  and gdf[c].nunique() <= max_features_per_column]
    try:
        main, side = detect_feature_columns(
            gdf, max_features_per_column=max_features_per_column)
    except Exception:
        main, side = (candidates[0] if candidates else None), []
    proposed: tuple[str, ...] = ()
    if main:
        proposed = (main,)
        if side:
            proposed = (main, side[0])
    # the notebook combination is preferred when its columns exist
    if all(k in gdf.columns for k in NOTEBOOK_KEYS):
        proposed = NOTEBOOK_KEYS
    for key in proposed:
        if key not in candidates:
            candidates.append(key)
    return proposed, candidates


def feature_analysis(gdf, feature_keys: tuple[str, ...] | list[str]):
    """Category counts per feature + the size of the combination.

    Mirrors the notebook's feature exploration: the user should see how many
    categories each column has and how many rows a (main, side) combination
    yields before seeding the cost table.
    Returns ``(per_column, n_combination_rows | None)`` where per_column is
    ``[(column, n_categories), ...]``.
    """
    per_column = []
    for key in feature_keys or ():
        if key in gdf.columns:
            per_column.append((key, int(gdf[key].nunique(dropna=True))))
    n_combo = None
    keys = [k for k in (feature_keys or ())[:2] if k in gdf.columns]
    if len(keys) == 2:
        pairs = gdf[keys].fillna("").drop_duplicates()
        n_combo = int(len(pairs))
    return per_column, n_combo


# -------------------------------------------------------------------- seeding
def seed_assumptions(gdf, feature_keys: tuple[str, ...]) -> dict:
    """Default cost dict: notebook values for ('nutzart','bez'), else zeros.

    The zero template enumerates the dataset's unique values and always adds
    the ``""`` catch-all per category (F4).
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    feature_keys = tuple(feature_keys)
    if feature_keys == NOTEBOOK_KEYS:
        return default_land_use_costs()[NOTEBOOK_KEYS]
    main = feature_keys[0]
    if len(feature_keys) == 1:
        import pandas as pd

        # a numeric column (e.g. a drawn cost layer's ``cost``, F5) seeds to
        # itself — the value already *is* the cost — instead of zero.
        numeric = pd.api.types.is_numeric_dtype(gdf[main])
        uniques = gdf[main].dropna().unique()
        values = sorted(str(v) for v in uniques)
        seeded: dict = {str(v): (_coerce_cost(v) if numeric else 0)
                        for v in uniques} if numeric else {v: 0
                                                           for v in values}
        seeded[""] = 0
        return seeded
    side = feature_keys[1]
    seeded = {}
    for main_val, group in gdf.dropna(subset=[main]).groupby(main):
        side_vals = sorted(str(v) for v in group[side].fillna("").unique())
        seeded[str(main_val)] = {v: 0 for v in side_vals}
        seeded[str(main_val)][""] = 0
    return seeded


# --------------------------------------------------------------- grid mapping
def _coerce_cost(value: Any) -> float | int:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0
    if isinstance(value, str):
        value = value.strip().replace(",", ".")
        value = float(value) if value else 0.0
    value = float(value)
    return int(value) if value.is_integer() else value


def grid_rows_from_assumptions(assumptions: dict,
                               feature_keys: tuple[str, ...]) -> list[dict]:
    """Nested cost dict -> flat AG-Grid rows (with the forbidden flag)."""
    rows: list[dict] = []
    main = feature_keys[0]
    if len(feature_keys) == 1:
        for value, cost in assumptions.items():
            cost = _coerce_cost(cost)
            rows.append({main: value, "cost": cost,
                         "forbidden": cost >= FORBIDDEN})
        return rows
    side = feature_keys[1]
    for main_val, sub in assumptions.items():
        if not isinstance(sub, dict):
            sub = {"": sub}
        for side_val, cost in sub.items():
            cost = _coerce_cost(cost)
            rows.append({main: main_val, side: side_val, "cost": cost,
                         "forbidden": cost >= FORBIDDEN})
    return rows


def assumptions_from_grid_rows(rows: list[dict],
                               feature_keys: tuple[str, ...]) -> dict:
    """Flat grid rows -> nested cost dict; forbidden flag wins over cost."""
    main = feature_keys[0]
    result: dict = {}
    for row in rows or []:
        cost = FORBIDDEN if row.get("forbidden") else _coerce_cost(
            row.get("cost"))
        main_val = "" if row.get(main) is None else str(row.get(main))
        if len(feature_keys) == 1:
            result[main_val] = cost
        else:
            side = feature_keys[1]
            side_val = "" if row.get(side) is None else str(row.get(side))
            result.setdefault(main_val, {})[side_val] = cost
    return result


def grid_column_defs(feature_keys: tuple[str, ...]) -> list[dict]:
    """AG-Grid columnDefs for the cost table (forbidden rows in red)."""
    defs = [{"field": key, "headerName": key, "editable": True}
            for key in feature_keys]
    defs.append({
        "field": "cost", "headerName": "cost (€/m)", "editable": True,
        "type": "numericColumn",
        "cellClassRules": {"text-danger fw-bold":
                           f"params.value >= {FORBIDDEN}"},
    })
    defs.append({
        "field": "forbidden", "headerName": "forbidden", "editable": True,
        "cellRenderer": "agCheckboxCellRenderer",
        "cellEditor": "agCheckboxCellEditor", "maxWidth": 110,
    })
    return defs


# ---------------------------------------------------------------- pyorps glue
def wrap_assumptions(assumptions: dict,
                     feature_keys: tuple[str, ...]) -> dict:
    """Unwrapped GUI dict -> the pyorps CostAssumptions source format."""
    feature_keys = tuple(feature_keys)
    if len(feature_keys) == 1:
        return {feature_keys[0]: assumptions}
    return {feature_keys: assumptions}


# ------------------------------------------------------------- import/export
def export_table(assumptions: dict, feature_keys: tuple[str, ...],
                 path: str | Path) -> str:
    """Write the cost table as CSV/JSON/XLSX via pyorps' CostAssumptions."""
    from pyorps import CostAssumptions

    ca = CostAssumptions(wrap_assumptions(assumptions, feature_keys))
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        ca.to_csv(str(path))
    elif suffix == ".json":
        ca.to_json(str(path))
    elif suffix in (".xlsx", ".xls"):
        ca.to_excel(str(path))
    else:
        raise ValueError(f"Unsupported cost-table format '{suffix}' — use "
                         ".csv, .json or .xlsx.")
    return str(path)


def import_table(path: str | Path) -> tuple[dict, tuple[str, ...]]:
    """Read a cost table file -> (unwrapped assumptions, feature_keys).

    pyorps loads CSV/XLSX into a flat dict with tuple keys
    (``{("Wald", "Nadelholz"): 405}``) and JSON into the nested form
    (``{"Wald": {"Nadelholz": 405}}``); column names come from the
    ``main_feature`` / ``side_features`` attributes. Both are normalized to
    the GUI's nested representation.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    from pyorps import CostAssumptions

    ca = CostAssumptions(str(path))
    source = ca.cost_assumptions
    if not isinstance(source, dict) or not source:
        raise ValueError(f"'{path}' contains no cost assumptions.")
    main = getattr(ca, "main_feature", None) or "feature"
    side = list(getattr(ca, "side_features", None) or [])[:1]

    first_key = next(iter(source))
    if isinstance(first_key, tuple):
        if len(first_key) >= 2:                       # CSV/XLSX combination
            nested: dict = {}
            for tup, cost in source.items():
                nested.setdefault(str(tup[0]), {})[str(tup[1])] = \
                    _coerce_cost(cost)
            return nested, (main, side[0] if side else "sub")
        return ({str(k[0]): _coerce_cost(v) for k, v in source.items()},
                (main,))                               # CSV single column
    if side and any(isinstance(v, dict) for v in source.values()):
        return source, (main, side[0])                 # JSON combination
    return source, (main,)                             # JSON single column


# ------------------------------------------------------------------ modifiers
def coerce_factor(value: Any) -> float:
    """The modifier factor/cost cell -> float (e.g. 1.25 or 65535 to forbid)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    try:
        return float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Modifier factor '{value}' is not a number — enter a factor "
            "(e.g. 1.25 to multiply) or a cost (e.g. 65535 to forbid).") from exc


def condition_label(column: str | None, operator: str | None,
                    value: Any) -> str:
    """Human-readable label for a modifier condition (logs / meta)."""
    operator = (operator or "all").strip()
    if operator in ("", "all"):
        return "all features"
    if operator == "is-empty":
        return f"{column} is empty"
    return f"{column} {operator} {value}"


def condition_mask(gdf, column: str | None, operator: str | None, value: Any):
    """Boolean row mask for a ``(column, operator, value)`` rule (or None).

    Returns ``None`` for the "all" (or blank) operator — the caller then means
    "the whole dataset" and can skip masking. pyorps only tests equality
    natively, so the GUI resolves every operator here (Section 21 / F5 pattern:
    shift work into the GUI where pyorps has no primitive). Shared by the
    conditional modifier editor (F5) and the preprocessing-step builder (F4).
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    import pandas as pd

    operator = (operator or "all").strip()
    if operator in ("", "all"):
        return None
    if not column or column not in gdf.columns:
        raise ValueError(
            f"Condition column '{column}' is not an attribute of the "
            "dataset — pick one of its columns.")
    col = gdf[column]
    if operator == "is-empty":
        return col.isna() | (col.astype(str).str.strip() == "")
    if operator == "in":
        wanted = {v.strip() for v in str(value or "").split(",") if v.strip()}
        return col.astype(str).str.strip().isin(wanted)
    if operator in ("==", "!="):
        target = str("" if value is None else value).strip()
        eq = col.astype(str).str.strip() == target
        return eq if operator == "==" else ~eq
    if operator in ("<", "<=", ">", ">="):
        num = pd.to_numeric(col, errors="coerce")
        try:
            threshold = float(str(value).replace(",", "."))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Operator '{operator}' needs a numeric value; got "
                f"'{value}'.") from exc
        cmp = {"<": num < threshold, "<=": num <= threshold,
               ">": num > threshold, ">=": num >= threshold}[operator]
        return cmp.fillna(False)
    raise ValueError(f"Unknown condition operator '{operator}'.")


def apply_condition(gdf, column: str | None, operator: str | None, value: Any):
    """Filter a modifier GeoDataFrame by a ``(column, operator, value)`` rule.

    ``operator`` "all" (or blank) returns the whole dataset. Thin wrapper over
    :func:`condition_mask` handing the already-filtered subset to
    ``modify_raster_from_dataset``.
    """
    mask = condition_mask(gdf, column, operator, value)
    return gdf if mask is None else gdf[mask]


def parse_modifier_values(text: Any) -> float | dict:
    """The modifier "Value(s)" cell: a scalar number or a JSON mapping."""
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        return float(text)
    if isinstance(text, dict):
        return {k: float(v) for k, v in text.items()}
    text = str(text or "").strip()
    if not text:
        raise ValueError("Modifier value is empty — enter a factor "
                         "(e.g. 1.25) or a JSON mapping.")
    try:
        return float(text.replace(",", "."))
    except ValueError:
        pass
    try:
        mapping = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Modifier value '{text}' is neither a number nor valid JSON "
            "(e.g. {\"1\": 100, \"2\": 2}).") from exc
    if not isinstance(mapping, dict):
        raise ValueError("A JSON modifier value must be an object "
                         "(zone -> factor).")
    return {k: float(v) for k, v in mapping.items()}
