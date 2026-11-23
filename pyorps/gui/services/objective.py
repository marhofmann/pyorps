"""
PYORPS GUI: objective weights, metric layers, a DEM and the weight precision for plain routing.

This is the logic behind the "Objective & terrain" card (gaps 1, 2 and 6 of the round-14 plan). The card
(``sections/objective.py`` and ``callbacks/objective.py``) only mirrors its controls into the settings dict
described below; everything that decides what is valid and what reaches the library lives here, so it is
unit-tested without a page. The module imports nothing from Dash, ``pyorps.gui.ids`` or the layout.

The settings dict (stored in ``state.gui_settings["objective"]``, saved in the project file)
------------------------------------------------------------------------------------------
All numbers are plain floats (or None where a blank is meaningful); the dict survives a JSON round trip::

    {"weights": {"cost": 1.0, "length": 0.0, "gradient": 0.0},
     "layers": [{"name": "landscape", "source": "<raster layer id>" | "derive:slope_from_dem",
                 "weight": 0.5, "hard_max": None, "hard_min": None}],
     "dem_layer_id": None,                       # id of a raster layer, or None
     "gradient": {"max_gradient_pct": None, "additive": None, "additive_scale": 2.0,
                  "multiplier": None, "mult_scale": None, "sig_center_deg": 15.0,
                  "sig_steepness": 0.2, "sig_max_penalty": 1000.0,
                  "bin_width_pct": 0.25, "s_max_pct": 200.0},
     "weight_precision": "uint16"}               # or "float32"

``normalise_objective_settings`` fills every missing key with its default and drops unknown keys; it never
judges a value. A blank (None or empty text) takes the default of its key; numeric text becomes a float; a
value that is not a number stays as it is, so that :func:`problems` can name it. ``{}`` and None give the
defaults, which are "classic cost routing": :func:`is_classic` is True, :func:`run_kwargs` returns ``{}`` and
the route is computed exactly as it was before this card existed.

What a setting means
--------------------
- ``weights``: the objective ``cost x raster value + length x metres + gradient x (slope response)``; a zero
  weight is dropped. ``gradient`` is the slope weight (needs a DEM).
- ``layers``: extra metric layers. ``source`` is a loaded raster layer (reprojected by the library onto the
  cost raster grid) or ``"derive:slope_from_dem"`` (per-cell slope in percent, needs the DEM). ``hard_max`` /
  ``hard_min`` forbid cells above / below the bound. A layer of weight 0 stays in the stack, so its bounds still
  forbid cells.
- ``dem_layer_id``: with a DEM every route length is the 3-D length. The GUI never passes a DEM without an
  explicit objective (its effect would differ per backend), so a DEM always switches the objective on.
- ``gradient``: the slope response (library ``GradientOptions``). Only the keys in
  :func:`applicable_gradient_keys` are read; the others are ignored and not checked.
- ``weight_precision``: ``"float32"`` keeps the combined surface lossless (graph libraries and GPU only).

Problems
--------
:func:`problems` returns ``list[ObjectiveProblem]``; ``ObjectiveProblem(key, label, message, value, allowed,
unit)`` is a NamedTuple. ``key`` names the setting (``"weights.cost"``, ``"layers.0.weight"``,
``"dem_layer_id"``, ``"gradient.max_gradient_pct"``, ``"weight_precision"``, ``"graph_api"``, ``"algorithm"``,
...: the full list is :data:`PROBLEM_KEYS`), ``label`` is the name the user sees and ``message`` finishes the
sentence "<label> <message>." (it states the value, the allowed range and the unit). The first problem of the
list stops a run.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable, Mapping
from typing import Any

# The data model (constants, specs, normalisation, accessors) lives in ``objective_model``; every public name
# is re-exported here, so ``pyorps.gui.services.objective`` stays the module the rest of the GUI imports.
from .objective_model import (  # noqa: F401
    _GRADIENT_CHOICES,
    _GRADIENT_OPTIONAL,
    ADDITIVE_SHAPES,
    DEM_FALLBACK_BACKENDS,
    DERIVE_SLOPE_LABEL,
    FIELD_SPECS,
    FLOAT_BACKENDS,
    GRADIENT_BACKENDS,
    LAYER_NAME_PATTERN,
    MULT_SCALE_DEFAULTS,
    MULTIPLIER_MODELS,
    OBJECTIVE_DEFAULTS,
    PRECISION_LABELS,
    PRECISIONS,
    PRESETS,
    PROBLEM_KEYS,
    RESERVED_LAYER_NAMES,
    SLOPE_NEEDS_DEM,
    SOURCE_DERIVE_SLOPE,
    WEIGHT_KEYS,
    FieldSpec,
    ObjectiveProblem,
    Support,
    _as_number,
    _cost_only,
    _is_zero,
    _number_problem,
    _positive,
    _shown,
    applicable_gradient_keys,
    apply_preset,
    check_layer_name,
    default_settings,
    describe_range,
    format_number,
    gradient_options,
    has_slope_terms,
    is_classic,
    label_with_unit,
    layer_row,
    mult_scale_default,
    needs_objective,
    normalise_objective_settings,
    preset_weights,
    source_label,
    weights,
)


# --------------------------------------------------------------------------- backend rules
def terrain_support(graph_api: str | None, algorithm: str | None) -> Support:
    """Whether a DEM can be used with this backend and algorithm, and why not.

    The GUI never relies on the library's fallbacks (raster_fim and delta-stepping fall back to Dijkstra with a
    warning): such a combination is refused, with a text that names the field to change.
    """
    issue = _terrain_backend_issue(graph_api or "cython", algorithm or "")
    return Support(True, "") if issue is None else Support(False, issue[3])


def _terrain_backend_issue(graph_api: str, algorithm: str) -> tuple[str, str, str, str] | None:
    """``(key, label, message, reason)`` for a DEM the backend or algorithm cannot take, else None."""
    if graph_api in DEM_FALLBACK_BACKENDS:
        reason = ("The raster_fim backend cannot route over a terrain model (DEM): its eikonal solution is " +
                  "continuous, not the discrete least-cost route. Choose cython, a graph library or raster_gpu " +
                  "in the Backend field.")
        return ("graph_api", "Backend", f"is {graph_api}, which cannot route over a terrain model (DEM): "
                "choose cython, a graph library or raster_gpu", reason)
    if graph_api not in GRADIENT_BACKENDS:
        reason = (f"The {graph_api} backend cannot apply the slope terms of a terrain model (DEM). Choose one of " +
                  f"{', '.join(GRADIENT_BACKENDS)} in the Backend field.")
        return ("graph_api", "Backend", f"is {graph_api}, which cannot apply the slope terms of a terrain "
                f"model (DEM); allowed: {', '.join(GRADIENT_BACKENDS)}", reason)
    if graph_api == "cython" and str(algorithm).startswith("delta"):
        reason = ("Delta-stepping has no slope terms: with a terrain model (DEM) the cython backend runs " +
                  "Dijkstra only. Choose Dijkstra in the Algorithm field.")
        return ("algorithm", "Algorithm", f"is {algorithm}, but delta-stepping has no slope terms: with a "
                "terrain model (DEM) the cython backend runs Dijkstra only; choose Dijkstra", reason)
    return None


def slope_support(settings: Any) -> Support:
    """Whether the slope weight and the slope response can apply (they need a DEM)."""
    s = normalise_objective_settings(settings)
    return Support(True) if s["dem_layer_id"] else Support(False, SLOPE_NEEDS_DEM)


def precision_support(settings: Any, graph_api: str | None) -> Support:
    """Whether the 32-bit choice is offered, and why not (the text of the disabled control).

    It applies only with an objective and on a backend that takes float weights. It is judged on the
    objective WITHOUT the precision itself, so choosing 32-bit never switches the choice off.
    """
    s = normalise_objective_settings(settings)
    api = graph_api or "cython"
    if _cost_only(s):
        return Support(False, "Only with an objective: classic cost routing always uses the 16-bit cost raster. "
                              "Add a length weight, an extra layer or a terrain model first.")
    if api not in FLOAT_BACKENDS:
        return Support(False, f"The {api} backend has 16-bit kernels. Use networkx, networkit, igraph, rustworkx "
                              "or raster_gpu for 32-bit weights.")
    if s["dem_layer_id"] and api in DEM_FALLBACK_BACKENDS:
        return Support(False, "The raster_fim backend cannot route over a terrain model (DEM).")
    return Support(True)


# --------------------------------------------------------------------------- problems
def _layer_label(index: int, row: Mapping[str, Any], what: str) -> str:
    name = row.get("name") or f"#{index + 1}"
    return f"Layer '{name}': {what}"


def _weight_problems(s: dict[str, Any]) -> list[ObjectiveProblem]:
    out = []
    has_dem = bool(s["dem_layer_id"])
    for key in WEIGHT_KEYS:
        if key == "gradient" and not has_dem:
            continue                       # the control is off without a DEM: the DEM rule says what is wrong
        problem = _number_problem(FIELD_SPECS[f"weights.{key}"], s["weights"][key])
        if problem is not None:
            out.append(problem)
    return out


def _layer_problems(s: dict[str, Any], raster_ids: set[str]) -> list[ObjectiveProblem]:
    # lizard forgives: one flat rule list per row; each rule is pinned by a test of its own
    out: list[ObjectiveProblem] = []
    seen: list[str] = []
    for i, row in enumerate(s["layers"]):
        prefix = f"layers.{i}"
        name_error = check_layer_name(row["name"], seen)
        if name_error is not None:
            out.append(ObjectiveProblem(f"{prefix}.name", f"Layer name #{i + 1}", name_error, row["name"]))
        seen.append(row["name"])
        source = row["source"]
        what = f"{prefix}.source"
        if source is None:
            out.append(ObjectiveProblem(
                what, _layer_label(i, row, "source"), "is empty: pick a raster layer or the slope from the DEM"))
        elif source == SOURCE_DERIVE_SLOPE:
            if not s["dem_layer_id"]:
                out.append(ObjectiveProblem(
                    what, _layer_label(i, row, "source"),
                    "is the slope from the DEM, which needs a terrain model (DEM): choose one first", source))
        elif source not in raster_ids:
            out.append(ObjectiveProblem(
                what, _layer_label(i, row, "source"),
                "points at a raster layer that is not loaded any more: pick another one", source))
        for field in ("weight", "hard_max", "hard_min"):
            spec = FIELD_SPECS[f"layers.{field}"]
            problem = _number_problem(spec, row[field], key=f"{prefix}.{field}",
                                      label=_layer_label(i, row, spec.label.lower()),
                                      optional=field != "weight")
            if problem is not None:
                out.append(problem)
        low, high = _as_number(row["hard_min"]), _as_number(row["hard_max"])
        if low is not None and high is not None and high < low:
            out.append(ObjectiveProblem(
                f"{prefix}.hard_max", _layer_label(i, row, "forbid above"),
                f"is {format_number(high)}, but must not be below 'forbid below' ({format_number(low)}): "
                "every cell would be forbidden", high))
    return out


def _slope_reasons(s: dict[str, Any]) -> list[str]:
    reasons = []
    if _positive(s["weights"]["gradient"]):
        reasons.append(f"the slope weight is {_shown(s['weights']['gradient'])}")
    if s["gradient"]["multiplier"] is not None:
        reasons.append(f"a slope multiplier ({s['gradient']['multiplier']}) is set")
    if s["gradient"]["max_gradient_pct"] is not None:
        reasons.append(f"a steepest allowed slope ({_shown(s['gradient']['max_gradient_pct'])} %) is set")
    return reasons


def _dem_crs_problems(label: str, raster_crs: Any, dem_crs: Any) -> list[ObjectiveProblem]:
    if raster_crs is None or dem_crs is None:
        return []
    if same_crs(raster_crs, dem_crs) is not False:
        return []
    return [ObjectiveProblem(
        "dem_layer_id", label,
        f"has the coordinate system {dem_crs}, but the cost raster uses {raster_crs}: the library does " +
        "not reproject a DEM file, so reproject the DEM to the project coordinate system first",
        str(dem_crs))]


def _dem_problems(s: dict[str, Any], raster_ids: set[str], raster_crs: Any, dem_crs: Any
                  ) -> list[ObjectiveProblem]:
    label = "Terrain model (DEM)"
    dem = s["dem_layer_id"]
    if not dem:
        reasons = _slope_reasons(s)
        if not reasons:
            return []
        return [ObjectiveProblem(
            "dem_layer_id", label,
            f"is not chosen, but the slope settings need one ({'; '.join(reasons)}): choose a terrain model or " +
            "set them back to 0 and blank")]
    if dem not in raster_ids:
        return [ObjectiveProblem("dem_layer_id", label,
                                 "points at a raster layer that is not loaded any more: pick another one", dem)]
    return _dem_crs_problems(label, raster_crs, dem_crs)


def _gradient_problems(s: dict[str, Any]) -> list[ObjectiveProblem]:
    # lizard forgives: one flat rule list; each rule is pinned by a test of its own
    gradient = s["gradient"]
    out: list[ObjectiveProblem] = []
    for key, choices, label in (("additive", ADDITIVE_SHAPES, "Slope shape"),
                                ("multiplier", MULTIPLIER_MODELS, "Slope penalty multiplier")):
        value = gradient[key]
        if value is not None and value not in choices:
            out.append(ObjectiveProblem(f"gradient.{key}", label,
                                        f"is {value!r}, but must be none or one of {', '.join(choices)}", value,
                                        ", ".join(choices)))
    for key in applicable_gradient_keys(gradient):
        if key in _GRADIENT_CHOICES:
            continue
        spec = FIELD_SPECS[f"gradient.{key}"]
        minimum = None
        if key == "s_max_pct":
            width = _as_number(gradient["bin_width_pct"])
            minimum = width if width is not None else FIELD_SPECS["gradient.bin_width_pct"].minimum
        problem = _number_problem(spec, gradient[key], optional=key in _GRADIENT_OPTIONAL, minimum=minimum)
        if problem is not None:
            out.append(problem)
    return out


def _precision_problems(s: dict[str, Any], graph_api: str) -> list[ObjectiveProblem]:
    label = "Weight precision"
    precision = s["weight_precision"]
    if precision not in PRECISIONS:
        return [ObjectiveProblem("weight_precision", label,
                                 f"is {precision!r}, but must be one of {', '.join(PRECISIONS)}", precision,
                                 ", ".join(PRECISIONS))]
    if precision == "float32":
        support = precision_support(s, graph_api)
        # a DEM on a backend that cannot take it is reported once, as a backend problem
        dem_on_fallback = bool(s["dem_layer_id"]) and graph_api in DEM_FALLBACK_BACKENDS
        if not support.enabled and not dem_on_fallback:
            return [ObjectiveProblem("weight_precision", label, f"cannot be 32-bit float here. {support.reason}",
                                     precision)]
    return []


def _library_problems(s: dict[str, Any]) -> list[ObjectiveProblem]:
    """What the library itself still rejects (the last safety net after every rule above passed)."""
    from pyorps.core.exceptions import ObjectiveError
    from pyorps.core.objective import GradientOptions, Objective

    try:
        options = GradientOptions.from_dict(gradient_options(s)) if s["dem_layer_id"] else None
        Objective(weights(s), options)
    except ObjectiveError as exc:
        text = str(exc)
        key = next((f"gradient.{k}" for k in ("max_gradient_pct", "bin_width_pct", "s_max_pct") if k in text),
                   "weights")
        return [ObjectiveProblem(key, "Objective", f"is not usable: {text}")]
    return []


def same_crs(first: Any, second: Any) -> bool | None:
    """True / False when two coordinate systems are / are not the same one, None when that cannot be told."""
    try:
        from pyproj import CRS

        return bool(CRS.from_user_input(first) == CRS.from_user_input(second))
    except Exception:  # pylint: disable=broad-exception-caught  # unknown or unparsable: do not judge
        return None


def _weights_rule_problems(s: dict[str, Any]) -> list[ObjectiveProblem]:
    found = _weight_problems(s)
    if not found and not any(_positive(s["weights"][k]) for k in WEIGHT_KEYS) and not any(
            _positive(row["weight"]) for row in s["layers"]):
        found.append(ObjectiveProblem(
            "weights", "Weights", "are all 0, but at least one must be above 0: an all-zero objective has no "
            "cheapest route"))
    return found


def _terrain_problems(s: dict[str, Any], api: str, algorithm: str | None) -> list[ObjectiveProblem]:
    if not s["dem_layer_id"]:
        return []
    found = _gradient_problems(s)
    issue = _terrain_backend_issue(api, algorithm or "")
    if issue is not None:
        found.append(ObjectiveProblem(issue[0], issue[1], issue[2], api if issue[0] == "graph_api" else algorithm))
    return found


def problems(settings: Any, *, graph_api: str | None = "cython", algorithm: str | None = None,
             raster_ids: Iterable[str] = (), raster_crs: Any = None, dem_crs: Any = None
             ) -> list[ObjectiveProblem]:
    """Every setting that cannot run, in the order of the card (an empty list: the run may start).

    ``graph_api`` / ``algorithm`` are the Backend and Algorithm fields of the page, ``raster_ids`` the ids of
    the raster layers that are loaded now, ``raster_crs`` / ``dem_crs`` the coordinate systems of the cost
    raster and of the DEM layer (both known: a DEM in another system is refused, because the library does not
    reproject it: a run with one stops with an error in the library). Classic settings have no problem.
    """
    s = normalise_objective_settings(settings)
    if is_classic(s):
        return []
    api = graph_api or "cython"
    loaded = set(raster_ids)
    found: list[ObjectiveProblem] = _weights_rule_problems(s)
    found += _layer_problems(s, loaded)
    found += _dem_problems(s, loaded, raster_crs, dem_crs)
    found += _terrain_problems(s, api, algorithm)
    found += _precision_problems(s, api)
    if not found:
        found += _library_problems(s)
    return found


def layer_problems(settings: Any, raster_ids: Iterable[str]) -> list[ObjectiveProblem]:
    """Only the problems of layers that are not loaded (what a route edit needs to rebuild its finder)."""
    s = normalise_objective_settings(settings)
    loaded = set(raster_ids)
    out = [p for p in _layer_problems(s, loaded) if p.key.endswith(".source")]
    out += [p for p in _dem_problems(s, loaded, None, None)]
    return out


def missing_files(settings: Any, layer_path: Callable[[str], str | None] | Mapping[str, str]
                  ) -> list[ObjectiveProblem]:
    """Problems for layers whose file is gone (a temporary file may have been cleaned up)."""
    s = normalise_objective_settings(settings)
    out: list[ObjectiveProblem] = []

    def check(layer_id: str, key: str, label: str) -> None:
        path = _resolve_path(layer_path, layer_id)
        if not path or not os.path.exists(str(path)):
            out.append(ObjectiveProblem(key, label, f"uses a raster file that no longer exists ({path or 'no path'}): "
                                        "load or rebuild that layer", layer_id))

    if s["dem_layer_id"]:
        check(s["dem_layer_id"], "dem_layer_id", "Terrain model (DEM)")
    for i, row in enumerate(s["layers"]):
        if row["source"] and row["source"] != SOURCE_DERIVE_SLOPE:
            check(row["source"], f"layers.{i}.source", _layer_label(i, row, "source"))
    return out


def _search_window(pts: list[tuple[float, float]], buffer_m: float,
                   raster_bounds: Iterable[float] | None) -> tuple[float, float, float, float]:
    """``(left, bottom, right, top)`` of the points' bounding box plus ``buffer_m``, cut to ``raster_bounds``."""
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    w_left, w_right = min(xs) - buffer_m, max(xs) + buffer_m
    w_bottom, w_top = min(ys) - buffer_m, max(ys) + buffer_m
    if raster_bounds is not None:
        r_left, r_bottom, r_right, r_top = (float(v) for v in raster_bounds)
        w_left, w_bottom = max(w_left, r_left), max(w_bottom, r_bottom)
        w_right, w_top = min(w_right, r_right), min(w_top, r_top)
    return w_left, w_bottom, w_right, w_top


def _window_leaves(window: tuple[float, float, float, float], bounds: tuple[float, float, float, float]) -> bool:
    """True when the search window is not wholly inside the DEM bounds."""
    w_left, w_bottom, w_right, w_top = window
    left, bottom, right, top = bounds
    return w_left < left or w_bottom < bottom or w_right > right or w_top > top


def _uncovered_problem(pts: list[tuple[float, float]],
                       bounds: tuple[float, float, float, float]) -> ObjectiveProblem | None:
    """The problem for routing points outside the DEM bounds (None when every point is covered)."""
    left, bottom, right, top = bounds
    outside = [p for p in pts if not (left <= p[0] <= right and bottom <= p[1] <= top)]
    if not outside:
        return None
    x, y = outside[0]
    return ObjectiveProblem(
        "dem_layer_id", "Terrain model (DEM)",
        f"does not cover {len(outside)} of {len(pts)} routing point(s), the first is ({x:,.1f}, {y:,.1f}): "
        "choose a DEM that covers every point", value=len(outside))


def dem_coverage(dem_bounds: Iterable[float], points: Iterable[Iterable[float]], *,
                 raster_bounds: Iterable[float] | None = None, buffer_m: float = 0.0
                 ) -> tuple[ObjectiveProblem | None, str | None]:
    """``(problem, warning)`` for how far the DEM reaches over the routing points.

    A DEM that does not contain every routing point is a problem (the library forbids the cells it does not
    cover, so such a route would start or end on a forbidden cell; a DEM that does not touch the search window
    at all stops the library with an error). A DEM that contains the points but not the whole search window
    (the points' bounding box plus ``buffer_m``, cut to ``raster_bounds``) is only a warning text: the run
    continues, but the route cannot leave the DEM. Bounds are ``(left, bottom, right, top)``.
    """
    left, bottom, right, top = (float(v) for v in dem_bounds)
    pts = [(float(p[0]), float(p[1])) for p in points]
    uncovered = _uncovered_problem(pts, (left, bottom, right, top))
    if uncovered is not None:
        return uncovered, None
    if not pts:
        return None, None
    window = _search_window(pts, buffer_m, raster_bounds)
    if _window_leaves(window, (left, bottom, right, top)):
        return None, ("The terrain model (DEM) covers the routing points but not the whole search window: cells " +
                      "outside it are forbidden, so the route cannot use them.")
    return None, None


# --------------------------------------------------------------------------- what reaches the library
def _resolve_path(layer_path: Callable[[str], str | None] | Mapping[str, str], layer_id: str) -> str | None:
    path = layer_path.get(layer_id) if isinstance(layer_path, Mapping) else layer_path(layer_id)
    return None if path is None else str(path)


def _metric_layers(s: dict[str, Any], layer_path: Callable[[str], str | None] | Mapping[str, str]
                   ) -> dict[str, dict[str, Any]]:
    layers: dict[str, dict[str, Any]] = {}
    for row in s["layers"]:
        if row["source"] == SOURCE_DERIVE_SLOPE:
            spec: dict[str, Any] = {"derive": "slope_from_dem"}
        else:
            path = _resolve_path(layer_path, row["source"] or "")
            if not path:
                raise ValueError(f"The raster layer of metric layer '{row['name']}' is not loaded.")
            spec = {"source": path}
        for key in ("hard_max", "hard_min"):
            number = _as_number(row[key])
            if number is not None:
                spec[key] = number
        layers[row["name"]] = spec
    return layers


def objective_spec(settings: Any) -> dict[str, Any]:
    """The settings as they are stored with a route (``params["objective_spec"]``; layer ids, no paths)."""
    return normalise_objective_settings(settings)


def objective_key(settings: Any) -> str:
    """Hash (sha256, hex) of what changes the route: weights, layers, DEM, slope response, precision.

    Settings that are not read (the exponential scale of a shape that is not exponential, the sigmoid
    parameters of another multiplier, ...) are left out, so they never change the key.
    """
    s = normalise_objective_settings(settings)
    dem = s["dem_layer_id"]
    canonical = {
        "weights": weights(s),
        "layers": [{"name": r["name"], "source": r["source"], "weight": _as_number(r["weight"]),
                    "hard_max": _as_number(r["hard_max"]), "hard_min": _as_number(r["hard_min"])}
                   for r in s["layers"]],
        "dem": dem,
        "gradient": gradient_options(s) if dem else None,
        "weight_precision": s["weight_precision"],
    }
    text = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def force_direct(settings: Any, graph_api: str | None = "cython") -> bool:
    """True when a route must run ``find_route`` itself instead of through a retained ``SearchSession``.

    - 32-bit weights: a session on the cython trees refuses a float32 finder, and one with a DEM refuses it on
      every backend.
    - extra metric layers (file layers and the slope derived from the DEM): a session on cython routes them like
      ``find_route`` does, so such a route is retained again and an edit is fast. The reason it was direct was
      a session that rebuilt its search window (a point moved out of it) and lost the metric layers; the library
      carries every finder setting over now. Proven by test (``tests/test_gui/test_gui_terrain_length_round16.py``,
      section B): for file layers and the derived slope, without and with a DEM, a session route has the cells
      of the direct route as a new route, after a window rebuild and after route edits. Only cython was run: a
      GPU or FIM session was not (no device), so on those backends the layers keep forcing a direct route.
    - a DEM on raster_gpu: a retained GPU field has no slope terms, the session would run the cython Dijkstra
      instead of the GPU.
    """
    s = normalise_objective_settings(settings)
    if is_classic(s):
        return False
    api = graph_api or "cython"
    if s["weight_precision"] == "float32":
        return True
    if s["dem_layer_id"] and api == "raster_gpu":
        return True
    return bool(s["layers"]) and api != "cython"


def run_kwargs(settings: Any, *, layer_path: Callable[[str], str | None] | Mapping[str, str],
               graph_api: str | None = "cython") -> dict[str, Any]:
    """What ``routing.run_routing`` takes for these settings: ``{}`` when classic, else the four keywords
    ``objective_kwargs`` (the ``PathFinder`` arguments ``objective``, ``weight_precision`` and, when set,
    ``dem``, ``gradient_options`` and ``metric_layers``), ``objective_spec``, ``objective_key``,
    ``force_direct``.

    ``layer_path`` maps a raster layer id to its file (a callable or a mapping). Call :func:`problems` first:
    an id that is not loaded raises ``ValueError`` here.
    """
    s = normalise_objective_settings(settings)
    if is_classic(s):
        return {}
    kwargs: dict[str, Any] = {"objective": weights(s), "weight_precision": s["weight_precision"]}
    dem = s["dem_layer_id"]
    if dem:
        path = _resolve_path(layer_path, dem)
        if not path:
            raise ValueError("The terrain model (DEM) layer is not loaded.")
        kwargs["dem"] = path
        kwargs["gradient_options"] = gradient_options(s)
    layers = _metric_layers(s, layer_path)
    if layers:
        kwargs["metric_layers"] = layers
    return {"objective_kwargs": kwargs, "objective_spec": objective_spec(s), "objective_key": objective_key(s),
            "force_direct": force_direct(s, graph_api)}


def stack_bands(settings: Any) -> int:
    """How many full-raster float32 bands the objective pipeline holds beyond the plain cost raster (rough).

    The metric stack covers the WHOLE raster, not the search window: extra layers, the derived slope (the DEM
    aligned to the raster plus the slope band) and the combine each cost 4 bytes per cell. 0 for a run that
    only weights the cost (zero-copy pass-through). An estimate for the heavy-run prognosis, not a measurement.
    """
    s = normalise_objective_settings(settings)
    if is_classic(s):
        return 0
    derived = any(row["source"] == SOURCE_DERIVE_SLOPE for row in s["layers"])
    weights_cost_only = (_positive(s["weights"]["cost"]) and _is_zero(s["weights"]["length"])
                         and _is_zero(s["weights"]["gradient"]))
    if weights_cost_only and not s["layers"]:
        return 0
    return 1 + len(s["layers"]) + (2 if derived else 0)


def _slope_extras(s: dict[str, Any]) -> list[str]:
    extras = []
    gradient = s["gradient"]
    limit = _as_number(gradient["max_gradient_pct"])
    if limit is not None:
        extras.append(f"slope limit {format_number(limit)} %")
    if gradient["additive"] and _positive(s["weights"]["gradient"]):
        extras.append(f"{gradient['additive']} slope shape")
    if gradient["multiplier"]:
        extras.append(f"{gradient['multiplier']} slope multiplier")
    return extras


def _summary_extras(s: dict[str, Any], graph_api: str | None, names: Mapping[str, str]) -> list[str]:
    extras = []
    dem = s["dem_layer_id"]
    if dem:
        extras.append(f"DEM '{names.get(dem, dem)}'")
        extras.extend(_slope_extras(s))
    if s["weight_precision"] == "float32":
        extras.append("32-bit float weights")
    if dem and (graph_api or "cython") == "cython":
        extras.append("cython runs Dijkstra only")
    return extras


def summary(settings: Any, graph_api: str | None = None, algorithm: str | None = None, *,
            layer_names: Mapping[str, str] | None = None) -> str:
    """One line for the card's status and the success notice, e.g. ``"Objective: cost 1 + length 0.2 +
    landscape 0.5; DEM 'DGM'; slope limit 30 %; cython runs Dijkstra only."``."""
    s = normalise_objective_settings(settings)
    if is_classic(s):
        return "Classic cost routing (no objective)."
    names = layer_names or {}
    labels = {"cost": "cost", "length": "length", "gradient": "slope"}
    parts = [f"{labels.get(name, name)} {format_number(value)}" for name, value in weights(s).items()]
    text = "Objective: " + (" + ".join(parts) if parts else "no positive weight")
    extras = _summary_extras(s, graph_api, names)
    return text + "".join(f"; {e}" for e in extras) + "."
