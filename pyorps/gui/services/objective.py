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

import copy
import hashlib
import json
import math
import os
import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any, NamedTuple

# --------------------------------------------------------------------------- constants
#: the three weights the card shows besides the layers, in display order
WEIGHT_KEYS = ("cost", "length", "gradient")
#: ``source`` of a layer row that is derived from the DEM instead of read from a raster layer
SOURCE_DERIVE_SLOPE = "derive:slope_from_dem"
DERIVE_SLOPE_LABEL = "Slope from the DEM (per cell, %)"
#: names a metric layer may not take (the library's reserved objective names and the cost layer)
RESERVED_LAYER_NAMES = ("cost", "length", "gradient", "smoothness", "tower")
LAYER_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")

#: Backends that apply every slope term themselves (library ``PathFinder.GRADIENT_BACKENDS``; a test compares)
GRADIENT_BACKENDS = ("cython", "networkit", "networkx", "igraph", "rustworkx", "raster_gpu")
#: Backends that take lossless float32 weights (library ``PathFinder.FLOAT_BACKENDS``; a test compares)
FLOAT_BACKENDS = ("networkit", "networkx", "igraph", "rustworkx", "raster_gpu", "raster_fim")
#: Backends that hand a DEM request to the cython Dijkstra (library ``PathFinder.DEM_FALLBACK_BACKENDS``)
DEM_FALLBACK_BACKENDS = ("raster_fim",)

ADDITIVE_SHAPES = ("linear", "quadratic", "exponential")
MULTIPLIER_MODELS = ("exponential", "power", "energy", "sigmoid", "squared")
#: the multipliers that read a single ``scale`` and the scale the library uses when none is given
MULT_SCALE_DEFAULTS: dict[str, float] = {"exponential": 2.0, "energy": 3.0, "squared": 400.0}
PRECISIONS = ("uint16", "float32")
PRECISION_LABELS = {"uint16": "16-bit (all backends)",
                    "float32": "32-bit float (lossless; graph libraries and GPU only)"}

SLOPE_NEEDS_DEM = "Needs a terrain model (DEM)"

OBJECTIVE_DEFAULTS: dict[str, Any] = {
    "weights": {"cost": 1.0, "length": 0.0, "gradient": 0.0},
    "layers": [],
    "dem_layer_id": None,
    "gradient": {"max_gradient_pct": None, "additive": None, "additive_scale": 2.0,
                 "multiplier": None, "mult_scale": None, "sig_center_deg": 15.0,
                 "sig_steepness": 0.2, "sig_max_penalty": 1000.0,
                 "bin_width_pct": 0.25, "s_max_pct": 200.0},
    "weight_precision": "uint16",
}

#: the three presets as the weights they write (the card's preset buttons)
PRESETS: dict[str, dict[str, float]] = {
    "classic": {"cost": 1.0, "length": 0.0, "gradient": 0.0},
    "shortest": {"cost": 0.0, "length": 1.0, "gradient": 0.0},
    "gentlest": {"cost": 0.0, "length": 0.01, "gradient": 1.0},
}

#: gradient keys whose value None is meaningful (blank = no limit / no shape / model default)
_GRADIENT_OPTIONAL = ("max_gradient_pct", "additive", "multiplier", "mult_scale")
_GRADIENT_CHOICES = ("additive", "multiplier")


class FieldSpec(NamedTuple):
    """One numeric setting: what the card labels it and what is allowed. P5 builds its controls from these."""

    key: str                 # problem key (a layer row's keys are written without the index here)
    label: str
    unit: str
    minimum: float | None
    maximum: float | None
    allow_equal: bool        # True: the minimum itself is allowed
    default: Any
    help: str


def _spec(key, label, unit, minimum, maximum, allow_equal, default, help_text=""):
    return FieldSpec(key, label, unit, minimum, maximum, allow_equal, default, help_text)


FIELD_SPECS: dict[str, FieldSpec] = {s.key: s for s in (
    _spec("weights.cost", "Cost weight", "x raster value", 0, 1e6, True, 1.0,
          "How much each metre costs per unit of the cost raster."),
    _spec("weights.length", "Length weight", "per m", 0, 1e6, True, 0.0,
          "A fixed price per metre, whatever the terrain."),
    _spec("weights.gradient", "Slope weight", "per % slope x m", 0, 1e6, True, 0.0,
          "How much each metre on a slope adds, times the slope shape. Needs a DEM."),
    _spec("layers.weight", "Layer weight", "per layer value x m", 0, 1e6, True, 1.0,
          "How much each metre costs per unit of the layer."),
    _spec("layers.hard_max", "Forbid above", "layer value", None, None, True, None,
          "Cells with a larger layer value are forbidden. Blank = no limit."),
    _spec("layers.hard_min", "Forbid below", "layer value", None, None, True, None,
          "Cells with a smaller layer value are forbidden. Blank = no limit."),
    _spec("gradient.max_gradient_pct", "Steepest allowed slope", "%", 0, 1000, False, None,
          "Steps steeper than this are forbidden. Blank = no limit."),
    _spec("gradient.additive_scale", "Exponential shape scale", "-", 0, 100, False, 2.0,
          "Scale of the exponential slope shape."),
    _spec("gradient.mult_scale", "Multiplier scale", "-", 0, 1e4, False, None,
          "Scale of the slope multiplier. Blank = the model's default (exponential 2, energy 3, squared 400)."),
    _spec("gradient.sig_center_deg", "Sigmoid centre", "deg", 0, 90, True, 15.0,
          "Slope angle at which the sigmoid multiplier is half way up."),
    _spec("gradient.sig_steepness", "Sigmoid steepness", "1/deg", 0, 10, False, 0.2,
          "How fast the sigmoid multiplier rises around its centre."),
    _spec("gradient.sig_max_penalty", "Sigmoid maximum factor", "-", 1, 1e6, True, 1000.0,
          "The multiplier the sigmoid approaches on very steep ground."),
    _spec("gradient.bin_width_pct", "Slope step of the look-up table", "%", 0.01, 10, True, 0.25,
          "Resolution of the tabulated slope response (advanced)."),
    _spec("gradient.s_max_pct", "Slope range up to", "%", None, 1000, True, 200.0,
          "Largest slope the response table represents; at least one step wide (advanced)."),
)}

#: every key :func:`problems` can return (``layers.<i>.<field>`` stands for each row)
PROBLEM_KEYS = (
    "weights", "weights.cost", "weights.length", "weights.gradient",
    "layers", "layers.<i>.name", "layers.<i>.source", "layers.<i>.weight", "layers.<i>.hard_max",
    "layers.<i>.hard_min", "dem_layer_id",
    "gradient.max_gradient_pct", "gradient.additive", "gradient.additive_scale", "gradient.multiplier",
    "gradient.mult_scale", "gradient.sig_center_deg", "gradient.sig_steepness", "gradient.sig_max_penalty",
    "gradient.bin_width_pct", "gradient.s_max_pct", "weight_precision", "graph_api", "algorithm",
)


class ObjectiveProblem(NamedTuple):
    """One unusable setting: ``"<label> <message>."`` is the sentence the user reads."""

    key: str
    label: str
    message: str
    value: Any = None
    allowed: str = ""
    unit: str = ""


class Support(NamedTuple):
    """Whether a control can apply, and why not (``reason`` is the text of a disabled control)."""

    enabled: bool
    reason: str = ""


# --------------------------------------------------------------------------- numbers and text
def format_number(value: float) -> str:
    """``1000000.0`` -> ``"1,000,000"``, ``0.25`` -> ``"0.25"``."""
    number = float(value)
    if math.isfinite(number) and number == int(number) and abs(number) < 1e15:
        return f"{int(number):,}"
    return f"{number:,.6g}"


def _as_number(value: Any) -> float | None:
    """``value`` as a finite float, or None when it is not a usable number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def describe_range(spec: FieldSpec, *, minimum: float | None = None) -> str:
    """The allowed range as words: ``"from 0 to 1,000,000"``, ``"above 0 up to 1,000"``, ``"at least 1"``."""
    low = spec.minimum if minimum is None else minimum
    high = spec.maximum
    if low is not None and high is not None:
        return (f"from {format_number(low)} to {format_number(high)}" if spec.allow_equal
                else f"above {format_number(low)} up to {format_number(high)}")
    if low is not None:
        return f"at least {format_number(low)}" if spec.allow_equal else f"above {format_number(low)}"
    if high is not None:
        return f"at most {format_number(high)}"
    return "any number"


def label_with_unit(spec: FieldSpec) -> str:
    """``"Cost weight (x raster value)"``."""
    return f"{spec.label} ({spec.unit})" if spec.unit and spec.unit != "-" else spec.label


def _shown(value: Any) -> str:
    number = _as_number(value)
    return format_number(number) if number is not None else repr(value)


def _unit_tail(unit: str) -> str:
    return f" ({unit})" if unit and unit != "-" else ""


def _number_problem(spec: FieldSpec, value: Any, *, key: str | None = None, label: str | None = None,
                    optional: bool = False, minimum: float | None = None) -> ObjectiveProblem | None:
    """A problem when ``value`` is not a number inside the spec's range (None is fine when ``optional``)."""
    key = key or spec.key
    label = label or spec.label
    if value is None:
        if optional:
            return None
        value = ""
    allowed = describe_range(spec, minimum=minimum)
    number = _as_number(value)
    if number is None:
        kind = "a finite number" if isinstance(value, float) else "a number"
        hint = "; ".join(part for part in (allowed if allowed != "any number" else "",
                                           spec.unit if spec.unit != "-" else "") if part)
        return ObjectiveProblem(
            key, label, f"is {_shown(value)}, but must be {kind}" + (f" ({hint})" if hint else ""),
            value, allowed, spec.unit)
    low = spec.minimum if minimum is None else minimum
    high = spec.maximum
    too_low = low is not None and (number < low or (number == low and not spec.allow_equal))
    too_high = high is not None and number > high
    if too_low or too_high:
        return ObjectiveProblem(
            key, label, f"is {format_number(number)}, but must be {allowed}{_unit_tail(spec.unit)}",
            number, allowed, spec.unit)
    return None


# --------------------------------------------------------------------------- normalising
def default_settings() -> dict[str, Any]:
    """A fresh copy of :data:`OBJECTIVE_DEFAULTS` (classic cost routing)."""
    return copy.deepcopy(OBJECTIVE_DEFAULTS)


def _coerce(value: Any, default: Any) -> Any:
    """Blank -> ``default``; numbers and numeric text -> float; anything else unchanged (judged later)."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return float(text)
        except ValueError:
            return text
    return str(value)


def _text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _choice(value: Any) -> str | None:
    """A select value: blank and ``"none"`` -> None, else the lower-case text."""
    text = _text_or_none(value)
    if text is None or text.lower() == "none":
        return None
    return text.lower()


def _normalise_layer(row: Any) -> dict[str, Any] | None:
    if not isinstance(row, Mapping):
        return None
    name = row.get("name")
    return {"name": "" if name is None else str(name).strip(),
            "source": _text_or_none(row.get("source")),
            "weight": _coerce(row.get("weight"), FIELD_SPECS["layers.weight"].default),
            "hard_max": _coerce(row.get("hard_max"), None),
            "hard_min": _coerce(row.get("hard_min"), None)}


def normalise_objective_settings(raw: Any) -> dict[str, Any]:
    """The objective settings of the page or of a project file, with every key present.

    Never raises and never judges a value (see the module docstring); ``{}``, None and anything that is not a
    mapping give :func:`default_settings`. The result is JSON-safe and normalising it again changes nothing.
    """
    out = default_settings()
    if not isinstance(raw, Mapping):
        return out
    weights = raw.get("weights")
    if isinstance(weights, Mapping):
        for key in WEIGHT_KEYS:
            if key in weights:
                out["weights"][key] = _coerce(weights[key], OBJECTIVE_DEFAULTS["weights"][key])
    layers = raw.get("layers")
    if isinstance(layers, (list, tuple)):
        out["layers"] = [row for row in map(_normalise_layer, layers) if row is not None]
    out["dem_layer_id"] = _text_or_none(raw.get("dem_layer_id"))
    gradient = raw.get("gradient")
    if isinstance(gradient, Mapping):
        for key, default in OBJECTIVE_DEFAULTS["gradient"].items():
            if key not in gradient:
                continue
            value = gradient[key]
            out["gradient"][key] = _choice(value) if key in _GRADIENT_CHOICES else _coerce(value, default)
    precision = _text_or_none(raw.get("weight_precision"))
    out["weight_precision"] = precision.lower() if precision else OBJECTIVE_DEFAULTS["weight_precision"]
    return out


def preset_weights(name: str) -> dict[str, float]:
    """The three weights a preset button writes (a copy). Raises ``KeyError`` for an unknown preset."""
    return dict(PRESETS[name])


def apply_preset(settings: Any, name: str) -> dict[str, Any]:
    """``settings`` with the weights of preset ``name``; layers, DEM and slope response are kept."""
    out = normalise_objective_settings(settings)
    out["weights"] = preset_weights(name)
    return out


def mult_scale_default(model: str | None) -> float | None:
    """The scale the library uses for ``model`` when none is given (None: the model has no single scale)."""
    return MULT_SCALE_DEFAULTS.get(model) if model else None


def applicable_gradient_keys(gradient: Mapping[str, Any]) -> tuple[str, ...]:
    """The slope-response keys that are read for the chosen shape and multiplier (the others are ignored)."""
    keys = ["max_gradient_pct", "additive", "multiplier", "bin_width_pct", "s_max_pct"]
    if gradient.get("additive") == "exponential":
        keys.append("additive_scale")
    model = gradient.get("multiplier")
    if model in MULT_SCALE_DEFAULTS:
        keys.append("mult_scale")
    elif model == "sigmoid":
        keys += ["sig_center_deg", "sig_steepness", "sig_max_penalty"]
    return tuple(keys)


# --------------------------------------------------------------------------- what the settings mean
def _positive(value: Any) -> bool:
    number = _as_number(value)
    return number is not None and number > 0


def _is_zero(value: Any) -> bool:
    return _as_number(value) == 0.0


def _cost_only(settings: dict[str, Any]) -> bool:
    """True when nothing but a positive cost weight is set (the precision is not looked at)."""
    weights, gradient = settings["weights"], settings["gradient"]
    cost = _as_number(weights["cost"])
    # a cost weight above its limit is not "classic": it must reach the checks and be named, not be ignored
    return (cost is not None and 0 < cost <= FIELD_SPECS["weights.cost"].maximum
            and _is_zero(weights["length"]) and _is_zero(weights["gradient"])
            and not settings["layers"] and not settings["dem_layer_id"]
            and gradient["multiplier"] is None and gradient["max_gradient_pct"] is None)


def is_classic(settings: Any) -> bool:
    """True for plain cost routing: only a cost weight above 0, no layer, no DEM, 16-bit weights.

    Such a route is computed with no objective at all (bit for bit as before the card existed). A different
    cost weight alone (``{"cost": 2}``) changes nothing about the route, so it is still classic.
    """
    s = normalise_objective_settings(settings)
    return _cost_only(s) and s["weight_precision"] == "uint16"


def needs_objective(settings: Any) -> bool:
    """True when something besides the cost weight is set, whatever the precision says."""
    return not _cost_only(normalise_objective_settings(settings))


def has_slope_terms(settings: Any) -> bool:
    """True when the settings ask for a slope response: a slope weight, a multiplier or a grade limit."""
    s = normalise_objective_settings(settings)
    gradient = s["gradient"]
    return (_positive(s["weights"]["gradient"]) or gradient["multiplier"] is not None
            or gradient["max_gradient_pct"] is not None)


def weights(settings: Any) -> dict[str, float]:
    """The weights dict handed to the library: positive reserved weights, then the positive layer weights."""
    s = normalise_objective_settings(settings)
    out: dict[str, float] = {}
    for key in WEIGHT_KEYS:
        number = _as_number(s["weights"][key])
        if number is not None and number > 0:
            out[key] = number
    for row in s["layers"]:
        number = _as_number(row["weight"])
        if number is not None and number > 0 and row["name"]:
            out[row["name"]] = number
    return out


def gradient_options(settings: Any) -> dict[str, Any]:
    """The dict for ``GradientOptions.from_dict``: the shapes with the parameters that belong to them."""
    gradient = normalise_objective_settings(settings)["gradient"]
    options: dict[str, Any] = {
        "additive": gradient["additive"], "additive_params": {},
        "multiplier": gradient["multiplier"], "multiplier_params": {},
        "max_gradient_pct": gradient["max_gradient_pct"],
        "bin_width_pct": gradient["bin_width_pct"], "s_max_pct": gradient["s_max_pct"]}
    if gradient["additive"] == "exponential":
        options["additive_params"] = {"scale": gradient["additive_scale"]}
    model = gradient["multiplier"]
    if model in MULT_SCALE_DEFAULTS:
        scale = gradient["mult_scale"]
        options["multiplier_params"] = {"scale": MULT_SCALE_DEFAULTS[model] if scale is None else scale}
    elif model == "sigmoid":
        options["multiplier_params"] = {"center_deg": gradient["sig_center_deg"],
                                        "steepness": gradient["sig_steepness"],
                                        "max_penalty": gradient["sig_max_penalty"]}
    return options


def layer_row(name: str, source: str | None, weight: float = 1.0, hard_max: float | None = None,
              hard_min: float | None = None) -> dict[str, Any]:
    """One ``layers`` entry as the card adds it (not checked: see :func:`check_layer_name`)."""
    return {"name": str(name or "").strip(), "source": _text_or_none(source),
            "weight": _coerce(weight, 1.0), "hard_max": _coerce(hard_max, None),
            "hard_min": _coerce(hard_min, None)}


def source_label(source: str | None, layer_names: Mapping[str, str] | None = None) -> str:
    """The text a layer row shows for its source."""
    if not source:
        return "(no source)"
    if source == SOURCE_DERIVE_SLOPE:
        return DERIVE_SLOPE_LABEL
    return (layer_names or {}).get(source, "(layer no longer loaded)")


def check_layer_name(name: Any, existing: Iterable[str] = ()) -> str | None:
    """Why ``name`` cannot be a new layer name (text that follows "Layer name"), or None when it is fine."""
    text = "" if name is None else str(name).strip()
    if not text:
        return "must not be empty"
    if not LAYER_NAME_PATTERN.match(text):
        return (f"is {text!r}, which is not allowed: start with a letter and use only letters, digits and "
                "underscores (at most 32 characters)")
    if text in RESERVED_LAYER_NAMES:
        return f"is {text!r}, which is reserved ({', '.join(RESERVED_LAYER_NAMES)})"
    if text in set(existing):
        return f"is {text!r}, which another layer already uses"
    return None


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
        reason = ("The raster_fim backend cannot route over a terrain model (DEM): its eikonal solution is "
                  "continuous, not the discrete least-cost route. Choose cython, a graph library or raster_gpu "
                  "in the Backend field.")
        return ("graph_api", "Backend", f"is {graph_api}, which cannot route over a terrain model (DEM): "
                "choose cython, a graph library or raster_gpu", reason)
    if graph_api not in GRADIENT_BACKENDS:
        reason = (f"The {graph_api} backend cannot apply the slope terms of a terrain model (DEM). Choose one of "
                  f"{', '.join(GRADIENT_BACKENDS)} in the Backend field.")
        return ("graph_api", "Backend", f"is {graph_api}, which cannot apply the slope terms of a terrain "
                f"model (DEM); allowed: {', '.join(GRADIENT_BACKENDS)}", reason)
    if graph_api == "cython" and str(algorithm).startswith("delta"):
        reason = ("Delta-stepping has no slope terms: with a terrain model (DEM) the cython backend runs "
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


def _dem_problems(s: dict[str, Any], raster_ids: set[str], raster_crs: Any, dem_crs: Any
                  ) -> list[ObjectiveProblem]:
    label = "Terrain model (DEM)"
    dem = s["dem_layer_id"]
    if not dem:
        reasons = []
        if _positive(s["weights"]["gradient"]):
            reasons.append(f"the slope weight is {_shown(s['weights']['gradient'])}")
        if s["gradient"]["multiplier"] is not None:
            reasons.append(f"a slope multiplier ({s['gradient']['multiplier']}) is set")
        if s["gradient"]["max_gradient_pct"] is not None:
            reasons.append(f"a steepest allowed slope ({_shown(s['gradient']['max_gradient_pct'])} %) is set")
        if not reasons:
            return []
        return [ObjectiveProblem(
            "dem_layer_id", label,
            f"is not chosen, but the slope settings need one ({'; '.join(reasons)}): choose a terrain model or "
            "set them back to 0 and blank")]
    if dem not in raster_ids:
        return [ObjectiveProblem("dem_layer_id", label,
                                 "points at a raster layer that is not loaded any more: pick another one", dem)]
    if raster_crs is not None and dem_crs is not None:
        same = same_crs(raster_crs, dem_crs)
        if same is False:
            return [ObjectiveProblem(
                "dem_layer_id", label,
                f"has the coordinate system {dem_crs}, but the cost raster uses {raster_crs}: the library does "
                "not reproject a DEM file, so reproject the DEM to the project coordinate system first",
                str(dem_crs))]
    return []


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
    found: list[ObjectiveProblem] = []
    found += _weight_problems(s)
    if not found and not any(_positive(s["weights"][k]) for k in WEIGHT_KEYS) and not any(
            _positive(row["weight"]) for row in s["layers"]):
        found.append(ObjectiveProblem(
            "weights", "Weights", "are all 0, but at least one must be above 0: an all-zero objective has no "
            "cheapest route"))
    found += _layer_problems(s, loaded)
    found += _dem_problems(s, loaded, raster_crs, dem_crs)
    if s["dem_layer_id"]:
        found += _gradient_problems(s)
        issue = _terrain_backend_issue(api, algorithm or "")
        if issue is not None:
            found.append(ObjectiveProblem(issue[0], issue[1], issue[2], api if issue[0] == "graph_api" else algorithm))
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
    outside = [p for p in pts if not (left <= p[0] <= right and bottom <= p[1] <= top)]
    if outside:
        x, y = outside[0]
        return ObjectiveProblem(
            "dem_layer_id", "Terrain model (DEM)",
            f"does not cover {len(outside)} of {len(pts)} routing point(s), the first is ({x:,.1f}, {y:,.1f}): "
            "choose a DEM that covers every point", value=len(outside)), None
    if not pts:
        return None, None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    w_left, w_right = min(xs) - buffer_m, max(xs) + buffer_m
    w_bottom, w_top = min(ys) - buffer_m, max(ys) + buffer_m
    if raster_bounds is not None:
        r_left, r_bottom, r_right, r_top = (float(v) for v in raster_bounds)
        w_left, w_bottom = max(w_left, r_left), max(w_bottom, r_bottom)
        w_right, w_top = min(w_right, r_right), min(w_top, r_top)
    if w_left < left or w_bottom < bottom or w_right > right or w_top > top:
        return None, ("The terrain model (DEM) covers the routing points but not the whole search window: cells "
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
    - extra metric layers: kept as a safeguard. A session that rebuilds its search window (a point moved out
      of it) used to build a finder without the metric layers; the library carries them over now (pinned in
      ``test_a_session_that_rebuilds_its_window_keeps_the_metric_layers``), but a direct run needs no retained
      tree for them, so the GUI keeps routing direct.
    - a DEM on raster_gpu: a retained GPU field has no slope terms, the session would run the cython Dijkstra
      instead of the GPU.
    """
    s = normalise_objective_settings(settings)
    if is_classic(s):
        return False
    if s["weight_precision"] == "float32" or s["layers"]:
        return True
    return bool(s["dem_layer_id"]) and (graph_api or "cython") == "raster_gpu"


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
    extras = []
    dem = s["dem_layer_id"]
    if dem:
        extras.append(f"DEM '{names.get(dem, dem)}'")
    gradient = s["gradient"]
    limit = _as_number(gradient["max_gradient_pct"])
    if dem and limit is not None:
        extras.append(f"slope limit {format_number(limit)} %")
    if dem and gradient["additive"] and _positive(s["weights"]["gradient"]):
        extras.append(f"{gradient['additive']} slope shape")
    if dem and gradient["multiplier"]:
        extras.append(f"{gradient['multiplier']} slope multiplier")
    if s["weight_precision"] == "float32":
        extras.append("32-bit float weights")
    if dem and (graph_api or "cython") == "cython":
        extras.append("cython runs Dijkstra only")
    return text + "".join(f"; {e}" for e in extras) + "."
