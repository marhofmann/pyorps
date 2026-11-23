"""
PYORPS GUI: the data model of the objective card (constants, field specs, normalisation, plain accessors).

This is the lower half of ``objective.py``, split off to keep each module short. It holds the setting
names and choices, the numeric field specs, :class:`ObjectiveProblem` / :class:`Support`, the number
helpers, ``normalise_objective_settings`` and the accessors that read a normalised settings dict. The rules
that judge a setting (``problems``) and what reaches the library (``run_kwargs``) stay in
``services/objective.py``, which re-exports every public name of this module. See that module for the
description of the settings dict. Nothing here imports Dash, ``pyorps.gui.ids`` or the layout.
"""
from __future__ import annotations

import copy
import math
import re
from collections.abc import Iterable, Mapping
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


def _not_a_number_problem(spec: FieldSpec, value: Any, key: str, label: str, allowed: str) -> ObjectiveProblem:
    kind = "a finite number" if isinstance(value, float) else "a number"
    hint = "; ".join(part for part in (allowed if allowed != "any number" else "",
                                       spec.unit if spec.unit != "-" else "") if part)
    return ObjectiveProblem(
        key, label, f"is {_shown(value)}, but must be {kind}" + (f" ({hint})" if hint else ""),
        value, allowed, spec.unit)


def _out_of_range(spec: FieldSpec, number: float, minimum: float | None) -> bool:
    low = spec.minimum if minimum is None else minimum
    high = spec.maximum
    too_low = low is not None and (number < low or (number == low and not spec.allow_equal))
    too_high = high is not None and number > high
    return too_low or too_high


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
        return _not_a_number_problem(spec, value, key, label, allowed)
    if _out_of_range(spec, number, minimum):
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


def _fill_weights(raw: Mapping[str, Any], out: dict[str, Any]) -> None:
    weights = raw.get("weights")
    if isinstance(weights, Mapping):
        for key in WEIGHT_KEYS:
            if key in weights:
                out["weights"][key] = _coerce(weights[key], OBJECTIVE_DEFAULTS["weights"][key])


def _fill_gradient(raw: Mapping[str, Any], out: dict[str, Any]) -> None:
    gradient = raw.get("gradient")
    if isinstance(gradient, Mapping):
        for key, default in OBJECTIVE_DEFAULTS["gradient"].items():
            if key not in gradient:
                continue
            value = gradient[key]
            out["gradient"][key] = _choice(value) if key in _GRADIENT_CHOICES else _coerce(value, default)


def normalise_objective_settings(raw: Any) -> dict[str, Any]:
    """The objective settings of the page or of a project file, with every key present.

    Never raises and never judges a value (see the module docstring); ``{}``, None and anything that is not a
    mapping give :func:`default_settings`. The result is JSON-safe and normalising it again changes nothing.
    """
    out = default_settings()
    if not isinstance(raw, Mapping):
        return out
    _fill_weights(raw, out)
    layers = raw.get("layers")
    if isinstance(layers, (list, tuple)):
        out["layers"] = [row for row in map(_normalise_layer, layers) if row is not None]
    out["dem_layer_id"] = _text_or_none(raw.get("dem_layer_id"))
    _fill_gradient(raw, out)
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
        return (f"is {text!r}, which is not allowed: start with a letter and use only letters, digits and " +
                "underscores (at most 32 characters)")
    if text in RESERVED_LAYER_NAMES:
        return f"is {text!r}, which is reserved ({', '.join(RESERVED_LAYER_NAMES)})"
    if text in set(existing):
        return f"is {text!r}, which another layer already uses"
    return None
