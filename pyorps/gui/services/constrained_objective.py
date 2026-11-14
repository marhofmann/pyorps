"""
PYORPS GUI: objective weights of constrained (overhead-line) routing (round 16, package R3-B).

The "Objective weights" block of the constrained card (``sections/profile_form.py`` builds it,
``callbacks/profile_form.py`` mirrors it, ``callbacks/constrained.py`` runs with it) only holds controls; every
rule about what is allowed and what reaches the library lives here, so it is unit-tested without a page. The
module imports nothing from Dash, ``pyorps.gui.ids`` or the layout.

What the library offers (``pyorps/graph/constrained_path_finder.py``, ``pyorps/core/objective.py``)
---------------------------------------------------------------------------------------------------
``ConstrainedPathFinder(..., objective=<dict of weights>)`` takes the reserved weight names ``smoothness`` and
``tower`` next to ``cost``. ``smoothness`` multiplies the turn-angle penalty table of the profile (the soft cost of
a bend; the hard turn limit is never scaled), ``tower`` multiplies the two tower cost tables (foundation cost by
ground value, tower-type cost by turn angle). Both default to 1, and a weight of 0 switches its term off. Every
weight must be finite and at least 0; the library sets no upper limit (this GUI does, see ``FIELD_SPECS``).
The library refuses an objective on a backend other than ``cython``, together with the clearance check of the
profile, and with a ``gradient`` weight; a terrain model (DEM) only makes it warn. The GUI offers the two weights
above and keeps ``cost`` at 1 (an objective without a cost weight prices the ground at 0). Only the ratios of
the weights matter: ``{"cost": 2}`` routes like ``{"cost": 1, "tower": 0.5, "smoothness": 0.5}``, so a separate
cost weight would add nothing to the two controls. A ``length`` weight is not offered either: the library takes
it (terrain, tower and turn costs are added in one unit since the round-16 fix; before, the quantised search
raster was mixed with unscaled tables), but it takes the search off the exact cost-only path and rounds the
terrain term to 16 bit. Reported tower costs follow the ``tower`` weight and the reported turn penalties are the
ones the search charged; a route stores and names the weights it was computed with.

The settings dict (stored in ``state.gui_settings["constrained"]["objective"]``, saved in the project file)
------------------------------------------------------------------------------------------------------
``{"smoothness": 1.0, "tower": 1.0}``. A project that never changed a weight has no ``objective`` key. A blank
(None or empty text) takes the default of its key; numeric text becomes a float; a value that is not a number
stays as it is, so that :func:`problems` can name it. ``{}`` and None give the defaults: :func:`is_default` is
True, :func:`library_objective` returns None, and the route is computed exactly as it was before the weights
existed (no ``objective`` reaches the library).

Problems
--------
:func:`problems` returns ``list[(key, message)]``. ``key`` is one of :data:`PROBLEM_KEYS` (a weight key for a
value that is not allowed, ``"backend"`` and ``"clearance"`` for a combination the library refuses), ``message``
is a whole sentence that names the setting, the value, the allowed range and the unit. The first problem stops a
run.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, NamedTuple

from pyorps.core.exceptions import InvalidInputError

from .errors import Notice

# --------------------------------------------------------------------------- constants
#: the weights the block shows, in display order (library names: reserved objective layers)
WEIGHT_KEYS = ("smoothness", "tower")
#: the only backend that takes an objective (``ConstrainedPathFinder``: per-cell tower costs)
SUPPORTED_BACKEND = "cython"
#: the cost weight the library objective always carries (a cost-only objective routes like the profile)
COST_WEIGHT = 1.0
#: the largest weight the GUI accepts (the library has no limit): beyond it one bend or one tower costs more than
#: the whole terrain, so the route would no longer be a weighing of anything
WEIGHT_MAXIMUM = 1000.0

#: every key :func:`problems` can return
PROBLEM_KEYS = (*WEIGHT_KEYS, "backend", "clearance")


class FieldSpec(NamedTuple):
    """One weight: what the block labels it and what is allowed (the controls are built from these)."""

    key: str
    label: str
    unit: str
    minimum: float | None
    maximum: float | None
    allow_equal: bool        # True: the minimum itself is allowed
    default: float
    help: str


FIELD_SPECS: dict[str, FieldSpec] = {s.key: s for s in (
    FieldSpec("smoothness", "Turn penalty weight", "x profile turn cost", 0.0, WEIGHT_MAXIMUM, True, 1.0,
              "Multiplies the cost of every bend in the line (the turn cost of the profile). 1 keeps the "
              "profile as written, 0 makes bends free, 2 makes them twice as expensive. The hard turn limit "
              "of the profile is never changed."),
    FieldSpec("tower", "Tower cost weight", "x profile tower cost", 0.0, WEIGHT_MAXIMUM, True, 1.0,
              "Multiplies the cost of every tower (the profile's foundation cost by ground value and its "
              "tower-type cost by turn angle). 1 keeps the profile as written, 0 makes towers free, 2 makes "
              "them twice as expensive. Towers exist only when the profile sets min_span_m and max_span_m."),
)}

DEFAULTS: dict[str, float] = {key: spec.default for key, spec in FIELD_SPECS.items()}

#: the label of a key as a problem names it ("Turn penalty weight (smoothness)")
LABELS: dict[str, str] = {
    **{key: f"{spec.label} ({key})" for key, spec in FIELD_SPECS.items()},
    "backend": "Backend", "clearance": "Clearance check"}


class Support(NamedTuple):
    """Whether the weights can apply, and why not (``reason`` is the text of the disabled block)."""

    enabled: bool
    reason: str = ""


class View(NamedTuple):
    """What the block shows for the current weights and profile: its status line and its disabled state."""

    text: str                # the status line
    problem: bool            # True: the text lists problems (shown as an error)
    enabled: bool            # False: the weights cannot apply, the number inputs are disabled
    reason: str              # why they are disabled ("" when enabled)


# --------------------------------------------------------------------------- numbers
def format_number(value: float) -> str:
    """``1000.0`` -> ``"1,000"``, ``0.25`` -> ``"0.25"``."""
    number = float(value)
    if math.isfinite(number) and number == int(number) and abs(number) < 1e15:
        return f"{int(number):,}"
    return f"{number:,.10g}"


def describe_range(spec: FieldSpec) -> str:
    """The allowed range as words with its unit: ``">= 0 and <= 1,000 (x profile turn cost)"``."""
    parts = []
    if spec.minimum is not None:
        parts.append(f"{'>=' if spec.allow_equal else '>'} {format_number(spec.minimum)}")
    if spec.maximum is not None:
        parts.append(f"<= {format_number(spec.maximum)}")
    text = " and ".join(parts) if parts else "any number"
    return f"{text} ({spec.unit})" if spec.unit else text


def label_with_unit(spec: FieldSpec) -> str:
    """``"Turn penalty weight (x profile turn cost)"``."""
    return f"{spec.label} ({spec.unit})" if spec.unit else spec.label


def _as_number(value: Any) -> float | None:
    """``value`` as a finite float, or None when it is not a usable number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _shown(value: Any) -> str:
    number = _as_number(value)
    return format_number(number) if number is not None else repr(value)


# --------------------------------------------------------------------------- normalising
def _coerce(value: Any, default: float) -> Any:
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


def default_settings() -> dict[str, float]:
    """A fresh copy of :data:`DEFAULTS` (the profile's own costs)."""
    return dict(DEFAULTS)


def normalise(raw: Any) -> dict[str, Any]:
    """The weights of the page or of a project file, with every key present.

    Never raises and never judges a value (see the module docstring); ``{}``, None and anything that is not a
    mapping give :func:`default_settings`; unknown keys are dropped. The result is JSON-safe (a text that is no
    number stays text) and normalising it again changes nothing.
    """
    out: dict[str, Any] = default_settings()
    if not isinstance(raw, Mapping):
        return out
    for key in WEIGHT_KEYS:
        if key in raw:
            out[key] = _coerce(raw[key], DEFAULTS[key])
    return out


def input_values(raw: Any) -> dict[str, float | None]:
    """What the number inputs show for these weights: a finite number as it is (also one outside the range, so
    that the user sees what is wrong), anything else blank."""
    settings = normalise(raw)
    return {key: _as_number(settings[key]) for key in WEIGHT_KEYS}


def is_default(raw: Any) -> bool:
    """True when every weight is its default: classic constrained routing, nothing goes to the library."""
    settings = normalise(raw)
    return all(_as_number(settings[key]) == DEFAULTS[key] for key in WEIGHT_KEYS)


# --------------------------------------------------------------------------- problems
def _value_problem(spec: FieldSpec, value: Any) -> str | None:
    """Why ``value`` is not an allowed weight, or None (the sentence names setting, value, range and unit)."""
    subject = f"{spec.label} ({spec.key})"
    allowed = f"Allowed: {describe_range(spec)}."
    number = _as_number(value)
    if number is None:
        kind = "a finite number" if isinstance(value, float) else "a number"
        return f"{subject}: {_shown(value)} is not {kind}. {allowed}"
    shown = f"{format_number(number)} {spec.unit}".rstrip()
    if spec.minimum is not None and (number < spec.minimum or (number == spec.minimum and not spec.allow_equal)):
        return f"{subject}: {shown} is too low. {allowed}"
    if spec.maximum is not None and number > spec.maximum:
        return f"{subject}: {shown} is too high. {allowed}"
    return None


def _changed_text(settings: Mapping[str, Any]) -> str:
    """``"turn penalty weight 0.5, tower cost weight 2"`` for the weights that are not at their default."""
    return ", ".join(f"{FIELD_SPECS[key].label.lower()} {_shown(settings[key])}" for key in WEIGHT_KEYS
                     if _as_number(settings[key]) != DEFAULTS[key])


def backend_reason(backend: str | None) -> str:
    """Why the weights cannot apply with this backend (the library runs an objective on ``cython`` only)."""
    return (f"Objective weights work with the {SUPPORTED_BACKEND} backend only: the parallel and GPU variants "
            f"do not take them yet. The backend is {backend or SUPPORTED_BACKEND}: choose {SUPPORTED_BACKEND} "
            "in the Backend field, or press Reset weights.")


CLEARANCE_REASON = (
    "The profile has the clearance check on (tower height, conductor weight, conductor tension and ground "
    "clearance are all set), and the library does not take objective weights together with it yet. Clear one "
    "of these four fields in the profile form, or press Reset weights.")


def support(*, backend: str | None = SUPPORTED_BACKEND, clearance_on: bool = False) -> Support:
    """Whether the weights can apply with this backend and this profile, and why not."""
    if (backend or SUPPORTED_BACKEND) != SUPPORTED_BACKEND:
        return Support(False, backend_reason(backend))
    if clearance_on:
        return Support(False, CLEARANCE_REASON)
    return Support(True)


def problems(raw: Any, *, backend: str | None = SUPPORTED_BACKEND, clearance_on: bool = False,
             has_dem: bool = False) -> list[tuple[str, str]]:
    """Everything that stops a run with these weights, as ``(key, message)``; ``[]`` when it can run.

    Weights that are not allowed values come first (one per weight). The two combinations the library refuses
    - a backend other than ``cython``, and the clearance check of the profile - are only judged when a weight
    is not at its default (an objective-free run has nothing to refuse) and when the values are usable.
    ``backend`` is the backend the run will use (after the experimental gate); ``clearance_on`` is
    ``profile_form.clearance_state(profile) == "on"``. ``has_dem`` is accepted so that a caller can pass the
    whole page state, but a terrain model (DEM) never makes a problem: the library only warns (see
    :func:`notes`).
    """
    del has_dem                       # a DEM is a note, never a refusal
    settings = normalise(raw)
    out = [(key, text) for key in WEIGHT_KEYS if (text := _value_problem(FIELD_SPECS[key], settings[key]))]
    if out or is_default(settings):
        return out
    changed = _changed_text(settings)
    if (backend or SUPPORTED_BACKEND) != SUPPORTED_BACKEND:
        out.append(("backend", f"Backend: {backend} cannot run the weights ({changed}). {backend_reason(backend)}"))
    if clearance_on:
        out.append(("clearance", f"Clearance check: on, so the weights ({changed}) cannot run. {CLEARANCE_REASON}"))
    return out


def problems_text(found: list[tuple[str, str]], limit: int = 3) -> str:
    """The first ``limit`` messages, one per line, plus a count of the rest (for the status line)."""
    lines = [message for _, message in found[:limit]]
    if len(found) > limit:
        lines.append(f"... and {len(found) - limit} more.")
    return "\n".join(lines)


def problem_notice(found: list[tuple[str, str]], *, controls: Mapping[str, str] | None = None,
                   tab: str = "tab-routes", impact: str = "No overhead-line route was computed.") -> dict | None:
    """The first problem as an error notice dict (``Notice.to_dict()``), or None without problems.

    ``controls`` maps a problem key to the component id the "Go fix" button takes the user to (the caller owns
    the ids, this service does not know them).
    """
    if not found:
        return None
    key, message = found[0]
    label = LABELS.get(key, key)
    if key in FIELD_SPECS:
        fix = f"Correct the {FIELD_SPECS[key].label} field in the Objective weights block, or press Reset weights."
    elif key == "backend":
        fix = f"Choose the {SUPPORTED_BACKEND} backend, or press Reset weights to run without weights."
    else:
        fix = "Clear one of the four clearance fields in the profile form, or press Reset weights."
    return Notice(
        severity="error", title=f"{label}: value not usable", meaning=message, impact=impact, fix=fix,
        focus_id=tab, focus_control=(controls or {}).get(key)).to_dict()


# --------------------------------------------------------------------------- what reaches the library
def library_objective(raw: Any) -> dict[str, float] | None:
    """The ``objective=`` dict for ``ConstrainedPathFinder``, or None for classic constrained routing.

    None when every weight is its default: the library is then called as it always was. Otherwise the cost
    weight 1 and every weight of the block (also the one that stays at 1, so what the library gets is what
    :func:`used_weights` reports). Raises ``InvalidInputError`` for a value that :func:`problems` would name;
    the callers check :func:`problems` first.
    """
    settings = normalise(raw)
    for key in WEIGHT_KEYS:
        text = _value_problem(FIELD_SPECS[key], settings[key])
        if text:
            raise InvalidInputError(text)
    if is_default(settings):
        return None
    return {"cost": COST_WEIGHT, **{key: float(settings[key]) for key in WEIGHT_KEYS}}


def used_weights(raw: Any) -> dict[str, float]:
    """The weights a run used, as stored with the route; ``{}`` for classic routing (nothing was passed)."""
    objective = library_objective(raw)
    return {} if objective is None else {key: objective[key] for key in WEIGHT_KEYS}


def weights_text(weights: Mapping[str, Any] | None) -> str:
    """``"turn penalty weight 0.5, tower cost weight 2"`` for the weights a route used; "" when none."""
    if not weights:
        return ""
    return ", ".join(f"{FIELD_SPECS[key].label.lower()} {_shown(weights[key])}"
                     for key in WEIGHT_KEYS if key in weights)


def summary(raw: Any) -> str:
    """One line: what the weights make of the run."""
    settings = normalise(raw)
    if is_default(settings):
        return "Both weights are 1: the turn and tower costs of the profile apply as written."
    changed = _changed_text(settings)
    return f"Weights: {changed}. The turn and tower costs of the profile are multiplied by them."


def notes(raw: Any, *, has_dem: bool = False, spans_on: bool | None = None) -> list[str]:
    """Facts worth knowing about weights that are not at their default (the run is not refused)."""
    settings = normalise(raw)
    out = []
    if is_default(settings):
        return out
    if has_dem:
        out.append("With a terrain model (DEM) the profile's own slope model applies; the weights do not change it.")
    if spans_on is False and _as_number(settings["tower"]) != DEFAULTS["tower"]:
        out.append("The profile sets no min_span_m and max_span_m, so it has no towers and the tower cost weight "
                   "changes nothing.")
    return out


def view(raw: Any, *, backend: str | None = SUPPORTED_BACKEND, clearance_on: bool = False,
         has_dem: bool = False, spans_on: bool | None = None) -> View:
    """The status line and the disabled state for the current weights, backend and profile.

    Problems come first (they are what a run would refuse with); without problems the summary and the notes.
    The inputs are disabled when the weights cannot apply (:func:`support`), which is also the text of the
    status line, so the reason is visible and not only a hover text.
    """
    sup = support(backend=backend, clearance_on=clearance_on)
    found = problems(raw, backend=backend, clearance_on=clearance_on)
    if found:
        return View(problems_text(found), True, sup.enabled, sup.reason)
    if not sup.enabled:
        return View(sup.reason, False, False, sup.reason)
    return View("\n".join([summary(raw), *notes(raw, has_dem=has_dem, spans_on=spans_on)]), False, True, "")
