"""
PYORPS GUI: the low-level part of the line-end settings (round 17, package P4) - constants, field specs and labels,
value coercion, the per-value problem checks and the facts read from a profile.

Split from :mod:`constrained_ends` (which re-exports every name here, so callers keep importing from it): this
module holds what needs no profile, no backend and no library - the two height keys and their limits, the
``terminal_span_rule`` labels, ``normalise`` (blank -> None, numeric text -> float, never raises) and the check
of one height value (``_value_problem``). It imports nothing from Dash, ``pyorps.gui.ids`` or the layout, and
nothing from :mod:`constrained_ends` (no cycle). The settings dict and its rules are described in the docstring
of :mod:`constrained_ends`.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, NamedTuple

from pyorps.core.exceptions import InvalidInputError
from pyorps.core.infrastructure_profile import TERMINAL_SPAN_RULES

from .constrained_objective import describe_range, format_number
from .profile_form import clearance_state

# --------------------------------------------------------------------------- constants
#: the two finder keywords, in display order (source first)
HEIGHT_KEYS = ("source_connection_height_m", "target_connection_height_m")
#: the largest height the GUI accepts (the library has no limit; the profile's own height fields stop at 300 m too)
HEIGHT_MAXIMUM = 300.0
#: the value of ``terminal_span_rule`` of a profile that does not name one
DEFAULT_RULE = "remove"
#: the rule value that needs the search (exact bins, CPU backends only)
ENFORCE_RULE = "enforce"

#: every key :func:`problems` and :func:`bins_problem` can return
PROBLEM_KEYS = (*HEIGHT_KEYS, "rule", "backend", "bins")


class FieldSpec(NamedTuple):
    """One height: what the block labels it and what is allowed (the controls are built from these)."""

    key: str
    label: str
    unit: str
    minimum: float | None
    maximum: float | None
    allow_equal: bool        # True: the minimum itself is allowed
    default: float | None    # None = blank = not set
    help: str


_HEIGHT_HELP = (
    "Height above the ground at the {end} point at which the conductor leaves the structure there ({example}). " +
    "Blank = not set: the line end behaves as before. With a terrain model (DEM) and a profile that has tower " +
    "height, conductor weight, conductor tension, ground clearance and span limits, the search chooses the " +
    "position and height of the {which} tower so that the {which} span clears the ground from this height; " +
    "otherwise the height is only reported as the height of the {end} tower and the route does not change.")

FIELD_SPECS: dict[str, FieldSpec] = {s.key: s for s in (
    FieldSpec("source_connection_height_m", "Source connection height", "m", 0.0, HEIGHT_MAXIMUM, False, None,
              _HEIGHT_HELP.format(end="source", which="first", example="for example a substation gantry")),
    FieldSpec("target_connection_height_m", "Target connection height", "m", 0.0, HEIGHT_MAXIMUM, False, None,
              _HEIGHT_HELP.format(end="target", which="last", example="for example the top of an existing tower"))
)}

DEFAULTS: dict[str, float | None] = {key: spec.default for key, spec in FIELD_SPECS.items()}

#: the label of a key as a problem names it ("Source connection height (source_connection_height_m)")
LABELS: dict[str, str] = {
    **{key: f"{spec.label} ({key})" for key, spec in FIELD_SPECS.items()},
    "rule": "End-span rule (terminal_span_rule)", "backend": "Backend", "bins": "Exact span bins"}


class RuleLabel(NamedTuple):
    """One value of ``terminal_span_rule`` as the page words it (the select is built from :data:`RULE_LABELS`)."""

    value: str
    label: str               # the option text of the select
    help: str                # one or two sentences for the help text


#: the four values in the library's order, from one table (the page package builds the select from it)
RULE_LABELS: dict[str, RuleLabel] = {r.value: r for r in (
    RuleLabel("remove", "Remove it (old behaviour; can leave a span above the longest span)",
              "A tower the search placed closer than the shortest span to a line end is removed afterwards. The " +
              "end span then becomes longer and can end up above the longest span; a warning says so."),
    RuleLabel("remove_if_valid", "Remove it only if every span stays valid",
              "The tower is removed only when the line stays valid without it (a straight stretch, a span below " +
              "the longest span, ground clearance kept); otherwise it is kept and flagged."),
    RuleLabel("keep", "Keep it (the end span may be shorter than the shortest span)",
              "Every tower the search placed is kept. A short end span is flagged in the route's notes."),
    RuleLabel("enforce", "Never place one (end spans at least the shortest span)",
              "The search itself keeps every end span at least as long as the shortest span, so nothing has to be " +
              "removed or repaired afterwards. It searches on exact span bins (more memory and time than a " +
              "default run) and works on the CPU backends only."),
)}


def rule_options() -> list[dict[str, str]]:
    """The options of the select, ``[{"label": ..., "value": ...}]``, in the library's order."""
    return [{"label": RULE_LABELS[value].label, "value": value} for value in TERMINAL_SPAN_RULES]


def rule_label(value: Any) -> str:
    """The option text of a rule value; the value itself (as text) when it is none of the four."""
    entry = RULE_LABELS.get(value) if isinstance(value, str) else None
    return entry.label if entry else str(value)


# --------------------------------------------------------------------------- numbers
def _as_number(value: Any) -> float | None:
    """``value`` as a finite float, or None when it is not a usable number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _shown(value: Any) -> str:
    number = _as_number(value)
    return format_number(number) if number is not None else repr(value)


def _coerce_text(value: str) -> Any:
    """A text height: blank -> None, numeric text -> float, anything else (also ``"nan"``) as stripped text."""
    text = value.strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return text
    return number if math.isfinite(number) else text


def _coerce(value: Any) -> Any:
    """Blank -> None; numbers and numeric text -> float; anything else unchanged (judged later).

    A non-finite float (or the text ``"nan"`` / ``"inf"``) is kept as text, so that the settings stay JSON-safe.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else repr(number)
    if isinstance(value, str):
        return _coerce_text(value)
    return str(value)


def default_settings() -> dict[str, float | None]:
    """A fresh copy of :data:`DEFAULTS` (no height set)."""
    return dict(DEFAULTS)


def normalise(raw: Any) -> dict[str, Any]:
    """The heights of the page or of a project file, with every key present.

    Never raises and never judges a value (see the module docstring); ``{}``, None and anything that is not a
    mapping give :func:`default_settings`; unknown keys are dropped. The result is JSON-safe and normalising it
    again changes nothing.
    """
    out: dict[str, Any] = default_settings()
    if not isinstance(raw, Mapping):
        return out
    for key in HEIGHT_KEYS:
        if key in raw:
            out[key] = _coerce(raw[key])
    return out


def input_values(raw: Any) -> dict[str, float | None]:
    """What the number inputs show: a finite number as it is (also one outside the range, so that the user sees
    what is wrong), anything else blank."""
    settings = normalise(raw)
    return {key: _as_number(settings[key]) for key in HEIGHT_KEYS}


def is_default(raw: Any) -> bool:
    """True when no height is set: classic constrained routing, nothing goes to the library."""
    settings = normalise(raw)
    return all(settings[key] is None for key in HEIGHT_KEYS)


def used_ends(raw: Any) -> dict[str, float]:
    """The heights a run was given, as stored with the route; ``{}`` when none is set.

    Raises ``InvalidInputError`` for a value that :func:`problems` would name (callers check it first).
    """
    settings = normalise(raw)
    _raise_for_bad_values(settings)
    return {key: float(settings[key]) for key in HEIGHT_KEYS if settings[key] is not None}


def _too_low(spec: FieldSpec, number: float) -> bool:
    """True when ``number`` is below the minimum (or equal to it where the minimum itself is not allowed)."""
    if spec.minimum is None:
        return False
    return number < spec.minimum or (number == spec.minimum and not spec.allow_equal)


def _too_high(spec: FieldSpec, number: float) -> bool:
    return spec.maximum is not None and number > spec.maximum


def _value_problem(spec: FieldSpec, value: Any) -> str | None:
    """Why ``value`` is not an allowed height, or None (blank is allowed: the height is not set)."""
    if value is None:
        return None
    subject = f"{spec.label} ({spec.key})"
    allowed = f"Allowed: {describe_range(spec)}, or leave it blank."
    number = _as_number(value)
    if number is None:
        return f"{subject}: {_shown(value)} is not a finite number. {allowed}"
    shown = f"{format_number(number)} {spec.unit}".rstrip()
    if _too_low(spec, number):
        return f"{subject}: {shown} is too low. {allowed}"
    if _too_high(spec, number):
        return f"{subject}: {shown} is too high. {allowed}"
    return None


def _raise_for_bad_values(settings: Mapping[str, Any]) -> None:
    for key in HEIGHT_KEYS:
        text = _value_problem(FIELD_SPECS[key], settings[key])
        if text:
            raise InvalidInputError(text)


# --------------------------------------------------------------------------- the profile, as facts
class ProfileFacts(NamedTuple):
    """What the end settings need to know about the profile (read from a dict or an ``InfrastructureProfile``)."""

    spans_on: bool                   # min_span_m and max_span_m both set: the profile places towers
    clearance_on: bool               # tower height, conductor weight, tension and clearance all set
    rule: str                        # terminal_span_rule (a missing key is "remove")
    min_span_m: Any
    max_span_m: Any
    n_heights: int                   # height classes of the search (1 unless several tower heights are set)


def _data(profile: Any) -> dict:
    if isinstance(profile, Mapping):
        return dict(profile)
    to_dict = getattr(profile, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    return {}


def profile_facts(profile: Any) -> ProfileFacts:
    """The facts of ``profile`` (a profile dict, an ``InfrastructureProfile`` or None), by the library's rules."""
    data = _data(profile)
    heights = data.get("tower_heights_m")
    n_heights = len(heights) if isinstance(heights, (list, tuple)) and len(heights) > 1 else 1
    rule = data.get("terminal_span_rule")
    return ProfileFacts(
        spans_on=data.get("min_span_m") is not None and data.get("max_span_m") is not None,
        clearance_on=clearance_state(data) == "on",
        rule=DEFAULT_RULE if rule is None else rule,
        min_span_m=data.get("min_span_m"), max_span_m=data.get("max_span_m"), n_heights=n_heights)


def _has_dem(dem: Any) -> bool:
    return dem is not None and not (isinstance(dem, str) and not dem.strip()) and dem is not False
