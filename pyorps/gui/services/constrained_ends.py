"""
PYORPS GUI: the two ends of an overhead line - connection heights and the end-span rule (round 17, package P4).

The "Line ends" block of the constrained card (the page package builds it, the callbacks mirror it, the run reads
it) only holds controls; every rule about what is allowed and what reaches the library lives here, so it is
unit-tested without a page. The module imports nothing from Dash, ``pyorps.gui.ids`` or the layout. It imports the
library (``ConstrainedPathFinder``) only inside the functions that need its constants, and only for those.

What the library offers (``pyorps/graph/constrained_path_finder.py``)
---------------------------------------------------------------------
* Two run settings, ``source_connection_height_m`` and ``target_connection_height_m`` (finder arguments and
  ``find_route`` keywords): the height in metres above the ground at which the conductor leaves the structure at the
  line end (a substation gantry, an existing tower). ``None`` is "not set" and means today's behaviour. A height
  reaches the SEARCH only when the clearance check is on (a terrain model, a profile with tower height, conductor
  weight, conductor tension, minimum clearance, and span limits); otherwise it is only reported and the finder warns
  "has no effect". The profile's ``terminal_tower_height_m`` stays a report-only value.
* The profile field ``terminal_span_rule`` with four values (``TERMINAL_SPAN_RULES``): ``remove`` (default),
  ``remove_if_valid``, ``keep`` and ``enforce``. It is a PROFILE field: this service never stores it a second time;
  it only knows what it means for the page (:data:`RULE_LABELS`, :func:`rule_support`) and what it implies for a
  run (:func:`problems`, :func:`bins_problem`).
* ``"enforce"`` and a height the search uses run on EXACT span bins (a bin of about one step instead of
  ``min_span_m``), so that these rules can never hide a feasible route. The search state space is cells x
  directions x bins x height classes; above ``ConstrainedPathFinder.EXACT_BINS_MAX_STATES`` (or above
  ``EXACT_BINS_MAX_COUNT`` bins, or ``EXACT_BINS_MAX_STEPS`` steps per longest span, or when the memory of the kernel
  it would call, ``ConstrainedPathFinder.exact_bins_memory``, is above ``EXACT_BINS_MAX_BYTES``) the finder raises
  ``InvalidInputError``. The GPU backends refuse ``"enforce"`` and a
  height the search would use (``AlgorithmNotImplementedError``). A bad height is ``InvalidTypeError`` or
  ``InvalidInputError``. Both limits are the library's; the GUI only adds the 300 m maximum of the profile's own
  height fields.

The settings dict (stored in ``state.gui_settings["constrained"]["ends"]``, saved in the project file)
-----------------------------------------------------------------------------------------------------
``{"source_connection_height_m": None, "target_connection_height_m": 25.0}``. A project that never set a height has
no ``ends`` key. A blank (None or empty text) is None ("not set"); numeric text becomes a float; a value that is
not a number stays as it is (a non-finite float becomes its text), so that :func:`problems` can name it. ``{}`` and
None give the defaults: :func:`is_default` is True, :func:`library_kwargs` returns ``{}``, nothing new reaches the
library and the route is computed exactly as before.

Problems
--------
:func:`problems` returns ``list[(key, message)]``. ``key`` is one of :data:`PROBLEM_KEYS`; ``message`` is a whole
sentence that names the setting, the value, the allowed range and the unit. A height that would have no effect is
NOT a problem: it is a note (:func:`notes`, :func:`view`: "reported only, the route does not change"). The size of
the exact-bin search needs the raster window, so it is a separate function, :func:`bins_problem`;
:func:`run_problems` joins the two.

Segments (waypoints)
--------------------
A run through waypoints is one search per consecutive pair of points. The source height belongs to the first
segment and the target height to the last one; the ends at waypoints are free (:func:`library_kwargs`). The end-span
rule is a profile field and applies to every segment.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any, NamedTuple

from pyorps.core.exceptions import InvalidInputError
from pyorps.core.infrastructure_profile import TERMINAL_SPAN_RULES

from .constrained_objective import Support, View, describe_range, format_number, problems_text
from .errors import Notice
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
    "Height above the ground at the {end} point at which the conductor leaves the structure there ({example}). "
    "Blank = not set: the line end behaves as before. With a terrain model (DEM) and a profile that has tower "
    "height, conductor weight, conductor tension, ground clearance and span limits, the search chooses the "
    "position and height of the {which} tower so that the {which} span clears the ground from this height; "
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
              "A tower the search placed closer than the shortest span to a line end is removed afterwards. The "
              "end span then becomes longer and can end up above the longest span; a warning says so."),
    RuleLabel("remove_if_valid", "Remove it only if every span stays valid",
              "The tower is removed only when the line stays valid without it (a straight stretch, a span below "
              "the longest span, ground clearance kept); otherwise it is kept and flagged."),
    RuleLabel("keep", "Keep it (the end span may be shorter than the shortest span)",
              "Every tower the search placed is kept. A short end span is flagged in the route's notes."),
    RuleLabel("enforce", "Never place one (end spans at least the shortest span)",
              "The search itself keeps every end span at least as long as the shortest span, so nothing has to be "
              "removed or repaired afterwards. It searches on exact span bins (more memory and time than a "
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
        text = value.strip()
        if not text:
            return None
        try:
            number = float(text)
        except ValueError:
            return text
        return number if math.isfinite(number) else text
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


# --------------------------------------------------------------------------- the profile and the run, as facts
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


class Effect(NamedTuple):
    """What the end settings of one run (or one segment) ask of the search."""

    enforce: bool            # terminal_span_rule "enforce" with span limits: a minimum last span in the search
    source_acts: bool        # the source height reaches the search
    target_acts: bool        # the target height reaches the search
    clearance: bool          # the clearance check is on (DEM + clearance data + span limits)

    @property
    def heights_act(self) -> bool:
        return self.source_acts or self.target_acts

    @property
    def exact_bins(self) -> bool:
        """True when the run needs exact span bins (``"enforce"`` or a height the search uses)."""
        return self.enforce or self.heights_act


def effect(raw: Any, *, profile: Any = None, dem: Any = None) -> Effect:
    """What the finder will do with these heights, this profile and this terrain model.

    Mirrors ``ConstrainedPathFinder._resolve_line_ends``: the clearance check is on with a DEM, the clearance data
    of the profile and span limits; a height acts only then, ``"enforce"`` acts with span limits alone.
    ``raw`` holds the heights of the run or segment in question (see :func:`segment_settings`).
    """
    settings = normalise(raw)
    facts = profile_facts(profile)
    clearance = _has_dem(dem) and facts.clearance_on and facts.spans_on
    return Effect(
        enforce=facts.spans_on and facts.rule == ENFORCE_RULE,
        source_acts=clearance and settings[HEIGHT_KEYS[0]] is not None,
        target_acts=clearance and settings[HEIGHT_KEYS[1]] is not None,
        clearance=clearance)


def segment_settings(raw: Any, segment_index: int, n_segments: int) -> dict[str, Any]:
    """The heights of one segment of a run: the source height only on the first, the target height only on the last.

    Raises ``InvalidInputError`` for a segment index outside ``0 <= index < n_segments``.
    """
    if not (isinstance(n_segments, int) and isinstance(segment_index, int) and 0 <= segment_index < n_segments):
        raise InvalidInputError(
            f"Segment {segment_index!r} of {n_segments!r}: the segment index must be a whole number from 0 up to "
            "the number of segments minus 1.")
    settings = normalise(raw)
    return {HEIGHT_KEYS[0]: settings[HEIGHT_KEYS[0]] if segment_index == 0 else None,
            HEIGHT_KEYS[1]: settings[HEIGHT_KEYS[1]] if segment_index == n_segments - 1 else None}


@lru_cache(maxsize=1)
def _library():
    """``ConstrainedPathFinder`` (imported when first needed: the pure rules above never import the library)."""
    from pyorps.graph.constrained_path_finder import ConstrainedPathFinder

    return ConstrainedPathFinder


def gpu_backends() -> tuple[str, ...]:
    """The backends that run a constrained GPU solver (the library's ``GPU_BACKENDS``)."""
    return tuple(_library().GPU_BACKENDS)


# --------------------------------------------------------------------------- problems
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
    if spec.minimum is not None and (number < spec.minimum or (number == spec.minimum and not spec.allow_equal)):
        return f"{subject}: {shown} is too low. {allowed}"
    if spec.maximum is not None and number > spec.maximum:
        return f"{subject}: {shown} is too high. {allowed}"
    return None


def _raise_for_bad_values(settings: Mapping[str, Any]) -> None:
    for key in HEIGHT_KEYS:
        text = _value_problem(FIELD_SPECS[key], settings[key])
        if text:
            raise InvalidInputError(text)


def _named(settings: Mapping[str, Any], keys: Sequence[str] = HEIGHT_KEYS) -> str:
    """``"source height 25 m, target height 12 m"`` for the heights that are set."""
    words = {HEIGHT_KEYS[0]: "source height", HEIGHT_KEYS[1]: "target height"}
    return ", ".join(f"{words[key]} {_shown(settings[key])} m" for key in keys if settings[key] is not None)


def gpu_reason(backend: str | None) -> str:
    """Why the heights cannot apply with this backend (no GPU solver takes them)."""
    return (f"The GPU backends ({', '.join(gpu_backends())}) take no connection heights and no end-span rule "
            f"'{ENFORCE_RULE}': their search has no end-span check. The backend is {backend}: choose a CPU backend "
            "(cython or cython_parallel) in the Backend field.")


def _gpu_causes(settings: Mapping[str, Any], eff: Effect) -> list[str]:
    causes = []
    if eff.enforce:
        causes.append(f"the end-span rule '{ENFORCE_RULE}'")
    if eff.heights_act:
        named = _named(settings, [k for k, acts in zip(HEIGHT_KEYS, (eff.source_acts, eff.target_acts)) if acts])
        causes.append(f"a connection height ({named})")
    return causes


def problems(raw: Any, *, backend: str | None = "cython", profile: Any = None,
             dem: Any = None) -> list[tuple[str, str]]:
    """Everything the library would refuse about these end settings, as ``(key, message)``; ``[]`` when it can run.

    In this order: heights that are not allowed values (one per height), a rule value that is none of the four;
    then, only when those are usable, the one combination the library refuses by backend: a GPU backend with the
    rule ``"enforce"`` (and span limits) or with a height the search would use (a DEM, the clearance data of the
    profile and span limits). The refusal is judged on the same conditions as the finder
    (:func:`effect`), so a run the library would take is never refused here.

    A height that would have no effect (no DEM, no clearance data, no span limits) is no problem: see
    :func:`notes`. The size of the exact-bin search is judged by :func:`bins_problem`, which needs the raster.
    ``backend`` is the backend the run will use (after the experimental gate); ``profile`` is the profile dict
    (or an ``InfrastructureProfile``); ``dem`` is the terrain model of the run (path or None).
    """
    settings = normalise(raw)
    out = [(key, text) for key in HEIGHT_KEYS if (text := _value_problem(FIELD_SPECS[key], settings[key]))]
    facts = profile_facts(profile)
    if not isinstance(facts.rule, str) or facts.rule not in TERMINAL_SPAN_RULES:
        out.append(("rule", f"End-span rule (terminal_span_rule): {facts.rule!r} is not one of "
                            f"{', '.join(TERMINAL_SPAN_RULES)}. Choose one of the four rules."))
    if out:
        return out
    eff = effect(settings, profile=profile, dem=dem)
    if (backend or "cython") in gpu_backends():
        causes = _gpu_causes(settings, eff)
        if causes:
            out.append(("backend", f"Backend: {backend} cannot run {' and '.join(causes)}. {gpu_reason(backend)}"))
    return out


# --------------------------------------------------------------------------- support, notes, view
def support(*, backend: str | None = "cython", profile: Any = None) -> Support:
    """Whether the connection heights can apply (``reason`` is the text of the disabled inputs).

    Disabled without span limits (the profile places no towers, so there is no tower for the height to act on) and
    on a GPU backend (none takes a height). Without a terrain model or without the clearance data the inputs stay
    enabled: the height is then reported and the status line says so (:func:`notes`).
    """
    if not profile_facts(profile).spans_on:
        return Support(False, "The profile places no towers (min_span_m and max_span_m are not both set), so a "
                              "connection height has no tower to act on. Set both span limits in the profile form.")
    if (backend or "cython") in gpu_backends():
        return Support(False, gpu_reason(backend))
    return Support(True)


class RuleSupport(NamedTuple):
    """Whether the values of ``terminal_span_rule`` can apply, and why not."""

    enabled: bool                    # False: no value can apply (the select is disabled)
    reason: str                      # why the select is disabled ("" when enabled)
    refused: tuple[str, ...]         # values the library refuses with this backend (``()`` or ``("enforce",)``)
    refused_reason: str              # why those values are refused ("" when none)


def rule_support(profile: Any = None, backend: str | None = "cython") -> RuleSupport:
    """Whether the end-span rule can apply for this profile and backend.

    Without span limits the profile places no towers, so no rule has anything to decide (select disabled).
    On a GPU backend the library refuses ``"enforce"`` (the other three values work).
    """
    facts = profile_facts(profile)
    if not facts.spans_on:
        return RuleSupport(False, "The profile places no towers (min_span_m and max_span_m are not both set), so "
                                  "the end-span rule has nothing to decide.", (), "")
    if (backend or "cython") in gpu_backends():
        return RuleSupport(True, "", (ENFORCE_RULE,),
                           f"The library refuses the rule '{ENFORCE_RULE}' on the GPU backend {backend}: its search "
                           "has no end-span check. Choose a CPU backend or another rule.")
    return RuleSupport(True, "", (), "")


def no_effect_reasons(profile: Any = None, dem: Any = None) -> list[str]:
    """Why a connection height would not reach the search (empty when it would)."""
    facts = profile_facts(profile)
    reasons = []
    if not _has_dem(dem):
        reasons.append("there is no terrain model (DEM)")
    if not facts.clearance_on:
        reasons.append("the profile lacks tower height, conductor weight, conductor tension or minimum clearance")
    if not facts.spans_on:
        reasons.append("the profile places no towers (it has no span limits)")
    return reasons


#: what the DEM and the DSM of the constrained card do (``ConstrainedPathFinder``), in plain words: the one text the
#: clearance line of the profile form, the help of the two inputs and the notes below agree with
DEM_ROLE = ("The terrain model (DEM) gives the ground height: the search uses it for slopes and, with a profile "
            "that has tower height, conductor weight, conductor tension and ground clearance, for the clearance "
            "check, which is off without a DEM.")
DSM_ROLE = ("The surface model (DSM) is optional and is read only together with a DEM: DSM minus DEM gives the "
            "height of trees and buildings, which the clearance check then counts as well. With a DEM alone the "
            "check counts the bare ground; a DSM without a DEM is ignored.")


def surface_model_notes(profile: Any = None, dem: Any = None, dsm: Any = None) -> list[str]:
    """What the finder does with a DSM that the run was given, when that is not what the user may expect.

    The library takes a DSM only together with a DEM (obstacle height = DSM minus DEM) and only the clearance
    check reads it. It neither warns nor refuses in the other cases, so the GUI says it:

    * a DSM without a DEM is ignored;
    * a DSM with a DEM, but without the clearance data of the profile, is not used (nothing reads it).

    No DSM, or a DSM the check will use: no note (a DEM without a DSM is a normal run, see :data:`DSM_ROLE`).
    """
    if not _has_dem(dsm):
        return []
    if not _has_dem(dem):
        return ["The surface model (DSM) is ignored: the library reads it only together with a terrain model (DEM), "
                "as the height of trees and buildings (DSM minus DEM). Enter a DEM as well, or clear the DSM."]
    if not profile_facts(profile).clearance_on:
        return ["The surface model (DSM) is not used: only the clearance check reads it, and the profile lacks tower "
                "height, conductor weight, conductor tension or ground clearance, so the check is off."]
    return []


def notes(raw: Any, *, backend: str | None = "cython", profile: Any = None, dem: Any = None) -> list[str]:
    """Facts worth knowing about these settings (the run is not refused).

    * a set height that would have no effect: "reported only, the route does not change", with the reasons;
    * a rule other than the default in a profile without span limits: it changes nothing;
    * a run that needs exact span bins: it takes more memory and time than a default run.
    """
    del backend                      # the refusals are problems, not notes
    settings = normalise(raw)
    facts = profile_facts(profile)
    eff = effect(settings, profile=profile, dem=dem)
    out = []
    set_keys = [key for key in HEIGHT_KEYS if settings[key] is not None and _as_number(settings[key]) is not None]
    if set_keys and not eff.clearance:
        reasons = no_effect_reasons(profile, dem)
        out.append(f"The {_named(settings, set_keys)} is reported only, the route does not change: "
                   f"{' and '.join(reasons)}. The height is still shown as the height of the end tower.")
    if facts.rule in TERMINAL_SPAN_RULES and facts.rule != DEFAULT_RULE and not facts.spans_on:
        out.append(f"The end-span rule '{facts.rule}' changes nothing: the profile places no towers.")
    if eff.exact_bins:
        out.append("This run searches with exact span bins (as short as one step) so that no feasible route is "
                   "hidden: it needs more memory and time than a default run, and the library refuses a window "
                   "whose search would not fit.")
    return out


def summary(raw: Any) -> str:
    """One line: what the heights make of the run."""
    settings = normalise(raw)
    if is_default(settings):
        return ("No connection heights are set: the line ends behave as before (the profile's terminal tower "
                "height is reported only).")
    return f"Connection heights: {_named(settings)} (metres above the ground at the end point)."


def view(raw: Any, *, backend: str | None = "cython", profile: Any = None, dem: Any = None) -> View:
    """The status line and the disabled state for the current heights, backend, profile and terrain model.

    Problems come first (they are what a run would refuse with); without problems the summary and the notes. The
    inputs are disabled when the heights cannot apply (:func:`support`), and that reason is the status line too, so
    it is visible and not only a hover text.
    """
    sup = support(backend=backend, profile=profile)
    found = problems(raw, backend=backend, profile=profile, dem=dem)
    if found:
        return View(problems_text(found), True, sup.enabled, sup.reason)
    extra = notes(raw, backend=backend, profile=profile, dem=dem)
    if not sup.enabled:
        return View("\n".join([sup.reason, *extra]), False, False, sup.reason)
    return View("\n".join([summary(raw), *extra]), False, True, "")


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
        title = f"{label}: value not usable"
        fix = f"Correct the {FIELD_SPECS[key].label} field in the Line ends block, or clear it."
    elif key == "rule":
        title = f"{label}: value not usable"
        fix = "Choose one of the four end-span rules in the profile form."
    elif key == "backend":
        title = "Line ends: this backend cannot run them"
        fix = ("Choose a CPU backend (cython or cython_parallel), or clear the connection heights and set the "
               "end-span rule to remove, remove_if_valid or keep.")
    else:
        title = "Line ends: the exact search is too large"
        fix = ("Use a coarser raster, a smaller search buffer or a smaller longest span; or set the end-span rule "
               "to keep or remove_if_valid and clear the connection heights (they keep the coarse bins).")
    return Notice(severity="error", title=title, meaning=message, impact=impact, fix=fix, focus_id=tab,
                  focus_control=(controls or {}).get(key)).to_dict()


# --------------------------------------------------------------------------- what reaches the library
def library_kwargs(raw: Any, segment_index: int = 0, n_segments: int = 1) -> dict[str, float | None]:
    """The ``find_route`` keywords of one segment, or ``{}`` for classic constrained routing.

    ``{}`` when no height is set: the library is then called as it always was. Otherwise BOTH keywords, so that an
    end that is not the segment's own is passed as an explicit ``None`` ("not set for this call", the
    ``find_route`` semantics; leaving a keyword out would mean the finder's own setting): the source height only
    to the first segment, the target height only to the last one, ``None`` for every end at a waypoint. A run
    without waypoints is one segment (``n_segments=1``) that gets both.

    Raises ``InvalidInputError`` for a value that :func:`problems` would name (the callers check it first) and for
    a segment index outside the run.
    """
    settings = normalise(raw)
    _raise_for_bad_values(settings)
    segment = segment_settings(settings, segment_index, n_segments)
    if is_default(settings):
        return {}
    return {key: None if segment[key] is None else float(segment[key]) for key in HEIGHT_KEYS}


# --------------------------------------------------------------------------- the exact-bin bound
class BinEstimate(NamedTuple):
    """The size of the exact-bin search of one window (the quantities ``_exact_span_bins`` of the finder uses)."""

    rows: int
    cols: int
    n_directions: int
    shortest_step_m: float
    steps_per_span: int
    n_bins: int
    bin_size_m: float
    n_heights: int
    states: int
    max_states: int                  # ConstrainedPathFinder.EXACT_BINS_MAX_STATES
    max_bins: int                    # ConstrainedPathFinder.EXACT_BINS_MAX_COUNT
    memory: Any = None               # the finder's ExactBinsMemory (kernel, form, nbytes, detail) of this window
    max_bytes: int = 0               # ConstrainedPathFinder.EXACT_BINS_MAX_BYTES
    max_steps: int = 0               # ConstrainedPathFinder.EXACT_BINS_MAX_STEPS

    @property
    def nbytes(self) -> int:
        """The finder's memory estimate of the kernel it would call, in bytes (0 when none was made)."""
        return int(self.memory.nbytes) if self.memory is not None else 0

    @property
    def exceeded(self) -> tuple[str, ...]:
        """The bounds the library would refuse this window on, in the order the finder checks them.

        Each is one of ``"bins"`` (more bins than the kernels' index holds), ``"steps"`` (more steps per longest
        span than the bins stay exact for), ``"states"`` and ``"memory"``; ``()`` when the window fits.
        """
        out = []
        if self.n_bins > self.max_bins:
            out.append("bins")
        if self.steps_per_span > self.max_steps:
            out.append("steps")
        if self.states > self.max_states:
            out.append("states")
        if self.nbytes > self.max_bytes:
            out.append("memory")
        return tuple(out)

    @property
    def exceeds(self) -> bool:
        """True when the library would refuse this window on any bound (bins, steps, states or memory)."""
        return bool(self.exceeded)


#: the backends whose finder builds the DIRECTED neighbourhood (``PathFinder._steps_from_neighborhood``)
_DIRECTED_BACKENDS = ("cython", "raster_gpu", "cugraph")


@lru_cache(maxsize=32)
def _steps(neighborhood: str, directed: bool):
    from pyorps.utils.neighborhood import get_neighborhood_steps

    return get_neighborhood_steps(neighborhood, directed=directed)


def estimate_exact_bins(rows: int, cols: int, *, max_span_m: float, cell_size_m: float = 1.0,
                        neighborhood: str = "r2", backend: str | None = "cython", n_heights: int = 1,
                        clearance: bool = False, has_dem: bool = False) -> BinEstimate:
    """The exact-bin state space of a ``rows`` x ``cols`` window, by the finder's own formula.

    Mirrors ``ConstrainedPathFinder._exact_span_bins``: the shortest step is the smallest step distance of the
    neighbourhood (directed for ``cython`` and ``raster_gpu``, the minimal set for ``cython_parallel``), the bins
    are ``steps_per_span = max(1, ceil(max_span_m / shortest))`` and ``K + 1`` of them, one bin
    ``shortest * K / (K + 0.25)`` long, and the states are ``cells x directions x bins x height classes``. The
    limits (states, bins, bytes, steps per span) are read from the finder's class attributes and the memory is the
    finder's own ``ConstrainedPathFinder.exact_bins_memory``: nothing is copied. ``clearance`` (the clearance
    check is on: a DEM and the profile's clearance data) and ``has_dem`` (a terrain model is given) choose the
    kernel the memory estimate is for, as in the finder.

    The estimate is EXACT when ``rows`` and ``cols`` are the window the finder will use (see
    :func:`window_shape`, which uses the finder's own window function) and ``n_heights`` is what it will use (see
    :func:`bins_problem`). It cannot be made before the run when the window is unknown (a raster that is not a
    file path, a point outside the raster); then the library's ``InvalidInputError`` is the answer and the page
    shows it unchanged.
    """
    directed = (backend or "cython") in _DIRECTED_BACKENDS
    steps = _steps(str(neighborhood), directed)
    shortest = float(min(math.sqrt(dr**2 + dc**2) * float(cell_size_m) for dr, dc in steps))
    steps_per_span = max(1, math.ceil(float(max_span_m) / shortest))
    bin_size = shortest * steps_per_span / (steps_per_span + 0.25)
    n_bins = steps_per_span + 1
    states = int(rows) * int(cols) * len(steps) * n_bins * int(n_heights)
    lib = _library()
    memory = lib.exact_bins_memory(int(rows) * int(cols), len(steps), n_bins, int(n_heights),
                                   backend=backend or "cython", clearance=bool(clearance), has_dem=bool(has_dem))
    return BinEstimate(int(rows), int(cols), len(steps), shortest, steps_per_span, n_bins, bin_size, int(n_heights),
                       states, int(lib.EXACT_BINS_MAX_STATES), int(lib.EXACT_BINS_MAX_COUNT), memory,
                       int(lib.EXACT_BINS_MAX_BYTES), int(lib.EXACT_BINS_MAX_STEPS))


def window_shape(raster_source: Any, source: Sequence[float], target: Sequence[float],
                 search_buffer_m: float | None = None) -> tuple[int, int, float] | None:
    """``(rows, cols, cell_size_m)`` of the search window of one segment, or None when it cannot be known.

    The window is the finder's: the buffer (automatic when blank, as ``run_constrained`` does) around the
    segment's two points, turned into pixels by ``RasterHandler.window_from_bounds`` from the raster header alone.
    ``raster_source`` must be a file path; anything else (a dataset object, a file that cannot be opened) gives
    None.
    """
    import os

    from pyorps.raster.handler import RasterHandler

    from .routing import default_search_buffer

    if not isinstance(raster_source, (str, os.PathLike)):
        return None
    try:
        import rasterio
        from shapely.geometry import LineString

        buffer_m = search_buffer_m
        if not buffer_m or buffer_m <= 0:
            buffer_m = default_search_buffer([tuple(source)], [tuple(target)])
        with rasterio.open(os.fspath(raster_source)) as src:
            transform, shape = src.transform, (src.height, src.width)
        geometry = LineString([tuple(source), tuple(target)]).buffer(distance=buffer_m, quad_segs=32)
        window = RasterHandler.window_from_bounds(geometry.bounds, transform, shape,
                                                  points=[tuple(source), tuple(target)])
        return int(window.height), int(window.width), abs(float(transform.a))
    except Exception:  # pylint: disable=broad-exception-caught  # an estimate must never stop a run
        return None


def bins_message(est: BinEstimate, why: str, *, buffer_m: float | None = None, segment: str = "") -> str:
    """The sentence of the exact-bin problem: the numbers, the limit and what to change."""
    prefix = f"{segment}: " if segment else ""
    buffer_text = f" (now {format_number(buffer_m)} m)" if buffer_m else ""
    rules = ("or set the end-span rule to \"Remove it\", \"Remove it only if every span stays valid\" or \"Keep it\" "
             "and clear the connection heights (they keep the coarse bins).")
    span_advice = (f"Use a coarser raster (the bins are as short as the shortest step) or a shorter longest span "
                   f"(max_span_m); {rules}")
    size_advice = (f"Use a coarser raster (the bins are as short as the shortest step), a smaller search buffer"
                   f"{buffer_text}, or a shorter longest span (max_span_m); {rules}")
    gib = 1024 ** 3
    steps_text = (f"the longest span is {est.steps_per_span:,} steps of the shortest step "
                  f"{est.shortest_step_m:.4g} m")
    space_text = (f"{est.states:,} states ({est.rows:,} x {est.cols:,} cells x {est.n_directions} directions x "
                  f"{est.n_bins:,} span bins of {est.bin_size_m:.4g} m x {est.n_heights} height classes)")
    exceeded = est.exceeded or ("states",)
    first = exceeded[0]
    if first == "bins":
        text = (f"{why} needs exact span bins, but {steps_text}, so {est.n_bins:,} span bins of "
                f"{est.bin_size_m:.4g} m are needed and the search holds at most {est.max_bins:,}. {span_advice}")
    elif first == "steps":
        text = (f"{why} needs exact span bins, but {steps_text}, above the limit of {est.max_steps:,} steps per "
                f"longest span (above it the bins no longer tell neighbouring spans apart). {span_advice}")
    elif first == "states":
        text = (f"{why} needs exact span bins, which make the search {space_text}, above the limit of "
                f"{est.max_states:,} states the library accepts. {size_advice}")
    else:
        memory = est.memory
        text = (f"{why} needs exact span bins, and the search kernel {memory.kernel} ({memory.form} form) would then "
                f"need about {est.nbytes / gib:.1f} GiB of memory, above the limit of {est.max_bytes / gib:.1f} GiB "
                f"the library accepts. The search has {space_text}. {size_advice}")
    others = exceeded[1:]
    if others:
        named = {"bins": f"{est.n_bins:,} span bins (limit {est.max_bins:,})",
                 "steps": f"{est.steps_per_span:,} steps per longest span (limit {est.max_steps:,})",
                 "states": f"{est.states:,} states (limit {est.max_states:,})",
                 "memory": f"{est.nbytes / gib:.1f} GiB (limit {est.max_bytes / gib:.1f} GiB)"}
        text += f" It is also above the limit of: {'; '.join(named[name] for name in others)}."
    return f"{prefix}Exact span bins: {text}"


def _why(eff: Effect) -> str:
    reasons = []
    if eff.enforce:
        reasons.append(f"the end-span rule '{ENFORCE_RULE}'")
    if eff.heights_act:
        reasons.append("a connection height")
    return " and ".join(reasons)


def bins_problem(raw: Any, *, raster_source: Any, points: Sequence[Sequence[float]], profile: Any = None,
                 backend: str | None = "cython", neighborhood: str = "r2", dem: Any = None,
                 search_buffer_m: float | None = None) -> tuple[str, str] | None:
    """The exact-bin problem of a run, or None when the search fits (or the size cannot be known before the run).

    Decides per segment of ``points`` (source, waypoints..., target) whether the run needs exact span bins
    (:func:`segment_settings`, :func:`effect`: ``"enforce"`` on every segment, a height on the end segment it
    belongs to), reads the window of that segment (:func:`window_shape`) and compares the state space
    (:func:`estimate_exact_bins`) with the finder's four limits (bins, steps per longest span, states and the
    memory of the kernel the run would call). The first segment above one is returned as ``("bins", message)``
    with the numbers, which bound it is and the advice (coarser raster, smaller search buffer, shorter span).

    The estimate uses the finder's window function and formula, so it agrees with the finder's own check. It
    cannot be made when ``raster_source`` is not a file path or the profile's longest span is not a number: the
    function returns None and the library's ``InvalidInputError`` (the same numbers) reaches the notice
    unchanged. Bad values (see :func:`problems`) give None as well: they are reported first.
    """
    settings = normalise(raw)
    facts = profile_facts(profile)
    max_span = _as_number(facts.max_span_m)
    if any(_value_problem(FIELD_SPECS[k], settings[k]) for k in HEIGHT_KEYS) or max_span is None or max_span <= 0:
        return None
    n_segments = len(points) - 1
    for index in range(max(0, n_segments)):
        segment = segment_settings(settings, index, n_segments)
        eff = effect(segment, profile=profile, dem=dem)
        if not eff.exact_bins:
            continue
        shape = window_shape(raster_source, points[index], points[index + 1], search_buffer_m)
        if shape is None:
            continue
        rows, cols, cell = shape
        n_heights = facts.n_heights if eff.clearance else 1
        est = estimate_exact_bins(rows, cols, max_span_m=max_span, cell_size_m=cell, neighborhood=neighborhood,
                                  backend=backend, n_heights=n_heights, clearance=eff.clearance,
                                  has_dem=_has_dem(dem))
        if est.exceeds:
            label = f"Segment {index + 1} of {n_segments}" if n_segments > 1 else ""
            return ("bins", bins_message(est, _why(eff), buffer_m=search_buffer_m, segment=label))
    return None


def run_problems(raw: Any, *, raster_source: Any = None, points: Sequence[Sequence[float]] | None = None,
                 backend: str | None = "cython", profile: Any = None, dem: Any = None,
                 neighborhood: str = "r2", search_buffer_m: float | None = None) -> list[tuple[str, str]]:
    """:func:`problems`, plus the exact-bin problem (:func:`bins_problem`) when raster and points are given.

    The bin problem is judged only when nothing else is wrong: a refused combination or a bad value is shown
    first, and the size of a search that cannot run anyway would only add noise.
    """
    found = problems(raw, backend=backend, profile=profile, dem=dem)
    if found or raster_source is None or points is None:
        return found
    extra = bins_problem(raw, raster_source=raster_source, points=points, profile=profile, backend=backend,
                         neighborhood=neighborhood, dem=dem, search_buffer_m=search_buffer_m)
    return [extra] if extra else []
