"""
PYORPS GUI: error/warning registry and capture (R14, Section 21).

Every failure mode is translated into a :class:`Notice` — a plain-language
notification box answering *what happened, what it means, how it affects your
result, and how to fix it* — instead of leaking raw exceptions or swallowing
warnings. Callbacks never call a service directly; they call it through
:func:`guard`, which captures both exceptions and ``warnings.warn`` and
appends translated notices.

The registry is pure data: unit tests assert every pyorps exception class maps
to the right box and that no class is left to the generic fallback.
"""
from __future__ import annotations

import threading
import traceback
import uuid
import warnings
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from pyorps.core import exceptions as pex

SEVERITIES = ("error", "warning", "info", "success")

#: the browser store keeps this many notices at most (every callback sends the
#: whole store back and forth, so an unbounded history slows the app down)
MAX_STORED_NOTICES = 20
#: longest ``details`` text a notice carries into the browser store
MAX_DETAILS_CHARS = 600
#: how many full (long) ``details`` texts are kept server-side for the log
_FULL_DETAILS_KEPT = 200
_full_details: OrderedDict[str, str] = OrderedDict()
_full_details_lock = threading.Lock()


def _short_details(text: str, limit: int = MAX_DETAILS_CHARS) -> str:
    """``text`` shortened for the browser: a traceback becomes its last line."""
    text = (text or "").strip()
    if "Traceback (most recent call last)" in text:
        tail: list[str] = []
        for line in reversed(text.splitlines()):
            if not line.strip() or line.startswith((" ", "\t")):
                break                         # a frame / code line: stop
            tail.append(line)
        text = "\n".join(reversed(tail)) or text.splitlines()[-1]
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _remember_full_details(notice_id: str, text: str) -> None:
    with _full_details_lock:
        _full_details[notice_id] = text
        _full_details.move_to_end(notice_id)
        while len(_full_details) > _FULL_DETAILS_KEPT:
            _full_details.popitem(last=False)


def full_details(notice_id: str | None) -> str | None:
    """The untruncated ``details`` of a notice made by :meth:`Notice.to_dict`.

    None when the notice had short details (nothing was cut) or is unknown.
    """
    if not notice_id:
        return None
    with _full_details_lock:
        return _full_details.get(notice_id)


def cap_notices(notices: list[dict] | None,
                limit: int = MAX_STORED_NOTICES) -> list[dict]:
    """The newest ``limit`` notices (the store is history, not an archive)."""
    kept = list(notices or [])
    return kept[-limit:] if limit > 0 else []


@dataclass
class Notice:
    """One notification box (Section 21.2). JSON-serializable via to_dict()."""

    severity: str          # "error" | "warning" | "info" | "success"
    title: str             # short, plain language
    meaning: str = ""      # what it means
    impact: str = ""       # how it affects the result
    fix: str = ""          # how to fix it (imperative)
    details: str = ""      # raw exception/warning text + traceback (collapsible)
    focus_id: str | None = None       # tab the "Go fix" button switches to
    focus_control: str | None = None  # component id it scrolls to + flashes
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_dict(self) -> dict:
        """The JSON-safe dict for the browser store.

        ``details`` is shortened to :data:`MAX_DETAILS_CHARS` (for a traceback:
        the exception line only); the full text is kept server-side, keyed by
        the notice id, so the log file still gets the whole traceback
        (:func:`full_details`).
        """
        data = asdict(self)
        full = self.details or ""
        short = _short_details(full)
        if short != full:
            _remember_full_details(self.id, full)
            data["details"] = short
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Notice:
        return cls(**{k: data.get(k) for k in
                      ("severity", "title", "meaning", "impact", "fix",
                       "details", "focus_id", "focus_control", "id")
                      if data.get(k) is not None})


# ---------------------------------------------------------------------------
# Exception registry: (class, message-substring | None, builder). Matching is
# by isinstance + optional case-insensitive substring, most specific first.
# ---------------------------------------------------------------------------
_Builder = Callable[[BaseException], Notice]


def _exc_details(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc,
                                              exc.__traceback__))


def _n(severity: str, title: str, meaning: str, impact: str, fix: str,
       focus_id: str | None = None) -> Callable[[BaseException], Notice]:
    def build(exc: BaseException) -> Notice:
        return Notice(severity=severity, title=title, meaning=meaning,
                      impact=impact, fix=fix, details=_exc_details(exc),
                      focus_id=focus_id)
    return build


def _wfs_layer_notice(exc: BaseException) -> Notice:
    return Notice(
        severity="error",
        title="Layer not found on the WFS server",
        meaning="That layer name doesn't exist on this WFS "
                "(and no close match was found).",
        impact="No data was loaded.",
        fix="Pick a layer from the server's list, or fix the spelling.",
        details=_exc_details(exc), focus_id="tab-data")


class ProjectLoadError(Exception):
    """A saved project could not be opened; the message names the failed asset.

    Raised by ``project_io.load_project`` BEFORE the current session is touched,
    so the user still has everything they had.
    """

    def __init__(self, message: str, *, asset: str = "", layer: str = "") -> None:
        super().__init__(message)
        self.asset = asset
        self.layer = layer


class ProjectLayersSkippedWarning(UserWarning):
    """Layers were left out while saving / opening a project (see project_io)."""


def _project_load_notice(exc: BaseException) -> Notice:
    return Notice(
        severity="error", title="Project could not be opened",
        meaning=str(exc),
        impact="Nothing was changed: your current session is exactly as it was.",
        fix="Restore or re-save the named file next to project.json, then open "
            "the project again.",
        details=_exc_details(exc), focus_id="tab-data")


class NothingToUndoError(Exception):
    """"Undo last edit" was asked for a route that has no edit to take back."""


def _nothing_to_undo_notice(exc: BaseException) -> Notice:
    return Notice(
        severity="info", title="Nothing to undo",
        meaning=str(exc) or "This route has not been edited since it was "
                            "computed, opened or last undone.",
        impact="Nothing changed.",
        fix="Edit the route first; 'Undo last edit' then brings back the "
            "version before that edit (one step only).",
        focus_id="tab-routes")


#: ordered list of (exception class, lowercase substring or None, builder)
EXCEPTION_RULES: list[tuple[type, str | None, _Builder]] = [
    (NothingToUndoError, None, _nothing_to_undo_notice),
    # ---- project files -----------------------------------------------------
    (ProjectLoadError, None, _project_load_notice),
    # ---- data loading / WFS (most specific first) --------------------------
    (pex.WFSConnectionError, None, _n(
        "error", "Can't reach the WFS server",
        "The server didn't respond (down, wrong URL, or you're offline).",
        "No data was loaded from this source.",
        "Check the URL and your internet connection; try again later or load " +
        "a local file instead.", "tab-data")),
    (pex.WFSLayerNotFoundError, None, _wfs_layer_notice),
    # empty answer for THIS area is benign (layer just doesn't cover it) — a
    # warning, not an error. Matched by the message data_io raises; must sit
    # BEFORE the generic "couldn't be read" rule below.
    (pex.WFSResponseParsingError, "returned no data", _n(
        "warning", "No data here for this layer",
        "The WFS answered, but this layer has no features in your study area " +
        "(e.g. a Hessen layer with an area outside Hessen).",
        "No data was loaded from this layer.",
        "Move/enlarge the study area to where the layer has data, or pick a " +
        "layer that covers your region.", "tab-data")),
    (pex.WFSResponseParsingError, "returned no features", _n(
        "warning", "No data here for this layer",
        "The WFS answered, but this layer has no features in your study area " +
        "(e.g. a Hessen layer with an area outside Hessen).",
        "No data was loaded from this layer.",
        "Move/enlarge the study area to where the layer has data, or pick a " +
        "layer that covers your region.", "tab-data")),
    (pex.WFSResponseParsingError, None, _n(
        "error", "The server's response couldn't be read",
        "The WFS returned unexpected or invalid XML.",
        "No data was loaded.",
        "Try a smaller area or a different WFS version; the server may be " +
        "misconfigured — contact the data provider.", "tab-data")),
    (pex.WFSError, None, _n(
        "error", "WFS request failed",
        "Something went wrong talking to the WFS server.",
        "No data was loaded from this source.",
        "Check the URL and layer name, then retry; details below.",
        "tab-data")),
    (FileNotFoundError, None, _n(
        "error", "File not found",
        "The path doesn't exist or the file was moved.",
        "The layer/raster/profile wasn't loaded.",
        "Re-select the file; check it wasn't moved or renamed.")),
    # ---- cost model ---------------------------------------------------------
    (pex.InvalidSourceError, None, _n(
        "error", "Cost source not supported",
        "The cost input isn't a table file or a valid mapping.",
        "The cost model was not built.",
        "Load a CSV/JSON/XLSX cost table, or edit costs directly in the " +
        "table.", "tab-cost")),
    (pex.FileLoadError, None, _n(
        "error", "Couldn't read the cost file",
        "The file's encoding, delimiter or format couldn't be parsed.",
        "The cost model was not loaded.",
        "Confirm it's a valid CSV/JSON/XLSX; re-export it from the cost table " +
        "and retry.", "tab-cost")),
    (pex.FormatError, None, _n(
        "error", "Cost table format is wrong",
        "The table is missing the cost column or a category column.",
        "Costs can't be mapped to the data.",
        "Ensure one numeric cost column and at least one category column; " +
        "select the feature column(s).", "tab-cost")),
    (pex.CostAssumptionsError, None, _n(
        "error", "Cost model problem",
        "The cost assumptions couldn't be processed.",
        "The cost model was not applied.",
        "Check the cost table and feature selection; details below.",
        "tab-cost")),
    (pex.NoSuitableColumnsError, None, _n(
        "warning", "Couldn't auto-detect a category column",
        "No attribute column looked suitable for cost categories.",
        "The cost table wasn't auto-seeded.",
        "Pick the feature column(s) manually from the dropdown.", "tab-cost")),
    (pex.ColumnAnalysisError, None, _n(
        "error", "Couldn't analyze the attribute columns",
        "Analyzing the dataset's columns for cost categories failed.",
        "The cost table wasn't auto-seeded.",
        "Pick the feature column(s) manually; details below.", "tab-cost")),
    (pex.FeatureColumnError, None, _n(
        "error", "Feature-column detection failed",
        "The dataset's columns couldn't be analyzed.",
        "The cost table wasn't auto-seeded.",
        "Pick the feature column(s) manually from the dropdown.", "tab-cost")),
    # ---- routing ------------------------------------------------------------
    (pex.NoPathFoundError, None, _n(
        "error", "No route could be found",
        "The source/target is blocked, outside the search window, or " +
        "separated by forbidden areas.",
        "No route was produced for this pair.",
        "Increase the search buffer, move the point off a forbidden cell, or " +
        "check the cost raster for a barrier.", "tab-routes")),
    (pex.PairwiseError, None, _n(
        "error", "Source and target counts don't match",
        "Pairwise mode needs exactly one target per source.",
        "No routes were computed.",
        "Add/remove points to equal counts, or turn off pairwise mode.",
        "tab-routes")),
    (pex.AlgorithmNotImplementedError, None, _n(
        "error", "That algorithm isn't available on this backend",
        "The selected algorithm has no implementation in the selected " +
        "graph backend.",
        "No route was computed.",
        "Pick a valid algorithm for this backend — the dropdown filters " +
        "them.", "tab-routes")),
    (pex.RasterShapeError, None, _n(
        "error", "Unexpected raster shape",
        "The cost raster isn't 2D (n, m) or 3D (n, m, 2).",
        "Routing can't run on this raster.",
        "Re-load the raster; if it persists the file may be malformed.",
        "tab-raster")),
    # ---- ValueError variants (matched by message substring) ----------------
    (ValueError, "unsupported vector data source", _n(
        "error", "Unsupported file type",
        "This extension isn't a vector format pyorps can read.",
        "The file wasn't loaded.",
        "Use a supported vector format (.shp/.geojson/.gpkg/.gml/.kml).",
        "tab-data")),
    (ValueError, "unsupported raster data source", _n(
        "error", "Unsupported file type",
        "This extension isn't a raster format pyorps can read.",
        "The file wasn't loaded.",
        "Use a supported raster format (.tif/.jp2/.img/.bil/.dem).",
        "tab-data")),
    (ValueError, "unable to determine appropriate dataset type", _n(
        "error", "Unsupported file type",
        "pyorps couldn't tell what kind of dataset this is.",
        "The file wasn't loaded.",
        "Use a supported vector (.shp/.geojson/.gpkg) or raster (.tif) " +
        "format.", "tab-data")),
    (ValueError, "hostname", _n(
        "error", "That WFS URL looks invalid",
        "The URL is malformed or has no host.",
        "Can't connect to the server.",
        "Paste the full https://... endpoint including the host.",
        "tab-data")),
    (ValueError, "must not be none", _n(
        "warning", "Set a source and a target first",
        "One or both routing points are missing.",
        "There is nothing to route yet.",
        "Place a source and a target on the map (or type coordinates).",
        "tab-routes")),
    (ValueError, "unsupported graph api", _n(
        "error", "Unknown routing backend",
        "An invalid graph backend was selected.",
        "Routing can't run.",
        "Choose a listed backend (cython / networkit / raster_gpu / ...).",
        "tab-routes")),
    (ValueError, "unknown angle cost function", _n(
        "error", "Unknown angle-cost function",
        "The infrastructure profile names an angle-cost function that " +
        "doesn't exist.",
        "Constrained routing can't run.",
        "Pick linear, quadratic or piecewise.", "tab-routes")),
    # ---- OpenStreetMap / Overpass (matched by message substring) -----------
    (ValueError, "overpass", _n(
        "warning", "OpenStreetMap server is busy",
        "The public Overpass API rate-limited or couldn't be reached " +
        "(the exact HTTP code is in Details).",
        "No OSM features were loaded.",
        "Wait a moment and retry, draw a smaller study area, or narrow the " +
        "tag (e.g. highway=primary instead of highway=*).", "tab-data")),
    (ValueError, "no osm features", _n(
        "warning", "No OSM features in this area",
        "Overpass returned nothing for that feature type in the study area.",
        "No OSM layer was loaded.",
        "Try a different feature type or a larger study area.", "tab-data")),
    (MemoryError, None, _n(
        "error", "Out of memory",
        "The operation needed more RAM than is available (raster too large " +
        "or search window too big).",
        "The operation failed part-way.",
        "Increase the resolution value (coarser), shrink the study area, or " +
        "set a smaller search buffer.", "tab-raster")),
    # ---- pyorps base fallback ----------------------------------------------
    (pex.PyorpsError, None, _n(
        "error", "pyorps reported an error",
        "A pyorps operation failed (details below).",
        "The action didn't complete.",
        "Check the inputs for this step and retry; details below.")),
]


def _generic_exception_notice(exc: BaseException) -> Notice:
    return Notice(
        severity="error", title="Something went wrong",
        meaning="An unexpected error occurred (details below).",
        impact="The action didn't complete.",
        fix="Try again; if it repeats, expand Details and report it.",
        details=_exc_details(exc))


def translate_exception(exc: BaseException) -> Notice:
    """Map any exception to its notification box (generic box if unmapped)."""
    msg = str(exc).lower()
    for cls, substr, builder in EXCEPTION_RULES:
        if isinstance(exc, cls) and (substr is None or substr in msg):
            return builder(exc)
    return _generic_exception_notice(exc)


# ---------------------------------------------------------------------------
# Warning registry: (category, lowercase message substring, builder(message)).
# pyorps warnings are plain UserWarnings, so matching is by message text.
# ---------------------------------------------------------------------------
def _w(severity: str, title: str, meaning: str, impact: str, fix: str,
       focus_id: str | None = None) -> Callable[[str], Notice]:
    def build(message: str) -> Notice:
        return Notice(severity=severity, title=title, meaning=meaning,
                      impact=impact, fix=fix, details=message,
                      focus_id=focus_id)
    return build


def _layers_skipped_notice(message: str) -> Notice:
    return Notice(
        severity="warning", title="Some layers were left out of the project",
        meaning=message,
        impact="Those layers are not part of the project file or the opened "
               "session; everything else was handled.",
        fix="Re-load or rebuild the listed layers (their source file is missing "
            "or empty), then save the project again.",
        details=message, focus_id="tab-data")


WARNING_RULES: list[tuple[type, str, Callable[[str], Notice]]] = [
    (ProjectLayersSkippedWarning, "", _layers_skipped_notice),
    (Warning, "crs mismatch between bbox", _w(
        "info", "Reprojected your area to the data's CRS",
        "Your study area was in a different CRS than the data.",
        "None — it was handled automatically.",
        "No action needed.")),
    (Warning, "crs mismatch between mask", _w(
        "info", "Reprojected your area to the data's CRS",
        "Your study area was in a different CRS than the data.",
        "None — it was handled automatically.",
        "No action needed.")),
    (Warning, "geographic crs", _w(
        "info", "Reprojected to a metric CRS for rasterizing",
        "The data was in lat/lon; rasterizing needs metres.",
        "The output raster is in a projected (UTM) CRS.",
        "No action needed, or set a project CRS explicitly.")),
    (Warning, "no search_space_buffer_m set", _w(
        "warning", "No search buffer set — routing the whole raster",
        "Without a buffer, the entire raster is searched.",
        "Correct results, but potentially very slow and memory-heavy (F1).",
        "Set a search-space buffer in the Routes tab.", "tab-routes")),
    (Warning, "maximum cost value", _w(
        "info", "Source/target moved to the nearest routable cell",
        "A point was on a forbidden/no-data cell, so it was snapped to the " +
        "nearest valid cell.",
        "The route starts/ends slightly off your click (shift shown in " +
        "Details).",
        "If the shift matters, move the point onto a routable area.",
        "tab-routes")),
    (Warning, "gpu", _w(
        "info", "GPU backend unavailable — using CPU",
        "CuPy / an NVIDIA GPU wasn't found, or the GPU backend fell back.",
        "Slower, but results are identical (or see Details for specifics).",
        "Keep CPU, or install CuPy with a CUDA GPU.", "tab-routes")),
    (Warning, "found no path", _w(
        "warning", "GPU constrained routing found no path",
        "The experimental GPU backend hit a limit or found nothing.",
        "No route was produced by the GPU backend.",
        "Use the CPU backend, or reduce the area/state space (F10).",
        "tab-routes")),
]


def _generic_warning_notice(message: str, category: type) -> Notice:
    return Notice(
        severity="warning", title="A step reported a warning",
        meaning="The operation completed but flagged something "
                f"({category.__name__}).",
        impact="Results may need a second look — see Details.",
        fix="Read the details; adjust the inputs if they apply.",
        details=message)


def translate_warning(w: warnings.WarningMessage) -> Notice:
    """Map a captured warning to its notification box."""
    message = str(w.message)
    lowered = message.lower()
    for category, substr, builder in WARNING_RULES:
        if issubclass(w.category, category) and substr in lowered:
            return builder(message)
    return _generic_warning_notice(message, w.category)


# ---------------------------------------------------------------------------
# Capture mechanism (Section 21.3)
# ---------------------------------------------------------------------------
#: pure-noise warnings never shown as boxes. guard's catch_warnings +
#: simplefilter("always") overrides module-level ignore filters (e.g. the
#: NoOverviewWarning filter in services.tiles), so they are re-suppressed
#: here by message substring (lowercase).
IGNORED_WARNING_SUBSTRINGS = (
    "has no overviews",                # rio-tiler NoOverviewWarning
    "dataset has no geotransform",     # rasterio NotGeoreferencedWarning
    "invalid value encountered in cast",   # numpy-internal NaN→int cast noise
)


def _is_noise(w: warnings.WarningMessage) -> bool:
    message = str(w.message).lower()
    return any(sub in message for sub in IGNORED_WARNING_SUBSTRINGS)


def _append_warnings(caught: list, notices: list[dict]) -> None:
    """Turn captured warnings into notices — ONE box per distinct warning.

    A single guarded action can raise the same warning dozens of times (e.g.
    numpy re-warns per chunk); the rule is one message per warning per user
    action, so duplicates (same category + text) collapse into one notice.
    """
    seen: set[tuple[str, str]] = set()
    for w in caught:
        if _is_noise(w):
            continue
        key = (w.category.__name__, str(w.message))
        if key in seen:
            continue
        seen.add(key)
        notices.append(translate_warning(w).to_dict())


def guard(fn: Callable[..., Any], *args: Any,
          notices: list[dict] | None = None, **kwargs: Any):
    """Run ``fn`` capturing exceptions AND warnings into notice dicts.

    Returns ``(result, notices)`` where ``notices`` is a list of
    ``Notice.to_dict()`` dicts ready for the ``notices`` dcc.Store. On any
    exception the result is None and an error notice is appended — a callback
    using guard can therefore never dump a bare traceback (C15).

    The returned history is capped at :data:`MAX_STORED_NOTICES` (newest kept),
    and each notice carries only short ``details`` (see :meth:`Notice.to_dict`).
    """
    notices = list(notices or [])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - translate everything (C15)  # pylint: disable=broad-exception-caught
            _append_warnings(caught, notices)
            notices.append(translate_exception(exc).to_dict())
            return None, cap_notices(notices)
    _append_warnings(caught, notices)
    return result, cap_notices(notices)


def success(title: str, meaning: str = "", impact: str = "",
            fix: str = "") -> dict:
    """Convenience: a green success notice dict."""
    return Notice(severity="success", title=title, meaning=meaning,
                  impact=impact, fix=fix).to_dict()
