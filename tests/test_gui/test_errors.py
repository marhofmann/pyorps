"""Unit tests: the error/warning registry (Section 21) covers everything."""
import inspect
import warnings

import pytest

from pyorps.core import exceptions as pex
from pyorps.gui.services import errors

GENERIC_TITLE = "Something went wrong"

# constructor args for pyorps exceptions that need them
_CTOR_ARGS = {
    pex.RasterShapeError: ((4, 4, 4, 4),),
    pex.NoPathFoundError: (1, 2),
    pex.AlgorithmNotImplementedError: ("dijkstra", "cython"),
    pex.PairwiseError: (),
}


def _instantiate(cls):
    args = _CTOR_ARGS.get(cls, ("boom",))
    return cls(*args)


def _pyorps_exception_classes():
    return [obj for _, obj in inspect.getmembers(pex, inspect.isclass)
            if issubclass(obj, Exception) and obj.__module__ == pex.__name__]


def test_every_pyorps_exception_is_mapped():
    """No pyorps exception may fall through to the generic box (21.7)."""
    for cls in _pyorps_exception_classes():
        notice = errors.translate_exception(_instantiate(cls))
        assert notice.severity in errors.SEVERITIES
        assert notice.title != GENERIC_TITLE, (
            f"{cls.__name__} is not mapped in EXCEPTION_RULES")
        assert notice.details  # traceback text present


@pytest.mark.parametrize("exc,expected_title", [
    (pex.WFSConnectionError("down"), "Can't reach the WFS server"),
    (pex.NoPathFoundError(3, 9), "No route could be found"),
    (pex.PairwiseError(), "Source and target counts don't match"),
    (pex.RasterShapeError((1, 2, 3, 4)), "Unexpected raster shape"),
    (pex.FormatError("No numeric column"), "Cost table format is wrong"),
    (FileNotFoundError("nope.shp"), "File not found"),
    (MemoryError(), "Out of memory"),
])
def test_specific_exception_boxes(exc, expected_title):
    assert errors.translate_exception(exc).title == expected_title


def test_value_error_variants_matched_by_message():
    exc = ValueError("Unsupported vector data source: foo.xyz")
    assert errors.translate_exception(exc).title == "Unsupported file type"
    exc = ValueError("Source and target coordinates must not be None")
    notice = errors.translate_exception(exc)
    assert notice.title == "Set a source and a target first"
    assert notice.severity == "warning"
    # an unmatched ValueError falls back to the generic box
    assert errors.translate_exception(
        ValueError("weird")).title == GENERIC_TITLE


def test_unknown_exception_gets_generic_box_with_traceback():
    try:
        raise RuntimeError("kaboom")
    except RuntimeError as exc:
        notice = errors.translate_exception(exc)
    assert notice.title == GENERIC_TITLE
    assert notice.severity == "error"
    assert "kaboom" in notice.details and "RuntimeError" in notice.details


# ------------------------------------------------------------------ warnings
def _warning_message(text, category=UserWarning):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn(text, category)
    return caught[0]


@pytest.mark.parametrize("text,title_part,severity", [
    ("CRS mismatch between bbox (EPSG:4326) and vector data (EPSG:25832). "
     "Auto-reprojecting bbox to match data CRS.",
     "Reprojected your area", "info"),
    ("Geographic CRS (EPSG:4326) detected. Auto-reprojecting to EPSG:32632 "
     "for accurate metric calculations.",
     "Reprojected to a metric CRS", "info"),
    ("No search_space_buffer_m set — using full raster (100x100 = "
     "10,000 cells). This may cause excessive memory usage.",
     "No search buffer set", "warning"),
    ("2 position(s) had maximum cost value (65535) and were corrected:",
     "nearest routable cell", "info"),
    ("GPU v3 unavailable (ImportError), falling back to Cython",
     "GPU", "info"),
])
def test_warning_registry(text, title_part, severity):
    notice = errors.translate_warning(_warning_message(text))
    assert title_part.lower() in notice.title.lower()
    assert notice.severity == severity
    assert text[:40] in notice.details


def test_unknown_warning_gets_generic_box():
    notice = errors.translate_warning(_warning_message("odd stuff"))
    assert notice.severity == "warning"
    assert "odd stuff" in notice.details


# --------------------------------------------------------------------- guard
def test_guard_success_captures_warnings():
    def fn(x):
        warnings.warn("No search_space_buffer_m set — using full raster")
        return x + 1

    result, notices = errors.guard(fn, 1, notices=[])
    assert result == 2
    assert len(notices) == 1
    assert notices[0]["severity"] == "warning"
    assert notices[0]["title"].startswith("No search buffer")


def test_guard_exception_returns_none_and_notice():
    def fn():
        warnings.warn("Geographic CRS (x) detected. Auto-reprojecting to y")
        raise pex.NoPathFoundError(1, 2)

    result, notices = errors.guard(fn, notices=[])
    assert result is None
    assert [n["severity"] for n in notices] == ["info", "error"]
    assert notices[-1]["title"] == "No route could be found"


def test_guard_appends_to_existing_notices():
    existing = [errors.success("done")]
    result, notices = errors.guard(lambda: 42, notices=existing)
    assert result == 42
    assert notices[0]["title"] == "done"


def test_notice_roundtrip():
    n = errors.Notice(severity="info", title="t", meaning="m", impact="i",
                      fix="f", details="d", focus_id="tab-data")
    d = n.to_dict()
    assert errors.Notice.from_dict(d) == n


# ------------------------------------------- tidy-up: two library warnings with their own boxes
def test_algorithm_fallback_warning_has_its_own_plain_box():
    text = ("algorithm='delta-stepping' cannot evaluate slope terms (a DEM or a gradient objective): "
            "Dijkstra runs instead. Using the gpu path is not possible here.")
    notice = errors.translate_warning(_warning_message(text, pex.AlgorithmFallbackWarning))
    assert notice.title == "A different search ran than the one you chose"
    assert notice.severity == "info"
    assert "Dijkstra" in notice.meaning and "cython" in notice.meaning
    assert "route is correct" in notice.impact
    assert "Dijkstra" in notice.fix and "cython" in notice.fix
    assert notice.focus_id == "tab-routes"
    assert "A step reported a warning" not in notice.title
    assert text[:40] in notice.details
    # the same words in a plain UserWarning are NOT a fallback notice (the rule matches the class)
    other = errors.translate_warning(_warning_message("Dijkstra runs instead."))
    assert other.title != notice.title


@pytest.mark.parametrize("text", [
    # several points, the table form of the library
    "2 position(s) had maximum cost value (65535) and were corrected:\n  Orig Coords ...\n"
    "1 of them lie where the DEM has no elevation (outside its extent or in a void of it); those cells are "
    "forbidden like any other.\n",
    # one point on a float weight raster
    "Position [3, 4] lies on a forbidden cell and was corrected to [3, 6]. The cell has no DEM elevation "
    "(outside its extent or in a void of it); those cells are forbidden like any other.",
])
def test_a_point_moved_because_the_dem_has_no_elevation_gets_its_own_box(text):
    # the library emits this correction as a CoverageWarning; the rule matches the class and the DEM sentence
    notice = errors.translate_warning(_warning_message(text, pex.CoverageWarning))
    assert "terrain model has no elevation" in notice.title
    assert notice.severity == "warning"
    assert "DEM" in notice.meaning and "nearest cell with elevation" in notice.impact
    assert notice.focus_id == "tab-routes"


@pytest.mark.parametrize("text", [
    "Metric layer 'landscape': 120 of 4000 cell(s) have no value (outside the layer or nodata); they count as 0.",
    "DEM has 35 non-finite cell(s); they are forbidden and filled with the mean height.",
])
def test_other_coverage_warnings_get_the_coverage_box(text):
    notice = errors.translate_warning(_warning_message(text, pex.CoverageWarning))
    assert notice.title == "A layer or the terrain model does not cover the whole search area"
    assert notice.severity == "warning" and notice.focus_id == "tab-routes"
    assert "forbidden" in notice.impact and "count as 0" in notice.impact
    assert text[:30] in notice.details
    # the same words in a plain UserWarning are not a coverage notice (the rule matches the class)
    assert errors.translate_warning(_warning_message(text)).title != notice.title


def test_a_point_moved_off_an_ordinary_forbidden_cell_keeps_the_old_box():
    notice = errors.translate_warning(_warning_message(
        "2 position(s) had maximum cost value (65535) and were corrected:"))
    assert "nearest routable cell" in notice.title
    notice = errors.translate_warning(_warning_message(
        "Position [3, 4] lies on a forbidden cell and was corrected to [3, 6]."))
    assert notice.title == "A step reported a warning"


# ------------------------------------------- round 17: the line ends of constrained routing
SPAN_RULE_TEXT = (
    "The line of profile 'gui_profile' breaks a span or clearance rule (terminal_span_rule='remove'); the route and "
    "its numbers are returned as they are: (1) terminal_span_rule='remove' removed the tower at (21.5, 1.5), 2.00 m "
    "from the target (min_span_m 3 m); it leaves a merged span from (14.5, 1.5) to (23.5, 1.5) of 9.00 m, not "
    "below max_span_m 8 m. The search does not check the last span, and checks the first span with the line height "
    "at the source.")


def test_the_span_rule_warning_has_its_own_plain_box_that_names_what_to_choose():
    notice = errors.translate_warning(_warning_message(SPAN_RULE_TEXT, pex.SpanRuleWarning))
    assert notice.title == "The route breaks a span or clearance rule of its profile"
    assert notice.severity == "warning" and notice.focus_id == "tab-routes"
    assert "longer than the longest span" in notice.meaning and "end-span rule" in notice.meaning
    assert "still shown and usable" in notice.impact
    assert "Line ends" in notice.fix and "Never place one" in notice.fix
    assert SPAN_RULE_TEXT[:40] in notice.details and "merged span" in notice.details
    assert "A step reported a warning" not in notice.title


def test_the_span_rule_warning_box_is_chosen_by_the_class_not_by_the_words():
    other = errors.translate_warning(_warning_message(SPAN_RULE_TEXT))                     # a plain UserWarning
    assert other.title != "The route breaks a span or clearance rule of its profile"
    # the words "gpu" in a span-rule text do not turn it into the GPU box (the class rule comes first)
    gpu = errors.translate_warning(_warning_message(SPAN_RULE_TEXT + " Run on a gpu.", pex.SpanRuleWarning))
    assert gpu.title == "The route breaks a span or clearance rule of its profile"


def test_the_span_rule_warning_is_a_pyorps_warning_the_registry_covers():
    assert issubclass(pex.SpanRuleWarning, pex.PyorpsWarning)
    assert not issubclass(pex.SpanRuleWarning, pex.AlgorithmFallbackWarning)


@pytest.mark.parametrize(("text", "title"), [
    ("source_connection_height_m=20 m has no effect on the route: the search checks the conductor clearance of a "
     "span only with a DEM and a profile that has tower height, conductor weight, conductor tension, minimum "
     "clearance and span limits, and here there is no DEM. The height is still reported as the height of the end "
     "tower.", "A connection height does not change the route"),
    ("The raster cells are not square (x size 1, y size 1.5). Constrained routing measures every distance with the "
     "x size (1): path lengths, spans, step costs and tower distances are computed as if a cell were 1 wide in both "
     "directions, so every length along y is off by the factor 1.5. Resample the raster to square cells for exact "
     "lengths.", "The raster cells are not square"),
    ("The raster cell size is 2, not 1. The constrained GPU solver 'raster_gpu' still prices the terrain per cell "
     "(its kernels take no cell size for the terrain term), while the CPU backends ('cython', 'cython_parallel') "
     "and the report price it per metre. Use a CPU backend for cell sizes other than 1.",
     "The GPU solver prices the terrain differently on this raster"),
])
def test_the_other_warnings_of_the_finder_have_their_own_boxes_and_the_gpu_words_do_not_mislead(text, title):
    notice = errors.translate_warning(_warning_message(text, pex.PyorpsWarning))
    assert notice.title == title and notice.focus_id in ("tab-routes", "tab-raster")
    assert text[:40] in notice.details
    assert "unavailable" not in notice.title                  # the "gpu" rule's box says the GPU is not there
    # the class decides: the same words in a plain UserWarning keep their old boxes
    assert errors.translate_warning(_warning_message(text)).title != title


#: the message of a restricted search as the library words it now (a ``NoPathFoundError`` with a message of its own)
NO_ROUTE_ENFORCE = (
    "No route from (8.0, 8.0) to (52.0, 52.0) satisfies the line-end settings of this run: the search was run with "
    "terminal_span_rule='enforce' (every end span at least min_span_m 6 m long, a line without a tower excepted) "
    "(exact span bins of 0.9877 m), and no route meets them. Move the end point, allow shorter end spans "
    "(terminal_span_rule 'keep' or 'remove_if_valid'), or change the connection height. If the same call fails "
    "without these settings too, no route exists between the points at all (check the forbidden cells and the search "
    "window, search_space_buffer_m).")
NO_ROUTE_HEIGHT = NO_ROUTE_ENFORCE.replace(
    "terminal_span_rule='enforce' (every end span at least min_span_m 6 m long, a line without a tower excepted)",
    "the end span clearance from target_connection_height_m 10 m")
NO_ROUTE_BOTH = NO_ROUTE_ENFORCE.replace(
    "without a tower excepted)", "without a tower excepted) and the end span clearance from "
                                  "source_connection_height_m 20 m")
#: the first wording of the library: the generic message plus a reason that ends "... and no route satisfies it"
OLD_NO_ROUTE_REASON = (
    " The search was run with terminal_span_rule='enforce' (every end span at least min_span_m 6 m long, a line "
    "without a tower excepted) (exact span bins of 0.9697 m), and no route satisfies it. Move the end point, allow "
    "shorter end spans (terminal_span_rule 'keep' or 'remove_if_valid'), or change the connection height; if the "
    "same call fails without these settings too, no route exists between the points at all.")


class _OwnMessageNoPath(pex.NoPathFoundError):
    """A ``NoPathFoundError`` whose message is exactly ``message`` (how the finder raises a restricted search)."""

    def __init__(self, message):
        super().__init__(1, 2, "")
        self.args = (message,)


@pytest.mark.parametrize(("reason", "what", "fix_part"), [
    (NO_ROUTE_ENFORCE, "the end-span rule", "Choose another end-span rule"),
    (NO_ROUTE_HEIGHT, "the connection heights", "connection heights"),
    (NO_ROUTE_BOTH, "the end-span rule and the connection heights", "Choose another end-span rule"),
])
def test_no_route_under_the_line_end_settings_names_them_and_gives_the_librarys_reason_in_the_details(
        reason, what, fix_part):
    notice = errors.translate_exception(_OwnMessageNoPath(reason))
    assert notice.title == f"No route satisfies {what}" and notice.severity == "error"
    assert notice.focus_id == "tab-routes" and fix_part in notice.fix
    assert "Without that setting a route may exist" in notice.meaning
    assert notice.details == reason                          # the library's own words, not a traceback
    assert "blocked" not in notice.meaning and "forbidden areas" not in notice.fix


def test_the_first_wording_of_the_library_is_still_recognised_and_its_details_leave_out_the_buffer_hint():
    notice = errors.translate_exception(pex.NoPathFoundError("(8.0, 8.0)", "(52.0, 52.0)", OLD_NO_ROUTE_REASON))
    assert notice.title == "No route satisfies the end-span rule"
    assert notice.details == OLD_NO_ROUTE_REASON.strip() and "increase buffer" not in notice.details


def test_the_generic_no_route_box_stays_for_every_other_reason():
    for exc in (pex.NoPathFoundError(3, 9), pex.NoPathFoundError(3, 9, " The source is on a forbidden cell.")):
        notice = errors.translate_exception(exc)
        assert notice.title == "No route could be found"
        assert "blocked" in notice.meaning and "NoPathFoundError" in notice.details


def test_the_no_route_notice_survives_the_browser_store_round_trip():
    notice = errors.translate_exception(_OwnMessageNoPath(NO_ROUTE_ENFORCE)).to_dict()
    assert errors.Notice.from_dict(notice).title == "No route satisfies the end-span rule"
