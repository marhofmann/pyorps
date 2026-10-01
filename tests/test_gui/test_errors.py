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
