"""Guards for the ways the 0.4.0 release tests failed on a clean machine.

Each test names the failure it prevents. They read the test sources only, so they run anywhere.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent

#: Names that only exist when CuPy is installed (sssp_gpu & co. guard their cupy import).
_GPU_NAMES = re.compile(r"\b(GpuSsspSession|sssp_raster_gpu\w*|eikonal_raster_gpu|import cupy|from cupy)\b")
#: Any of these in the file means its GPU tests skip themselves without a GPU.
_GPU_GUARDS = re.compile(r"gpu_only|importorskip\(\s*[\"']cupy|skipif|skipUnless|skipIf|skipTest|unittest\.skip|pytest\.skip|requires_gpu|needs_gpu|HAS_GPU|HAS_CUPY")
#: Modules whose import needs an optional extra: a module-level importorskip in a conftest aborts the session.
_OPTIONAL_IMPORTORSKIP = re.compile(r"^\s*pytest\.importorskip\(", re.MULTILINE)


def _test_files():
    return sorted(p for p in TESTS.rglob("test_*.py") if p.name != Path(__file__).name)


def test_gpu_tests_skip_themselves_without_cupy():
    """9 arena-overflow tests raised NameError on the CI runners (no GPU) because the class had no marker."""
    offenders = []
    for path in _test_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        if _GPU_NAMES.search(text) and not _GPU_GUARDS.search(text):
            offenders.append(str(path.relative_to(TESTS)))
    assert not offenders, (
        "These test files use CuPy-only names without any skip guard (add @gpu_only or "
        f"pytest.importorskip('cupy')): {offenders}")


def test_conftests_do_not_importorskip_at_module_level():
    """A skipped import in a conftest.py cancels the whole pytest session, not just its folder."""
    offenders = []
    for path in sorted(TESTS.rglob("conftest.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if _OPTIONAL_IMPORTORSKIP.search(text):
            offenders.append(str(path.relative_to(TESTS)))
    assert not offenders, (
        f"Use collect_ignore_glob when an optional package is missing instead of importorskip: {offenders}")


def test_tests_do_not_import_from_benchmarks():
    """benchmarks/ is not part of the repository; the wheel-test job copies only tests/ and profiles/."""
    offenders = []
    for path in _test_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        if re.search(r"^\s*(from|import)\s+benchmarks\b", text, re.MULTILINE):
            offenders.append(str(path.relative_to(TESTS)))
    assert not offenders, f"Move the helper into tests/: {offenders}"


@pytest.mark.parametrize("folder", ["profiles"])
def test_folders_that_tests_read_are_not_gitignored(folder):
    """profiles/ was gitignored for months, so a clean checkout had no profile files at all."""
    root = TESTS.parent
    gitignore = root / ".gitignore"
    if not gitignore.exists():
        pytest.skip("no source tree (installed-wheel test run)")
    lines = [ln.strip() for ln in gitignore.read_text(encoding="utf-8").splitlines()]
    blanket = {f"{folder}/", f"/{folder}/", folder, f"/{folder}"}
    assert not blanket.intersection(lines), f"{folder}/ is ignored as a whole; tracked files in it would be invisible"
