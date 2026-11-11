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
_GPU_GUARDS = re.compile(
    r"gpu_only|importorskip\(\s*[\"']cupy|skipif|skipUnless|skipIf|skipTest|unittest\.skip|pytest\.skip"
    r"|requires_gpu|needs_gpu|HAS_GPU|HAS_CUPY"
)
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


# ---------------------------------------------------------------------------- the workflow design (2026-11)
# Two workflows only: "Tests" (lint + test matrix + weekly newest-dependency run) and the release pipeline.
# These tests read .github/ and skip in an installed-wheel test run, where there is no source tree.
_WORKFLOWS = TESTS.parent / ".github" / "workflows"
_SUPPORTED_PYTHONS = ("3.12", "3.13", "3.14", "3.15")


def _workflow_text(name):
    path = _WORKFLOWS / name
    if not path.exists():
        pytest.skip("no source tree (installed-wheel test run)")
    return path.read_text(encoding="utf-8")


def test_only_the_two_workflows_exist():
    """Every extra workflow is one more place that goes stale.

    The former code-quality and weekly newest-dependencies files are merged into tests.yml.
    """
    if not _WORKFLOWS.exists():
        pytest.skip("no source tree (installed-wheel test run)")
    found = sorted(p.name for p in _WORKFLOWS.glob("*.y*ml"))
    assert found == ["build_and_publish_to_pypi.yml", "tests.yml"]


def test_tests_workflow_lints_with_ruff_only():
    text = _workflow_text("tests.yml")
    assert "python -m ruff check ." in text
    assert not re.search(r"flake8|pylint", text, re.IGNORECASE)


def test_tests_workflow_covers_every_supported_python_and_system():
    text = _workflow_text("tests.yml")
    for version in _SUPPORTED_PYTHONS:
        assert f'python: "{version}"' in text, f"Python {version} is missing from the test matrix"
    for system in ("ubuntu-latest", "windows-latest", "macos-15"):
        assert re.search(rf"os: {system}\b", text), f"{system} is missing from the test matrix"


def test_tests_workflow_runs_the_weekly_newest_dependency_job():
    text = _workflow_text("tests.yml")
    assert re.search(r"^\s*schedule:", text, re.MULTILINE)
    assert "--upgrade-strategy eager" in text


def test_workflows_do_not_depend_on_the_ubuntu_package_servers():
    """A run must not fail because a runner cannot reach the package server.

    All four Linux jobs of one run failed in "apt-get update" with "Temporary failure resolving". The
    build needs only gcc and g++ (gcc brings libgomp), which the runner images ship, so apt-get may appear
    only as a fallback: behind a check for the compiler and with retries.
    """
    for name in ("tests.yml", "build_and_publish_to_pypi.yml"):
        text = _workflow_text(name)
        commands = [line for line in text.splitlines() if "apt-get" in line and not line.lstrip().startswith("#")]
        if not commands:
            continue
        assert "command -v g++" in text, f"{name}: apt-get runs without a check for an installed compiler"
        for line in commands:
            if re.search(r"apt-get.*(update|install)", line):
                assert "Acquire::Retries" in line, f"{name}: apt-get without retries: {line.strip()}"
        assert not any("libomp-dev" in line for line in commands), (
            f"{name}: libomp-dev is not needed with gcc (libgomp)")


def test_release_workflow_builds_and_tests_every_supported_python():
    text = _workflow_text("build_and_publish_to_pypi.yml")
    for version in _SUPPORTED_PYTHONS:
        assert f"cp{version.replace('.', '')}-*" in text, f"no wheel is built for Python {version}"
        assert f'"{version}"' in text, f"the wheel tests skip Python {version}"
    assert not re.search(r"flake8|pylint", text, re.IGNORECASE)
