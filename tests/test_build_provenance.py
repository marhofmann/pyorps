# tests/test_build_provenance.py
"""Guard that the compiled extensions were built from the checked-out sources.

``setup.py``'s build_ext command writes a ``_build_provenance.json`` next to the
binaries it produces, recording the SHA-256 of every ``.pyx`` it compiled and of
every binary it emitted. setuptools decides what to recompile from mtimes alone,
so without this the repository can hold a ``.pyd`` that no longer corresponds to
any source on disk - and every benchmark taken against it is silently invalid.

These tests fail until ``python setup.py build_ext --inplace`` has been run
against the current working tree.
"""

import hashlib
import importlib.machinery
import json
import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "pyorps"
PROVENANCE_FILENAME = "_build_provenance.json"
REBUILD_HINT = (
    f"Rebuild with:  python setup.py build_ext --inplace   (in {PROJECT_ROOT})"
)
FAST_MATH_VALUES = {"1", "true", "yes", "on"}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifests():
    """Return every provenance manifest shipped alongside a compiled extension."""
    paths = sorted(PACKAGE_ROOT.rglob(PROVENANCE_FILENAME))
    if not paths:
        pytest.fail(
            f"No {PROVENANCE_FILENAME} found under {PACKAGE_ROOT}.\n"
            "The compiled extensions carry no record of the sources they were "
            "built from, so nothing measured against them can be trusted.\n"
            f"{REBUILD_HINT}"
        )
    return [(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]


def resolve_recorded_file(manifest_path, manifest, name):
    """Locate a recorded source/header next to the binaries or in its source dir."""
    candidate = manifest_path.parent / name
    if candidate.exists():
        return candidate
    for source_dir in manifest.get("source_dirs", []):
        candidate = Path(source_dir) / name
        if candidate.exists():
            return candidate
    return None


def importable_stem(path):
    """Return the module name a file would be imported as, or None if it is not one."""
    for suffix in sorted(importlib.machinery.EXTENSION_SUFFIXES, key=len, reverse=True):
        if path.name.endswith(suffix):
            stem = path.name[: -len(suffix)]
            return stem if stem.isidentifier() else None
    return None


def fail_with(headline, problems, hint=REBUILD_HINT):
    listing = "\n".join(f"  - {problem}" for problem in problems)
    pytest.fail(f"{headline}\n{listing}\n{hint}")


def test_provenance_manifest_exists():
    manifests = load_manifests()
    assert all(manifest["modules"] for _, manifest in manifests)


def test_every_extension_built_from_the_current_pyx():
    problems = []
    for manifest_path, manifest in load_manifests():
        for module, entry in sorted(manifest["modules"].items()):
            if entry["source_kind"] != "pyx":
                continue
            source = resolve_recorded_file(manifest_path, manifest, entry["source"])
            if source is None:
                problems.append(
                    f"{module}: was built from {entry['source']}, which is no "
                    "longer anywhere in the tree"
                )
                continue
            on_disk = sha256_file(source)
            if on_disk != entry["source_sha256"]:
                problems.append(
                    f"{module}: {source} has changed since the build "
                    f"(built from {entry['source_sha256'][:12]}, "
                    f"on disk {on_disk[:12]})"
                )
    if problems:
        fail_with(
            "Compiled extensions do not match their Cython sources - the binaries "
            "in this tree are stale:",
            problems,
        )


def test_binaries_match_the_recorded_build():
    problems = []
    for manifest_path, manifest in load_manifests():
        for module, entry in sorted(manifest["modules"].items()):
            binary = manifest_path.parent / entry["binary"]
            if not binary.exists():
                problems.append(f"{module}: {entry['binary']} is missing")
                continue
            on_disk = sha256_file(binary)
            if on_disk != entry["binary_sha256"]:
                problems.append(
                    f"{module}: {entry['binary']} is not the binary this manifest "
                    "describes (replaced or rebuilt without updating the manifest)"
                )
    if problems:
        fail_with(
            "Compiled extensions do not match the build that produced this "
            "provenance manifest:",
            problems,
        )


def test_no_extension_built_from_a_pregenerated_cpp():
    problems = []
    for _, manifest in load_manifests():
        for module, entry in sorted(manifest["modules"].items()):
            if entry["source_kind"] != "pyx":
                problems.append(
                    f"{module}: built from {entry['source']} because no .pyx was "
                    "present - the binary has no reviewable source"
                )
    if problems:
        fail_with(
            "Extensions were built from pre-generated C++ instead of Cython "
            "sources:",
            problems,
            hint="Restore the missing .pyx files before taking any measurement.",
        )


def test_shared_cython_headers_unchanged():
    problems = []
    for manifest_path, manifest in load_manifests():
        for name, recorded in sorted(manifest.get("headers", {}).items()):
            header = resolve_recorded_file(manifest_path, manifest, name)
            if header is None:
                problems.append(f"{name}: removed since the build")
            elif sha256_file(header) != recorded:
                problems.append(f"{name}: changed since the build")
    if problems:
        fail_with(
            "Shared Cython/C headers changed since the extensions were built; "
            "every extension that includes them is stale:",
            problems,
        )


def test_no_compiled_module_without_source():
    """A binary with no .pyx shadows any same-named Python module - flag, never guess.

    Do not delete anything this reports: some of these binaries are imported in
    preference to a pure-Python fallback, so removing one changes which code
    runs. Restore the .pyx, or remove the binary as a deliberate decision.
    """
    orphans = []
    for path in sorted(PACKAGE_ROOT.rglob("*")):
        if not path.is_file():
            continue
        stem = importable_stem(path)
        if stem is None:
            continue
        if not (path.parent / f"{stem}.pyx").exists():
            orphans.append(
                f"{path.relative_to(PROJECT_ROOT)} - no {stem}.pyx in "
                f"{path.parent.relative_to(PROJECT_ROOT)}"
            )
    if orphans:
        fail_with(
            "Importable compiled modules exist with no Cython source in the tree. "
            "They cannot be rebuilt, cannot be reviewed, and take import "
            "precedence over any same-named Python module:",
            orphans,
            hint=(
                "Restore the .pyx, or remove the binary deliberately after "
                "checking what imports it."
            ),
        )


def test_extensions_not_built_with_fast_math():
    """The kernels compare against +inf sentinels; -ffast-math makes that undefined."""
    requested = (
        os.environ.get("PYORPS_FAST_MATH", "").strip().lower() in FAST_MATH_VALUES
    )
    problems = []
    for manifest_path, manifest in load_manifests():
        built_fast = bool(manifest.get("fast_math"))
        if built_fast != requested:
            problems.append(
                f"{manifest_path.relative_to(PROJECT_ROOT)}: built with "
                f"fast_math={built_fast} "
                f"(compile args: {' '.join(manifest.get('compile_args') or [])})"
            )
    if problems:
        fail_with(
            f"Floating-point build mode does not match PYORPS_FAST_MATH="
            f"{'1' if requested else '0'}. Unsafe FP mode breaks the +inf "
            "sentinels the kernels rely on and the bit-identical weights the "
            "parity suites assume:",
            problems,
            hint=(
                "Set PYORPS_FAST_MATH identically for the build and the test run "
                f"(the parity job builds both ways).\n{REBUILD_HINT}"
            ),
        )
