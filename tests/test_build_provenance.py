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

import ast
import hashlib
import importlib.machinery
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "pyorps"
SETUP_PY = PROJECT_ROOT / "setup.py"
PROVENANCE_FILENAME = "_build_provenance.json"
REBUILD_HINT = (
    f"Rebuild with:  python setup.py build_ext --inplace   (in {PROJECT_ROOT})"
)
FAST_MATH_VALUES = {"1", "true", "yes", "on"}

#: ``cdef extern from "foo.h"`` / ``include "foo.pxi"`` - quoted forms only.
#: Angle-bracket externs name system headers and are not ours to track.
EXTERN_RE = re.compile(r'cdef\s+extern\s+from\s+["\']([^"\']+)["\']')
INCLUDE_RE = re.compile(r'^\s*include\s+["\']([^"\']+)["\']', re.MULTILINE)


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


def tracked_files():
    """Repo-relative posix paths git has under version control.

    Skips (rather than fails) outside a work tree: an installed sdist or wheel
    has no repository to interrogate.
    """
    if not (PROJECT_ROOT / ".git").exists():
        pytest.skip("not a git work tree - nothing to check tracking against")
    try:
        completed = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=PROJECT_ROOT, capture_output=True, check=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"git unavailable: {exc}")
    return {name for name in completed.stdout.decode("utf-8").split("\0") if name}


def declared_modules():
    """The (module, stem-path) pairs setup.py's MODULES declares.

    Parsed rather than imported: setup.py calls setup() at module scope, so
    importing it here would run a build.
    """
    tree = ast.parse(SETUP_PY.read_text(encoding="utf-8"), filename=str(SETUP_PY))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(getattr(t, "id", None) == "MODULES" for t in node.targets):
            return [tuple(entry) for entry in ast.literal_eval(node.value)]
    pytest.fail(f"No module-level MODULES assignment found in {SETUP_PY}")


def local_dependencies(pyx_path):
    """Quoted headers/includes referenced by a .pyx that resolve inside the repo."""
    text = pyx_path.read_text(encoding="utf-8", errors="replace")
    found = set()
    for name in EXTERN_RE.findall(text) + INCLUDE_RE.findall(text):
        for base in (pyx_path.parent, PACKAGE_ROOT / "utils"):
            candidate = (base / name).resolve()
            if candidate.is_file():
                found.add(candidate)
                break
    return found


def as_repo_path(path):
    return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()


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


def test_git_tracking_probe_is_meaningful():
    """The tracking checks are only worth anything if the probe discriminates."""
    tracked = tracked_files()
    assert "setup.py" in tracked, (
        "git ls-files did not report setup.py as tracked; the tracking checks "
        "below would pass vacuously"
    )
    assert "pyorps/utils/_no_such_source.pyx" not in tracked


def test_every_declared_module_source_is_tracked():
    """setup.py must not declare a module whose source only exists locally.

    A working tree that holds the .pyx builds happily while a fresh clone dies
    in cythonize with "doesn't match any files". Both the .pyx and any .pxd
    beside it count: the .pxd is what other modules cimport.
    """
    tracked = tracked_files()
    problems = []
    for module, stem in declared_modules():
        for suffix in (".pyx", ".pxd"):
            source = PROJECT_ROOT / f"{stem}{suffix}"
            if not source.is_file():
                if suffix == ".pyx":
                    problems.append(
                        f"{module}: declared in MODULES but {stem}.pyx does not "
                        "exist at all"
                    )
                continue
            rel = as_repo_path(source)
            if rel not in tracked:
                problems.append(f"{module}: {rel} exists but is NOT tracked by git")
    if problems:
        fail_with(
            "setup.py declares extensions whose sources are not in the "
            "repository. This tree builds; a clean checkout does not:",
            problems,
            hint=(
                "Commit the sources, or remove the entry from MODULES. "
                "Run 'git status --porcelain' to see them as untracked."
            ),
        )


def test_every_local_build_dependency_is_tracked():
    """Headers and .pxi includes a declared .pyx pulls in must be tracked too.

    Committing the .pyx alone is not enough: a 'cdef extern from "atomic_cas.h"'
    whose header is untracked, or one committed without a function the kernel
    calls, fails at compile time on a clean checkout rather than in cythonize.
    """
    tracked = tracked_files()
    problems = []
    for module, stem in declared_modules():
        pyx = PROJECT_ROOT / f"{stem}.pyx"
        if not pyx.is_file():
            continue  # reported by the previous test
        for dependency in sorted(local_dependencies(pyx)):
            rel = as_repo_path(dependency)
            if rel not in tracked:
                problems.append(f"{module}: needs {rel}, which is NOT tracked by git")
    if problems:
        fail_with(
            "Declared extensions depend on local headers/includes that are not "
            "in the repository:",
            problems,
            hint="Commit the headers, or drop the dependency from the .pyx.",
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
