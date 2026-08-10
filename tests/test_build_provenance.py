# tests/test_build_provenance.py
"""Guard that what is committed can be built and imported from a clean checkout.

Two families of check live here, both answering the same question - *does this
repository contain everything it references, or only my working tree?*

**Build provenance.** ``setup.py``'s build_ext command writes a
``_build_provenance.json`` next to the binaries it produces, recording the
SHA-256 of every ``.pyx`` it compiled and of every binary it emitted. setuptools
decides what to recompile from mtimes alone, so without this the repository can
hold a ``.pyd`` that no longer corresponds to any source on disk - and every
benchmark taken against it is silently invalid. These tests fail until
``python setup.py build_ext --inplace`` has been run against the current
working tree. The tracking half additionally checks that every module
``setup.py`` declares, and every header its ``.pyx`` externs, is tracked by git.

**Import provenance.** The same defect class reaches pure Python: a module can
be committed while a module it imports is not, so ``pip install`` from a fresh
clone yields a package that raises ``ModuleNotFoundError`` on import. That has
happened repeatedly here (an extension declared for an uncommitted ``.pyx``; a
kernel needing an uncommitted header; a subpackage committed one file out of
forty-four, ``__init__.py`` included). ``ast`` is used rather than a regex:
regexes miss ``from .a.b import c`` and fire inside strings and comments.
"""

import ast
import hashlib
import importlib.machinery
import json
import os
import re
import subprocess
import warnings
from pathlib import Path
from typing import NamedTuple

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

#: The distribution's import root. Only imports resolving inside it are ours to
#: police; third-party and stdlib imports are the dependency metadata's problem.
PACKAGE_NAME = PACKAGE_ROOT.name

#: Every top-level directory in this repo that a tracked module may import
#: from. Restricting the resolver to ``pyorps`` alone left a hole: tracked
#: ``tests/test_exactness_referee.py`` imports ``benchmarks.exactness_referee``
#: at module level (a PEP-420 namespace package, reachable via conftest's
#: sys.path insert), so an untracked benchmark would kill collection on a clean
#: checkout without this guard noticing. Widening what is SEEN is independent of
#: IMPORT_CRITICAL_ROOTS, which decides what FAILS.
RESOLVABLE_ROOTS = ("pyorps", "tests", "benchmarks", "case_studies")

#: Trees whose tracked modules a clean checkout must be able to import before it
#: can do anything at all: the package itself, and the suite that gates releases.
#: ``benchmarks/`` and ``case_studies/`` are deliberately *not* here - see
#: ``test_deferred_and_out_of_scope_untracked_imports``.
IMPORT_CRITICAL_ROOTS = ("pyorps", "tests")

#: Suffixes under which a tracked file provides an importable module. ``.pyx``
#: counts because ``setup.py`` compiles it into an extension of the same name,
#: and the build guards above already prove that source is present and current.
MODULE_SUFFIXES = (".py", ".pyx")

#: ``ast.TryStar`` only exists from 3.11; treat both as import guards.
TRY_NODES = tuple(
    node for node in (getattr(ast, "Try", None), getattr(ast, "TryStar", None))
    if node is not None
)

#: Exception names whose handler makes a module-level import optional rather
#: than fatal.
GUARD_EXCEPTIONS = frozenset(
    {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}
)

IMPORT_HINT = (
    "Commit the missing modules, or drop the import. "
    "'git status --porcelain' lists them as untracked; "
    "'git ls-files <path>' confirms whether one is really in the repository."
)


class UntrackedImportWarning(UserWarning):
    """A tracked module references a module git does not track.

    Raised only for references that do *not* break importing the package -
    deferred imports, and files outside ``IMPORT_CRITICAL_ROOTS``. pytest prints
    it in the warnings summary of every run, so the gap stays visible without
    failing the suite; add a ``filterwarnings = error`` entry to make the
    informational tier binding once the backlog is empty.
    """


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


# --------------------------------------------------------------------------
# Import provenance: resolve intra-package imports to files and check tracking.
# --------------------------------------------------------------------------


class ImportRef(NamedTuple):
    """One intra-package module reference found in one tracked file."""

    importer: str  # repo-relative posix path of the file doing the importing
    dotted: str  # fully-qualified module name, always rooted at PACKAGE_NAME
    lineno: int
    deferred: str  # "" when module-level and unguarded, else why it is deferred
    speculative: bool  # a name after ``from X import name``: module or attribute?


def package_of(rel_posix):
    """The dotted package a repo-relative module lives in, as a parts tuple."""
    parts = rel_posix.split("/")
    return tuple(parts[:-1])  # ``__init__.py`` included: its package is its dir


def deferral_reason(ancestors, node):
    """Why this import does not run at package-import time, or "" if it does.

    Three deferrals, all of which mean *a feature breaks* rather than *the
    package cannot be imported*: inside a function or class body, inside the
    body of a ``try`` that swallows ``ImportError``, and under
    ``if TYPE_CHECKING`` (which never executes at runtime at all).
    """
    for ancestor in reversed(ancestors):
        if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return f"deferred inside function {ancestor.name}()"
        if isinstance(ancestor, ast.ClassDef):
            return f"deferred inside class {ancestor.name}"
    for index, ancestor in enumerate(ancestors):
        child = ancestors[index + 1] if index + 1 < len(ancestors) else node
        if isinstance(ancestor, ast.If) and _is_type_checking(ancestor.test):
            if any(child is stmt for stmt in ancestor.body):
                return "type-checking only (if TYPE_CHECKING)"
        if not isinstance(ancestor, TRY_NODES):
            continue
        if not any(child is stmt for stmt in ancestor.body):
            continue
        for handler in ancestor.handlers:
            caught = _handler_names(handler.type)
            if caught & GUARD_EXCEPTIONS or "*bare*" in caught:
                return "guarded by try/except " + "/".join(sorted(caught))
    return ""


def _is_type_checking(test):
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _handler_names(node):
    if node is None:
        return {"*bare*"}
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Attribute):
        return {node.attr}
    if isinstance(node, ast.Tuple):
        names = set()
        for element in node.elts:
            names |= _handler_names(element)
        return names
    return set()


def intra_package_imports(rel_posix, source):
    """Every ``PACKAGE_NAME.*`` module one source file imports.

    Handles ``import pyorps.a.b``, ``from pyorps.a import b``,
    ``from . import x``, ``from .x import y`` and ``from ..a.b import c``.
    Imports that resolve outside the package (stdlib, third party, relative
    imports from a non-package tree such as ``tests/``) are dropped.

    Pure: takes text, returns refs. That is what makes the resolver testable
    against a synthetic tree in ``test_import_resolver_detects_a_missing_module``.
    """
    tree = ast.parse(source, filename=rel_posix)
    package = package_of(rel_posix)
    refs = []
    stack = [(tree, [])]
    while stack:
        node, ancestors = stack.pop()
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            refs.extend(
                _refs_for_node(node, rel_posix, package,
                               deferral_reason(ancestors, node))
            )
        child_ancestors = ancestors + [node]
        stack.extend((child, child_ancestors) for child in ast.iter_child_nodes(node))
    return sorted(refs, key=lambda ref: (ref.lineno, ref.dotted))


def _refs_for_node(node, rel_posix, package, deferred):
    if isinstance(node, ast.Import):
        return [
            ImportRef(rel_posix, alias.name, node.lineno, deferred, False)
            for alias in node.names
            if alias.name.split(".")[0] in RESOLVABLE_ROOTS
        ]

    if node.level:  # relative: walk up `level - 1` packages from this file's own
        parts = list(package)
        for _ in range(node.level - 1):
            if not parts:
                return []  # escapes the tree root; not a reference we can resolve
            parts.pop()
        parts += node.module.split(".") if node.module else []
    elif node.module:
        parts = node.module.split(".")
    else:
        return []
    if not parts or parts[0] not in RESOLVABLE_ROOTS:
        return []

    dotted = ".".join(parts)
    refs = [ImportRef(rel_posix, dotted, node.lineno, deferred, False)]
    # ``from X import name``: `name` is either a submodule of X or an attribute
    # bound in its ``__init__``. Resolve it as a submodule, but speculatively -
    # see ``untracked_prefix``.
    refs += [
        ImportRef(rel_posix, f"{dotted}.{alias.name}", node.lineno, deferred, True)
        for alias in node.names
        if alias.name != "*"
    ]
    return refs


def module_candidates(dotted):
    """The repo-relative paths that could provide ``dotted`` as a module."""
    stem = dotted.replace(".", "/")
    return [stem + suffix for suffix in MODULE_SUFFIXES] + [stem + "/__init__.py"]


def untracked_prefix(ref, tracked):
    """The shallowest ancestor of ``ref.dotted`` git does not track, or None.

    Reports the shallowest one because that is the root cause: importing
    ``pyorps.a.b`` imports ``pyorps.a`` first, so an untracked ``pyorps/a`` is
    the single fix for every reference beneath it.

    A speculative trailing name is only accused when a file on disk proves it is
    a module. Static analysis cannot tell ``from . import cost`` (submodule) from
    ``from . import cost`` (name bound in ``__init__``), and a false accusation
    is what gets a guard disabled. This is not a hole in practice: the mistake is
    made in a tree where the file *does* exist but was never added, which is
    exactly the case this branch catches.
    """
    parts = ref.dotted.split(".")
    for depth in range(1, len(parts) + 1):
        prefix = ".".join(parts[:depth])
        candidates = module_candidates(prefix)
        if any(candidate in tracked for candidate in candidates):
            continue
        # PEP 420 namespace package: a directory with tracked modules under it
        # and no __init__.py. `benchmarks/` and `tests/` are exactly this -
        # importable via conftest's sys.path insert - so demanding an
        # __init__.py for them would accuse three correct imports.
        if depth < len(parts):
            as_dir = prefix.replace(".", "/") + "/"
            if any(rel.startswith(as_dir) for rel in tracked):
                continue
        if ref.speculative and depth == len(parts):
            if not any((PROJECT_ROOT / c).is_file() for c in candidates):
                return None
        return prefix, candidates
    return None


def tracked_python_sources(tracked, roots=None):
    """Repo-relative tracked ``.py`` paths, optionally restricted to ``roots``."""
    prefixes = tuple(f"{root}/" for root in roots) if roots else None
    return sorted(
        rel
        for rel in tracked
        if rel.endswith(".py") and (prefixes is None or rel.startswith(prefixes))
    )


def scan_untracked_imports(tracked, roots=None):
    """Resolve every intra-package import in ``roots``.

    Returns ``(problems, unreadable, scanned)`` where each problem is
    ``(ref, prefix, candidates)``, deduplicated per (importer, prefix) so one
    missing module named ten times is one line to fix.
    """
    problems, unreadable, scanned = [], [], 0
    seen = set()
    for rel in tracked_python_sources(tracked, roots):
        path = PROJECT_ROOT / rel
        if not path.is_file():
            unreadable.append(f"{rel}: tracked by git but not present in the tree")
            continue
        try:
            refs = intra_package_imports(
                rel, path.read_text(encoding="utf-8", errors="replace")
            )
        except SyntaxError as exc:
            unreadable.append(f"{rel}: could not be parsed ({exc})")
            continue
        scanned += 1
        for ref in refs:
            found = untracked_prefix(ref, tracked)
            if found is None:
                continue
            prefix, candidates = found
            if (rel, prefix) in seen:
                continue
            seen.add((rel, prefix))
            problems.append((ref, prefix, candidates))
    return problems, unreadable, scanned


def describe_import_problem(ref, prefix, candidates):
    on_disk = [c for c in candidates if (PROJECT_ROOT / c).is_file()]
    where = f"{ref.importer}:{ref.lineno}"
    resolved = (
        f"present but untracked: {', '.join(on_disk)}"
        if on_disk
        else f"no file at any of: {', '.join(candidates)}"
    )
    detail = f" [{ref.deferred}]" if ref.deferred else ""
    return f"{where} imports {ref.dotted} -> {prefix} ({resolved}){detail}"


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


def test_import_resolver_detects_a_missing_module():
    """The import guards below are worth nothing unless the resolver discriminates.

    Runs the resolver on a synthetic file against a synthetic tracked set, so it
    proves detection rather than inheriting whatever the real tree happens to
    look like. It also pins the two deliberate design decisions - deferred
    imports are classified, not dropped; a trailing ``from X import name`` is
    only accused when a file proves it is a module - so neither can be softened
    by accident.
    """
    source = (
        "from pyorps.core.path import Path\n"  # 1: tracked module + attribute
        "import pyorps.utils.phantom\n"  # 2: absolute, nothing there
        "from . import ghost\n"  # 3: could be an attribute; no file
        "from pyorps.graph import path_finder\n"  # 4: real file, untracked here
        "if TYPE_CHECKING:\n"
        "    from pyorps.core.wraith import Wraith\n"  # 6: never executes
        "def factory():\n"
        "    from pyorps.graph.api.spectre import Spectre\n"  # 8: lazy
        "import os, numpy, pyorps\n"  # 9: foreign imports must be ignored
    )
    tracked = {
        "pyorps/__init__.py",
        "pyorps/core/__init__.py",
        "pyorps/core/path.py",
        "pyorps/graph/__init__.py",
        "pyorps/graph/api/__init__.py",
        "pyorps/utils/__init__.py",
    }
    refs = intra_package_imports("pyorps/graph/_synthetic_probe.py", source)
    resolved = {ref.dotted: (ref.deferred, untracked_prefix(ref, tracked)) for ref in refs}

    assert not any(d.startswith(("os", "numpy")) for d in resolved), (
        f"resolver leaked non-package imports: {sorted(resolved)}"
    )
    # Compare the reported *prefixes*: the prefix is the module to commit, and
    # collapsing ``pyorps.a.ghost`` and ``pyorps.a.ghost.Name`` onto one prefix
    # is what keeps the failure list one line per fix.
    hard = {found[0] for _, (why, found) in resolved.items() if found and not why}
    assert hard == {"pyorps.utils.phantom", "pyorps.graph.path_finder"}, (
        f"resolver missed or invented a module-level offender: {sorted(hard)}"
    )
    deferred = {found[0] for _, (why, found) in resolved.items() if found and why}
    assert deferred == {"pyorps.core.wraith", "pyorps.graph.api.spectre"}, (
        f"deferred offenders misclassified: {sorted(deferred)}"
    )
    assert "function factory()" in resolved["pyorps.graph.api.spectre"][0]
    assert "TYPE_CHECKING" in resolved["pyorps.core.wraith"][0]
    # Negative controls: a tracked module reached via an attribute, and a name
    # that no file proves to be a module, must both stay silent.
    assert resolved["pyorps.core.path"][1] is None
    assert resolved["pyorps.graph.ghost"][1] is None


def test_no_tracked_module_imports_an_untracked_module():
    """A committed module must not import one that lives only in a working tree.

    This is the Python twin of ``test_every_declared_module_source_is_tracked``.
    The failure mode is worse than a build error, because it is invisible until
    someone installs from a clean clone: the import raises
    ``ModuleNotFoundError`` for a file the author can see on their own disk.

    Only module-level, unguarded imports fail here - those are the ones that
    make ``import pyorps`` itself impossible. Deferred ones are reported by
    ``test_deferred_and_out_of_scope_untracked_imports`` instead.
    """
    tracked = tracked_files()
    problems, _, scanned = scan_untracked_imports(tracked, IMPORT_CRITICAL_ROOTS)
    assert scanned >= 20, (
        f"only {scanned} tracked modules were parsed under "
        f"{IMPORT_CRITICAL_ROOTS}; this guard would pass vacuously"
    )
    offenders = [
        describe_import_problem(ref, prefix, candidates)
        for ref, prefix, candidates in problems
        if not ref.deferred
    ]
    if offenders:
        fail_with(
            "Tracked modules import modules that are NOT in the repository. "
            "This tree imports; a clean checkout raises ModuleNotFoundError:",
            sorted(offenders),
            hint=IMPORT_HINT,
        )


def test_every_tracked_module_lives_in_a_tracked_package():
    """Every ancestor ``__init__.py`` of a tracked module must be tracked too.

    Committing part of a package is the same defect seen from the other side: a
    file lands in the repository while the ``__init__.py`` that makes its
    directory importable does not, so nothing under it can ever be imported.
    Cheaper and more direct than following the imports, and it fires even when
    the orphaned module imports nothing at all.
    """
    tracked = tracked_files()
    modules = tracked_python_sources(tracked, (PACKAGE_NAME,))
    assert modules, f"no tracked modules found under {PACKAGE_NAME}/"
    problems = []
    for rel in modules:
        directories = rel.split("/")[:-1]
        for depth in range(1, len(directories) + 1):
            init = "/".join(directories[:depth]) + "/__init__.py"
            if init not in tracked:
                problems.append(f"{rel}: {init} is NOT tracked by git")
                break
    if problems:
        fail_with(
            "Tracked modules sit in packages whose __init__.py is not in the "
            "repository - a clean checkout cannot import any of them:",
            sorted(problems),
            hint=IMPORT_HINT,
        )


def test_deferred_and_out_of_scope_untracked_imports():
    """Report - never fail on - untracked imports that do not break the package.

    Two kinds land here. **Deferred** imports (inside a function or class, under
    a ``try/except ImportError``, or under ``if TYPE_CHECKING``) cost one backend
    or one optional feature, not the import of ``pyorps``; failing on them would
    force either committing work in progress or adding suppressions, and
    suppressions rot. **Out-of-scope** files (``benchmarks/``, ``case_studies/``,
    ``examples/``) are not on the install-and-run path at all.

    Reported through a warning so pytest surfaces it in every run's warnings
    summary: a known gap stays visible, and the strict guard above stays quiet
    enough that nobody reaches for ``-p no:cacheprovider`` on it.
    """
    tracked = tracked_files()
    everything, unreadable, scanned = scan_untracked_imports(tracked)
    assert scanned >= 20, f"only {scanned} tracked modules were parsed"
    critical = tuple(f"{root}/" for root in IMPORT_CRITICAL_ROOTS)
    notes = [
        describe_import_problem(ref, prefix, candidates)
        for ref, prefix, candidates in everything
        # module-level offenders inside the critical roots are the strict
        # guard's business; do not report them twice.
        if ref.deferred or not ref.importer.startswith(critical)
    ]
    notes += unreadable
    if notes:
        report = "\n".join(f"  - {note}" for note in sorted(notes))
        message = (
            f"{len(notes)} untracked-module reference(s) that do not break "
            f"importing {PACKAGE_NAME}:\n{report}\n{IMPORT_HINT}"
        )
        print("\n" + message)
        warnings.warn(message, UntrackedImportWarning, stacklevel=2)


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
