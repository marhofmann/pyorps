"""Static check of the package layering: which pyorps package may import which.

The core (``core``, ``graph``, ``io``, ``raster``, ``utils``) imports no
research package, and the research packages form one direction only::

    core  <-  collector  <-  certify
    core  <-  collector  <-  costmodel
    siting imports no other research package

Every ``pyorps/**/*.py`` is parsed with :mod:`ast`; no module is imported, so
the test also sees imports inside functions, under ``TYPE_CHECKING`` and in
modules that need an optional dependency. Absolute and relative imports
count, as does ``from pyorps import siting``. Imports built at run time
(``importlib.import_module``) cannot be seen and are not checked.

To allow a new dependency between research packages, add it to
``ALLOWED_RESEARCH_IMPORTS`` and say why in the commit message.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

import pyorps

# The allowed edges. Key: a research package; value: the OTHER research
# packages it may import (every package may import itself and the core).
# Every unit not listed here (core, graph, io, raster, utils, gui, webviz and
# the package root ``pyorps/__init__.py``) may import no research package.
ALLOWED_RESEARCH_IMPORTS: dict[str, frozenset[str]] = {
    "collector": frozenset(),
    "certify": frozenset({"collector"}),
    "costmodel": frozenset({"collector"}),
    "siting": frozenset(),
}

PACKAGE = "pyorps"
# The installed package, not a path next to this file: the release job runs the tests from a copy
# of tests/ without the repository around it.
PACKAGE_DIR = Path(pyorps.__file__).resolve().parent


def _module_parts(path: Path) -> tuple[list[str], bool]:
    """Dotted name parts of the module at ``path`` and whether it is a package."""
    rel = path.relative_to(PACKAGE_DIR.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        return parts[:-1], True
    return parts, False


def _imported_modules(tree: ast.AST, module_parts: list[str],
                      is_package: bool) -> Iterator[tuple[int, str]]:
    """``(line, absolute module name)`` for every import anywhere in the tree.

    ``from a import b`` yields ``a`` and ``a.b``: ``b`` may be a submodule.
    """
    package_parts = module_parts if is_package else module_parts[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                keep = len(package_parts) - (node.level - 1)
                if keep < 0:
                    continue
                base_parts = package_parts[:keep]
            else:
                base_parts = []
            if node.module:
                base_parts = base_parts + node.module.split(".")
            base = ".".join(base_parts)
            yield node.lineno, base
            for alias in node.names:
                yield node.lineno, f"{base}.{alias.name}" if base else alias.name


def _unit(module: str) -> str | None:
    """The first package below ``pyorps`` that ``module`` names, else ``None``."""
    parts = module.split(".")
    if parts[0] == PACKAGE and len(parts) > 1:
        return parts[1]
    return None


def layering_violations(source: str, module_parts: list[str], is_package: bool,
                        allowed=None, label: str | None = None) -> list[str]:
    """One message per import in ``source`` that breaks the layering."""
    allowed = ALLOWED_RESEARCH_IMPORTS if allowed is None else allowed
    own = module_parts[1] if len(module_parts) > 1 else None
    others = allowed.get(own, frozenset())
    may_import = others | {own}
    where = label or "/".join(module_parts)
    tree = ast.parse(source)
    found: dict[tuple[int, str], str] = {}
    for line, module in _imported_modules(tree, module_parts, is_package):
        target = _unit(module)
        if target not in allowed or target in may_import:
            continue
        who = "package root" if own is None else f"'{own}'"
        found.setdefault(
            (line, target),
            f"{where}:{line} imports {module}: {who} may not import the "
            f"research package '{target}' (allowed: "
            f"{', '.join(sorted(others)) or 'none'})")
    return [found[key] for key in sorted(found)]


def test_package_layering():
    problems = []
    files = sorted(PACKAGE_DIR.rglob("*.py"))
    assert files, f"no sources found under {PACKAGE_DIR}"
    for path in files:
        parts, is_package = _module_parts(path)
        source = path.read_text(encoding="utf-8")
        label = path.relative_to(PACKAGE_DIR.parent).as_posix()
        problems += layering_violations(source, parts, is_package, label=label)
    assert not problems, (
        "import layering broken (core <- collector <- certify / costmodel; "
        "siting alone):\n" + "\n".join(problems))


def test_allowed_edges_have_no_cycle():
    """The table itself must stay a one-way layering."""
    assert set().union(*ALLOWED_RESEARCH_IMPORTS.values()) <= set(
        ALLOWED_RESEARCH_IMPORTS)
    done: set[str] = set()

    def visit(node: str, stack: tuple[str, ...]):
        assert node not in stack, f"cycle in ALLOWED_RESEARCH_IMPORTS: {stack + (node,)}"
        if node in done:
            return
        for dep in ALLOWED_RESEARCH_IMPORTS[node]:
            visit(dep, stack + (node,))
        done.add(node)

    for name in ALLOWED_RESEARCH_IMPORTS:
        visit(name, ())


@pytest.mark.parametrize("source, parts, is_package, expect", [
    # absolute, at the top of the module: the old graph -> certify edge
    ("from pyorps.certify.windows import drain\n",
     ["pyorps", "graph", "mixed_legs"], False, ["certify"]),
    # inside a function
    ("def f():\n    from pyorps.certify.rv import relax_and_verify\n",
     ["pyorps", "collector", "voltage_rv"], False, ["certify"]),
    # relative, from a package __init__ and from a module
    ("from ..certify import rv\n", ["pyorps", "collector", "x"], False,
     ["certify"]),
    ("from .. import siting\n", ["pyorps", "costmodel"], True, ["siting"]),
    # a submodule imported through the package root
    ("from pyorps import collector\n", ["pyorps", "utils", "x"], False,
     ["collector"]),
    ("import pyorps.siting.bounds as b\n", ["pyorps", "certify", "x"], False,
     ["siting"]),
    # allowed or unrelated imports stay silent
    ("from pyorps.collector.design import reprice\n",
     ["pyorps", "certify", "x"], False, []),
    ("from pyorps.certify.windows import drain\n",
     ["pyorps", "certify", "y"], False, []),
    ("from .windows import drain\nfrom pyorps.graph.drain import drain\n",
     ["pyorps", "certify", "z"], False, []),
    ("from .costmodel import x\n", ["pyorps", "gui", "callbacks", "io"], False,
     []),
    ("from pyorps.certify import rv\n", ["pyorps", "costmodel", "x"], False,
     ["certify"]),
    ("import numpy\n", ["pyorps", "graph", "x"], False, []),
])
def test_checker_sees_every_import_form(source, parts, is_package, expect):
    got = layering_violations(source, parts, is_package)
    assert [m.split("research package '")[1].split("'")[0] for m in got] == expect
    for message in got:
        assert message.startswith("/".join(parts) + ":")
        assert " imports pyorps." in message
