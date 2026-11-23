"""Lazy imports of optional third-party packages, with one clear error when one is missing.

Nothing here imports a heavy package at import time. ``import_yaml`` imports PyYAML
when it is called, so a plain install can use everything that does not read or write
YAML.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from pyorps.core.exceptions import MissingDependencyError


def import_yaml() -> ModuleType:
    """Return the ``yaml`` module (PyYAML).

    PyYAML is optional: install it with ``pip install "pyorps[yaml]"`` (the ``gui``,
    ``case_studies`` and ``full`` extras include it).

    Raises
    ------
    MissingDependencyError
        PyYAML cannot be imported. Its ``extra`` attribute is ``"yaml"``.
    """
    try:
        return importlib.import_module("yaml")
    except ImportError as exc:
        raise MissingDependencyError(
            "PyYAML is needed to read or write this YAML file but is not installed: "
            'pip install "pyorps[yaml]"',
            extra="yaml", name="yaml") from exc
