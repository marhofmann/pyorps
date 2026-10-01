"""Shared setup for the collector tests.

Registers the ``slow`` marker here rather than in ``pyproject.toml``: the
long oracle sweeps (four turbines, the strict MILP, the n = 4 enumerators)
are skipped unless the run selects them with ``-m slow`` or sets
``PYORPS_RUN_SLOW=1``, so the default run stays well under a minute.
"""
import os

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: long oracle sweeps; run with -m slow or "
                   "PYORPS_RUN_SLOW=1")


def pytest_collection_modifyitems(config, items):
    wanted = "slow" in (config.getoption("-m") or "")
    if wanted or os.environ.get("PYORPS_RUN_SLOW") == "1":
        return
    skip = pytest.mark.skip(reason="slow oracle sweep; run with -m slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
