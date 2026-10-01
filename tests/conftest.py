# tests/conftest.py
import sys
import os
from pathlib import Path

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Shared test helpers that live next to the tests (e.g. exactness_referee.py)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Tests that need files outside tests/ are collected only when those files are there: a run of the
# installed wheel (CI copies tests/ away from the source tree) or a clone without the local case study.
_ROOT = Path(__file__).resolve().parent.parent
collect_ignore = []
if not (_ROOT / "setup.py").exists():
    collect_ignore += ["test_build_provenance.py"]          # reads the source tree and the built-extension manifest
if not (_ROOT / "case_studies" / "runkel_free_siting" / "config" / "voltage_2026.yaml").exists():
    collect_ignore += ["test_collector/test_voltage.py", "test_collector/test_voltage_batch.py",
                       "test_collector/test_voltage_plant.py"]  # read the (untracked) Runkel case-study config
