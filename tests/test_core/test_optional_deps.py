"""PyYAML is an optional dependency: one helper, one clear error when it is missing."""
import sys

import pytest

from pyorps.core.exceptions import MissingDependencyError
from pyorps.core.optional_deps import import_yaml


def test_import_yaml_returns_the_module_when_installed():
    pytest.importorskip("yaml")
    import yaml

    assert import_yaml() is yaml


@pytest.fixture
def no_yaml(monkeypatch):
    """Make ``import yaml`` fail as it does in an install without the ``yaml`` extra."""
    monkeypatch.setitem(sys.modules, "yaml", None)


def _assert_names_the_extra(exc_info):
    assert exc_info.value.extra == "yaml"
    assert "pyorps[yaml]" in str(exc_info.value)
    assert isinstance(exc_info.value, ImportError)


def test_helper_raises_missing_dependency_error(no_yaml):
    with pytest.raises(MissingDependencyError) as exc_info:
        import_yaml()
    _assert_names_the_extra(exc_info)


def test_core_profile_loader_names_the_extra(no_yaml, tmp_path):
    from pyorps.core.infrastructure_profile import InfrastructureProfile

    path = tmp_path / "profile.yaml"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(MissingDependencyError) as exc_info:
        InfrastructureProfile.load(str(path))
    _assert_names_the_extra(exc_info)


def test_core_profile_save_without_pyyaml_leaves_no_file(no_yaml, tmp_path):
    """The error comes before the file is opened: no empty or truncated profile is left behind."""
    from pyorps.core.infrastructure_profile import InfrastructureProfile

    profile = InfrastructureProfile.from_dict({
        "name": "test", "description": "test", "soft_angle_limit_deg": 30.0,
        "hard_angle_limit_deg": 60.0, "angle_cost_function": "linear"})
    target = tmp_path / "profile.yaml"
    target.write_text("kept: true", encoding="utf-8")
    with pytest.raises(MissingDependencyError) as exc_info:
        profile.save(str(target))
    _assert_names_the_extra(exc_info)
    assert target.read_text(encoding="utf-8") == "kept: true"
    profile.save(str(tmp_path / "profile.json"))  # JSON needs no PyYAML
    assert (tmp_path / "profile.json").stat().st_size > 0


def test_collector_loader_names_the_extra(no_yaml, tmp_path):
    from pyorps.collector.voltage import voltage_model_from_yaml

    path = tmp_path / "voltage.yaml"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(MissingDependencyError) as exc_info:
        voltage_model_from_yaml(path, u_kv=33, cable_mm2=[240])
    _assert_names_the_extra(exc_info)


def test_costmodel_loader_names_the_extra(no_yaml, tmp_path):
    from pyorps.costmodel.catalogue import CostCatalogue

    path = tmp_path / "catalogue.yaml"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(MissingDependencyError) as exc_info:
        CostCatalogue.load(path)
    _assert_names_the_extra(exc_info)


def test_siting_loader_names_the_extra(no_yaml, tmp_path):
    from pyorps.siting.data_manifest import DataManifest

    path = tmp_path / "manifest.yaml"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(MissingDependencyError) as exc_info:
        DataManifest.load(path)
    _assert_names_the_extra(exc_info)
