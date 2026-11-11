"""One exception root for the whole library, and complete public name lists.

* Every exception class pyorps defines derives from ``PyorpsError``; every
  warning class from ``PyorpsWarning``.
* The classes that replace a builtin exception also derive from that builtin,
  so ``except ValueError`` (and the other old clauses) keep working.
* Errors cross process boundaries, so each class must pickle.
* ``__all__`` of the four packages lists only names that exist, lists each name
  once, and lists every exception class.

The front-end packages (gui, webviz) are skipped on purpose (see ``_OUT_OF_SCOPE``).
"""
import importlib
import inspect
import pickle
import pkgutil
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import from_origin

import pyorps
from pyorps.core import exceptions as pex
from pyorps.core.exceptions import (
    AlgorithmNotImplementedError,
    ConsistencyCheckError,
    FeatureNotImplementedError,
    FormatError,
    InputFileNotFoundError,
    InsufficientMemoryError,
    InvalidInputError,
    InvalidTypeError,
    MissingDependencyError,
    MissingKeyError,
    NoPathFoundError,
    NumericalPrecisionError,
    PairwiseError,
    PyorpsError,
    PyorpsRuntimeError,
    PyorpsWarning,
    RasterShapeError,
)
from pyorps.core.warning_types import PyorpsWarning as _WarningRoot
from pyorps.graph.api.graph_library_api import reject_unknown_options
from pyorps.graph.path_finder import PathFinder, get_graph_api_class
from pyorps.io.provenance import ProvenanceMismatch
from pyorps.utils.neighborhood import get_neighborhood_steps

#: Packages whose exception classes are covered here.
_IN_SCOPE = ("core", "graph", "raster", "io", "utils",
             "certify", "collector", "costmodel", "siting")
#: Front-end packages: they keep their own exception classes.
_OUT_OF_SCOPE = ("pyorps.gui", "pyorps.webviz")

#: The new classes, with the builtin each one also derives from.
DUAL_BASE = [
    (InvalidInputError, ValueError),
    (InvalidTypeError, TypeError),
    (MissingKeyError, KeyError),
    (MissingDependencyError, ImportError),
    (FeatureNotImplementedError, NotImplementedError),
    (PyorpsRuntimeError, RuntimeError),
    (InsufficientMemoryError, MemoryError),
    (InputFileNotFoundError, FileNotFoundError),
    (ConsistencyCheckError, AssertionError),
    (NumericalPrecisionError, ArithmeticError),
]

CRS = "EPSG:32632"
N = 60
SRC = (5.5, N - 10.5)
TGT = (50.5, N - 50.5)


# ---------------------------------------------------------------------------
# Finding the classes
# ---------------------------------------------------------------------------

def _import_in_scope_modules() -> None:
    """Import every module of the in-scope packages (optional ones may not load)."""
    for name in _IN_SCOPE:
        package = importlib.import_module(f"pyorps.{name}")
        for info in pkgutil.walk_packages(package.__path__, f"pyorps.{name}."):
            try:
                importlib.import_module(info.name)
            except ImportError:      # an optional dependency (GPU, ...) is not installed
                continue


def _defined_error_classes() -> dict[str, type]:
    """``qualified name -> class`` for every Exception/Warning defined in pyorps.*"""
    _import_in_scope_modules()
    found = {}
    for module_name, module in list(sys.modules.items()):
        if (module is None or not module_name.startswith("pyorps.")
                or module_name.startswith(_OUT_OF_SCOPE)):
            continue
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (issubclass(obj, BaseException) and obj.__module__ == module_name
                    and not obj.__module__.startswith(_OUT_OF_SCOPE)):
                found[f"{obj.__module__}.{obj.__qualname__}"] = obj
    return found


def _is_warning(cls: type) -> bool:
    return issubclass(cls, Warning)


ALL_CLASSES = _defined_error_classes()
EXCEPTION_CLASSES = {k: v for k, v in ALL_CLASSES.items() if not _is_warning(v)}
WARNING_CLASSES = {k: v for k, v in ALL_CLASSES.items() if _is_warning(v)}


def _classes_of_module(module) -> list[type]:
    return [obj for _, obj in inspect.getmembers(module, inspect.isclass)
            if issubclass(obj, BaseException) and obj.__module__ == module.__name__]


# ---------------------------------------------------------------------------
# The hierarchy
# ---------------------------------------------------------------------------

def test_the_scan_finds_the_known_classes():
    """Guard against a scan that silently finds nothing."""
    names = {cls.__name__ for cls in EXCEPTION_CLASSES.values()}
    assert {"PyorpsError", "NoPathFoundError", "EnsembleError", "ProvenanceMismatch",
            "ThinForbiddenFeatureError", "SealedOpeningError", "InvalidInputError",
            "WFSServerError"} <= names
    assert {"PyorpsWarning", "ThinForbiddenFeatureWarning", "SealedOpeningWarning",
            "ForbiddenBurnRoutingAssessment"} <= {c.__name__ for c in WARNING_CLASSES.values()}


@pytest.mark.parametrize("qualified_name", sorted(EXCEPTION_CLASSES))
def test_every_exception_class_derives_from_the_root(qualified_name):
    assert issubclass(EXCEPTION_CLASSES[qualified_name], PyorpsError)


def test_every_exceptions_module_class_derives_from_the_root():
    for cls in _classes_of_module(pex):
        assert issubclass(cls, PyorpsError), cls


@pytest.mark.parametrize("qualified_name", sorted(WARNING_CLASSES))
def test_every_warning_class_derives_from_the_warning_root(qualified_name):
    cls = WARNING_CLASSES[qualified_name]
    assert issubclass(cls, PyorpsWarning)
    assert issubclass(cls, UserWarning)          # the bases the classes had before


def test_the_warning_root_is_one_class_in_two_places():
    assert pex.PyorpsWarning is _WarningRoot is pyorps.PyorpsWarning
    assert issubclass(PyorpsWarning, UserWarning)
    assert not issubclass(PyorpsWarning, PyorpsError)


def test_the_warning_root_filters_every_pyorps_warning():
    from pyorps.raster.thinness import ThinForbiddenFeatureWarning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("ignore", PyorpsWarning)
        warnings.warn("quiet", ThinForbiddenFeatureWarning)
        warnings.warn("loud", UserWarning)
    assert [str(w.message) for w in caught] == ["loud"]


def test_classes_that_had_a_builtin_base_keep_it():
    from pyorps.raster.thinness import SealedOpeningError, ThinForbiddenFeatureError
    for cls in (ThinForbiddenFeatureError, SealedOpeningError, ProvenanceMismatch):
        assert issubclass(cls, ValueError), cls
        assert issubclass(cls, PyorpsError), cls
    assert issubclass(pyorps.EnsembleError, PyorpsError)


def test_the_old_classes_did_not_gain_builtin_bases():
    """``except ValueError`` and friends must not start catching these."""
    for cls in (AlgorithmNotImplementedError, NoPathFoundError, PairwiseError,
                RasterShapeError, FormatError):
        assert not issubclass(cls, (ValueError, TypeError, KeyError, ImportError,
                                    NotImplementedError, RuntimeError, MemoryError)), cls


# ---------------------------------------------------------------------------
# The dual-base classes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls, builtin", DUAL_BASE, ids=[c.__name__ for c, _ in DUAL_BASE])
class TestDualBase:
    def test_derives_from_both(self, cls, builtin):
        assert issubclass(cls, PyorpsError)
        assert issubclass(cls, builtin)
        assert cls.__mro__.index(PyorpsError) < cls.__mro__.index(builtin)

    def test_caught_by_both(self, cls, builtin):
        for catching in (PyorpsError, builtin, Exception):
            with pytest.raises(catching):
                raise cls("the message")

    def test_message_is_the_builtins_message(self, cls, builtin):
        """Byte-identical text: the GUI and the tests match on it."""
        assert str(cls("the message")) == str(builtin("the message"))

    def test_pickles_and_round_trips_the_message(self, cls, builtin):
        original = cls("the message")
        clone = pickle.loads(pickle.dumps(original))
        assert type(clone) is cls
        assert str(clone) == str(original)
        assert clone.args == original.args


def test_missing_dependency_error_carries_the_extra():
    error = MissingDependencyError("needs networkx", extra="graph", name="networkx")
    assert (error.extra, error.name) == ("graph", "networkx")
    assert MissingDependencyError("plain").extra is None
    clone = pickle.loads(pickle.dumps(error))
    assert (clone.extra, clone.name, str(clone)) == ("graph", "networkx", "needs networkx")


def test_missing_key_error_quotes_its_message_like_a_key_error():
    assert str(MissingKeyError("no route 'a'")) == str(KeyError("no route 'a'"))


@pytest.mark.parametrize("error", [
    NoPathFoundError(3, 9),
    NoPathFoundError(3, 9, add_message=" Extra."),
    RasterShapeError((1, 2, 3, 4)),
    AlgorithmNotImplementedError("dijkstra", "cython"),
    PairwiseError(),
    ProvenanceMismatch("the field", ["seeds.a", "storage.b"]),
], ids=lambda e: type(e).__name__)
def test_classes_with_their_own_constructor_still_pickle(error):
    clone = pickle.loads(pickle.dumps(error))
    assert type(clone) is type(error)
    assert str(clone) == str(error)
    assert clone.__dict__ == error.__dict__


def test_custom_constructor_arguments_are_kept_as_attributes():
    error = NoPathFoundError(3, 9, add_message=" Extra.")
    assert (error.source, error.target) == (3, 9)
    assert str(error) == ("No path found from 3 to 9! Choose different source and "
                          "target or increase buffer! Extra.")
    assert RasterShapeError((1, 2)).raster_shape == (1, 2)
    assert AlgorithmNotImplementedError("a", "b").graph_library == "b"
    assert str(AlgorithmNotImplementedError("a", "b")) == "Algorithm a for b not supported!"
    assert ProvenanceMismatch("w", ["x"]).differences == ["x"]


# ---------------------------------------------------------------------------
# One behavioural test per boundary: caught by the root AND by the old type
# ---------------------------------------------------------------------------

def _caught_by_root_and_old_type(old_type, call):
    """``call`` must raise something both ``PyorpsError`` and ``old_type`` catch."""
    for catching in (PyorpsError, old_type):
        with pytest.raises(catching) as info:
            call()
    return info.value


def _finder(**kwargs):
    raster = np.random.default_rng(0).integers(1, 10, size=(N, N)).astype(np.uint16)
    args = dict(dataset_source=raster, source_coords=SRC, target_coords=TGT,
                search_space_buffer_m=15, crs=CRS,
                transform=from_origin(0.0, float(N), 1.0, 1.0))
    args.update(kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PathFinder(**args)


def test_unknown_backend_name():
    error = _caught_by_root_and_old_type(ValueError, lambda: get_graph_api_class("nonsense"))
    assert isinstance(error, InvalidInputError)
    assert "Unsupported graph API" in str(error)


def test_unimplemented_backend():
    error = _caught_by_root_and_old_type(NotImplementedError,
                                         lambda: get_graph_api_class("cugraph"))
    assert isinstance(error, FeatureNotImplementedError)


def test_point_outside_the_raster():
    finder = _finder(search_space_buffer_m=5)
    outside = (N - 2.5, N - 2.5)     # inside the raster, outside the 5 m search window
    with finder.cost_field(SRC) as field:
        error = _caught_by_root_and_old_type(ValueError, lambda: field.costs_to([outside]))
    assert isinstance(error, InvalidInputError)
    assert "outside" in str(error)


def test_bad_neighbourhood():
    error = _caught_by_root_and_old_type(ValueError, lambda: get_neighborhood_steps("r999"))
    assert isinstance(error, InvalidInputError)
    _caught_by_root_and_old_type(TypeError, lambda: get_neighborhood_steps(None))
    # PathFinder reports it as a PyorpsError, with the original as its cause.
    with pytest.raises(PyorpsError) as info:
        _finder(neighborhood_str="r999")
    assert isinstance(info.value.__cause__, ValueError)


@pytest.mark.parametrize("resolution", [0, -2.0, float("nan")])
def test_non_positive_resolution(resolution):
    from pyorps.raster.rasterizer import GeoRasterizer
    error = _caught_by_root_and_old_type(
        ValueError, lambda: GeoRasterizer._check_resolution(resolution))
    assert isinstance(error, InvalidInputError)
    assert "resolution_in_m" in str(error)
    _caught_by_root_and_old_type(TypeError, lambda: GeoRasterizer._check_resolution("1"))


def test_out_of_range_cost():
    import geopandas as gpd
    from shapely.geometry import box

    from pyorps.io.geo_dataset import InMemoryVectorDataset
    from pyorps.raster.rasterizer import GeoRasterizer
    frame = gpd.GeoDataFrame({"category": ["a"]}, geometry=[box(0, 0, 4, 4)], crs=CRS)
    rasterizer = GeoRasterizer(InMemoryVectorDataset(frame, crs=CRS),
                               {"category": {"a": 70000}})
    # FormatError was already a PyorpsError; it has no builtin base to keep.
    error = _caught_by_root_and_old_type(FormatError, rasterizer.rasterize)
    assert "0 to 65535" in str(error)


def test_misspelled_keyword():
    error = _caught_by_root_and_old_type(
        TypeError, lambda: _finder(neighbourhood_str="r1"))
    assert isinstance(error, InvalidTypeError)
    assert "did you mean 'neighborhood_str'" in str(error)
    error = _caught_by_root_and_old_type(
        TypeError, lambda: reject_unknown_options({"detla": 1}, ("delta",), "here()"))
    assert isinstance(error, InvalidTypeError)


def test_missing_optional_backend(monkeypatch):
    monkeypatch.setitem(sys.modules, "networkx", None)
    monkeypatch.delitem(sys.modules, "pyorps.graph.api.networkx_api", raising=False)
    error = _caught_by_root_and_old_type(ImportError, lambda: get_graph_api_class("networkx"))
    assert isinstance(error, MissingDependencyError)
    assert error.extra == "graph"
    assert 'pip install "pyorps[graph]"' in str(error)
    assert isinstance(error.__cause__, ImportError)


def test_a_closed_session_is_a_pyorps_runtime_error():
    finder = _finder()
    field = finder.cost_field(SRC)
    field.close()
    error = _caught_by_root_and_old_type(RuntimeError, lambda: field.costs_to([TGT]))
    assert isinstance(error, PyorpsRuntimeError)
    assert str(error) == "CostField is closed"


def test_not_enough_memory_is_a_pyorps_memory_error():
    from pyorps.utils.resources import resource_budget
    error = _caught_by_root_and_old_type(
        MemoryError, lambda: resource_budget(1 << 40, free_bytes=1 << 30))
    assert isinstance(error, InsufficientMemoryError)


def test_a_missing_dict_key_is_a_pyorps_key_error():
    finder = _finder()
    error = _caught_by_root_and_old_type(
        KeyError, lambda: finder.find_route(simplify={"tolerance": 1.0}))
    assert isinstance(error, MissingKeyError)
    assert "must contain key 'method'" in str(error)


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

PACKAGES = ["pyorps", "pyorps.core", "pyorps.io", "pyorps.utils"]
#: The research packages keep their classes in their own namespaces; the root
#: package does not import them (they are optional and heavy).
_RESEARCH = ("pyorps.certify.", "pyorps.collector.", "pyorps.costmodel.", "pyorps.siting.")


@pytest.mark.parametrize("package_name", PACKAGES)
class TestAllLists:
    def test_every_name_exists(self, package_name):
        package = importlib.import_module(package_name)
        missing = [n for n in package.__all__ if not hasattr(package, n)]
        assert not missing

    def test_no_duplicates(self, package_name):
        names = importlib.import_module(package_name).__all__
        assert len(names) == len(set(names)), sorted(n for n in names if names.count(n) > 1)

    def test_every_name_is_a_string(self, package_name):
        assert all(isinstance(n, str) for n in importlib.import_module(package_name).__all__)

    def test_star_import_exposes_exactly_all(self, package_name):
        package = importlib.import_module(package_name)
        namespace: dict = {}
        exec(f"from {package_name} import *", namespace)          # noqa: S102
        namespace.pop("__builtins__", None)
        assert set(namespace) == set(package.__all__)

    def test_lists_every_exception_class_defined_in_its_own_modules(self, package_name):
        package = importlib.import_module(package_name)
        prefix = package_name + "."
        own = {cls.__name__ for qualified, cls in ALL_CLASSES.items()
               if qualified.startswith(prefix) and not cls.__name__.startswith("_")
               and not (package_name == "pyorps" and qualified.startswith(_RESEARCH))}
        assert own <= set(package.__all__), sorted(own - set(package.__all__))


def test_the_root_package_lists_every_exception_and_warning_class():
    missing = sorted({cls.__name__ for qualified, cls in ALL_CLASSES.items()
                      if not cls.__name__.startswith("_")
                      and not qualified.startswith(_RESEARCH)} - set(pyorps.__all__))
    assert not missing


def test_core_lists_every_class_of_the_exceptions_module():
    names = {cls.__name__ for cls in _classes_of_module(pex)} | {"PyorpsWarning"}
    assert names <= set(pyorps.core.__all__)


@pytest.mark.parametrize("package_name", ["pyorps.certify", "pyorps.collector",
                                          "pyorps.costmodel", "pyorps.siting"])
def test_research_packages_list_their_exception_classes(package_name):
    package = importlib.import_module(package_name)
    prefix = package_name + "."
    own = {cls.__name__ for qualified, cls in ALL_CLASSES.items()
           if qualified.startswith(prefix) and not cls.__name__.startswith("_")}
    assert own <= set(package.__all__), sorted(own - set(package.__all__))


def test_io_lists_the_wfs_errors_and_the_raster_loader_functions():
    assert {"WFSServerError", "WFSRequestRejectedError", "ProvenanceMismatch",
            "read_raster_files", "combine_rasters", "create_combined_raster_dataset",
            "find_raster_files", "validate_raster_compatibility"} <= set(pyorps.io.__all__)


def test_new_exception_class_must_be_exported():
    """Fails when a class is added to exceptions.py and forgotten in the name lists."""
    defined = {cls.__name__ for cls in _classes_of_module(pex)}
    defined |= {cls.__name__ for cls in _classes_of_module(importlib.import_module(
        "pyorps.core.warning_types"))}
    assert defined, "the scan found no classes"
    for where, names in (("pyorps.core.exceptions.__all__", pex.__all__),
                         ("pyorps.core.__all__", pyorps.core.__all__),
                         ("pyorps.__all__", pyorps.__all__)):
        assert not defined - set(names), f"missing from {where}: {sorted(defined - set(names))}"
    for name in defined:
        assert getattr(pyorps, name) is getattr(pex, name)


def test_the_docs_page_lists_every_exception_class():
    page = Path(pyorps.__file__).parents[1] / "docs" / "source" / "api" / "exceptions.md"
    if not page.exists():
        pytest.skip("docs are not part of this checkout")
    text = page.read_text(encoding="utf-8")
    names = {cls.__name__ for cls in EXCEPTION_CLASSES.values()}
    names |= {cls.__name__ for cls in WARNING_CLASSES.values()}
    missing = sorted(n for n in names if not re.search(rf"\b{n}\b", text))
    assert not missing, f"docs/source/api/exceptions.md does not mention {missing}"
