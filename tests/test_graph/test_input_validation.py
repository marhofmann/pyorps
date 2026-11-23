"""Input validation at the public routing boundary (audit T4 #1, #2, #7, #14,
#16, #22, #24).

Every case here used to fail deep inside, with an unrelated message, or kill
the interpreter. Inputs that crashed the interpreter before the fix run in a
child process, so a regression fails the test instead of killing pytest.
"""
import importlib.util
import json
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest
from rasterio.transform import from_origin

from pyorps import PathFinder
from pyorps.core.exceptions import AlgorithmNotImplementedError, PyorpsError
from pyorps.graph.api.cython_api import CythonAPI
from pyorps.graph.path_finder import GRAPH_APIS, get_graph_api_class
from pyorps.raster.handler import RasterHandler
from pyorps.utils.neighborhood import get_neighborhood_steps

CRS = "EPSG:32632"
ORIGIN_X, ORIGIN_Y = 500000.0, 5600000.0
N = 60
SRC = (500005.5, 5599990.5)
TGT = (500050.5, 5599950.5)


def _raster(dtype=np.uint16):
    return np.random.default_rng(0).integers(1, 10, size=(N, N)).astype(dtype)


def _finder(**kwargs):
    args = dict(dataset_source=_raster(), source_coords=SRC, target_coords=TGT,
                search_space_buffer_m=10, crs=CRS,
                transform=from_origin(ORIGIN_X, ORIGIN_Y, 1, 1))
    args.update(kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PathFinder(**args)


# ---------------------------------------------------------------------------
# Child process for every input that crashed the interpreter (exit code 139)
# ---------------------------------------------------------------------------

_CHILD = r'''
import json, sys, warnings
warnings.simplefilter("ignore")
import numpy as np
from rasterio.transform import from_origin
from pyorps import PathFinder
from pyorps.core.exceptions import PyorpsError
from pyorps.graph.api.cython_api import CythonAPI
from pyorps.utils.neighborhood import get_neighborhood_steps

RASTER = np.random.default_rng(0).integers(1, 10, size=(60, 60)).astype(np.uint16)
TR = from_origin(500000.0, 5600000.0, 1, 1)
SRC = (500005.5, 5599990.5)
TGT = (500050.5, 5599950.5)
NAN, INF = float("nan"), float("inf")


def finder(**kw):
    args = dict(dataset_source=RASTER, source_coords=SRC, target_coords=TGT,
                search_space_buffer_m=10, crs="EPSG:32632", transform=TR)
    args.update(kw)
    return PathFinder(**args)


def route(**kw):
    pairwise = kw.pop("pairwise", False)
    result = finder(**kw).find_route(pairwise=pairwise)
    return getattr(result, "total_length", result)


def session(points):
    return finder().search_session().route(points).total_length


def cython(steps, replace_steps=False):
    api = CythonAPI(RASTER, get_neighborhood_steps("r2") if replace_steps else steps)
    if replace_steps:
        api.steps = steps
    return api.shortest_path(61, 3000)


CASES = {
    "src_nan": lambda: route(source_coords=(NAN, SRC[1])),
    "src_pinf": lambda: route(source_coords=(INF, SRC[1])),
    "src_ninf": lambda: route(source_coords=(SRC[0], -INF)),
    "tgt_list_nan": lambda: route(target_coords=[TGT, (NAN, TGT[1])]),
    "public_nan": lambda: finder().get_node_indices_from_coords((NAN, SRC[1])),
    "lonlat": lambda: route(source_coords=(8.5, 50.1)),
    "out_east": lambda: route(source_coords=(500060.5, SRC[1])),
    "out_west": lambda: route(source_coords=(499999.5, SRC[1])),
    "out_north": lambda: route(source_coords=(SRC[0], 5600000.5)),
    "out_south": lambda: route(target_coords=(TGT[0], 5599939.5)),
    "last_cell": lambda: route(target_coords=(500059.5, 5599940.5)),
    "multi_out": lambda: route(source_coords=[SRC, (500006.5, 5599985.5)],
                               target_coords=[TGT, (500070.5, TGT[1])]),
    "pairwise_out": lambda: route(source_coords=[SRC, (500006.5, 5599985.5)],
                                  target_coords=[TGT, (500070.5, TGT[1])],
                                  pairwise=True),
    "session_nan": lambda: session([SRC, (NAN, TGT[1])]),
    "session_out": lambda: session([SRC, (500070.5, TGT[1])]),
    "neigh_0": lambda: route(neighborhood_str=0),
    "neigh_none": lambda: route(neighborhood_str=None),
    "neigh_empty": lambda: route(neighborhood_str=""),
    "cy_steps_none": lambda: cython(None),
    "cy_steps_empty": lambda: cython(np.zeros((0, 2), dtype=np.int8)),
    "cy_steps_shape": lambda: cython(np.zeros((4, 1), dtype=np.int8)),
    "cy_steps_replaced_none": lambda: cython(None, replace_steps=True),
}

for name in sys.argv[1:]:
    try:
        out = CASES[name]()
        rec = {"case": name, "ok": True, "result": repr(out)[:200]}
    except Exception as exc:
        rec = {"case": name, "ok": False, "type": type(exc).__name__,
               "pyorps": isinstance(exc, PyorpsError), "msg": str(exc)}
    print(json.dumps(rec), flush=True)
'''

_CRASH_CASES = (
    "src_nan", "src_pinf", "src_ninf", "tgt_list_nan", "public_nan", "lonlat",
    "out_east", "out_west", "out_north", "out_south", "last_cell", "multi_out",
    "pairwise_out", "session_nan", "session_out", "neigh_0", "neigh_none",
    "neigh_empty", "cy_steps_none", "cy_steps_empty", "cy_steps_shape",
    "cy_steps_replaced_none",
)


@pytest.fixture(scope="module")
def child_results():
    """Run every crash-prone case in ONE child process; one JSON line each."""
    env = dict(os.environ, NUMBA_NUM_THREADS="2", OMP_NUM_THREADS="2",
               PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, *_CRASH_CASES],
        capture_output=True, text=True, timeout=900, env=env)
    records = {}
    for line in proc.stdout.splitlines():
        if line.startswith("{"):
            rec = json.loads(line)
            records[rec["case"]] = rec
    return records, proc


def _record(child_results, case):
    records, proc = child_results
    if case not in records:
        pytest.fail(
            f"child process died before case {case!r} finished "
            f"(return code {proc.returncode}); stderr tail:\n"
            f"{proc.stderr[-2000:]}")
    return records[case]


def _raised(child_results, case):
    rec = _record(child_results, case)
    assert not rec["ok"], f"{case} should raise, returned {rec.get('result')}"
    assert rec["pyorps"], f"{case}: {rec['type']} is not a PyorpsError: {rec['msg']}"
    return rec["msg"]


# --- #1 coordinates ---------------------------------------------------------

@pytest.mark.parametrize("case, shown", [
    ("src_nan", "nan"), ("src_pinf", "inf"), ("src_ninf", "-inf"),
    ("tgt_list_nan", "nan"), ("public_nan", "nan"),
    ("session_nan", "nan"),
])
def test_non_finite_coordinate_raises(child_results, case, shown):
    msg = _raised(child_results, case)
    assert "not finite" in msg
    assert shown in msg
    assert "Allowed: finite" in msg


@pytest.mark.parametrize("case, point", [
    ("out_east", "500060.5"), ("out_west", "499999.5"),
    ("out_north", "5600000.5"), ("out_south", "5599939.5"),
    ("multi_out", "500070.5"), ("pairwise_out", "500070.5"),
    ("session_out", "500070.5"),
])
def test_point_outside_raster_raises_with_bounds_and_crs(child_results, case,
                                                         point):
    msg = _raised(child_results, case)
    assert "lies outside the search window" in msg
    assert point in msg
    assert "full raster bounds = (500000.000, 5599940.000, 500060.000, " \
           "5600000.000)" in msg
    assert "raster CRS = EPSG:32632" in msg
    assert "longitude/latitude" not in msg


def test_lonlat_on_projected_raster_hints_at_crs_mismatch(child_results):
    msg = _raised(child_results, "lonlat")
    assert "(8.5, 50.1)" in msg
    assert "raster CRS = EPSG:32632" in msg
    assert "longitude/latitude" in msg
    assert "CRS mismatch" in msg


def test_point_on_last_valid_cell_still_routes(child_results):
    rec = _record(child_results, "last_cell")
    assert rec["ok"], rec
    assert np.isfinite(float(rec["result"]))


def test_coords_to_indices_keeps_raw_indices_by_default():
    """SearchSession and the GUI detect outside points from raw indices."""
    finder = _finder()
    handler = finder.raster_handler
    rows, cols = handler.data.shape[-2:]
    idx = handler.coords_to_indices([(500070.5, TGT[1])])
    assert idx[0][1] >= cols or idx[0][0] >= rows or (idx[0] < 0).any()
    with pytest.raises(PyorpsError, match="outside the search window"):
        handler.coords_to_indices([(500070.5, TGT[1])], check_bounds=True)
    with pytest.raises(PyorpsError, match="not finite"):
        handler.coords_to_indices([(float("nan"), TGT[1])])


# --- #2 neighbourhood that defines no steps ------------------------------------

def test_neighborhood_zero_is_r0(child_results):
    rec = _record(child_results, "neigh_0")
    assert rec["ok"], rec
    np.testing.assert_array_equal(_finder(neighborhood_str=0).steps,
                                  get_neighborhood_steps("r0"))


@pytest.mark.parametrize("case", ["neigh_none", "neigh_empty"])
def test_missing_neighborhood_raises(child_results, case):
    msg = _raised(child_results, case)
    assert "defines no neighbourhood" in msg
    assert "'r2'" in msg


def test_missing_neighborhood_allowed_on_raster_fim():
    """The eikonal solver has no neighbourhood; None keeps meaning that."""
    finder = PathFinder.__new__(PathFinder)
    finder.graph_api_name = "raster_fim"
    assert finder._steps_from_neighborhood(None) is None


@pytest.mark.parametrize("case", ["cy_steps_none", "cy_steps_empty",
                                  "cy_steps_shape", "cy_steps_replaced_none"])
def test_cython_wrapper_rejects_bad_steps(child_results, case):
    msg = _raised(child_results, case)
    assert "steps" in msg
    assert "int8 numpy array of shape (n, 2)" in msg


def test_cython_wrapper_rejects_wrong_steps_dtype():
    with pytest.raises(PyorpsError, match="dtype 'int64'"):
        CythonAPI(_raster(), get_neighborhood_steps("r1").astype(np.int64))


# --- #22 neighbourhood strings and numbers ------------------------------------

@pytest.mark.parametrize("value", ["r999", "x", "r2.5", "r2abc", 2.7, True])
def test_invalid_neighborhood_raises_with_allowed_forms(value):
    with pytest.raises(PyorpsError, match="0 <= K <= 127") as info:
        _finder(neighborhood_str=value)
    assert repr(value) in str(info.value)
    assert isinstance(info.value.__cause__, (TypeError, ValueError))


@pytest.mark.parametrize("value, same_as", [
    ("r3", "r3"), ("R3", "r3"), (" r3 ", "r3"), ("8", "r8"), (2.0, "r2"),
    (np.int64(1), "r1"), ("k=1", "r1"),
])
def test_valid_neighborhood_spellings_unchanged(value, same_as):
    np.testing.assert_array_equal(get_neighborhood_steps(value),
                                  get_neighborhood_steps(same_as))


def test_neighborhood_utility_names_value_and_forms():
    with pytest.raises(ValueError, match=r"'r999'.*too large"):
        get_neighborhood_steps("r999")
    with pytest.raises(ValueError, match=r"'x'.*'r2'"):
        get_neighborhood_steps("x")
    with pytest.raises(TypeError, match="NoneType"):
        get_neighborhood_steps(None)


# --- #7 raster dtype -------------------------------------------------------------

@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.uint8, np.int32])
def test_unsupported_raster_dtype_raises_listing_allowed(dtype):
    with pytest.raises(PyorpsError, match="'uint16'") as info:
        _finder(dataset_source=_raster(dtype))
    assert repr(np.dtype(dtype).name) in str(info.value)


def test_unsupported_dtype_on_cython_wrapper():
    with pytest.raises(PyorpsError, match="'float32'.*'uint16' only"):
        CythonAPI(_raster(np.float32), get_neighborhood_steps("r1"))


def test_float_raster_handler_without_outside_value_explains():
    from pyorps.io.geo_dataset import InMemoryRasterDataset
    dataset = InMemoryRasterDataset(_raster(np.float32), CRS,
                                    from_origin(ORIGIN_X, ORIGIN_Y, 1, 1))
    with pytest.raises(PyorpsError, match="'float32'.*outside_value"):
        RasterHandler(dataset, SRC, TGT, search_space_buffer_m=10)


# --- #14 graph_api name ------------------------------------------------------------

def test_graph_api_any_case_works_like_lower_case():
    upper = _finder(graph_api="Cython")
    assert upper.graph_api_name == "cython"
    reference = _finder().find_route()
    path = upper.find_route()
    assert path.total_cost == reference.total_cost
    assert list(path.path_indices) == list(reference.path_indices)


@pytest.mark.parametrize("value", ["nope", None, 3])
def test_unknown_graph_api_raises_at_construction_listing_names(value):
    with pytest.raises(PyorpsError, match="Unsupported graph API") as info:
        _finder(graph_api=value)
    for name in GRAPH_APIS:
        assert repr(name) in str(info.value)
    assert isinstance(info.value.__cause__, ValueError)


def test_get_graph_api_class_unknown_lists_names():
    """The module-level resolver keeps raising ValueError (callers catch it)."""
    with pytest.raises(ValueError, match="Valid graph_api names"):
        get_graph_api_class("nope")


def test_subclass_backend_names_are_accepted():
    """ConstrainedPathFinder routes 'cython_parallel' itself; PathFinder must
    not refuse a name its subclass declares in SUPPORTED_BACKENDS."""
    class Sub(PathFinder):
        SUPPORTED_BACKENDS = ("cython", "cython_parallel")

    finder = Sub(dataset_source=_raster(), source_coords=SRC, target_coords=TGT,
                 search_space_buffer_m=10, crs=CRS,
                 transform=from_origin(ORIGIN_X, ORIGIN_Y, 1, 1),
                 graph_api="Cython_Parallel")
    assert finder.graph_api_name == "cython_parallel"


# --- #24 algorithm name --------------------------------------------------------

_LIBRARY_BACKENDS = {
    "networkx": "networkx", "networkit": "networkit", "igraph": "igraph",
    "rustworkx": "rustworkx",
}


def _backend_params():
    params = [pytest.param("cython", id="cython")]
    for api, module in _LIBRARY_BACKENDS.items():
        missing = importlib.util.find_spec(module) is None
        params.append(pytest.param(
            api, id=api,
            marks=pytest.mark.skipif(missing, reason=f"{module} not installed")))
    return params


@pytest.mark.parametrize("graph_api", _backend_params())
def test_algorithm_name_any_case(graph_api):
    finder = _finder(graph_api=graph_api)
    lower = finder.find_route(algorithm="dijkstra")
    upper = finder.find_route(algorithm="DIJKSTRA")
    assert upper.total_cost == pytest.approx(lower.total_cost)


@pytest.mark.parametrize("graph_api", _backend_params())
def test_unknown_algorithm_lists_allowed_names(graph_api):
    finder = _finder(graph_api=graph_api)
    with pytest.raises(AlgorithmNotImplementedError, match="nope") as info:
        finder.find_route(algorithm="nope")
    assert "allowed: 'dijkstra'" in str(info.value)


# --- #16 optional backend not installed -------------------------------------------

def test_missing_graph_library_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "networkx", None)
    monkeypatch.delitem(sys.modules, "pyorps.graph.api.networkx_api",
                        raising=False)
    with pytest.raises(ImportError, match=r'pip install "pyorps\[graph\]"') as info:
        get_graph_api_class("networkx")
    assert isinstance(info.value.__cause__, ImportError)
    finder = _finder(graph_api="networkx")
    with pytest.raises(ImportError, match=r"pyorps\[graph\]"):
        finder.find_route()


def test_missing_gpu_backend_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyorps.graph.api.raster_gpu_api", None)
    with pytest.raises(ImportError, match=r'pip install "pyorps\[gpu\]"'):
        get_graph_api_class("raster_gpu")
