"""Result handling and option forwarding of PathFinder (audit T4 #8, #9, #10,
#11, #12, #15, #30).

Each test fails on the behaviour before the fix: a finder built without
coordinates could not route, unreachable pairs vanished from multi-route
results, save_paths wrote a stale frame, misspelt keywords were dropped,
backend settings could not be set, and fallbacks were silent.
"""
import inspect
import types
import warnings

import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import box

from pyorps import PathFinder
from pyorps.core.exceptions import NoPathFoundError, PyorpsError
from pyorps.graph import path_finder as path_finder_module
from pyorps.graph.api import graph_library_api
from pyorps.graph.api.cython_api import CythonAPI
from pyorps.graph.api.graph_library_api import GraphLibraryAPI, reject_unknown_options
from pyorps.graph.path_finder import FIND_ROUTE_OPTIONS, GRAPH_API_OPTIONS
from pyorps.io.geo_dataset import InMemoryRasterDataset
from pyorps.utils.neighborhood import get_neighborhood_steps

CRS = "EPSG:32632"
ORIGIN_X, ORIGIN_Y = 500000.0, 5600000.0
TRANSFORM = from_origin(ORIGIN_X, ORIGIN_Y, 1, 1)
N = 30


def cell(row, col):
    """Centre coordinate of a raster cell."""
    return (ORIGIN_X + col + 0.5, ORIGIN_Y - row - 0.5)


SRC = cell(2, 2)
TGT = cell(25, 25)
OTHER = cell(3, 20)
#: Cells inside the walled enclosure of _raster(walled=True).
WALLED = cell(15, 15)
WALLED_2 = cell(14, 14)


def _raster(walled=False):
    """Uniform 30 x 30 cost raster; ``walled`` adds a closed ring of 65535
    (rows/cols 12..18) around a 5 x 5 block of ordinary cells."""
    raster = np.full((N, N), 5, dtype=np.uint16)
    if walled:
        raster[12:19, 12:19] = 65535
        raster[13:18, 13:18] = 5
    return raster


def _finder(source=SRC, target=TGT, raster=None, **kwargs):
    args = dict(dataset_source=_raster() if raster is None else raster,
                source_coords=source, target_coords=target,
                search_space_buffer_m=40, crs=CRS, transform=TRANSFORM)
    args.update(kwargs)
    return PathFinder(**args)


def _messages(record, text):
    return [str(w.message) for w in record if text in str(w.message)]


# ---------------------------------------------------------------------------
# #8  a finder built without coordinates routes later
# ---------------------------------------------------------------------------

def test_late_coordinates_find_route_matches_constructor_coordinates():
    late = _finder(None, None)
    assert late.raster_handler is None

    path = late.find_route(SRC, TGT)

    reference = _finder().find_route()
    assert list(path.path_indices) == list(reference.path_indices)
    assert path.total_cost == pytest.approx(reference.total_cost)
    assert late.source_coords == SRC and late.target_coords == TGT
    assert late.raster_handler is not None


def test_late_coordinates_keep_the_constructor_raster_parameters(tmp_path):
    saved = tmp_path / "search_raster.tif"
    late = _finder(None, None, raster_save_path=str(saved))
    assert not saved.exists()

    late.find_route(SRC, TGT)

    assert saved.exists()


def test_late_coordinates_cost_field_and_cost_fields():
    single = _finder(None, None)
    with single.cost_field(SRC, algorithm="dijkstra") as field:
        assert field is not None
    assert single.source_coords == SRC and single.target_coords == SRC

    several = _finder(None, None)
    with several.cost_fields([SRC, TGT], algorithm="dijkstra") as fields:
        assert fields is not None
    assert several.source_coords == [SRC, TGT]


def test_late_coordinates_build_corridor_graph():
    late = _finder(None, None)
    graph = late.build_corridor_graph([SRC, TGT, OTHER])
    assert len(graph.segments) >= 2
    assert late.source_coords == [SRC, TGT, OTHER]


def test_missing_coordinates_are_named():
    with pytest.raises(PyorpsError, match="source_coords and target_coords are None"):
        _finder(None, None).search_session()
    with pytest.raises(PyorpsError, match="target_coords is None"):
        _finder(SRC, None).create_raster_handler()
    with pytest.raises(ValueError, match="must not be None! Missing: target"):
        _finder(SRC, None).find_route()
    with pytest.raises(PyorpsError, match="no raster to save"):
        _finder(None, None).save_raster("unused.tif")


# ---------------------------------------------------------------------------
# #9  unreachable pairs of a multi-route call are reported
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("graph_api", ["cython", "networkx"])
def test_unreachable_pair_among_reachable_ones_is_reported(graph_api):
    if graph_api != "cython":
        pytest.importorskip(graph_api)
    finder = _finder(SRC, [TGT, WALLED], raster=_raster(walled=True),
                     ignore_max_cost=True, graph_api=graph_api)

    with pytest.warns(UserWarning, match="unreachable") as record:
        result = finder.find_route()

    reported = _messages(record, "unreachable")
    assert len(reported) == 1
    assert "returns 1 route(s) for 2 (source, target) pair(s)" in reported[0]
    assert f"({SRC[0]!r}, {SRC[1]!r}) -> ({WALLED[0]!r}, {WALLED[1]!r})" in reported[0]
    assert len(result) == 1
    assert result.unreachable_pairs == [(SRC, WALLED)]
    assert finder.unreachable_pairs == [(SRC, WALLED)]


def test_unreachable_pair_in_a_pairwise_run_keeps_its_identity():
    finder = _finder([SRC, OTHER], [WALLED, TGT], raster=_raster(walled=True),
                     ignore_max_cost=True)

    with pytest.warns(UserWarning, match="unreachable"):
        result = finder.find_route(pairwise=True)

    assert len(result) == 1
    assert result.unreachable_pairs == [(SRC, WALLED)]


def test_all_pairs_unreachable_raises_no_path_found():
    finder = _finder(SRC, [WALLED, WALLED_2], raster=_raster(walled=True),
                     ignore_max_cost=True)
    with pytest.raises(NoPathFoundError, match="None of the 2"):
        finder.find_route()
    assert finder.unreachable_pairs == [(SRC, WALLED), (SRC, WALLED_2)]

    pairwise = _finder([SRC, OTHER], [WALLED, WALLED_2],
                       raster=_raster(walled=True), ignore_max_cost=True)
    with pytest.raises(NoPathFoundError, match="None of the 2"):
        pairwise.find_route(pairwise=True)


def test_long_list_of_unreachable_pairs_is_truncated_with_the_count():
    walled = [cell(row, col) for row in (13, 14, 15) for col in (13, 14, 15, 16)]
    finder = _finder(SRC, [TGT, *walled], raster=_raster(walled=True),
                     ignore_max_cost=True)

    with pytest.warns(UserWarning, match="unreachable") as record:
        result = finder.find_route()

    message = _messages(record, "unreachable")[0]
    assert "12 pair(s) are unreachable" in message
    assert "... and 2 more" in message
    assert len(result) == 1
    assert len(result.unreachable_pairs) == 12


def test_reachable_pairs_give_no_warning_and_an_empty_list():
    finder = _finder(SRC, [TGT, OTHER])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = finder.find_route()
    assert len(result) == 2
    assert result.unreachable_pairs == []
    assert finder.unreachable_pairs == []


def test_same_cell_pair_is_not_called_unreachable():
    finder = _finder(SRC, [TGT, SRC])
    with pytest.warns(UserWarning, match="same raster cell") as record:
        result = finder.find_route()
    assert not _messages(record, "unreachable")
    assert len(result) == 1
    assert result.unreachable_pairs == []


# ---------------------------------------------------------------------------
# #15  save_paths: nothing to save, and no stale frame
# ---------------------------------------------------------------------------

def test_save_paths_without_routes_raises(tmp_path):
    target_file = tmp_path / "routes.gpkg"
    finder = _finder()
    with pytest.raises(PyorpsError, match="no routes to save"):
        finder.save_paths(str(target_file))
    assert not target_file.exists()
    finder.save_paths(None)       # documented: no file is written
    finder.save_paths("")


def test_save_paths_after_a_second_route_writes_the_new_routes(tmp_path):
    finder = _finder()
    finder.find_route()
    first_file = tmp_path / "first.gpkg"
    finder.save_paths(str(first_file))
    assert len(gpd.read_file(first_file)) == 1

    finder.find_route(SRC, OTHER)
    second_file = tmp_path / "second.gpkg"
    finder.save_paths(str(second_file))

    written = gpd.read_file(second_file)
    assert len(written) == 2
    assert len(finder.path_gdf) == 2
    ends = sorted(geometry.coords[-1] for geometry in written.geometry)
    assert ends == sorted([OTHER, TGT])


def test_cached_frame_is_dropped_when_a_route_is_added():
    finder = _finder()
    finder.find_route()
    assert len(finder.create_path_geodataframe()) == 1
    finder.find_route(SRC, OTHER)
    assert finder.path_gdf is None
    assert len(finder.create_path_geodataframe()) == 2


# ---------------------------------------------------------------------------
# #11  misspelt keywords raise, with a suggestion
# ---------------------------------------------------------------------------

def test_constructor_typo_raises_with_suggestion():
    with pytest.raises(TypeError, match=r"'neighbourhood_str' \(did you mean 'neighborhood_str'\?\)"):
        _finder(neighbourhood_str="r1")
    # ... also when the raster is not loaded yet (no coordinates)
    with pytest.raises(TypeError, match="'neighbourhood_str'"):
        _finder(None, None, neighbourhood_str="r1")


def test_vector_source_typo_suggests_the_rasterize_parameter():
    frame = gpd.GeoDataFrame({"cost": [5]}, geometry=[box(ORIGIN_X, ORIGIN_Y - N, ORIGIN_X + N, ORIGIN_Y)],
                             crs=CRS)
    with pytest.raises(TypeError, match=r"'resolution_in_meter' \(did you mean 'resolution_in_m'\?\)"):
        PathFinder(frame, None, None, crs=CRS, resolution_in_meter=2)
    # the real parameter is accepted
    PathFinder(frame, None, None, crs=CRS, resolution_in_m=2)


def test_find_route_typo_raises_with_suggestion():
    finder = _finder()
    with pytest.raises(TypeError, match=r"'detla' \(did you mean 'delta'\?\)"):
        finder.find_route(algorithm="delta-stepping", detla=3)
    with pytest.raises(TypeError, match=r"'pairwize' \(did you mean 'pairwise'\?\)"):
        finder.find_route(pairwize=True)
    # the settings the cython backend reads still pass
    path = finder.find_route(algorithm="delta-stepping", delta=50, num_threads=1,
                             margin=1.00001)
    assert path.total_cost > 0


def test_find_route_typo_on_a_graph_library_backend():
    pytest.importorskip("networkx")
    finder = _finder(graph_api="networkx")
    with pytest.raises(TypeError, match=r"'heu_weigth' \(did you mean 'heu_weight'\?\)"):
        finder.find_route(algorithm="astar", heu_weigth=2.0)


def test_backend_shortest_path_rejects_unknown_keywords():
    api = CythonAPI(_raster(), get_neighborhood_steps("r2", directed=True))
    with pytest.raises(TypeError, match=r"'detla' \(did you mean 'delta'\?\)"):
        api.shortest_path(0, N * N - 1, algorithm="delta-stepping", detla=3)
    assert len(api.shortest_path(0, N * N - 1, algorithm="delta-stepping", delta=3,
                                 num_threads=1, pairwise=False)) > 1


def test_session_and_field_typos_raise_and_the_allow_list_is_accepted():
    finder = _finder()
    with pytest.raises(TypeError, match=r"'num_thread' \(did you mean 'num_threads'\?\)"):
        finder.search_session(num_thread=2)
    with pytest.raises(TypeError, match=r"'dleta' \(did you mean 'delta'\?\)"):
        finder.cost_field(SRC, algorithm="dijkstra", dleta=1)
    with pytest.raises(TypeError, match=r"'label' \(did you mean 'labels'\?\)"):
        finder.cost_fields([SRC, TGT], label=["a", "b"])
    # Keys the GUI passes although nothing reads them stay accepted.
    session = finder.search_session(algorithm="dijkstra", delta=100, num_threads=1,
                                    use_astar=False, margin=1.1, max_trees=2)
    session.close()


def test_reject_unknown_options_names_key_suggestion_and_accepted_names():
    with pytest.raises(TypeError) as info:
        reject_unknown_options({"detla": 1, "zzz": 2}, ("delta", "margin"), "here()",
                               ignored=("legacy",))
    message = str(info.value)
    assert "here() got unexpected keyword argument(s)" in message
    assert "'detla' (did you mean 'delta'?)" in message and "'zzz'" in message
    assert "Accepted here: 'delta', 'margin'." in message
    reject_unknown_options({"legacy": 1, "delta": 2}, ("delta",), "here()", ignored=("legacy",))


def test_route_option_tables_match_the_backends():
    assert set(FIND_ROUTE_OPTIONS["cython"]) | {"pairwise"} == set(CythonAPI.SHORTEST_PATH_OPTIONS)
    for name in ("networkit", "igraph", "rustworkx", "networkx"):
        assert set(FIND_ROUTE_OPTIONS[name]) | {"pairwise"} == set(GraphLibraryAPI.SHORTEST_PATH_OPTIONS)


# ---------------------------------------------------------------------------
# #12  backend settings through graph_api_options
# ---------------------------------------------------------------------------

def test_graph_api_options_reach_the_backend_constructor(monkeypatch):
    received = {}

    class RecordingAPI:
        def __init__(self, raster_data, steps, **kwargs):
            received.update(kwargs)

    monkeypatch.setattr(path_finder_module, "get_graph_api_class", lambda name: RecordingAPI)
    finder = _finder(graph_api="raster_gpu",
                     graph_api_options={"delta": 50.0, "arena_factor": 2.0})

    finder.create_graph()

    assert received["delta"] == 50.0 and received["arena_factor"] == 2.0
    assert received["ignore_max"] is True          # PathFinder's own arguments stay
    assert finder.graph_api_options == {"delta": 50.0, "arena_factor": 2.0}


def test_allow_large_raster_is_settable_through_pathfinder(monkeypatch):
    pytest.importorskip("networkx")
    monkeypatch.setattr(graph_library_api, "MAX_LIBRARY_BACKEND_CELLS", 100)

    with pytest.raises(ValueError, match="graph_api_options="):
        _finder(graph_api="networkx").find_route()

    path = _finder(graph_api="networkx",
                   graph_api_options={"allow_large_raster": True}).find_route()
    assert path.total_cost > 0


def test_unknown_graph_api_option_raises_with_suggestion():
    with pytest.raises(TypeError, match=r"'allow_large_rasters' \(did you mean 'allow_large_raster'\?\)"):
        _finder(graph_api="networkx", graph_api_options={"allow_large_rasters": True})
    with pytest.raises(TypeError, match="graph_api='cython'.*Accepted here: none"):
        _finder(graph_api_options={"delta": 5})
    with pytest.raises(TypeError, match="must be a dict"):
        _finder(graph_api_options=[("delta", 5)])
    assert _finder().graph_api_options == {}


@pytest.mark.parametrize("name, module, class_name", [
    ("raster_gpu", "pyorps.graph.api.raster_gpu_api", "RasterGPUAPI"),
    ("raster_fim", "pyorps.graph.api.raster_fim_api", "RasterFIMAPI"),
    ("networkx", "pyorps.graph.api.networkx_api", "NetworkxAPI"),
])
def test_graph_api_option_table_names_real_constructor_parameters(name, module, class_name):
    backend = getattr(pytest.importorskip(module), class_name)
    parameters = set(inspect.signature(backend.__init__).parameters)
    create_graph_keys = {"n", "remove_isolated_nodes"}     # read from **kwargs in create_graph
    assert set(GRAPH_API_OPTIONS[name]) <= parameters | create_graph_keys


# ---------------------------------------------------------------------------
# #10  bbox/mask are not handed to a raster DEM
# ---------------------------------------------------------------------------

def test_vector_source_with_raster_dem_gives_no_ignored_bbox_warning():
    extent = (ORIGIN_X, ORIGIN_Y - N, ORIGIN_X + N, ORIGIN_Y)
    frame = gpd.GeoDataFrame({"cost": [5]}, geometry=[box(*extent)], crs=CRS)
    dem = np.zeros((N, N), dtype=np.float32)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        finder = PathFinder(frame, None, None, crs=CRS, bbox=extent, dem=dem,
                            transform=TRANSFORM)

    assert not _messages(record, "ignored for a raster")
    assert isinstance(finder.dem_dataset, InMemoryRasterDataset)
    assert finder.dataset.bbox == extent           # still applied to the vector source


def test_raster_source_with_raster_dem_warns_once_about_bbox():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _finder(dem=np.zeros((N, N), dtype=np.float32),
                bbox=(ORIGIN_X, ORIGIN_Y - N, ORIGIN_X + N, ORIGIN_Y))
    assert len(_messages(record, "ignored for a raster")) == 1


# ---------------------------------------------------------------------------
# #30  fallbacks are visible, messages state the real cause
# ---------------------------------------------------------------------------

def test_negative_buffer_message_names_the_sign():
    with pytest.raises(ValueError, match="a negative buffer shrinks"):
        _finder(search_space_buffer_m=-5)
    with pytest.raises(ValueError, match="buffering it by zero removes it"):
        _finder(search_space_buffer_m=0)


def test_estimated_buffer_on_a_large_window_warns_with_the_real_numbers():
    size = 1100
    raster = np.full((size, size), 5, dtype=np.uint16)
    with pytest.warns(UserWarning, match="No search_space_buffer_m set") as record:
        finder = _finder(cell(50, 50), cell(1050, 1050), raster=raster,
                         search_space_buffer_m=None)

    message = _messages(record, "No search_space_buffer_m set")[0]
    rows, cols = finder.raster_handler.data.shape[-2:]
    assert rows * cols > 1_000_000
    assert f"the buffer estimator chose {finder.search_space_buffer_m} m" in message
    assert f"{rows}x{cols} = {rows * cols:,} cells" in message
    assert "full raster" not in message


def test_corridor_burn_fallback_warns():
    finder = _finder()
    frame = gpd.GeoDataFrame({"cost": [5]}, geometry=[box(ORIGIN_X, ORIGIN_Y - N, ORIGIN_X + N, ORIGIN_Y)],
                             crs=CRS)
    # A rasterizer without the grid helpers the corridor certificate needs.
    finder.geo_rasterizer = types.SimpleNamespace(base_data=frame)

    with pytest.warns(UserWarning, match="Rasterizing the full data extent instead") as record:
        assert finder._certified_corridor_bounding_box() is None

    message = _messages(record, "Rasterizing the full data extent")[0]
    assert "corridor_first=True" in message and "search_space_buffer_m=40" in message
    assert "AttributeError" in message
