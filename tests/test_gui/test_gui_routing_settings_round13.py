"""Round 13: the routing settings of the GUI reach the library, and no control lies.

Headless callback and service tests (the harness of ``conftest.py`` drives the callbacks as Dash would).
Each new setting has a test that the value set in the page arrives at the library boundary (a spy or a
monkeypatch at the library call), one for its default and one for an invalid value (a field notice, no
exception). The dead controls are gone, and routes that carry the old ``use_astar`` key still load and edit.
"""
from __future__ import annotations

import importlib.util
import json

import pytest
from conftest import SOURCE, TARGET, TEST_CRS, invoke
from shapely.geometry import LineString

from pyorps import PathFinder
from pyorps.graph.api import cython_api
from pyorps.gui import ids
from pyorps.gui.callbacks import edit as edit_cb
from pyorps.gui.callbacks.routes import add_route_layer, number_or_default
from pyorps.gui.layout import build_layout
from pyorps.gui.services import constrained, project_io, routing
from pyorps.gui.services.cost import RouteCost

RUN_WIRING = (f"{ids.ROUTING_STATUS}.children", ids.RUN_ROUTING_BTN)
OHL_WIRING = (f"{ids.OHL_STATUS}.children", ids.RUN_ROUTING_BTN)


# ------------------------------------------------------------------ helpers
def _installed(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _draft(source=SOURCE, target=TARGET):
    return {"sources": [{"lat": 0, "lng": 0, "x": source[0], "y": source[1]}],
            "targets": [{"lat": 0, "lng": 0, "x": target[0], "y": target[1]}],
            "waypoints": []}


def _run_args(raster_id, *, algorithm="dijkstra", graph_api="cython", k=1, buffer=80, full_window=False,
              delta=100, threads=0, margin=None, simplify=False, simplify_tol=1.0, draft=None):
    return (1, False, draft or _draft(), raster_id, algorithm, graph_api, k, buffer, full_window, True, False,
            simplify, simplify_tol, delta, threads, margin, [], [])


def _run(app, raster_id, **kwargs):
    return invoke(app, RUN_WIRING, *_run_args(raster_id, **kwargs), triggered=[f"{ids.RUN_ROUTING_BTN}.n_clicks"])


def _notices(resp) -> list[dict]:
    return resp[ids.NOTICES]["data"]


@pytest.fixture()
def raster_layer(app, state, raster_path):
    invoke(app, ("layers-view.data", ids.RASTER_LOAD_BTN), 1, raster_path, "viridis", [],
           triggered=[f"{ids.RASTER_LOAD_BTN}.n_clicks"])
    return state.layers_of_kind("raster")[0]


@pytest.fixture()
def run_calls(monkeypatch):
    """Replace the service call: the keywords the callback hands to ``run_routing``."""
    calls: list[dict] = []

    def fake_run(_path, **kwargs):
        calls.append(kwargs)
        return None, [], []

    monkeypatch.setattr(routing, "run_routing", fake_run)
    return calls


def _spy(monkeypatch, owner, name: str) -> list[tuple]:
    """Record every call of ``owner.name`` (class or module attribute) and run it as before."""
    calls: list[tuple] = []
    original = getattr(owner, name)

    if isinstance(owner, type):
        def wrapper(self, *args, **kwargs):
            calls.append((args, kwargs))
            return original(self, *args, **kwargs)
    else:
        def wrapper(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, wrapper)
    return calls


def _library_classes():
    """``{graph_api: API class}`` of the library backends that are installed here."""
    from pyorps.graph.path_finder import get_graph_api_class

    found = {}
    for name in ("networkx", "networkit", "igraph", "rustworkx"):
        if _installed(name):
            found[name] = get_graph_api_class(name)
    return found


# ======================================== A: the backend really runs (4b.1)
def test_finding_4b1_a_session_on_a_library_backend_runs_the_cython_dijkstra(raster_path, monkeypatch):
    """The reason the GUI routes library backends directly: a SearchSession picks its kernel from the
    backend KIND, so on networkx it solves with the Cython Dijkstra and never calls the library."""
    pytest.importorskip("networkx")
    from pyorps.graph.api.networkx_api import NetworkxAPI
    from pyorps.utils import _dijkstra

    library = _spy(monkeypatch, NetworkxAPI, "shortest_path")
    cython = _spy(monkeypatch, _dijkstra, "make_dijkstra_solver")
    finder = PathFinder(raster_path, source_coords=SOURCE, target_coords=TARGET, graph_api="networkx",
                        neighborhood_str="r1", search_space_buffer_m=80)
    session = finder.search_session(algorithm="bidirectional_dijkstra")
    routing.route_through_points(finder, [SOURCE, TARGET], algorithm="bidirectional_dijkstra", session=session)
    session.close()
    assert library == []
    assert len(cython) == 1


@pytest.mark.parametrize("graph_api,algorithm", [
    ("networkx", "bidirectional_dijkstra"), ("networkx", "astar"), ("networkx", "dijkstra"),
    ("networkit", "bidirectional_dijkstra"), ("igraph", "bellman_ford"), ("rustworkx", "astar")])
def test_library_backend_runs_find_route_with_its_own_algorithm(raster_path, monkeypatch, graph_api, algorithm):
    api_class = _library_classes().get(graph_api)
    if api_class is None:
        pytest.skip(f"{graph_api} is not installed")
    from pyorps.utils import _dijkstra

    library = _spy(monkeypatch, api_class, "shortest_path")
    cython = _spy(monkeypatch, _dijkstra, "make_dijkstra_solver")
    _finder, results, failed = routing.run_routing(
        raster_path, sources=[SOURCE], targets=[TARGET], algorithm=algorithm, graph_api=graph_api,
        neighborhood="r1", search_buffer_m=80)
    assert failed == [] and len(results) == 1
    assert [call[1].get("algorithm") for call in library] == [algorithm]      # the library's own path method ran
    assert cython == []                                                       # and not the Cython Dijkstra
    assert results[0].session is None                                         # no retained session
    assert results[0].params["graph_api"] == graph_api and results[0].params["algorithm"] == algorithm
    assert results[0].cost.total_cost > 0


def test_cython_stays_on_a_search_session(raster_path):
    for algorithm in ("dijkstra", "delta-stepping"):
        _finder, results, failed = routing.run_routing(
            raster_path, sources=[SOURCE], targets=[TARGET], algorithm=algorithm, graph_api="cython",
            neighborhood="r1", search_buffer_m=80)
        assert failed == []
        assert results[0].session is not None and results[0].session.has_route
        results[0].session.close()


@pytest.mark.parametrize("graph_api,algorithm", [("raster_gpu", "delta-stepping"), ("raster_gpu", "dijkstra"),
                                                 ("raster_fim", "fim")])
def test_gpu_and_fim_stay_on_a_search_session(raster_path, monkeypatch, graph_api, algorithm):
    """No GPU is needed: the session is opened (and recorded) but the run stops there."""
    seen: list[tuple] = []

    def stop(self, algorithm="dijkstra", **kwargs):
        seen.append((self.graph_api_name, algorithm))
        raise RuntimeError("stop before any kernel runs")

    monkeypatch.setattr(PathFinder, "search_session", stop)
    with pytest.raises(RuntimeError, match="stop before"):
        routing.run_routing(raster_path, sources=[SOURCE], targets=[TARGET], algorithm=algorithm,
                            graph_api=graph_api, neighborhood="r1", search_buffer_m=80)
    assert seen == [(graph_api, algorithm)]


def test_run_callback_routes_a_networkx_run_and_stores_backend_and_algorithm(app, state, raster_layer,
                                                                             monkeypatch):
    pytest.importorskip("networkx")
    from pyorps.graph.api.networkx_api import NetworkxAPI

    library = _spy(monkeypatch, NetworkxAPI, "shortest_path")
    resp = _run(app, raster_layer.id, algorithm="bidirectional_dijkstra", graph_api="networkx")
    assert "1 route(s) built" in resp[ids.ROUTING_STATUS]["children"]
    assert len(library) == 1
    route = state.layers_of_kind("route")[0]
    assert route.meta["params"]["graph_api"] == "networkx"
    assert route.meta["params"]["algorithm"] == "bidirectional_dijkstra"
    assert route.meta["params"]["neighborhood"] == "r1"
    assert route.id not in state.search_sessions


def test_an_edit_of_a_library_route_recomputes_on_the_same_backend(app, state, raster_layer, monkeypatch):
    pytest.importorskip("networkx")
    from pyorps.graph.api.networkx_api import NetworkxAPI

    _run(app, raster_layer.id, algorithm="bidirectional_dijkstra", graph_api="networkx")
    route = state.layers_of_kind("route")[0]
    library = _spy(monkeypatch, NetworkxAPI, "shortest_path")
    moved = [list(SOURCE), [TARGET[0] - 10.0, TARGET[1] + 10.0]]
    new_layer, notices = edit_cb.recompute_variant(state, route, moved, "moved target", [])
    assert new_layer is not None, notices
    assert len(library) == 1                                  # the library path method, again
    assert new_layer.meta["params"]["graph_api"] == "networkx"
    assert new_layer.id not in state.search_sessions


def test_an_edit_ignores_a_cached_finder_of_another_backend(app, state, raster_layer, raster_path, monkeypatch):
    """The finder cache holds one finder per raster; a later cython run must not serve a networkx route."""
    pytest.importorskip("networkx")
    from pyorps.graph.api.networkx_api import NetworkxAPI

    _run(app, raster_layer.id, algorithm="dijkstra", graph_api="networkx")
    route = state.layers_of_kind("route")[0]
    state.finders[raster_layer.id] = PathFinder(
        raster_path, source_coords=SOURCE, target_coords=TARGET, graph_api="cython", neighborhood_str="r1",
        search_space_buffer_m=80)
    library = _spy(monkeypatch, NetworkxAPI, "shortest_path")
    new_layer, notices = edit_cb.recompute_variant(
        state, route, [list(SOURCE), [TARGET[0] - 5.0, TARGET[1]]], "moved target", [])
    assert new_layer is not None, notices
    assert len(library) == 1
    assert state.finders[raster_layer.id].graph_api_name == "networkx"


# ================================ A: the Backend dropdown filters the algorithms
def test_backend_table_follows_the_library(raster_path):
    from pyorps.graph.api import raster_fim_api
    from pyorps.graph.path_finder import GRAPH_APIS, get_graph_api_class

    assert set(routing.BACKENDS) <= set(GRAPH_APIS)
    assert "cugraph" not in routing.BACKENDS                     # raises NotImplementedError
    assert routing.DEFAULT_BACKEND == "cython"
    assert list(routing.BACKENDS) == ["cython", "raster_gpu", "raster_fim", "networkx", "networkit", "igraph",
                                      "rustworkx"]
    assert routing.BACKENDS["cython"].algorithms == tuple(get_graph_api_class("cython").ALGORITHMS)
    for name, api_class in _library_classes().items():
        assert routing.BACKENDS[name].algorithms == tuple(api_class.ALGORITHMS), name
    assert routing.BACKENDS["raster_fim"].algorithms == tuple(raster_fim_api._ACCEPTED_ALGORITHMS)
    # raster_gpu.shortest_path only accepts these two: delta-stepping-circular would be rejected
    assert routing.BACKENDS["raster_gpu"].algorithms == ("delta-stepping", "dijkstra")
    for backend in routing.BACKENDS.values():
        assert backend.default_algorithm in backend.algorithms


def test_unavailable_backends_are_disabled_with_the_install_hint(monkeypatch):
    hidden = {"networkx", "cupy"}
    real = importlib.util.find_spec

    def find_spec(name, package=None):
        return None if name.split(".")[0] in hidden else real(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    options = {o["value"]: o for o in routing.backend_options()}
    assert [o["value"] for o in routing.backend_options()] == list(routing.BACKENDS)
    assert options["cython"].get("disabled") is not True
    for name in ("networkx", "raster_gpu", "raster_fim"):
        assert options[name]["disabled"] is True
        assert routing.BACKENDS[name].hint in options[name]["label"]
    assert 'pip install "pyorps[graph]"' in options["networkx"]["label"]
    assert 'pip install "pyorps[gpu]"' in options["raster_gpu"]["label"]
    assert routing.backend_available("networkx") is False and routing.backend_available("cython") is True


def test_the_install_hint_is_the_one_the_library_prints(monkeypatch):
    from pyorps.core.exceptions import MissingDependencyError
    from pyorps.graph import path_finder

    def missing():
        raise ImportError("no module named networkx")

    monkeypatch.setattr(path_finder, "_networkx_class", missing)
    with pytest.raises(MissingDependencyError) as err:
        path_finder.get_graph_api_class("networkx")
    assert routing.BACKENDS["networkx"].hint in str(err.value)
    assert routing.BACKENDS["raster_gpu"].hint.endswith(f'[{path_finder._BACKEND_EXTRAS["raster_gpu"][1]}]"')


def test_backend_dropdown_is_filled_from_the_service_and_defaults_to_cython(state):
    dropdown = next(c for c in build_layout(state)._traverse() if getattr(c, "id", None) == ids.GRAPH_API)  # noqa: SLF001
    assert [o["value"] for o in dropdown.options] == list(routing.BACKENDS)
    assert dropdown.value == "cython"


@pytest.mark.parametrize("graph_api", list(routing.BACKENDS))
def test_changing_the_backend_resets_the_algorithm_to_its_default(app, graph_api):
    resp = invoke(app, (f"{ids.ALGORITHM}.options", ids.GRAPH_API), graph_api, "astar",
                  triggered=[f"{ids.GRAPH_API}.value"])
    values = [o["value"] for o in resp[ids.ALGORITHM]["options"]]
    assert values == routing.backend_algorithms(graph_api)
    assert resp[ids.ALGORITHM]["value"] == routing.BACKENDS[graph_api].default_algorithm
    if graph_api == "raster_gpu":
        assert "delta-stepping-circular" not in values


def test_the_first_paint_keeps_a_restored_algorithm_that_the_backend_accepts(app):
    keep = invoke(app, (f"{ids.ALGORITHM}.options", ids.GRAPH_API), "networkx", "astar")
    assert keep[ids.ALGORITHM]["value"] == "astar"
    reset = invoke(app, (f"{ids.ALGORITHM}.options", ids.GRAPH_API), "networkx", "delta-stepping")
    assert reset[ids.ALGORITHM]["value"] == "dijkstra"


def test_a_backend_that_is_not_installed_gives_a_notice_and_runs_nothing(app, raster_layer, run_calls, monkeypatch):
    monkeypatch.setattr(routing, "backend_available", lambda name: name != "networkx")
    resp = _run(app, raster_layer.id, algorithm="dijkstra", graph_api="networkx")
    assert run_calls == []
    notice = _notices(resp)[-1]
    assert notice["title"] == "This backend is not installed"
    assert 'pip install "pyorps[graph]"' in notice["fix"]
    assert notice["focus_control"] == ids.GRAPH_API


def test_an_algorithm_the_backend_does_not_offer_gives_a_notice_and_runs_nothing(app, raster_layer, run_calls,
                                                                                   monkeypatch):
    # Without CuPy (every CI runner) raster_gpu counts as "not installed", and that notice would come first.
    monkeypatch.setattr(routing, "backend_available", lambda name: True)
    resp = _run(app, raster_layer.id, algorithm="delta-stepping-circular", graph_api="raster_gpu")
    assert run_calls == []
    notice = _notices(resp)[-1]
    assert notice["title"] == "Algorithm not offered by this backend"
    assert notice["focus_control"] == ids.ALGORITHM


def test_run_routing_refuses_an_algorithm_the_backend_does_not_run(raster_path):
    with pytest.raises(ValueError, match="does not offer"):
        routing.run_routing(raster_path, sources=[SOURCE], targets=[TARGET], algorithm="bellman_ford",
                            graph_api="cython")
    with pytest.raises(ValueError, match="Unknown backend"):
        routing.run_routing(raster_path, sources=[SOURCE], targets=[TARGET], algorithm="dijkstra",
                            graph_api="cugraph")


def test_the_chosen_backend_reaches_the_service(app, raster_layer, run_calls):
    _run(app, raster_layer.id, algorithm="astar", graph_api="rustworkx")
    assert run_calls[0]["graph_api"] == "rustworkx" and run_calls[0]["algorithm"] == "astar"
    _run(app, raster_layer.id, algorithm=None, graph_api=None)                    # defaults
    assert run_calls[1]["graph_api"] == "cython" and run_calls[1]["algorithm"] == "delta-stepping"


# ======================================================== B: delta-stepping margin
def test_margin_reaches_the_cython_kernel(raster_path, monkeypatch):
    kernel = _spy(monkeypatch, cython_api, "delta_stepping_2d_persistent")
    _finder, results, failed = routing.run_routing(
        raster_path, sources=[SOURCE], targets=[TARGET], algorithm="delta-stepping", graph_api="cython",
        neighborhood="r1", search_buffer_m=80, margin=2.5)
    assert failed == []
    assert [call[1]["margin"] for call in kernel] == [2.5]          # the keyword of the kernel call itself
    assert results[0].session is None                              # a session would have ignored it
    assert results[0].params["margin"] == 2.5


def test_without_a_margin_the_session_runs_and_the_kernel_default_applies(raster_path, monkeypatch):
    kernel = _spy(monkeypatch, cython_api, "delta_stepping_2d_persistent")
    _finder, results, _failed = routing.run_routing(
        raster_path, sources=[SOURCE], targets=[TARGET], algorithm="delta-stepping", graph_api="cython",
        neighborhood="r1", search_buffer_m=80)
    assert kernel == []                                            # the session's own kernel call ran
    assert results[0].session is not None and results[0].params["margin"] is None
    results[0].session.close()
    # one-shot, margin left empty: the library's exact termination margin
    finder = PathFinder(raster_path, source_coords=SOURCE, target_coords=TARGET, neighborhood_str="r1",
                        search_space_buffer_m=80)
    finder.find_route(algorithm="delta-stepping")
    assert kernel[-1][1]["margin"] == cython_api.EXACT_TERMINATION_MARGIN


def test_a_margin_on_the_gpu_goes_to_graph_api_options(raster_path, monkeypatch):
    """The GPU margin is a backend constructor setting; no GPU is needed to see it passed."""
    import pyorps

    seen: list[dict] = []

    def stop(*_args, **kwargs):
        seen.append(kwargs)
        raise RuntimeError("stop")

    monkeypatch.setattr(pyorps, "PathFinder", stop)
    with pytest.raises(RuntimeError):
        routing.run_routing(raster_path, sources=[SOURCE], targets=[TARGET], algorithm="delta-stepping",
                            graph_api="raster_gpu", neighborhood="r1", search_buffer_m=80, margin=1.7, delta=50)
    assert seen[0]["graph_api_options"] == {"delta": 50, "margin": 1.7}
    seen.clear()
    with pytest.raises(RuntimeError):
        routing.run_routing(raster_path, sources=[SOURCE], targets=[TARGET], algorithm="delta-stepping",
                            graph_api="raster_gpu", neighborhood="r1", search_buffer_m=80)
    assert "graph_api_options" not in seen[0]


def test_margin_from_the_field_reaches_the_service_and_blank_means_default(app, raster_layer, run_calls):
    _run(app, raster_layer.id, algorithm="delta-stepping", margin=1.5)
    assert run_calls[0]["margin"] == 1.5
    _run(app, raster_layer.id, algorithm="delta-stepping", margin=None)
    assert run_calls[1]["margin"] is None
    _run(app, raster_layer.id, algorithm="delta-stepping", margin="")
    assert run_calls[2]["margin"] is None


@pytest.mark.parametrize("margin", [0, -1, -0.5, "abc"])
def test_a_margin_of_zero_or_less_is_a_field_error(app, raster_layer, run_calls, margin):
    resp = _run(app, raster_layer.id, algorithm="delta-stepping", margin=margin)
    assert run_calls == []
    notice = _notices(resp)[-1]
    assert "margin" in notice["title"] and notice["focus_control"] == ids.MARGIN
    assert notice["focus_id"] == ids.TAB_ROUTES


def test_a_margin_is_not_checked_where_it_is_disabled(app, raster_layer, run_calls):
    _run(app, raster_layer.id, algorithm="dijkstra", margin=-3)          # cython dijkstra ignores it
    assert len(run_calls) == 1


# ====================================================== C: neighbourhood k
def test_k_reaches_pathfinder_as_rk_and_sets_the_steps(raster_path):
    finder, results, _failed = routing.run_routing(
        raster_path, sources=[SOURCE], targets=[TARGET], algorithm="dijkstra", graph_api="cython",
        neighborhood="r3", search_buffer_m=80)
    assert finder.neighborhood_str == "r3" and len(finder.steps) == 32
    assert results[0].params["neighborhood"] == "r3"
    results[0].session.close()


def test_k_from_the_field_reaches_the_service(app, state, raster_layer, run_calls):
    _run(app, raster_layer.id, k=5)
    assert run_calls[0]["neighborhood"] == "r5"
    _run(app, raster_layer.id, k=0)
    assert run_calls[1]["neighborhood"] == "r0"                       # zero is a value, not blank
    # the largest k is an enormous run: it waits for the user's go-ahead, with k in the pending kwargs
    resp = _run(app, raster_layer.id, k=127)
    assert resp[ids.ROUTING_CONFIRM_MODAL]["is_open"] is True
    assert state.pending_routing["kwargs"]["neighborhood"] == "r127"
    assert len(run_calls) == 2


def test_a_blank_k_is_the_default_r2(app, raster_layer, run_calls):
    _run(app, raster_layer.id, k=None)
    _run(app, raster_layer.id, k="")
    assert [call["neighborhood"] for call in run_calls] == ["r2", "r2"]


@pytest.mark.parametrize("k", [-1, 128, 2.5, "abc"])
def test_an_invalid_k_is_a_field_error(app, raster_layer, run_calls, k):
    resp = _run(app, raster_layer.id, k=k)
    assert run_calls == []
    notice = _notices(resp)[-1]
    assert "k" in notice["title"] and notice["focus_control"] == ids.NEIGHBORHOOD


def test_the_estimate_counts_the_edges_of_any_k(raster_path):
    edges = {k: routing.edges_per_cell(f"r{k}") for k in (0, 1, 2, 3, 4)}
    assert edges == {0: 4, 1: 8, 2: 16, 3: 32, 4: 48}
    est = routing.estimate_routing(raster_path, sources=[SOURCE], targets=[TARGET], neighborhood="r4",
                                   search_buffer_m=80)
    assert est["neighborhood"] == "r4"
    assert routing.neighborhood_string(None) == "r2" and routing.neighborhood_string(0) == "r0"
    assert routing.neighborhood_string("r1") == "r1"                  # a stored route's own spelling


# ================================================= D: whole raster as search window
def test_full_window_passes_the_librarys_whole_raster_buffer(raster_path, monkeypatch):
    import pyorps
    from pyorps.graph.search_session import full_window_buffer_m

    seen: list[dict] = []

    def stop(*_args, **kwargs):
        seen.append(kwargs)
        raise RuntimeError("stop")

    monkeypatch.setattr(pyorps, "PathFinder", stop)
    with pytest.raises(RuntimeError):
        routing.run_routing(raster_path, sources=[SOURCE], targets=[TARGET], algorithm="dijkstra",
                            graph_api="cython", neighborhood="r1", search_buffer_m=80, full_window=True)
    assert seen[0]["search_space_buffer_m"] == full_window_buffer_m(raster_path)
    assert seen[0]["search_space_buffer_m"] > 80


def test_full_window_switch_reaches_the_service_and_disables_the_buffer(app, raster_layer, raster_path, run_calls):
    from pyorps.graph.search_session import full_window_buffer_m

    _run(app, raster_layer.id, buffer=80, full_window=True)
    assert run_calls[0]["full_window"] is True
    assert run_calls[0]["search_buffer_m"] == full_window_buffer_m(raster_path)
    _run(app, raster_layer.id, buffer=80, full_window=False)                # default: the typed buffer
    assert run_calls[1]["full_window"] is False and run_calls[1]["search_buffer_m"] == 80.0
    resp = invoke(app, (f"{ids.SEARCH_BUFFER}.disabled",), True)
    assert resp[ids.SEARCH_BUFFER]["disabled"] is True
    resp = invoke(app, (f"{ids.SEARCH_BUFFER}.disabled",), False)
    assert resp[ids.SEARCH_BUFFER]["disabled"] is False


def test_an_invalid_buffer_is_ignored_while_the_whole_raster_is_the_window(app, raster_layer, run_calls):
    resp = _run(app, raster_layer.id, buffer=-5, full_window=False)
    assert run_calls == [] and "Search buffer" in _notices(resp)[-1]["title"]
    _run(app, raster_layer.id, buffer=-5, full_window=True)
    assert len(run_calls) == 1


def test_the_buffer_readout_names_the_whole_raster_window(app):
    resp = invoke(app, (f"{ids.SEARCH_BUFFER_INFO}.children",), None, _draft(), True)
    assert "whole raster" in resp[ids.SEARCH_BUFFER_INFO]["children"]


def test_a_full_window_route_keeps_its_window_when_it_is_edited(app, state, raster_layer, raster_path):
    from pyorps.graph.search_session import full_window_buffer_m

    _run(app, raster_layer.id, full_window=True)
    route = state.layers_of_kind("route")[0]
    assert route.meta["params"]["full_window"] is True
    assert route.meta["params"]["search_space_buffer_m"] == full_window_buffer_m(raster_path)


# ============================================= F: what was dead is gone, old data still loads
def _component_ids(state) -> set:
    return {c.id for c in build_layout(state)._traverse() if isinstance(getattr(c, "id", None), str)}  # noqa: SLF001


def test_the_astar_checkbox_and_the_hardware_radio_are_gone(state):
    found = _component_ids(state)
    assert "route-use-astar" not in found
    assert "route-hardware" not in found
    assert {ids.GRAPH_API, ids.ALGORITHM, ids.NEIGHBORHOOD, ids.FULL_WINDOW, ids.DELTA, ids.NUM_THREADS,
            ids.MARGIN} <= found


def test_dead_id_constants_and_stores_are_gone(state):
    for name in ("DRAG_EVENT", "RECOMPUTE_FLASH", "EDIT_MODE", "DRAW_TARGET_RADIO", "LAYER_TABLE_HEIGHT",
                 "HARDWARE"):
        assert not hasattr(ids, name), name
    found = _component_ids(state)
    assert not {"drag-event", "recompute-flash"} & found
    assert ids.GRAPH_API in found            # the old unused constant is now the Backend dropdown


def test_no_code_reads_use_astar_any_more():
    from pathlib import Path

    gui = Path(routing.__file__).resolve().parents[1]
    hits = [f"{path.relative_to(gui)}:{number}" for path in gui.rglob("*.py")
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if "use_astar" in line and 'pop("use_astar"' not in line and "dead key" not in line]
    assert hits == []                  # only the line that DROPS the old key from a stored route is left


def test_run_routing_stores_no_astar_param(raster_path):
    _finder, results, _failed = routing.run_routing(
        raster_path, sources=[SOURCE], targets=[TARGET], algorithm="dijkstra", neighborhood="r1",
        search_buffer_m=80)
    assert "use_astar" not in results[0].params
    results[0].session.close()


def _legacy_route(state, raster_layer, **extra):
    """A route as an older project stored it: params with ``use_astar`` and no backend / margin keys."""
    return add_route_layer(
        state, line=LineString([SOURCE, TARGET]),
        route_cost=RouteCost(total_length=226.0, total_cost=300.0, total_cell_cost=200.0, geodesic_length_m=226.0),
        params={"algorithm": "dijkstra", "graph_api": "cython", "neighborhood": "r1", "search_space_buffer_m": 80,
                "ignore_max_cost": True, "hardware": "cpu", "delta": 100, "num_threads": 0, "use_astar": False,
                "raster_layer_id": raster_layer.id, **extra},
        control_points=[list(SOURCE), list(TARGET)], crs=TEST_CRS)


def test_an_old_route_with_use_astar_edits_and_drops_the_dead_key(state, raster_layer):
    route = _legacy_route(state, raster_layer)
    new_layer, notices = edit_cb.recompute_variant(
        state, route, [list(SOURCE), [TARGET[0] - 10.0, TARGET[1] + 10.0]], "moved target", [])
    assert new_layer is not None, notices
    assert "use_astar" not in new_layer.meta["params"]
    assert new_layer.meta["params"]["graph_api"] == "cython"


def test_an_old_project_with_use_astar_still_loads(state, raster_layer, tmp_path):
    _legacy_route(state, raster_layer)
    path = project_io.save_project(state, tmp_path / "old")
    manifest = json.loads(open(path, encoding="utf-8").read())
    route_entry = next(layer for layer in manifest["layers"] if layer["kind"] == "route")
    assert route_entry["meta"]["params"]["use_astar"] is False
    # an old file: no routing settings, the dead key in every route
    manifest.pop("routing_settings", None)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)
    state.clear()
    project_io.load_project(state, path)
    route = state.layers_of_kind("route")[0]
    assert route.meta["params"]["use_astar"] is False                  # kept as stored; nothing reads it
    assert state.routing_settings == {}


# ============================== G: conditional controls say so, zero is a value
@pytest.mark.parametrize("graph_api,algorithm,delta_on,threads_on,margin_on", [
    ("cython", "dijkstra", False, False, False),
    ("cython", "delta-stepping", True, True, True),
    ("cython", "delta-stepping-circular", True, True, True),
    ("networkx", "dijkstra", False, False, False),
    ("igraph", "bellman_ford", False, False, False),
    ("raster_gpu", "delta-stepping", True, False, True),
    ("raster_gpu", "dijkstra", True, False, True),
    ("raster_fim", "fim", False, False, False)])
def test_delta_threads_and_margin_are_disabled_where_they_do_nothing(app, graph_api, algorithm, delta_on,
                                                                     threads_on, margin_on):
    resp = invoke(app, (f"{ids.DELTA}.disabled",), algorithm, graph_api)
    assert resp[ids.DELTA]["disabled"] is (not delta_on)
    assert resp[ids.NUM_THREADS]["disabled"] is (not threads_on)
    assert resp[ids.MARGIN]["disabled"] is (not margin_on)
    for wrap, disabled in ((ids.DELTA_WRAP, not delta_on), (ids.THREADS_WRAP, not threads_on),
                           (ids.MARGIN_WRAP, not margin_on)):
        title = resp[wrap]["title"]
        assert title and (title.startswith("Not used") is disabled)    # a one-line reason when switched off


def test_the_help_titles_are_on_the_new_controls(state):
    layout = build_layout(state)
    by_id = {c.id: c for c in layout._traverse() if isinstance(getattr(c, "id", None), str)}  # noqa: SLF001
    for wrap in (ids.DELTA_WRAP, ids.THREADS_WRAP, ids.MARGIN_WRAP):
        assert by_id[wrap].title
    titles = " | ".join(str(c.title) for c in layout._traverse() if isinstance(getattr(c, "title", None), str))  # noqa: SLF001
    for fragment in ("graph backend", "k = 0", "entire raster", "Bucket width", "Worker threads", "termination margin"):
        assert fragment in titles, fragment


def test_zero_is_a_value_in_the_routing_numbers(app, raster_layer, run_calls):
    _run(app, raster_layer.id, algorithm="delta-stepping", threads=0, k=0, simplify=True, simplify_tol=0)
    call = run_calls[0]
    assert call["num_threads"] == 0 and call["neighborhood"] == "r0"
    assert number_or_default(0, 100) == 0 and number_or_default(None, 100) == 100
    assert number_or_default("", 100) == 100


def test_zero_simplify_tolerance_is_kept_not_replaced_by_the_default(app, state, raster_layer):
    _run(app, raster_layer.id, simplify=True, simplify_tol=0)
    assert state.layers_of_kind("route")[0].meta["simplify_tol"] == 0.0


# ================================================================ H: constrained runs
@pytest.fixture(scope="module")
def tiny_raster(tmp_path_factory):
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds

    path = tmp_path_factory.mktemp("r13_tiny") / "uniform.tif"
    with rasterio.open(str(path), "w", driver="GTiff", height=100, width=100, count=1, dtype="uint16",
                       crs=TEST_CRS, transform=from_bounds(0, 0, 100, 100, 100, 100)) as dst:
        dst.write(np.full((100, 100), 100, dtype=np.uint16), 1)
    return str(path)


MINI_PROFILE = {
    "soft_angle_limit_deg": 20.0, "hard_angle_limit_deg": 60.0, "angle_cost_function": "linear",
    "angle_cost_params": {"scale": 10.0}, "min_span_m": 20, "max_span_m": 40, "span_bin_size_m": 1.0,
    "tower_cost_function": "terrain_plus_angle",
    "tower_cost_params": {"terrain_cost_map": {"0": 100, "500": 200}, "terrain_interpolation": "linear",
                          "angle_types": {"suspension": {"max_angle_deg": 20.0, "base_cost": 100},
                                          "strain": {"max_angle_deg": 60.0, "base_cost": 300}}}}


def _ohl(app, raster_id, draft, *, k=1, buffer=100):
    import yaml

    return invoke(app, OHL_WIRING, 1, True, draft, raster_id, yaml.safe_dump(MINI_PROFILE), "cython", False, k,
                  buffer, None, None, [], [], triggered=[f"{ids.RUN_ROUTING_BTN}.n_clicks"])


@pytest.fixture()
def ohl_layer(app, state, tiny_raster):
    invoke(app, ("layers-view.data", ids.RASTER_LOAD_BTN), 1, tiny_raster, "viridis", [],
           triggered=[f"{ids.RASTER_LOAD_BTN}.n_clicks"])
    return state.layers_of_kind("raster")[0]


def _point(x, y):
    return {"lat": 0, "lng": 0, "x": x, "y": y}


def test_a_constrained_run_with_several_points_says_which_are_left_out(app, ohl_layer, monkeypatch):
    seen: list[dict] = []

    def fake(*_args, **kwargs):
        seen.append(kwargs)
        return None

    monkeypatch.setattr(constrained, "run_constrained_path", fake)
    draft = {"sources": [_point(10.0, 10.0), _point(20.0, 10.0), _point(30.0, 10.0)],
             "targets": [_point(90.0, 90.0), _point(80.0, 90.0)], "waypoints": []}
    resp = _ohl(app, ohl_layer.id, draft)
    assert seen[0]["points"] == [(10.0, 10.0), (90.0, 90.0)]           # first source, first target
    notice = next(n for n in _notices(resp) if n["title"] == "Only the first source and target are routed")
    assert "2 more source" in notice["meaning"] and "1 more target" in notice["meaning"]
    assert notice["severity"] == "warning"


def test_a_constrained_run_with_one_pair_gives_no_such_notice(app, ohl_layer, monkeypatch):
    monkeypatch.setattr(constrained, "run_constrained_path", lambda *_a, **_k: None)
    resp = _ohl(app, ohl_layer.id, _draft((10.0, 10.0), (90.0, 90.0)))
    assert "Only the first source and target are routed" not in [n["title"] for n in _notices(resp)]


def test_constrained_k_reaches_the_service_and_a_bad_k_is_a_field_error(app, ohl_layer, monkeypatch):
    seen: list[dict] = []
    monkeypatch.setattr(constrained, "run_constrained_path", lambda *_a, **kw: seen.append(kw))
    draft = _draft((10.0, 10.0), (90.0, 90.0))
    _ohl(app, ohl_layer.id, draft, k=3)
    assert seen[0]["neighborhood"] == "r3"
    resp = _ohl(app, ohl_layer.id, draft, k=200)
    assert len(seen) == 1
    assert _notices(resp)[-1]["focus_control"] == ids.NEIGHBORHOOD


def test_an_overhead_line_route_is_not_silently_recomputed_as_a_plain_route(app, state, raster_layer):
    route = _legacy_route(state, raster_layer, constrained=True, profile=MINI_PROFILE, backend="cython")
    before = len(state.layers_of_kind("route"))
    new_layer, notices = edit_cb.recompute_variant(
        state, route, [list(SOURCE), [TARGET[0] - 10.0, TARGET[1] + 10.0]], "moved target", [])
    assert new_layer is None
    assert notices[-1]["title"] == "Overhead-line routes cannot be edited here"
    assert notices[-1]["focus_control"] == ids.OHL_ENABLE
    assert len(state.layers_of_kind("route")) == before and state.get(route.id) is route     # untouched


def test_run_busy_flag_follows_each_runner(app):
    resp = invoke(app, (f"{ids.RUN_ROUTING_BUSY}.data", ids.RUN_PLAIN_BUSY), False, True)
    assert resp[ids.RUN_ROUTING_BUSY]["data"] is True                 # a constrained run is busy
    resp = invoke(app, (f"{ids.RUN_ROUTING_BUSY}.data", ids.RUN_PLAIN_BUSY), True, False)
    assert resp[ids.RUN_ROUTING_BUSY]["data"] is True                 # a plain run is busy
    constrained_specs = [s for s in app._callback_list if ids.OHL_STATUS in str(s["output"])]  # noqa: SLF001
    assert constrained_specs[0]["running"]["running"][f"{ids.RUN_OHL_BUSY}.data"] is True


# ============================== I: the settings are saved, restored and prefilled
CUSTOM_SETTINGS = {"graph_api": "networkx", "algorithm": "astar", "neighborhood_k": 4, "search_buffer_m": 120.0,
                   "full_window": True, "ignore_max_cost": False, "pairwise": True, "simplify": True,
                   "simplify_tol": 2.5, "delta": 50, "num_threads": 3, "margin": 1.5}


def test_the_controls_are_mirrored_into_the_server_state(app, state):
    resp = invoke(app, (f"{ids.ROUTING_SETTINGS}.data",), "networkx", "astar", 4, 120.0, True, False, True, True,
                  2.5, 50, 3, 1.5)
    assert state.routing_settings == CUSTOM_SETTINGS == resp[ids.ROUTING_SETTINGS]["data"]
    invoke(app, (f"{ids.ROUTING_SETTINGS}.data",), None, None, None, None, False, True, False, False, None, None,
           None, None)                                             # blank fields: the defaults
    assert state.routing_settings["graph_api"] == "cython"
    assert state.routing_settings["algorithm"] == "delta-stepping"
    assert state.routing_settings["neighborhood_k"] == 2 and state.routing_settings["margin"] is None


def test_the_settings_round_trip_through_the_project_file(state, tmp_path):
    state.routing_settings = routing.normalise_routing_settings(CUSTOM_SETTINGS)
    path = project_io.save_project(state, tmp_path / "proj")
    manifest = json.loads(open(path, encoding="utf-8").read())
    assert manifest["routing_settings"] == CUSTOM_SETTINGS
    state.clear()
    assert state.routing_settings == {}
    project_io.load_project(state, path)
    assert state.routing_settings == CUSTOM_SETTINGS


def test_a_project_without_the_settings_loads_with_the_defaults(state, tmp_path):
    path = project_io.save_project(state, tmp_path / "old")
    manifest = json.loads(open(path, encoding="utf-8").read())
    manifest.pop("routing_settings")                               # a file written before this round
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)
    state.routing_settings = routing.normalise_routing_settings(CUSTOM_SETTINGS)
    project_io.load_project(state, path)
    assert state.routing_settings == {}
    by_id = {c.id: c for c in build_layout(state)._traverse() if isinstance(getattr(c, "id", None), str)}  # noqa: SLF001
    assert by_id[ids.GRAPH_API].value == "cython" and by_id[ids.NEIGHBORHOOD].value == 2
    assert by_id[ids.DELTA].value == 100 and getattr(by_id[ids.MARGIN], "value", None) is None


def test_the_layout_is_prefilled_from_the_state_after_a_reload(state):
    state.routing_settings = routing.normalise_routing_settings(CUSTOM_SETTINGS)
    by_id = {c.id: c for c in build_layout(state)._traverse() if isinstance(getattr(c, "id", None), str)}  # noqa: SLF001
    assert by_id[ids.GRAPH_API].value == "networkx" and by_id[ids.ALGORITHM].value == "astar"
    assert [o["value"] for o in by_id[ids.ALGORITHM].options] == routing.backend_algorithms("networkx")
    assert by_id[ids.NEIGHBORHOOD].value == 4 and by_id[ids.SEARCH_BUFFER].value == 120.0
    assert by_id[ids.FULL_WINDOW].value is True and by_id[ids.IGNORE_MAX_COST].value is False
    assert by_id[ids.PAIRWISE].value is True and by_id[ids.SIMPLIFY].value is True
    assert by_id[ids.SIMPLIFY_TOL].value == 2.5 and by_id[ids.DELTA].value == 50
    assert by_id[ids.NUM_THREADS].value == 3 and by_id[ids.MARGIN].value == 1.5


def test_unusable_saved_settings_fall_back_to_usable_ones():
    out = routing.normalise_routing_settings({"graph_api": "cugraph", "algorithm": "banana", "neighborhood_k": "x",
                                              "margin": float("nan"), "full_window": "yes", "delta": True})
    assert out["graph_api"] == "cython" and out["algorithm"] == "delta-stepping"
    assert out["neighborhood_k"] == 2 and out["margin"] is None
    assert out["full_window"] is False and out["delta"] == 100
    mismatch = routing.normalise_routing_settings({"graph_api": "raster_gpu", "algorithm": "delta-stepping-circular"})
    assert mismatch["algorithm"] == "delta-stepping"
    assert routing.normalise_routing_settings(None) == {} == routing.normalise_routing_settings({})
