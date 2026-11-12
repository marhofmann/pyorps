"""Round 14, P9 - the shared-trench corridor service (``services/corridor.py``).

Every raster here is a tiny GeoTIFF whose open cells form a one-cell-wide network and whose other cells are forbidden
(65535, hard barriers). With the 4-neighbour step table (``r0``) the route between two points on such a network is
unique, so every length asserted below can be counted by hand and checked with an independent breadth-first search
instead of being read back from the code under test.

Networks (cell size 2 m, cells of value 100):

* Y: a trunk along row 10 (columns 0 to 39) and a branch up column 20 (rows 0 to 9). Points A (row 10, column 0),
  B (row 0, column 20, the end of the branch) and C (row 10, column 39). The library joins A-B and B-C; both routes
  use the branch, so the branch (10 steps = 20 m) is the one shared section and the corridor is 49 steps = 98 m.
* split: two separate lines, 10 and 15 steps long. No route can use a cell of the other line, so nothing is shared.
* line: one row of 60 open cells with points 10 and 30 steps apart (pair costs 2000 and 6000 cost units x m).
"""
from __future__ import annotations

import ast
import hashlib
import json
import threading
from collections import deque
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

from pyorps.core.exceptions import InvalidInputError
from pyorps.gui.services import corridor, jobs

CELL = 2.0
X0, Y0 = 442000.0, 5587000.0
CRS = "EPSG:25832"
WALL = 65535
OPEN = 100
JOIN_S = 60


def xy(row: int, col: int) -> tuple[float, float]:
    """The centre of a cell of the test rasters."""
    return (X0 + (col + 0.5) * CELL, Y0 - (row + 0.5) * CELL)


def _write(path: Path, array: np.ndarray) -> str:
    transform = Affine(CELL, 0.0, X0, 0.0, -CELL, Y0)
    with rasterio.open(path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1], count=1,
                       dtype=str(array.dtype), crs=CRS, transform=transform) as dataset:
        dataset.write(array, 1)
    return str(path)


def _network(rows: int, cols: int, open_cells: list[tuple[slice | int, slice | int]]) -> np.ndarray:
    array = np.full((rows, cols), WALL, dtype=np.uint16)
    for r, c in open_cells:
        array[r, c] = OPEN
    return array


Y_A, Y_B, Y_C = (10, 0), (0, 20), (10, 39)
SPLIT_A, SPLIT_B, SPLIT_C, SPLIT_D = (3, 0), (3, 10), (12, 5), (12, 20)


@pytest.fixture(scope="module")
def y_net(tmp_path_factory) -> str:
    array = _network(16, 44, [(10, slice(0, 40)), (slice(0, 10), 20)])
    return _write(tmp_path_factory.mktemp("corridor_y") / "y.tif", array)


@pytest.fixture(scope="module")
def split_net(tmp_path_factory) -> str:
    array = _network(16, 44, [(3, slice(0, 11)), (12, slice(5, 21))])
    return _write(tmp_path_factory.mktemp("corridor_split") / "split.tif", array)


@pytest.fixture(scope="module")
def line_net(tmp_path_factory) -> str:
    array = _network(5, 60, [(2, slice(0, 60))])
    return _write(tmp_path_factory.mktemp("corridor_line") / "line.tif", array)


def _build(path, cells, settings=None, **kwargs):
    kwargs.setdefault("neighborhood", "r0")
    return corridor.build_corridor(path, [xy(*c) for c in cells], settings, **kwargs)


@pytest.fixture(scope="module")
def y_result(y_net):
    return _build(y_net, [Y_A, Y_B, Y_C])


@pytest.fixture(scope="module")
def split_result(split_net):
    return _build(split_net, [SPLIT_A, SPLIT_B, SPLIT_C, SPLIT_D])


def _draft(sources=(), waypoints=(), targets=()):
    def points(cells):
        return [{"lat": 0.0, "lng": 0.0, "x": xy(*c)[0], "y": xy(*c)[1], "name": ""} for c in cells]
    return {"sources": points(sources), "waypoints": points(waypoints), "targets": points(targets)}


Y_DRAFT = _draft(sources=[Y_A], waypoints=[Y_B], targets=[Y_C])


# ------------------------------------------------------------------ the independent oracle
def _shortest_path(open_cells: set, start: tuple, goal: tuple) -> list:
    """Breadth-first search over 4-neighbour open cells (the route is unique on these networks)."""
    previous = {start: None}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        if cell == goal:
            break
        r, c = cell
        for step in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if step in open_cells and step not in previous:
                previous[step] = cell
                queue.append(step)
    path, cell = [], goal
    while cell is not None:
        path.append(cell)
        cell = previous[cell]
    return path[::-1]


def _oracle(path: str, order: list[tuple], pairs) -> tuple[float, float]:
    """``(total_m, shared_m)`` of the given point pairs, counted edge by edge from the raster alone.

    ``order`` maps a terminal id to its (row, column); ``pairs`` are the (id, id) pairs the library connected.
    """
    with rasterio.open(path) as src:
        data = src.read(1)
    open_cells = {(int(r), int(c)) for r, c in zip(*np.nonzero(data != WALL))}
    use: dict[frozenset, int] = {}
    for a, b in pairs:
        route = _shortest_path(open_cells, order[a], order[b])
        for edge in {frozenset(step) for step in zip(route, route[1:])}:
            use[edge] = use.get(edge, 0) + 1
    return len(use) * CELL, sum(1 for n in use.values() if n > 1) * CELL


# ================================================================ settings
def test_defaults_and_specs_are_one_table():
    assert corridor.SETTING_KEYS == ("terminals", "k_per_pair", "min_shared_length_m", "max_pair_cost",
                                     "validate_pair_costs")
    assert corridor.CORRIDOR_DEFAULTS == {"terminals": "all", "k_per_pair": 1, "min_shared_length_m": 0.0,
                                          "max_pair_cost": None, "validate_pair_costs": False}
    assert corridor.corridor_problems(corridor.CORRIDOR_DEFAULTS) == []
    for spec in corridor.FIELD_SPECS.values():
        assert spec.label and spec.help and corridor.describe_range(spec)
    assert corridor.FIELD_SPECS["min_shared_length_m"].unit == "m"
    assert corridor.FIELD_SPECS["max_pair_cost"].unit == "cost units x m"


def test_normalise_gives_empty_for_nothing_and_a_full_dict_otherwise():
    for raw in (None, {}, [], "x", 3):
        assert corridor.normalise_corridor_settings(raw) == {}
    out = corridor.normalise_corridor_settings({"k_per_pair": 3})
    assert out == {**corridor.CORRIDOR_DEFAULTS, "k_per_pair": 3}
    assert list(out) == list(corridor.SETTING_KEYS)


def test_normalise_keeps_usable_values_and_replaces_unusable_ones():
    out = corridor.normalise_corridor_settings({
        "terminals": "sources_targets", "k_per_pair": "2", "min_shared_length_m": 12.5,
        "max_pair_cost": "250.5", "validate_pair_costs": True})
    assert out == {"terminals": "sources_targets", "k_per_pair": 2, "min_shared_length_m": 12.5,
                   "max_pair_cost": 250.5, "validate_pair_costs": True}
    bad = corridor.normalise_corridor_settings({
        "terminals": "everything", "k_per_pair": float("nan"), "min_shared_length_m": "far",
        "max_pair_cost": "", "validate_pair_costs": "yes"})
    assert bad == corridor.CORRIDOR_DEFAULTS


def test_normalise_never_clamps_and_survives_a_json_round_trip():
    raw = {"k_per_pair": 9, "min_shared_length_m": -4, "max_pair_cost": 0}
    out = corridor.normalise_corridor_settings(raw)
    assert (out["k_per_pair"], out["min_shared_length_m"], out["max_pair_cost"]) == (9, -4, 0)
    full = corridor.normalise_corridor_settings({"k_per_pair": 2, "max_pair_cost": 99.5})
    assert json.loads(json.dumps(full)) == full


@pytest.mark.parametrize(("key", "raw", "fragments"), [
    ("k_per_pair", 7, ["Crossings kept per point pair", "k_per_pair", "7 is too high", "<= 5", "whole numbers"]),
    ("k_per_pair", 0, ["0 is too low", ">= 1"]),
    ("k_per_pair", 2.5, ["2.5 is not a whole number"]),
    ("k_per_pair", "many", ["'many' is not a usable number"]),
    ("k_per_pair", None, ["blank, but a value is required", ">= 1 and <= 5"]),
    ("k_per_pair", "", ["blank, but a value is required"]),
    ("k_per_pair", True, ["true is not a usable number"]),
    ("min_shared_length_m", -5, ["Ignore shared runs shorter than", "-5 m is too low", ">= 0 and <= 10000 m"]),
    ("min_shared_length_m", 10001, ["10001 m is too high"]),
    ("min_shared_length_m", float("inf"), ["is not a usable number"]),
    ("min_shared_length_m", None, ["blank, but a value is required"]),
    ("max_pair_cost", 0, ["Skip pairs costing more than", "0 cost units x m is too low", "> 0 cost units x m",
                          "blank = every pair is kept"]),
    ("max_pair_cost", -3, ["-3 cost units x m is too low"]),
    ("max_pair_cost", "lots", ["'lots' is not a usable number"]),
    ("validate_pair_costs", "yes", ["Also compute exact pair costs", "'yes' is not on or off", "on or off"]),
    ("terminals", "none", ["Points to connect", "'none' is not an option", "all, sources_targets"]),
    ("terminals", None, ["blank is not an option"]),
])
def test_a_bad_value_is_named_with_value_range_and_unit(key, raw, fragments):
    problems = corridor.corridor_problems({key: raw})
    assert [k for k, _ in problems] == [key]
    for fragment in fragments:
        assert fragment in problems[0][1], (fragment, problems[0][1])


@pytest.mark.parametrize(("key", "raw"), [
    ("k_per_pair", 1), ("k_per_pair", 5), ("k_per_pair", "3"), ("k_per_pair", 2.0),
    ("min_shared_length_m", 0), ("min_shared_length_m", 10000), ("min_shared_length_m", "7.5"),
    ("max_pair_cost", None), ("max_pair_cost", ""), ("max_pair_cost", 0.01), ("max_pair_cost", "1e6"),
    ("validate_pair_costs", True), ("validate_pair_costs", False), ("validate_pair_costs", None),
    ("terminals", "all"), ("terminals", "sources_targets"),
])
def test_values_inside_the_range_are_fine(key, raw):
    assert corridor.corridor_problems({key: raw}) == []


def test_absent_keys_count_as_defaults_and_unknown_keys_are_ignored():
    assert corridor.corridor_problems(None) == []
    assert corridor.corridor_problems({}) == []
    assert corridor.corridor_problems({"colour": "red"}) == []
    assert [k for k, _ in corridor.corridor_problems({"k_per_pair": 0, "min_shared_length_m": -1})] == [
        "k_per_pair", "min_shared_length_m"]
    assert [k for k, _ in corridor.corridor_problems("nope")] == ["settings"]


def test_problem_notice_and_text():
    problems = corridor.corridor_problems({"k_per_pair": 9, "max_pair_cost": 0})
    notice = corridor.problem_notice(problems, controls={"k_per_pair": "some-component"})
    assert notice["severity"] == "warning" and "Crossings kept per point pair" in notice["title"]
    assert "k_per_pair" in notice["meaning"] and notice["focus_control"] == "some-component"
    assert notice["focus_id"] == "tab-routes" and notice["fix"] and notice["impact"]
    assert corridor.problem_notice([]) is None
    text = corridor.problems_text([("a", "one"), ("b", "two"), ("c", "three"), ("d", "four")], limit=2)
    assert text == "one\ntwo\n... and 2 more."
    assert corridor.problem_notice([("points", "x")])["title"] == "The points cannot be connected"
    assert corridor.label_for("k_per_pair") == "Crossings kept per point pair"
    assert corridor.label_for("raster") == "Cost raster" and corridor.label_for("zzz") == "zzz"


def test_the_service_imports_nothing_from_dash_ids_or_the_layout():
    tree = ast.parse(Path(corridor.__file__).read_text("utf-8"))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules += [("." * node.level) + (node.module or "") + "." + alias.name for alias in node.names]
    banned = ("dash", "ids", "layout", "sections", "callbacks")
    offenders = [m for m in modules if any(part in banned for part in m.replace("..", ".").split("."))]
    assert not offenders, offenders


# ================================================================ the points
def test_terminals_from_draft_orders_sources_waypoints_targets():
    draft = _draft(sources=[(1, 1), (1, 2)], waypoints=[(2, 2)], targets=[(3, 3)])
    everything = corridor.terminals_from_draft(draft, "all")
    assert everything == [xy(1, 1), xy(1, 2), xy(2, 2), xy(3, 3)]
    assert corridor.terminals_from_draft(draft, "sources_targets") == [xy(1, 1), xy(1, 2), xy(3, 3)]
    assert corridor.terminals_from_draft(draft) == everything
    assert all(isinstance(v, float) for point in everything for v in point)


def test_terminals_from_draft_removes_nothing_and_takes_pairs_too():
    draft = {"sources": [[5.0, 6.0], (5.0, 6.0)], "targets": [{"x": "7", "y": 8}]}
    assert corridor.terminals_from_draft(draft) == [(5.0, 6.0), (5.0, 6.0), (7.0, 8.0)]
    assert corridor.terminals_from_draft(None) == [] and corridor.terminals_from_draft({}) == []


@pytest.mark.parametrize("draft", [{"sources": [{"x": 1}]}, {"targets": ["ab"]}, {"waypoints": [None]},
                                   {"sources": [{"x": "a", "y": 2}]}, [1, 2]])
def test_terminals_from_draft_refuses_an_entry_that_is_not_a_point(draft):
    with pytest.raises(InvalidInputError):
        corridor.terminals_from_draft(draft)


def test_terminals_from_draft_refuses_an_unknown_choice():
    with pytest.raises(InvalidInputError, match="not an option"):
        corridor.terminals_from_draft({}, "everything")


def test_good_points_have_no_problems(y_net):
    assert corridor.terminal_problems(y_net, [xy(*Y_A), xy(*Y_B), xy(*Y_C)]) == []


@pytest.mark.parametrize("cells", [[], [Y_A]])
def test_fewer_than_two_points_are_refused(y_net, cells):
    problems = corridor.terminal_problems(y_net, [xy(*c) for c in cells])
    assert [k for k, _ in problems] == ["points"]
    assert "at least 2 points" in problems[0][1] and f"gives {len(cells)}" in problems[0][1]


def test_a_point_outside_the_raster_is_refused_with_its_number_and_the_raster_extent(y_net):
    problems = corridor.terminal_problems(y_net, [xy(*Y_A), (X0 - 50.0, Y0), xy(*Y_C), (X0 + 500.0, Y0 - 3.0)])
    assert [k for k, _ in problems] == ["points"]
    message = problems[0][1]
    assert "2 point(s) lie outside the cost raster" in message
    assert "point 2 " in message and "point 4 " in message and "point 1 " not in message
    assert "442,000.0 to 442,088.0" in message and CRS in message


def test_the_far_edge_of_the_raster_is_outside_but_the_near_edge_is_inside(y_net):
    right_edge = X0 + 44 * CELL
    assert corridor.terminal_problems(y_net, [xy(*Y_A), (right_edge, Y0 - 1.0)])
    assert not [m for k, m in corridor.terminal_problems(y_net, [xy(*Y_A), (X0, Y0 - 1.0)]) if "outside" in m]


def test_a_point_on_a_forbidden_cell_is_refused_only_while_forbidden_cells_are_barriers(y_net):
    on_wall = xy(0, 0)
    problems = corridor.terminal_problems(y_net, [xy(*Y_A), on_wall, xy(*Y_C)])
    assert [k for k, _ in problems] == ["points"]
    assert "1 point(s) lie on forbidden cells" in problems[0][1] and "point 2 " in problems[0][1]
    assert "65535" in problems[0][1]
    assert corridor.terminal_problems(y_net, [xy(*Y_A), on_wall, xy(*Y_C)], ignore_max_cost=False) == []


def test_a_float_raster_counts_non_finite_cells_as_forbidden(tmp_path):
    array = np.full((6, 8), 3.0, dtype=np.float32)
    array[2, 2] = np.nan
    path = _write(tmp_path / "f.tif", array)
    problems = corridor.terminal_problems(path, [xy(0, 0), xy(2, 2), xy(5, 7)])
    assert len(problems) == 1 and "point 2 " in problems[0][1] and "forbidden" in problems[0][1]


def test_duplicate_points_are_refused_by_cell_and_exact_repeats_too(y_net):
    near = (xy(*Y_B)[0] + 0.4, xy(*Y_B)[1] - 0.4)            # another spot in the same raster cell
    problems = corridor.terminal_problems(y_net, [xy(*Y_A), xy(*Y_B), near, xy(*Y_C), xy(*Y_A)])
    assert [k for k, _ in problems] == ["points"]
    message = problems[0][1]
    assert "duplicates" in message and "points 2, 3 are on the same raster cell (row 0, column 20)" in message
    assert "points 1, 5 are on the same raster cell (row 10, column 0)" in message


def test_all_points_in_one_cell_are_a_duplicate_not_a_corridor(y_net):
    problems = corridor.terminal_problems(y_net, [xy(*Y_A), xy(*Y_A)])
    assert len(problems) == 1 and "duplicates" in problems[0][1]


def test_a_coordinate_that_is_not_finite_is_named(y_net):
    problems = corridor.terminal_problems(y_net, [xy(*Y_A), (float("nan"), 1.0), xy(*Y_C)])
    assert any("not a finite number" in m and "point 2 " in m for _, m in problems)


def test_the_numbers_stay_the_drafts_numbers_when_a_bad_coordinate_comes_first(y_net):
    problems = corridor.terminal_problems(y_net, [(float("nan"), 1.0), xy(*Y_A), (X0 - 99.0, Y0), xy(*Y_C)])
    outside = next(m for _, m in problems if "outside" in m)
    assert "point 3 " in outside


def test_an_unreadable_or_missing_raster_is_one_problem(tmp_path):
    missing = corridor.terminal_problems(str(tmp_path / "gone.tif"), [(1.0, 1.0), (2.0, 2.0)])
    assert [k for k, _ in missing] == ["raster"] and "cannot be read" in missing[0][1]
    none = corridor.terminal_problems(None, [(1.0, 1.0), (2.0, 2.0)])
    assert [k for k, _ in none] == ["raster"] and "No cost raster" in none[0][1]
    junk = tmp_path / "junk.tif"
    junk.write_bytes(b"not a raster")
    assert [k for k, _ in corridor.terminal_problems(str(junk), [(1.0, 1.0), (2.0, 2.0)])] == ["raster"]


def test_the_backend_must_be_cython():
    assert corridor.backend_problem("cython") is None and corridor.backend_problem(None) is None
    assert corridor.backend_problem(" Cython ") is None
    key, message = corridor.backend_problem("networkx")
    assert key == "graph_api" and "cython" in message and "networkx" in message


# ================================================================ cost limit and cell size
def test_the_cost_limit_converts_by_the_cell_size():
    assert corridor.max_pair_distance_cells(2000.0, 2.0) == 1000.0
    assert corridor.max_pair_distance_cells(750, 0.5) == 1500.0
    assert corridor.max_pair_distance_cells(None, 2.0) is None
    for bad in (0, -1, float("nan"), None, "2"):
        with pytest.raises(InvalidInputError, match="cell size"):
            corridor.max_pair_distance_cells(10.0, bad)


def test_the_cost_limit_cuts_pairs_at_exactly_the_converted_value(line_net):
    """Points 10 and 30 steps apart on value-100 cells of 2 m: the pairs cost 2000 and 6000 cost units x m
    (1000 and 3000 cell units). A limit read as cell units, or not converted, would cut elsewhere."""
    cells = [(2, 0), (2, 10), (2, 40)]
    keep_all = _build(line_net, cells)
    assert keep_all.summary["n_pairs"] == 2 and keep_all.summary["total_length_m"] == 80.0

    for limit, pairs, length in [(6500.0, 2, 80.0), (6000.0, 2, 80.0), (2100.0, 1, 20.0), (2000.0, 1, 20.0)]:
        result = _build(line_net, cells, {"max_pair_cost": limit})
        assert (result.summary["n_pairs"], result.summary["total_length_m"]) == (pairs, length), limit
        assert result.summary["settings"]["max_pair_cost"] == limit

    nothing = _build(line_net, cells, {"max_pair_cost": 1500.0})        # 1000 cell units would pass if not converted
    assert nothing.graph is None and nothing.summary["n_corridors"] == 0


def test_a_limit_that_cuts_everything_is_a_warning_not_an_error(line_net):
    result = _build(line_net, [(2, 0), (2, 10), (2, 40)], {"max_pair_cost": 100.0})
    assert result.graph is None and result.segments_gdf is None and result.segments_geojson is None
    assert result.summary["n_corridors"] == 0 and result.summary["text"] == "No corridors were built."
    [notice] = result.notices
    assert notice["severity"] == "warning" and "100" in notice["meaning"] and "limit" in notice["meaning"]
    assert "Raise" in notice["fix"] and "unchanged" in notice["impact"]


# ================================================================ the graph: shared and not shared
def test_two_pairs_share_the_branch_and_the_numbers_match_an_independent_count(y_net, y_result):
    summary, graph = y_result.summary, y_result.graph
    pairs = sorted(graph.pair_routes)
    assert pairs == [(0, 1), (1, 2)], "the library joins A-B and B-C: both routes use the branch"
    total_m, shared_m = _oracle(y_net, [Y_A, Y_B, Y_C], pairs)
    assert (total_m, shared_m) == (49 * CELL, 10 * CELL)             # counted by hand: 20 + 19 trunk steps, 10 branch
    assert summary["total_length_m"] == pytest.approx(total_m) == pytest.approx(98.0)
    assert summary["shared_length_m"] == pytest.approx(shared_m) == pytest.approx(20.0)
    assert summary["shared_fraction"] == pytest.approx(20.0 / 98.0)
    assert (summary["n_terminals"], summary["n_pairs"], summary["n_corridors"]) == (3, 2, 3)
    assert (summary["n_shared_sections"], summary["n_junctions"]) == (1, 1)
    assert summary["per_route_length_m"] == pytest.approx(118.0)       # A-B 30 steps + B-C 29 steps, each on its own
    assert summary["saved_length_m"] == pytest.approx(20.0)            # the branch dug once instead of twice
    assert summary["total_construction_cost"] == pytest.approx(98.0 * OPEN)
    assert summary["shared_construction_cost"] == pytest.approx(20.0 * OPEN)
    assert summary["cell_size_m"] == CELL and summary["crs"] == CRS and summary["n_points"] == 3


def test_the_summary_equals_the_graphs_own_totals(y_result):
    graph, summary = y_result.graph, y_result.summary
    assert summary["total_length_m"] == graph.total_length
    assert summary["n_corridors"] == len(graph.segments)
    assert summary["total_construction_cost"] == graph.total_construction_cost
    assert summary["n_shared_sections"] == len(graph.shared_segments)
    assert summary["shared_length_m"] == sum(s.length for s in graph.shared_segments)
    assert summary["n_junctions"] == len(graph.junctions)
    assert summary["n_routes"] == len(graph.pair_routes)
    frame = y_result.segments_gdf
    assert float(frame["length_m"].sum()) == pytest.approx(summary["total_length_m"])
    assert float(frame.loc[frame["shared"] == 1, "length_m"].sum()) == pytest.approx(summary["shared_length_m"])
    assert "3 points joined by 3 trench sections; 1 of them shared" in summary["text"]
    assert corridor.summary_text(summary) == summary["text"]


def test_pairs_that_cannot_share_report_zero_shared_length(split_net, split_result):
    summary, graph = split_result.summary, split_result.graph
    pairs = sorted(graph.pair_routes)
    assert pairs == [(0, 1), (2, 3)]
    total_m, shared_m = _oracle(split_net, [SPLIT_A, SPLIT_B, SPLIT_C, SPLIT_D], pairs)
    assert (total_m, shared_m) == (25 * CELL, 0.0)
    assert summary["total_length_m"] == pytest.approx(50.0) and summary["shared_length_m"] == 0.0
    assert (summary["n_corridors"], summary["n_shared_sections"], summary["n_junctions"]) == (2, 0, 0)
    assert summary["shared_fraction"] == 0.0 and summary["saved_length_m"] == pytest.approx(0.0)
    assert split_result.shared_gdf is None and split_result.shared_geojson is None
    assert split_result.notices == []


def test_routes_that_only_meet_at_a_point_do_not_share(y_net):
    """The middle point sits on the trunk where the branch joins: both routes end there and run on opposite sides."""
    result = _build(y_net, [Y_A, (10, 20), Y_C])
    assert sorted(result.graph.pair_routes) == [(0, 1), (1, 2)]
    assert result.summary["total_length_m"] == pytest.approx(39 * CELL)
    assert result.summary["shared_length_m"] == 0.0 and result.summary["n_shared_sections"] == 0
    assert result.summary["n_corridors"] == 2


def test_two_points_make_one_route_and_nothing_to_share(y_net):
    result = _build(y_net, [Y_A, Y_C])
    assert result.summary["n_corridors"] == 1 and result.summary["total_length_m"] == pytest.approx(39 * CELL)
    assert result.summary["n_shared_sections"] == 0 and result.summary["shared_length_m"] == 0.0


def test_the_minimum_shared_length_is_in_metres(y_net):
    """The shared branch is 20 m (10 cells of 2 m): a threshold of 19 m keeps it, 21 m drops it."""
    kept = _build(y_net, [Y_A, Y_B, Y_C], {"min_shared_length_m": 19.0})
    assert kept.summary["shared_length_m"] == pytest.approx(20.0)
    dropped = _build(y_net, [Y_A, Y_B, Y_C], {"min_shared_length_m": 21.0})
    assert dropped.summary["shared_length_m"] == 0.0 and dropped.summary["n_shared_sections"] == 0


def test_k_per_pair_reaches_the_library(y_net):
    result = _build(y_net, [Y_A, Y_B, Y_C], {"k_per_pair": 2})
    assert result.graph.provenance["k_per_pair"] == 2 and result.settings["k_per_pair"] == 2


def test_the_validation_switch_adds_what_the_exact_search_found(y_net, y_result):
    assert y_result.summary["validation"] is None
    result = _build(y_net, [Y_A, Y_B, Y_C], {"validate_pair_costs": True})
    validation = result.summary["validation"]
    assert validation["n_pairs"] == 2
    assert validation["max_excess"] == pytest.approx(0.0) and validation["min_excess"] == pytest.approx(0.0)
    assert validation["worst_pair_points"] in ([1, 2], [2, 3])
    assert "Exact pair costs were computed for 2 pair(s)" in validation["text"]
    assert validation["text"] in result.summary["text"]
    assert result.notices == []                      # an exact network gives no warning


# ================================================================ the result for the poll side
def test_the_frames_and_the_graph_are_what_the_layers_and_the_save_need(y_result):
    graph, segments, nodes = y_result.graph, y_result.segments_gdf, y_result.nodes_gdf
    assert len(segments) == len(graph.segments) == 3 and len(nodes) == len(graph.nodes) == 4
    assert str(segments.crs) == CRS == str(nodes.crs) == str(y_result.crs)
    assert {"segment_id", "length_m", "construction_cost", "use_count", "shared", "members"} <= set(segments.columns)
    assert sorted(nodes["kind"]) == ["junction", "terminal", "terminal", "terminal"]
    shared = y_result.shared_gdf
    assert len(shared) == 1 and int(shared["use_count"].iloc[0]) == 2
    assert shared["length_m"].iloc[0] == pytest.approx(20.0)


def test_the_geojson_is_wgs84_and_matches_the_frames(y_result):
    for frame, geojson in ((y_result.segments_gdf, y_result.segments_geojson),
                           (y_result.nodes_gdf, y_result.nodes_geojson),
                           (y_result.shared_gdf, y_result.shared_geojson)):
        assert geojson["type"] == "FeatureCollection" and len(geojson["features"]) == len(frame)
        expected = frame.to_crs("EPSG:4326").geometry
        for feature, geometry in zip(geojson["features"], expected):
            flat = np.asarray(feature["geometry"]["coordinates"], dtype=float).reshape(-1, 2)
            lon, lat = flat[:, 0], flat[:, 1]
            assert ((-180 <= lon) & (lon <= 180) & (-90 <= lat) & (lat <= 90)).all()
            want = np.asarray(geometry.coords if geometry.geom_type != "Point" else [geometry.coords[0]])
            assert np.allclose(flat, want, atol=1e-5)
    props = y_result.segments_geojson["features"][0]["properties"]
    assert {"shared", "use_count", "length_m"} <= set(props)
    assert sum(f["properties"]["shared"] for f in y_result.segments_geojson["features"]) == 1


def test_progress_reports_four_stages_and_says_the_run_cannot_be_stopped(y_net):
    calls = []
    _build(y_net, [Y_A, Y_B, Y_C], progress=lambda *args: calls.append(args))
    assert [c[0] for c in calls] == [0, 1, 2, 3] and {c[1] for c in calls} == {4}
    assert all(corridor.CANNOT_STOP_NOTE in c[2] for c in calls)
    assert calls[1][2].startswith("reading the search window") and calls[2][2].startswith("building the shared")


def test_build_refuses_what_the_checks_refuse(y_net):
    with pytest.raises(InvalidInputError, match="at least 2 points"):
        _build(y_net, [Y_A])
    with pytest.raises(InvalidInputError, match="forbidden cells"):
        _build(y_net, [Y_A, (0, 0), Y_C])
    with pytest.raises(InvalidInputError, match="Crossings kept per point pair"):
        _build(y_net, [Y_A, Y_C], {"k_per_pair": 0})
    with pytest.raises(InvalidInputError, match="outside the cost raster"):
        corridor.build_corridor(y_net, [xy(*Y_A), (0.0, 0.0)], neighborhood="r0")
    with pytest.raises(InvalidInputError, match="list of"):
        corridor.build_corridor(y_net, "ab")
    with pytest.raises(InvalidInputError):
        corridor.build_corridor(y_net, [(1.0, 2.0, 3.0), (1.0, 2.0)])


def test_a_point_on_a_wall_is_allowed_when_forbidden_cells_are_not_barriers(tmp_path):
    """With ignore_max_cost off a forbidden cell costs 65535 per metre but can be crossed, so the check steps aside."""
    array = np.full((6, 12), OPEN, dtype=np.uint16)
    array[2, 5] = WALL
    path = _write(tmp_path / "soft.tif", array)
    result = corridor.build_corridor(path, [xy(2, 5), xy(2, 11)], {}, neighborhood="r0", ignore_max_cost=False)
    assert result.summary["n_corridors"] == 1


# ================================================================ prepare_run
def test_prepare_run_builds_a_plain_snapshot(y_net):
    plan = corridor.prepare_run(y_net, Y_DRAFT, {"k_per_pair": 2}, {"graph_api": "cython", "neighborhood_k": 0})
    assert plan.ok and plan.problems == [] and plan.notices == []
    assert plan.params["raster_path"] == y_net and plan.params["neighborhood"] == "r0"
    assert plan.params["terminals"] == [list(xy(*c)) for c in (Y_A, Y_B, Y_C)]
    assert plan.params["settings"] == {**corridor.CORRIDOR_DEFAULTS, "k_per_pair": 2}
    assert (plan.params["search_buffer_m"], plan.params["full_window"], plan.params["ignore_max_cost"]) == (
        None, False, True)
    assert json.loads(json.dumps(plan.params)) == plan.params
    assert plan.estimate["window_cells"] == 16 * 44 and plan.estimate["cell_size_m"] == CELL
    assert plan.estimate["heavy"] is False


def test_prepare_run_uses_the_routing_defaults_when_it_gets_none(y_net):
    plan = corridor.prepare_run(y_net, Y_DRAFT, None)
    assert plan.ok and plan.params["neighborhood"] == "r2"
    assert plan.params["settings"] == corridor.CORRIDOR_DEFAULTS


def test_prepare_run_passes_the_routing_windows_through(y_net):
    plan = corridor.prepare_run(y_net, Y_DRAFT, {}, {"search_buffer_m": 30, "full_window": True,
                                                    "ignore_max_cost": False})
    assert plan.ok and plan.params["search_buffer_m"] == 30.0 and plan.params["full_window"] is True
    assert plan.params["ignore_max_cost"] is False
    blank = corridor.prepare_run(y_net, Y_DRAFT, {}, {"search_buffer_m": 0})
    assert blank.params["search_buffer_m"] is None


def test_prepare_run_leaves_the_waypoints_out_when_asked(y_net):
    plan = corridor.prepare_run(y_net, Y_DRAFT, {"terminals": "sources_targets"})
    assert plan.ok and plan.params["terminals"] == [list(xy(*Y_A)), list(xy(*Y_C))]
    waypoint_only = _draft(sources=[Y_A], waypoints=[Y_B])
    only_waypoint = corridor.prepare_run(y_net, waypoint_only, {"terminals": "sources_targets"})
    assert [k for k, _ in only_waypoint.problems] == ["points"] and only_waypoint.params is None


def test_prepare_run_refuses_a_backend_other_than_cython(y_net):
    plan = corridor.prepare_run(y_net, Y_DRAFT, {}, {"graph_api": "networkx"})
    assert not plan.ok and plan.params is None
    assert [k for k, _ in plan.problems] == ["graph_api"]


def test_prepare_run_reports_every_kind_of_problem_at_once(y_net):
    draft = _draft(sources=[(0, 0)], targets=[(0, 0)])          # one point on a wall, twice
    plan = corridor.prepare_run(y_net, draft, {"k_per_pair": 9}, {"graph_api": "igraph", "neighborhood_k": 500})
    assert [k for k, _ in plan.problems[:3]] == ["graph_api", "neighborhood_k", "k_per_pair"]
    assert [k for k, _ in plan.problems[3:]] == ["points", "points"]
    notice = corridor.problem_notice(plan.problems)
    assert notice["title"] == "The corridor graph needs the cython backend"


@pytest.mark.parametrize("value", ["abc", -1, 128, float("nan")])
def test_prepare_run_names_an_unusable_neighbourhood(y_net, value):
    plan = corridor.prepare_run(y_net, Y_DRAFT, {}, {"neighborhood_k": value})
    assert [k for k, _ in plan.problems] == ["neighborhood_k"]
    assert "Neighbourhood order k" in plan.problems[0][1] and "127" in plan.problems[0][1]


def test_prepare_run_refuses_too_few_points_and_a_missing_raster(y_net, tmp_path):
    assert [k for k, _ in corridor.prepare_run(y_net, _draft(sources=[Y_A]), {}).problems] == ["points"]
    assert [k for k, _ in corridor.prepare_run(y_net, None, {}).problems] == ["points"]
    plan = corridor.prepare_run(str(tmp_path / "gone.tif"), Y_DRAFT, {})
    assert [k for k, _ in plan.problems] == ["raster"]
    bad = corridor.prepare_run(y_net, {"sources": [{"x": 1}]}, {})
    assert [k for k, _ in bad.problems] == ["points"] and "not a point" in bad.problems[0][1]


def test_prepare_run_warns_about_a_heavy_window_but_still_allows_the_run(y_net, monkeypatch):
    monkeypatch.setattr(corridor, "HEAVY_WINDOW_CELLS", 100)
    plan = corridor.prepare_run(y_net, Y_DRAFT, {})
    assert plan.ok and plan.estimate["heavy"] is True
    [notice] = plan.notices
    assert notice["severity"] == "warning" and "704" in notice["meaning"] and "cannot be stopped" in notice["impact"]


def test_estimate_run_clips_the_window_to_the_raster_and_honours_the_buffer(y_net):
    points = [xy(*Y_A), xy(*Y_C)]
    whole = corridor.estimate_run(y_net, points, full_window=True)
    assert whole["window_cells"] == 16 * 44
    small = corridor.estimate_run(y_net, points, search_buffer_m=4.0)
    # the two points on row 10 are 78 m apart (cell centres) and on one line; +- 4 m is 83 m x 8 m, inside the raster
    assert small["window_cells"] == int(83 / CELL * (8 / CELL)) == 166
    assert small["window_cells"] < whole["window_cells"] and small["buffer_m"] == 4.0
    assert corridor.estimate_run(y_net, points)["buffer_m"] == 1000.0


# ================================================================ the job
def _plan(y_net, **settings):
    plan = corridor.prepare_run(y_net, Y_DRAFT, settings, {"neighborhood_k": 0})
    assert plan.ok, plan.problems
    return plan


def _start(state, plan):
    job = jobs.start_job(state, corridor.JOB_KIND, corridor.run_job, label=corridor.JOB_LABEL,
                         params=plan.params, can_stop=False)
    assert job is not None
    job.thread.join(JOIN_S)
    assert not job.thread.is_alive(), "the corridor job did not end"
    return job


def test_a_full_job_cycle_adds_the_layers_and_keeps_the_graph(state, y_net):
    job = _start(state, _plan(y_net))
    assert job.status == "done" and job.can_stop is False and job.kind == "corridor"
    done = jobs.take_finished(state, "corridor")
    assert done is job and state.jobs == {}
    assert isinstance(done.result, corridor.CorridorResult)
    assert state.layers == {}                                  # the worker changed nothing in the project

    outcome = corridor.finalize_corridor(state, done.result, done.params)
    assert [state.layers[i].name for i in outcome.layer_ids] == [
        "Corridor segments", "Corridor shared runs", "Corridor nodes"]
    assert outcome.segments_layer_id == outcome.layer_ids[0] and outcome.nodes_layer_id == outcome.layer_ids[-1]
    assert outcome.shared_layer_id == outcome.layer_ids[1]
    assert state.corridor_graphs == {outcome.segments_layer_id: done.result.graph}
    assert all(state.layers[i].kind == "vector" for i in outcome.layer_ids)
    assert state.dirty is True
    assert [row["id"] for row in state.layers_view()] == outcome.layer_ids      # nodes drawn last, on top


def test_the_layers_carry_geometry_style_and_json_meta(state, y_net):
    result = _start(state, _plan(y_net, k_per_pair=1)).result
    outcome = corridor.finalize_corridor(state, result, {"raster_layer_id": "raster-7"})
    segments = state.layers[outcome.segments_layer_id]
    shared = state.layers[outcome.shared_layer_id]
    nodes = state.layers[outcome.nodes_layer_id]
    assert segments.gdf is result.segments_gdf and segments.geojson is result.segments_geojson
    assert str(segments.crs) == CRS and len(segments.geojson["features"]) == 3
    assert len(shared.geojson["features"]) == 1 and len(nodes.geojson["features"]) == 4
    assert segments.style == corridor.SEGMENTS_STYLE and shared.style["color"] != segments.style["color"]
    assert shared.style["weight"] > segments.style["weight"]
    assert segments.meta["corridor_role"] == "segments" and segments.meta["corridor_run"] == 1
    assert shared.meta["corridor_for"] == segments.id and nodes.meta["corridor_for"] == segments.id
    assert segments.meta["raster_layer_id"] == "raster-7"
    assert segments.meta["summary"]["n_features"] == 3 and "shared" in segments.meta["summary"]["columns"]
    for layer in (segments, shared, nodes):
        json.dumps(layer.meta)                                  # plain types: the project file can save them


def test_the_outcome_has_the_status_text_and_a_success_notice(state, y_net):
    result = _start(state, _plan(y_net)).result
    outcome = corridor.finalize_corridor(state, result)
    assert outcome.status_text == result.summary["text"] == outcome.summary["text"]
    assert outcome.notices[-1]["severity"] == "success"
    assert "3 trench sections" in outcome.notices[-1]["title"]
    assert "GeoPackage" in outcome.notices[-1]["impact"]
    assert outcome.notices[:-1] == result.notices


def test_a_second_run_gets_numbered_layers_and_dead_graphs_are_dropped(state, y_net):
    first = corridor.finalize_corridor(state, _start(state, _plan(y_net)).result)
    jobs.take_finished(state, "corridor")
    second_result = _start(state, _plan(y_net, k_per_pair=2)).result
    second = corridor.finalize_corridor(state, second_result)
    assert [state.layers[i].name for i in second.layer_ids] == [
        "Corridor segments (2)", "Corridor shared runs (2)", "Corridor nodes (2)"]
    assert state.layers[second.segments_layer_id].meta["corridor_run"] == 2
    assert set(state.corridor_graphs) == {first.segments_layer_id, second.segments_layer_id}

    state.remove_layer(first.segments_layer_id)
    jobs.take_finished(state, "corridor")
    third = corridor.finalize_corridor(state, second_result)
    assert set(state.corridor_graphs) == {second.segments_layer_id, third.segments_layer_id}
    assert state.layers[third.segments_layer_id].name == "Corridor segments (3)"


def test_a_run_without_corridors_leaves_the_project_alone(state, line_net):
    result = _build(line_net, [(2, 0), (2, 10), (2, 40)], {"max_pair_cost": 1.0})
    before = dict(state.layers)
    state.dirty = False
    outcome = corridor.finalize_corridor(state, result)
    assert state.layers == before and state.corridor_graphs == {} and state.dirty is False
    assert outcome.layer_ids == [] and outcome.segments_layer_id is None
    assert outcome.notices == result.notices and outcome.status_text == "No corridors were built."


def test_the_job_cannot_be_stopped_and_its_status_text_says_so(state, y_net, monkeypatch):
    gate = threading.Event()

    def slow(**kwargs):
        progress = kwargs["progress"]
        progress(1, 4, "reading the search window")
        assert gate.wait(JOIN_S)
        return corridor.CorridorResult(graph=None, summary={"text": "x"})
    monkeypatch.setattr(corridor, "build_corridor", slow)
    job = jobs.start_job(state, corridor.JOB_KIND, corridor.run_job, label=corridor.JOB_LABEL,
                         params=_plan(y_net).params, can_stop=False)
    try:
        assert job is not None and job.can_stop is False
        text = corridor.status_text(job)
        assert text.startswith("Corridor graph...") and "cannot be stopped; you can keep working" in text
        assert text.count("cannot be stopped") == 1                # said once, even after a stage text carried it
        stop = jobs.stop_job(state, corridor.JOB_KIND)
        assert stop.accepted is False and "cannot be stopped" in stop.text
        assert not job.cancel.is_set() and job.status == "running"
        assert jobs.refusal_notice(state, "corridor")["title"].endswith("is still running")
    finally:
        gate.set()
        job.thread.join(JOIN_S)
    assert job.status == "done"
    assert "cannot be stopped" not in corridor.status_text(job) and "finished after" in corridor.status_text(job)


def test_the_job_reads_only_the_snapshot_and_ignores_extra_keys(state, y_net):
    plan = _plan(y_net)
    params = {**plan.params, "raster_layer_id": "r1", "anything": object()}
    job = jobs.start_job(state, corridor.JOB_KIND, corridor.run_job, label=corridor.JOB_LABEL, params=params,
                         can_stop=False)
    job.thread.join(JOIN_S)
    assert job.status == "done" and job.result.summary["n_corridors"] == 3


def test_a_failing_build_ends_the_job_failed_with_a_notice(state, y_net):
    params = {**_plan(y_net).params, "terminals": [list(xy(*Y_A))]}      # one point: build_corridor refuses
    job = jobs.start_job(state, corridor.JOB_KIND, corridor.run_job, label=corridor.JOB_LABEL, params=params,
                         can_stop=False)
    job.thread.join(JOIN_S)
    assert job.status == "failed" and isinstance(job.error, InvalidInputError)
    assert job.error_notice()["severity"] == "error"


# ================================================================ save
def test_save_writes_a_geopackage_with_the_two_layers(tmp_path, y_result):
    import geopandas as gpd

    target = tmp_path / "out" / "corridors.gpkg"                   # the folder does not exist yet
    written = corridor.save_corridor(y_result.graph, target)
    assert written == str(target) and target.is_file()
    layers = set(gpd.list_layers(target)["name"])
    assert layers == {corridor.SEGMENTS_LAYER, corridor.NODES_LAYER} == {"corridor_segments", "corridor_nodes"}
    segments = gpd.read_file(target, layer="corridor_segments")
    nodes = gpd.read_file(target, layer="corridor_nodes")
    assert len(segments) == len(y_result.graph.segments) and len(nodes) == len(y_result.graph.nodes)
    assert float(segments["length_m"].sum()) == pytest.approx(y_result.summary["total_length_m"])
    assert str(segments.crs.to_epsg()) == "25832"
    assert [p.name for p in target.parent.iterdir()] == ["corridors.gpkg"]       # no scratch file left behind


def test_save_never_replaces_a_file_silently(tmp_path, y_result, split_result):
    import geopandas as gpd

    target = tmp_path / "keep.gpkg"
    corridor.save_corridor(y_result.graph, target)
    before = hashlib.sha256(target.read_bytes()).hexdigest()
    with pytest.raises(InvalidInputError, match="already exists"):
        corridor.save_corridor(split_result.graph, target)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == before
    assert [p.name for p in tmp_path.iterdir()] == ["keep.gpkg"]

    corridor.save_corridor(split_result.graph, target, overwrite=True)
    assert len(gpd.read_file(target, layer="corridor_segments")) == 2               # the split graph now
    assert [p.name for p in tmp_path.iterdir()] == ["keep.gpkg"]


def test_save_problems_name_every_refusal(tmp_path, y_result):
    graph = y_result.graph
    existing = tmp_path / "there.gpkg"
    existing.write_bytes(b"x")
    assert corridor.save_problems(graph, tmp_path / "new.gpkg") == []
    assert corridor.save_problems(graph, existing, overwrite=True) == []
    [(key, message)] = corridor.save_problems(graph, existing)
    assert key == "save_path" and "there.gpkg" in message and "already exists" in message
    assert [k for k, _ in corridor.save_problems(None, tmp_path / "new.gpkg")] == ["graph"]
    assert [k for k, _ in corridor.save_problems(graph, "")] == ["save_path"]
    assert [k for k, _ in corridor.save_problems(graph, None)] == ["save_path"]
    assert [k for k, _ in corridor.save_problems(None, "  ")] == ["graph", "save_path"]
    [(key, message)] = corridor.save_problems(graph, tmp_path / "x.shp")
    assert key == "save_path" and ".gpkg" in message
    [(key, message)] = corridor.save_problems(graph, tmp_path)
    assert key == "save_path"
    folder = tmp_path / "folder.gpkg"
    folder.mkdir()
    [(key, message)] = corridor.save_problems(graph, folder, overwrite=True)
    assert "folder" in message
    notice = corridor.problem_notice([("graph", "g")], controls={"graph": "c"})
    assert notice["title"] == "There is no corridor graph to save" and notice["focus_control"] == "c"


def test_save_raises_what_save_problems_reports(tmp_path, y_result):
    with pytest.raises(InvalidInputError, match="no corridor graph"):
        corridor.save_corridor(None, tmp_path / "a.gpkg")
    with pytest.raises(InvalidInputError, match="gpkg"):
        corridor.save_corridor(y_result.graph, tmp_path / "a.txt")
    with pytest.raises(InvalidInputError, match="No file name"):
        corridor.save_corridor(y_result.graph, "")
    assert list(tmp_path.iterdir()) == []


def test_a_failed_write_keeps_the_old_file_and_leaves_no_scratch_file(tmp_path, y_result):
    target = tmp_path / "old.gpkg"
    corridor.save_corridor(y_result.graph, target)
    before = target.read_bytes()

    class Broken:
        def save(self, path):
            Path(path).write_bytes(b"half a file")
            raise RuntimeError("disk full")

    with pytest.raises(RuntimeError, match="disk full"):
        corridor.save_corridor(Broken(), target, overwrite=True)
    assert target.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["old.gpkg"]
