"""Round 14, P8 - the cost-field service (``services/cost_field.py``).

Headless, small synthetic rasters only. Reference values come from an independent calculation: a plain Dijkstra on
the 8-neighbour graph ("r1") written out below from the library's documented edge weight (the mean of the two cell
values for a straight step, and the four cell values of the step with its two corner cells, times sqrt(2) / 4, for
a diagonal one), and ``PathFinder.find_route`` for the larger neighbourhood "r2". Limits are changed by
monkeypatching the module's names, never by building a large raster. GPU variants are skipped without a GPU.
"""
from __future__ import annotations

import ast
import heapq
import math
import re
import threading
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from pyorps import PathFinder
from pyorps.core.exceptions import FeatureNotImplementedError, InvalidInputError
from pyorps.gui.services import cost_field as cf
from pyorps.gui.services import errors, geo, jobs, tiles

CRS = "EPSG:32632"
X0, Y0 = 500000.0, 5000000.0
FORBIDDEN = 65535
JOIN_S = 60


# ------------------------------------------------------------------ helpers
def _write_raster(path: Path, data: np.ndarray, *, dtype: str = "uint16") -> str:
    rows, cols = data.shape
    with rasterio.open(path, "w", driver="GTiff", height=rows, width=cols, count=1, dtype=dtype, crs=CRS,
                       transform=from_origin(X0, Y0 + rows, 1, 1)) as dst:
        dst.write(data.astype(dtype), 1)
    return str(path)


def _xy(rows: int, row: int, col: int) -> tuple[float, float]:
    """Centre of cell (row, col) of a raster written by ``_write_raster``."""
    return X0 + col + 0.5, Y0 + rows - row - 0.5


def _reference(raster: np.ndarray, origin_rc: tuple[int, int]) -> np.ndarray:
    """Independent Dijkstra on the r1 graph: the field of one origin, float64, inf where unreachable."""
    rows, cols = raster.shape
    dist = np.full((rows, cols), np.inf)
    dist[origin_rc] = 0.0
    queue = [(0.0, origin_rc)]
    while queue:
        d, (r, c) = heapq.heappop(queue)
        if d > dist[r, c]:
            continue
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if (dr == 0 and dc == 0) or not (0 <= rr < rows and 0 <= cc < cols):
                    continue
                corners = [(r + dr, c), (r, c + dc)] if dr and dc else []
                cells = [(r, c), (rr, cc), *corners]
                if any(raster[a, b] == FORBIDDEN for a, b in cells):
                    continue
                weight = sum(float(raster[a, b]) for a, b in cells) * math.hypot(dr, dc) / (2 + len(corners))
                if d + weight < dist[rr, cc]:
                    dist[rr, cc] = d + weight
                    heapq.heappush(queue, (d + weight, (rr, cc)))
    return dist


def _read(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def _assert_field(path: str, expected: np.ndarray) -> None:
    """The file holds ``expected`` (float64, inf = unreachable) to float32 precision, nodata -1 where inf."""
    got = _read(path)
    finite = np.isfinite(expected)
    assert np.array_equal(got == cf.NODATA, ~finite)
    np.testing.assert_allclose(got[finite], expected[finite], rtol=2e-5, atol=1e-4)


def _run(path: str, origins, out: Path, **kwargs) -> dict:
    options = {"combine": "sum", "algorithm": "dijkstra", "neighborhood": "r1", "search_buffer_m": 50.0,
               "out_path": out}
    options.update(kwargs)
    return cf.compute_cost_field(path, origins, **options)


def _limit_for(cells: int) -> float:
    """A ``MEMORY_LIMIT_MB`` that fits exactly ``cells`` window cells (half a cell of slack: no float edge)."""
    return (cells + 0.5) * cf.FIELD_BYTES_PER_CELL / 1e6


@pytest.fixture()
def walled(tmp_path):
    """5 x 5 with random costs 1..19 and a forbidden wall in row 2 (columns 1 to 3), as array and file."""
    data = np.random.default_rng(3).integers(1, 20, (5, 5)).astype(np.uint16)
    data[2, 1:4] = FORBIDDEN
    return data, _write_raster(tmp_path / "walled.tif", data)


@pytest.fixture()
def split(tmp_path):
    """6 x 6 uniform cost with a full forbidden column 3: the left and the right half cannot reach each other."""
    data = np.ones((6, 6), dtype=np.uint16)
    data[:, 3] = FORBIDDEN
    return data, _write_raster(tmp_path / "split.tif", data)


def _gpu_available() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0 and cf.routing.backend_available("raster_gpu")
    except Exception:  # noqa: BLE001  # any failure to reach a device means "no GPU" here
        return False


GPU = _gpu_available()


# ================================================================== module rules
def test_the_service_imports_nothing_from_ids_dash_or_the_layout():
    source = Path(cf.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(("." * node.level) + (node.module or ""))
            imported.update(f"{'.' * node.level}{node.module or ''}.{alias.name}" for alias in node.names)
    forbidden = {"dash", "dash_bootstrap_components", "dash_ag_grid", "dash_leaflet"}
    assert not {name for name in imported if name.split(".")[0] in forbidden}
    assert not [name for name in imported if name.endswith("ids") or name.endswith("layout")
                or "callbacks" in name or "sections" in name]
    assert "dashGridOptions" not in source


def test_the_option_lists_and_defaults_agree_with_the_choices():
    assert [o["value"] for o in cf.ORIGIN_OPTIONS] == list(cf.ORIGIN_CHOICES)
    assert [o["value"] for o in cf.COMBINE_OPTIONS] == list(cf.COMBINE_CHOICES)
    assert {o["value"] for o in cf.ALGORITHM_OPTIONS} <= set(cf.ALGORITHM_CHOICES)
    assert set(cf.COST_FIELD_DEFAULTS) == {"origins", "combine", "algorithm", "threads"}
    assert cf.validate_cost_field_settings(dict(cf.COST_FIELD_DEFAULTS)) == []
    assert cf.FIELD_BACKENDS == ("cython", "raster_gpu", "raster_fim")


# ================================================================== settings
def test_normalise_gives_nothing_for_an_old_project_and_every_key_otherwise():
    assert cf.normalise_cost_field_settings({}) == {}
    assert cf.normalise_cost_field_settings(None) == {}
    assert cf.normalise_cost_field_settings("junk") == {}
    kept = cf.normalise_cost_field_settings(
        {"origins": "all", "combine": "min", "algorithm": "dijkstra", "threads": 1, "extra": 1})
    assert kept == {"origins": "all", "combine": "min", "algorithm": "dijkstra", "threads": 1}
    fallback = cf.normalise_cost_field_settings(
        {"origins": "nowhere", "combine": 3, "algorithm": "astar", "threads": 10 ** 6})
    assert fallback == cf.COST_FIELD_DEFAULTS
    assert cf.normalise_cost_field_settings({"combine": "min"})["origins"] == "sources"
    assert cf.normalise_cost_field_settings({"threads": True})["threads"] == 0
    assert cf.normalise_cost_field_settings({"threads": 1.0})["threads"] == 1


def test_every_bad_setting_is_named_with_its_value_range_and_unit():
    problems = dict(cf.validate_cost_field_settings(
        {"origins": "nowhere", "combine": "max", "algorithm": "astar", "threads": 999}))
    assert set(problems) == {"origins", "combine", "algorithm", "threads"}
    assert "Fixed points" in problems["origins"] and "'nowhere'" in problems["origins"]
    assert "Allowed: sources, all" in problems["origins"]
    assert "Combine" in problems["combine"] and "'max'" in problems["combine"] and "sum, min" in problems["combine"]
    assert "Kernel" in problems["algorithm"] and "auto, dijkstra, delta-stepping" in problems["algorithm"]
    limit = cf.max_threads()
    assert "Threads" in problems["threads"] and "999 threads" in problems["threads"]
    assert f"0 to {limit} threads" in problems["threads"]


@pytest.mark.parametrize("value, why", [("abc", "is not a number"), (2.5, "is not a whole number"),
                                        (True, "is not a number"), (float("nan"), "is not a finite number")])
def test_threads_must_be_a_whole_number(value, why):
    ((key, message),) = cf.validate_cost_field_settings({"threads": value})
    assert key == "threads" and why in message and "whole numbers" in message


def test_threads_range_and_blank_values():
    assert cf.validate_cost_field_settings({"threads": cf.max_threads()}) == []
    assert cf.validate_cost_field_settings({"threads": 0}) == []
    assert cf.validate_cost_field_settings({"threads": 3.0}) == [] or cf.max_threads() < 3
    assert [k for k, _m in cf.validate_cost_field_settings({"threads": -1})] == ["threads"]
    assert [k for k, _m in cf.validate_cost_field_settings({"threads": cf.max_threads() + 1})] == ["threads"]
    # blank or missing keys count as their defaults
    assert cf.validate_cost_field_settings({"threads": None, "combine": "", "origins": " "}) == []
    assert cf.validate_cost_field_settings(None) == []


def test_the_thread_limit_is_one_core_less_than_the_machine(monkeypatch):
    monkeypatch.setattr(cf.os, "cpu_count", lambda: 16)
    assert cf.max_threads() == 15
    monkeypatch.setattr(cf.os, "cpu_count", lambda: 1)
    assert cf.max_threads() == 1
    monkeypatch.setattr(cf.os, "cpu_count", lambda: None)
    assert cf.max_threads() == 1


def test_problem_notice_names_the_field_and_lists_the_rest():
    assert cf.problem_notice([]) is None
    problems = cf.validate_cost_field_settings({"combine": "max", "threads": -4})
    notice = cf.problem_notice(problems, controls={"combine": "some-control"})
    assert notice["severity"] == "warning" and notice["title"] == "Combine: value not usable"
    assert notice["meaning"] == problems[0].message and notice["focus_id"] == "tab-routes"
    assert notice["focus_control"] == "some-control"
    assert problems[1].message in notice["details"]
    assert cf.problem_notice(problems[1:])["focus_control"] is None
    assert cf.problem_notice([cf.Problem("memory", "too big")])["severity"] == "error"


def test_a_problem_is_a_two_tuple_like_the_profile_form_problems():
    problem = cf.Problem("combine", "text")
    key, message = problem
    assert (key, message) == ("combine", "text") and isinstance(problem, tuple) and len(problem) == 2


# ================================================================== fixed points
def _point(x, y, name=""):
    """A draft point as the map click makes it (lat/lng from the CRS, x/y in the CRS)."""
    lon, lat = geo.crs_transformer(CRS, "EPSG:4326").transform(x, y)
    return {"lat": lat, "lng": lon, "x": x, "y": y, "name": name}


def test_origins_from_draft_picks_sources_or_every_control_point_without_duplicates():
    draft = {"sources": [_point(500001.5, 5000001.5), _point(500002.5, 5000002.5), _point(500001.5, 5000001.5)],
             "targets": [_point(500010.5, 5000010.5)], "waypoints": [_point(500020.5, 5000020.5)]}
    assert cf.origins_from_draft(draft, "sources") == [(500001.5, 5000001.5), (500002.5, 5000002.5)]
    assert cf.origins_from_draft(draft, "all") == [
        (500001.5, 5000001.5), (500002.5, 5000002.5), (500010.5, 5000010.5), (500020.5, 5000020.5)]
    assert cf.origins_from_draft(draft) == cf.origins_from_draft(draft, "sources")


def test_origins_from_draft_copes_with_empty_and_odd_drafts():
    assert cf.origins_from_draft(None, "sources") == []
    assert cf.origins_from_draft({}, "all") == []
    odd = {"sources": [None, {"name": "x"}, {"x": "a", "y": 1}, {"x": 1, "y": 2}]}
    assert cf.origins_from_draft(odd) == [(1.0, 2.0)]
    with pytest.raises(ValueError, match="sources, all"):
        cf.origins_from_draft({}, "targets")


def test_origins_from_draft_converts_lat_lng_into_the_raster_crs_when_given():
    point = _point(500123.0, 5000456.0)
    stale = {**point, "x": 1.0, "y": 2.0}              # clicked while another raster (CRS) was selected
    assert cf.origins_from_draft({"sources": [stale]}, "sources") == [(1.0, 2.0)]
    (x, y), = cf.origins_from_draft({"sources": [stale]}, "sources", crs=CRS)
    assert x == pytest.approx(500123.0, abs=1e-3) and y == pytest.approx(5000456.0, abs=1e-3)
    # a point without lat/lng keeps its x/y even when a CRS is given
    assert cf.origins_from_draft({"sources": [{"x": 5.0, "y": 6.0}]}, "sources", crs=CRS) == [(5.0, 6.0)]


def test_study_area_geometry_reprojects_the_wgs84_polygon():
    assert cf.study_area_geometry(None, CRS) is None
    corners = [_point(X0 + 1, Y0 + 1), _point(X0 + 4, Y0 + 1), _point(X0 + 4, Y0 + 4), _point(X0 + 1, Y0 + 4)]
    geojson = {"type": "Polygon", "coordinates": [[[p["lng"], p["lat"]] for p in [*corners, corners[0]]]]}
    area = cf.study_area_geometry(geojson, CRS)
    assert area.bounds == pytest.approx((X0 + 1, Y0 + 1, X0 + 4, Y0 + 4), abs=1e-2)
    assert cf.study_area_geometry(geojson, None).bounds[0] == pytest.approx(min(p["lng"] for p in corners))


# ================================================================== backend checks
def test_graph_libraries_and_unknown_backends_are_refused_with_the_allowed_list():
    for name in ("networkx", "networkit", "igraph", "rustworkx"):
        ((key, message),) = cf.check_backend(name)
        assert key == "graph_api" and repr(name) in message and "cython, raster_gpu, raster_fim" in message
    ((_, message),) = cf.check_backend("nonsense")
    assert "not a known backend" in message and "cython" in message
    assert cf.check_backend("cython") == []


def test_a_backend_that_is_not_installed_is_refused_with_the_install_hint(monkeypatch):
    monkeypatch.setattr(cf.routing, "backend_available", lambda name: False)
    ((key, message),) = cf.check_backend("raster_gpu")
    assert key == "graph_api" and "not installed" in message and 'pip install "pyorps[gpu]"' in message


def test_with_a_dem_only_cython_dijkstra_is_allowed(monkeypatch):
    monkeypatch.setattr(cf.routing, "backend_available", lambda name: True)
    for api in ("raster_gpu", "raster_fim"):
        ((key, message),) = cf.check_backend(api, "auto", has_dem=True)
        assert key == "graph_api" and "DEM" in message and "slope" in message
        assert "Allowed with a DEM: cython" in message
    ((key, message),) = cf.check_backend("cython", "delta-stepping", has_dem=True)
    assert key == "algorithm" and "'delta-stepping'" in message and "auto, dijkstra" in message
    assert cf.check_backend("cython", "auto", has_dem=True) == []
    assert cf.check_backend("cython", "dijkstra", has_dem=True) == []
    assert cf.check_backend("cython", "delta-stepping", has_dem=False) == []


# ================================================================== fixed points against the raster
def test_origins_inside_a_passable_cell_are_fine(walled):
    _data, path = walled
    assert cf.check_origins(path, [_xy(5, 0, 0), _xy(5, 4, 4)]) == []


def test_no_origin_or_too_many_is_refused_with_the_range(walled, monkeypatch):
    _data, path = walled
    ((key, message),) = cf.check_origins(path, [])
    assert key == "origins" and "none given" in message and f"1 to {cf.MAX_ORIGINS} points" in message
    monkeypatch.setattr(cf, "MAX_ORIGINS", 2)
    points = [_xy(5, 0, 0), _xy(5, 0, 1), _xy(5, 0, 2)]
    ((key, message),) = cf.check_origins(path, points)
    assert key == "origins" and "3 points" in message and "1 to 2 points" in message


def test_a_point_outside_the_raster_is_named_with_its_coordinates_and_the_bounds(walled):
    _data, path = walled
    ((key, message),) = cf.check_origins(path, [_xy(5, 0, 0), (X0 - 3.0, Y0 + 2.0), (X0 + 99.0, Y0 + 1.0)])
    assert key == "origins" and "2 of 3 points are outside the raster" in message
    assert f"point 2 at ({X0 - 3.0:,.1f}, {Y0 + 2.0:,.1f})" in message and "point 3" in message
    assert f"x from {X0:,.1f} to {X0 + 5:,.1f} m" in message and f"y from {Y0:,.1f} to {Y0 + 5:,.1f} m" in message


def test_a_point_that_is_not_finite_is_refused(walled):
    _data, path = walled
    ((key, message),) = cf.check_origins(path, [_xy(5, 0, 0), (float("nan"), 1.0)])
    assert key == "origins" and "point 2 is not finite" in message


def test_a_point_on_a_forbidden_cell_is_refused_because_the_library_would_move_it(walled):
    _data, path = walled
    ((key, message),) = cf.check_origins(path, [_xy(5, 0, 0), _xy(5, 2, 2)])
    assert key == "origins" and "1 of 2 points stand on a forbidden cell (value 65535)" in message
    assert "point 2 at" in message and "Forbidden cells are barriers" in message
    # with the barrier option off the cell is crossable, so a field may start there
    assert cf.check_origins(path, [_xy(5, 2, 2)], ignore_max_cost=False) == []


def test_a_missing_raster_file_is_a_raster_problem(tmp_path):
    ((key, message),) = cf.check_origins(str(tmp_path / "gone.tif"), [(1.0, 1.0)])
    assert key == "raster" and "no longer exists" in message


# ================================================================== memory estimate and refusal
def test_the_estimate_equals_the_window_the_library_builds(tmp_path):
    data = np.random.default_rng(1).integers(1, 30, (30, 40)).astype(np.uint16)
    path = _write_raster(tmp_path / "window.tif", data)
    points = [_xy(30, 10, 10), _xy(30, 15, 25)]
    for options in ({"buffer_m": 3.2}, {"buffer_m": 7.3}, {"full_window": True}, {"buffer_m": None}):
        est = cf.estimate(path, points, **options)
        finder = PathFinder(path, source_coords=points, target_coords=points, search_space_buffer_m=est["buffer_m"])
        finder.create_raster_handler()
        rows, cols = finder.raster_handler.data.shape[-2:]
        assert (est["rows"], est["cols"], est["cells"]) == (rows, cols, rows * cols), options
        assert est["mb"] == pytest.approx(est["cells"] * cf.FIELD_BYTES_PER_CELL / 1e6)
        assert est["raster_cells"] == 30 * 40 and est["resolution_m"] == 1.0
    assert cf.estimate(path, points, buffer_m=3.2)["cells"] < cf.estimate(path, points, buffer_m=7.3)["cells"]
    assert cf.estimate(path, points, full_window=True)["cells"] == 30 * 40


def test_the_estimate_never_under_counts_when_the_window_edge_falls_on_a_cell_edge(tmp_path):
    """Points at cell centres and a buffer of x.5 m put the window edge exactly on a cell edge, where the
    library's floating point bound may fall one cell short: the estimate is then at most one row and column
    larger, never smaller."""
    data = np.random.default_rng(1).integers(1, 30, (30, 40)).astype(np.uint16)
    path = _write_raster(tmp_path / "edge.tif", data)
    points = [_xy(30, 10, 10), _xy(30, 15, 25)]
    est = cf.estimate(path, points, buffer_m=7.5)
    finder = PathFinder(path, source_coords=points, target_coords=points, search_space_buffer_m=7.5)
    finder.create_raster_handler()
    rows, cols = finder.raster_handler.data.shape[-2:]
    assert est["rows"] - rows in (0, 1) and est["cols"] - cols in (0, 1)


def test_the_buffer_is_automatic_for_blank_or_zero_and_the_whole_raster_on_request(tmp_path):
    path = _write_raster(tmp_path / "b.tif", np.ones((10, 10), dtype=np.uint16))
    one = [_xy(10, 5, 5)]
    assert cf.resolve_buffer_m(path, one, None) == 1000.0
    assert cf.resolve_buffer_m(path, one, 0) == 1000.0
    assert cf.resolve_buffer_m(path, one, "") == 1000.0
    assert cf.resolve_buffer_m(path, one, 12.5) == 12.5
    two = [(0.0, 0.0), (3000.0, 4000.0)]
    assert cf.resolve_buffer_m(path, two, None) == 7500.0            # 1.5 x the 5000 m distance
    assert cf.resolve_buffer_m(path, one, 12.5, full_window=True) == pytest.approx(math.hypot(10, 10))


def test_the_verdict_follows_the_two_memory_names(tmp_path, monkeypatch):
    path = _write_raster(tmp_path / "v.tif", np.ones((40, 60), dtype=np.uint16))
    points = [_xy(40, 20, 30)]
    est = cf.estimate(path, points, full_window=True)
    assert est["verdict"] == "ok" and est["limit_mb"] == cf.MEMORY_LIMIT_MB and est["warn_mb"] == cf.MEMORY_WARN_MB
    monkeypatch.setattr(cf, "MEMORY_WARN_MB", est["mb"] / 2)
    assert cf.estimate(path, points, full_window=True)["verdict"] == "warn"
    monkeypatch.setattr(cf, "MEMORY_LIMIT_MB", est["mb"] / 2)
    refused = cf.estimate(path, points, full_window=True)
    assert refused["verdict"] == "refuse"
    assert refused["max_cells"] == int(est["mb"] / 2 * 1e6 // cf.FIELD_BYTES_PER_CELL)


def test_the_gpu_backends_also_have_a_device_limit(tmp_path, monkeypatch):
    path = _write_raster(tmp_path / "g.tif", np.ones((40, 60), dtype=np.uint16))
    points = [_xy(40, 20, 30)]
    assert "device_mb" not in cf.estimate(path, points, full_window=True, graph_api="cython")
    est = cf.estimate(path, points, full_window=True, graph_api="raster_gpu")
    assert est["device_mb"] == pytest.approx(2400 * cf.GPU_BYTES_PER_CELL / 1e6) and est["verdict"] == "ok"
    monkeypatch.setattr(cf, "GPU_MEMORY_LIMIT_MB", est["device_mb"] / 2)
    over = cf.estimate(path, points, full_window=True, graph_api="raster_gpu")
    assert over["verdict"] == "refuse"
    ((key, message),) = cf.memory_problems(over, path, points, full_window=True)
    assert key == "memory" and "graphics card" in message and f"{over['device_limit_mb']:.3g} MB" in message
    assert f"{over['device_mb']:.3g} MB" in message and f"{cf.GPU_BYTES_PER_CELL} bytes per cell" in message
    assert cf.estimate(path, points, full_window=True, graph_api="cython")["verdict"] == "ok"


def test_a_refusal_gives_the_cells_the_megabytes_the_limit_and_a_buffer_that_fits(tmp_path, monkeypatch):
    path = _write_raster(tmp_path / "m.tif", np.ones((200, 200), dtype=np.uint16))
    points = [_xy(200, 100, 90), _xy(200, 100, 110)]
    est = cf.estimate(path, points, buffer_m=60.0)
    limit_mb = _limit_for(900)                                         # room for 900 cells
    monkeypatch.setattr(cf, "MEMORY_LIMIT_MB", limit_mb)
    refused = cf.estimate(path, points, buffer_m=60.0)
    assert refused["verdict"] == "refuse" and refused["max_cells"] == 900
    ((key, message),) = cf.memory_problems(refused, path, points)
    assert key == "memory"
    assert f"{est['rows']:,} x {est['cols']:,} = {est['cells']:,} cells" in message
    assert f"{est['mb']:.3g} MB" in message and f"{limit_mb:.3g} MB" in message and "about 900 cells" in message
    assert f"{cf.FIELD_BYTES_PER_CELL} bytes per cell" in message
    suggested = float(re.search(r"at most ([\d,]+) m", message).group(1).replace(",", ""))
    assert 0 < suggested < 60
    assert cf.estimate(path, points, buffer_m=suggested)["cells"] <= 900          # the advice fits ...
    assert cf.estimate(path, points, buffer_m=suggested + 3.0)["cells"] > 900     # ... and is nearly the largest


def test_a_refusal_for_the_whole_raster_says_to_switch_the_option_off(tmp_path, monkeypatch):
    path = _write_raster(tmp_path / "w.tif", np.ones((100, 100), dtype=np.uint16))
    points = [_xy(100, 50, 45), _xy(100, 50, 55)]
    monkeypatch.setattr(cf, "MEMORY_LIMIT_MB", _limit_for(2000))
    est = cf.estimate(path, points, full_window=True)
    ((_, message),) = cf.memory_problems(est, path, points, full_window=True)
    assert "Switch off the whole-raster option" in message and "search buffer to at most" in message


def test_when_even_the_points_do_not_fit_the_advice_is_a_coarser_raster(tmp_path, monkeypatch):
    path = _write_raster(tmp_path / "c.tif", np.ones((100, 100), dtype=np.uint16))
    points = [_xy(100, 10, 10), _xy(100, 90, 90)]
    monkeypatch.setattr(cf, "MEMORY_LIMIT_MB", _limit_for(100))
    est = cf.estimate(path, points, buffer_m=5.0)
    ((_, message),) = cf.memory_problems(est, path, points)
    assert "coarser cost raster" in message and "fewer, closer fixed points" in message
    expected = math.sqrt(est["cells"] / 100)
    assert f"at least {expected:,.1f} m per cell" in message


def test_ok_and_warn_have_no_memory_problem(tmp_path, monkeypatch):
    path = _write_raster(tmp_path / "o.tif", np.ones((20, 20), dtype=np.uint16))
    points = [_xy(20, 10, 10)]
    assert cf.memory_problems(cf.estimate(path, points, full_window=True), path, points) == []
    monkeypatch.setattr(cf, "MEMORY_WARN_MB", 0.0001)
    est = cf.estimate(path, points, full_window=True)
    assert est["verdict"] == "warn" and cf.memory_problems(est, path, points) == []


# ================================================================== the plan of a run
def _plan(path, settings=None, origins=None, **kwargs):
    options = {"graph_api": "cython", "neighborhood": "r1", "search_buffer_m": 50.0, "out_path": "unused.tif"}
    options.update(kwargs)
    return cf.plan_run(path, settings or {}, origins if origins is not None else [_xy(5, 0, 0)], **options)


def test_a_good_plan_carries_the_arguments_of_the_computation(walled):
    _data, path = walled
    plan = _plan(path, {"combine": "min", "algorithm": "dijkstra", "threads": 1},
                 [_xy(5, 0, 0), _xy(5, 4, 4)], neighborhood=3, out_path="out.tif")
    assert plan.ok and plan.problems == [] and plan.estimate["verdict"] == "ok"
    assert plan.kwargs == {
        "raster_path": path, "origins": [_xy(5, 0, 0), _xy(5, 4, 4)], "combine": "min", "algorithm": "dijkstra",
        "num_threads": 1, "neighborhood": "r3", "ignore_max_cost": True, "search_buffer_m": 50.0,
        "graph_api": "cython", "out_path": "out.tif", "study_area": None, "finder_options": None}
    assert plan.settings["origins"] == "sources"


def test_the_plan_resolves_an_automatic_buffer_and_the_whole_raster(walled):
    _data, path = walled
    assert _plan(path, search_buffer_m=None).kwargs["search_buffer_m"] == 1000.0
    assert _plan(path, search_buffer_m=0).kwargs["search_buffer_m"] == 1000.0
    assert _plan(path, full_window=True).kwargs["search_buffer_m"] == pytest.approx(math.hypot(5, 5))


def test_a_plan_with_problems_lists_all_of_them_and_has_no_kwargs(walled):
    _data, path = walled
    plan = _plan(path, {"combine": "max", "threads": 999}, [_xy(5, 0, 0), (1.0, 1.0)], graph_api="networkx",
                 search_buffer_m=-5, neighborhood="rubbish")
    keys = [k for k, _m in plan.problems]
    assert not plan.ok and plan.kwargs is None
    assert set(keys) == {"combine", "threads", "graph_api", "search_buffer_m", "neighborhood", "origins"}
    assert len(keys) == len(set(keys))
    buffer_message = dict(plan.problems)["search_buffer_m"]
    assert "-5 m" in buffer_message and "0 (automatic) or a positive number of metres" in buffer_message
    assert "Allowed: whole numbers from 0 to 127" in dict(plan.problems)["neighborhood"]


def test_a_bad_origins_setting_is_a_problem_even_with_good_points(walled):
    _data, path = walled
    plan = _plan(path, {"origins": "nowhere"})
    assert [k for k, _m in plan.problems] == ["origins"]
    assert "'nowhere'" in plan.problems[0].message and "sources, all" in plan.problems[0].message


def test_the_plan_refuses_a_raster_that_is_not_uint16(tmp_path):
    path = _write_raster(tmp_path / "f.tif", np.ones((5, 5), dtype=np.float32), dtype="float32")
    plan = _plan(path)
    ((key, message),) = plan.problems
    assert key == "raster" and "float32" in message and "uint16" in message
    assert _plan(path, finder_options={"objective": {"cost": 1.0}}).ok      # the objective pipeline builds its own


def test_the_plan_refuses_a_run_over_the_memory_limit_and_warns_below_it(walled, monkeypatch):
    _data, path = walled
    monkeypatch.setattr(cf, "MEMORY_LIMIT_MB", _limit_for(10))
    plan = _plan(path, full_window=True)
    assert [k for k, _m in plan.problems] == ["memory"] and plan.kwargs is None
    assert plan.estimate["verdict"] == "refuse"
    monkeypatch.setattr(cf, "MEMORY_LIMIT_MB", 4000.0)
    monkeypatch.setattr(cf, "MEMORY_WARN_MB", _limit_for(10))
    plan = _plan(path, full_window=True)
    assert plan.ok and [n["title"] for n in plan.notices] == ["This cost field needs a lot of memory"]
    assert plan.notices[0]["severity"] == "warning" and "25 cells" in plan.notices[0]["meaning"]


def test_the_plan_names_the_terrain_model_kernel_and_the_continuous_fim_field(walled, monkeypatch):
    _data, path = walled
    dem = {"dem": "some_dem.tif", "objective": {"cost": 1.0}}
    assert [n["title"] for n in _plan(path, finder_options=dem).notices] == [
        "With a terrain model the cost field uses the Dijkstra solver"]
    assert _plan(path, {"algorithm": "dijkstra"}, finder_options=dem).notices == []
    monkeypatch.setattr(cf.routing, "backend_available", lambda name: True)
    plan = _plan(path, graph_api="raster_fim")
    assert plan.ok and [n["title"] for n in plan.notices] == ["raster_fim computes a continuous field"]


def test_the_plan_refuses_a_dem_with_the_gpu_or_an_explicit_delta_stepping(walled, monkeypatch):
    _data, path = walled
    monkeypatch.setattr(cf.routing, "backend_available", lambda name: True)
    dem = {"dem": "some_dem.tif", "objective": {"cost": 1.0}}
    assert [k for k, _m in _plan(path, graph_api="raster_gpu", finder_options=dem).problems] == ["graph_api"]
    plan = _plan(path, {"algorithm": "delta-stepping"}, finder_options=dem)
    assert [k for k, _m in plan.problems] == ["algorithm"]
    assert _plan(path, {"algorithm": "delta-stepping"}).ok


# ================================================================== the field itself
@pytest.mark.parametrize("algorithm", ["dijkstra", "delta-stepping", "auto"])
def test_one_origin_equals_an_independent_dijkstra(walled, tmp_path, algorithm):
    data, path = walled
    out = tmp_path / "one.tif"
    result = _run(path, [_xy(5, 0, 0)], out, algorithm=algorithm, num_threads=1)
    _assert_field(str(out), _reference(data, (0, 0)))
    assert result["path"] == str(out) and result["n_origins"] == 1
    expected_kernel = "dijkstra" if algorithm == "dijkstra" else "delta-stepping"
    assert result["algorithm"] == expected_kernel and result["algorithm_requested"] == algorithm


def test_the_field_at_a_cell_equals_find_route_to_that_cell_on_the_r2_neighbourhood(tmp_path):
    data = np.random.default_rng(5).integers(1, 40, (14, 14)).astype(np.uint16)
    data[5:9, 7] = FORBIDDEN
    path = _write_raster(tmp_path / "r2.tif", data)
    origin = _xy(14, 2, 2)
    out = tmp_path / "r2_field.tif"
    _run(path, [origin], out, neighborhood="r2", algorithm="dijkstra")
    field = _read(str(out))
    for row, col in [(0, 13), (13, 0), (13, 13), (7, 3), (7, 11), (10, 10)]:
        site = _xy(14, row, col)
        finder = PathFinder(path, source_coords=site, target_coords=origin, search_space_buffer_m=50.0,
                            neighborhood_str="r2")
        route = finder.find_route(algorithm="dijkstra", calculate_metrics=True)
        assert field[row, col] == pytest.approx(route.total_cost, rel=2e-5), (row, col)


def test_sum_and_min_of_two_origins_equal_the_independent_calculation(walled, tmp_path):
    data, path = walled
    a, b = (0, 0), (4, 4)
    points = [_xy(5, *a), _xy(5, *b)]
    ref_a, ref_b = _reference(data, a), _reference(data, b)
    summed = _run(path, points, tmp_path / "sum.tif", combine="sum")
    _assert_field(summed["path"], ref_a + ref_b)
    lowest = _run(path, points, tmp_path / "min.tif", combine="min")
    _assert_field(lowest["path"], np.minimum(ref_a, ref_b))
    assert summed["combine"] == "sum" and lowest["combine"] == "min"
    assert lowest["min"] == 0.0 and summed["min"] == pytest.approx(np.min(ref_a + ref_b), rel=2e-5)


def test_three_origins_sum_in_float32_to_the_reference(tmp_path):
    data = np.random.default_rng(11).integers(1, 30, (12, 12)).astype(np.uint16)
    path = _write_raster(tmp_path / "three.tif", data)
    cells = [(1, 1), (10, 3), (6, 10)]
    result = _run(path, [_xy(12, *c) for c in cells], tmp_path / "three_out.tif")
    _assert_field(result["path"], sum(_reference(data, c) for c in cells))


def test_the_result_holds_what_the_poll_side_needs(walled, tmp_path):
    data, path = walled
    out = tmp_path / "sub" / "dir" / "res.tif"                       # the parent folders are created
    ref = _reference(data, (0, 0)) + _reference(data, (4, 4))
    result = _run(path, [_xy(5, 0, 0), _xy(5, 4, 4)], out)
    finite = np.isfinite(ref)
    assert out.is_file() and not Path(str(out) + ".part").exists()
    assert result["min"] == pytest.approx(ref[finite].min(), rel=2e-5)
    assert result["max"] == pytest.approx(ref[finite].max(), rel=2e-5)
    assert result["unit"] == "cost units" and "cell value x metres" in result["unit_note"]
    assert result["nodata"] == -1.0 and result["graph_api"] == "cython" and result["neighborhood"] == "r1"
    assert result["origins"] == [list(_xy(5, 0, 0)), list(_xy(5, 4, 4))]
    assert (result["rows"], result["cols"], result["cells"]) == (5, 5, 25)
    assert result["valid_cells"] == int(finite.sum()) == 22 and result["nodata_cells"] == 3
    assert result["forbidden_cells"] == 3 and result["unreachable_cells"] == 0
    assert result["resolution_m"] == 1.0 and result["bounds"] == [X0, Y0, X0 + 5, Y0 + 5]
    assert result["crs"] == CRS and result["estimate"]["cells"] == 25 and result["threads_requested"] == 0
    assert isinstance(result["notices"], list)
    assert all(isinstance(v, (str, int, float, list, dict, type(None))) for v in result.values())


def test_the_file_is_a_georeferenced_float32_geotiff_with_nodata_minus_one(walled, tmp_path):
    data, path = walled
    result = _run(path, [_xy(5, 0, 0), _xy(5, 4, 4)], tmp_path / "geo.tif")
    with rasterio.open(result["path"]) as src:
        assert src.nodata == -1.0 and src.dtypes[0] == "float32" and src.count == 1
        assert src.crs.to_string() == CRS and src.transform == from_origin(X0, Y0 + 5, 1, 1)
        values = src.read(1)
    assert np.isfinite(values).all()
    # forbidden cells are nodata, not a huge number that would flatten the colour scale
    assert (values[data == FORBIDDEN] == -1.0).all()
    assert values[values != -1.0].min() == pytest.approx(result["min"]) and values.max() == pytest.approx(result["max"])


def test_a_window_smaller_than_the_raster_has_its_own_transform(tmp_path):
    data = np.random.default_rng(2).integers(1, 20, (30, 40)).astype(np.uint16)
    path = _write_raster(tmp_path / "win.tif", data)
    origin = _xy(30, 15, 20)
    result = _run(path, [origin], tmp_path / "win_out.tif", search_buffer_m=5.0)
    assert 0 < result["cells"] < 30 * 40 and result["cells"] == result["estimate"]["cells"]
    with rasterio.open(result["path"]) as src:
        assert (src.height, src.width) == (result["rows"], result["cols"])
        left, bottom, right, top = src.bounds
        assert left <= origin[0] <= right and bottom <= origin[1] <= top
        at_origin = src.read(1)[src.index(*origin)]
    assert at_origin == 0.0
    # the window's corners lie outside the buffered area around the point: masked, so nodata, never a made-up cost
    assert result["forbidden_cells"] > 0 and result["valid_cells"] < result["cells"]


def test_unreachable_cells_are_nodata_and_reported(split, tmp_path):
    data, path = split
    result = _run(path, [_xy(6, 1, 0)], tmp_path / "one_side.tif")
    ref = _reference(data, (1, 0))
    _assert_field(result["path"], ref)
    assert result["valid_cells"] == 18 and result["forbidden_cells"] == 6 and result["unreachable_cells"] == 12
    assert [n["title"] for n in result["notices"]] == ["Some cells cannot be reached"]
    assert "12 of 36" in result["notices"][0]["meaning"]
    assert result["max"] < 100          # the cut-off half is not in the colour range


def test_origins_in_separate_halves_leave_nothing_for_the_sum(split, tmp_path):
    data, path = split
    points = [_xy(6, 1, 0), _xy(6, 1, 5)]
    summed = _run(path, points, tmp_path / "none.tif", combine="sum")
    assert summed["valid_cells"] == 0 and summed["min"] == 0.0 and summed["max"] == 0.0
    assert (_read(summed["path"]) == -1.0).all()
    assert summed["best"] is None and "No cell is reachable from every fixed point" in summed["best_note"]
    assert [n["title"] for n in summed["notices"]] == ["The cost field has no reachable cell"]
    lowest = _run(path, points, tmp_path / "lowest.tif", combine="min")
    _assert_field(lowest["path"], np.minimum(_reference(data, (1, 0)), _reference(data, (1, 5))))
    assert lowest["valid_cells"] == 30 and lowest["unreachable_cells"] == 0


def test_forbidden_cells_are_nodata_even_when_they_may_be_crossed(tmp_path):
    data = np.ones((7, 7), dtype=np.uint16)
    data[3, 3] = FORBIDDEN
    path = _write_raster(tmp_path / "cross.tif", data)
    plan = cf.plan_run(path, {}, [_xy(7, 0, 0)], graph_api="cython", neighborhood="r1", search_buffer_m=50.0,
                       ignore_max_cost=False, out_path=tmp_path / "cross_out.tif")
    assert plan.ok and plan.kwargs["ignore_max_cost"] is False
    result = cf.compute_cost_field(**plan.kwargs)
    values = _read(result["path"])
    assert values[3, 3] == -1.0 and result["forbidden_cells"] == 1       # finite in the library, nodata here
    assert (values[values != -1.0] >= 0).all() and result["valid_cells"] == 48
    assert np.isfinite(values).all() and values.max() < 1000        # no 65535-per-metre values in the scale


def test_the_origin_cell_costs_nothing_and_neighbours_cost_their_edge(tmp_path):
    path = _write_raster(tmp_path / "flat.tif", np.full((5, 5), 4, dtype=np.uint16))
    result = _run(path, [_xy(5, 2, 2)], tmp_path / "flat_out.tif")
    values = _read(result["path"])
    assert values[2, 2] == 0.0
    assert values[2, 3] == pytest.approx(4.0)                         # (4 + 4) / 2 x 1 m
    assert values[3, 3] == pytest.approx(4.0 * math.sqrt(2.0))        # diagonal: sqrt(2) x 4


# ================================================================== the cheapest cell
def test_the_best_cell_is_the_cheapest_cell_of_the_sum(walled, tmp_path):
    data, path = walled
    points = [(0, 0), (4, 4), (4, 0)]
    result = _run(path, [_xy(5, *c) for c in points], tmp_path / "best.tif")
    ref = sum(_reference(data, c) for c in points)
    best = result["best"]
    assert best["cost"] == pytest.approx(np.nanmin(np.where(np.isfinite(ref), ref, np.nan)), rel=2e-5)
    assert ref[best["row"], best["col"]] == pytest.approx(best["cost"], rel=2e-5)      # a cell of that cost
    assert (best["x"], best["y"]) == _xy(5, best["row"], best["col"])
    assert result["best_note"] == ""
    assert cf.best_text(result) == (
        f"Cheapest cell: ({best['x']:,.1f}, {best['y']:,.1f}), total {best['cost']:,.0f} cost units")


def test_a_drawn_study_area_limits_the_best_cell(tmp_path):
    data = np.ones((9, 9), dtype=np.uint16)
    path = _write_raster(tmp_path / "area.tif", data)
    points = [_xy(9, 4, 0), _xy(9, 4, 8)]                  # the cheapest cells lie on the straight line between them
    free = _run(path, points, tmp_path / "free.tif")
    assert free["best"]["cost"] == pytest.approx(8.0)
    area = box(X0, Y0 + 5, X0 + 9, Y0 + 9)                 # only the upper rows 0..3 of the raster
    limited = _run(path, points, tmp_path / "limited.tif", study_area=area)
    assert limited["best"]["row"] <= 3 and Y0 + 5 <= limited["best"]["y"] <= Y0 + 9
    ref = _reference(data, (4, 0)) + _reference(data, (4, 8))
    assert limited["best"]["cost"] == pytest.approx(ref[:4].min(), rel=2e-5)
    assert limited["best"]["cost"] > free["best"]["cost"]
    # the layer itself is not clipped to the study area
    assert limited["valid_cells"] == free["valid_cells"] == 81


def _geojson_polygon(x0, y0, x1, y1):
    """A WGS84 GeoJSON polygon (what ``state.study_area`` holds) around a rectangle given in the raster CRS."""
    corners = [_point(x0, y0), _point(x1, y0), _point(x1, y1), _point(x0, y1)]
    return {"type": "Polygon", "coordinates": [[[p["lng"], p["lat"]] for p in [*corners, corners[0]]]]}


def test_the_study_area_may_be_given_as_the_wgs84_geojson_of_the_project_state(tmp_path):
    path = _write_raster(tmp_path / "geojson_area.tif", np.ones((9, 9), dtype=np.uint16))
    points = [_xy(9, 4, 0), _xy(9, 4, 8)]
    geojson = _geojson_polygon(X0 + 0.2, Y0 + 5.2, X0 + 8.8, Y0 + 8.8)       # the cell centres of rows 0 to 3
    plan = _plan(path, {}, points, study_area=geojson, out_path=tmp_path / "geojson_out.tif")
    assert plan.ok and plan.kwargs["study_area"].bounds == pytest.approx((X0 + 0.2, Y0 + 5.2, X0 + 8.8, Y0 + 8.8),
                                                                         abs=1e-2)
    via_plan = cf.compute_cost_field(**plan.kwargs)
    direct = _run(path, points, tmp_path / "geojson_direct.tif", study_area=geojson)     # the worker converts too
    for result in (via_plan, direct):
        assert result["best"]["row"] <= 3 and result["best"]["cost"] > 8.0
    assert _plan(path, {}, points).kwargs["study_area"] is None


def test_a_fresh_output_name_lies_in_the_work_folder_and_is_not_created(tmp_path):
    one, two = cf.new_out_path(tmp_path), cf.new_out_path(tmp_path)
    assert one != two and one.parent == tmp_path and one.suffix == ".tif" and not one.exists()


def test_a_study_area_without_reachable_cells_gives_no_best_cell_and_a_notice(tmp_path):
    path = _write_raster(tmp_path / "nowhere.tif", np.ones((9, 9), dtype=np.uint16))
    points = [_xy(9, 4, 0), _xy(9, 4, 8)]
    far = box(X0 + 100, Y0 + 100, X0 + 120, Y0 + 120)
    result = _run(path, points, tmp_path / "nowhere_out.tif", study_area=far)
    assert result["best"] is None and "inside the study area" in result["best_note"]
    assert "No cheapest cell inside the study area" in [n["title"] for n in result["notices"]]
    assert result["valid_cells"] == 81


def test_a_single_origin_or_the_lowest_of_the_points_names_no_cheapest_cell(walled, tmp_path):
    _data, path = walled
    single = _run(path, [_xy(5, 0, 0)], tmp_path / "s.tif")
    assert single["best"] is None and "a single fixed point" in single["best_note"]
    lowest = _run(path, [_xy(5, 0, 0), _xy(5, 4, 4)], tmp_path / "l.tif", combine="min")
    assert lowest["best"] is None and "lowest of the points" in lowest["best_note"]
    assert cf.best_text(lowest) == lowest["best_note"] and "Use 'Sum over the points'" in cf.best_text(lowest)
    assert cf.best_text({}) == "No cheapest cell." and cf.best_text(None) == "No cheapest cell."


def test_the_summary_line_names_points_kernel_cells_and_range(walled, tmp_path):
    _data, path = walled
    result = _run(path, [_xy(5, 0, 0), _xy(5, 4, 4)], tmp_path / "sum_line.tif")
    text = cf.summary_text(result)
    assert "sum over 2 fixed point(s)" in text and "kernel dijkstra" in text and "22 of 25 cells" in text
    assert text.endswith("cost units.")
    assert "lowest of 2" in cf.summary_text({**result, "combine": "min"})


# ================================================================== progress and stop
def test_progress_is_reported_at_every_step_with_a_stage_text(walled, tmp_path):
    _data, path = walled
    calls = []
    _run(path, [_xy(5, 0, 0), _xy(5, 4, 4)], tmp_path / "p.tif", progress=lambda *a: calls.append(a))
    assert [(d, t) for d, t, _s in calls] == [(0, 3), (0, 3), (1, 3), (2, 3)]
    stages = [s for _d, _t, s in calls]
    assert stages[0] == "opening the raster window"
    assert "point 1 of 2" in stages[1] and "point 2 of 2" in stages[2]
    assert all("Stop acts between points" in s for s in stages[1:3])
    assert stages[3] == "writing the GeoTIFF"


def _count_fields(monkeypatch, after_first=None):
    """Wrap ``PathFinder.cost_field`` to count the solves (and run ``after_first`` once the first one is open)."""
    calls = []
    original = PathFinder.cost_field

    def wrapper(self, origin, algorithm="auto", **kwargs):
        calls.append((origin, algorithm, kwargs))
        field = original(self, origin, algorithm, **kwargs)
        if len(calls) == 1 and after_first is not None:
            after_first()
        return field

    monkeypatch.setattr(PathFinder, "cost_field", wrapper)
    return calls


def test_stop_before_the_start_does_nothing_and_writes_no_file(walled, tmp_path):
    _data, path = walled
    stop = threading.Event()
    stop.set()
    out = tmp_path / "stopped.tif"
    with pytest.raises(jobs.JobCancelled):
        _run(path, [_xy(5, 0, 0)], out, cancel=stop)
    assert not out.exists() and not Path(str(out) + ".part").exists()


def test_stop_between_origins_ends_after_the_point_in_flight_and_writes_nothing(walled, tmp_path, monkeypatch):
    _data, path = walled
    stop = threading.Event()
    calls = _count_fields(monkeypatch, after_first=stop.set)         # Stop pressed while point 1 is being solved
    out = tmp_path / "between.tif"
    with pytest.raises(jobs.JobCancelled):
        _run(path, [_xy(5, 0, 0), _xy(5, 4, 4), _xy(5, 4, 0)], out, cancel=stop)
    assert len(calls) == 1                           # the point in flight finished, the next one never started
    assert not out.exists() and not Path(str(out) + ".part").exists()


def test_stop_during_the_last_point_still_writes_nothing(walled, tmp_path, monkeypatch):
    _data, path = walled
    stop = threading.Event()
    _count_fields(monkeypatch, after_first=stop.set)
    out = tmp_path / "last.tif"
    with pytest.raises(jobs.JobCancelled):
        _run(path, [_xy(5, 0, 0)], out, cancel=stop)
    assert not out.exists()


def test_cancel_may_be_a_callable(walled, tmp_path):
    _data, path = walled
    with pytest.raises(jobs.JobCancelled):
        _run(path, [_xy(5, 0, 0)], tmp_path / "callable.tif", cancel=lambda: True)
    assert _run(path, [_xy(5, 0, 0)], tmp_path / "callable_ok.tif", cancel=lambda: False)["valid_cells"] > 0


def test_thread_count_is_passed_only_to_the_cpu_kernel_when_given(walled, tmp_path, monkeypatch):
    _data, path = walled
    calls = _count_fields(monkeypatch)
    _run(path, [_xy(5, 0, 0)], tmp_path / "t0.tif", algorithm="auto", num_threads=0)
    _run(path, [_xy(5, 0, 0)], tmp_path / "t1.tif", algorithm="auto", num_threads=1)
    assert calls[0][2] == {} and calls[1][2] == {"num_threads": 1}
    assert calls[0][1] == "auto" and calls[1][1] == "auto"


def test_threads_that_cannot_be_used_are_reported_as_a_notice(walled, tmp_path):
    _data, path = walled
    ignored = _run(path, [_xy(5, 0, 0)], tmp_path / "ign.tif", algorithm="dijkstra", num_threads=1)
    notice = [n for n in ignored["notices"] if n["title"] == "The thread count was not used"]
    assert len(notice) == 1 and "Threads were set to 1" in notice[0]["meaning"]
    used = _run(path, [_xy(5, 0, 0)], tmp_path / "used.tif", algorithm="delta-stepping", num_threads=1)
    assert not [n for n in used["notices"] if n["title"] == "The thread count was not used"]


# ================================================================== refusals inside the computation
def test_the_computation_repeats_every_check_before_it_allocates_anything(walled, tmp_path, monkeypatch):
    _data, path = walled

    def boom(*_a, **_k):
        raise AssertionError("the window was loaded although the run must be refused")

    monkeypatch.setattr(cf, "_finder_for", boom)
    monkeypatch.setattr(cf, "MEMORY_LIMIT_MB", _limit_for(5))
    out = tmp_path / "refused.tif"
    with pytest.raises(cf.CostFieldRefused) as raised:
        _run(path, [_xy(5, 0, 0)], out, full_window=True)
    ((key, message),) = raised.value.problems
    assert key == "memory" and "25 cells" in message and "about 5 cells" in message
    assert str(raised.value) == message and isinstance(raised.value, InvalidInputError)
    assert not out.exists()
    monkeypatch.setattr(cf, "MEMORY_LIMIT_MB", 4000.0)
    for kwargs, expected in (({"graph_api": "networkx"}, "graph_api"), ({"combine": "max"}, "combine"),
                             ({"num_threads": 999}, "threads")):
        with pytest.raises(cf.CostFieldRefused) as raised:
            _run(path, [_xy(5, 0, 0)], out, **kwargs)
        assert [k for k, _m in raised.value.problems] == [expected]
    with pytest.raises(cf.CostFieldRefused, match="forbidden cell"):
        _run(path, [_xy(5, 2, 2)], out)
    with pytest.raises(cf.CostFieldRefused, match="outside the raster"):
        _run(path, [(X0 - 5, Y0)], out)
    with pytest.raises(cf.CostFieldRefused, match="none given"):
        _run(path, [], out)


def test_the_computation_needs_a_file_to_write(walled):
    _data, path = walled
    with pytest.raises(InvalidInputError, match="out_path"):
        cf.compute_cost_field(path, [_xy(5, 0, 0)], neighborhood="r1", search_buffer_m=50.0)


@pytest.mark.parametrize("exc, key, fragment", [
    (InvalidInputError("CostField origin (1, 2) is outside the search window; widen it"), "origins",
     "point 2 lies outside the search window"),
    (InvalidInputError("origin (1, 2) snapped to an impassable cell; a root the exclude mask"), "origins",
     "point 2 snapped to a forbidden cell"),
    (FeatureNotImplementedError("the delta full-field kernel does not apply the DEM's per-edge slope terms"),
     "algorithm", "does not apply the slope terms"),
    (InvalidInputError("CostField needs symmetric edge weights: the step set"), "graph_api", "not symmetric"),
])
def test_library_refusals_become_field_level_problems(walled, tmp_path, monkeypatch, exc, key, fragment):
    _data, path = walled
    calls = []
    original = PathFinder.cost_field

    def wrapper(self, origin, algorithm="auto", **kwargs):
        calls.append(origin)
        if len(calls) == 2:
            raise exc
        return original(self, origin, algorithm, **kwargs)

    monkeypatch.setattr(PathFinder, "cost_field", wrapper)
    with pytest.raises(cf.CostFieldRefused) as raised:
        _run(path, [_xy(5, 0, 0), _xy(5, 4, 4)], tmp_path / "lib.tif")
    ((got_key, message),) = raised.value.problems
    assert got_key == key and fragment in message
    assert raised.value.__cause__ is exc


def test_other_library_errors_are_not_disguised(walled, tmp_path, monkeypatch):
    _data, path = walled

    def boom(self, origin, algorithm="auto", **kwargs):
        raise InvalidInputError("something else entirely")

    monkeypatch.setattr(PathFinder, "cost_field", boom)
    with pytest.raises(InvalidInputError, match="something else entirely") as raised:
        _run(path, [_xy(5, 0, 0)], tmp_path / "other.tif")
    assert not isinstance(raised.value, cf.CostFieldRefused)


def test_job_error_notice_names_the_field_of_a_refusal_and_translates_anything_else():
    assert cf.job_error_notice(None) is None
    refused = cf.CostFieldRefused([cf.Problem("graph_api", "Backend (graph_api): no")])
    notice = cf.job_error_notice(refused)
    assert notice["title"] == "Backend: value not usable" and notice["meaning"] == "Backend (graph_api): no"
    other = cf.job_error_notice(RuntimeError("boom"))
    assert other["severity"] == "error" and other["title"] == errors.translate_exception(RuntimeError("x")).title


# ================================================================== with a terrain model
@pytest.fixture()
def dem_run(walled, tmp_path):
    data, path = walled
    dem = np.add.outer(np.arange(5) * 3.0, np.arange(5) * 1.0).astype(np.float32)
    dem_path = _write_raster(tmp_path / "dem.tif", dem, dtype="float32")
    return data, path, {"dem": dem_path, "objective": {"cost": 1.0}}


def test_with_a_dem_auto_resolves_to_dijkstra_and_the_field_is_computed(dem_run, tmp_path):
    _data, path, options = dem_run
    plan = _plan(path, {"algorithm": "auto"}, [_xy(5, 0, 0), _xy(5, 4, 4)], finder_options=options,
                 out_path=tmp_path / "dem_out.tif")
    assert plan.ok and plan.kwargs["finder_options"] == options
    result = cf.compute_cost_field(**plan.kwargs)
    assert result["algorithm"] == "dijkstra" and result["algorithm_requested"] == "auto"
    assert result["valid_cells"] > 0 and result["min"] > 0
    flat = _run(path, [_xy(5, 0, 0), _xy(5, 4, 4)], tmp_path / "flat_ref.tif")
    assert result["max"] != pytest.approx(flat["max"])        # the slope terms change the field


def test_with_a_dem_an_explicit_delta_stepping_is_a_refusal_not_a_traceback(dem_run, tmp_path):
    _data, path, options = dem_run
    with pytest.raises(cf.CostFieldRefused) as raised:
        _run(path, [_xy(5, 0, 0)], tmp_path / "dem_delta.tif", algorithm="delta-stepping", finder_options=options)
    assert [k for k, _m in raised.value.problems] == ["algorithm"]
    assert "terrain model (DEM)" in raised.value.problems[0].message


# ================================================================== jobs
def _plan_for_job(path, tmp_path, name, origins, **settings):
    plan = _plan(path, settings, origins, out_path=tmp_path / name)
    assert plan.ok, plan.problems
    return plan


def test_the_job_runs_the_computation_and_keeps_the_result(state, walled, tmp_path):
    data, path = walled
    plan = _plan_for_job(path, tmp_path, "job.tif", [_xy(5, 0, 0), _xy(5, 4, 4)], algorithm="dijkstra")
    job = jobs.start_job(state, "cost_field", cf.job_work, label="Cost field", params={"compute": plan.kwargs})
    assert job is not None
    job.thread.join(JOIN_S)
    assert job.status == jobs.DONE and job.error is None
    assert (job.done, job.total) == (2, 3) and job.stage == "writing the GeoTIFF"
    _assert_field(job.result["path"], _reference(data, (0, 0)) + _reference(data, (4, 4)))
    assert "cost_field" in state.jobs and jobs.take_finished(state, "cost_field") is job


def test_stop_ends_the_job_cancelled_with_no_file(state, walled, tmp_path, monkeypatch):
    _data, path = walled
    plan = _plan_for_job(path, tmp_path, "job_stop.tif", [_xy(5, 0, 0), _xy(5, 4, 4), _xy(5, 4, 0)])
    outcome = {}
    _count_fields(monkeypatch, after_first=lambda: outcome.update(stop=jobs.stop_job(state, "cost_field")))
    job = jobs.start_job(state, "cost_field", cf.job_work, label="Cost field", params={"compute": plan.kwargs})
    job.thread.join(JOIN_S)
    assert outcome["stop"].accepted and job.status == jobs.CANCELLED
    assert job.result is None and job.error is None
    assert not Path(plan.kwargs["out_path"]).exists()


def test_a_failing_job_gives_a_translated_notice(state, walled, tmp_path, monkeypatch):
    _data, path = walled
    plan = _plan_for_job(path, tmp_path, "job_fail.tif", [_xy(5, 0, 0)])

    def boom(self, origin, algorithm="auto", **kwargs):
        raise RuntimeError("the kernel fell over")

    monkeypatch.setattr(PathFinder, "cost_field", boom)
    job = jobs.start_job(state, "cost_field", cf.job_work, label="Cost field", params={"compute": plan.kwargs})
    job.thread.join(JOIN_S)
    assert job.status == jobs.FAILED
    notice = cf.job_error_notice(job.error)
    assert notice["severity"] == "error" and "the kernel fell over" in (notice["details"] or "")
    jobs.take_finished(state, "cost_field")


def test_a_refusal_inside_the_job_reaches_the_poll_side_as_a_field_notice(state, walled, tmp_path):
    _data, path = walled
    plan = _plan_for_job(path, tmp_path, "job_refused.tif", [_xy(5, 0, 0)])
    plan.kwargs["graph_api"] = "networkx"                 # the state of the page changed after the check
    job = jobs.start_job(state, "cost_field", cf.job_work, label="Cost field", params={"compute": plan.kwargs})
    job.thread.join(JOIN_S)
    assert job.status == jobs.FAILED and isinstance(job.error, cf.CostFieldRefused)
    assert cf.job_error_notice(job.error)["title"] == "Backend: value not usable"
    jobs.take_finished(state, "cost_field")


# ================================================================== finishing: the map layer
def test_finalize_serves_the_file_as_a_raster_layer_owned_by_the_tile_manager(state, walled, tmp_path):
    _data, path = walled
    result = _run(path, [_xy(5, 0, 0), _xy(5, 4, 4)], tmp_path / "layer.tif")
    layer, notices = cf.finalize_cost_field(state, result, [{"title": "earlier"}], raster_layer_id="abc123")
    assert layer is not None and layer.kind == "raster" and state.get(layer.id) is layer
    assert layer.name == "Cost field (sum of 2 point(s))" and layer.crs.to_string() == CRS
    assert layer.tile.nodata == -1.0 and layer.tile.source_path == result["path"]
    assert (layer.tile.vmin, layer.tile.vmax) == (result["min"], result["max"])
    assert state.tiles._clients[layer.id] is layer.tile.tile_client      # released with the layer
    best = result["best"]
    assert layer.meta["source_path"] == result["path"] and layer.meta["vmin"] == result["min"]
    assert layer.meta["cost_field"] == {
        "origins": result["origins"], "combine": "sum", "algorithm": "dijkstra", "best_xy": [best["x"], best["y"]],
        "best_cost": best["cost"], "unit": "cost units", "raster_layer_id": "abc123"}
    assert notices[0] == {"title": "earlier"}
    assert [n["severity"] for n in notices if n.get("severity") == "success"] == ["success"]
    assert "Cost field layer 'Cost field (sum of 2 point(s))' added" in notices[-1]["title"]
    assert "cell value x metres" in notices[-1]["impact"] and "Cheapest cell" in notices[-1]["meaning"]
    client = layer.tile.tile_client
    state.remove_layer(layer.id)
    assert getattr(client, "_pyorps_released", False) is True


def test_finalize_without_a_best_cell_and_with_a_custom_name_and_colormap(state, walled, tmp_path):
    _data, path = walled
    result = _run(path, [_xy(5, 0, 0)], tmp_path / "single.tif")
    layer, notices = cf.finalize_cost_field(state, result, name="Catchment", colormap="plasma")
    assert layer.name == "Catchment" and layer.tile.colormap == "plasma"
    assert layer.meta["cost_field"]["best_xy"] is None and layer.meta["cost_field"]["best_cost"] is None
    assert layer.meta["cost_field"]["raster_layer_id"] is None
    assert "a single fixed point" in notices[-1]["meaning"]
    state.remove_layer(layer.id)


def test_finalize_adds_the_notices_of_the_result(state, split, tmp_path):
    _data, path = split
    result = _run(path, [_xy(6, 1, 0)], tmp_path / "with_notice.tif")
    layer, notices = cf.finalize_cost_field(state, result, None)
    titles = [n["title"] for n in notices]
    assert "Some cells cannot be reached" in titles and titles[-1].startswith("Cost field layer")
    state.remove_layer(layer.id)


def test_a_file_that_cannot_be_served_gives_an_error_notice_and_no_layer(state, walled, tmp_path, monkeypatch):
    _data, path = walled
    result = _run(path, [_xy(5, 0, 0)], tmp_path / "unserved.tif")

    def boom(*_a, **_k):
        raise ValueError("no tile server today")

    monkeypatch.setattr(tiles, "build_tile_layer", boom)
    layer, notices = cf.finalize_cost_field(state, result, [])
    assert layer is None and state.layers == {}
    assert notices[-1]["severity"] == "error" and not any(n["severity"] == "success" for n in notices)


# ================================================================== GPU (skipped without a GPU)
@pytest.mark.skipif(not GPU, reason="needs CuPy and a CUDA GPU")
def test_raster_gpu_equals_the_independent_dijkstra(walled, tmp_path):
    data, path = walled
    result = _run(path, [_xy(5, 0, 0), _xy(5, 4, 4)], tmp_path / "gpu.tif", graph_api="raster_gpu", algorithm="auto")
    _assert_field(result["path"], _reference(data, (0, 0)) + _reference(data, (4, 4)))
    assert result["graph_api"] == "raster_gpu"


@pytest.mark.skipif(not GPU, reason="needs CuPy and a CUDA GPU")
def test_raster_fim_runs_through_the_service_and_marks_forbidden_cells(walled, tmp_path):
    """raster_fim at GUI level [V on a GPU]: the eikonal field is continuous, so only its shape is checked."""
    data, path = walled
    plan = _plan(path, {}, [_xy(5, 0, 0)], graph_api="raster_fim", out_path=tmp_path / "fim.tif")
    assert plan.ok
    result = cf.compute_cost_field(**plan.kwargs)
    values = _read(result["path"])
    assert values[0, 0] == 0.0 and (values[data == FORBIDDEN] == -1.0).all()
    assert (values[data != FORBIDDEN] >= 0).all() and result["graph_api"] == "raster_fim"
