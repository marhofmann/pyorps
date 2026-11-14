"""Round 14 (P4): ``python -m pyorps.gui --project <file>`` opens a saved project at start.

The flag is parsed, the project is loaded BEFORE the server starts (``launch`` is replaced by a recorder), and a missing
or broken file ends in one error line and exit code 1 before anything is served. A project opened by the flag must be
in exactly the state that the "Open project" button leaves, and the first page load (also a reload) must show it,
including the cost table, which the browser holds client-side. Everything is headless; only the two exit-code cases
start a subprocess, and neither reaches a server. The projects are written by the save service itself.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import geopandas as gpd
import pytest
from conftest import invoke
from plotly.utils import PlotlyJSONEncoder
from shapely.geometry import Polygon

from pyorps.gui import __main__ as cli
from pyorps.gui import ids, layout
from pyorps.gui.app import LaunchError, build_app
from pyorps.gui.callbacks import io as io_cb
from pyorps.gui.services import data_io, geo, project_io, rasterize, routing
from pyorps.gui.state import ProjectState

AREA = {"type": "Feature", "properties": {"type": "rectangle"},
        "geometry": {"type": "Polygon", "coordinates": [[
            [8.6, 50.0], [8.8, 50.0], [8.8, 50.2], [8.6, 50.2], [8.6, 50.0]]]}}
LINE = {"type": "FeatureCollection", "features": [
    {"type": "Feature", "properties": {},
     "geometry": {"type": "LineString", "coordinates": [[8.6, 50.0], [8.7, 50.1]]}}]}
ROUTING = {"graph_api": "cython", "algorithm": "dijkstra", "neighborhood_k": 3, "search_buffer_m": 250.0,
           "full_window": True, "ignore_max_cost": False, "pairwise": True, "simplify": True, "simplify_tol": 2.5,
           "delta": 50, "num_threads": 3, "margin": 1.5}
RASTER = {"resolution_m": 5.0, "fill_value": 65535, "dtype": "uint32", "geometry_buffer_m": 2,
          "widen_thin_forbidden": True, "all_touched": True, "on_thin_features": "warn", "crop_border": True}
GUI_SETTINGS = {"objective": {"preset": "gentlest"}, "cost_field": {"threads": 2}, "corridor": {"k": 4}}
ROWS = [{"use": "forest", "cost": 10, "forbidden": False}, {"use": "road", "cost": 5, "forbidden": False}]
MODIFIERS = [{"dataset": "landuse", "column": "use", "operator": "==", "value": "road", "factor": "2"}]
OPEN = ("project-status.children", ids.GO_OPEN_PROJECT)
_CLI_TIMEOUT = 120      # seconds; a server that did start would never end


# ------------------------------------------------------------------ helpers
@pytest.fixture(autouse=True)
def _temp_work_dirs(monkeypatch, tmp_path):
    """The work folder of a state without ``--work-dir`` goes under ``tmp_path``."""
    made: list[Path] = []

    def fake_mkdtemp(*_args, **_kwargs):
        folder = tmp_path / f"auto_work_{len(made)}"
        folder.mkdir()
        made.append(folder)
        return str(folder)

    monkeypatch.setattr(tempfile, "mkdtemp", fake_mkdtemp)
    return made


@pytest.fixture()
def launched(monkeypatch):
    """Replace ``launch`` by a recorder; each call snapshots what the state held AT that moment."""
    calls: list[dict] = []

    def fake_launch(state, **kwargs):
        calls.append({"state": state, "kwargs": kwargs, "at_launch": _snapshot(state) if state else None})

    monkeypatch.setattr(cli, "launch", fake_launch)
    yield calls
    for call in calls:
        if call["state"] is not None:
            call["state"].shutdown()


@pytest.fixture()
def saved(tmp_path):
    """A project written by the save service: vector layer, route, study area, settings, cost table."""
    maker = ProjectState(work_dir=tmp_path / "maker_work")
    try:
        gdf = gpd.GeoDataFrame(
            {"use": ["forest", "road"]},
            geometry=[Polygon([(500000, 5530000), (500100, 5530000), (500100, 5530100)]),
                      Polygon([(500200, 5530200), (500300, 5530200), (500300, 5530300)])],
            crs="EPSG:25832")
        vector = maker.add_layer("landuse", "vector", gdf=gdf, crs=gdf.crs, geojson=geo.gdf_to_wgs84_geojson(gdf),
                                 meta={"summary": data_io.dataset_summary(gdf), "source": {}})
        maker.study_area = AREA
        route = maker.add_layer("Route 1", "route", geojson=LINE)
        maker.active_route_id = route.id
        maker.project_crs = "EPSG:32632"
        maker.routing_settings = routing.normalise_routing_settings(ROUTING)
        maker.raster_settings = rasterize.normalise_raster_settings(RASTER)
        maker.gui_settings = {key: dict(value) for key, value in GUI_SETTINGS.items()}
        cost_table = {"feature_keys": ["use"], "dataset_id": vector.id, "rows": ROWS, "modifiers": MODIFIERS}
        manifest = Path(project_io.save_project(maker, tmp_path / "proj", cost_table=cost_table))
        return types.SimpleNamespace(
            manifest=manifest, folder=manifest.parent, dataset_id=vector.id, cost_table=cost_table,
            routing=dict(maker.routing_settings), raster=dict(maker.raster_settings),
            n_layers=len(maker.layers))
    finally:
        maker.shutdown()


def _snapshot(state) -> dict:
    """Everything of a state that a project load or the page can see, as plain comparable data."""
    layers = {}
    for layer in state.ordered_layers():
        layers[layer.id] = {
            "name": layer.name, "kind": layer.kind, "visible": layer.visible, "z": layer.z,
            "opacity": layer.opacity, "style": layer.style, "meta": layer.meta,
            "crs": str(layer.crs) if layer.crs else None, "geojson": layer.geojson,
            "wkt": None if layer.gdf is None else list(layer.gdf.geometry.to_wkt())}
    return {
        "layers": layers, "view": state.layers_view(), "project_crs": str(state.project_crs),
        "study_area": state.study_area, "active_route_id": state.active_route_id,
        "routing_settings": state.routing_settings, "raster_settings": state.raster_settings,
        "gui_settings": state.gui_settings, "loaded_cost_table": state.loaded_cost_table,
        "cost_tables": state.cost_tables, "dirty": state.dirty,
        "study_area_not_drawn": getattr(state, "_study_area_not_drawn", False),
        "routing_restore_backend": state.routing_restore_backend,
        "jobs": dict(state.jobs), "corridor_graphs": dict(state.corridor_graphs)}


def _first_paint(state) -> dict:
    """``{component id: component}`` of the page that a browser gets for ``state``."""
    return {c.id: c for c in layout.build_layout(state)._traverse()  # noqa: SLF001
            if isinstance(getattr(c, "id", None), str)}


def _open_by_button(app, path) -> dict:
    return invoke(app, OPEN, 1, str(path), [], triggered=[f"{ids.GO_OPEN_PROJECT}.data"])


def _run_cli(tmp_path, *argv):
    scratch = tmp_path / "cli_tmp"
    scratch.mkdir(exist_ok=True)
    env = {**os.environ, "TEMP": str(scratch), "TMP": str(scratch)}
    return subprocess.run([sys.executable, "-m", "pyorps.gui", *argv], cwd=tmp_path, env=env, capture_output=True,
                          text=True, timeout=_CLI_TIMEOUT, check=False), scratch


def _error_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("error:")]


# ================================================================ the flag
def test_no_flag_means_no_project():
    assert cli.parse_args([]).project is None


def test_a_relative_path_is_taken_from_the_current_folder(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    args = cli.parse_args(["--project", os.path.join("sub", "project.json")])
    assert args.project == Path.cwd() / "sub" / "project.json"
    assert args.project.is_absolute()


def test_an_absolute_path_is_kept_and_a_home_folder_is_expanded(monkeypatch, tmp_path):
    absolute = tmp_path / "a" / "project.json"
    assert cli.parse_args(["--project", str(absolute)]).project == absolute
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    assert cli.parse_args(["--project", "~/p.json"]).project == tmp_path / "p.json"


def test_the_path_need_not_exist_for_parsing(tmp_path):
    """A missing file is NOT a usage error (exit 2): main() ends it with exit code 1, like a broken file."""
    assert cli.parse_args(["--project", str(tmp_path / "nothing.json")]).project == tmp_path / "nothing.json"


def test_the_flag_needs_a_value():
    with pytest.raises(SystemExit) as exc:
        cli.parse_args(["--project"])
    assert exc.value.code == 2


def test_help_documents_the_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.parse_args(["--help"])
    assert exc.value.code == 0
    out = " ".join(capsys.readouterr().out.split())
    assert "--project FILE" in out
    for fragment in ("project.json", "folder", "current folder", "exit code 1"):
        assert fragment in out, fragment
    action = next(a for a in cli.build_parser()._actions if "--project" in a.option_strings)  # noqa: SLF001
    assert action.help and action.metavar == "FILE" and action.default is None


# ============================================================ main(): loading
def test_the_project_is_loaded_before_the_server_starts(launched, saved, capsys):
    assert cli.main(["--project", str(saved.manifest)]) == 0
    assert len(launched) == 1
    state, at_launch = launched[0]["state"], launched[0]["at_launch"]
    # all of it was in the state when launch() was called
    assert [row["name"] for row in at_launch["view"]] == [row["name"] for row in state.layers_view()]
    assert sorted(v["name"] for v in at_launch["layers"].values()) == ["Route 1", "landuse"]
    assert at_launch["project_crs"] == "EPSG:32632"
    assert at_launch["study_area"] == AREA
    assert at_launch["routing_settings"] == saved.routing
    assert at_launch["raster_settings"] == saved.raster
    assert at_launch["gui_settings"] == GUI_SETTINGS
    assert at_launch["loaded_cost_table"] == saved.cost_table
    assert at_launch["dirty"] is False                      # freshly loaded == on disk
    assert at_launch["active_route_id"] is None             # the opened project has no route selected
    assert at_launch["study_area_not_drawn"] is True        # the area is not a shape in the draw control
    out = capsys.readouterr()
    assert f"opened project: {saved.manifest} ({len(state.layers)} layers)" in out.out
    assert out.err == ""


def test_a_folder_that_holds_a_project_json_is_opened(launched, saved, capsys):
    assert cli.main(["--project", str(saved.folder)]) == 0
    assert len(launched[0]["state"].layers) == saved.n_layers
    assert f"opened project: {saved.manifest}" in capsys.readouterr().out


def test_a_relative_path_is_opened_from_the_current_folder(monkeypatch, launched, saved, capsys):
    monkeypatch.chdir(saved.folder.parent)
    assert cli.main(["--project", os.path.join(saved.folder.name, "project.json")]) == 0
    assert len(launched[0]["state"].layers) == saved.n_layers
    assert "opened project:" in capsys.readouterr().out


def test_one_layer_is_not_called_layers(launched, tmp_path, capsys):
    maker = ProjectState(work_dir=tmp_path / "w1")
    maker.add_layer("Route 1", "route", geojson=LINE)
    manifest = project_io.save_project(maker, tmp_path / "one")
    maker.shutdown()
    assert cli.main(["--project", manifest]) == 0
    assert "(1 layer)" in capsys.readouterr().out


def test_the_flag_works_together_with_every_other_flag(launched, saved, tmp_path):
    work = tmp_path / "session"
    code = cli.main(["--project", str(saved.manifest), "--desktop", "--host", "localhost", "--port", "9100",
                     "--threads", "5", "--work-dir", str(work), "--no-compress", "--debug", "--allow-remote"])
    assert code == 0
    call = launched[0]
    assert call["state"].work_dir == work and work.is_dir()
    assert call["kwargs"] == {"host": "localhost", "port": 9100, "threads": 5, "compress": False, "debug": True,
                              "desktop": True, "allow_remote": True}
    assert len(call["state"].layers) == saved.n_layers


def test_without_the_flag_nothing_changes(launched, tmp_path):
    assert cli.main([]) == 0
    assert launched[0]["state"] is None
    work = tmp_path / "w"
    assert cli.main(["--work-dir", str(work)]) == 0
    assert launched[1]["state"].work_dir == work and launched[1]["state"].loaded_cost_table == {}


def test_layers_that_could_not_be_restored_are_printed_as_warnings(launched, tmp_path, capsys):
    folder = tmp_path / "old"
    folder.mkdir()
    (folder / "project.json").write_text(json.dumps({"version": 1, "layers": [
        {"id": "r1", "name": "lost raster", "kind": "raster"},
        {"id": "x1", "name": "odd one", "kind": "hologram"},
        {"id": "g1", "name": "kept", "kind": "route", "geojson": LINE}]}), encoding="utf-8")
    assert cli.main(["--project", str(folder)]) == 0
    err = capsys.readouterr().err
    warning_lines = [line for line in err.splitlines() if line.startswith("warning:")]
    assert len(warning_lines) == 1
    assert "2 layer(s)" in warning_lines[0] and "lost raster" in warning_lines[0] and "odd one" in warning_lines[0]
    assert [v["name"] for v in launched[0]["state"].layers_view()] == ["kept"]


# ========================================== main(): a missing or broken file
def _fails(argv, capsys, launched) -> str:
    """``main`` returned 1 with ONE error line and no traceback, and nothing was served."""
    assert cli.main(argv) == 1
    assert launched == []
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err and "Traceback" not in captured.out
    assert len(_error_lines(captured.err)) == 1 and len(captured.err.strip().splitlines()) == 1
    assert captured.out == ""
    return captured.err.strip()


def test_a_missing_file_exits_1_before_any_state_exists(monkeypatch, launched, tmp_path, capsys):
    def boom(*_a, **_k):
        raise AssertionError("no state may be created for a file that does not exist")

    monkeypatch.setattr(cli, "ProjectState", boom)
    missing = tmp_path / "nowhere" / "project.json"
    line = _fails(["--project", str(missing)], capsys, launched)
    assert str(missing) in line and "does not exist" in line and line.startswith("error: could not open the project")


def test_a_folder_without_a_project_json_exits_1(launched, tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    line = _fails(["--project", str(tmp_path / "empty")], capsys, launched)
    assert str(tmp_path / "empty" / "project.json") in line and "no project.json" in line


def test_a_file_that_is_not_json_exits_1_and_names_the_file(launched, tmp_path, capsys, _temp_work_dirs):
    broken = tmp_path / "broken.json"
    broken.write_text("{ this is not json", encoding="utf-8")
    line = _fails(["--project", str(broken)], capsys, launched)
    assert str(broken) in line and "could not be read" in line
    assert [p for p in _temp_work_dirs if p.exists()] == []      # the empty temporary work folder is removed again


@pytest.mark.parametrize("text", ["[]", "42", '"text"', '{"layers": 7}', '{"layers": ["not a layer"]}'])
def test_json_that_is_not_a_project_exits_1_without_a_traceback(launched, tmp_path, capsys, text):
    odd = tmp_path / "odd.json"
    odd.write_text(text, encoding="utf-8")
    line = _fails(["--project", str(odd)], capsys, launched)
    assert str(odd) in line and "not a valid project" in line


def test_a_missing_asset_exits_1_and_names_the_layer_and_the_file(launched, saved, capsys):
    asset = next((saved.folder / "assets").glob("*.gpkg"))
    asset.unlink()
    line = _fails(["--project", str(saved.manifest)], capsys, launched)
    assert str(saved.manifest) in line and "landuse" in line and str(asset) in line and "is missing" in line


def test_a_work_folder_the_user_named_is_never_removed(launched, saved, tmp_path, capsys):
    next((saved.folder / "assets").glob("*.gpkg")).unlink()
    work = tmp_path / "mine"
    _fails(["--project", str(saved.manifest), "--work-dir", str(work)], capsys, launched)
    assert work.is_dir()


def test_a_launch_error_after_a_good_load_is_still_one_line(monkeypatch, saved, capsys):
    def fail(*_a, **_k):
        raise LaunchError("no free port")

    monkeypatch.setattr(cli, "launch", fail)
    assert cli.main(["--project", str(saved.manifest)]) == 1
    assert "error: no free port" in capsys.readouterr().err


def test_exit_code_1_for_a_missing_file_in_a_real_process(tmp_path):
    result, scratch = _run_cli(tmp_path, "--project", os.path.join("nowhere", "project.json"))
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    lines = _error_lines(result.stderr)
    assert len(lines) == 1 and "does not exist" in lines[0]
    assert os.path.normcase(str(tmp_path)) in os.path.normcase(lines[0])      # taken from the current folder
    assert "PYORPS GUI" not in result.stdout                                   # no server banner
    assert [p for p in scratch.iterdir() if p.name.startswith("pyorps_gui_")] == []      # no state, no work folder


def test_exit_code_1_for_a_broken_file_in_a_real_process(tmp_path):
    (tmp_path / "project.json").write_text("{ this is not json", encoding="utf-8")
    result, scratch = _run_cli(tmp_path, "--project", "project.json", "--desktop")
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    lines = _error_lines(result.stderr)
    assert len(lines) == 1 and "could not be read" in lines[0] and "project.json" in lines[0]
    assert "PYORPS GUI" not in result.stdout
    assert [p for p in scratch.iterdir() if p.name.startswith("pyorps_gui_")] == []


# ====================================== the same state as the "Open project" button
def test_flag_and_button_leave_exactly_the_same_state(launched, saved, tmp_path):
    assert cli.main(["--project", str(saved.manifest)]) == 0
    by_flag = _snapshot(launched[0]["state"])

    other = ProjectState(work_dir=tmp_path / "button_work")
    try:
        # the button replaces a session that holds other things: none of them may survive
        other.add_layer("stale", "vector", geojson={"type": "FeatureCollection", "features": []})
        other.study_area = {"type": "Feature", "properties": {"type": "rectangle"},
                            "geometry": {"type": "Polygon", "coordinates": [[
                                [9.0, 50.0], [9.1, 50.0], [9.1, 50.1], [9.0, 50.1], [9.0, 50.0]]]}}
        other.active_route_id = "stale-route"
        other.dirty = True
        other.gui_settings = {"objective": {"stale": True}}
        other.loaded_cost_table = {"feature_keys": ["stale"]}
        resp = _open_by_button(build_app(other), saved.manifest)
        assert resp[ids.PROJECT_STATUS]["children"] == f"opened: {saved.manifest}"
        by_button = _snapshot(other)
    finally:
        other.shutdown()

    assert by_flag.keys() == by_button.keys()
    for key in by_flag:
        assert by_flag[key] == by_button[key], key
    assert len(by_flag["layers"]) == saved.n_layers and by_flag["loaded_cost_table"] == saved.cost_table
    table_id = f"tbl-{saved.dataset_id}-use"
    assert by_flag["cost_tables"][table_id]["rows"] == ROWS     # registered for the Raster tab, as a cell edit does


def test_both_ways_run_the_same_service_function(monkeypatch, launched, saved, state, app):
    seen = []
    real = project_io.load_project

    def spy(target_state, path):
        seen.append(Path(path))
        return real(target_state, path)

    monkeypatch.setattr(project_io, "load_project", spy)
    cli.main(["--project", str(saved.manifest)])
    _open_by_button(app, saved.manifest)
    assert seen == [saved.manifest, saved.manifest]


# =============================================== the first page load shows it
def test_the_first_paint_shows_the_opened_project(launched, saved):
    cli.main(["--project", str(saved.manifest)])
    state = launched[0]["state"]
    page = _first_paint(state)
    assert page[ids.PROJECT_CRS].value == "EPSG:32632"
    assert [row["name"] for row in page[ids.LAYERS_VIEW].data] == [row["name"] for row in state.layers_view()]
    assert "landuse" in [row["name"] for row in page[ids.LAYERS_VIEW].data]
    assert [o["label"] for o in page[ids.EDIT_ROUTE_SELECT].options] == ["Route 1"]
    # settings
    assert (page[ids.GRAPH_API].value, page[ids.ALGORITHM].value) == ("cython", "dijkstra")
    assert page[ids.NEIGHBORHOOD].value == 3 and page[ids.SEARCH_BUFFER].value == 250.0
    assert page[ids.FULL_WINDOW].value is True and page[ids.IGNORE_MAX_COST].value is False
    assert page[ids.RES_M].value == 5.0 and page[ids.RASTER_DTYPE].value == "uint32"
    assert page[ids.CROP_BORDER].value is True
    assert page[ids.ROUTING_SETTINGS].data == saved.routing and page[ids.RASTER_SETTINGS].data == saved.raster
    # the cost table as loaded
    assert page[ids.COST_GRID].rowData == ROWS
    assert [c["field"] for c in page[ids.COST_GRID].columnDefs] == ["use", "cost", "forbidden"]
    assert page[ids.COST_GRID_STATE].data == {"dataset_id": saved.dataset_id, "feature_keys": ["use"]}
    assert page[ids.MODIFIER_GRID].rowData == MODIFIERS
    assert page[ids.COST_COVERAGE_INFO].children.startswith("✓")
    assert "landuse" in json.dumps(page[ids.DATASET_LIST].children, default=str)
    assert "km" in page[ids.STUDY_AREA_INFO].children


def test_the_first_paint_is_what_the_button_sends_to_the_browser(launched, saved, state, app):
    cli.main(["--project", str(saved.manifest)])
    page = _first_paint(launched[0]["state"])
    resp = _open_by_button(app, saved.manifest)
    for component, prop in ((ids.COST_GRID, "columnDefs"), (ids.COST_GRID, "rowData"), (ids.COST_GRID_STATE, "data"),
                            (ids.MODIFIER_GRID, "rowData"), (ids.COST_COVERAGE_INFO, "children"),
                            (ids.LAYERS_VIEW, "data"), (ids.PROJECT_CRS, "value"), (ids.GRAPH_API, "value"),
                            (ids.ALGORITHM, "value"), (ids.NEIGHBORHOOD, "value"), (ids.RES_M, "value"),
                            (ids.RASTER_DTYPE, "value"), (ids.EDIT_ROUTE_SELECT, "options"),
                            (ids.DATASET_LIST, "children"), (ids.STUDY_AREA_INFO, "children")):
        sent = json.loads(json.dumps(getattr(page[component], prop), cls=PlotlyJSONEncoder))
        assert sent == resp[component][prop], (component, prop)


def test_a_reload_shows_the_project_again(launched, saved):
    """A browser reload evaluates the page layout again (``PageLayout``): same table, same settings."""
    cli.main(["--project", str(saved.manifest)])
    state = launched[0]["state"]
    page_layout = build_app(state).layout
    first, second = page_layout(), page_layout()
    for page in (first, second):
        by_id = {c.id: c for c in page._traverse() if isinstance(getattr(c, "id", None), str)}  # noqa: SLF001
        assert by_id[ids.COST_GRID].rowData == ROWS
        assert by_id[ids.MODIFIER_GRID].rowData == MODIFIERS
        assert by_id[ids.COST_GRID_STATE].data["feature_keys"] == ["use"]
        assert by_id[ids.RES_M].value == 5.0 and by_id[ids.NEIGHBORHOOD].value == 3
        assert len(by_id[ids.LAYERS_VIEW].data) == saved.n_layers


def test_open_project_by_button_remembers_the_cost_table_for_a_reload(saved, state, app):
    assert state.loaded_cost_table == {}
    _open_by_button(app, saved.manifest)
    assert state.loaded_cost_table == saved.cost_table
    page = _first_paint(state)
    assert page[ids.COST_GRID].rowData == ROWS and page[ids.MODIFIER_GRID].rowData == MODIFIERS
    assert page[ids.COST_COVERAGE_INFO].children.startswith("✓")


def test_a_project_without_a_cost_table_leaves_the_empty_grids(launched, tmp_path):
    maker = ProjectState(work_dir=tmp_path / "w")
    maker.add_layer("Route 1", "route", geojson=LINE)
    manifest = project_io.save_project(maker, tmp_path / "nt", include=["raster", "vector"])
    maker.shutdown()
    cli.main(["--project", manifest])
    state = launched[0]["state"]
    assert state.loaded_cost_table == {}
    page = _first_paint(state)
    assert page[ids.COST_GRID].rowData == [] and page[ids.COST_GRID].columnDefs == []
    assert page[ids.COST_GRID_STATE].data == {} and page[ids.MODIFIER_GRID].rowData == []
    values = layout._state_prefill(state)  # noqa: SLF001
    assert ids.COST_GRID not in values and ids.COST_GRID_STATE not in values


def test_open_project_without_a_cost_table_forgets_the_one_of_the_project_before(saved, tmp_path, state, app):
    _open_by_button(app, saved.manifest)
    assert state.loaded_cost_table
    maker = ProjectState(work_dir=tmp_path / "w")
    maker.add_layer("Route 1", "route", geojson=LINE)
    empty = project_io.save_project(maker, tmp_path / "nt", include=[])
    maker.shutdown()
    _open_by_button(app, empty)
    assert state.loaded_cost_table == {} and state.cost_tables == {}
    assert ids.COST_GRID not in layout._state_prefill(state)  # noqa: SLF001


def test_new_project_clears_the_remembered_cost_table(saved, state, app):
    _open_by_button(app, saved.manifest)
    invoke(app, ("project-status.children", ids.GO_NEW_PROJECT), 1, [], triggered=[f"{ids.GO_NEW_PROJECT}.data"])
    assert state.loaded_cost_table == {}
    assert _first_paint(state)[ids.COST_GRID].rowData == []


def test_a_blank_state_prefills_no_cost_grid(state):
    values = layout._state_prefill(state)  # noqa: SLF001
    assert not {ids.COST_GRID, ids.COST_GRID_STATE, ids.MODIFIER_GRID, ids.COST_COVERAGE_INFO} & set(values)


def test_a_failing_coverage_text_does_not_hide_the_table(monkeypatch, saved, state, app):
    from pyorps.gui.callbacks import costmodel

    _open_by_button(app, saved.manifest)

    def fail(*_a, **_k):
        raise RuntimeError("coverage exploded")

    monkeypatch.setattr(costmodel, "_coverage_text", fail)
    page = _first_paint(state)
    assert page[ids.COST_GRID].rowData == ROWS and page[ids.COST_COVERAGE_INFO].children == ""


@pytest.mark.parametrize("table", ["text", 7, ["a"], {"feature_keys": 5, "rows": "x", "modifiers": None},
                                   {"feature_keys": ["use", 3], "rows": [{"use": "a"}]}])
def test_a_broken_cost_table_never_takes_the_page_down(state, table):
    state.loaded_cost_table = table
    page = _first_paint(state)                                  # must not raise
    assert ids.COST_GRID in page


def test_apply_opened_project_keeps_only_a_mapping_as_the_cost_table(state):
    for raw in (None, "x", [1], 3):
        io_cb.apply_opened_project(state, {"cost_table": raw})
        assert state.loaded_cost_table == {}
    io_cb.apply_opened_project(state, {})
    assert state.loaded_cost_table == {}
    table = {"feature_keys": ["use"], "rows": ROWS}
    io_cb.apply_opened_project(state, {"cost_table": table})
    assert state.loaded_cost_table == table and state.loaded_cost_table is not table


def test_cost_table_view_matches_the_values_open_project_sends(saved):
    manifest = json.loads(saved.manifest.read_text(encoding="utf-8"))
    view = io_cb.cost_table_view(manifest["cost_table"])
    assert view["rows"] == ROWS and view["modifiers"] == MODIFIERS and view["keys"] == ("use",)
    assert view["grid_state"] == {"dataset_id": saved.dataset_id, "feature_keys": ["use"]}
    assert [c["field"] for c in view["column_defs"]] == ["use", "cost", "forbidden"]
    assert io_cb.cost_table_view(None) == {
        "keys": (), "dataset_id": None, "rows": [], "grid_state": {}, "column_defs": [], "modifiers": []}


def test_gui_settings_of_the_project_survive_the_flag_and_the_page(launched, saved):
    cli.main(["--project", str(saved.manifest)])
    state = launched[0]["state"]
    _first_paint(state)                                         # the cards' own prefill runs; it never changes them
    assert state.gui_settings == GUI_SETTINGS


# ========================== tidy-up: the first page load names the opened project (state.project_path)
def test_a_new_state_has_no_project_path(state):
    assert state.project_path is None


def test_the_flag_sets_the_path_and_the_first_page_load_names_the_project(launched, saved):
    cli.main(["--project", str(saved.manifest)])
    state = launched[0]["state"]
    assert state.project_path == str(saved.manifest)
    status = _first_paint(state)[ids.PROJECT_STATUS].children
    assert status == f"opened: {saved.manifest}" and "session restored" not in status
    assert _first_paint(state)[ids.LAYERS_VIEW].data                      # the layers are still there too


def test_the_flag_with_a_folder_names_the_project_file_inside_it(launched, saved):
    cli.main(["--project", str(saved.folder)])
    assert launched[0]["state"].project_path == str(saved.manifest)


def test_open_project_by_button_sets_the_path_and_keeps_its_status_text(saved, state, app):
    resp = _open_by_button(app, saved.manifest)
    assert resp[ids.PROJECT_STATUS]["children"] == f"opened: {saved.manifest}"            # the text of today
    assert state.project_path == str(saved.manifest)
    assert _first_paint(state)[ids.PROJECT_STATUS].children == f"opened: {saved.manifest}"


def test_a_failed_open_never_names_the_file_that_failed(state, app, tmp_path):
    missing = tmp_path / "missing" / "project.json"
    _open_by_button(app, missing)
    assert state.project_path != str(missing)


def test_save_sets_the_path_and_keeps_its_status_text(app, state, tmp_path):
    state.add_layer("a route", "route", geojson=LINE)
    target = tmp_path / "saved_session"
    resp = invoke(app, (f"{ids.PROJECT_STATUS}.children", ids.GO_SAVE_PROJECT), 1, str(target),
                  ["raster", "vector", "cost_table"], [], {}, [], [], triggered=[f"{ids.GO_SAVE_PROJECT}.data"])
    manifest = target / "project.json"
    assert resp[ids.PROJECT_STATUS]["children"] == f"saved: {manifest}"                    # the text of today
    assert state.project_path == str(manifest)
    assert _first_paint(state)[ids.PROJECT_STATUS].children == f"opened: {manifest}"


def test_a_save_without_a_path_leaves_the_project_path_alone(app, state):
    state.project_path = "some/earlier/project.json"
    invoke(app, (f"{ids.PROJECT_STATUS}.children", ids.GO_SAVE_PROJECT), 1, "", ["raster", "vector", "cost_table"],
           [], {}, [], [], triggered=[f"{ids.GO_SAVE_PROJECT}.data"])
    assert state.project_path == "some/earlier/project.json"


def test_new_project_clears_the_path_and_the_page_says_nothing_about_a_project(saved, state, app):
    _open_by_button(app, saved.manifest)
    assert state.project_path
    resp = invoke(app, ("project-status.children", ids.GO_NEW_PROJECT), 1, [], triggered=[f"{ids.GO_NEW_PROJECT}.data"])
    assert resp[ids.PROJECT_STATUS]["children"] == "new project"
    assert state.project_path is None
    assert not getattr(_first_paint(state)[ids.PROJECT_STATUS], "children", None)


def test_a_session_without_a_project_file_still_says_session_restored(state):
    state.add_layer("a route", "route", geojson=LINE)
    assert state.project_path is None
    status = _first_paint(state)[ids.PROJECT_STATUS].children
    assert status.startswith("session restored: 1 layer(s) are still loaded on the server")


def test_flag_and_button_agree_on_the_path_of_a_project_file(launched, saved, tmp_path):
    cli.main(["--project", str(saved.manifest)])
    other = ProjectState(work_dir=tmp_path / "button_work2")
    try:
        _open_by_button(build_app(other), saved.manifest)
        assert other.project_path == launched[0]["state"].project_path
    finally:
        other.shutdown()
