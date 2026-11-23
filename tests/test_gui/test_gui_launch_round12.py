"""Round 12: start-up code and native file dialogs (audit T5 #1 layout part,
#27, #28, #32).

* host policy: a non-loopback ``--host`` needs ``--allow-remote``
* port probe on a wildcard host (``0.0.0.0``) and bind test
* desktop mode waits until the server listens before the window opens
* ``python -m pyorps.gui --help`` lists every option; every ``launch`` setting
  is reachable from the command line
* file dialogs run in a child process: path / cancelled / unavailable, timeout
  kills the child before the lock is released
* the browse buttons answer with a notice when no dialog can be shown
* the app layout is a function evaluated on every page load (F5)
* at exit the work folder is reported (and nothing is deleted)

No real Tk window, pywebview window or network server is used.
"""
from __future__ import annotations

import inspect
import json
import socket
import subprocess
import sys
import threading
import types
from pathlib import Path

import pytest
from conftest import invoke
from dash.exceptions import PreventUpdate

from pyorps.gui import __main__ as cli
from pyorps.gui import app as app_module
from pyorps.gui import ids
from pyorps.gui.services import dialog_worker, dialogs
from pyorps.gui.state import ProjectState

SQUARE = {"type": "FeatureCollection", "features": [{
    "type": "Feature",
    "geometry": {"type": "Polygon", "coordinates": [[
        [8.6, 50.0], [8.8, 50.0], [8.8, 50.2], [8.6, 50.2], [8.6, 50.0]]]},
    "properties": {"name": "sq"}}]}


# ------------------------------------------------------------------ helpers
def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture()
def held_port():
    """A port held open (listening) on 127.0.0.1 only."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    try:
        yield server.getsockname()[1]
    finally:
        server.close()


# ====================================================== #28a host policy
@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "LOCALHOST", "::1",
                                  "[::1]", "127.0.0.5"])
def test_loopback_hosts_are_allowed(host):
    assert app_module.is_loopback_host(host)
    assert app_module.check_host(host) is False


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "", "192.168.1.20",
                                  "gui.example.org"])
def test_other_hosts_are_refused_without_the_flag(host):
    assert not app_module.is_loopback_host(host)
    with pytest.raises(app_module.LaunchError, match="--allow-remote"):
        app_module.check_host(host)
    assert app_module.check_host(host, allow_remote=True) is True


def test_wildcard_detection():
    for host in ("0.0.0.0", "::", "", "[::]"):
        assert app_module.is_wildcard_host(host)
    for host in ("127.0.0.1", "localhost", "192.168.1.20"):
        assert not app_module.is_wildcard_host(host)


def test_cli_refuses_remote_host(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.parse_args(["--host", "0.0.0.0"])
    assert exc.value.code == 2
    assert "--allow-remote" in capsys.readouterr().err


def test_cli_accepts_remote_host_with_the_flag():
    args = cli.parse_args(["--host", "0.0.0.0", "--allow-remote"])
    assert args.host == "0.0.0.0" and args.allow_remote is True


def test_remote_warning_names_every_risk():
    text = app_module.remote_warning("0.0.0.0", 8050)
    assert "NO authentication" in text
    assert "SERVER machine" in text and "file dialogs" in text
    assert "localhost" in text and "will not load" in text
    assert "Werkzeug" not in text
    assert "Werkzeug" in app_module.remote_warning("0.0.0.0", 8050, debug=True)


def test_launch_refuses_before_creating_any_state(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("must not run for a refused host")

    monkeypatch.setattr(app_module, "ProjectState", boom)
    monkeypatch.setattr(app_module, "build_app", boom)
    with pytest.raises(app_module.LaunchError, match="--allow-remote"):
        app_module.launch(host="0.0.0.0")


def _stub_launch(monkeypatch, served):
    """launch() without a server, an app, a port probe or an exit hook."""
    registered = []
    monkeypatch.setattr(app_module, "build_app", lambda *_a, **_k: "APP")
    monkeypatch.setattr(app_module, "_resolve_port", lambda _h, port: port)
    monkeypatch.setattr(app_module, "_serve",
                        lambda *args: served.append(args))
    monkeypatch.setattr(app_module.atexit, "register",
                        lambda fn, *a: registered.append((fn, a)))
    return registered


def test_launch_prints_the_warning_with_allow_remote(monkeypatch, tmp_path,
                                                     capsys):
    served = []
    _stub_launch(monkeypatch, served)
    state = ProjectState(work_dir=tmp_path / "w")
    app_module.launch(state, host="0.0.0.0", port=9000, allow_remote=True,
                      threads=3)
    err = capsys.readouterr().err
    assert "NO authentication" in err and "0.0.0.0:9000" in err
    assert served == [("APP", "0.0.0.0", 9000, False, 3)]


def test_launch_loopback_prints_no_warning(monkeypatch, tmp_path, capsys):
    _stub_launch(monkeypatch, [])
    app_module.launch(ProjectState(work_dir=tmp_path / "w"), port=9001)
    assert "NO authentication" not in capsys.readouterr().err


# =================================================== #28b port probe
def test_wildcard_host_is_probed_on_loopback(held_port):
    assert app_module._connect_host("0.0.0.0") == "127.0.0.1"
    assert app_module._connect_host("::") == "::1"
    assert app_module._connect_host("127.0.0.1") == "127.0.0.1"
    assert app_module._accepts_connections("0.0.0.0", held_port)
    assert not app_module._accepts_connections("0.0.0.0", _free_port())


def test_busy_port_is_skipped_for_a_wildcard_host(held_port, capsys):
    chosen = app_module._resolve_port("0.0.0.0", held_port)
    assert chosen > held_port
    assert f"port {held_port} is already in use" in capsys.readouterr().out


def test_free_port_is_kept_for_a_wildcard_host():
    free = _free_port()
    assert app_module._resolve_port("0.0.0.0", free) == free


def test_bind_test_sees_a_socket_that_does_not_listen():
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))              # bound, never listening
    port = blocker.getsockname()[1]
    try:
        assert not app_module._accepts_connections("127.0.0.1", port)
        assert app_module._cannot_bind("127.0.0.1", port)
        assert app_module._resolve_port("127.0.0.1", port) != port
    finally:
        blocker.close()
    assert not app_module._cannot_bind("127.0.0.1", port)


def test_resolve_port_gives_up_when_everything_is_busy(monkeypatch):
    monkeypatch.setattr(app_module, "_port_in_use", lambda _h, _p: True)
    with pytest.raises(app_module.LaunchError, match="no free port"):
        app_module._resolve_port("127.0.0.1", 8050, attempts=3)


def test_browse_url_is_something_a_browser_can_open():
    assert app_module._browse_url("0.0.0.0", 8050) == "http://localhost:8050"
    assert app_module._browse_url("127.0.0.1", 1) == "http://127.0.0.1:1"
    assert app_module._browse_url("::1", 8050) == "http://[::1]:8050"


def test_serve_passes_host_port_and_threads_to_waitress(monkeypatch):
    calls = []
    monkeypatch.setattr("waitress.serve",
                        lambda wsgi, **kw: calls.append((wsgi, kw)))
    app = types.SimpleNamespace(server="WSGI")
    app_module._serve(app, "127.0.0.1", 8123, False, 3)
    assert calls == [("WSGI", {"host": "127.0.0.1", "port": 8123,
                               "threads": 3, "ident": "pyorps-gui"})]


# ============================================== #28c desktop wait loop
class _Clock:
    """Fake monotonic clock that advances only through ``sleep``."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def test_wait_returns_as_soon_as_the_port_answers():
    clock = _Clock()
    answers = iter([False, False, True])
    app_module._wait_for_server(
        "127.0.0.1", 1, probe=lambda _h, _p: next(answers),
        clock=clock, sleep=clock.sleep, interval=0.1)
    assert clock.sleeps == [0.1, 0.1]


def test_wait_times_out_with_a_clear_error():
    clock = _Clock()
    with pytest.raises(app_module.LaunchError,
                       match=r"did not start listening on http://127.0.0.1:1 within 2 s"):
        app_module._wait_for_server(
            "127.0.0.1", 1, timeout=2.0, interval=0.5,
            probe=lambda _h, _p: False, clock=clock, sleep=clock.sleep)
    assert clock.now >= 2.0


def test_wait_fails_fast_when_the_server_thread_died():
    dead = types.SimpleNamespace(is_alive=lambda: False)
    clock = _Clock()
    with pytest.raises(app_module.LaunchError, match="stopped before"):
        app_module._wait_for_server(
            "127.0.0.1", 1, server=dead, probe=lambda _h, _p: False,
            clock=clock, sleep=clock.sleep)
    assert clock.sleeps == []


def test_wait_probes_the_loopback_for_a_wildcard_host(held_port):
    clock = _Clock()
    app_module._wait_for_server("0.0.0.0", held_port, clock=clock,
                                sleep=clock.sleep)
    assert clock.sleeps == []


def _fake_webview(events):
    module = types.ModuleType("webview")

    def create_window(title, url, **kwargs):
        events.append(("create_window", url))
        return types.SimpleNamespace(
            events=types.SimpleNamespace(closing=_Hook()))

    module.create_window = create_window
    module.start = lambda: events.append(("start",))
    return module


class _Hook:
    def __iadd__(self, _fn):
        return self


def test_desktop_window_opens_only_after_the_server_listens(monkeypatch,
                                                            tmp_path):
    events = []
    _stub_launch(monkeypatch, [])
    monkeypatch.setitem(sys.modules, "webview", _fake_webview(events))
    monkeypatch.setattr(
        app_module, "_wait_for_server",
        lambda host, port, **kw: events.append(("wait", host, port)))
    app_module.launch(ProjectState(work_dir=tmp_path / "w"), host="0.0.0.0",
                      port=5555, desktop=True, allow_remote=True)
    assert events == [("wait", "0.0.0.0", 5555),
                      ("create_window", "http://localhost:5555"),
                      ("start",)]


def test_desktop_without_pywebview_is_a_clear_error(monkeypatch, tmp_path):
    _stub_launch(monkeypatch, [])
    monkeypatch.setitem(sys.modules, "webview", None)       # import fails
    with pytest.raises(app_module.LaunchError, match="pywebview"):
        app_module.launch(ProjectState(work_dir=tmp_path / "w"), desktop=True)


# ================================================= #28d command line
def test_help_lists_every_option_with_a_description(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.parse_args(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for option in ("--desktop", "--host", "--port", "--allow-remote",
                   "--threads", "--work-dir", "--compress", "--no-compress",
                   "--debug"):
        assert option in out
    parser = cli.build_parser()
    for action in parser._actions:                           # noqa: SLF001
        assert action.help, f"{action.option_strings} has no help text"


def test_defaults_are_todays_values():
    args = cli.parse_args([])
    assert (args.host, args.port, args.threads) == ("127.0.0.1", 8050, 8)
    assert (args.desktop, args.debug, args.allow_remote) == (False, False, False)
    assert args.compress is None and args.work_dir is None


@pytest.mark.parametrize("argv", [["--port", "0"], ["--port", "70000"],
                                  ["--threads", "0"]])
def test_out_of_range_numbers_are_usage_errors(argv):
    with pytest.raises(SystemExit) as exc:
        cli.parse_args(argv)
    assert exc.value.code == 2


def test_every_launch_setting_is_reachable_from_main(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(cli, "launch",
                        lambda state, **kw: seen.update(kw, state=state))
    work = tmp_path / "session"
    code = cli.main(["--host", "localhost", "--port", "9100", "--threads", "5",
                     "--work-dir", str(work), "--no-compress", "--debug",
                     "--desktop", "--allow-remote"])
    assert code == 0
    assert seen["state"].work_dir == work and work.is_dir()
    assert seen == {
        "state": seen["state"], "host": "localhost", "port": 9100,
        "threads": 5, "compress": False, "debug": True, "desktop": True,
        "allow_remote": True}
    settings = set(inspect.signature(app_module.launch).parameters) - {"state"}
    assert settings <= set(seen), "a launch() setting has no command-line flag"


def test_main_reports_a_launch_error_without_a_traceback(monkeypatch, capsys):
    def fail(*_a, **_k):
        raise app_module.LaunchError("no free port")

    monkeypatch.setattr(cli, "launch", fail)
    assert cli.main([]) == 1
    assert "error: no free port" in capsys.readouterr().err


# ===================================================== #27 dialog service
class _FakeProc:
    """Stands in for ``subprocess.Popen``; records the order of events and
    whether the one-dialog-at-a-time lock was held at each step."""

    def __init__(self, events, stdout="", stderr="", returncode=0, hang=False):
        self.events = events
        self._stdout, self._stderr = stdout, stderr
        self._returncode, self._hang = returncode, hang
        self.returncode = None
        self.killed = False

    def _note(self, what, **extra):
        self.events.append((what, dialogs._dialog_lock.locked(), extra))  # noqa: SLF001

    def communicate(self, input=None, timeout=None):          # noqa: A002
        self._note("communicate", input=input, timeout=timeout)
        if self._hang and not self.killed:
            raise subprocess.TimeoutExpired(cmd="dialog", timeout=timeout)
        self.returncode = -9 if self.killed else self._returncode
        return self._stdout, self._stderr

    def kill(self):
        self._note("kill")
        self.killed = True

    def poll(self):
        return self.returncode


@pytest.fixture()
def popen(monkeypatch):
    """Patch Popen; ``popen.proc`` is the next fake child, ``popen.calls`` the
    (argv, kwargs) it was started with, ``popen.events`` the event log."""
    class Handle:
        events: list = []
        calls: list = []
        proc: _FakeProc | None = None
        error: Exception | None = None

    handle = Handle()
    handle.events, handle.calls = [], []

    def factory(argv, **kwargs):
        handle.calls.append((argv, kwargs))
        if handle.error is not None:
            raise handle.error
        return handle.proc

    monkeypatch.setattr(dialogs.subprocess, "Popen", factory)
    return handle


def _reply(popen, payload, **kw):
    popen.proc = _FakeProc(popen.events, stdout=json.dumps(payload) + "\n", **kw)


def test_dialog_path_chosen(popen):
    _reply(popen, {"status": "path", "path": "C:/data/x.tif"})
    result = dialogs.ask_open("raster")
    assert result.status == "path" and result.path == "C:/data/x.tif"
    assert tuple(result) == ("C:/data/x.tif", None)         # legacy unpacking
    argv, kwargs = popen.calls[0]
    assert argv[0] == sys.executable and "-I" in argv
    assert Path(argv[-1]).name == "dialog_worker.py" and Path(argv[-1]).is_file()
    assert kwargs["stdin"] == subprocess.PIPE
    sent = json.loads(popen.events[0][2]["input"])
    assert sent["mode"] == "open" and sent["filetypes"] == [
        list(t) for t in dialogs.FILETYPES["raster"]]
    # the lock was held while the child ran and is free again afterwards
    assert popen.events[0][1] is True
    assert dialogs._dialog_lock.locked() is False            # noqa: SLF001


def test_dialog_save_and_directory_requests(popen):
    _reply(popen, {"status": "path", "path": "C:/out/r.geojson"})
    dialogs.ask_save("routes", defaultextension=".geojson")
    sent = json.loads(popen.events[-1][2]["input"])
    assert (sent["mode"], sent["defaultextension"]) == ("save", ".geojson")
    _reply(popen, {"status": "path", "path": "C:/out"})
    dialogs.ask_directory()
    assert json.loads(popen.events[-1][2]["input"])["mode"] == "directory"


def test_dialog_cancelled(popen):
    _reply(popen, {"status": "cancelled"})
    result = dialogs.ask_open()
    assert result.status == "cancelled" and result.path is None
    assert tuple(result) == (None, None)


def test_dialog_tkinter_missing_is_unavailable_with_a_reason(popen):
    _reply(popen, {"status": "unavailable",
                   "reason": "tkinter is not installed (No module named 'tkinter')"})
    result = dialogs.ask_open()
    assert result.status == "unavailable"
    assert "tkinter is not installed" in result.reason


def test_dialog_helper_crash_is_unavailable(popen):
    popen.proc = _FakeProc(popen.events, stdout="", stderr="x\nAbort trap: 6\n",
                           returncode=-6)
    result = dialogs.ask_open()
    assert result.status == "unavailable"
    assert "exit code -6" in result.reason and "Abort trap" in result.reason


def test_dialog_ignores_noise_before_the_json_line(popen):
    popen.proc = _FakeProc(
        popen.events,
        stdout='Tk deprecation warning\n{"status": "cancelled"}\n')
    assert dialogs.ask_open().status == "cancelled"


def test_dialog_helper_cannot_start(popen):
    popen.error = FileNotFoundError("python is gone")
    result = dialogs.ask_open()
    assert result.status == "unavailable" and "python is gone" in result.reason
    assert dialogs._dialog_lock.locked() is False            # noqa: SLF001


def test_dialog_timeout_kills_the_child_before_the_lock_is_released(popen):
    popen.proc = _FakeProc(popen.events, hang=True)
    result = dialogs.ask_open()
    assert result.status == "unavailable" and "left open" in result.reason
    steps = [(what, locked) for what, locked, _ in popen.events]
    assert steps == [("communicate", True),      # waited with the timeout ...
                     ("kill", True),             # ... killed, lock still held
                     ("communicate", True)]      # ... reaped, lock still held
    assert popen.events[0][2]["timeout"] == dialogs.DIALOG_TIMEOUT_S
    assert popen.proc.killed and popen.proc.returncode is not None
    assert dialogs._dialog_lock.locked() is False            # noqa: SLF001


def test_dialog_lock_is_released_after_an_unexpected_error(popen):
    class Exploding(_FakeProc):
        def communicate(self, input=None, timeout=None):      # noqa: A002
            if not self.killed:
                raise RuntimeError("interrupted")
            return "", ""

    popen.proc = Exploding(popen.events)
    with pytest.raises(RuntimeError):
        dialogs.ask_open()
    assert popen.proc.killed                                  # no window left behind
    assert dialogs._dialog_lock.locked() is False            # noqa: SLF001


def test_second_dialog_waits_for_the_first(popen):
    started, release = threading.Event(), threading.Event()

    class Slow(_FakeProc):
        def communicate(self, input=None, timeout=None):      # noqa: A002
            started.set()
            release.wait(5)
            self.returncode = 0
            return json.dumps({"status": "cancelled"}), ""

    popen.proc = Slow(popen.events)
    first = threading.Thread(target=dialogs.ask_open, daemon=True)
    first.start()
    assert started.wait(5)
    assert dialogs._dialog_lock.acquire(blocking=False) is False  # noqa: SLF001
    release.set()
    first.join(5)
    assert dialogs._dialog_lock.acquire(blocking=False) is True   # noqa: SLF001
    dialogs._dialog_lock.release()                                # noqa: SLF001


def test_worker_protocol_over_a_real_pipe_without_tkinter():
    """The real helper script, a real child process, tkinter made unimportable
    (so no Tk window can open): the JSON line parses into 'unavailable'."""
    runner = ("import runpy, sys; sys.modules['tkinter'] = None; "
              "runpy.run_path(sys.argv[1], run_name='__main__')")
    done = subprocess.run(
        [sys.executable, "-I", "-c", runner, str(dialogs._WORKER)],  # noqa: SLF001
        input=json.dumps({"mode": "open", "title": "t", "filetypes": []}),
        capture_output=True, text=True, timeout=60, check=False)
    result = dialogs._parse_reply(done.returncode, done.stdout, done.stderr)  # noqa: SLF001
    assert result.status == "unavailable"
    assert "tkinter is not installed" in result.reason


# ------------------------------------------- the worker itself (fake Tk)
class _FakeTk:
    instances: list = []

    def __init__(self):
        self.destroyed = False
        _FakeTk.instances.append(self)

    def withdraw(self):
        pass

    def attributes(self, *_a):
        pass

    def destroy(self):
        self.destroyed = True


def _install_fake_tk(monkeypatch, *, tk_error=None, answer="", boom=None):
    tk = types.ModuleType("tkinter")
    fd = types.ModuleType("tkinter.filedialog")
    _FakeTk.instances = []
    seen = {}

    def make_root():
        if tk_error is not None:
            raise tk_error
        return _FakeTk()

    def pick(name):
        def fn(**kw):
            seen[name] = kw
            if boom is not None:
                raise boom
            return answer
        return fn

    tk.Tk = make_root
    fd.askopenfilename = pick("open")
    fd.asksaveasfilename = pick("save")
    fd.askdirectory = pick("directory")
    tk.filedialog = fd
    monkeypatch.setitem(sys.modules, "tkinter", tk)
    monkeypatch.setitem(sys.modules, "tkinter.filedialog", fd)
    return seen


def test_worker_returns_the_chosen_path(monkeypatch):
    seen = _install_fake_tk(monkeypatch, answer="C:/a/b.json")
    reply = dialog_worker.handle({"mode": "open", "title": "T",
                                  "filetypes": [["Any", "*.*"]]})
    assert reply == {"status": "path", "path": "C:/a/b.json"}
    assert seen["open"]["filetypes"] == [("Any", "*.*")]
    assert _FakeTk.instances[0].destroyed


@pytest.mark.parametrize("answer", ["", ()])
def test_worker_cancel_is_silent(monkeypatch, answer):
    _install_fake_tk(monkeypatch, answer=answer)
    assert dialog_worker.handle({"mode": "save"}) == {"status": "cancelled"}
    assert _FakeTk.instances[0].destroyed


def test_worker_without_display(monkeypatch):
    _install_fake_tk(monkeypatch, tk_error=RuntimeError("no display name"))
    reply = dialog_worker.handle({"mode": "open"})
    assert reply["status"] == "unavailable" and "no display" in reply["reason"]


def test_worker_without_tkinter(monkeypatch):
    monkeypatch.setitem(sys.modules, "tkinter", None)
    reply = dialog_worker.handle({"mode": "open"})
    assert reply["status"] == "unavailable"
    assert "tkinter is not installed" in reply["reason"]


def test_worker_dialog_failure_still_destroys_the_root(monkeypatch):
    _install_fake_tk(monkeypatch, boom=RuntimeError("tcl exploded"))
    reply = dialog_worker.handle({"mode": "directory"})
    assert reply["status"] == "unavailable" and "tcl exploded" in reply["reason"]
    assert _FakeTk.instances[0].destroyed


def test_worker_unknown_mode(monkeypatch):
    _install_fake_tk(monkeypatch)
    assert dialog_worker.handle({"mode": "bogus"})["status"] == "unavailable"


# ====================================== #27 browse callback (headless harness)
_BROWSE_OUTPUTS = [{"id": ids.LOCAL_PATH, "property": "value"},
                   {"id": ids.NOTICES, "property": "data"}]


def _browse(app):
    return invoke(app, (f"{ids.LOCAL_PATH}.value", "browse-local-path"), 1,
                  outputs_list=_BROWSE_OUTPUTS,
                  triggered=[f"browse-{ids.LOCAL_PATH}.n_clicks"])


def _appended_notice(resp):
    """The notice a callback APPENDED to the notices store (a Dash Patch)."""
    patch = resp[ids.NOTICES]["data"]
    assert patch["__dash_patch_update"] == "__dash_patch_update"
    (operation,) = patch["operations"]
    assert operation["operation"] == "Append" and operation["location"] == []
    return operation["params"]["value"]


def test_browse_returns_a_notice_when_no_dialog_is_available(app, monkeypatch):
    monkeypatch.setattr(
        dialogs, "ask_open",
        lambda kind, title="": dialogs.DialogResult.unavailable(
            "no display is available (no $DISPLAY)"))
    resp = _browse(app)
    assert ids.LOCAL_PATH not in resp                  # the field is untouched
    shown = _appended_notice(resp)                     # history is kept (append)
    assert shown["severity"] == "warning"
    assert "not available" in shown["title"]
    assert "no display is available" in shown["meaning"]
    assert "type" in shown["fix"].lower() and "path" in shown["fix"].lower()
    assert shown["id"]                                 # a real Notice dict


def test_browse_cancel_is_silent(app, monkeypatch):
    monkeypatch.setattr(dialogs, "ask_open",
                        lambda kind, title="": dialogs.DialogResult.cancelled())
    resp = _browse(app)
    assert ids.LOCAL_PATH not in resp and ids.NOTICES not in resp


def test_browse_fills_the_field_without_a_notice(app, monkeypatch):
    monkeypatch.setattr(
        dialogs, "ask_open",
        lambda kind, title="": dialogs.DialogResult.chosen("C:/data/x.gpkg"))
    resp = _browse(app)
    assert resp[ids.LOCAL_PATH]["value"] == "C:/data/x.gpkg"
    assert ids.NOTICES not in resp


def test_every_browse_button_can_report_unavailable(app, monkeypatch):
    from pyorps.gui.callbacks import browse

    unavailable = dialogs.DialogResult.unavailable("no display")
    for name in ("ask_open", "ask_save", "ask_directory"):
        monkeypatch.setattr(dialogs, name, lambda *a, **k: unavailable)
    for target, *_ in browse.BROWSE_TARGETS:
        resp = invoke(app, (f"{target}.value", browse.browse_button_id(target)),
                      1,
                      outputs_list=[{"id": target, "property": "value"},
                                    {"id": ids.NOTICES, "property": "data"}],
                      triggered=[f"{browse.browse_button_id(target)}.n_clicks"])
        assert _appended_notice(resp)["title"].startswith("The file dialog")


# ============================================ #1 layout is a function (F5)
def test_layout_is_a_function_returning_a_fresh_layout_each_time(app, state):
    assert callable(app.layout)
    first, second = app.layout(), app.layout()
    assert first is not second
    found = {getattr(c, "id", None) for c in first._traverse()}   # noqa: SLF001
    for component_id in (ids.MAP, ids.LAYER_HOST, ids.NOTICES, ids.LAYERS_VIEW):
        assert component_id in found


def test_layout_call_resets_the_render_signature(app, state):
    state._host_sig = ("rendered before the reload",)          # noqa: SLF001
    app.layout()
    assert state._host_sig is None                              # noqa: SLF001


def test_page_load_route_resets_the_signature(app, state):
    """The real page-load path: Dash serves /_dash-layout (Flask test client,
    no network) and that evaluates the layout function."""
    state._host_sig = ("stale",)                                # noqa: SLF001
    response = app.server.test_client().get("/_dash-layout")
    assert response.status_code == 200
    assert ids.LAYER_HOST in response.get_data(as_text=True)
    assert state._host_sig is None                              # noqa: SLF001


def test_reload_renders_the_map_again(app, state):
    """F5: the second render with an unchanged state is skipped, but after the
    layout function ran (a reload) the host is shipped again."""
    state.add_layer("Square", "vector", geojson=SQUARE,
                    style={"color": "#00aa88"})
    wiring = ("layer-host.children", ids.LAYERS_VIEW)
    first = invoke(app, wiring, [])
    assert first[ids.LAYER_HOST]["children"]
    with pytest.raises(PreventUpdate):
        invoke(app, wiring, [])                 # unchanged: skipped (perf)
    app.layout()                                # browser reload
    again = invoke(app, wiring, [])
    assert again[ids.LAYER_HOST]["children"] == first[ids.LAYER_HOST]["children"]


def test_validation_layout_is_set_for_callback_validation(app):
    ids_in_validation = {getattr(c, "id", None)
                         for c in app.validation_layout._traverse()}  # noqa: SLF001
    assert {ids.MAP, ids.LAYER_HOST, ids.NOTICES} <= ids_in_validation


def test_traverse_hook_still_pre_seeds_every_later_layout(app):
    forced = {"id": "forced-1", "severity": "error", "title": "Forced"}
    for component in app.layout._traverse():                     # noqa: SLF001
        if getattr(component, "id", None) == ids.NOTICES:
            component.data = [forced]
        elif getattr(component, "id", None) == ids.MAP:
            component.zoom = 11
    for _ in range(2):                          # every page load, not just one
        layout = app.layout()
        by_id = {getattr(c, "id", None): c for c in layout._traverse()}  # noqa: SLF001
        assert by_id[ids.NOTICES].data == [forced]
        assert by_id[ids.MAP].zoom == 11
        assert by_id[ids.LAYERS_VIEW].data == []        # untouched stays default


# ======================================================= #32 exit handling
def test_exit_report_stops_tiles_and_names_the_folder_without_deleting(
        tmp_path, capsys, monkeypatch):
    state = ProjectState(work_dir=tmp_path / "work")
    (state.work_dir / "cost.tif").write_bytes(b"x" * 3000)
    (state.work_dir / "sub").mkdir()
    (state.work_dir / "sub" / "dem.tif").write_bytes(b"y" * 2000)
    stopped = []
    monkeypatch.setattr(state, "shutdown", lambda: stopped.append(True))
    app_module._exit_report(state)                              # noqa: SLF001
    out = capsys.readouterr().out
    assert stopped == [True]
    assert str(state.work_dir) in out and "4.9 KB" in out
    assert "delete it yourself" in out
    assert "Nothing was deleted" in out
    assert (state.work_dir / "cost.tif").exists()               # nothing removed
    assert (state.work_dir / "sub" / "dem.tif").exists()


def test_exit_report_survives_a_failing_shutdown(tmp_path, capsys, monkeypatch):
    state = ProjectState(work_dir=tmp_path / "work")

    def fail():
        raise RuntimeError("tile server stuck")

    monkeypatch.setattr(state, "shutdown", fail)
    app_module._exit_report(state)                              # noqa: SLF001
    out = capsys.readouterr().out
    assert "tile server stuck" in out and str(state.work_dir) in out


def test_exit_report_with_a_missing_folder_is_silent(tmp_path, capsys,
                                                     monkeypatch):
    state = ProjectState(work_dir=tmp_path / "work")
    state.work_dir = tmp_path / "gone"
    monkeypatch.setattr(state, "shutdown", lambda: None)
    app_module._exit_report(state)                              # noqa: SLF001
    assert capsys.readouterr().out == ""


def test_human_size():
    assert app_module._human_size(0) == "0 B"                   # noqa: SLF001
    assert app_module._human_size(1536) == "1.5 KB"             # noqa: SLF001
    assert app_module._human_size(5 * 1024 ** 2) == "5.0 MB"    # noqa: SLF001
    assert app_module._human_size(3 * 1024 ** 3) == "3.0 GB"    # noqa: SLF001


def test_launch_registers_the_exit_report(monkeypatch, tmp_path):
    registered = _stub_launch(monkeypatch, [])
    state = ProjectState(work_dir=tmp_path / "w")
    app_module.launch(state, port=9002)
    assert (app_module._exit_report, (state,)) in registered    # noqa: SLF001


def test_building_the_app_registers_no_exit_hook(monkeypatch, state):
    registered = []
    monkeypatch.setattr(app_module.atexit, "register",
                        lambda fn, *a: registered.append(fn))
    app_module.build_app(state)
    assert registered == []
