"""
PYORPS GUI: app factory and launcher.

``build_app(state)`` creates the Dash app around a single server-side
:class:`~pyorps.gui.state.ProjectState`; ``launch()`` runs it in the browser
or (with ``desktop=True``) inside a pywebview window.
"""
from __future__ import annotations

import atexit
import errno
import ipaddress
import json
import os
import socket
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dash import Dash

from .callbacks import register_all
from .layout import build_layout
from .state import ProjectState

#: Custom index: Bootstrap 5.3 colour modes hang off ``data-bs-theme`` on the
#: root element. The inline script applies the persisted choice BEFORE first
#: paint (no light-flash on reload); the header toggle updates it afterwards.
#: Bootstrap itself is vendored into ``assets/00-bootstrap.min.css`` (assets
#: load alphabetically, so gui.css overrides it) — the app is fully offline.
_INDEX_STRING = """<!DOCTYPE html>
<html data-bs-theme="light">
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <script>
        (function () {
            try {
                var t = window.localStorage.getItem("pyorps-theme");
                if (t === "dark" || t === "light") {
                    document.documentElement.setAttribute(
                        "data-bs-theme", t);
                }
            } catch (e) { /* private mode etc. — default to light */ }
        })();
        </script>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""


class PageLayout:
    """The app layout as a callable: Dash evaluates it on EVERY page load.

    A static layout is built once, so after F5 (or a second tab, or a pywebview
    reload) the browser got the blank initial stores while the server-side
    ``ProjectState`` still held every layer, and ``render_host`` skipped its
    first render because the host signature was unchanged: full layer list,
    empty map. Each evaluation therefore (1) forgets the last rendered host
    signature, so the initial ``render_host`` call of the new page always
    ships the host, and (2) returns a fresh ``build_layout(state)``.

    ``_traverse`` keeps tools and tests that inspect or pre-seed the layout
    working (``for c in app.layout._traverse(): c.data = ...``): it walks a
    pinned layout, and every property ASSIGNED on one of its components is
    copied into each layout returned afterwards. (Assign; in-place changes of
    a list or dict are not seen.)
    """

    def __init__(self, state: ProjectState) -> None:
        self._state = state
        self._pinned: Any = None
        self._baseline: list[tuple[Any, str, Any]] = []

    def __call__(self) -> Any:
        self._state._host_sig = None  # pylint: disable=protected-access  # view cache owned by callbacks/layers.py
        layout = build_layout(self._state)
        if self._pinned is not None:
            self._copy_pinned_edits(layout)
        return layout

    def _traverse(self):
        if self._pinned is None:
            self._pinned = build_layout(self._state)
            self._baseline = [
                (comp, prop, getattr(comp, prop, None))
                for comp in self._with_root(self._pinned)
                if getattr(comp, "id", None) is not None
                for prop in comp._prop_names]  # pylint: disable=protected-access
        return self._pinned._traverse()  # pylint: disable=protected-access

    @staticmethod
    def _with_root(layout: Any) -> list[Any]:
        return [layout, *layout._traverse()]  # pylint: disable=protected-access

    @staticmethod
    def _key(component_id: Any) -> str:
        return json.dumps(component_id, sort_keys=True, default=str)

    def _copy_pinned_edits(self, layout: Any) -> None:
        fresh = {self._key(comp.id): comp for comp in self._with_root(layout)
                 if getattr(comp, "id", None) is not None}
        for comp, prop, original in self._baseline:
            current = getattr(comp, prop, None)
            target = fresh.get(self._key(comp.id))
            if current is not original and target is not None:
                setattr(target, prop, current)


def build_app(state: ProjectState | None = None, *,
              compress: bool | None = None, **dash_kwargs: Any) -> Dash:
    """Create the Dash app; the state is captured by callback closures.

    ``compress=None`` (default) enables response compression whenever
    flask-compress is installed — a large-layer callback response measured
    18 MB uncompressed shrinks to a fraction on the wire. Pass
    ``compress=False`` for loopback-only serving (desktop/pywebview), where
    bandwidth is free and compression only costs CPU.
    """
    state = state or ProjectState()
    if compress is None:
        compress = _has_flask_compress()
    app = Dash(
        __name__,
        title="PYORPS GUI",
        index_string=_INDEX_STRING,
        compress=bool(compress and _has_flask_compress()),
        **dash_kwargs,
    )
    # NB: flask-compress caches its mimetype set when Dash constructs it, so
    # the /_gj route serves plain "application/json" (already on the default
    # compress list) rather than "application/geo+json".
    # a function layout: re-evaluated on every page load (see PageLayout). Dash
    # evaluates it once on assignment and keeps that value as the
    # ``validation_layout`` for callback-id validation.
    app.layout = PageLayout(state)
    register_all(app, state)
    _register_geojson_route(app, state)
    # test/debug access to the server-side state
    app._pyorps_state = state  # pylint: disable=protected-access  # test/debug hook
    return app


def _has_flask_compress() -> bool:
    try:
        import flask_compress  # noqa: F401  # pylint: disable=unused-import
        return True
    except ImportError:
        return False


def _json_bytes(payload: Any) -> bytes:
    """orjson when available (~5-10x stdlib for multi-MB GeoJSON)."""
    try:
        import orjson
        return orjson.dumps(payload)  # pylint: disable=no-member  # optional dependency
    except ImportError:
        import json
        return json.dumps(payload).encode("utf-8")


def _register_geojson_route(app: Dash, state: ProjectState) -> None:
    """Serve a vector layer's WGS84 GeoJSON at ``/_gj/<id>`` and the same
    data geobuf-encoded (binary protobuf, far smaller + faster to parse in
    the browser) at ``/_gb/<id>`` (perf: large layers).

    Big vector layers are rendered as ``dl.GeoJSON(url=…)`` instead of inline
    ``data=…`` so their (multi-MB) payload is fetched ONCE by the browser and
    is NOT re-serialized into every ``render_host`` callback response. The URL
    carries a ``?v=<rev>`` cache-buster that changes whenever the layer's
    GeoJSON is reassigned; ETag/304 handling covers reloads on top of that.
    The geobuf encoding is cached per layer revision.
    """
    from flask import Response, abort, request

    def _etag(layer) -> str:
        return f'"{layer.id}-{layer.geojson_rev}"'

    def _cached(layer) -> Response | None:
        if request.if_none_match and _etag(layer).strip('"') in \
                request.if_none_match:
            return Response(status=304)
        return None

    def _finish(response: Response, layer) -> Response:
        response.headers["Cache-Control"] = "public, max-age=3600"
        response.set_etag(_etag(layer).strip('"'))
        return response

    @app.server.route("/_gj/<layer_id>")
    def _serve_geojson(layer_id):                       # noqa: ANN202
        layer = state.get(layer_id)
        if layer is None or layer.geojson is None:
            abort(404)
        cached = _cached(layer)
        if cached is not None:
            return cached
        # plain application/json so flask-compress (whose mimetype set is
        # frozen at Dash construction) compresses it in browser mode
        response = Response(_json_bytes(layer.geojson),
                            mimetype="application/json")
        return _finish(response, layer)

    @app.server.route("/_gb/<layer_id>")
    def _serve_geobuf(layer_id):                        # noqa: ANN202
        import geobuf

        layer = state.get(layer_id)
        if layer is None or layer.geojson is None:
            abort(404)
        cached = _cached(layer)
        if cached is not None:
            return cached
        # encode once per geojson revision (encoding is the expensive part)
        if getattr(layer, "_geobuf_rev", None) != layer.geojson_rev:
            layer._geobuf_bytes = geobuf.encode(layer.geojson)  # pylint: disable=protected-access  # per-layer cache
            layer._geobuf_rev = layer.geojson_rev  # pylint: disable=protected-access  # per-layer cache
        response = Response(layer._geobuf_bytes,  # pylint: disable=protected-access  # per-layer cache
                            mimetype="application/octet-stream")
        return _finish(response, layer)


class LaunchError(RuntimeError):
    """The GUI cannot start: refused host, no free port, server never listened."""


# ---------------------------------------------------------------------------
# host policy (the path fields read and write ANY file this user can reach, and
# there is no authentication: only loopback is allowed unless --allow-remote)
# ---------------------------------------------------------------------------
def _bare(host: str) -> str:
    return (host or "").strip().strip("[]").lower()


def is_loopback_host(host: str) -> bool:
    """True for ``127.0.0.1`` (any 127/8 address), ``localhost`` and ``::1``."""
    name = _bare(host)
    if name == "localhost":
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def is_wildcard_host(host: str) -> bool:
    """True for ``0.0.0.0``, ``::`` and the empty host (listen everywhere)."""
    name = _bare(host)
    if not name:
        return True
    try:
        return ipaddress.ip_address(name).is_unspecified
    except ValueError:
        return False


def check_host(host: str, allow_remote: bool = False) -> bool:
    """Refuse a non-loopback ``host`` unless ``allow_remote``.

    Returns True when the host is remote AND allowed (the caller then prints
    :func:`remote_warning`), False for loopback; raises :class:`LaunchError`
    when a remote host is refused.
    """
    if is_loopback_host(host):
        return False
    if not allow_remote:
        raise LaunchError(
            f"refusing to listen on {host!r}: the GUI has no authentication and "
            "its path fields read and write any file this user can access, so "
            "anyone who can reach the port could use them. Use --host 127.0.0.1 "
            "(the default), or add --allow-remote if you really mean to expose "
            "it (trusted network only).")
    return True


def remote_warning(host: str, port: int, debug: bool = False) -> str:
    """The warning printed when the GUI is deliberately exposed."""
    lines = [
        f"WARNING: --allow-remote: the GUI listens on {host}:{port} and is open "
        "to every machine that can reach that port.",
        " - There is NO authentication: anybody who connects can read and write "
        "any file this user can access through the path fields.",
        " - File paths refer to the SERVER machine, and native file dialogs "
        "open on the server, not in the remote browser.",
        " - Raster tiles are served from the server's 'localhost' and will not "
        "load in a remote browser.",
    ]
    if debug:
        lines.append(" - --debug also exposes the Werkzeug debugger on the "
                     "network. Do not combine it with --allow-remote.")
    return "\n".join(lines)


def _browse_url(host: str, port: int) -> str:
    """URL a browser on THIS machine opens (wildcard -> localhost, IPv6 bracketed)."""
    name = "localhost" if is_wildcard_host(host) else _bare(host)
    return f"http://[{name}]:{port}" if ":" in name else f"http://{name}:{port}"


# ---------------------------------------------------------------------------
# port probing
# ---------------------------------------------------------------------------
_BUSY_ERRNOS = frozenset(
    getattr(errno, name) for name in
    ("EADDRINUSE", "EACCES", "WSAEADDRINUSE", "WSAEACCES") if hasattr(errno, name))


def _connect_host(host: str) -> str:
    """Address to CONNECT to for ``host``: a wildcard cannot be connected to
    (``connect_ex(("0.0.0.0", p))`` never succeeds on Windows), so probe loopback."""
    if is_wildcard_host(host):
        return "::1" if ":" in _bare(host) else "127.0.0.1"
    return _bare(host)


def _accepts_connections(host: str, port: int, timeout: float = 0.5) -> bool:
    """Does something already answer on ``host:port``? (wildcard: on loopback)"""
    try:
        infos = socket.getaddrinfo(_connect_host(host), port,
                                   type=socket.SOCK_STREAM)
    except socket.gaierror:
        return False
    for family, kind, proto, _canon, sockaddr in infos:
        try:
            with socket.socket(family, kind, proto) as probe:
                probe.settimeout(timeout)
                if probe.connect_ex(sockaddr) == 0:
                    return True
        except OSError:
            continue
    return False


def _cannot_bind(host: str, port: int) -> bool:
    """Is ``host:port`` taken, judged by trying to bind it (without listening)?

    Catches listeners a connect probe misses (a socket on another interface,
    one that does not accept). No SO_REUSEADDR on Windows, where it would let
    the probe share a busy port; on POSIX it only lets the probe ignore
    TIME_WAIT leftovers (a listening socket still blocks it).
    """
    try:
        infos = socket.getaddrinfo(_bare(host) or None, port,
                                   type=socket.SOCK_STREAM,
                                   flags=socket.AI_PASSIVE)
    except socket.gaierror:
        return False
    for family, kind, proto, _canon, sockaddr in infos:
        try:
            probe = socket.socket(family, kind, proto)
        except OSError:     # e.g. IPv6 not available here: nothing to learn
            continue
        with probe:
            if os.name != "nt":
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(sockaddr)
            except OSError as exc:
                if exc.errno in _BUSY_ERRNOS:
                    return True
    return False


def _port_in_use(host: str, port: int) -> bool:
    return _accepts_connections(host, port) or _cannot_bind(host, port)


def _resolve_port(host: str, port: int, attempts: int = 20) -> int:
    """Refuse to double-bind a busy port (Windows SO_REUSEADDR silently
    allows two servers on one port and the OLD one keeps the traffic).

    If something already uses ``port``, pick the next free one and say so
    loudly — otherwise a stale server (e.g. yesterday's viewer) hijacks the
    browser while this process believes it is serving. A wildcard host
    (``0.0.0.0``) is probed on loopback, and every candidate is also bind-tested.
    """
    for candidate in range(port, min(port + attempts, 65536)):
        if not _port_in_use(host, candidate):
            break
    else:       # every candidate is busy (or none is left below 65536)
        raise LaunchError(f"no free port found from {port} upwards; "
                          "stop the other server or pick --port.")
    if candidate != port:
        print(f"WARNING: port {port} is already in use by another server "
              f"(a stale PYORPS viewer?). Serving on port {candidate} "
              f"instead -> {_browse_url(host, candidate)}")
    return candidate


def _wait_for_server(host: str, port: int, *, timeout: float = 30.0,
                     interval: float = 0.1,
                     server: threading.Thread | None = None,
                     probe: Callable[[str, int], bool] = _accepts_connections,
                     clock: Callable[[], float] = time.monotonic,
                     sleep: Callable[[float], None] = time.sleep) -> None:
    """Block until ``host:port`` accepts connections (desktop mode: the window
    must not open on a port nobody listens on yet = blank page).

    Raises :class:`LaunchError` when the server thread died or ``timeout``
    seconds pass. ``probe``/``clock``/``sleep`` are injectable for tests.
    """
    deadline = clock() + timeout
    while not probe(host, port):
        if server is not None and not server.is_alive():
            raise LaunchError(
                f"the GUI server stopped before it listened on "
                f"{_browse_url(host, port)} (see the error above).")
        if clock() >= deadline:
            raise LaunchError(
                f"the GUI server did not start listening on "
                f"{_browse_url(host, port)} within {timeout:.0f} s.")
        sleep(interval)


# ---------------------------------------------------------------------------
# serving + exit handling
# ---------------------------------------------------------------------------
def _serve(app: Dash, host: str, port: int, debug: bool,
           threads: int = 8) -> None:
    """Serve the app: waitress (production WSGI, parallel threads for
    simultaneous callback + GeoJSON/tile traffic) when available, the
    Flask dev server for ``debug`` or as fallback."""
    if debug:
        app.run(host=host, port=port, debug=True)  # nosec B201 - development server, only when debug mode is requested
        return
    try:
        from waitress import serve
    except ImportError:
        app.run(host=host, port=port)
        return
    print(f"PYORPS GUI (waitress) -> {_browse_url(host, port)}")
    serve(app.server, host=host, port=port, threads=threads, ident="pyorps-gui")


def _folder_size(folder: Path) -> int:
    """Total bytes of the regular files under ``folder`` (read-only walk)."""
    total = 0
    for root, _dirs, files in os.walk(folder):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                continue
    return total


def _human_size(size: float) -> str:
    size = float(size)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _exit_report(state: ProjectState) -> None:
    """``atexit``: stop the tile servers, then say where the session files are.

    Deliberately deletes NOTHING (recursive deletes are not ours to do); the
    work folder keeps every cost raster / DEM / combined raster of the session.
    """
    try:
        state.shutdown()
    except Exception as exc:  # pylint: disable=broad-exception-caught  # never fail at exit
        print(f"PYORPS GUI: could not stop all tile servers cleanly ({exc}).")
    try:
        work_dir = Path(state.work_dir)
        if work_dir.is_dir():
            print(f"PYORPS GUI work folder: {work_dir} "
                  f"({_human_size(_folder_size(work_dir))}). It holds this "
                  "session's rasters and is not needed once you have saved your "
                  "project; delete it yourself when you no longer want them. "
                  "Nothing was deleted.")
    except Exception:  # nosec B110  # pylint: disable=broad-exception-caught
        pass


def launch(state: ProjectState | None = None, *, host: str = "127.0.0.1",
           port: int = 8050, debug: bool = False, desktop: bool = False,
           allow_remote: bool = False, threads: int = 8,
           compress: bool | None = None) -> None:
    """Run the GUI. ``desktop=True`` wraps it in a pywebview window.

    ``host`` must be a loopback address unless ``allow_remote`` (no
    authentication; see :func:`remote_warning`). ``threads`` is the waitress
    worker count; ``compress=None`` keeps the default (on in the browser when
    flask-compress is installed, off for the desktop window).
    """
    from .services import logbook

    remote = check_host(host, allow_remote)         # before any state exists
    state = state or ProjectState()
    atexit.register(_exit_report, state)
    # desktop = loopback-only: skip response compression (CPU for nothing)
    app = build_app(state, compress=(not desktop) if compress is None else compress)
    print(f"PYORPS GUI log: {logbook.log_file_path()}")
    port = _resolve_port(host, port)
    if remote:
        print(remote_warning(host, port, debug), file=sys.stderr)
    if not desktop:
        _serve(app, host, port, debug, threads)
        return

    try:
        import webview
    except ImportError as exc:
        raise LaunchError("the desktop window needs pywebview: "
                          'pip install "pyorps[gui]" (or pip install pywebview).') from exc

    server = threading.Thread(
        target=lambda: _serve(app, host, port, False, threads),
        daemon=True)
    server.start()
    _wait_for_server(host, port, server=server)
    window = webview.create_window("PYORPS GUI", _browse_url(host, port),
                                   width=1400, height=900)

    def _confirm_close():
        """Save-before-closing guard: block the close while work is unsaved
        unless the user explicitly confirms (returning False cancels)."""
        if not getattr(state, "dirty", False):
            return True
        try:
            return bool(window.create_confirmation_dialog(
                "Unsaved changes",
                "The cost raster, routes or project have NOT been saved.\n"
                "Close anyway and lose them?\n\n"
                "(Cancel to go back and use Data tab → Project → "
                "Save project.)"))
        except Exception:      # dialog unavailable — never trap the user  # pylint: disable=broad-exception-caught
            return True

    try:
        window.events.closing += _confirm_close
    except Exception:  # nosec B110  # pylint: disable=broad-exception-caught
        # older pywebview without the closing event
        pass
    webview.start()
