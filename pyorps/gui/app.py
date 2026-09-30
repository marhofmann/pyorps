"""
PYORPS GUI: app factory and launcher.

``build_app(state)`` creates the Dash app around a single server-side
:class:`~pyorps.gui.state.ProjectState`; ``launch()`` runs it in the browser
or (with ``desktop=True``) inside a pywebview window.
"""
from __future__ import annotations

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
    app.layout = build_layout(state)
    register_all(app, state)
    _register_geojson_route(app, state)
    # test/debug access to the server-side state
    app._pyorps_state = state
    return app


def _has_flask_compress() -> bool:
    try:
        import flask_compress  # noqa: F401
        return True
    except ImportError:
        return False


def _json_bytes(payload: Any) -> bytes:
    """orjson when available (~5-10x stdlib for multi-MB GeoJSON)."""
    try:
        import orjson
        return orjson.dumps(payload)
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
            layer._geobuf_bytes = geobuf.encode(layer.geojson)
            layer._geobuf_rev = layer.geojson_rev
        response = Response(layer._geobuf_bytes,
                            mimetype="application/octet-stream")
        return _finish(response, layer)


def _resolve_port(host: str, port: int) -> int:
    """Refuse to double-bind a busy port (Windows SO_REUSEADDR silently
    allows two servers on one port and the OLD one keeps the traffic).

    If something already answers on ``port``, pick the next free one and say
    so loudly — otherwise a stale server (e.g. yesterday's viewer) hijacks
    the browser while this process believes it is serving.
    """
    import socket

    probe_port = port
    for _ in range(20):
        with socket.socket() as probe:
            probe.settimeout(0.5)
            if probe.connect_ex((host, probe_port)) != 0:
                break
        probe_port += 1
    if probe_port != port:
        print(f"WARNING: port {port} is already in use by another server "
              f"(a stale PYORPS viewer?). Serving on port {probe_port} "
              f"instead -> http://{host}:{probe_port}")
    return probe_port


def _serve(app: Dash, host: str, port: int, debug: bool) -> None:
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
    print(f"PYORPS GUI (waitress) -> http://{host}:{port}")
    serve(app.server, host=host, port=port, threads=8, ident="pyorps-gui")


def launch(state: ProjectState | None = None, *, host: str = "127.0.0.1",
           port: int = 8050, debug: bool = False,
           desktop: bool = False) -> None:
    """Run the GUI. ``desktop=True`` wraps it in a pywebview window."""
    from .services import logbook

    state = state or ProjectState()
    # desktop = loopback-only: skip response compression (CPU for nothing)
    app = build_app(state, compress=not desktop)
    print(f"PYORPS GUI log: {logbook.log_file_path()}")
    port = _resolve_port(host, port)
    if not desktop:
        _serve(app, host, port, debug)
        return

    import threading

    import webview

    server = threading.Thread(
        target=lambda: _serve(app, host, port, False),
        daemon=True)
    server.start()
    window = webview.create_window("PYORPS GUI", f"http://{host}:{port}",
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
        except Exception:      # dialog unavailable — never trap the user
            return True

    try:
        window.events.closing += _confirm_close
    except Exception:          # older pywebview without the closing event  # nosec B110
        pass
    webview.start()
