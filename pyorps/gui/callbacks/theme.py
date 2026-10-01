"""
PYORPS GUI callbacks: app-shell chrome — theme toggle + sidebar collapse.

Pure client-side: flipping Bootstrap's ``data-bs-theme`` and collapsing the
sidebar never need a server round-trip. The chosen theme is persisted to
``localStorage`` (``pyorps-theme``) and re-applied before first paint by the
inline script in :data:`pyorps.gui.app._INDEX_STRING`. Both callbacks fire a
window ``resize`` so Leaflet re-measures the map after the layout changes.
"""
from __future__ import annotations

from dash import Input, Output, State

from .. import ids


def register(app, state) -> None:
    # ------------------------------------------------- light / dark toggle
    app.clientside_callback(
        """
        function(n_clicks) {
            if (!n_clicks) { return window.dash_clientside.no_update; }
            var root = document.documentElement;
            var next = root.getAttribute("data-bs-theme") === "dark"
                ? "light" : "dark";
            root.setAttribute("data-bs-theme", next);
            try { window.localStorage.setItem("pyorps-theme", next); }
            catch (e) { /* storage unavailable — theme still applies */ }
            window.dispatchEvent(new Event("resize"));
            return next;
        }
        """,
        Output(ids.THEME_STORE, "data"),
        Input(ids.THEME_TOGGLE, "n_clicks"),
        prevent_initial_call=True)

    # ------------------------------------------------- sidebar collapse
    app.clientside_callback(
        """
        function(n_clicks, cls) {
            if (!n_clicks) { return window.dash_clientside.no_update; }
            cls = cls || "";
            var collapsed = cls.indexOf("sidebar-collapsed") >= 0;
            var next = collapsed
                ? cls.replace(/\\s*sidebar-collapsed/g, "")
                : (cls + " sidebar-collapsed");
            // let the CSS transition finish before Leaflet re-measures
            setTimeout(function () {
                window.dispatchEvent(new Event("resize"));
            }, 320);
            return next;
        }
        """,
        Output(ids.APP_SHELL, "className"),
        Input(ids.SIDEBAR_TOGGLE, "n_clicks"),
        State(ids.APP_SHELL, "className"),
        prevent_initial_call=True)
