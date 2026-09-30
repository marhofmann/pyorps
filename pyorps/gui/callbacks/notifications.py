"""
PYORPS GUI callbacks: the notification stack (R14, Section 21).

One render callback maps the ``notices`` store to a stack of severity-coloured
toast boxes (title / Means / Impact / Fix / collapsible Details / optional
"Go fix" button that jumps to the offending tab). Dismissing a toast prunes it
from the store; warnings/info/success auto-dismiss client-side via
``duration`` but stay in the store history until dismissed or replaced.
"""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, ctx, html, no_update
from dash.exceptions import PreventUpdate

from .. import ids
from ..services import logbook

_COLORS = {"error": "danger", "warning": "warning", "info": "info",
           "success": "success"}
_ICONS = {"error": "danger", "warning": "warning", "info": "info",
          "success": "success"}
#: auto-dismiss (ms): warnings 5 s, errors 10 s (user request); info/success
#: shorter. Every notice is ALSO written to the persistent log (logbook), so a
#: dismissed toast is never lost.
_DURATIONS = {"error": 10_000, "warning": 5_000, "info": 5_000,
              "success": 4_000}


def notice_toast(notice: dict) -> dbc.Toast:
    """Render one notice dict (services.errors.Notice.to_dict) as a Toast."""
    severity = notice.get("severity", "error")
    body = []
    for label, key in (("Means", "meaning"), ("Impact", "impact"),
                       ("Fix", "fix")):
        text = notice.get(key)
        if text:
            body.append(html.Div([html.B(f"{label}: "), text],
                                 className="small"))
    if notice.get("focus_id") or notice.get("focus_control"):
        body.append(dbc.Button(
            "Go fix →", size="sm", color=_COLORS.get(severity, "secondary"),
            outline=True, className="mt-1 me-2",
            id={"type": ids.TYPE_NOTICE_FIX,
                "tab": notice.get("focus_id") or "",
                "control": notice.get("focus_control") or "",
                "index": notice.get("id", "")}))
    if notice.get("details"):
        body.append(html.Details([
            html.Summary("Details", className="small text-muted"),
            html.Pre(notice["details"], className="notice-details"),
        ], className="mt-1"))
    return dbc.Toast(
        body, header=notice.get("title", "Notice"),
        icon=_ICONS.get(severity, "danger"),
        duration=_DURATIONS.get(severity),
        dismissable=True, is_open=True,
        className=f"notice-{severity}",
        id={"type": ids.TYPE_NOTICE, "index": notice.get("id", "")})


def register(app, state) -> None:
    #: ids of notices already written to the log (one write per notice)
    logged_ids: set[str] = set()

    @app.callback(Output(ids.NOTICE_STACK, "children"),
                  Input(ids.NOTICES, "data"))
    def render_notices(notices):
        # persist every new notice to the log file before it is rendered, so a
        # warning/error survives its 5-10 s toast and stays readable later.
        for notice in notices or []:
            nid = notice.get("id")
            if nid and nid not in logged_ids:
                logged_ids.add(nid)
                try:
                    logbook.log_notice(notice)
                except Exception:  # pragma: no cover - logging must never break UI
                    pass
        return [notice_toast(n) for n in (notices or [])]

    # ------------------------------------------------- the full-log viewer
    @app.callback(
        Output(ids.LOG_OFFCANVAS, "is_open"),
        Output(ids.LOG_CONTENT, "children"),
        Output(ids.LOG_PATH_INFO, "children"),
        Input(ids.LOG_VIEW_BTN, "n_clicks"),
        Input(ids.LOG_REFRESH_BTN, "n_clicks"),
        State(ids.LOG_OFFCANVAS, "is_open"),
        prevent_initial_call=True)
    def show_log(view_clicks, refresh_clicks, is_open):
        trigger = ctx.triggered_id
        if trigger == ids.LOG_VIEW_BTN and not view_clicks:
            raise PreventUpdate
        # the header button toggles; refresh keeps it open and reloads
        open_now = (not is_open) if trigger == ids.LOG_VIEW_BTN else True
        if not open_now:
            return False, no_update, no_update
        return (True, logbook.read_log(),
                f"Saved to {logbook.log_file_path()} "
                f"(kept ~{logbook.BACKUP_DAYS} days).")

    @app.callback(Output(ids.NOTICES, "data", allow_duplicate=True),
                  Input({"type": ids.TYPE_NOTICE, "index": ALL}, "n_dismiss"),
                  State(ids.NOTICES, "data"),
                  prevent_initial_call=True)
    def dismiss_notice(n_dismiss, notices):
        trigger = ctx.triggered_id
        if not trigger or not any(n_dismiss or []):
            raise PreventUpdate
        dismissed = trigger["index"]
        return [n for n in (notices or []) if n.get("id") != dismissed]

    @app.callback(Output(ids.TABS, "active_tab", allow_duplicate=True),
                  Output(ids.FOCUS_FLASH, "data"),
                  Input({"type": ids.TYPE_NOTICE_FIX, "tab": ALL,
                         "control": ALL, "index": ALL}, "n_clicks"),
                  prevent_initial_call=True)
    def go_fix(n_clicks):
        from dash import no_update

        trigger = ctx.triggered_id
        if not trigger or not any(n_clicks or []):
            raise PreventUpdate
        tab = trigger.get("tab") or no_update
        control = trigger.get("control") or ""
        flash = ({"control": control, "seq": sum(filter(None, n_clicks))}
                 if control else no_update)
        return tab, flash

    # scroll to + flash the offending control (works even when the target
    # tab is already active — switching tabs alone is invisible then)
    app.clientside_callback(
        """
        function(data) {
            if (!data || !data.control) {
                return window.dash_clientside.no_update;
            }
            setTimeout(function() {
                var el = document.getElementById(data.control);
                if (el) {
                    el.scrollIntoView({behavior: 'smooth',
                                       block: 'center'});
                    el.classList.add('pyorps-flash');
                    setTimeout(function() {
                        el.classList.remove('pyorps-flash');
                    }, 2400);
                }
            }, 250);  // give the tab switch a moment to render
            return window.dash_clientside.no_update;
        }
        """,
        Output(ids.FOCUS_FLASH, "data", allow_duplicate=True),
        Input(ids.FOCUS_FLASH, "data"),
        prevent_initial_call=True)
