"""
PYORPS GUI callbacks: the notification stack (R14, Section 21).

One render callback maps the ``notices`` store to a stack of severity-coloured
toast boxes (title / Means / Impact / Fix / collapsible Details / optional
"Go fix" button that jumps to the offending tab). Dismissing a toast prunes it
from the store; warnings/info/success auto-dismiss client-side via
``duration`` but stay in the store history (capped at the newest
``errors.MAX_STORED_NOTICES``) until dismissed or replaced. Only notices that
are NEW since the previous render open as toasts; an older one stays open only
while its own auto-dismiss timer is running, so an earlier warning never pops
up again.
"""
from __future__ import annotations

import time

import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, ctx, html, no_update
from dash.exceptions import PreventUpdate

from .. import ids
from ..services import errors, logbook

_COLORS = {"error": "danger", "warning": "warning", "info": "info",
           "success": "success"}
_ICONS = {"error": "danger", "warning": "warning", "info": "info",
          "success": "success"}
#: auto-dismiss (ms): warnings 5 s, errors 10 s (user request); info/success
#: shorter. Every notice is ALSO written to the persistent log (logbook), so a
#: dismissed toast is never lost.
_DURATIONS = {"error": 10_000, "warning": 5_000, "info": 5_000,
              "success": 4_000}
#: monotonic clock for the toast timers (a module attribute so tests can drive it)
_clock = time.monotonic


def notice_toast(notice: dict, is_open: bool = True) -> dbc.Toast:
    """Render one notice dict (services.errors.Notice.to_dict) as a Toast.

    ``is_open=False`` renders it closed (an old notice kept in the history).
    """
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
        dismissable=True, is_open=is_open,
        className=f"notice-{severity}",
        id={"type": ids.TYPE_NOTICE, "index": notice.get("id", "")})


def _still_showing(first_shown: dict[str, float], notice: dict,
                   now: float) -> bool:
    """Whether a notice's toast should be open in this render.

    A NEW notice opens. An older one stays open only while its auto-dismiss
    timer (``_DURATIONS``) is still running, so a toast on screen is not
    closed by the next notice and one that already timed out (or was closed)
    does not pop up again. A severity without a timer stays open.
    """
    nid = notice.get("id")
    if not nid:
        return True
    started = first_shown.setdefault(nid, now)
    duration = _DURATIONS.get(notice.get("severity", "error"))
    return duration is None or (now - started) * 1000.0 < duration


def register(app, state) -> None:  # pylint: disable=unused-argument  # uniform register(app, state) signature
    #: ids of notices already written to the log (one write per notice)
    logged_ids: set[str] = set()
    #: notice id -> when its toast first opened (``_clock`` seconds)
    first_shown: dict[str, float] = {}

    @app.callback(Output(ids.NOTICE_STACK, "children"),
                  Input(ids.NOTICES, "data"))
    def render_notices(notices):
        notices = errors.cap_notices(notices)
        now = _clock()
        toasts = []
        for notice in notices:
            nid = notice.get("id")
            # persist every new notice to the log file before it is rendered,
            # so a warning/error survives its 5-10 s toast and stays readable
            # later. The browser store only holds a short ``details``; the full
            # traceback is still known server-side and goes to the log.
            if nid and nid not in logged_ids:
                logged_ids.add(nid)
                try:
                    full = errors.full_details(nid)
                    logbook.log_notice({**notice, "details": full}
                                       if full else notice)
                except Exception:  # pragma: no cover  # nosec B110  # pylint: disable=broad-exception-caught
                    pass
            toasts.append(notice_toast(
                notice, is_open=_still_showing(first_shown, notice, now)))
        if len(first_shown) > 500:           # bounded: keep only the live ids
            live = {n.get("id") for n in notices}
            for stale in [k for k in first_shown if k not in live]:
                del first_shown[stale]
        return toasts

    # ------------------------------------------------- the full-log viewer
    @app.callback(
        Output(ids.LOG_OFFCANVAS, "is_open"),
        Output(ids.LOG_CONTENT, "children"),
        Output(ids.LOG_PATH_INFO, "children"),
        Input(ids.LOG_VIEW_BTN, "n_clicks"),
        Input(ids.LOG_REFRESH_BTN, "n_clicks"),
        State(ids.LOG_OFFCANVAS, "is_open"),
        prevent_initial_call=True)
    def show_log(view_clicks, refresh_clicks, is_open):  # pylint: disable=unused-argument  # Dash callback input
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
