"""PYORPS GUI sections: the layout of one card each, kept out of ``layout.py``.

Round 14 scaffold. Every module here holds the layout of one group of controls
and exposes the same two functions:

``section() -> list``
    The cards (``layout._collapsible_card`` / ``layout._card``) that the
    layout splices into its tab. The private layout helpers (``_card``,
    ``_collapsible_card``, ``_lbl``, ``_addon``, ``_help``, ``_browse``) are
    imported INSIDE ``section()``: ``layout`` imports this package, so a
    module-level import would be a cycle.

``prefill(state, put) -> None``
    Writes the first-paint values the server-side ``state`` holds (a page
    reload, or a project opened with ``--project``) with
    ``put(component_id, prop, value)``, the same contract as the
    ``layout._prefill_*`` steps. Read ``state.gui_settings[<key>]``.

The callbacks of a section live in ``callbacks/<name>.py`` (each has
``register(app, state)``) and its logic in ``services/<name>.py``.
"""
