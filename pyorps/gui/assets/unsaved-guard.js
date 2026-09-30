/* Save-before-closing guard (browser).
 *
 * window.__pyorpsDirty mirrors the server's ProjectState.dirty flag (set on
 * every layer add/remove, cleared by project save/open/new — clientside
 * callback in callbacks/io.py). While it is true, closing or reloading the
 * tab triggers the browser's native "unsaved changes" confirmation; custom
 * text is not allowed by modern browsers. The desktop (pywebview) window
 * additionally asks via its own closing handler (app.py).
 */
(function () {
    "use strict";
    window.addEventListener("beforeunload", function (ev) {
        if (window.__pyorpsDirty) {
            ev.preventDefault();
            ev.returnValue = "";          // legacy engines need a value
            return "";
        }
    });
    window.__pyorpsUnsavedGuard = true;   // wiring marker (tests)
})();
