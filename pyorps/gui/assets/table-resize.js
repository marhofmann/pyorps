/* Free drag-resize for the bottom layer-attribute panel.
 *
 * Dragging the .table-resizer strip (the panel's top edge) sets the
 * offcanvas height in PIXELS (no percent steps), persisted to localStorage
 * and re-applied on load. The inner AG grid follows via flex CSS.
 * Delegated pointer events — they work whenever Dash renders the panel.
 */
(function () {
    "use strict";
    var KEY = "pyorps-table-h";
    var MIN = 140;                          // header + a few rows
    var dragging = false;

    function panel() {
        return document.getElementById("layer-table-offcanvas");
    }

    function clamp(px) {
        var max = Math.max(MIN, window.innerHeight - 120);
        return Math.min(Math.max(px, MIN), max);
    }

    function apply(px) {
        var el = panel();
        if (el) { el.style.height = Math.round(px) + "px"; }
    }

    function restore() {
        try {
            var saved = parseInt(window.localStorage.getItem(KEY), 10);
            if (saved > 0 && panel()) { apply(clamp(saved)); return true; }
        } catch (e) { /* storage unavailable — default height stands */ }
        return !!panel();
    }
    if (!restore()) {
        var timer = window.setInterval(function () {
            if (restore()) { window.clearInterval(timer); }
        }, 300);
    }

    document.addEventListener("pointerdown", function (ev) {
        var t = ev.target;
        if (!t || !t.classList ||
                !t.classList.contains("table-resizer")) { return; }
        dragging = true;
        document.body.classList.add("table-resizing");
        if (t.setPointerCapture) {
            try { t.setPointerCapture(ev.pointerId); } catch (e) { }
        }
        ev.preventDefault();
    });

    document.addEventListener("pointermove", function (ev) {
        if (!dragging) { return; }
        apply(clamp(window.innerHeight - ev.clientY));
    });

    function stop() {
        if (!dragging) { return; }
        dragging = false;
        document.body.classList.remove("table-resizing");
        var el = panel();
        if (el) {
            try {
                window.localStorage.setItem(
                    KEY, String(parseInt(el.style.height, 10)));
            } catch (e) { /* persistence is best-effort */ }
        }
    }
    document.addEventListener("pointerup", stop);
    document.addEventListener("pointercancel", stop);
})();
