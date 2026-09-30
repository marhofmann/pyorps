/* Drag-resizable sidebar.
 *
 * Dragging the .sidebar-resizer strip (the map/sidebar boundary) rewrites the
 * --sidebar-w design token, which everything derives its width from (the
 * sidebar column, the layer-table offcanvas). The chevron collapse handle is
 * untouched. The chosen width persists in localStorage and is re-applied on
 * load. Delegated pointer events: they work whenever Dash (re)renders.
 */
(function () {
    "use strict";
    var KEY = "pyorps-sidebar-w";
    var MIN = 300;                       // keep the tab bar usable
    var MAX_FRACTION = 0.75;             // always leave some map visible
    var root = document.documentElement;
    var dragging = false;

    function clamp(px) {
        var max = Math.max(MIN, window.innerWidth * MAX_FRACTION);
        return Math.min(Math.max(px, MIN), max);
    }

    function apply(px) {
        root.style.setProperty("--sidebar-w", Math.round(px) + "px");
    }

    try {
        var saved = parseInt(window.localStorage.getItem(KEY), 10);
        if (saved > 0) { apply(clamp(saved)); }
    } catch (e) { /* storage unavailable — default width stands */ }

    document.addEventListener("pointerdown", function (ev) {
        var t = ev.target;
        if (!t || !t.classList ||
                !t.classList.contains("sidebar-resizer")) { return; }
        dragging = true;
        document.body.classList.add("sidebar-resizing");
        if (t.setPointerCapture) {
            try { t.setPointerCapture(ev.pointerId); } catch (e) { }
        }
        ev.preventDefault();
    });

    document.addEventListener("pointermove", function (ev) {
        if (!dragging) { return; }
        apply(clamp(window.innerWidth - ev.clientX));
    });

    function stop() {
        if (!dragging) { return; }
        dragging = false;
        document.body.classList.remove("sidebar-resizing");
        try {
            var w = getComputedStyle(root).getPropertyValue("--sidebar-w");
            window.localStorage.setItem(KEY, String(parseInt(w, 10)));
        } catch (e) { /* persistence is best-effort */ }
        window.dispatchEvent(new Event("resize"));   // Leaflet re-measures
    }
    document.addEventListener("pointerup", stop);
    document.addEventListener("pointercancel", stop);
})();
