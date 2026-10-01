/* Map-corner basemap switcher: close the <details> panel on outside clicks
 * (native details only closes via its summary, which feels sticky for a
 * map control). Delegated, so it works whenever Dash renders the control. */
(function () {
    "use strict";
    document.addEventListener("pointerdown", function (ev) {
        var open = document.querySelector("details.basemap-control[open]");
        if (open && !open.contains(ev.target)) {
            open.removeAttribute("open");
        }
    });
})();
