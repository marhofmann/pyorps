/**
 * PYORPS GUI: global-L arbitration shim (hard rule C4').
 *
 * Two Dash component bundles fight over the global `L`:
 *
 *  - dash-leaflet's lazy EditControl chunk bundles leaflet-draw, which
 *    reads/writes the BARE GLOBAL `L` and needs it to be the real Leaflet
 *    (otherwise: "Cannot read properties of undefined (reading 'extend')"
 *    and the draw toolbar never appears).
 *  - dash-ag-grid's minified bundle leaks an object-spread helper into `L`
 *    (sloppy `L = ...` without var) and READS it back from its own async
 *    chunks ("L is not a function" if we simply pin Leaflet).
 *
 * Fix: grab dash-leaflet's *bundled* Leaflet from its webpack runtime and
 * install an accessor on `window.L` that serves Leaflet to dash_leaflet
 * callers (decided by the caller's script URL on the stack) and the
 * last-written value to everyone else. The E2E tests exercise both the draw
 * toolbar and an AG grid on one page to guard this against version bumps.
 */
(function () {
    "use strict";
    var chunks = (self.webpackChunkdash_leaflet =
        self.webpackChunkdash_leaflet || []);

    function isLeaflet(m) {
        return !!(m && m.Handler && m.Control && m.DomUtil && m.version);
    }

    function install(realLeaflet) {
        var other = window.L; // whatever some other bundle already leaked
        // the bundled Leaflet under a collision-free name, for our own asset
        // scripts (draw-preview.js): the arbitrated `window.L` would hand
        // THEM the foreign leak, not Leaflet
        window.__pyorpsLeaflet = realLeaflet;
        try {
            Object.defineProperty(window, "L", {
                configurable: true,
                get: function () {
                    var stack = String(new Error().stack || "");
                    if (stack.indexOf("dash_leaflet") !== -1) {
                        return realLeaflet;
                    }
                    return other !== undefined ? other : realLeaflet;
                },
                set: function (value) {
                    // leaflet(-draw) never reassigns L; remember foreign leaks
                    if (!isLeaflet(value)) { other = value; }
                },
            });
            window.__pyorpsLeafletShim = "arbitrating " + realLeaflet.version;
        } catch (e) {
            window.L = realLeaflet;
            window.__pyorpsLeafletShim = "assigned " + realLeaflet.version;
        }
    }

    chunks.push([["pyorps-leaflet-global-shim"], {}, function (req) {
        window.__pyorpsLeafletShim = "runtime-ran";
        try {
            // module id of the bundled Leaflet in dash-leaflet 1.1.3
            var L = req(3481);
            if (isLeaflet(L)) { install(L); return; }
        } catch (e) { /* id changed in a newer dash-leaflet — scan below */ }
        try {
            if (req.m) {
                for (var id in req.m) {
                    try {
                        var m = req(id);
                        if (isLeaflet(m)) { install(m); return; }
                    } catch (e) { /* skip modules that fail to init */ }
                }
            }
            window.__pyorpsLeafletShim = "no-leaflet-found";
        } catch (e) {
            window.__pyorpsLeafletShim = "scan-failed: " + e;
        }
    }]);
})();
