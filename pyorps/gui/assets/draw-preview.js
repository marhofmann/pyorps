/* Live draw preview (leaflet-draw).
 *
 * DRAW_SHAPE_OPTIONS (layout.py) styles the in-progress rubber band VISIBLY,
 * so every click / mouse move previews the polygon or rectangle edges. But
 * the same options would also style the CREATED layer, doubling the styled
 * study-area / cost-polygon host layer (the round-7 "duplicate window").
 * So: patch the one place every draw handler funnels through —
 * L.Draw.Feature.prototype._fireCreatedEvent — and restyle the finished
 * layer to invisible before it is added to the edit FeatureGroup. The result:
 * visible while drawing, host-layer-only once done.
 *
 * leaflet-draw is bundled lazily inside dash-leaflet's EditControl chunk, so
 * poll until L.Draw exists. Use window.__pyorpsLeaflet (exposed by
 * leaflet-global-shim.js): the arbitrated window.L hands non-dash-leaflet
 * callers — including this script — AG Grid's leaked value, never Leaflet.
 */
(function () {
    "use strict";
    var HIDDEN = {color: "#3388ff", weight: 0, opacity: 0,
                  fill: false, fillOpacity: 0};

    function patch() {
        var L = window.__pyorpsLeaflet;
        if (!L || !L.Draw || !L.Draw.Feature ||
                !L.Draw.Feature.prototype ||
                L.Draw.Feature.prototype._pyorpsHideCreated) {
            return !!(L && L.Draw && L.Draw.Feature &&
                      L.Draw.Feature.prototype &&
                      L.Draw.Feature.prototype._pyorpsHideCreated);
        }
        var fireCreated = L.Draw.Feature.prototype._fireCreatedEvent;
        L.Draw.Feature.prototype._fireCreatedEvent = function (layer) {
            try {
                if (layer && typeof layer.setStyle === "function") {
                    layer.setStyle(HIDDEN);
                }
            } catch (e) { /* cosmetic only — never block shape creation */ }
            return fireCreated.call(this, layer);
        };
        L.Draw.Feature.prototype._pyorpsHideCreated = true;
        return true;
    }

    if (!patch()) {
        var timer = window.setInterval(function () {
            if (patch()) { window.clearInterval(timer); }
        }, 200);
    }
})();
