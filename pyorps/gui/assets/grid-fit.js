/* Fit the columns of the AG grids marked .gui-grid-fit when they are shown.
 *
 * dash-ag-grid's columnSize ("sizeToFit", "responsiveSizeToFit") fits the
 * columns right when the grid starts. Most grids of the app start in a hidden
 * tab or a closed <details> card and have no width then: AG Grid warns (#29),
 * or squeezes the columns to their minimum width and does not fit again when
 * the card opens. The grids built in layout.py carry a class instead
 * (layout.FIT_COLUMNS, layout.FIT_COLUMNS_RESPONSIVE) and this script calls
 * sizeColumnsToFit() once the grid is rendered with a width:
 *   .gui-grid-fit             the first time only, as "sizeToFit" did (a
 *                             column the user resized keeps its width);
 *   .gui-grid-fit-responsive  again whenever the width changes, as
 *                             "responsiveSizeToFit" did.
 * No column is squeezed below MIN_WIDTH px (or its own maxWidth): a grid with
 * more columns than fit scrolls sideways instead of showing "R…" columns.
 * A new grid component (page reload, re-render) and new column definitions
 * are fitted again.
 */
(function () {
    "use strict";
    var MARK = "gui-grid-fit";
    var RESPONSIVE = "gui-grid-fit-responsive";
    var MIN_WIDTH = 60;
    var observed = new WeakSet();
    var fittedWidth = new WeakMap();     // host -> width it was fitted at
    var listening = new WeakSet();       // grid APIs with the column listener

    function shown(host) {
        if (!host.isConnected || host.closest("details:not([open])")) {
            return false;
        }
        if (typeof host.checkVisibility === "function") {
            return host.checkVisibility();
        }
        return host.offsetParent !== null;
    }

    function check(host) {
        if (!host.id || !shown(host)) { return; }
        var width = Math.round(host.getBoundingClientRect().width);
        if (width <= 0) { return; }
        if (fittedWidth.has(host) && (!host.classList.contains(RESPONSIVE) ||
                fittedWidth.get(host) === width)) {
            return;
        }
        var grids = window.dash_ag_grid;
        if (!grids || typeof grids.getApiAsync !== "function") { return; }
        fittedWidth.set(host, width);
        grids.getApiAsync(host.id).then(function (api) {
            if (!api || api.isDestroyed() || !shown(host)) {
                fittedWidth.delete(host);    // try again at the next change
                return;
            }
            if (!listening.has(api)) {
                // new column definitions (the cost table is seeded with its
                // columns later): fit them again, as when the grid starts
                listening.add(api);
                api.addEventListener("newColumnsLoaded", function () {
                    fittedWidth.delete(host);
                    check(host);
                });
            }
            api.sizeColumnsToFit({defaultMinWidth: MIN_WIDTH});
        }).catch(function () { fittedWidth.delete(host); });
    }

    var resizes = new ResizeObserver(function (entries) {
        entries.forEach(function (entry) { check(entry.target); });
    });

    function watch(root) {
        if (!root || root.nodeType !== 1) { return; }
        var hosts = root.classList.contains(MARK) ? [root] : [];
        hosts.push.apply(hosts, root.querySelectorAll("." + MARK));
        hosts.forEach(function (host) {
            if (!observed.has(host)) {
                observed.add(host);
                resizes.observe(host);      // first callback comes at once
            }
        });
    }

    new MutationObserver(function (records) {
        records.forEach(function (record) {
            record.addedNodes.forEach(watch);
        });
    }).observe(document.documentElement, {childList: true, subtree: true});
    watch(document.documentElement);

    // a <details> card that opens may leave the grid's box size unchanged,
    // so the ResizeObserver stays silent: check the grids inside it
    document.addEventListener("toggle", function (ev) {
        var card = ev.target;
        if (card && card.tagName === "DETAILS" && card.open) {
            card.querySelectorAll("." + MARK).forEach(check);
        }
    }, true);
})();
