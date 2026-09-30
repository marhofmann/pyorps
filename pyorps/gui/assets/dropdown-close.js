/**
 * PYORPS GUI: close a single-select dropdown when an option is clicked —
 * including the one already selected (task 51).
 *
 * dcc.Dropdown (react-select) leaves the menu open when you click the option
 * that is already selected (value unchanged → no close). Users expect any
 * pick, same or different, to close the menu. We blur the control's input on
 * the tick after the click so react-select closes the menu; multi-selects are
 * left alone (they should stay open to pick several). Defensive throughout so
 * it can never break the page.
 */
(function () {
    "use strict";

    function isMultiSelect(root) {
        // react-select multi shows removable value chips
        return !!root.querySelector(
            '[class*="multiValue"], [class*="multi-value"]');
    }

    function findRoot(option) {
        return option.closest(".dash-dropdown")
            || option.closest('[class*="-container"]')
            || option.closest(".Select");
    }

    document.addEventListener("click", function (event) {
        try {
            var option = event.target.closest('[role="option"]');
            if (!option) { return; }
            var root = findRoot(option);
            if (root && isMultiSelect(root)) { return; }
            // let react-select apply the selection first, then close the menu
            setTimeout(function () {
                var input = root && root.querySelector("input");
                if (!input && document.activeElement
                        && document.activeElement.tagName === "INPUT") {
                    input = document.activeElement;
                }
                if (input) {
                    input.dispatchEvent(new KeyboardEvent("keydown", {
                        key: "Escape", code: "Escape", keyCode: 27,
                        which: 27, bubbles: true
                    }));
                    input.blur();
                }
            }, 0);
        } catch (e) { /* never let a UX nicety break the app */ }
    }, true);
})();
