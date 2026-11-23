"""Bound-and-prune driver of the case study: the lower bound of every pad position from the real fields (paper math.md M4, M12).

Stage 2 of the plan's pipeline on the 5 m position lattice, scenario ALL-CABLE (no overhead leg), terminals on lattice nodes:

    LB(s) = A_site(s)                                            exact pad cost, best rotation (site field, int cents)
          + min_{|o| <= 50 m} LB_acc(s + o)                      access road       (GPU float32 -> float64 bound)
          + min_{|o| <= 50 m} LB_HV (s + o)                      110 kV cable leg  (PCCs and verified towers as seeds)
          + min_{|o| <= 50 m} LB_MV (s + o)                      collector trench  (7 turbine drains, Steiner/MST bound)

The offset filter is the offset-free relaxation of math.md eq. (4.2): a pad terminal lies within the pad's half-diagonal
(50 m) of its anchor. Dropped as non-negative: bays, transformer, system and junction costs of the collector, tap works,
consent and delay (these only loosen the bound; they never change the answer). eps_adm is 0 here: the float32 slack is
already subtracted inside every GPU bound and the site field is integer.

Without an incumbent ``--z-ub`` it reports what stage 2 can say alone: the lowest bound (a certified lower bound of the
optimum of the all-cable scenario, within the stated relaxations), the 20 lowest-bound positions (the F0 candidates that
the exact stage must price) and the survivor-count-versus-Z_UB curve (plan phase F). With ``--z-ub`` it also prunes by
``LB - eps_adm > Z_UB`` (``pyorps.certify.prune``) and reports the kept set. The exact stages (incumbents E0/E1, collector
engines A and B, float64 re-solves) are not in this script; see the pipeline plan.

Writes ``results/bound_and_prune_<date>.json`` (small). Stand-in prices: nothing here is quotable as a result.

    .venv/Scripts/python.exe case_studies/runkel_free_siting/scripts/run_bound_and_prune.py [--z-ub EUR]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASE = HERE.parent
MANIFEST = CASE / "data" / "data_manifest.yaml"
RASTERS = CASE / "data" / "rasters"
RESULTS = CASE / "results"
SENTINEL_U32 = 4294967295
NODATA = 1e29
PAD_HALF_DIAGONAL_M = 50.0                    # half-diagonal of the 60 x 80 m pad
LATTICE_M = 5


def load_lattice_site(m, shape, x0, y1):
    """Site field (EUR, best rotation) sampled at the lattice nodes; inf where infeasible or outside the envelope."""
    import numpy as np
    import rasterio

    out = np.full(shape, np.inf)
    with rasterio.open(m.file_of("site_field_envelope_cents")) as s:
        ex0, ey1, w, h = int(s.bounds.left), int(s.bounds.top), s.width, s.height
        dx, dy = ex0 - x0, y1 - ey1                                 # envelope origin in full-grid cells
        j_lo, j_hi = -(-dy // LATTICE_M), (dy + h - 1) // LATTICE_M
        i_lo, i_hi = -(-dx // LATTICE_M), (dx + w - 1) // LATTICE_M
        cols = np.arange(i_lo, i_hi + 1) * LATTICE_M - dx
        for j0 in range(j_lo, j_hi + 1, 200):
            js = np.arange(j0, min(j_hi, j0 + 199) + 1)
            r0, r1 = js[0] * LATTICE_M - dy, js[-1] * LATTICE_M - dy + 1
            from rasterio.windows import Window

            block = s.read(1, window=Window(0, int(r0), w, int(r1 - r0)))
            vals = block[(js * LATTICE_M - dy) - r0][:, cols].astype(np.float64)
            vals[vals == SENTINEL_U32] = np.inf
            out[js[0]:js[-1] + 1, i_lo:i_hi + 1] = np.where(np.isfinite(vals), vals / 100.0, np.inf)
    return out


def collector_cable_floor(m):
    """``(alpha, const)``: the cable material floor of any collector at any root (math.md M7.6b).

    ``alpha = min over MV levels and non-empty turbine sets S of rho_hat(S) / |S|`` (EUR per metre and turbine); every
    turbine's power runs in a cable along its whole path to the root, so the cable cost is at least
    ``alpha * sum_i |t_i - g|``. ``const``: each turbine pays one panel and the root at least one bay (cheapest level).
    """
    import math

    from pyorps.collector.bounds import rho_hat
    from pyorps.costmodel import CostCatalogue, LossCategories
    from pyorps.costmodel.collector_inputs import CollectorInputs, build_collector_model

    alphas, consts = [], []
    for u in (20.0, 30.0, 33.0):
        model = build_collector_model(CollectorInputs.load(CASE / "config" / "collector_inputs_2026.yaml"),
                                      CostCatalogue.load(CASE / "config" / "cost_catalogue_2026.yaml"),
                                      LossCategories.load(CASE / "data" / "loss_categories.npz"), u_kv=u, rated_mw=[6.8] * 7).model
        rho, _ = rho_hat(model)
        alphas.append(min(rho[s] / bin(s).count("1") for s in range(1, model.full + 1) if math.isfinite(rho[s])))
        consts.append(model.n * model.turbine_panel_eur + model.bay_eur)
    return float(min(alphas)), float(min(consts))


def branch_constants():
    from pyorps.costmodel import CostCatalogue
    from pyorps.costmodel.branch_constants import BranchConstants

    return BranchConstants.load(CASE / "config" / "branch_constants_2026.yaml", CostCatalogue.load(CASE / "config" / "cost_catalogue_2026.yaml"))


def assemble(m):
    """Lattice arrays of the bound: bound terms, their raw fields and the lattice geometry (shared with price_incumbents.py)."""
    import numpy as np
    import rasterio

    from pyorps.certify.prune import branch_lower_bound
    from pyorps.siting.lattice_bounds import float32_lower_bound, offset_min_filter

    hashes = m.verify(["site_field_envelope_cents", "access_field_5m_lattice", "hv_field_taps_5m_lattice", "mv_lb_5m_lattice"])
    side = json.loads((RASTERS / "mv_hv_bound_sidecar.json").read_text(encoding="utf-8"))["stats"]
    acc_side = json.loads((RASTERS / "access_field_sidecar.json").read_text(encoding="utf-8"))

    def find(d, key):
        if isinstance(d, dict):
            if key in d and not isinstance(d[key], (dict, list)):
                return d[key]
            for v in d.values():
                r = find(v, key)
                if r is not None:
                    return r
        return None

    k_acc, k_hv = int(find(acc_side, "k_max_rigorous")), int(side["hv_taps"]["k_max_rigorous"])
    with rasterio.open(m.file_of("access_field_5m_lattice")) as s:
        acc, shape = s.read(1), (s.height, s.width)
        x0, y1 = int(s.bounds.left), int(s.bounds.top)
    with rasterio.open(m.file_of("hv_field_taps_5m_lattice")) as s:
        hv = s.read(1)
    with rasterio.open(m.file_of("mv_lb_5m_lattice")) as s:
        mv = s.read(1)
    assert acc.shape == hv.shape == mv.shape == shape
    site = load_lattice_site(m, shape, x0, y1)
    gj, gi = np.indices(shape)
    gx, gy = x0 + float(LATTICE_M) * gi, y1 - float(LATTICE_M) * gj
    acc_lb = float32_lower_bound(acc, k_acc)
    hv_lb = float32_lower_bound(hv, k_hv)
    mv_lb = np.where(mv >= NODATA, np.inf, mv.astype(np.float64))      # slack already applied by the builder
    # cable floor of the collector (math.md M7.6b): alpha * sum_i |t_i - g| + 7 panels + 1 bay, added at every node
    alpha, const = collector_cable_floor(m)
    turb = m.read_vector("turbines", verify=False)
    cable = np.zeros(shape)
    for g in turb.geometry:
        cable += np.hypot(gx - g.x, gy - g.y)
    mv_lb = mv_lb + alpha * cable + const
    window_info = None
    dtr_layer = m.layers.get("mv_window_dtr_5m_lattice")
    wside = RASTERS / "mv_window_sidecar.json"
    if dtr_layer is not None and dtr_layer.status == "present" and wside.exists():
        ws = json.loads(wside.read_text(encoding="utf-8"))["stats"]
        if ws.get("levels") and ws.get("dtr"):
            m.verify(["mv_window_dtr_5m_lattice"])
            with rasterio.open(m.file_of("mv_window_dtr_5m_lattice")) as s:
                dtr = float32_lower_bound(s.read(1), int(ws["dtr"]["k_max_rigorous"]))
            w = ws["window_full_grid_cells"]
            xmin, xmax, ymax, ymin = x0 + w["col0"], x0 + w["col1"], y1 - w["row0"], y1 - w["row1"]
            ex_ = np.maximum(np.maximum(xmin - gx, gx - xmax), 0.0)
            ey_ = np.maximum(np.maximum(ymin - gy, gy - ymax), 0.0)
            e_win = np.maximum(np.hypot(ex_, ey_) - 2.0, 0.0)           # distance to the window rectangle, 2 m for the cell geometry
            outside = (ex_ > 0) | (ey_ > 0)
            per_level = [lv["win_eur"] + lv["alpha_eur_per_m_turbine"] * lv["n_turbines"] * e_win + lv["bay_eur"] for lv in ws["levels"].values()]
            new = np.min(per_level, axis=0) + dtr                         # math.md M7.6d: inside part + outside trench + outside cable + bay
            mv_lb = np.where(outside, np.maximum(mv_lb, new), mv_lb)
            window_info = {"levels": sorted(ws["levels"]), "outside_nodes": int(outside.sum())}
    radius = int(round(PAD_HALF_DIAGONAL_M / LATTICE_M))
    terms = [site] + [offset_min_filter(t, radius) for t in (acc_lb, hv_lb, mv_lb)]
    ledger = branch_constants()
    lb = branch_lower_bound(ledger.k_min, terms)
    with rasterio.open(m.file_of("hv_field_min_5m_lattice")) as s:
        hv_pcc = s.read(1)
    k_hv_pcc = int(find(json.loads((RASTERS / "hv_field_sidecar.json").read_text(encoding="utf-8")), "k_max_rigorous"))
    return {"lb": lb, "ledger": ledger, "window": window_info, "hv_pcc_raw": hv_pcc, "k_hv_pcc": k_hv_pcc, "terms": terms, "site": site, "acc_raw": acc, "hv_raw": hv, "k_acc": k_acc, "k_hv": k_hv, "shape": shape,
            "x0": x0, "y1": y1, "hashes": hashes, "side": side}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--z-ub", type=float, default=None, help="incumbent cost in EUR (a complete, re-priced design)")
    ap.add_argument("--eps-adm", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import numpy as np

    from pyorps.certify.prune import prune_mask
    from pyorps.siting.data_manifest import DataManifest

    m = DataManifest.load(MANIFEST)
    A = assemble(m)
    lb, terms, site, shape, x0, y1, hashes, side = (A[k] for k in ("lb", "terms", "site", "shape", "x0", "y1", "hashes", "side"))
    k_acc, k_hv = A["k_acc"], A["k_hv"]
    feasible_site = np.isfinite(site)
    finite = np.isfinite(lb)
    flat = lb.ravel()
    order = np.argsort(flat, kind="stable")[:20]
    order = order[np.isfinite(flat[order])]
    jj, ii = np.divmod(order, shape[1])

    def xy(j, i):
        return float(x0 + LATTICE_M * i), float(y1 - LATTICE_M * j)

    top = []
    for k, (j, i) in enumerate(zip(jj, ii)):
        top.append({"rank": k + 1, "x": xy(j, i)[0], "y": xy(j, i)[1], "lb_eur": float(lb[j, i]),
                    "site": float(terms[0][j, i]), "access": float(terms[1][j, i]), "hv": float(terms[2][j, i]), "collector_trench": float(terms[3][j, i])})
    lb_min = float(lb[finite].min()) if finite.any() else float("inf")
    curve = []
    for delta in (0, 1e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7):
        z = lb_min + delta
        curve.append({"z_ub_minus_lb_min_eur": delta, "kept": int((prune_mask(lb, z, args.eps_adm) & finite).sum())})
    res = {
        "date": dt.date.today().isoformat(),
        "scenario": "all-cable (no overhead or mixed leg); terminals on 5 m lattice nodes within the pad half-diagonal (assumption)",
        "stand_in_inputs": "site prices, access constraints (7), loss price and tap works are stand-ins or dropped: NOT QUOTABLE",
        "anchors": {"lattice_cells": int(lb.size), "site_feasible": int(feasible_site.sum()), "all_bounds_finite": int(finite.sum())},
        "bounds_dropped_as_nonnegative": ["bays", "transformer", "collector system and junction costs", "tap works", "consent and delay", "K(b)"],
        "eps_adm_eur": args.eps_adm,
        "slack": {"access": k_acc * 3 * 2.0 ** -24, "hv_taps": k_hv * 3 * 2.0 ** -24, "mv": side["mv"]["slack_relative"]},
        "lb_min_eur": lb_min, "lb_percentiles_eur": {str(q): float(np.percentile(lb[finite], q)) for q in (1, 5, 25, 50, 75)} if finite.any() else {},
        "f0_candidates_lowest_lb": top,
        "survivor_curve": curve,
        "inputs_sha256_prefix": {k: v[:16] for k, v in hashes.items()},
    }
    if args.z_ub is not None:
        keep = prune_mask(lb, args.z_ub, args.eps_adm) & finite
        jk, ik = np.nonzero(keep)
        res["prune"] = {"z_ub_eur": args.z_ub, "kept": int(keep.sum()), "dropped": int(finite.sum() - keep.sum()),
                        "kept_bbox_xy": [xy(jk.max(), ik.min()), xy(jk.min(), ik.max())] if keep.any() else None,
                        "rule": "dropped iff LB - eps_adm > Z_UB"}
    print(json.dumps({k: v for k, v in res.items() if k not in ("f0_candidates_lowest_lb",)}, indent=1))
    print("lowest-bound positions (F0 candidates):")
    for t in top[:5]:
        print(" ", t)
    out = Path(args.out) if args.out else RESULTS / f"bound_and_prune_{dt.date.today().isoformat()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
