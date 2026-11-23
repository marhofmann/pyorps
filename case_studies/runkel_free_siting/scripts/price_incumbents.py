"""Incumbent Z_UB, the F0 loop and the kept set E_b (paper math.md M4.2, M4.3; plan phases E and F0).

Reduced objective of the bound-and-prune driver: the four position-dependent bills with the branch constants K(b) dropped
(bays at the connection point, transformer, switchyard equipment, consent). Terminals of the pad lie on the anchor cell.
Scenario all-cable. Every number below is an UPPER bound of the reduced optimum because each piece is the cost of a real
design:

* pad site: the exact integer-cent field at the anchor (best rotation);
* access road and 110 kV leg: the float32 GPU field divided by ``1 - k_max eta`` (an upper bound of the cost of the path
  the drain found, math.md M9.1); the 110 kV leg uses the same field as the lower bound (connection points and verified
  towers as seeds, tap works zero), so the two bounds see the same reduced objective;
* MV collector: the exact ``D_share`` optimum (engine B, one cable size per cable, switching stations, bays, derating) on
  the route graph: the union of the exact float64 least-cost 1 m routes between the turbines and to the root; terminals and
  branching cells are nodes, chains are contracted, an edge carries the exact trench weight times ``1 + omega`` and the
  length of its stretch. Every edge is a real trench stretch, so any design on the graph is a real layout and costs at
  least the optimum: an upper bound; where routes share cells the sharing is priced once. The window is the bounding box of turbines and candidates plus a buffer.

F0 (plan phase F0): price the ``--f0`` lowest-LB positions, set ``Z_UB`` to the smallest total, then price the next ``--f0``
positions whose bound is still not above ``Z_UB`` (``--passes`` times). ``E_b`` is every lattice position with
``LB - eps_adm <= Z_UB``. These are bounds of the REDUCED objective: constants K(b) are not in them, so a position outside
``E_b`` is excluded for that objective only (math.md M4.6). Stand-in prices: not quotable.

    .venv/Scripts/python.exe case_studies/runkel_free_siting/scripts/price_incumbents.py --f0 20 --passes 2
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "runtime_study"))

CASE = HERE.parent
MANIFEST = CASE / "data" / "data_manifest.yaml"
RESULTS = CASE / "results"
INPUTS = CASE / "config" / "collector_inputs_2026.yaml"
CATALOGUE = CASE / "config" / "cost_catalogue_2026.yaml"
LOSS_TABLE = CASE / "data" / "loss_categories.npz"
NODATA = 1e29
ETA = 3.0 * 2.0 ** -24


def walk(prev, target):
    """Cells of the route from the seed to ``target`` along the predecessor forest."""
    import numpy as np

    path = [int(target)]
    c = int(target)
    while prev[c] != -1:
        c = int(prev[c])
        path.append(c)
    return np.asarray(path[::-1], dtype=np.int64)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--f0", type=int, default=20)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--pool", type=int, default=100, help="lowest-bound positions whose routes are prepared")
    ap.add_argument("--buffer", type=int, default=1000, help="window buffer in m")
    ap.add_argument("--eps-adm", type=float, default=0.0)
    ap.add_argument("--u-kv", type=float, default=33.0)
    ap.add_argument("--zero-wind-q", choices=("conservative", "optimistic"), default="conservative",
                    help="reactive power of the converters at zero wind: none (conservative) or up to the rated apparent power (optimistic)")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--pool-mode", choices=("lb", "proxy", "cluster"), default="lb",
                    help="order of the candidate pool: lowest bound, or lowest site + access + 110 kV + ratio * collector bound")
    ap.add_argument("--mv-ratio", type=float, default=3.3, help="collector UB/LB ratio seen in the first pass (ranking only)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--block-m", type=int, default=1000, help="cluster mode: one representative (lowest proxy) per block of this size")
    ap.add_argument("--bbox", type=float, nargs=4, default=None, metavar=("X0", "Y0", "X1", "Y1"), help="restrict the candidate pool to this box (m)")
    ap.add_argument("--z-guess", type=float, default=None, help="cluster mode: only positions with LB below this are candidates")
    ap.add_argument("--cache", default=None, help="pickle of the extracted routes (re-used if present)")
    ap.add_argument("--graph", choices=("terminal", "route"), default="terminal",
                    help="terminal: edges are the exact routes between turbines and to the root (fast); route: shared stretches with branching cells (an 11-node graph did not finish in 20 minutes: engine B at n = 7)")
    ap.add_argument("--root-routes", type=int, default=2, help="root routes to the cheapest turbines kept in the route graph")
    ap.add_argument("--no-feasibility", action="store_true",
                    help="skip the plant voltage check of each incumbent (then Z_UB is NOT certified feasible, math.md M15)")
    ap.add_argument("--operator", default="operator_a", choices=("operator_a", "operator_b"), help="grid rules of the point of connection")
    ap.add_argument("--leg-step-km", type=float, default=0.025, help="grid step over the 110 kV leg length interval")
    args = ap.parse_args(argv)

    import numpy as np
    import rasterio
    import run_bound_and_prune as rbp
    from rasterio.windows import Window

    from pyorps.collector.reference import CollectorGraph, solve_collector
    from pyorps.costmodel import CostCatalogue, LossCategories
    from pyorps.costmodel.collector_inputs import CollectorInputs, build_collector_model
    from pyorps.graph.drain import drain
    from pyorps.siting.data_manifest import DataManifest
    from pyorps.utils.neighborhood import get_neighborhood_steps

    m = DataManifest.load(MANIFEST)
    A = rbp.assemble(m)
    lb, site, shape, x0, y1 = A["lb"], A["site"], A["shape"], A["x0"], A["y1"]
    finite_site = np.isfinite(site)
    flat_lb = np.where(np.isfinite(lb) & finite_site, lb, np.inf).ravel()
    if args.pool_mode in ("proxy", "cluster"):
        # the ORDER in which positions are priced is a search aid only; every price below is the cost of a real design
        t_ = A["terms"]
        key = np.where(np.isfinite(flat_lb), (t_[0] + t_[1] + t_[2] + args.mv_ratio * t_[3]).ravel(), np.inf)
        if args.bbox is not None:
            jj_, ii_ = np.indices(shape)
            xx_, yy_ = (x0 + 5.0 * ii_).ravel(), (y1 - 5.0 * jj_).ravel()
            key = np.where((xx_ >= args.bbox[0]) & (xx_ <= args.bbox[2]) & (yy_ >= args.bbox[1]) & (yy_ <= args.bbox[3]), key, np.inf)
        if args.z_guess is not None:
            key = np.where(flat_lb <= args.z_guess, key, np.inf)
        order = np.argsort(key, kind="stable")
        if args.pool_mode == "cluster":
            # one representative per block: the lowest proxy in the block (the ORDER is a search aid, every price is a real design)
            blk = args.block_m // 5
            jj_, ii_ = np.divmod(order, shape[1])
            bid = (jj_ // blk) * 100000 + ii_ // blk
            _, first = np.unique(bid, return_index=True)
            order = order[np.sort(first)]
            order = order[np.isfinite(key[order])]
    else:
        order = np.argsort(flat_lb, kind="stable")
    pool = order[:args.pool][np.isfinite(flat_lb[order[:args.pool]])]

    def lat_xy(idx):
        j, i = divmod(int(idx), shape[1])
        return j, i

    # upper bounds of the access and 110 kV terms at the pad anchors (terminals on the anchor)
    def ub_of(raw, k, j, i):
        v = float(raw[j, i])
        return float("inf") if v >= NODATA else v / (1.0 - k * ETA)

    sidecar = json.loads((CASE / "data" / "rasters" / "hv_cost_sidecar.json").read_text(encoding="utf-8"))
    import re
    omega = float(re.search(r"omega ([0-9.]+)", sidecar["note"]).group(1))

    # the collector model of one MV level
    built = build_collector_model(CollectorInputs.load(INPUTS), CostCatalogue.load(CATALOGUE), LossCategories.load(LOSS_TABLE),
                                  u_kv=args.u_kv, rated_mw=[6.8] * 7)
    model = built.model
    built.check_condition_c((1.0 + omega) * 125.0)
    print(built.report(), flush=True)

    # feasibility context (math.md M15): the plant must meet the voltage rules at the point of connection
    feas = None
    if not args.no_feasibility:
        import yaml

        from pyorps.certify.feasible_incumbent import leg_length_interval_km, length_grid_km, make_feasible
        from pyorps.collector.voltage import voltage_model_from_yaml
        from pyorps.collector.voltage_plant import HVLeg, grid_rules_from_yaml, plant_model_from_yaml
        from pyorps.siting.terminals import load_terminals

        vfile = CASE / "config" / "voltage_2026.yaml"
        sections = [int(s_) for s_ in yaml.safe_load(INPUTS.read_text(encoding="utf-8"))["cable"]["sections_mm2"]["value"]]
        vmodel = voltage_model_from_yaml(vfile, u_kv=args.u_kv, cable_mm2=sections, n_turbines=7)
        rules = grid_rules_from_yaml(vfile, args.operator)
        leg = yaml.safe_load(vfile.read_text(encoding="utf-8"))["hv_leg"]["cable_n2xsfl2y_1x300"]["value"]

        s_n_mva = float(yaml.safe_load(vfile.read_text(encoding="utf-8"))["turbine"]["rated_apparent_power_kva"]["value"]) / 1e3
        q_zero = s_n_mva if args.zero_wind_q == "optimistic" else 0.0

        def plant_for_leg(length_km):
            pm = plant_model_from_yaml(vfile, vmodel, hv_leg=HVLeg.from_type(
                r_ohm_per_km=leg["r_ohm_per_km"], x_ohm_per_km=leg["x_ohm_per_km"], c_nf_per_km=leg["c_nf_per_km"], length_km=length_km))
            return dataclasses.replace(pm, q_zero_power_mvar=q_zero)

        tap_xy = np.array([t_.xy for t_ in load_terminals(m.file_of("terminals"))]
                          + [(g_.x, g_.y) for g_ in m.read_vector("towers_110_verified", verify=False).geometry], dtype=float)
        w_min = float(re.search(r"free-cell weight (\d+)\.\.", sidecar["note"]).group(1))        # cheapest HV weight per metre
        feas = {"vmodel": vmodel, "rules": rules, "plant": plant_for_leg, "tap_xy": tap_xy, "w_min": w_min}
        print(f"feasibility check on: {args.operator}, {args.u_kv:g} kV, {len(tap_xy)} taps, cheapest HV weight {w_min:g} EUR/m", flush=True)

    # routes on the 1 m priced raster, window = turbines + candidates + buffer
    turb = m.read_vector("turbines", verify=False)
    with rasterio.open(m.file_of("cost_priced_1m")) as s:
        gx0, gy1, gw, gh = int(s.bounds.left), int(s.bounds.top), s.width, s.height
        tcol = [int(np.floor(g.x - gx0)) for g in turb.geometry]
        trow = [int(np.floor(gy1 - g.y)) for g in turb.geometry]
        pcell = [(5 * lat_xy(p)[0], 5 * lat_xy(p)[1]) for p in pool]            # full-grid (row, col) of each candidate
        rows = trow + [r for r, _ in pcell]
        cols = tcol + [c for _, c in pcell]
        r0, r1 = max(0, min(rows) - args.buffer), min(gh, max(rows) + args.buffer + 1)
        c0, c1 = max(0, min(cols) - args.buffer), min(gw, max(cols) + args.buffer + 1)
        vals = s.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
    h, w = vals.shape
    print(f"window {w} x {h} = {w * h / 1e6:.1f} M cells; {len(pool)} candidates; omega {omega:.4f}", flush=True)
    steps = np.asarray(get_neighborhood_steps(args.k, directed=True), dtype=np.int8)
    tcell = [(r - r0) * w + (c - c0) for r, c in zip(trow, tcol)]
    pc = [((r - r0) * w + (c - c0)) for r, c in pcell]
    n_t = len(tcell)

    def route_stats(path):
        rr, cc = np.divmod(path, w)
        return float(np.hypot(np.diff(rr), np.diff(cc)).sum())

    paths_tt: dict = {}                                      # (i, j) -> (cells, cumulative exact trench weight)
    paths_tr: dict = {}                                      # (i, candidate k) -> (cells, cumulative weight)
    import pickle
    cache = Path(args.cache) if args.cache else None
    if cache is not None and cache.exists():
        paths_tt, paths_tr = pickle.loads(cache.read_bytes())
        print("routes loaded from", cache, flush=True)
    for i in (range(n_t) if not paths_tt else []):
        dist, prev = drain(vals, steps, [tcell[i]], [0.0], return_prev=True)
        dflat = dist.ravel()
        for j in range(i + 1, n_t):
            if np.isfinite(dflat[tcell[j]]):
                path = walk(prev, tcell[j])
                paths_tt[(i, j)] = (path, dflat[path].copy())
        for k_, c in enumerate(pc):
            if np.isfinite(dflat[c]):
                path = walk(prev, c)
                paths_tr[(i, k_)] = (path, dflat[path].copy())
        del dist, prev, dflat
        print(f"  turbine {i}: routes extracted", flush=True)
    if cache is not None and not cache.exists():
        cache.write_bytes(pickle.dumps((paths_tt, paths_tr)))
    del vals

    def route_graph(path_list, root_cell):
        """Graph of the union of exact routes: terminals, the root and every branching cell are nodes, chains are contracted.

        Edge weight of a step = difference of the exact float64 cumulative weights (the maximum where two routes use the same
        step); lengths from the step geometry. Every edge is a real trench stretch, so any design on the graph is a real layout.
        """
        from collections import defaultdict

        stepw: dict = {}
        for cells, cum in path_list:
            rr, cc = np.divmod(cells, w)
            ln = np.hypot(np.diff(rr), np.diff(cc))
            wd = np.diff(cum)
            for a_, b_, x_, l_ in zip(cells[:-1].tolist(), cells[1:].tolist(), wd.tolist(), ln.tolist()):
                key = (a_, b_) if a_ < b_ else (b_, a_)
                old = stepw.get(key)
                if old is None or x_ > old[0]:
                    stepw[key] = (x_, l_)
        adj = defaultdict(list)
        for (a_, b_), (x_, l_) in stepw.items():
            adj[a_].append((b_, x_, l_))
            adj[b_].append((a_, x_, l_))
        terminals = list(tcell) + [root_cell]
        nodes = set(terminals) | {c_ for c_, nb in adj.items() if len(nb) != 2}
        ids = {c_: n for n, c_ in enumerate(terminals)}
        for c_ in sorted(nodes - set(terminals)):
            ids[c_] = len(ids)
        edges: dict = {}
        for s_ in nodes:
            for nb, x_, l_ in adj[s_]:
                cost, length, prev_c, cur = x_, l_, s_, nb
                while cur not in nodes:
                    nxt = [e for e in adj[cur] if e[0] != prev_c]
                    if not nxt:
                        break
                    e = nxt[0]
                    cost += e[1]
                    length += e[2]
                    prev_c, cur = cur, e[0]
                if cur != s_ and cur in nodes:
                    ia, ib = ids[s_], ids[cur]
                    key = (min(ia, ib), max(ia, ib))
                    if key not in edges or cost < edges[key][0]:
                        edges[key] = (cost, length)
        return len(ids), [(a_, b_, (1.0 + omega) * c_, l_) for (a_, b_), (c_, l_) in edges.items()]

    ub_cache: dict = {}

    def price(k_):
        idx = int(pool[k_])
        if k_ in ub_cache:
            return ub_cache[k_]
        j, i = lat_xy(idx)
        if args.graph == "terminal":
            edges = [(a_, b_, (1.0 + omega) * float(cum[-1]), route_stats(cells)) for (a_, b_), (cells, cum) in paths_tt.items()]
            edges += [(a_, n_t, (1.0 + omega) * float(cum[-1]), route_stats(cells)) for (a_, kk), (cells, cum) in paths_tr.items() if kk == k_]
            n_nodes = n_t + 1
        else:
            # a graph the exact engine can solve in minutes: the cheapest spanning tree of the turbine-to-turbine routes plus the
            # root routes to the ``--root-routes`` cheapest turbines (every edge is still a real stretch, so it is still an upper bound)
            pair = sorted(paths_tt.items(), key=lambda kv: float(kv[1][1][-1]))
            parent = list(range(n_t))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            mst = []
            for (a_, b_), v_ in pair:
                ra, rb = find(a_), find(b_)
                if ra != rb:
                    parent[ra] = rb
                    mst.append(v_)
            roots = sorted(((i_, v_) for (i_, kk), v_ in paths_tr.items() if kk == k_), key=lambda iv: float(iv[1][1][-1]))[:args.root_routes]
            plist = mst + [v_ for _, v_ in roots]
            n_nodes, edges = route_graph(plist, pc[k_])
            print(f"    candidate {k_}: route graph with {n_nodes} nodes, {len(edges)} edges", flush=True)
        cgraph = CollectorGraph(n_nodes, edges)
        res = solve_collector(cgraph, list(range(n_t)), model, engine="B", root=n_t, keep_trace=feas is not None)
        mv_ub = float(res.mv[n_t])
        acc_ub, hv_ub = ub_of(A["acc_raw"], A["k_acc"], j, i), ub_of(A["hv_raw"], A["k_hv"], j, i)
        feas_rec = None
        if feas is not None and np.isfinite(mv_ub) and np.isfinite(hv_ub):
            lo_km, hi_km = leg_length_interval_km((float(x0 + 5 * i), float(y1 - 5 * j)), feas["tap_xy"], hv_ub, feas["w_min"])
            fr = make_feasible(res.design(n_t), cgraph, list(range(n_t)), model, feas["vmodel"], feas["rules"], feas["plant"],
                               length_grid_km(lo_km, hi_km, args.leg_step_km))
            feas_rec = {"feasible": fr.feasible, "repair_extra_eur": fr.extra_eur, "leg_interval_km": [lo_km, hi_km], **fr.record}
            mv_ub = mv_ub + fr.extra_eur                                      # inf if no repair passes: never an incumbent
        total = float(site[j, i]) + acc_ub + hv_ub + mv_ub + A["ledger"].k_min            # the 110 kV field already carries the connection-point bay or the tap works as its seed label
        rec = {"x": float(x0 + 5 * i), "y": float(y1 - 5 * j), "lb_eur": float(lb[j, i]), "site": float(site[j, i]),
               "access_ub": acc_ub, "hv_ub": hv_ub, "collector_ub": mv_ub, "total_ub": total,
               "collector_lb_trench_at_node": float(m_lb(j, i)), "graph_nodes": n_nodes, "graph_edges": len(edges),
               "feasibility": feas_rec}
        ub_cache[k_] = rec
        return rec

    import rasterio as _r
    with _r.open(m.file_of("mv_lb_5m_lattice")) as s:
        mv_lat = s.read(1)

    def m_lb(j, i):
        v = float(mv_lat[j, i])
        return float("inf") if v >= NODATA else v

    z = float("inf")
    history, priced = [z], []
    for ps in range(args.passes):
        cand = [k_ for k_ in range(len(pool)) if k_ not in ub_cache and flat_lb[pool[k_]] - args.eps_adm <= z][:args.f0]
        for k_ in cand:
            t0 = time.time()
            rec = price(k_)
            rec["price_seconds"] = round(time.time() - t0, 2)
            priced.append(rec)
            if np.isfinite(rec["total_ub"]):
                z = min(z, rec["total_ub"])
            print(f"  pass {ps + 1}: ({rec['x']:.0f}, {rec['y']:.0f}) LB {rec['lb_eur']:.0f}  UB {rec['total_ub']:.0f}  "
                  f"(site {rec['site']:.0f}, access {rec['access_ub']:.0f}, 110 kV {rec['hv_ub']:.0f}, collector {rec['collector_ub']:.0f})"
                  f"  [{rec['price_seconds']:.1f} s]"
                  + ("" if rec["feasibility"] is None else
                     f"  voltage rules: {'pass' if rec['feasibility']['feasible'] else 'FAIL'}, repair +{rec['feasibility']['repair_extra_eur']:.0f}"), flush=True)
        history.append(z)
        if not cand:
            break
    bad = [r for r in priced if np.isfinite(r["total_ub"]) and (r["lb_eur"] - args.eps_adm > r["total_ub"] + 1e-6
                                                                    or r["collector_lb_trench_at_node"] > r["collector_ub"] + 1e-6)]
    keep = (np.isfinite(lb) & (lb - args.eps_adm <= z)) if np.isfinite(z) else np.zeros(lb.shape, dtype=bool)
    jk, ik = np.nonzero(keep)
    res = {
        "date": dt.date.today().isoformat(),
        "objective": "F_model: pad site + access + 110 kV cable to a connection point + collector + K(b) of the branch-constants ledger",
        "ledger": A["ledger"].record(),
        "window_bound": A["window"],
        "scenario": f"all-cable; pad terminals on the anchor cell; MV level {args.u_kv:.0f} kV; engine B on the turbine-and-root route graph",
        "pool_mode": args.pool_mode,
        "feasibility": (f"every incumbent checked against the plant voltage rules ({args.operator}, {args.u_kv:g} kV), repaired by thicker cables if needed; "
                        "Z_UB is the cost of a feasible design (math.md M15, T-F). Reactive shortfalls are priced outside F_model and only listed.") if feas is not None else "NOT CHECKED: Z_UB is the cost of the collector optimum without the voltage rules, so the prune is not certified (math.md M15)",
        "stand_in_inputs": "NOT QUOTABLE", "condition_c": built.record.get("condition_c"), "omega": omega,
        "z_ub_history_eur": history, "z_ub_eur": z, "f0_positions_priced": priced,
        "admissibility_check_violations": bad,
        "E_b": {"rule": "LB - eps_adm <= Z_UB", "eps_adm": args.eps_adm, "kept": int(keep.sum()),
                "lattice_cells_finite_lb": int(np.isfinite(lb).sum()),
                "bbox_xy": [[float(x0 + 5 * ik.min()), float(y1 - 5 * jk.max())], [float(x0 + 5 * ik.max()), float(y1 - 5 * jk.min())]] if keep.any() else None},
    }
    out = RESULTS / f"incumbents_{dt.date.today().isoformat()}{args.tag}.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("f0_positions_priced",)}, indent=1))
    print("wrote", out)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
