"""Voltage on the #1277 incumbent tree: old LinDistFlow against exact AC.

Plan ``2026-09-25-voltage-limit-c5.md`` step 3 asks to reproduce the
appendix's LinDistFlow values on the incumbent tree (M2-05: 1.0053 p.u. at
66 kV, 1.0098 at 33 kV, 1.0156 at 20 kV) and to record the difference to an
exact AC load flow. Three columns:

1. **LinDistFlow** exactly as the 2026-09-23 review computed it: the
   TopoMILP catalogue resistance (``reff = r / n``), every segment carrying
   the rated power of the turbines below it, ``V^2 = 1 + sum 2 R P / U^2``,
   unity power factor, busbar at 1.0 p.u.;
2. **exact AC, same data**: the backward--forward sweep
   (:func:`pyorps.collector.voltage.load_flow`) on the same tree and the
   same resistances, x = 0, no charging, busbar 1.0 -- the linearisation
   error alone;
3. **exact AC, the C5 model**: the cables' 90 degC resistance, reactance and
   charging from ``voltage_2026.yaml`` (the same cross-sections, rounded up
   to the file's 150-630 mm2), the four corners with the turbine
   transformer and its taps (20 and 33 kV only; the file has no 66 kV data);
4. **the plant at the NAP** (:mod:`pyorps.collector.voltage_plant`): the
   same collector behind the UW transformer (the file's default: 63 MVA,
   115 kV, uk 12.5 %, +/-9 x 1.78 %) and a 110 kV leg, checked against the
   operators' TABs: operator A at the incumbent's station ST3 on the
   Niederselters line (0.65 km from S15, overhead or cable) and, for
   comparison, operator B at PCC2 (6.41 km).

The tree is a TopoMILP result, a radial feeder tree with joints at corridor
nodes (not a ``D_share`` design): every segment is its own cable.

Inputs (read only, retired by ``retire_pre_v3.py``):
``TopoMILP/results/cired2027/corridor_free_sites_pre_v3/milp_input/
corridor_graph_recommended.json`` and
``TopoMILP/results/cired2027/free_siting_variants_pre_v3.json``.

Run with the pyorps venv::

    .venv/Scripts/python.exe case_studies/runkel_free_siting/scripts/compare_1277_voltage.py
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from pyorps.collector.voltage import (
    Branch,
    ElectricalTree,
    check_tree,
    load_flow,
    voltage_model_from_yaml,
)
from pyorps.collector.voltage_plant import (
    HVLeg,
    check_plant_tree,
    grid_rules_from_yaml,
    plant_model_from_yaml,
)

HERE = Path(__file__).resolve().parent
CASE = HERE.parent
TOPO = CASE.parents[2] / "TopoMILP" / "results" / "cired2027"
GRAPH = TOPO / "corridor_free_sites_pre_v3/milp_input/corridor_graph_recommended.json"
VARIANTS = TOPO / "free_siting_variants_pre_v3.json"
YAML = CASE / "config" / "voltage_2026.yaml"
OMEGA = 2 * math.pi * 50

# TopoMILP/topomilp/formulations/cable_types.py, as transcribed by the
# 2026-09-23 review (scratch m2_incumbent.py): (mm2, kA, ohm/km, EUR/m)
CAT = {
    20.0: [(120, 0.286, 0.256, 40.0), (150, 0.323, 0.210, 45.0),
           (185, 0.361, 0.169, 50.0), (240, 0.419, 0.130, 60.0),
           (300, 0.475, 0.103, 70.0), (400, 0.538, 0.078, 80.0),
           (500, 0.600, 0.062, 95.0), (620, 0.675, 0.050, 110.0)],
    33.0: [(120, 0.270, 0.256, 50.0), (150, 0.307, 0.210, 55.0),
           (185, 0.345, 0.169, 60.0), (240, 0.400, 0.130, 70.0),
           (300, 0.455, 0.103, 80.0), (400, 0.515, 0.078, 95.0),
           (500, 0.575, 0.062, 110.0), (620, 0.645, 0.050, 130.0)],
    66.0: [(120, 0.255, 0.256, 70.0), (150, 0.290, 0.210, 80.0),
           (185, 0.320, 0.169, 90.0), (240, 0.375, 0.130, 100.0),
           (300, 0.430, 0.103, 120.0), (400, 0.490, 0.078, 140.0),
           (500, 0.550, 0.062, 160.0), (620, 0.620, 0.050, 190.0)],
}
PF, SF, MAXPAR, NBP, P_WT = 1.0, 1.10, 3, 6, 6.8
THETA = 1955.84
_Q = 1 / 1.04
LAM = 60.0 * _Q * (1 - _Q ** 25) / (1 - _Q)
YAML_MM2 = (150, 185, 240, 300, 400, 500, 630)


def combos(u):
    return [dict(cs=cs, n=n, cap=math.sqrt(3) * u * i * n * PF / SF,
                 reff=r / n, mat=m * n)
            for cs, i, r, m in CAT[u] for n in range(1, MAXPAR + 1)]


def rho(P, u):
    """The review's cable choice per segment: cheapest material + tangent
    loss among the options with capacity."""
    def tangent(cap):
        return max(2 * pb * P - pb * pb
                   for pb in [cap * k / (NBP - 1) for k in range(1, NBP)])
    best = (math.inf, None)
    for c in combos(u):
        if c["cap"] + 1e-9 < P:
            continue
        v = c["mat"] + LAM * THETA * c["reff"] / 1e3 * tangent(c["cap"]) / u ** 2
        if v < best[0]:
            best = (v, c)
    return best[1]


def load_tree():
    G = json.loads(GRAPH.read_text(encoding="utf-8"))
    R = json.loads(VARIANTS.read_text(encoding="utf-8"))
    edges = {e["edge_id"]: e for e in G["edges"]}
    adj = defaultdict(list)
    for e in R["segments"]:
        E = edges[e]
        adj[E["from_node"]].append((E["to_node"], e))
        adj[E["to_node"]].append((E["from_node"], e))
    root = G["terminal_nodes"]["S15"]
    turb = {G["terminal_nodes"][f"WT{i}"]: i for i in range(7)}
    parent, down, order = {}, {}, [root]
    stack, seen = [root], {root}
    while stack:                                   # iterative DFS
        v = stack.pop()
        for w, e in adj[v]:
            if w not in seen:
                seen.add(w)
                parent[w] = (v, e)
                order.append(w)
                stack.append(w)
    for v in reversed(order):                      # turbines below each edge
        s = {turb[v]} if v in turb else set()
        for w, e in adj[v]:
            if parent.get(w, (None,))[0] == v:
                s |= down[e]
        if v != root:
            down[parent[v][1]] = frozenset(s)
    return edges, root, turb, parent, down, order


def build(u, *, physical, vm=None):
    edges, root, turb, parent, down, order = load_tree()
    index = {root: 0}
    labels = [("root", 0)]
    for t in range(7):
        labels.append(("turbine", t))
    tnode = {v: 1 + i for v, i in turb.items()}
    for v in order:
        if v == root:
            continue
        if v in tnode:
            index[v] = tnode[v]
        else:
            index[v] = len(labels)
            labels.append(("junction", len(labels) - 8))
    branches, ldf = [], {}
    for v in order:
        if v == root:
            continue
        pv, e = parent[v]
        P = P_WT * len(down[e])
        c = rho(P, u)
        L = edges[e]["length_m"]
        if physical:
            mm2 = next(m for m in YAML_MM2 if m >= min(c["cs"], 630))
            cab = vm.cables[YAML_MM2.index(mm2)]
            n = c["n"]
            br = Branch(index[v], index[pv], cab.r_ohm_per_m * L / n,
                        cab.x_ohm_per_m * L / n,
                        OMEGA * cab.c_f_per_m * L * n)
        else:
            br = Branch(index[v], index[pv], c["reff"] * L / 1e3, 0.0, 0.0)
        branches.append(br)
        ldf[v] = (pv, 2 * c["reff"] * L / 1e3 * P / u ** 2)
    tree = ElectricalTree(u, labels, branches, {t: 1 + t for t in range(7)})
    # LinDistFlow of the review: worst sum over turbine paths
    worst = 0.0
    for v in turb:
        s, x = 0.0, v
        while x != root:
            px, d = ldf[x]
            s += d
            x = px
        worst = max(worst, s)
    return tree, math.sqrt(1 + worst)


def yaml_data():
    import yaml

    return yaml.safe_load(YAML.read_text(encoding="utf-8"))


def main():
    rows = {}
    for u in (66.0, 33.0, 20.0):
        tree, v_ldf = build(u, physical=False)
        s = [0j] * tree.n_nodes
        for t in range(7):
            s[1 + t] = complex(P_WT, 0.0)
        lf = load_flow(tree, s, 1.0)
        v_ac = max(abs(lf.v_pu[1 + t]) for t in range(7))
        rows[u] = {"ldf_v_pu": v_ldf, "exact_same_data_v_pu": v_ac,
                   "ldf_minus_exact": v_ldf - v_ac,
                   "segments": len(tree.branches)}
        if u in (20.0, 33.0):
            vm = voltage_model_from_yaml(YAML, u_kv=u,
                                         cable_mm2=list(YAML_MM2),
                                         n_turbines=7)
            ptree, _ = build(u, physical=True, vm=vm)
            chk = check_tree(ptree, vm)
            rows[u]["c5"] = {
                "passed": chk.passed,
                "max_v_pu": {k: c.max_v_pu for k, c in chk.corners.items()},
                "tap_kv": chk.tap_kv,
                "margin_pu": chk.margin_pu,
                "um_margin_kv": chk.um_margin_kv,
                "violations": chk.violations,
                "cable_class": vm.meta["cable_class"],
            }
            d = yaml_data()["hv_leg"]
            ohl = d["ohl_243_al1_39"]["value"]
            cab = d["cable_n2xsfl2y_1x300"]["value"]
            plants = {}
            for op, km in (("operator_a", 0.0), ("operator_a", 0.65),
                           ("operator_b", 6.41)):
                legs = ([("co-located", HVLeg())] if km == 0 else
                        [(f"{km} km OHL", HVLeg.from_type(
                            r_ohm_per_km=ohl["r_ohm_per_km"],
                            x_ohm_per_km=ohl["x_ohm_per_km"],
                            c_nf_per_km=ohl["c_nf_per_km"], length_km=km)),
                         (f"{km} km cable", HVLeg.from_type(
                             r_ohm_per_km=cab["r_ohm_per_km"],
                             x_ohm_per_km=cab["x_ohm_per_km"],
                             c_nf_per_km=cab["c_nf_per_km"], length_km=km))])
                for leg_name, leg in legs:
                    plant = plant_model_from_yaml(YAML, vm, hv_leg=leg)
                    pc = check_plant_tree(ptree, vm, plant,
                                          grid_rules_from_yaml(YAML, op))
                    plants[f"{op}, {leg_name}"] = pc.as_record()
            rows[u]["plant"] = plants
    print("| kV | LinDistFlow (review) | exact AC, same data | LDF - exact |")
    print("|---|---:|---:|---:|")
    for u, r in rows.items():
        print(f"| {u:.0f} | {r['ldf_v_pu']:.5f} | "
              f"{r['exact_same_data_v_pu']:.5f} | "
              f"{r['ldf_minus_exact']:+.2e} |")
    for u in (20.0, 33.0):
        print(f"\n{u:.0f} kV at the NAP:")
        print("| operator, HV leg | passed | Q deficit (Mvar) | no-load Q "
              "(Mvar) | reactor (Mvar) | S3 max U/U_NS | turbine taps (kV) |")
        print("|---|---|---:|---:|---:|---:|---|")
        for name, rec in rows[u]["plant"].items():
            s3 = rec["corners"].get("S3", {}).get("max_ns", float("nan"))
            print(f"| {name} | {rec['passed']} | {rec['q_deficit_mvar']:.2f} "
                  f"| {rec['no_load_q_mvar']:+.2f} | {rec['reactor_mvar']:.2f}"
                  f" | {s3:.4f} | {sorted(set(rec['tap_kv'].values()))} |")
            for v in rec["violations"][:4]:
                print("   -", v)
            if len(rec["violations"]) > 4:
                print(f"   - ... {len(rec['violations']) - 4} more")
    for u in (20.0, 33.0):
        c = rows[u]["c5"]
        print(f"\n{u:.0f} kV, C5 model: passed {c['passed']}; highest |V| "
              + ", ".join(f"{k} {v:.4f}" for k, v in c["max_v_pu"].items())
              + f"; taps {sorted(set(c['tap_kv'].values()))} kV; worst band "
              f"margin {min(c['margin_pu'].values()):+.4f} p.u.; U_m margin "
              f"{c['um_margin_kv']:.2f} kV")
        for v in c["violations"]:
            print("   ", v)
    out = CASE / "results" / "voltage_1277_comparison.json"
    out.write_text(json.dumps({str(k): v for k, v in rows.items()},
                              indent=1, default=str), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
