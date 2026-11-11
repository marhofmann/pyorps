#!/usr/bin/env python
"""Joint, time-step-matched per-turbine power categories and their square sums W(X).

Plan 2026-09-23 (revision 5), Phase C11 -- the Runkel seven-turbine case study.

A category k is a class of hours of the DWD 06344 record, binned by free-stream
hub-height speed (width dv) and wind direction (width dtheta).  It carries its
hours per year h_k and the simultaneous, wake-resolved output of every turbine
P_{t,k}, averaged over the SAME hours (time-step matched), so the joint
behaviour of the turbines is kept (plan check K2).  The engines consume only
the category square sums

    W(X) = sum_k h_k |sum_{t in X} S_{t,k}|^2      [MVA^2 h per year]

for the 127 non-empty turbine subsets X, with S = P + jQ.  W is a quadratic
form in the subset indicator, so W(X) = 1_X^T M 1_X with the 7x7 matrix
M = sum_k h_k (P_k P_k^T + Q_k Q_k^T); M is stored as well.

Exactness.  The wake model is a deterministic function of the recorded state
(F, D), and the record is quantised (0.1 m/s, 10 deg), so dv = 0 with
dtheta = 10 -- one category per distinct record state -- IS the hourly limit.
For any width the within-category covariance C_k of S is stored, and
W_var = W_cat + sum_k h_k 1_X^T C_k 1_X reproduces the hourly W exactly;
W_cat <= W_exact for every X (variance decomposition): the mean-only table
errs low, never high.

Conventions (also written into the meta JSON):
* hub speed  v = s * F * (179/10)^alpha, power law, alpha = 0.20 by default;
* level      s solved so that the NET (waked) farm capacity factor over the
             record equals --cf (0.30); wakes therefore remove energy once;
* wakes      PyWake Bastankhah_PorteAgel_2014, UniformSite(p_wd=[1], ti=0.10),
             k = 0.0324555, GenericWindTurbine(D = 175 m, H = 179 m, 6800 kW),
             i.e. the TopoMILP PyWakeModel.  The effective speed WS_eff of
             every turbine is mapped through the chosen curve; thrust always
             comes from the generic turbine (no public Ct curve);
* cut-out    whole farm off when the free stream exceeds 20 m/s (a stopped
             machine casts no wake, so by induction from the most upstream
             machine every machine stops); cut-in: the PyWake curve as PyWake
             gives it (TI-smoothed, as in the TopoMILP export), cubic curves
             at 3 m/s;
* reactive   Q = P tan(acos(cosphi)), default cosphi = 1 -> Q = 0; Q(P=0) = 0
             (Nordex Rev. 11 standard: converter off at standstill);
* direction  bins centred on multiples of dtheta (the TopoMILP sector rule);
             D = 0 (103 calm-coded hours, F <= 1.6 m/s) is binned as north;
* hours      h_k = n_k * 8760 / N_valid; gaps stay gaps (no infilling).

Runs in the TopoMILP venv (numpy, py_wake 2.6.x).  Imports neither topomilp
nor pyorps.  Example:
    python build_loss_categories.py --dv 0.5 --dtheta 10 --curve pywake \
        --alpha 0.20 --cf 0.30 --out loss_categories.npz --report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HOURS_PER_YEAR = 8760.0
SENSOR_HEIGHT_M = 10.0
HUB_HEIGHT_M = 179.0
ROTOR_DIAMETER_M = 175.0          # TopoMILP N175_ROTOR_DIAMETER_M
RATED_MW = 6.8
CUT_IN_MS = 3.0
CUT_OUT_MS = 20.0
AMBIENT_TI = 0.10
WAKE_K = 0.0324555
AIR_DENSITY = 1.225
DWD_MISSING = -999.0
#: Deutsche WindGuard reference-yield regression (plan prototype refyield.py):
#: VLS_ref = 3918.1 - 5.89 * specific power [W/m^2] + 8.88 * hub height [m].
SPECIFIC_POWER_W_M2 = 282.7

_TOPO = Path(r"C:\Users\mhnn82\Documents\2_python_projects\TopoMILP\data\cired2026\input")
DEFAULT_DWD = _TOPO / "produkt_ff_stunde_20101001_20241231_06344.txt"
DEFAULT_WEA = _TOPO / "wea.geojson"

GRID_DV = (0.0, 0.25, 0.5, 1.0, 2.0)
GRID_DTHETA = (10.0, 30.0)
BUILDER_VERSION = "c11-build_loss_categories 1.0 (2026-09-24)"


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
def read_dwd(path: Path) -> dict:
    """DWD produkt_ff_stunde file WITH timestamps (MESS_DATUM, UTC)."""
    raw = path.read_bytes()
    lines = raw.decode("latin-1").splitlines()
    head = [c.strip() for c in lines[0].split(";")]
    i_t, i_f, i_d = head.index("MESS_DATUM"), head.index("F"), head.index("D")
    stamp, f, d = [], [], []
    for line in lines[1:]:
        if not line.strip():
            continue
        cols = line.split(";")
        stamp.append(cols[i_t].strip())
        f.append(float(cols[i_f]))
        d.append(float(cols[i_d]))
    iso = [f"{s[:4]}-{s[4:6]}-{s[6:8]}T{s[8:10]}" for s in stamp]
    unix = np.array(iso, dtype="datetime64[h]").astype("datetime64[s]").astype(np.int64)
    if np.any(np.diff(unix) <= 0):
        raise ValueError("MESS_DATUM is not strictly increasing (duplicates?)")
    f = np.asarray(f, dtype=float)
    d = np.asarray(d, dtype=float)
    valid = (f != DWD_MISSING) & np.isfinite(f) & (f >= 0) & (d >= 0) & (d <= 360)
    t = unix[valid]
    step_h = np.diff(t) // 3600
    n_span = int((t[-1] - t[0]) // 3600 + 1)
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "path": str(path),
        "n_rows": int(len(f)),
        "n_flagged": int((~valid).sum()),
        "unix": t,
        "F": f[valid],
        "D": d[valid],
        "n_valid": int(valid.sum()),
        "n_span_hours": n_span,
        "n_missing_hours": n_span - int(valid.sum()),
        "n_gaps": int((step_h > 1).sum()),
        "max_gap_hours": int(step_h.max() - 1),
        "first_utc": str(np.datetime64(int(t[0]), "s")),
        "last_utc": str(np.datetime64(int(t[-1]), "s")),
        "n_calm_coded_D0": int((d[valid] == 0).sum()),
    }


def read_positions(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Turbine E/N read as TopoMILP load_wind_turbines reads them (properties)."""
    fc = json.loads(path.read_text(encoding="utf-8"))
    xy, ids, rated = [], [], []
    for feat in fc["features"]:
        p = feat["properties"]
        e = float(p["OSTWERT_UTM32N"])
        if e > 30_000_000:
            e -= 32_000_000
        xy.append((e, float(p["NORDWERT_UTM32N"])))
        ids.append(str(p["ID"]))
        rated.append(float(str(p["ELEKTR_LEISTUNG"]).replace(",", ".")))
    return np.asarray(xy, dtype=float), ids, np.asarray(rated, dtype=float)


# --------------------------------------------------------------------------
# turbine, curve, wakes
# --------------------------------------------------------------------------
class Farm:
    """The TopoMILP PyWake set-up, evaluated per hour (time=True)."""

    def __init__(self, xy: np.ndarray, rotor_d: float = ROTOR_DIAMETER_M,
                 hub_m: float = HUB_HEIGHT_M, rated_mw: float = RATED_MW):
        from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
        from py_wake.site import UniformSite
        from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
        self.turbine = GenericWindTurbine(
            name=f"generic_{rated_mw:.1f}MW", diameter=rotor_d,
            hub_height=hub_m, power_norm=rated_mw * 1000.0)
        self.wfm = Bastankhah_PorteAgel_2014(
            UniformSite(p_wd=[1.0], ti=AMBIENT_TI), self.turbine, k=WAKE_K)
        self.x = xy[:, 0] - xy[:, 0].mean()
        self.y = xy[:, 1] - xy[:, 1].mean()
        self.n = len(xy)
        self.rotor_d = rotor_d
        self.rated_mw = rated_mw

    def ws_eff(self, ws: np.ndarray, wd: np.ndarray, chunk: int = 20000) -> np.ndarray:
        """Effective hub speed of every turbine, one row per (ws, wd) pair."""
        out = np.empty((len(ws), self.n))
        for a in range(0, len(ws), chunk):
            sim = self.wfm(self.x, self.y, wd=wd[a:a + chunk], ws=ws[a:a + chunk], time=True)
            out[a:a + chunk] = np.asarray(sim.WS_eff).reshape(self.n, -1).T
        return out


def make_curve(name: str, farm: Farm):
    """Per-unit output lambda(v) WITHOUT the cut-out (applied to the free stream)."""
    if name == "pywake":
        norm = farm.rated_mw * 1e6

        def lam(v):
            v = np.asarray(v, dtype=float)
            out = np.asarray(farm.turbine.power(v.ravel()), dtype=float).reshape(v.shape)
            return np.clip(out / norm, 0.0, 1.0)
        return lam
    if name.startswith("cubic"):
        vr = float(name[len("cubic"):].lstrip(":"))
        if not CUT_IN_MS < vr <= CUT_OUT_MS:
            raise ValueError(f"cubic rated speed {vr} outside ({CUT_IN_MS}, {CUT_OUT_MS}]")

        def lam(v):
            v = np.asarray(v, dtype=float)
            out = np.zeros_like(v)
            r = (v >= CUT_IN_MS) & (v < vr)
            out[r] = (v[r] ** 3 - CUT_IN_MS ** 3) / (vr ** 3 - CUT_IN_MS ** 3)
            out[v >= vr] = 1.0
            return out
        return lam
    raise ValueError(f"unknown curve {name!r}: pywake | cubic<rated>, e.g. cubic12.5")


def free_curve(lam):
    """The curve with the cut-out applied (free-stream use)."""
    def f(v):
        v = np.asarray(v, dtype=float)
        return np.where(v > CUT_OUT_MS, 0.0, lam(v))
    return f


def state_outputs(farm: Farm, lam, v_free: np.ndarray, wd: np.ndarray):
    """Per-turbine MW per distinct (v, wd) state, and the hour -> state map.

    PyWake runs once per distinct state; identical states give identical
    outputs, so this is the exact per-hour evaluation, only deduplicated.
    """
    uniq, inv = np.unique(np.column_stack([v_free, wd]), axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    p = np.zeros((len(uniq), farm.n))
    run = (uniq[:, 0] > 0.0) & (uniq[:, 0] <= CUT_OUT_MS)
    if run.any():
        p[run] = farm.rated_mw * lam(farm.ws_eff(uniq[run, 0], uniq[run, 1]))
    return p, uniq, inv


# --------------------------------------------------------------------------
# level
# --------------------------------------------------------------------------
def solve_scale(cf_of_s, target: float, s0: float = 0.5, step: float = 0.05,
                s_max: float = 4.0, tol: float = 1e-12) -> float:
    """Lowest s with cf(s) = target (cf is not monotone past the cut-out)."""
    lo = s0
    if cf_of_s(lo) >= target:
        raise ValueError(f"target CF {target} already exceeded at s = {s0}")
    hi = lo + step
    while cf_of_s(hi) < target:
        lo, hi = hi, hi + step
        if hi > s_max:
            raise ValueError(f"target CF {target} unreachable for s <= {s_max}")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if cf_of_s(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * hi:
            break
    return hi


# --------------------------------------------------------------------------
# categories and square sums
# --------------------------------------------------------------------------
def subsets(n: int):
    """Masks 1 .. 2^n - 1; bit t (value 2^t) is turbine t (wea feature order)."""
    masks = np.arange(1, 2 ** n, dtype=np.int64)
    member = ((masks[:, None] >> np.arange(n)) & 1).astype(bool)
    return masks, member


def w_of(m: np.ndarray, member: np.ndarray) -> np.ndarray:
    """W(X) = 1_X^T M 1_X for every subset."""
    b = member.astype(float)
    return np.einsum("si,ij,sj->s", b, m, b)


def w_min_by_size(w: np.ndarray, member: np.ndarray) -> np.ndarray:
    size = member.sum(1)
    return np.array([w[size == k].min() for k in range(1, member.shape[1] + 1)])


def categorise(v: np.ndarray, d: np.ndarray, dv: float, dtheta: float):
    """Category of every hour: (floor(v/dv), direction bin centred on k*dtheta)."""
    n_dir = int(round(360.0 / dtheta))
    if abs(n_dir * dtheta - 360.0) > 1e-9:
        raise ValueError("dtheta must divide 360")
    di = np.floor(((d % 360.0) + 0.5 * dtheta) / dtheta).astype(np.int64) % n_dir
    if dv > 0:
        vi = np.floor(v / dv).astype(np.int64)
        speed_values = None
    else:   # every distinct speed is its own bin (the exact limit)
        speed_values, vi = np.unique(v, return_inverse=True)
        vi = np.asarray(vi).reshape(-1).astype(np.int64)
    key = vi * n_dir + di
    uk, cat = np.unique(key, return_inverse=True)
    return np.asarray(cat).reshape(-1), uk // n_dir, uk % n_dir, n_dir, speed_values


def category_table(P: np.ndarray, Q: np.ndarray, v: np.ndarray, d: np.ndarray,
                   dv: float, dtheta: float) -> dict:
    """Time-matched category means, hours and within-category covariances."""
    n_hours, n = P.shape
    cat, vbin, dbin, n_dir, speed_values = categorise(v, d, dv, dtheta)
    K = len(vbin)
    cnt = np.bincount(cat, minlength=K).astype(float)
    h = cnt * HOURS_PER_YEAR / n_hours
    pm = np.stack([np.bincount(cat, P[:, t], K) for t in range(n)], 1) / cnt[:, None]
    qm = np.stack([np.bincount(cat, Q[:, t], K) for t in range(n)], 1) / cnt[:, None]
    dp, dq = P - pm[cat], Q - qm[cat]
    cov = np.zeros((K, n, n))
    for i in range(n):
        for j in range(i, n):
            c = np.bincount(cat, dp[:, i] * dp[:, j] + dq[:, i] * dq[:, j], K) / cnt
            cov[:, i, j] = c
            cov[:, j, i] = c
    m_cat = (pm * h[:, None]).T @ pm + (qm * h[:, None]).T @ qm
    m_within = np.einsum("k,kij->ij", h, cov)
    return {"K": K, "cat": cat, "vbin": vbin, "dbin": dbin, "n_dir": n_dir,
            "speed_values": speed_values, "count": cnt, "h": h, "P": pm, "Q": qm,
            "cov": cov, "v_mean": np.bincount(cat, v, K) / cnt,
            "M_cat": m_cat, "M_within": m_within}


# --------------------------------------------------------------------------
# one configuration
# --------------------------------------------------------------------------
def build(rec: dict, xy: np.ndarray, *, curve: str, alpha: float, cf: float,
          cosphi: float = 1.0, rotor_d: float = ROTOR_DIAMETER_M,
          farm: Farm | None = None, level: str = "net") -> dict:
    """Hourly per-turbine output for one (curve, alpha, cf) at the solved level."""
    farm = farm or Farm(xy, rotor_d=rotor_d)
    lam = make_curve(curve, farm)
    base = rec["F"] * (HUB_HEIGHT_M / SENSOR_HEIGHT_M) ** alpha
    wd = rec["D"] % 360.0
    n_cap = farm.n * farm.rated_mw * len(base)

    def cf_net(s):
        p, _, inv = state_outputs(farm, lam, s * base, wd)
        return float(p[inv].sum()) / n_cap

    def cf_gross(s):
        return float(np.mean(free_curve(lam)(s * base)))

    if level == "net":
        s = solve_scale(cf_net, cf)
    elif level == "gross":
        s = solve_scale(cf_gross, cf)
    else:
        s = 1.0
    v = s * base
    p_state, states, inv = state_outputs(farm, lam, v, wd)
    P = p_state[inv]
    tanphi = math.tan(math.acos(cosphi))
    Q = P * tanphi
    lam_free = free_curve(lam)(v)
    m_exact = HOURS_PER_YEAR / len(v) * (P.T @ P + Q.T @ Q)
    return {"farm": farm, "lam": lam, "s": s, "v": v, "wd": wd, "P": P, "Q": Q,
            "p_state": p_state, "states": states, "state_of_hour": inv,
            "tanphi": tanphi, "cf_net": float(P.sum() / n_cap),
            "cf_gross": float(lam_free.mean()),
            "cf_turbine": P.mean(0) / farm.rated_mw,
            "theta_coherent_free_h": float(np.mean(lam_free ** 2) * HOURS_PER_YEAR),
            "M_exact": m_exact, "curve": curve, "alpha": alpha, "cf_target": cf,
            "cosphi": cosphi, "rotor_d": farm.rotor_d}


# --------------------------------------------------------------------------
# cross-checks
# --------------------------------------------------------------------------
def weibull_moments(a: float, k: float, lam_free, v_max: float = 40.0,
                    n_steps: int = 4000) -> tuple[float, float]:
    """CF and Theta/8760 of a Weibull site (Simpson, as TopoMILP theta_from_weibull)."""
    v = np.linspace(0.0, v_max, n_steps + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdf = (k / a) * (v / a) ** (k - 1.0) * np.exp(-((v / a) ** k))
    pdf[0] = 0.0 if k > 1 else pdf[1]
    pdf = np.nan_to_num(pdf)
    hstep = v[1] - v[0]

    def simpson(y):
        return float(hstep / 3.0 * (y[0] + y[-1] + 4.0 * y[1:-1:2].sum()
                                    + 2.0 * y[2:-2:2].sum()))
    lam = lam_free(v)
    mass = simpson(pdf)
    return simpson(lam * pdf) / mass, simpson(lam ** 2 * pdf) / mass


def weibull_calibration(rec: dict, alpha: float, lam_free, cf: float) -> dict:
    """The TopoMILP route: Justus k from the record, A solved to the FREE-STREAM CF."""
    v = rec["F"] * (HUB_HEIGHT_M / SENSOR_HEIGHT_M) ** alpha
    v = v[v > 0]
    k = (float(np.std(v)) / float(np.mean(v))) ** -1.086
    a_raw = float(np.mean(v)) / math.gamma(1.0 + 1.0 / k)
    lo, hi = 1.0, 25.0
    while hi - lo > 1e-9:
        mid = 0.5 * (lo + hi)
        if weibull_moments(mid, k, lam_free)[0] < cf:
            lo = mid
        else:
            hi = mid
    a_cal = 0.5 * (lo + hi)
    cf_w, th_w = weibull_moments(a_cal, k, lam_free)
    return {"k": k, "A_raw": a_raw, "A_cal": a_cal, "ratio": a_cal / a_raw,
            "cf": cf_w, "theta_h": th_w * HOURS_PER_YEAR}


def curve_report(lam, rotor_d: float) -> dict:
    """Per-unit points, implied Cp*eta, knee speeds, EEG reference-site FLH."""
    lf = free_curve(lam)
    area = math.pi * (rotor_d / 2.0) ** 2
    out = {f"pu@{v:g}": float(lf(np.array([v]))[0])
           for v in (8.0, 10.0, 10.3, 12.0, 12.5, 13.0)}
    for v in (6.0, 8.0, 10.0):
        out[f"cp_eta@{v:g}"] = (float(lf(np.array([v]))[0]) * RATED_MW * 1e6
                                / (0.5 * AIR_DENSITY * area * v ** 3))
    grid = np.arange(0.0, CUT_OUT_MS + 1e-9, 0.001)
    g = lf(grid)
    for q in (0.95, 0.99, 0.995):
        out[f"v@{q:g}"] = float(grid[np.argmax(g >= q)])
    # EEG reference site: Rayleigh, 6.45 m/s at 100 m, Hellmann 0.25, 1.225 kg/m^3.
    vmean = 6.45 * (HUB_HEIGHT_M / 100.0) ** 0.25
    cf_ref, _ = weibull_moments(vmean / math.gamma(1.5), 2.0, lf)
    flh_wg = 3918.1 - 5.89 * SPECIFIC_POWER_W_M2 + 8.88 * HUB_HEIGHT_M
    out["FLH_ref_h"] = cf_ref * HOURS_PER_YEAR
    out["FLH_ref_windguard_h"] = flh_wg
    out["FLH_ref_ratio"] = cf_ref * HOURS_PER_YEAR / flh_wg
    return out


def width_grid(res: dict, member: np.ndarray, grid_dv, grid_dtheta):
    """W_cat for every width setting, its gap to the hourly W, and W_min."""
    w_exact = w_of(res["M_exact"], member)
    rows = []
    for dth in grid_dtheta:
        for dv in grid_dv:
            tab = category_table(res["P"], res["Q"], res["v"], res["wd"], dv, dth)
            w_cat = w_of(tab["M_cat"], member)
            w_var = w_of(tab["M_cat"] + tab["M_within"], member)
            gap = w_cat / w_exact - 1.0
            rows.append({"dv": dv, "dtheta": dth, "K": int(tab["K"]), "W_cat": w_cat,
                         "gap": gap, "gap_all": float(gap[-1]),
                         "worst_gap": float(gap.min()),
                         "worst_mask": int(np.argmin(gap) + 1),
                         "var_err": float(np.abs(w_var / w_exact - 1.0).max()),
                         "W_min": w_min_by_size(w_cat, member)})
    return w_exact, rows


# --------------------------------------------------------------------------
# report and main
# --------------------------------------------------------------------------
#: (label, curve, alpha, cf, rotor); None takes the command-line value.
SENSITIVITIES = (
    ("headline", None, None, None, None),
    ("knee 12.5 m/s (cubic)", "cubic12.5", None, None, None),
    ("10.3 m/s cubic (comparison)", "cubic10.3", None, None, None),
    ("alpha 0.14", None, 0.14, None, None),
    ("alpha 0.26", None, 0.26, None, None),
    ("CF 0.25", None, None, 0.25, None),
    ("CF 0.36", None, None, 0.36, None),
    ("rotor 176 m", None, None, None, 176.0),
)


def report(args, rec, xy, farm, res, member, w_exact, grid, tab, level):
    s_cap = (len(xy) * RATED_MW) ** 2
    keys = ("sha256", "n_rows", "n_flagged", "n_valid", "n_span_hours", "n_missing_hours",
            "n_gaps", "max_gap_hours", "first_utc", "last_utc", "n_calm_coded_D0")
    print("DWD", {k: rec[k] for k in keys})
    print("distinct (F, D) states:", len(res["states"]))
    print("\n== power curves (free stream, cut-out 20 m/s) ==")
    for cname in ("pywake", "cubic12.5", "cubic10.3"):
        cr = curve_report(make_curve(cname, farm), farm.rotor_d)
        print(f"{cname:10s}", {k: round(v, 4) for k, v in cr.items()})
    print("\n== level ==")
    print(json.dumps(level, indent=1))
    print("per-turbine CF_net", np.round(res["cf_turbine"], 5))
    e_hour = res["P"].sum(0) * HOURS_PER_YEAR / len(res["v"])
    e_cat = (tab["P"] * tab["h"][:, None]).sum(0)
    print("energy MWh/a per turbine", np.round(e_hour, 3), "farm", round(float(e_hour.sum()), 3),
          "max abs(cat - hourly)", f"{np.abs(e_cat - e_hour).max():.2e}")
    print(f"\n== width grid; W_exact(all) = {w_exact[-1]:.6e} MVA^2 h/a,"
          f" theta_exact = {w_exact[-1] / s_cap:.3f} h ==")
    print("dtheta    dv       K    W_cat(all)    gap_all    worst_gap (mask)  var_err  theta_cat")
    for g in grid:
        print(f"{g['dtheta']:6.0f} {g['dv']:5.2f} {g['K']:7d}  {g['W_cat'][-1]:.6e}  "
              f"{g['gap_all']:+.3e}  {g['worst_gap']:+.3e} ({g['worst_mask']:3d})  "
              f"{g['var_err']:.1e}  {g['W_cat'][-1] / s_cap:.3f}")
    wme = w_min_by_size(w_exact, member)
    print("W_min exact, k = 1..7:", np.array2string(wme, precision=6, max_line_width=250))
    for g in grid:
        print(f"  W_min rel gap dtheta {g['dtheta']:.0f} dv {g['dv']:.2f}:",
              np.array2string(g["W_min"] / wme - 1.0, precision=3, max_line_width=250))
    print("W_exact singletons:",
          np.array2string(w_exact[[0, 1, 3, 7, 15, 31, 63]], precision=6, max_line_width=250))
    print("M_exact:\n", np.array2string(res["M_exact"], precision=1, max_line_width=250))
    print("\n== sensitivities (W in MVA^2 h/a; theta = W(all) / (7 x 6.8 MW)^2) ==")
    for lab, c, a, f, r in SENSITIVITIES:
        c = c or args.curve
        a = args.alpha if a is None else a
        f = args.cf if f is None else f
        r = args.rotor if r is None else r
        fm = farm if r == farm.rotor_d else Farm(xy, rotor_d=r)
        rr = build(rec, xy, curve=c, alpha=a, cf=f, cosphi=args.cosphi, farm=fm)
        we = w_of(rr["M_exact"], member)
        tb = category_table(rr["P"], rr["Q"], rr["v"], rr["wd"], args.dv, args.dtheta)
        wc = w_of(tb["M_cat"], member)
        wbs = weibull_calibration(rec, a, free_curve(rr["lam"]), f)
        prod = rr["s"] * (HUB_HEIGHT_M / SENSOR_HEIGHT_M) ** a
        print(f"{lab:28s} s {rr['s']:.6f} (s*17.9^a {prod:.6f}) CFnet {rr['cf_net']:.6f} "
              f"CFfree {rr['cf_gross']:.5f} W_exact(all) {we[-1]:.6e} theta {we[-1] / s_cap:8.2f} h;"
              f" W_cat {wc[-1]:.6e} theta {wc[-1] / s_cap:8.2f} h worst {np.min(wc / we - 1.0):+.2e};"
              f" Weibull A_cal/A_raw {wbs['ratio']:.4f} theta_W {wbs['theta_h']:.1f} h")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dv", type=float, default=0.5, help="hub-speed bin width, m/s (0 = exact)")
    ap.add_argument("--dtheta", type=float, default=10.0, help="direction bin width, deg")
    ap.add_argument("--curve", default="pywake", help="pywake, cubic12.5, cubic10.3 or cubic<v>")
    ap.add_argument("--alpha", type=float, default=0.20, help="shear exponent")
    ap.add_argument("--cf", type=float, default=0.30, help="NET farm capacity factor target")
    ap.add_argument("--cosphi", type=float, default=1.0, help="reactive rule Q = P tan(acos(cosphi))")
    ap.add_argument("--rotor", type=float, default=ROTOR_DIAMETER_M, help="rotor diameter, m")
    ap.add_argument("--dwd", type=Path, default=DEFAULT_DWD)
    ap.add_argument("--wea", type=Path, default=DEFAULT_WEA)
    ap.add_argument("--out", type=Path, default=None, help=".npz to write (omit: write nothing)")
    ap.add_argument("--report", action="store_true", help="print curves, level, grid, sensitivities")
    args = ap.parse_args(argv)

    import py_wake
    t0 = time.time()
    rec = read_dwd(args.dwd)
    xy, ids, rated = read_positions(args.wea)
    if not np.allclose(rated, RATED_MW):
        raise ValueError(f"wea ratings {rated} are not all {RATED_MW} MW")
    masks, member = subsets(len(xy))
    farm = Farm(xy, rotor_d=args.rotor)
    res = build(rec, xy, curve=args.curve, alpha=args.alpha, cf=args.cf,
                cosphi=args.cosphi, farm=farm)
    w_exact, grid = width_grid(res, member, GRID_DV, GRID_DTHETA)
    tab = category_table(res["P"], res["Q"], res["v"], res["wd"], args.dv, args.dtheta)
    w_cat = w_of(tab["M_cat"], member)
    w_var = w_of(tab["M_cat"] + tab["M_within"], member)
    lam_free = free_curve(res["lam"])
    base = rec["F"] * (HUB_HEIGHT_M / SENSOR_HEIGHT_M) ** args.alpha
    s_gross = solve_scale(lambda s: float(np.mean(lam_free(s * base))), args.cf)
    s_cap = (len(xy) * RATED_MW) ** 2
    level = {
        "rule": "single hub-speed scale s: NET (waked) farm CF over the record = cf",
        "s_net": res["s"], "cf_net": res["cf_net"], "cf_free_stream_at_s_net": res["cf_gross"],
        "wake_loss_fraction": 1.0 - res["cf_net"] / res["cf_gross"],
        "theta_coherent_free_at_s_net_h": res["theta_coherent_free_h"],
        "s_gross_free_stream_cf": s_gross,
        "theta_coherent_free_at_s_gross_h":
            float(np.mean(lam_free(s_gross * base) ** 2) * HOURS_PER_YEAR),
        "cf_free_at_s1": float(np.mean(lam_free(base))),
        "weibull_topomilp_route": weibull_calibration(rec, args.alpha, lam_free, args.cf),
    }
    try:
        script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except (NameError, OSError):
        script_sha = None
    meta = {
        "builder": BUILDER_VERSION, "builder_sha256": script_sha,
        "plan": "2026-09-23 proven-optimal substation plan rev. 5, Phase C11 (2.1, 2.7, B6)",
        "curve": args.curve, "alpha": args.alpha, "cf_target_net": args.cf, "level": level,
        "cf_turbine_net": res["cf_turbine"].tolist(),
        "hub_speed": "v = s * F * (179/10)^alpha, F = DWD 10 m hourly mean",
        "reactive_rule": f"Q = P tan(acos({args.cosphi})); Q(P=0) = 0", "cosphi": args.cosphi,
        "wake_model": {"py_wake": py_wake.__version__, "model": "Bastankhah_PorteAgel_2014",
                       "site": "UniformSite(p_wd=[1], ti=0.10)", "k": WAKE_K,
                       "turbine": "GenericWindTurbine (PyWake defaults)", "rotor_d_m": args.rotor,
                       "hub_m": HUB_HEIGHT_M, "rated_kw": RATED_MW * 1000,
                       "evaluation": "exact per hour: PyWake time=True on the distinct (v, wd) states",
                       "power": "curve(WS_eff) per turbine; thrust from the generic turbine",
                       "cut_out": "farm off when the free-stream hub speed > 20 m/s",
                       "cut_in": "pywake curve as PyWake gives it; cubic curves 3 m/s"},
        "positions_utm32n": {"ids": ids, "E": xy[:, 0].tolist(), "N": xy[:, 1].tolist(),
                             "source": str(args.wea),
                             "rule": "OSTWERT_UTM32N - 32e6, NORDWERT_UTM32N (as TopoMILP)"},
        "dwd": {k: rec[k] for k in ("path", "sha256", "n_rows", "n_flagged", "n_valid",
                                     "n_span_hours", "n_missing_hours", "n_gaps",
                                     "max_gap_hours", "first_utc", "last_utc", "n_calm_coded_D0")},
        "timestamps": "MESS_DATUM read as UTC; gaps kept as gaps; hour_unix in the file",
        "n_valid_hours": rec["n_valid"], "n_distinct_states": int(len(res["states"])),
        "width": {"dv": args.dv, "dtheta": args.dtheta, "K": int(tab["K"])},
        "width_grid": [{"dv": g["dv"], "dtheta": g["dtheta"], "K": g["K"],
                        "gap_all": g["gap_all"], "worst_gap": g["worst_gap"],
                        "worst_mask": g["worst_mask"]} for g in grid],
        "units": {"h": "h/a", "P": "MW", "Q": "Mvar", "cov": "MVA^2", "W": "MVA^2 h/a",
                  "M": "MVA^2 h/a"},
        "subset_order": "mask m = 1..127; bit t (value 2^t) = turbine t = wea feature t",
        "theta_def": "W(all) / (7 x 6.8 MW)^2, hours",
        "theta_h_exact": float(w_exact[-1] / s_cap), "theta_h_cat": float(w_cat[-1] / s_cap),
        "W_all_exact": float(w_exact[-1]), "W_all_cat": float(w_cat[-1]),
        "numpy": np.__version__,
    }
    if args.report:
        report(args, rec, xy, farm, res, member, w_exact, grid, tab, level)
    if args.out is not None:
        speed_edges = (np.arange(int(tab["vbin"].max()) + 2) * args.dv if args.dv > 0
                       else np.zeros(0))
        payload = {
            "h": tab["h"], "count": tab["count"], "P": tab["P"], "Q": tab["Q"],
            "cov": tab["cov"], "v_mean": tab["v_mean"],
            "speed_bin": tab["vbin"], "dir_bin": tab["dbin"], "speed_edges": speed_edges,
            "speed_values": (tab["speed_values"] if args.dv == 0 else np.zeros(0)),
            "dir_centres": np.arange(tab["n_dir"]) * args.dtheta,
            "dir_edges": np.arange(tab["n_dir"] + 1) * args.dtheta - 0.5 * args.dtheta,
            "subset_mask": masks, "subset_member": member,
            "W_cat": w_cat, "W_var": w_var, "W_exact": w_exact,
            "W_min_cat": w_min_by_size(w_cat, member),
            "W_min_exact": w_min_by_size(w_exact, member),
            "M_cat": tab["M_cat"], "M_within": tab["M_within"], "M_exact": res["M_exact"],
            "grid_dv": np.array([g["dv"] for g in grid]),
            "grid_dtheta": np.array([g["dtheta"] for g in grid]),
            "grid_K": np.array([g["K"] for g in grid]),
            "grid_W": np.stack([g["W_cat"] for g in grid]),
            "grid_W_min": np.stack([g["W_min"] for g in grid]),
            "state_v": res["states"][:, 0], "state_wd": res["states"][:, 1],
            "state_P": res["p_state"], "state_Q": res["p_state"] * res["tanphi"],
            "hour_unix": rec["unix"], "hour_state": res["state_of_hour"].astype(np.int32),
            "hour_cat": tab["cat"].astype(np.int32), "positions_EN": xy,
            "meta": np.array(json.dumps(meta, sort_keys=True)),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "wb") as fh:
            np.savez_compressed(fh, **payload)
        digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
        print(f"wrote {args.out} ({args.out.stat().st_size} B) sha256 {digest}")
    print(f"runtime {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
