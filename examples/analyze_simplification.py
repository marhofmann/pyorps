"""Detailed comparison of routed paths with vs. without simplification.

Joins ``all_mv_routes.geojson`` and
``all_mv_routes_simplified_douglas_peucker_2.0.geojson`` on (source, target),
then reports:

1. Per-path deltas in path length, total cost (distance-weighted), and raw
   cell cost; with summary statistics over all paths.
2. Vertex-count reduction and geometry-length difference.
3. Per-terrain-category length shifts in absolute metres and percentage
   points (PP) -i.e. how much each category's share of every route changed.
4. The top-10 paths with the largest absolute cost shift.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


OUT = Path("output/example_2025_eamex/route_planning")
RAW = OUT / "all_mv_routes.geojson"
SIMP = OUT / "all_mv_routes_simplified_douglas_peucker_2.0.geojson"

# Human-readable labels for the cost integers used in the example raster.
LABEL = {
    125: "Heide / Unland",
    200: "Segelfluggelände",
    300: "Weg / Fußweg",
    310: "Platz / Parkplatz",
    320: "Grünanlage / Rastplatz",
    332: "Graben / Fließgewässer",
    340: "Straßenverkehr",
    346: "Bach",
    350: "Teich",
    380: "Brachland / Gehölz",
    400: "Stehendes Gewässer",
    405: "Nadelholz / Wald",
    425: "Laub-+Nadelwald",
    437: "Ackerland / Grünland",
    475: "Laubwald",
    500: "Bundesstraße",
    581: "Gartenbau / Streuobst",
    586: "Fluss",
    590: "Stausee / Baggersee",
    750: "Autobahn",
    800: "Bahnverkehr",
    65535: "No-Go / Schutzgebiet",
}


def _norm_key(s):
    """Robust merge key from heterogenous source/target string formats.

    The source/target columns are either ``str(np.array)`` like
    ``"[451405.56 5614266.57]"`` or ``str(tuple)`` like
    ``"(451405.56, 5614266.57)"``. Strip parens/brackets/commas, then take
    the first two floats and round to 0.01 m.
    """
    if isinstance(s, str):
        cleaned = (
            s.replace("[", " ").replace("]", " ")
             .replace("(", " ").replace(")", " ")
             .replace(",", " ").split()
        )
    elif hasattr(s, "tolist"):
        return tuple(round(float(v), 2) for v in s.tolist()[:2])
    else:
        cleaned = list(s)
    return tuple(round(float(v), 2) for v in cleaned[:2])


def _length_cols(df):
    return [c for c in df.columns if c.startswith("length_cost_")]


def _to_cat(c):
    return int(c.removeprefix("length_cost_"))


def load_and_join():
    raw = gpd.read_file(RAW)
    simp = gpd.read_file(SIMP)

    raw = raw.copy()
    simp = simp.copy()
    raw["key"] = list(zip(raw["source"].map(_norm_key),
                          raw["target"].map(_norm_key)))
    simp["key"] = list(zip(simp["source"].map(_norm_key),
                           simp["target"].map(_norm_key)))

    keep_metric_cols = (
        ["path_length", "path_cost", "path_cell_cost", "geometry"]
        + _length_cols(raw)
    )
    raw_keep = raw[["key"] + keep_metric_cols].set_index("key")
    keep_metric_cols_s = (
        ["path_length", "path_cost", "path_cell_cost", "geometry"]
        + _length_cols(simp)
    )
    simp_keep = simp[["key"] + keep_metric_cols_s].set_index("key")

    merged = raw_keep.join(simp_keep, lsuffix="_raw", rsuffix="_simp",
                           how="inner")
    return raw, simp, merged


def vertex_count(geom):
    return len(geom.coords)


def aggregate_summary(merged):
    rows = len(merged)
    print(f"Joined paths: {rows}")
    print()
    print("=" * 78)
    print("1. Per-path totals -descriptive statistics over all routes")
    print("=" * 78)
    stats = pd.DataFrame({
        "length_raw_m": merged["path_length_raw"],
        "length_simp_m": merged["path_length_simp"],
        "length_delta_m": merged["path_length_simp"] - merged["path_length_raw"],
        "cost_raw":   merged["path_cost_raw"],
        "cost_simp":  merged["path_cost_simp"],
        "cost_delta": merged["path_cost_simp"] - merged["path_cost_raw"],
        "cell_cost_raw":   merged["path_cell_cost_raw"],
        "cell_cost_simp":  merged["path_cell_cost_simp"],
    })
    stats["length_delta_pct"] = (
        100 * stats["length_delta_m"] / stats["length_raw_m"]
    )
    stats["cost_delta_pct"] = (
        100 * stats["cost_delta"] / stats["cost_raw"].replace(0, np.nan)
    )

    desc = stats.describe(percentiles=[0.05, 0.5, 0.95]).T
    desc = desc[["mean", "std", "min", "5%", "50%", "95%", "max"]]
    print(desc.round(2).to_string())


def vertex_reduction(merged):
    print()
    print("=" * 78)
    print("2. Vertex count and geometry length")
    print("=" * 78)
    nv_raw = merged["geometry_raw"].apply(vertex_count)
    nv_simp = merged["geometry_simp"].apply(vertex_count)
    geom_len_raw = merged["geometry_raw"].length
    geom_len_simp = merged["geometry_simp"].length
    print(f"  Vertices RAW:  total={int(nv_raw.sum()):>10,}  "
          f"mean/path={nv_raw.mean():7.1f}  median={nv_raw.median():5.0f}")
    print(f"  Vertices SIMP: total={int(nv_simp.sum()):>10,}  "
          f"mean/path={nv_simp.mean():7.1f}  median={nv_simp.median():5.0f}")
    reduction = 1.0 - nv_simp.sum() / nv_raw.sum()
    print(f"  Total vertex reduction: {100 * reduction:.1f}%")
    print(f"  Geometry length sum  RAW: {geom_len_raw.sum():>12,.1f} m")
    print(f"  Geometry length sum SIMP: {geom_len_simp.sum():>12,.1f} m  "
          f"(delta = {geom_len_simp.sum() - geom_len_raw.sum():+,.1f} m, "
          f"{100*(geom_len_simp.sum()/geom_len_raw.sum() - 1):+.2f}%)")


def category_shifts(raw, simp, merged):
    print()
    print("=" * 78)
    print("3. Per-terrain-category length shifts (aggregated over all 386 routes)")
    print("=" * 78)
    cats_raw = {_to_cat(c) for c in _length_cols(raw)}
    cats_simp = {_to_cat(c) for c in _length_cols(simp)}
    cats = sorted(cats_raw | cats_simp)

    tot_len_raw = merged["path_length_raw"].sum()
    tot_len_simp = merged["path_length_simp"].sum()

    rows = []
    for cat in cats:
        col = f"length_cost_{cat}"
        len_raw = (
            merged[f"{col}_raw"].fillna(0).sum() if f"{col}_raw" in merged
            else 0.0
        )
        len_simp = (
            merged[f"{col}_simp"].fillna(0).sum() if f"{col}_simp" in merged
            else 0.0
        )
        pct_raw = 100 * len_raw / tot_len_raw if tot_len_raw > 0 else 0.0
        pct_simp = 100 * len_simp / tot_len_simp if tot_len_simp > 0 else 0.0
        rows.append({
            "category":  cat,
            "label":     LABEL.get(cat, f"cost {cat}"),
            "len_raw_m":  round(len_raw, 1),
            "len_simp_m": round(len_simp, 1),
            "delta_m":    round(len_simp - len_raw, 1),
            "pct_raw":   round(pct_raw, 3),
            "pct_simp":  round(pct_simp, 3),
            "delta_pp":  round(pct_simp - pct_raw, 3),
        })
    df = pd.DataFrame(rows).sort_values("len_raw_m", ascending=False)
    print(df.to_string(index=False))

    print()
    print("Categories sorted by abs |delta percentage points| (top 8):")
    df["abs_dpp"] = df["delta_pp"].abs()
    print(
        df.sort_values("abs_dpp", ascending=False)
        [["category", "label", "len_raw_m", "len_simp_m", "delta_m",
          "pct_raw", "pct_simp", "delta_pp"]]
        .head(8).to_string(index=False)
    )


def biggest_cost_movers(merged):
    print()
    print("=" * 78)
    print("4. Top 10 paths with the largest absolute cost shift")
    print("=" * 78)
    delta = (
        (merged["path_cost_simp"] - merged["path_cost_raw"])
        .rename("cost_delta")
        .to_frame()
    )
    delta["abs_cost_delta"] = delta["cost_delta"].abs()
    delta["cost_raw"] = merged["path_cost_raw"]
    delta["cost_simp"] = merged["path_cost_simp"]
    delta["length_raw_m"] = merged["path_length_raw"]
    delta["length_simp_m"] = merged["path_length_simp"]
    delta["length_delta_m"] = delta["length_simp_m"] - delta["length_raw_m"]
    delta["cost_delta_pct"] = (
        100 * delta["cost_delta"]
        / delta["cost_raw"].replace(0, np.nan)
    )
    top = delta.sort_values("abs_cost_delta", ascending=False).head(10)
    print(top.round(1).to_string())

    print()
    print("=" * 78)
    print("5. Sanity -does cost go up or down when the line is simplified?")
    print("=" * 78)
    up = int((delta["cost_delta"] > 0).sum())
    down = int((delta["cost_delta"] < 0).sum())
    same = int((delta["cost_delta"] == 0).sum())
    print(f"  Paths with cost INCREASE after simplification: {up:3d}")
    print(f"  Paths with cost DECREASE after simplification: {down:3d}")
    print(f"  Paths with cost UNCHANGED:                     {same:3d}")
    print(f"  Mean cost delta: {delta['cost_delta'].mean():+.1f}  "
          f"(median {delta['cost_delta'].median():+.1f})")


def nogo_analysis(merged):
    """Quantify how often DP shortcuts plough through 65535 (no-go) cells
    that the router carefully avoided."""
    print()
    print("=" * 78)
    print("6. No-go (65535) traversal -- the likely culprit for cost blow-up")
    print("=" * 78)
    col = "length_cost_65535"
    raw_col = f"{col}_raw"
    simp_col = f"{col}_simp"
    if raw_col not in merged or simp_col not in merged:
        print("  No no-go column present in either dataset.")
        return
    raw_len = merged[raw_col].fillna(0)
    simp_len = merged[simp_col].fillna(0)
    raw_total = float(raw_len.sum())
    simp_total = float(simp_len.sum())
    print(f"  Total length through no-go cells, RAW:  {raw_total:>10,.1f} m  "
          f"({100 * raw_total / merged['path_length_raw'].sum():.3f}% of "
          f"network)")
    print(f"  Total length through no-go cells, SIMP: {simp_total:>10,.1f} m  "
          f"({100 * simp_total / merged['path_length_simp'].sum():.3f}% of "
          f"network)")
    print(f"  Increase: factor {simp_total / max(raw_total, 1e-9):,.1f}x")

    diff = (simp_len - raw_len).reset_index(drop=True)
    raw_pos = raw_len.reset_index(drop=True)
    simp_pos = simp_len.reset_index(drop=True)
    length_raw_pos = merged["path_length_raw"].reset_index(drop=True)
    affected = int((diff > 0).sum())
    print(f"  Paths whose no-go length INCREASED: {affected} / {len(merged)} "
          f"({100*affected/len(merged):.1f}%)")
    top = diff.sort_values(ascending=False).head(5)
    print()
    print("  Top 5 paths by additional no-go metres after simplification:")
    for rank, pos in enumerate(top.index, 1):
        dlen = float(diff.iloc[pos])
        rl = float(raw_pos.iloc[pos])
        sl = float(simp_pos.iloc[pos])
        path_len = float(length_raw_pos.iloc[pos])
        share = 100 * sl / path_len if path_len > 0 else 0.0
        print(f"   #{rank}  delta_no_go=+{dlen:7.1f} m  "
              f"(raw={rl:6.1f}, simp={sl:7.1f}, "
              f"path_length={path_len:7.1f}, "
              f"no_go_share_simp={share:5.1f}%)")


def main():
    raw, simp, merged = load_and_join()
    aggregate_summary(merged)
    vertex_reduction(merged)
    category_shifts(raw, simp, merged)
    biggest_cost_movers(merged)
    nogo_analysis(merged)


if __name__ == "__main__":
    main()
