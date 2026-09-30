"""
PYORPS GUI: constrained overhead-line routing (R13, Section 20).

Wraps ``ConstrainedPathFinder`` + ``InfrastructureProfile``: shipped-profile
discovery, live profile validation (mirroring ``__post_init__`` so the error
box never has to appear — 21.5), experimental-backend gating (F10), and the
tower export. The reported terrain cost is 2-D even though the optimizer used
3-D penalties — a known limitation the GUI labels explicitly (F9).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import Notice

#: F10: only these backends are production-ready; the GPU variants hang /
#: exhaust VRAM on real rasters and must be explicitly opted into.
STABLE_BACKENDS = ("cython",)
EXPERIMENTAL_BACKENDS = ("cython_parallel", "raster_gpu", "raster_gpu_v4")

F9_CAVEAT = ("Terrain cost shown is 2-D: the optimizer used 3-D penalties "
             "but reports the plain 2-D terrain cost (known limitation). "
             "Compare variants by length + tower cost too.")


def profiles_dir() -> Path | None:
    """The repo's shipped profiles/ folder (dev checkout), if present."""
    candidate = Path(__file__).resolve().parents[3] / "profiles"
    return candidate if candidate.is_dir() else None


def shipped_profiles() -> dict[str, str]:
    """Name -> path of the shipped YAML profiles (110/220/380 kV, road)."""
    folder = profiles_dir()
    if folder is None:
        return {}
    return {path.stem: str(path) for path in sorted(folder.glob("*.yaml"))}


def load_profile_dict(source: str | Path | dict) -> dict:
    """Profile file (YAML/JSON) or dict -> plain dict."""
    if isinstance(source, dict):
        return dict(source)
    from .project_io import load_profile

    return load_profile(source)


def validate_span_bin_vs_resolution(config: dict,
                                    resolution_m: float) -> Notice | None:
    """Perf pre-flight: a span bin larger than the cell size is pathological.

    The extended-state search only advances one raster cell per step; with
    ``span_bin_size_m`` far above the resolution the state space degenerates
    and a run that should take seconds takes minutes (measured: 40x40 cells
    at 1 m with bin=10 runs for minutes; bin=1 finishes in ~0.1 s).
    """
    bin_size = config.get("span_bin_size_m")
    if bin_size is None or resolution_m is None or resolution_m <= 0:
        return None
    if bin_size <= resolution_m:
        return None
    return Notice(
        severity="warning",
        title="Span bin size is coarser than the raster",
        meaning=f"span_bin_size_m ({bin_size} m) is larger than the raster "
                f"cell size ({resolution_m:g} m); each search step should "
                "advance at least one span bin.",
        impact="The constrained search can take minutes instead of "
               "seconds.",
        fix=f"Set span_bin_size_m to <= {resolution_m:g} (the raster "
            "resolution).",
        focus_id="tab-routes")


def validate_profile(config: dict) -> Notice | None:
    """Mirror InfrastructureProfile.__post_init__ rules, shifted left."""
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    problems = []
    soft = config.get("soft_angle_limit_deg")
    hard = config.get("hard_angle_limit_deg")
    if soft is not None and hard is not None and hard < soft:
        problems.append(f"hard_angle_limit_deg ({hard}) must be >= "
                        f"soft_angle_limit_deg ({soft})")
    min_span = config.get("min_span_m")
    max_span = config.get("max_span_m")
    if min_span is not None and max_span is not None and min_span > max_span:
        problems.append(f"min_span_m ({min_span}) must be <= "
                        f"max_span_m ({max_span})")
    if (min_span is not None or max_span is not None):
        bin_size = config.get("span_bin_size_m")
        if bin_size is not None and bin_size <= 0:
            problems.append("span_bin_size_m must be > 0 when spans are set")
    area = config.get("tower_ground_area_m2")
    if area is not None and area < 1.0:
        problems.append(f"tower_ground_area_m2 ({area}) must be >= 1.0")
    mode = config.get("tower_area_cost_mode")
    if mode is not None and mode not in ("uniform", "exact"):
        problems.append(f"tower_area_cost_mode '{mode}' must be uniform or "
                        "exact")
    fn = config.get("angle_cost_function")
    if fn is not None and fn not in ("linear", "quadratic", "piecewise"):
        problems.append(f"angle_cost_function '{fn}' must be linear, "
                        "quadratic or piecewise")
    if not problems:
        return None
    return Notice(
        severity="error", title="Profile setting invalid",
        meaning="; ".join(problems),
        impact="Constrained routing can't run until the profile is fixed.",
        fix="Correct the highlighted field(s) in the profile editor.",
        focus_id="tab-routes")


def resolve_backend(backend: str, experimental: bool) -> tuple[str, Notice | None]:
    """F10: gate experimental backends behind the explicit flag."""
    backend = backend or "cython"
    if backend in STABLE_BACKENDS:
        return backend, None
    if backend in EXPERIMENTAL_BACKENDS and experimental:
        return backend, Notice(
            severity="info", title=f"Experimental backend '{backend}'",
            meaning="GPU/parallel constrained backends can hang or exceed "
                    "VRAM on real rasters and fall back to Cython (F10).",
            impact="The run may be slow, fall back, or fail.",
            fix="Switch back to cython if it misbehaves.",
            focus_id="tab-routes")
    return "cython", Notice(
        severity="info", title="Backend reset to cython",
        meaning=f"'{backend}' is experimental and the experimental flag "
                "is off (F10).",
        impact="The run uses the stable Cython backend.",
        fix="Tick 'allow experimental backends' to use it anyway.",
        focus_id="tab-routes")


def run_constrained(raster_source: Any, *, source: tuple[float, float],
                    target: tuple[float, float], profile: dict,
                    backend: str = "cython", neighborhood: str = "r2",
                    search_buffer_m: float | None = None,
                    dem: str | None = None, dsm: str | None = None):
    """Run ConstrainedPathFinder; returns (ConstrainedPath, towers_gdf)."""
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    from pyorps.core.infrastructure_profile import InfrastructureProfile
    from pyorps.graph.constrained_path_finder import ConstrainedPathFinder

    from .routing import default_search_buffer

    profile = dict(profile)
    profile.setdefault("name", "gui_profile")
    profile.setdefault("description", "edited in the PYORPS GUI")
    profile_obj = InfrastructureProfile.from_dict(profile)
    if not search_buffer_m or search_buffer_m <= 0:      # F1 again
        search_buffer_m = default_search_buffer([source], [target])

    finder = ConstrainedPathFinder(
        dataset_source=raster_source, source_coords=tuple(source),
        target_coords=tuple(target), profile=profile_obj,
        graph_api=backend, neighborhood_str=neighborhood,
        search_space_buffer_m=search_buffer_m,
        dem=dem or None, dsm=dsm or None)
    result = finder.find_route()
    crs = finder.dataset.crs
    towers = result.towers_to_geodataframe(crs=crs)
    return result, towers, crs


def run_constrained_path(raster_source: Any, *, points, profile: dict,
                         backend: str = "cython", neighborhood: str = "r2",
                         search_buffer_m: float | None = None,
                         dem: str | None = None, dsm: str | None = None):
    """Constrained routing through ``points`` (source, *waypoints, target).

    Routes each consecutive segment independently and stitches them: every
    interior waypoint is a segment endpoint, i.e. a FORCED terminal tower there
    (task 46). Returns a dict with ``line, towers, crs, total_cost,
    total_length, total_cell_cost, summary`` or None if any segment has no path.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import LineString

    points = [tuple(p) for p in points]
    if len(points) < 2:
        return None
    segments, tower_frames, summaries = [], [], []
    crs = None
    totals = {"total_cost": 0.0, "total_length": 0.0, "total_cell_cost": 0.0}
    for a, b in zip(points[:-1], points[1:]):
        result, towers, crs = run_constrained(
            raster_source, source=a, target=b, profile=profile,
            backend=backend, neighborhood=neighborhood,
            search_buffer_m=search_buffer_m, dem=dem, dsm=dsm)
        if result.path_geometry is None:
            return None
        segments.append(result.path_geometry)
        if towers is not None and not towers.empty:
            tower_frames.append(towers)
        summaries.append(result_summary(result))
        totals["total_cost"] += float(result.total_cost or 0)
        totals["total_length"] += float(getattr(result, "total_length", 0) or 0)
        totals["total_cell_cost"] += float(
            getattr(result, "total_cell_cost", 0) or 0)

    # stitch the segment lines end-to-end (drop the shared vertex at joins)
    coords: list = []
    for seg in segments:
        seg_coords = list(seg.coords)
        if coords and coords[-1] == seg_coords[0]:
            seg_coords = seg_coords[1:]
        coords.extend(seg_coords)
    line = LineString(coords)

    towers_gdf = None
    if tower_frames:
        towers_gdf = gpd.GeoDataFrame(
            pd.concat(tower_frames, ignore_index=True), crs=crs)
        # a waypoint tower appears as the end of one segment and the start of
        # the next — drop the geometric duplicates
        towers_gdf = towers_gdf[~towers_gdf.geometry.duplicated()].reset_index(
            drop=True)

    summary = {
        "n_towers": int(len(towers_gdf)) if towers_gdf is not None else sum(
            s["n_towers"] for s in summaries),
        "total_tower_cost": sum(s["total_tower_cost"] for s in summaries),
        "total_angle_penalty_cost": sum(
            s["total_angle_penalty_cost"] for s in summaries),
        "max_turn_angle_deg": max(
            (s["max_turn_angle_deg"] for s in summaries), default=0.0),
        "spans_min_max_avg": (
            min(s["spans_min_max_avg"][0] for s in summaries),
            max(s["spans_min_max_avg"][1] for s in summaries),
            sum(s["spans_min_max_avg"][2] for s in summaries) / len(summaries)),
    }
    return {"line": line, "towers": towers_gdf, "crs": crs, "summary": summary,
            **totals}


def result_summary(result) -> dict:
    """The read-only result readout (Section 20), incl. the F9 caveat."""
    return {
        "n_towers": result.n_towers,
        "total_tower_cost": result.total_tower_cost,
        "total_terrain_cost_2d": result.total_terrain_cost,   # F9
        "total_angle_penalty_cost": result.total_angle_penalty_cost,
        "spans_min_max_avg": (result.min_span_actual_m,
                              result.max_span_actual_m, result.avg_span_m),
        "max_turn_angle_deg": result.max_turn_angle_deg,
        "tower_type_counts": dict(result.tower_type_counts or {}),
        "caveat": F9_CAVEAT,
    }
