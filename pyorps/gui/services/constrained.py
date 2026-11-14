"""
PYORPS GUI: constrained overhead-line routing (R13, Section 20).

Wraps ``ConstrainedPathFinder`` + ``InfrastructureProfile``: shipped-profile
discovery, live profile validation (mirroring ``__post_init__`` so the error
box never has to appear — 21.5), experimental-backend gating (F10), and the
tower export. The reported terrain cost is 2-D even though the optimizer used
3-D penalties — a known limitation the GUI labels explicitly (F9).

Round 14 (gap 7): ``is_heavy`` decides which runs go to a background job,
``snapshot_run_inputs`` / ``background_run`` / ``status_text`` are the job's
worker side, and ``run_constrained_path`` takes ``cancel`` / ``progress``.
The library has no cancel point inside one search, so Stop acts between the
segments of a run (``STOP_NOTE`` says so).

Round 16 (R3-B): ``run_constrained`` / ``run_constrained_path`` / ``snapshot_run_inputs`` take ``objective``,
the weights dict of ``constrained_objective.library_objective``; None (every weight at its default) passes no
``objective`` keyword to the library at all, so a classic run is called exactly as before.

Round 17 (P4): the two connection heights of the line ends (``constrained_ends``). ``run_constrained_path`` and
``snapshot_run_inputs`` take ``ends`` (the settings dict of ``constrained_ends``), ``run_constrained`` takes the
``find_route`` keywords of its one segment (``constrained_ends.library_kwargs``). Without a height nothing new is
passed, and the snapshot of a classic run has no ``ends`` key. ``result_summary`` reports what the library now
says about the line ends (removed towers, span notes, search notes, heights, bin size, premiums), and only where
it says something: every old key stays, and a route with nothing to report gets no new key.

Round 17 (fixes): a search without a feasible route returns the library's empty result, which ``run_constrained``
recognises before it builds any tower frame (``run_constrained_path`` then answers None, as a "no route" run always
did; ``outcome`` names the segment of a chain). In a chain of waypoints one physical tower stands at every interior
waypoint and its cost is counted once (``run_constrained_path``, ``_merge_waypoint_towers``).
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

from pyorps.core.exceptions import InvalidInputError

from . import constrained_ends, jobs, routing
from .errors import Notice
from .jobs import RUNNING, JobCancelled

#: F10: only these backends are production-ready; the GPU variants hang /
#: exhaust VRAM on real rasters and must be explicitly opted into.
STABLE_BACKENDS = ("cython",)
EXPERIMENTAL_BACKENDS = ("cython_parallel", "raster_gpu", "raster_gpu_v4")

F9_CAVEAT = ("Terrain cost shown is 2-D: the optimizer used 3-D penalties "
             "but reports the plain 2-D terrain cost (known limitation). "
             "Compare variants by length + tower cost too.")

#: the key of the background job in ``ProjectState.jobs`` (the one the page's
#: ``layout.JOB_CONTROLS`` knows) and the name the user reads in its status
JOB_KIND = "constrained"
JOB_LABEL = "Constrained routing"
#: what a Stop is promised to do, for the status line and the Stop answer. The
#: library offers no cancel point inside one search, so Stop acts between segments.
STOP_NOTE = ("A segment (one search from a point to the next one) cannot be "
             "interrupted: the run stops when the segment in flight has "
             "finished (a run without waypoints is one single segment), and "
             "then adds no route.")


def is_heavy(est: dict | None) -> bool:
    """Does a constrained run with this prognosis go to a background job?

    ``est`` is ``routing.estimate_routing(...)``; the criterion is the one of
    plain routing (``routing.needs_confirmation``, read when called). No
    prognosis (the estimate itself failed, audit #35) counts as heavy: an unknown
    effort must not block the page.
    """
    return est is None or routing.needs_confirmation(est)


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
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
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
                    dem: str | None = None, dsm: str | None = None,
                    objective: dict[str, float] | None = None,
                    ends: dict[str, float | None] | None = None):
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Run ConstrainedPathFinder; returns (ConstrainedPath, towers_gdf, crs).

    A search that finds no feasible route returns the library's empty result (``path_geometry`` is None); its
    ``towers_gdf`` is None (round 17: no frame is built from an empty tower list).

    ``objective`` is the weights dict of ``constrained_objective.library_objective`` (None: classic
    constrained routing; then no ``objective`` keyword reaches the library at all).

    ``ends`` holds the ``find_route`` keywords of this one search, as ``constrained_ends.library_kwargs`` gives
    them (``source_connection_height_m`` and ``target_connection_height_m``; an end that is not set for this
    call is an explicit ``None``). Empty or None: ``find_route()`` is called without a keyword, as it always was.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    from pyorps.core.infrastructure_profile import InfrastructureProfile
    from pyorps.graph.constrained_path_finder import ConstrainedPathFinder

    from .routing import default_search_buffer

    unknown = sorted(set(ends or ()) - set(constrained_ends.HEIGHT_KEYS))
    if unknown:
        raise InvalidInputError(f"run_constrained: unknown line-end keyword(s) {unknown}; allowed: "
                                f"{list(constrained_ends.HEIGHT_KEYS)}")
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
        dem=dem or None, dsm=dsm or None,
        **({"objective": dict(objective)} if objective else {}))
    result = finder.find_route(**dict(ends)) if ends else finder.find_route()
    crs = finder.dataset.crs
    if result.path_geometry is None:
        # the library's empty result (no feasible route): there are no towers to export, and the library's own
        # frame of an empty tower list is refused by geopandas (no geometry column), so none is built here
        return result, None, crs
    towers = result.towers_to_geodataframe(crs=crs)
    return result, towers, crs


def _cancel_point(cancel) -> None:
    """A segment boundary: ends the run (``JobCancelled``) once Stop was pressed."""
    if cancel is not None and cancel.is_set():
        raise JobCancelled(f"{JOB_LABEL} was stopped")


def run_constrained_path(raster_source: Any, *, points, profile: dict,
                         backend: str = "cython", neighborhood: str = "r2",
                         search_buffer_m: float | None = None,
                         dem: str | None = None, dsm: str | None = None,
                         cancel=None, progress=None,
                         objective: dict[str, float] | None = None,
                         ends: dict[str, Any] | None = None,
                         outcome: dict | None = None):
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Constrained routing through ``points`` (source, *waypoints, target).

    Routes each consecutive segment independently and stitches them: every
    interior waypoint is a segment endpoint, i.e. a FORCED terminal tower there
    (task 46). Returns a dict with ``line, towers, crs, total_cost,
    total_length, total_cell_cost, cost_breakdown, summary`` or None if any
    segment has no path. ``outcome`` (an empty dict the caller keeps) learns
    which one: ``outcome["no_route"] = {"segment": k, "of": n}`` (1-based) when
    segment k of n found no route; :func:`no_route_segment_text` words it.

    One physical tower stands at an interior waypoint, although both segments
    report a terminal tower there (round 17): its cost is counted once. The
    two reports can disagree about its height or premium (a source tower can
    carry a height premium, with several tower heights); the DEARER one is
    kept (at equal cost the TALLER one), the other is dropped from the tower
    table and from every total, and ``summary["search_notes"]`` says so
    (:func:`_merge_waypoint_towers`).
    So ``total_cost`` = terrain + tower table + turn penalties
    (``cost_breakdown`` holds the three terms).
    ``objective`` (``constrained_objective.library_objective``) goes to every
    segment's finder; None keeps classic constrained routing.

    ``ends`` is the settings dict of ``constrained_ends`` (the two connection
    heights). Each segment gets ``constrained_ends.library_kwargs(ends, index,
    n_segments)``: the source height only on the first segment, the target
    height only on the last, an explicit ``None`` for the ends at waypoints.
    None, ``{}`` or no height set: nothing new reaches the library.

    ``cancel`` (anything with ``is_set()``, e.g. a ``threading.Event``) and
    ``progress`` are for a background run. One segment is one library call that
    cannot be interrupted, so Stop is looked at between the segments (and once
    more after the last one): a stopped run raises ``jobs.JobCancelled`` and
    returns nothing, never a partial tower chain. ``progress(done, total,
    stage)`` is called when each segment starts, with the segments finished so
    far. Without them (the synchronous run) the function behaves as before.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import LineString

    points = [tuple(p) for p in points]
    if len(points) < 2:
        return None
    segments, tower_frames, tower_lists, summaries = [], [], [], []
    crs = None
    totals = {"total_cost": 0.0, "total_length": 0.0, "total_cell_cost": 0.0}
    terms = {"terrain": 0.0, "angle_penalties": 0.0}
    n_segments = len(points) - 1
    for index, (a, b) in enumerate(zip(points[:-1], points[1:])):
        _cancel_point(cancel)
        if progress is not None:
            progress(index, n_segments, "segments done")
        segment_ends = constrained_ends.library_kwargs(ends, index, n_segments)
        result, towers, crs = run_constrained(
            raster_source, source=a, target=b, profile=profile,
            backend=backend, neighborhood=neighborhood,
            search_buffer_m=search_buffer_m, dem=dem, dsm=dsm,
            **({"objective": objective} if objective else {}),
            **({"ends": segment_ends} if segment_ends else {}))
        if result.path_geometry is None:
            if outcome is not None:
                outcome["no_route"] = {"segment": index + 1, "of": n_segments}
            return None
        segments.append(result.path_geometry)
        tower_frames.append(towers)
        tower_lists.append(list(getattr(result, "towers", None) or ()))
        summaries.append(result_summary(result))
        terms["terrain"] += float(getattr(result, "total_terrain_cost", 0) or 0)
        terms["angle_penalties"] += float(getattr(result, "total_angle_penalty_cost", 0) or 0)
        totals["total_cost"] += float(result.total_cost or 0)
        totals["total_length"] += float(getattr(result, "total_length", 0) or 0)
        totals["total_cell_cost"] += float(
            getattr(result, "total_cell_cost", 0) or 0)
    _cancel_point(cancel)             # Stop came during the last segment

    # stitch the segment lines end-to-end (drop the shared vertex at joins)
    coords: list = []
    for seg in segments:
        seg_coords = list(seg.coords)
        if coords and coords[-1] == seg_coords[0]:
            seg_coords = seg_coords[1:]
        coords.extend(seg_coords)
    line = LineString(coords)

    # a waypoint tower appears as the end of one segment and the start of the next: one structure stands there
    drop, dropped, notes = _merge_waypoint_towers(tower_frames, tower_lists, segments)
    towers_gdf = None
    kept = [frame.drop(index=frame.index[sorted(rows)]) if rows else frame
            for frame, rows in zip(tower_frames, drop) if frame is not None and not frame.empty]
    kept = [frame for frame in kept if not frame.empty]
    if kept:
        towers_gdf = gpd.GeoDataFrame(pd.concat(kept, ignore_index=True), crs=crs)
    dropped_cost = sum(float(t.total_cost or 0) for t in dropped)
    totals["total_cost"] -= dropped_cost
    tower_cost = sum(s["total_tower_cost"] for s in summaries) - dropped_cost

    summary = {
        "n_towers": int(len(towers_gdf)) if towers_gdf is not None else sum(
            s["n_towers"] for s in summaries),
        "total_tower_cost": tower_cost,
        "total_angle_penalty_cost": sum(
            s["total_angle_penalty_cost"] for s in summaries),
        "max_turn_angle_deg": max(
            (s["max_turn_angle_deg"] for s in summaries), default=0.0),
        "spans_min_max_avg": (
            min(s["spans_min_max_avg"][0] for s in summaries),
            max(s["spans_min_max_avg"][1] for s in summaries),
            sum(s["spans_min_max_avg"][2] for s in summaries) / len(summaries)),
    }
    summary.update(merge_end_summaries(summaries))
    if "height_premiums" in summary and dropped:
        summary["height_premiums"] -= sum(float(t.height_premium or 0) for t in dropped)
    if notes:
        summary["search_notes"] = [*summary.get("search_notes", []), *notes]
    breakdown = {"terrain": terms["terrain"], "towers": tower_cost, "angle_penalties": terms["angle_penalties"]}
    return {"line": line, "towers": towers_gdf, "crs": crs, "summary": summary,
            "cost_breakdown": breakdown, **totals}


#: two tower locations closer than this (in the units of the raster's CRS) are the same place
_SAME_PLACE = 1e-6


def no_route_segment_text(info: dict | None) -> str:
    """Which segment of a chain found no route, as a plain sentence; "" for a run of one segment or no answer.

    ``info`` is what ``run_constrained_path`` leaves in ``outcome["no_route"]`` (``{"segment": k, "of": n}``,
    1-based). The segment is named by its two ends: the source, waypoint 1, 2, ... and the target.
    """
    if not info or int(info.get("of", 1)) < 2:
        return ""
    k, n = int(info["segment"]), int(info["of"])

    def name(point: int) -> str:               # point 0 is the source, point n the target
        return "the source" if point == 0 else "the target" if point == n else f"waypoint {point}"

    return f"Segment {k} of {n} (from {name(k - 1)} to {name(k)}) found no feasible tower-by-tower path."


def _tower_text(tower) -> str:
    height = getattr(tower, "height_m", None)
    return f"{'height not stated' if height is None else f'{height:g} m'}, cost {float(tower.total_cost or 0):,.0f}"


def _waypoint_pair(frames, tower_lists, lines, j):
    """The two towers at interior waypoint j (0-based: between segment j and j + 1), or None unless both exist."""
    first, second = tower_lists[j], tower_lists[j + 1]
    if not first or not second:
        return None
    place = lines[j].coords[-1]
    end, start = first[-1], second[0]
    same = all(abs(t.location.x - place[0]) <= _SAME_PLACE and abs(t.location.y - place[1]) <= _SAME_PLACE
               for t in (end, start))
    aligned = all(frame is not None and len(frame) == len(towers)
                  for frame, towers in ((frames[j], first), (frames[j + 1], second)))
    return (end, start) if same and aligned else None


def _merge_waypoint_towers(frames, tower_lists, lines):
    """One physical tower per interior waypoint (round 17).

    Each segment's search reports a terminal tower at both of its ends, so the structure at an interior waypoint is
    reported twice: as the target tower of one segment and as the source tower of the next. Rule: it is ONE tower,
    its cost is counted once. When the two reports disagree (the source tower of a segment can carry a height
    premium, and the two ends can report different heights, several tower heights) the dearer one is kept (more
    cost, i.e. the one that carries a premium), and when they cost the same the taller one; if nothing differs the
    earlier one stays. The other one leaves the tower table and every total.

    ``frames[i]`` is the tower table of segment i (None without towers) and ``tower_lists[i]`` its ``Tower`` list
    (same order); a waypoint where either side has no tower at the waypoint's place is left alone.

    Returns ``(drop, dropped, notes)``: ``drop[i]`` the set of row positions to drop from ``frames[i]``, ``dropped``
    the dropped towers, ``notes`` one sentence for every waypoint where the two reports disagreed.
    """
    drop = [set() for _ in frames]
    dropped, notes = [], []
    for j in range(len(frames) - 1):
        pair = _waypoint_pair(frames, tower_lists, lines, j)
        if pair is None:
            continue
        end, start = pair
        cost_end, cost_start = float(end.total_cost or 0), float(start.total_cost or 0)
        height_end = -math.inf if end.height_m is None else float(end.height_m)
        height_start = -math.inf if start.height_m is None else float(start.height_m)
        costs_differ = not math.isclose(cost_end, cost_start, rel_tol=1e-9, abs_tol=1e-9)
        heights_differ = height_end != height_start
        keep_start = (cost_start > cost_end) if costs_differ else (height_start > height_end)
        loser, segment = (end, j) if keep_start else (start, j + 1)
        drop[segment].add(len(tower_lists[segment]) - 1 if keep_start else 0)
        dropped.append(loser)
        if costs_differ or heights_differ:
            what = "heights and costs" if costs_differ and heights_differ else "costs" if costs_differ else "heights"
            notes.append(
                f"Waypoint {j + 1}: its two segments report different {what} for the structure there (segment "
                f"{j + 1}: {_tower_text(end)}; segment {j + 2}: {_tower_text(start)}). One structure stands there: "
                f"the dearer one (at equal cost the taller one; here segment {j + 2 if keep_start else j + 1}'s) is "
                f"kept and its cost is counted once.")
    return drop, dropped, notes


#: the summary keys the line ends add (each only where the library reports something)
END_SUMMARY_KEYS = ("terminal_span_rule", "span_notes", "removed_towers", "search_notes", "source_connection_height_m",
                    "target_connection_height_m", "span_bin_size_m", "height_premiums")


def _end_summary(result) -> dict:
    """What the library reports about the line ends of one route, only where it says something.

    ``terminal_span_rule`` (when not the default ``"remove"``), ``span_notes`` and ``search_notes`` (lists of
    sentences, when not empty), ``removed_towers`` (``{"count", "cost"}``, when the rule removed a tower), the two
    connection heights (when set; a height the search could not use is still the reported end height),
    ``span_bin_size_m`` (only together with a search note: the finder says in ``search_notes`` that it took
    exact bins, and only then is the bin worth a line, a default run's bin being its own coarse one), and
    ``height_premiums`` (the share of the tower cost that several tower heights charged, when the library
    reports it). A run without a rule, a height or a note adds none of these keys; a default ``"remove"``
    run still shows ``removed_towers`` and ``span_notes`` when the library removed a tower and says why.
    """
    out: dict[str, Any] = {}
    rule = getattr(result, "terminal_span_rule", None)
    if isinstance(rule, str) and rule != constrained_ends.DEFAULT_RULE:
        out["terminal_span_rule"] = rule
    span_notes = list(getattr(result, "span_notes", None) or [])
    if span_notes:
        out["span_notes"] = span_notes
    breakdown = getattr(result, "cost_breakdown", None) or {}
    removed = list(getattr(result, "removed_towers", None) or [])
    if removed:
        cost = breakdown.get("removed_towers")
        if cost is None:
            cost = sum(float(getattr(t, "total_cost", 0) or 0) for t in removed)
        out["removed_towers"] = {"count": len(removed), "cost": float(cost)}
    search_notes = list(getattr(result, "search_notes", None) or [])
    if search_notes:
        out["search_notes"] = search_notes
    heights = {key: getattr(result, key, None) for key in constrained_ends.HEIGHT_KEYS}
    out.update({key: float(value) for key, value in heights.items() if value is not None})
    bin_size = getattr(result, "span_bin_size_m", None)
    if bin_size is not None and search_notes:
        out["span_bin_size_m"] = float(bin_size)
    if breakdown.get("height_premiums") is not None:
        out["height_premiums"] = float(breakdown["height_premiums"])
    return out


def merge_end_summaries(summaries: list[dict]) -> dict:
    """The line-end keys of a whole run from those of its segments (see :func:`_end_summary`).

    Notes are collected in segment order (a span note of a run with several segments starts with "Segment k of
    n:"; a search note that repeats is said once), removed towers and height premiums are summed, the bin size is
    the smallest one, a connection height is the first one a segment reports (only the first segment holds the
    source height and only the last the target height), the rule is the one every segment shares.
    """
    n = len(summaries)
    out: dict[str, Any] = {}
    for key in ("terminal_span_rule", *constrained_ends.HEIGHT_KEYS):
        found = [s[key] for s in summaries if key in s]
        if found:
            out[key] = found[0]
    span_notes = [(f"Segment {i + 1} of {n}: {text}" if n > 1 else text)
                  for i, s in enumerate(summaries) for text in s.get("span_notes", [])]
    if span_notes:
        out["span_notes"] = span_notes
    search_notes = list(dict.fromkeys(text for s in summaries for text in s.get("search_notes", [])))
    if search_notes:
        out["search_notes"] = search_notes
    removed = [s["removed_towers"] for s in summaries if "removed_towers" in s]
    if removed:
        out["removed_towers"] = {"count": sum(r["count"] for r in removed), "cost": sum(r["cost"] for r in removed)}
    sizes = [s["span_bin_size_m"] for s in summaries if "span_bin_size_m" in s]
    if sizes:
        out["span_bin_size_m"] = min(sizes)
    premiums = [s["height_premiums"] for s in summaries if "height_premiums" in s]
    if premiums:
        out["height_premiums"] = sum(premiums)
    return out


def result_summary(result) -> dict:
    """The read-only result readout (Section 20), incl. the F9 caveat.

    The keys of :data:`END_SUMMARY_KEYS` are added where the library reports something about the line ends
    (:func:`_end_summary`); every other key is what it always was.
    """
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
        **_end_summary(result),
    }


# ------------------------------------------------------ background run (P10)
def snapshot_run_inputs(raster_source: Any, *, points, profile: dict,
                        backend: str, neighborhood: str,
                        search_buffer_m: float | None, dem: str | None,
                        dsm: str | None,
                        objective: dict[str, float] | None = None,
                        ends: dict[str, Any] | None = None) -> dict[str, Any]:
    """The ``run_constrained_path`` inputs of a background run, as the worker may keep them.

    The worker thread must never see the project change under it (the user can
    edit the profile text or move the points while a long run works), so it gets
    its own copy of the points, of the profile dict, of the objective weights and
    of the line-end heights. Everything else is a string or a number. Returns the
    keywords of :func:`run_constrained_path` plus ``raster_source``; the
    ``objective`` key is there only when weights were given and the ``ends`` key
    only when a connection height is set (the normalised settings dict), so a
    classic run's snapshot is what it always was.
    """
    snapshot = {
        "raster_source": raster_source,
        "points": [tuple(float(c) for c in p) for p in points],
        "profile": copy.deepcopy(profile), "backend": backend,
        "neighborhood": neighborhood, "search_buffer_m": search_buffer_m,
        "dem": dem, "dsm": dsm}
    if objective:
        snapshot["objective"] = copy.deepcopy(dict(objective))
    if ends and not constrained_ends.is_default(ends):
        snapshot["ends"] = copy.deepcopy(constrained_ends.normalise(ends))
    return snapshot


def background_run(job) -> dict[str, Any]:
    """Worker body of a heavy constrained run (``jobs.start_job(..., work=background_run)``).

    Computes only: it reads ``job.params["run"]`` (the snapshot from
    :func:`snapshot_run_inputs`), reports each segment with ``job.report`` and
    stops between segments when ``job.cancel`` is set (``JobCancelled``: no
    route). Returns ``{"result": <run_constrained_path result or None for "no
    path">, "notices": [dict]}``; the notices are the warnings the library
    raised, which ``errors.guard`` cannot collect from another thread. A run
    without a route also has ``"no_route": {"segment": k, "of": n}`` (the
    segment that found none, see :func:`no_route_segment_text`). The poll
    callback registers the route from it.
    """
    run = dict(job.params["run"])
    raster_source = run.pop("raster_source")
    outcome: dict = {}
    result, notices = jobs.run_collecting_warnings(
        run_constrained_path, raster_source, **run, cancel=job.cancel, progress=job.report, outcome=outcome)
    # a run without a route names the segment that had none (only then does the answer carry the extra key)
    answer = {"result": result, "notices": notices}
    if "no_route" in outcome:
        answer["no_route"] = outcome["no_route"]
    return answer


def status_text(job) -> str:
    """The status line of a running constrained job: its progress, and where a Stop takes effect.

    The job service's own line says "Stop ends it after the current step"; for this job the step is a
    segment, which cannot be interrupted, so the line names it. Once Stop was pressed the line says
    that the run is ending and adds no route.
    """
    text = job.progress_text()
    if job.status != RUNNING:
        return text
    text = text.replace("after the current step", "after the segment in flight")
    if job.cancel.is_set():
        text += " (stop requested, the run ends after the segment in flight and adds no route)"
    return text
