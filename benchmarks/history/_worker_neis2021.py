"""The pre-pyorps prototype from the NEIS 2021 paper, as a benchmark job.

Reference: M. Hofmann, M. Franz, T. Stetz, M. Hajdu, "Comparison of Best
Practices for Evaluation of New Power Line Connections in Automated Power
Systems Planning", NEIS 2021. The paper's technique 2) rasterises the land
registry data to 1 m, connects each cell to its R = 2 neighbourhood ("due to
the effect of raster network connectivity on the elongation error"), and runs
Dijkstra. Its tooling references are scikit-image [7] and SciPy [5].

The solver is the author's own ``MyMCP`` from that era, transcribed from
``my_mcp/_my_mcp.pyx`` (2022): it subclasses ``skimage.graph.MCP_Geometric``
but calls ``MCP.__init__`` directly, which skips MCP_Geometric's
"all offset components must be 0, 1, or -1" validation while keeping its
C-level geometric cost. That bypass is what made the paper's R = 2 reachable at
native speed. Its R1 and R2 offset sets are identical to the ones pyorps uses
today, so this comparison isolates the cost model rather than connectivity.

TWO SEMANTIC DIFFERENCES to today's pyorps are deliberately preserved, because
they are the substance of what changed:

1. Edge cost. MCP_Geometric charges ``(c[a] + c[b]) / 2 * ||offset||`` -- the
   mean of the two ENDPOINTS. pyorps charges
   ``(c[a] + c[b] + sum(c[intermediates])) * ||offset|| / (2 + n_inter)`` --
   the mean over every cell the step actually crosses. For R2 knight moves the
   two disagree, because MCP never looks at the cells in between.
2. Passability. Because MCP never looks at the intermediate cells, an R2 step
   can jump diagonally THROUGH a one-cell barrier that pyorps refuses.

Both effects are quantified by ``score.py``, which re-evaluates every route --
prototype and pyorps alike -- under one single canonical cost model.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback

JOB = json.loads(open(sys.argv[1], encoding="utf-8").read())

REC = {k: JOB[k] for k in ("job_id", "engine", "version", "kind", "neighborhood",
                           "graph_api", "algorithm", "repeat", "exec_mode")
       if k in JOB}
REC["pid"] = os.getpid()
REC["status"] = "started"


def peak_rss_gb():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        peak = getattr(mi, "peak_wset", None) or getattr(mi, "rss", None)
        return round(peak / 2 ** 30, 3) if peak else None
    except Exception:
        return None


def dump(status, **kw):
    REC.update(kw)
    REC["status"] = status
    REC["peak_rss_gb"] = peak_rss_gb()
    tmp = JOB["out_json"] + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(REC, fh, indent=1, default=str)
    os.replace(tmp, JOB["out_json"])
    print(json.dumps({k: v for k, v in REC.items() if k != "paths"},
                     default=str)[:1500], flush=True)


def offsets_for(neighborhood: str):
    """Return the R-neighbourhood offsets, identical to the ones pyorps uses.

    Imported from pyorps when available so the two solvers see the SAME move
    set and the comparison isolates the cost model rather than connectivity.
    Falls back to the standard construction (all steps whose components are
    coprime, so no move is a repeat of a shorter one).

    NOTE ``directed=True``: pyorps returns only one step per undirected pair
    when directed=False, because the graph libraries add the reverse edge
    themselves. MCP has no such convention -- handed the halved set it can
    only ever move in half the directions, and most targets become
    unreachable. The symmetry check below makes that failure loud.
    """
    origin = "pyorps"
    try:
        sys.path.insert(0, JOB["pyorps_for_steps"])
        from pyorps.utils.neighborhood import get_neighborhood_steps
        offs = [(int(a), int(b))
                for a, b in get_neighborhood_steps(neighborhood, directed=True)]
    except Exception:
        origin = "fallback"
        r = {"r0": 1, "r1": 1, "r2": 2, "r3": 3}[neighborhood]
        offs = []
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                if dr == 0 and dc == 0:
                    continue
                if neighborhood == "r0" and dr != 0 and dc != 0:
                    continue
                if math.gcd(abs(dr), abs(dc)) != 1:
                    continue
                offs.append((dr, dc))

    as_set = set(offs)
    missing = [o for o in offs if (-o[0], -o[1]) not in as_set]
    if missing:
        raise ValueError(
            f"{neighborhood} offsets from '{origin}' are not symmetric "
            f"({len(offs)} offsets, missing reverses for {missing[:4]}) -- "
            f"MCP would search a one-way graph and most targets would be "
            f"unreachable")
    return offs, origin


try:
    import numpy as np
    from skimage.graph import MCP_Geometric, MCP_Flexible, _mcp
    import skimage
    from shapely.geometry import LineString
    from affine import Affine

    REC["skimage_version"] = skimage.__version__

    class MyMCP(MCP_Geometric):
        """The original prototype's R2 solver, reproduced verbatim.

        Transcribed from the author's own ``my_mcp/_my_mcp.pyx`` (2022):

            class MyMCP(_mcp.MCP_Geometric):
                def __init__(self, costs, additional_moves):
                    offsets = make_offsets(2, True)
                    offsets.extend([am for am in additional_moves])
                    self.offsets = np.array(offsets, dtype=OFFSET_D)
                    _mcp.MCP.__init__(self, costs, offsets=self.offsets,
                                      fully_connected=True, sampling=None)

        The trick is the last line. ``MCP_Geometric.__init__`` is where the
        ``all offset components must be 0, 1, or -1`` check lives, so calling
        ``MCP.__init__`` instead skips the validation while still inheriting
        ``MCP_Geometric._travel_cost`` -- the C-level
        ``offset_length * 0.5 * (old + new)``.

        So the R = 2 the NEIS 2021 paper reports WAS reachable at native speed;
        it just needed this bypass. Verified against skimage 0.26: identical
        cumulative cost and identical path to the MCP_Flexible formulation,
        3.4x faster. Note it is a plain Python class even in the .pyx file --
        no compilation is involved in the trick itself.
        """

        def __init__(self, costs, offsets):
            self.offsets = np.array(offsets, dtype=np.int8)
            _mcp.MCP.__init__(self, costs, offsets=self.offsets,
                              fully_connected=True, sampling=None)

    class MCPFlexibleGeometric(MCP_Flexible):
        """Same cost model through the public, supported API.

        Kept as the fallback if a future skimage breaks the bypass above, and
        as the cross-check that the bypass still means what it should. Accepts
        arbitrary offsets but calls back into Python per relaxation.
        """

        def travel_cost(self, old_cost, new_cost, offset_length):
            return offset_length * 0.5 * (old_cost + new_cost)

    def verify_bypass():
        """Confirm MyMCP still computes the documented cost model.

        A silently-changed base class would leave the benchmark reporting a
        different cost function under the prototype's name, which is exactly
        the sort of thing this whole harness exists to catch.
        """
        rng = np.random.default_rng(0)
        probe = rng.integers(1, 500, size=(48, 48)).astype(np.float64)
        po = np.array([(-2, -1), (-1, 2), (1, 1), (0, 1), (1, 0),
                       (2, 1), (1, -2), (-1, -1), (0, -1), (-1, 0),
                       (-1, 1), (1, -1), (2, -1), (-2, 1), (-1, -2),
                       (1, 2)], dtype=np.int8)
        a = MyMCP(probe.copy(), po)
        b = MCPFlexibleGeometric(probe.copy(), offsets=po)
        ca, _ = a.find_costs([(0, 0)], [(47, 47)])
        cb, _ = b.find_costs([(0, 0)], [(47, 47)])
        return bool(np.isclose(ca[47, 47], cb[47, 47], rtol=1e-9))

    meta = json.loads(open(JOB["inputs"]["window_meta"], encoding="utf-8").read())
    transform = Affine(*meta["transform"])

    t_load0 = time.perf_counter()
    costs_u16 = np.load(JOB["inputs"]["window_cost"])
    t_load1 = time.perf_counter()

    # ignore_max_cost=False in the case study: the 65535 class stays traversable
    # but ruinous. Reproduced here rather than masking those cells to inf, so
    # both solvers face the identical feasible set.
    costs = costs_u16.astype(np.float64)
    if JOB.get("max_cost_impassable"):
        costs[costs_u16 == 65535] = np.inf

    def rc(x, y):
        col, row = ~transform * (x, y)
        return int(math.floor(row)), int(math.floor(col))

    src_xy = JOB["source_xy"]
    start = rc(*src_xy)
    ends = {int(b): rc(x, y) for b, (x, y) in JOB["target_xy"].items()}

    offs, offs_origin = offsets_for(JOB["neighborhood"])
    REC["n_offsets"] = len(offs)
    REC["offsets_origin"] = offs_origin

    # Use the prototype's own solver at every neighbourhood. For unit-only
    # offsets it is behaviourally identical to plain MCP_Geometric (same C
    # _travel_cost, same offsets); above R1 it is the only native option.
    offs_arr = np.array(offs, dtype=np.int8)
    try:
        ok = verify_bypass()
        if not ok:
            raise RuntimeError("MyMCP and MCP_Flexible disagree on the cost "
                               "model; the MCP_Geometric bypass no longer "
                               "computes offset_length * 0.5 * (old + new)")
        cls, native = MyMCP, True
    except Exception as exc:
        # Never silently downgrade: the fallback is 3.4x slower, so a runtime
        # measured with it is not comparable to one measured without it.
        REC["mcp_bypass_error"] = repr(exc)
        cls, native = MCPFlexibleGeometric, False

    REC["mcp_class"] = cls.__name__
    REC["mcp_native"] = native

    t0 = time.perf_counter()
    mcp = (cls(costs, offs_arr) if native
           else cls(costs, offsets=offs_arr))
    t1 = time.perf_counter()
    cum, _tb = mcp.find_costs([start], list(ends.values()))
    t2 = time.perf_counter()

    paths_out = []
    for bus, end in ends.items():
        tb0 = time.perf_counter()
        idx = mcp.traceback(end)
        tb1 = time.perf_counter()
        rows = np.array([p[0] for p in idx])
        cols = np.array([p[1] for p in idx])
        # pixel centres, matching how pyorps writes path geometry
        xs = transform.c + (cols + 0.5) * transform.a + (rows + 0.5) * transform.b
        ys = transform.f + (cols + 0.5) * transform.d + (rows + 0.5) * transform.e
        geom = LineString(np.column_stack([xs, ys]))
        d = np.hypot(np.diff(xs), np.diff(ys))
        paths_out.append(dict(
            group=0, target_row=None, bus=int(bus),
            graph_api="skimage." + cls.__name__, algorithm="dijkstra",
            total_cost=float(cum[end]),          # the prototype's OWN metric
            total_length=float(d.sum()),
            geom_length_m=float(geom.length),
            n_vertices=len(idx),
            euclidean_distance=float(math.dist(
                (xs[0], ys[0]), (xs[-1], ys[-1]))),
            search_space_buffer_m=meta.get("search_space_buffer_m"),
            runtimes={"window_load": t_load1 - t_load0,
                      "mcp_init": t1 - t0,
                      "find_costs": t2 - t1,
                      "traceback": tb1 - tb0},
            wkt=geom.wkt,
        ))

    t_all1 = time.perf_counter()
    dump("ok",
         routing_total_s=t_all1 - t0,
         groups=[dict(group=0, n_targets=len(ends), init_s=t1 - t0,
                      find_route_s=t2 - t1, n_paths=len(paths_out))],
         window_shape=list(costs_u16.shape),
         n_paths=len(paths_out), paths=paths_out)
except Exception:
    dump("error", error=traceback.format_exc())
    sys.exit(1)
