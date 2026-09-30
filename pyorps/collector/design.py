"""An explicit ``D_share`` collector design and its independent re-pricer.

The DP (:mod:`pyorps.collector.reference`) returns a value per root; the
traceback turns the argmin into a :class:`Design`: a trench tree mapped
onto graph edges, the systems (cables) with their options, the
switching-station junctions, and where each system starts and ends.

:func:`reprice` prices a design from scratch under
:class:`~pyorps.collector.model.CollectorModel` -- it routes every system
along its trench-tree path, recomputes each trench step's cable count and
derating, checks every constraint and sums the costs -- sharing no code
path with the DP's recursion. The plan (CODE-02, D7) never trusts a DP
value it has not re-priced this way.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from pyorps.collector.model import INF, CollectorModel

__all__ = ["Design", "DesignError", "Junction", "Layout", "System",
           "layout", "reprice"]


class DesignError(ValueError):
    """A design violates a rule of ``D_share``."""


@dataclass
class System:
    """One electrical connection (a cable, or ``p`` parallel cables).

    ``source`` / ``sink`` are ``("turbine", i)``, ``("junction", j)`` or
    ``("root", 0)``; power flows from source to sink.
    """
    mask: int
    option: tuple[int, int]          # (type index, p)
    source: tuple[str, int] | None = None
    sink: tuple[str, int] | None = None


@dataclass
class Junction:
    """One busbar section of a switching station."""
    tnode: int
    inputs: list[int] = field(default_factory=list)
    output: int = -1


@dataclass
class Design:
    """A trench tree plus the electrical network laid in it.

    ``tnodes[k]`` is the graph node of trench-tree node ``k``;
    ``tedges`` are ``(child tnode, parent tnode, graph edge index)``,
    oriented towards ``root_tnode``.
    """
    tnodes: list[int]
    tedges: list[tuple[int, int, int]]
    root_tnode: int
    turbine_tnode: dict[int, int]
    systems: list[System]
    junctions: list[Junction]
    cost: float = INF

    def summary(self) -> dict:
        """Counts a result table prints beside the cost."""
        stations = {j.tnode for j in self.junctions}
        return {
            "trench_edges": len(self.tedges),
            "systems": len(self.systems),
            "stations": len(stations),
            "junctions": len(self.junctions),
            "parallel_systems": sum(1 for s in self.systems
                                    if s.option[1] > 1),
        }


def _edge_pricer(graph):
    """``price(edge key, cell a, cell b) -> (trench EUR, length m)``.

    A :class:`~pyorps.collector.reference.CollectorGraph` prices by edge
    index; anything with a ``price_step`` method (the raster pricer of
    plan D7) prices the step between the two cells itself.
    """
    if hasattr(graph, "price_step"):
        return graph.price_step

    def price(ei, a, b):
        u, w, c, ln = graph.edges[ei]
        if {a, b} != {u, w}:
            raise DesignError(f"trench edge between {a} and {b} is not on "
                              f"graph edge {ei}")
        return c, ln
    return price


@dataclass
class Layout:
    """A validated design with every system routed through its trench tree.

    Built by :func:`layout`, shared by :func:`reprice` and the voltage check
    (:mod:`pyorps.collector.voltage`), so both read the same routes.

    Attributes:
        priced: ``(trench EUR, length m)`` per trench edge of the design.
        node_of: The trench node of every electrical node (``("root", 0)``,
            ``("turbine", i)``, ``("junction", j)``).
        outs, ins: Out- and in-systems of every electrical node.
        ups, dns: Per trench edge, the systems crossing it towards and away
            from the root.
        routes: Per system, the trench edges it runs through.
    """
    priced: list[tuple[float, float]]
    node_of: dict
    outs: dict
    ins: dict
    ups: dict
    dns: dict
    routes: list[list[int]]

    def length_m(self, sid: int) -> float:
        """Trench length of system ``sid`` (0 for a system that stays on one
        trench node)."""
        return float(sum(self.priced[e][1] for e in self.routes[sid]))


def layout(design: Design, graph, turbines, model: CollectorModel, *,
           root_transit: bool = True) -> Layout:
    """Validate ``design`` and route its systems; raise :class:`DesignError`.

    Every structural rule :func:`reprice` enforces is checked here (tree,
    terminals, no-transit, flow conservation, radial network, systems not
    passing a turbine or the root); the pricing rules (derating, R-once,
    panels, bays) stay in :func:`reprice`.
    """
    price = _edge_pricer(graph)
    tn = design.tnodes
    root = design.root_tnode
    # ---- the trench tree
    parent: dict[int, int] = {}
    pedge: dict[int, int] = {}
    priced: list[tuple[float, float]] = []
    for idx, (c, p, ei) in enumerate(design.tedges):
        if c in parent:
            raise DesignError(f"trench node {c} has two parents")
        if c == root:
            raise DesignError("the root trench node has a parent")
        priced.append(price(ei, tn[c], tn[p]))
        parent[c] = p
        pedge[c] = idx
    depth: dict[int, int] = {root: 0}
    for x in range(len(tn)):                 # iterative: any tree depth
        path: list[int] = []
        on_path: set[int] = set()
        y = x
        while y not in depth:
            if y not in parent:
                raise DesignError(f"trench node {y} is not connected to "
                                  f"the root")
            if y in on_path:
                raise DesignError("the trench network has a cycle")
            on_path.add(y)
            path.append(y)
            y = parent[y]
        d = depth[y]
        for z in reversed(path):
            d += 1
            depth[z] = d
    # ---- terminals and no-transit
    turb_cells = {int(c): i for i, c in enumerate(turbines)}
    if sorted(design.turbine_tnode) != list(range(model.n)):
        raise DesignError("every turbine needs exactly one trench node")
    for i, x in design.turbine_tnode.items():
        if tn[x] != turbines[i]:
            raise DesignError(f"turbine {i} trench node is off its cell")
    special = set(design.turbine_tnode.values()) | {root}
    for x, cell in enumerate(tn):
        if cell in turb_cells and x not in special:
            raise DesignError("a trench passes through a turbine cell")
        if (not root_transit and cell == tn[root] and x != root):
            raise DesignError("a trench passes through the UW cell")
    # ---- electrical network
    node_of: dict[tuple[str, int], int] = {("root", 0): root}
    for i, x in design.turbine_tnode.items():
        node_of[("turbine", i)] = x
    for j, jn in enumerate(design.junctions):
        if tn[jn.tnode] in turb_cells or jn.tnode == root:
            raise DesignError(f"junction {j} sits on a turbine or the root")
        node_of[("junction", j)] = jn.tnode
    outs: dict = defaultdict(list)
    ins: dict = defaultdict(list)
    for sid, s in enumerate(design.systems):
        if s.source is None or s.sink is None:
            raise DesignError(f"system {sid} lacks an end")
        ti, p = s.option
        if not (0 <= ti < len(model.types)) or not (1 <= p <= model.p_max):
            raise DesignError(f"system {sid} has option {s.option}, not in "
                              f"the catalogue ({len(model.types)} types, "
                              f"p <= {model.p_max})")
        if s.source[0] == "root":
            raise DesignError(f"system {sid} starts at the root")
        outs[s.source].append(sid)
        ins[s.sink].append(sid)
    for i in range(model.n):
        if len(outs[("turbine", i)]) != 1:
            raise DesignError(f"turbine {i} needs exactly one out-system")
    for j, jn in enumerate(design.junctions):
        if len(outs[("junction", j)]) != 1:
            raise DesignError(f"junction {j} needs exactly one output")
        if len(ins[("junction", j)]) < 2:
            raise DesignError(f"junction {j} has fewer than two inputs")
        if not model.allow_stations:
            raise DesignError("stations are not allowed")
    # flow conservation: out mask = own turbine | inputs
    for key, sids in outs.items():
        own = (1 << key[1]) if key[0] == "turbine" else 0
        acc = own
        for sid in ins[key]:
            acc |= design.systems[sid].mask
        for sid in sids:
            if design.systems[sid].mask != acc:
                raise DesignError(f"flow conservation fails at {key}")
    # radial into the root: follow sinks
    for sid, s in enumerate(design.systems):
        seen = set()
        cur = s
        while cur.sink[0] != "root":
            if cur.sink in seen:
                raise DesignError("the electrical network has a cycle")
            seen.add(cur.sink)
            cur = design.systems[outs[cur.sink][0]]
    covered = 0
    for sid in ins[("root", 0)]:
        covered |= design.systems[sid].mask
    if covered != model.full:
        raise DesignError("not every turbine reaches the root")
    # ---- route every system along its trench path
    ups: dict = defaultdict(list)
    dns: dict = defaultdict(list)
    routes: list[list[int]] = []
    for sid, s in enumerate(design.systems):
        a, b = node_of[s.source], node_of[s.sink]
        x, y = a, b
        up, dn = [], []
        na, nb = [a], [b]
        while depth[x] > depth[y]:
            up.append(pedge[x]); x = parent[x]; na.append(x)
        while depth[y] > depth[x]:
            dn.append(pedge[y]); y = parent[y]; nb.append(y)
        while x != y:
            up.append(pedge[x]); x = parent[x]; na.append(x)
            dn.append(pedge[y]); y = parent[y]; nb.append(y)
        for z in (na + nb[-2::-1])[1:-1]:
            if z in special:
                raise DesignError(f"system {sid} passes a turbine or the root")
        for e in up:
            ups[e].append(sid)
        for e in dn:
            dns[e].append(sid)
        routes.append(up + dn)
    return Layout(priced=priced, node_of=node_of, outs=outs, ins=ins,
                  ups=ups, dns=dns, routes=routes)


def reprice(design: Design, graph, turbines, model: CollectorModel, *,
            root_transit: bool = True) -> tuple[float, dict]:
    """Price ``design`` from scratch; raise :class:`DesignError` if invalid.

    ``graph`` is the :class:`~pyorps.collector.reference.CollectorGraph`
    the design was traced on, or a
    :class:`~pyorps.collector.raster_pricer.RasterStepPricer` for a design
    traced on a raster.

    Returns ``(cost, breakdown)`` with the trench, cable, turbine-panel,
    station and bay parts.
    """
    lay = layout(design, graph, turbines, model, root_transit=root_transit)
    priced, outs, ins = lay.priced, lay.outs, lay.ins
    ups, dns = lay.ups, lay.dns
    # ---- price
    trench = cable = 0.0
    for e in range(len(design.tedges)):
        crossing = ups[e] + dns[e]
        if not crossing:
            raise DesignError(f"trench edge {e} carries no system")
        for lst in (ups[e], dns[e]):
            acc = 0
            for sid in lst:
                mk = design.systems[sid].mask
                if acc & mk:
                    raise DesignError(f"R-once fails on trench edge {e}")
                acc |= mk
        r = model.rate([(design.systems[s].mask, design.systems[s].option)
                        for s in crossing])
        if r == INF:
            raise DesignError(f"trench edge {e} is infeasible "
                              f"(m_max or derated ampacity)")
        cc, ln = priced[e]
        trench += cc
        cable += r * ln
    panels = 0.0
    for i in range(model.n):
        key = ("turbine", i)
        out = design.systems[outs[key][0]]
        cond = out.option[1] + sum(design.systems[s].option[1]
                                   for s in ins[key])
        if not model.turbine_ok(out.mask, cond):
            raise DesignError(f"turbine {i} exceeds switchgear or panels")
        panels += model.turbine_panel_eur * cond
    stations = 0.0
    for tnode in {j.tnode for j in design.junctions}:
        stations += model.station_building_eur
    for j, jn in enumerate(design.junctions):
        cond = sum(design.systems[s].option[1] for s in ins[("junction", j)])
        cond += design.systems[outs[("junction", j)][0]].option[1]
        stations += model.station_panel_eur * cond
    bays = 0.0
    for sid in ins[("root", 0)]:
        s = design.systems[sid]
        if not model.bay_ok(s.mask, s.option):
            raise DesignError(f"system {sid} exceeds the bay rating")
        bays += model.bay_eur * s.option[1]
    total = trench + cable + panels + stations + bays
    return total, {"trench": trench, "cable": cable, "turbine_panels": panels,
                   "stations": stations, "bays": bays}
