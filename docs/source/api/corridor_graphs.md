---
title: "Corridor Graphs"
summary: "Reduce many routes to the trenches they share so each trench is priced once."
status: unreleased
since: "unreleased"
available_in: source
module: "pyorps.graph.corridor"
api:
  - pyorps.corridor_graph_from_routes
  - pyorps.route_metrics
  - pyorps.cell_sharing_profile
  - pyorps.supercover_cells
---
# 🛤️ Corridor Graphs: Routes Reduced to Shared Trenches

A least-cost path is computed for one source-target pair at a time. In real
terrain the results are not independent: several routes follow the same field
track, road verge or gap between obstacles for part of their length. A planning
model that charges construction cost **per route** excavates those shared
stretches more than once.

A *corridor graph* removes that double count. Its nodes are the terminals plus
the cells where routes **merge** or **diverge**; its edges are the stretches
between them, each carrying its construction cost exactly once.

```python
from pyorps import PathFinder, corridor_graph_from_routes

finder = PathFinder(raster, source_coords=points, target_coords=points,
                    search_space_buffer_m=2000)
graph = finder.build_corridor_graph(terminals=points)

print(graph)
print(graph.overlap_report())
graph.save("corridor.gpkg")          # segments + nodes, two layers
finder.plot_corridor_graph()
```

:::{warning}
"Corridor" means two different things in pyorps. `PathFinder.corridor_geometry`,
`corridor_bounds` and the `corridor_first` constructor argument are about the
**search window** — the buffered polygon the raster is read inside. This page is
about the other sense: a stretch of ground several routes have in common.
:::

---

(corridor-graphs-no-tolerance-anywhere)=
## No tolerance anywhere

A raster route is an ordered list of cells over one grid, so *"these two routes
coincide here"* is an **identity**, not a proximity test. There is no snapping
radius, no buffer width and no threshold in the construction. That is what makes
a reported junction count reproducible rather than a function of a parameter
nobody can defend.

The unit of coincidence is the **step**, not the cell. pyorps prices a route
step by step:

```
step_cost = (raster[u] + sum(raster[intermediates]) + raster[v]) * factor
factor    = sqrt(dr² + dc²) / (2 + n_intermediates)
```

From `r2` upward a step spans more than two cells, so *"both routes use this
cell"* does not imply *"both pay this step"*. Cutting on cells would split a
step and force its cost to be apportioned between two segments by an invented
rule. Cutting on step identity cannot: a junction only ever lands on a step
endpoint, every step belongs to exactly one segment, and

> **segment metrics sum back to route metrics exactly, not approximately.**

`tests/test_graph/test_corridor.py` pins that as a conservation law across
`r0`–`r3`.

---

(corridor-graphs-two-constructions)=
## Two constructions

### Overlay — segment routes you already have

```python
from pyorps import corridor_graph_from_routes, route_metrics

routes = {(a, b): path.path_indices for (a, b), path in my_routes.items()}
graph = corridor_graph_from_routes(
    routes, finder.raster_handler.data[0],
    finder.raster_handler.window_transform,
    crs=finder.dataset.crs, steps=finder.steps,
)
report = graph.overlap_report(
    route_metrics=route_metrics(routes, raster, cell_size))
```

Use this to measure how much of an **existing layout** is shared. Every route
must be expressed in cells of the **same search window** — cell indices are
window-local, so routes cut on different windows cannot be compared at all. Route
them through one `PathFinder`.

### Distance network — derive the candidate graph

```python
graph = finder.build_corridor_graph(terminals=points, k_per_pair=1,
                                    validate_pair_costs=True)
```

One multi-source sweep seeds every terminal at distance 0 and labels each cell
with the terminal that reaches it most cheaply, partitioning the raster into
routing-metric Voronoi regions. Every step whose two ends fall in different
regions is a candidate connection priced at `dist[u] + step(u→v) + dist[v]`;
keeping the cheapest per terminal pair is Mehlhorn's terminal distance
network[^mehlhorn]. The routes those steps stand for are then overlaid and
contracted by the same code path as above.

The distance network is an **approximation** of the true pairwise metric — it
connects two terminals through the cheapest step on their shared boundary, which
is a real route and therefore an upper bound, tight whenever the true shortest
path crosses the bisector. `validate_pair_costs=True` runs the pairwise search
as well and records the gap in `graph.provenance["pair_cost_gap"]`. Report that
gap; it is a property of the construction, not a defect.

`cython` only. `raster_gpu` would need multi-source seeding in
`GpuSsspSession.solve` (it takes one source today) and `raster_fim` carries no
region label, so both raise rather than silently substituting a different edge
model.

---

(corridor-graphs-determinism)=
## Determinism

Ties are pervasive on rasters with few cost categories and the binary heap has
no stable order for equal priorities. Without a rule the region partition — and
therefore the whole graph — would vary with the order the terminals were listed
in. `MultiSourceSolver` applies a documented lexicographic rule on
`(new_dist, terminal_cell[region[current]], current)`: the key is the terminal's
**cell**, never its position in the argument array.

With strictly positive step costs that makes the forest permutation-independent:
any `u` that can tie for `v` has `dist[u] < dist[v]`, so it is popped before `v`
settles and its candidate is always seen. Zero-cost steps break that argument and
are reported by `solver.has_zero_cost_steps()`.

---

(corridor-graphs-what-a-segment-carries)=
## What a segment carries

`CorridorSegment.metrics` is a metric **vector**, not a single `cost` float:

| Key | Meaning |
|---|---|
| `length_m` | 2D length in CRS units, as `Path.total_length` reports it |
| `construction_cost` | `Σ(cell value × metres)` — EUR when the raster holds EUR/m |
| `routing_cost` | what the SEARCH minimized, in **cell** units |
| `length_by_category` | metres per raster cost value, so a segment can be re-priced under a different cost table without re-routing |

`use_count`, `members` and `CorridorGraph.pair_routes` are **diagnostic**.
Handing them to an optimiser reintroduces the pairwise view the graph exists to
remove: a segment is trenched once regardless of how many connections run over
it, so the count must not enter a cost term.

---

(corridor-graphs-robustness-knobs)=
## Robustness knobs

Both are off by default, because the exact construction is the defensible one.

`min_shared_length_m`
: Discard a shared run shorter than this and revert those steps to per-route
  membership, so two routes grazing for a couple of cells do not generate a
  junction. A run is demoted only when it is short in *every* route that carries
  it.

`cell_sharing_profile(routes, shape, radii=(0, 1, 2, 3))`
: The sensitivity a reviewer asks for. Repeats the coincidence test with each
  route's footprint dilated by `r` cells and reports how the shared fraction
  moves. `min_run=2` implements the rule that a transversal crossing is not
  sharing.

---

(corridor-graphs-scope-note)=
## Scope note

Once trenches are shared, the marginal cost of following an existing corridor is
material only, so the underlying problem is a Steiner tree problem and
shortest-path structure is a **candidate generator**. An optimiser on this graph
is optimal over the candidate graph, not over the raster — the same status the
pairwise pipeline already has with respect to its own candidate routes. The
corridor graph does not introduce that gap; it makes it visible.

[^mehlhorn]: Mehlhorn, K.: 'A faster approximation algorithm for the Steiner
    problem in graphs', *Inf. Process. Lett.*, 1988, **27**, (3), pp. 125–128
