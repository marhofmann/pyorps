---
title: "Cost Fields and Search Sessions"
summary: "Solve one field once and price many endpoint pairs from it."
status: experimental
since: "0.4.0"
available_in: pypi
module: "pyorps.graph.search_session"
api:
  - pyorps.CostField
  - pyorps.CostFieldSet
  - pyorps.SavedCostField
  - pyorps.SearchSession
  - pyorps.Leg
  - pyorps.full_window_buffer_m
  - pyorps.fields_fit_in_memory
---
# 📡 Retained Search Fields: `SearchSession` and `CostField`

`find_route` is one-shot: every call starts Dijkstra (or delta-stepping, or the
GPU/FIM backends) from scratch. Two opt-in classes retain the search state
between calls when you know you'll be asking related questions repeatedly.

| Use case | Class |
|---|---|
| One route whose control points get dragged interactively | `SearchSession` |
| One fixed terminal, many candidate endpoints to rank | `CostField` |

---

(cost-fields-searchsession-editing-a-route)=
## `SearchSession`: editing a route

```python
session = finder.search_session(algorithm="dijkstra")
path = session.route([source, target])
# ... the user drags the target ...
path = session.update([source, new_target])
```

`update(points)` returns exactly the same cell sequence as chaining fresh
`find_route` calls on each consecutive pair — retention is only a speedup, not
an approximation. Unchanged legs are skipped entirely; a dirty leg resumes
the existing search tree instead of restarting it.

### Reverse trees (`reuse_reverse`)

By default, `SearchSession` keeps one tree per leg, rooted at the leg's
**start**. Moving the *target* is cheap (the tree just resumes); moving the
*source* is not, because the tree is rooted at the old source and a new one
has to grow from scratch.

```python
session = finder.search_session(algorithm="dijkstra", reuse_reverse=True)
```

With `reuse_reverse=True`, `SearchSession` additionally builds a tree rooted
at each leg's **end** on the first source move for that leg, then reuses it
for every subsequent drag of that source. This is sound because the raster
graph is undirected with symmetric edge weights: `d(u → v) == d(v → u)`, so a
tree rooted at the *end* answers exactly the same distances as one rooted at
the start.

The one place this is *not* free: the cell sequence may differ from a fresh
`find_route` when several shortest paths tie (neighbor visitation order
differs by direction). The **cost** never differs. Because a warm forward
tree is always preferred over a reverse tree, `reuse_reverse=False` (the
default) is bit-identical to `find_route` in every case — turning it on
trades that tie-breaking guarantee for far cheaper source drags.

`max_trees` (default 4) caps the number of resident trees with LRU eviction —
a Dijkstra tree is 13 bytes/cell, so an unbounded cache on a large raster is
not a cache.

---

(cost-fields-costfieldset-many-fixed-terminals-the-siting-ent)=
## `CostFieldSet`: many fixed terminals — the siting entry point

`finder.cost_fields(origins)` is the natural shape of a siting problem. The
substation may stand anywhere; the turbines and grid connection points do not
move. So the search is rooted at **them** and read in the other direction —
one sweep per terminal instead of one per candidate.

```python
with finder.cost_fields(turbines + pccs, labels=names) as fields:
    costs = fields.costs_to(candidate_sites)      # (n_terminals, n_candidates)
    value, which = fields.nearest(candidate_sites, labels=pcc_names)
    leg = fields.path_to("WT0", best_site)        # cells, coords, length_m, cost
```

Ten terminals against forty million candidates is **ten searches, not forty
million**. The inversion is exact, not an approximation: the raster graph is
undirected, so `d(terminal → candidate) == d(candidate → terminal)`. Measured
against independently computed pairwise routes on the CIRED case study:
agreement **5.5e-09** relative, symmetry **exactly 0.0**.

Fields are settled with delta-stepping by default, and the set decides for
itself whether to hold them in memory or page them through disk —
`fields_fit_in_memory` makes the call, `spill=True/False` forces it. A spilled
set settles each field once, writes it, and reopens it on demand, so only one
field is ever resident.

### Is the parallel kernel safe to preprocess with?

Yes, and it is checked rather than assumed. Over the whole 41.8 M-cell field,
against the float64 Dijkstra:

| threads | cells worse | cells better | reachability differences |
|---|---|---|---|
| 1 | 0 | 0 | 0 |
| 8 | 0 | 0 | 0 |
| 12 | 0 | 0 | 0 |

Two runs at 12 threads are **bit-identical**, and 1 thread against 12 is
**bit-identical**. The residual ±1.5e-05 against Dijkstra is float32 label
storage, nothing more. CPU and GPU delta-stepping agree bit-for-bit with each
other, so the backend is purely a speed choice.

---

(cost-fields-costfield-one-fixed-terminal-many-candidates)=
## `CostField`: one fixed terminal, many candidates

`SearchSession` amortizes across edits to *one* route. `CostField` amortizes
across an unbounded number of *different* endpoints against *one* terminal
that never moves — the shape of substation siting, service-area maps, or
ranking the k cheapest connections among many candidates.

```python
import pyorps

with finder.cost_field(substation_coords) as field:
    costs = field.costs_to(candidate_sites)      # (n,) float64, EUR
    best = candidate_sites[int(np.argmin(costs))]
    route = field.path_to(best, calculate_metrics=True)
```

`CostField`, `SearchSession` and `full_window_buffer_m` are exported from the
package root, so `import pyorps` is enough — no reaching into subpackages.

### Picking the kernel: `algorithm="auto"`

The default is `"auto"`, which picks the fastest **full-field** kernel the
backend in use offers. That is the right default because the whole point of a
field is to settle once and then look up many times:

```python
with finder.cost_field(pcc) as field:
    field.to_geotiff("reach_from_pcc.tif")
print(field.algorithm)          # "delta-stepping" -- what auto resolved to
```

Pass `algorithm="dijkstra"` for the opposite case. It is the only backend that
can **stop early**, so pricing a handful of nearby candidates never settles the
rest of the window:

```python
with finder.cost_field(pcc, algorithm="dijkstra") as field:
    costs = field.costs_to(two_or_three_points)
```

Measured on a 41.8 M-cell raster at `r2`, one field, same root, on an
otherwise idle 16-core machine (under 4 % foreign CPU):

| Backend | Algorithm | Time | Rate | Speedup |
|---|---|---|---|---|
| `cython` | `dijkstra` (opt-in) | 38.1 s | 1.10 M cells/s | 1.0x |
| `cython` | `delta-stepping` (auto, default) | 4.7 s | 8.96 M cells/s | **8.1x** |
| `raster_gpu` | `delta-stepping` (auto, default) | 2.6 s | 15.79 M cells/s | **14.7x** |

The three agree to a maximum relative deviation of 1.3e-05 — float32 rounding,
not a different answer.

#### How many threads

`"auto"` asks for **three quarters** of the cores. More is not better here:
the kernel synchronises every bucket on a spin barrier, so a worker that
cannot get a core holds all the others spinning. On the same 16-core machine,
same field, same load:

| threads | 4 | 8 | 12 | 13 | 14 | 16 |
|---|---|---|---|---|---|---|
| seconds | 7.03 | 5.09 | **4.61** | 4.69 | 6.78 | **917** |

Asking for every core is 199x slower than asking for twelve. Pass
`num_threads=` to override the default, but do not set it to the core count.

`graph_api="raster_gpu"` and `"raster_fim"` choose their kernel from the graph
API rather than from the algorithm string, so on those backends `"auto"` simply
names something the backend accepts.

### Georeferencing the field

`field_array()` is a plain window-shaped array; `transform` and `crs`
georeference it, and `to_geotiff` writes it out in one call.

```python
with finder.cost_field(origin) as field:
    arr = field.field_array()                 # (rows, cols) float32
    x, y = rasterio.transform.xy(field.transform, row, col)
    field.to_geotiff("field.tif")             # nodata=-1.0 where unreachable
```

The field covers the **search window**, not the whole source raster, so
`transform` is the window's. To make the window the entire raster — which is
what a siting run over scattered candidates needs — size the buffer with
`full_window_buffer_m`:

```python
finder = pyorps.PathFinder(
    dataset_source=raster,
    source_coords=xy, target_coords=[xy],
    search_space_buffer_m=pyorps.full_window_buffer_m(raster),
)
```

`search_space_buffer_m=None` does **not** mean "the whole raster": it runs a
heuristic estimator sized for a single source/target pair, which is the wrong
shape for a field queried all over the map.

### Saving a field, when they don't all fit

Keeping fields live is always better — an open `CostField` answers everything a
saved one does and needs no decode. `save` / `open` exist for the case where
the set doesn't **fit**: ten 240 M-cell fields are about 19 GB live.

```python
if pyorps.fields_fit_in_memory(len(terminals), rows * cols):
    fields = [finder.cost_field(t, algorithm="auto") for t in terminals]
else:
    for t in terminals:                       # page them through
        with finder.cost_field(t, algorithm="auto") as f:
            f.save(cache / f"field_{t}.npz")

with pyorps.CostField.open(cache / "field_WT0.npz") as field:
    costs = field.costs_to(candidate_sites)   # same units, same inf rule
    cells = field.path_cells(best)            # route, from stored steps
    xy = field.path_coords(best)
    metres = field.path_length_m(best)
```

A saved field stores the distance (4 B/cell) and, unless `with_paths=False`,
the predecessor as **one byte** — the index of the step that reached the cell,
rather than a 4-byte absolute id. So it is 5 B/cell on disk and *smaller in
memory than the live kernel workspace it came from* (209 MB against 836 MB on
41.8 M cells).

Measured on 41.8 M cells against a 4.67 s settle of the same field:

| | save | open | size |
|---|---|---|---|
| `compress=False` (default) | 2.59 s | **0.19 s** | 5.00 B/cell |
| `compress=True` | 12.1 s | 0.94 s | 3.07 B/cell |

Uncompressed is the default because it is the only setting cheaper than simply
searching again: writing costs about half a settle, reading back a
twenty-fifth. Deflating costs more than two settles to save 39 % of the disk.

**One difference from a live field.** A live `CostField` still has the raster,
so a coordinate landing on an impassable cell is nudged to the nearest passable
one. A saved field has no raster, so such a point reads as unreachable instead.
Measured over 5,000 random probes: 788 reachability disagreements, **all 788**
on impassable cells and none anywhere else; off those cells reachability is
identical and costs agree to 5.8e-08.

`with_paths=True` works on `dijkstra`, `delta-stepping` and the GPU kernel
(whose `v5_repair_pred` post-pass makes every predecessor agree with the final
labels, not just the ones a walk touches). The eikonal backend traces by
descent rather than by predecessors, so there it raises and `with_paths=False`
saves costs alone.

### Why the direction doesn't matter

The naive way to price `n` candidate sites against one fixed terminal is `n`
separate route solves, each rooted at a *different* candidate. `CostField`
instead solves **once**, rooted at the fixed terminal, and reads every
candidate's distance back out of that single field:

```
F(s) = d(fixed_terminal, s)
```

This is valid because the raster graph is undirected with symmetric edge
weights: `d(a → b) == d(b → a)` for every pair. A field rooted at the fixed
terminal therefore answers `d(terminal, candidate)` for *every* candidate in
one solve, instead of one solve per candidate.

### Preconditions

`CostField` checks this precondition at construction and raises `ValueError`
if it does not hold:

* the neighborhood step set must be closed under negation (`directed=True`,
  the default for the cython and raster_gpu backends — a
  hand-passed or `directed=False` half step set is rejected);
* the backend must not declare asymmetric edge weights
  (`symmetric_edge_weights = False`, e.g. a future signed slope-response
  objective);
* a FIM backend must not be using a *lazy* grade limit — that enforces the
  constraint per source/target pair, which one field cannot reproduce (an
  *eager* grade limit, pre-masking the raster, is fine).

A DEM/gradient objective does **not** break symmetry: the slope term bins
`|height(v) − height(u)|`, which is direction-free by construction.

### Units and precision

`costs_to()` returns the same units as `Path.total_cost` (cell value × cell
size). It matches `Path.total_cost` to floating-point rounding on the exact
(cython) backends. On the `raster_fim` (eikonal) backend, the field value is
the **continuous** quantity the search actually minimized; `Path.total_cost`
recomputes a **discrete** retrace of the traced path, which prices the
continuous value upward. Don't compare the two on that backend — see
`PathFinder._total_cost_basis`.

### Settling and memory

The cython Dijkstra backend settles on demand: `costs_to()` resumes the
search only as far as needed for the points you asked about. Below 1024
points this is per-point resume; at or above it, one `settle_all()` is
cheaper than as many `search_until` calls, so `costs_to()` switches
automatically (`settle="auto"`; override with `settle="each"` / `"all"` /
`"none"`). All other backends (delta-stepping, GPU, FIM) are full-field on
construction — there is nothing to settle incrementally.

A Dijkstra tree is 13 bytes/cell — roughly 1 GB on a 73-million-cell window.
Hold one field at a time, gather everything you need from it, then `close()`
(or use it as a context manager) before opening the next one:

```python
for terminal in terminals:
    with finder.cost_field(terminal) as field:
        results[terminal] = field.costs_to(all_candidates)
```

`field_array()` returns the whole field as a window-shaped array for map
output (float32 by default — half the memory of float64, and no downstream
raster consumer needs the extra digits).
