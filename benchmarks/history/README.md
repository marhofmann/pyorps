# MV-Oberrhein history benchmark

Runs the **CIRED 2025 case study** — connect one PV plant to the MV-Oberrhein
grid, evaluating 8 candidate points of common coupling on a 1 m ALKIS cost
raster — against **every relevant pyorps version** and against the
**pre-pyorps scikit-image prototype from the NEIS 2021 paper**.

It answers three questions with measurements rather than release notes:

1. **Runtime** — how long the same job took, version by version.
2. **Cost / accuracy** — whether the routes actually got better, measured
   under one canonical cost model rather than each version's own yardstick.
3. **Throughput** — what changes when the same matrix is run in parallel
   instead of serially.

## Run it

```bash
# from the repository root
.venv/Scripts/python.exe benchmarks/history/run_history_benchmark.py --profile standard
```

That is the whole command. Preparation, execution (serial *and* parallel) and
scoring all happen inside it, and it is safe to re-run — every prepared
artefact is cached.

| profile | versions | neighbourhoods | repeats | jobs | rough serial+parallel wall time |
|---|---|---|---|---|---|
| `quick` | 5 | r1, r2 | 1 | 16 | ~40 min |
| `standard` | 8 | r0, r1, r2 | 2 | 84 | **~3 h** |
| `full` | 13 | r0, r1, r2, r3 | 3 | ~230 | ~15–25 h |

Estimates come from measured single-job times on this machine (rasterise ~33 s,
cython R2 ~58 s, networkit R2 ~200 s, prototype R2 ~18 s). `--dry-run` prints
the exact matrix before committing to it.

Useful switches:

```bash
--exec serial            # only the comparable timings, skip the throughput pass
--profile quick          # short run
--versions HEAD v0.1.0   # explicit subset
--neighborhoods r2       # explicit subset
--workers 3              # parallel worker slots (default: CPUs//4)
--mem-budget-gb 12       # ceiling the parallel scheduler respects
--timeout 5400           # per-job kill timeout, seconds
--dry-run                # print the job matrix and exit
--prepare-only           # just build the inputs and the version checkouts
```

`--profile full` includes R3, which is the expensive one: the case study runs
R3 in clusters of two targets "due to memory constraints", and this harness
reproduces that clustering exactly.

## Where everything lives

Nothing is written into the repository. The data root is
`~/Documents/pyorps_benchmarks` (override with the `PYORPS_BENCH_ROOT`
environment variable):

```
inputs/     bbox / source / targets geojson, alkis_mvo.gpkg,
            ref_raster.tiff (1.1 GiB), window_cost.npy, window_meta.json
checkouts/  one directory per version, exported with `git archive`
runs/<id>/  manifest.json      what was run, on what hardware
            records.jsonl      one line per job, appended as it finishes
            routes/*.json      full route geometry (WKT) per job
            logs/*.log         stdout/stderr per job
            summary_jobs.csv   every job, every timing
            summary_paths.csv  every route, reported and canonical cost
            REPORT.md          the readable summary
```

`records.jsonl` is appended and flushed after every job, so a run that is
interrupted still leaves everything it had already measured.

## What is measured

| track | what it isolates |
|---|---|
| `raster` | vector → 1 m cost raster, per version |
| `evolution` | each version on **its own default backend**, R0–R3 — what a user actually got at the time |
| `control` | every version pinned to **one common backend** (networkit), R2 — separates algorithmic change from backend change |
| `backends` | HEAD across every installed backend, R2 |
| `baseline` | the NEIS 2021 scikit-image prototype |

The `evolution` and `control` split matters: pyorps changed its default from
`networkit` (v0.1.x) to `cython` (v0.2.1+), so the headline speedup mixes a
backend swap with an architectural change. `control` holds the backend fixed
so the rest is visible on its own.

For v0.1.x the default *is* networkit, so a control job would repeat the
evolution job byte for byte (~200 s each, twice per repeat, twice per exec
mode). Those are skipped; read the v0.1.x `evolution` R2 row as the control
row — the `graph_api` column in `summary_paths.csv` confirms which backend
actually ran.

## The canonical cost model

Versions do not agree on what `total_cost` means — the metric changed over the
project's life, and the 2021 prototype uses a different edge model entirely.
So `score.py` **ignores every reported cost** and re-evaluates each route
geometry under one model, today's pyorps semantics:

```
step cost = (c[from] + c[to] + Σ c[intermediates]) · ‖step‖ / (2 + n_inter)
```

Each engine's self-reported cost is kept beside the canonical one, so where
they disagree that disagreement is itself a result.

Two derived columns are worth reading:

- **`excess_pct`** — how much more the route costs than the cheapest route
  *anyone* found to that bus. This is the accuracy number.
- **`barrier_tunnels`** — steps that pass through a maximum-cost (65535) cell
  while both endpoints avoid it. A route crossing a barrier it never pays for.
  It is the signature of an edge model that ignores intermediate cells, and it
  is what separates `MCP_Geometric` from pyorps on R2 moves.

## The NEIS 2021 baseline

The paper (Hofmann, Franz, Stetz, Hajdu, NEIS 2021) rasterises land registry
data to 1 m, connects each cell to its **R = 2** neighbourhood, and runs
Dijkstra, citing scikit-image. `_worker_neis2021.py` reproduces that with
`skimage.graph.MCP_Geometric` on **the identical search window** pyorps uses —
same cells, same costs, same source and target pixels, exported from a real
`PathFinder` construction rather than reimplemented.

Two differences are preserved on purpose, because they are the substance of
what changed since:

1. **Edge cost.** `MCP_Geometric` charges the mean of the two *endpoints*.
   pyorps charges the mean over *every cell the step crosses*. For R2 knight
   moves these disagree.
2. **Passability.** Because MCP never looks at intermediate cells, an R2 step
   can jump diagonally through a one-cell barrier that pyorps refuses.

### How R = 2 is reached: the original `MyMCP` bypass

`skimage.graph.MCP_Geometric` rejects any offset with a component outside
{-1, 0, 1} (`ValueError: all offset components must be 0, 1, or -1`), so R = 2
is not reachable through its constructor. The original work solved this, and
this harness uses the author's own solution, transcribed from
`my_mcp/_my_mcp.pyx` (2022):

```python
class MyMCP(_mcp.MCP_Geometric):
    def __init__(self, costs, additional_moves):
        offsets = make_offsets(2, True)              # the 8 unit neighbours
        offsets.extend([am for am in additional_moves])   # + 8 knight moves
        self.offsets = np.array(offsets, dtype=OFFSET_D)
        _mcp.MCP.__init__(self, costs, offsets=self.offsets,
                          fully_connected=True, sampling=None)
```

The trick is the last line: the validation lives in `MCP_Geometric.__init__`,
so calling `MCP.__init__` instead skips it while still inheriting
`MCP_Geometric._travel_cost`, the C-level `offset_length * 0.5 * (old + new)`.
`MyMCP` is a plain Python class even inside that `.pyx`, so no compilation is
involved in the trick itself.

Verified against scikit-image 0.26 on the real 38 Mcell window:

| solver | R2 solve time | result |
|---|---:|---|
| `MyMCP` (bypass) | **13.8 s** | — |
| `MCP_Flexible` subclass | 48.2 s | bit-identical: same cost *and* same WKT for all 8 routes |

So the paper's R = 2 **was** reachable at native speed, and the harness uses
that path at every neighbourhood (`mcp_class`, `mcp_native` are recorded). The
`MCP_Flexible` formulation is kept as a fallback and as a startup cross-check:
if the two ever disagree on a 48×48 probe, the job fails loudly rather than
silently measuring a different cost function under the prototype's name.

The R1 and R2 offset **sets are identical** to the ones pyorps uses today
(verified by set comparison), so the comparison isolates the cost model rather
than connectivity. Offsets are taken from pyorps' `get_neighborhood_steps` for
all neighbourhoods, so their *order* may differ from the 2022 construction —
that can only change which of several equal-cost routes is traced back.

Requires `scikit-image`. If it is missing the run continues and the baseline
track is reported as unavailable — never silently dropped.

## Serial vs parallel

Both modes run the same job matrix.

- **serial** — one job at a time. These are the timings that are comparable
  between versions, and the ones the report ranks.
- **parallel** — several jobs at once, admitted by a scheduler that respects
  both a worker-slot count and a memory budget. This measures **throughput on
  this machine**, not per-job latency: jobs contend for cores and memory
  bandwidth, so individual numbers get worse while total wall time improves.

The two are recorded under separate `exec_mode` values and the report never
mixes them. A job whose own memory estimate exceeds the budget still runs — it
just runs alone, so heavy R2/R3 graph-library jobs are never silently dropped.

Defaults are deliberately conservative (`CPUs // 4` workers, 55 % of RAM) so a
run left going overnight does not make the machine unusable.

## Notes on how this interacts with git

The harness only ever **reads** the repository: `git archive`, `git rev-parse`,
`git tag`. It creates no worktree, changes no branch, stages nothing and
commits nothing. Version checkouts are plain directories under the data root.

`HEAD` means the working tree exactly as it is, including uncommitted work —
that is the only way to measure work that is not committed yet.

## Known caveats

- **networkit `addEdges` shim.** networkit ≥ 11 segfaults on uint32 index
  arrays; pyorps only started casting to uint64 later. Without a shim every
  pre-fix version dies with SIGSEGV on today's networkit. The shim is applied
  and recorded as `nk_uint64_shim` in the result, so a shimmed measurement is
  never mistaken for unpatched historical behaviour.
- **One machine, one day.** Every number comes from today's hardware, so this
  measures *the code's* evolution, not what users experienced on the hardware
  of the time.
- **Timing noise.** Repeats exist because this machine has shown run-to-run
  spread of several × on identical code. Read medians, and read the min–max
  column before believing any single ratio.
- **`igraph` / `rustworkx`** are not installed here, so those backends are
  reported as unavailable rather than measured.
- **`networkx` at R2 is refused, not run.** A 38 Mcell window at R2 is ~305 M
  edges; networkx stores those as nested Python dicts, an estimated ~36 GiB on
  a 31 GiB machine. The runner refuses any job whose estimate exceeds 90 % of
  physical RAM and records it as `skipped_insufficient_memory` — it would not
  fail fast, it would swap for hours and take the desktop with it. Lower
  neighbourhoods still run. Override with `--mem-budget-gb` if you disagree
  with the model.
