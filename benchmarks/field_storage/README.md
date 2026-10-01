# Field storage probes and benchmark

Evidence behind `docs/superpowers/plans/2026-09-20-generalized-free-siting-facility-chains.md`
sections 2 and 3.5, which are now **implemented** in
`pyorps/io/field_codec.py` and wired into `CostField.save(codec=...)` /
`SavedCostField`.

**Start with `bench_field_codec.py`** — it measures the shipped codec. The
`probe_*` scripts below are the exploration that produced it, kept because
three of them are negative results worth not repeating.

## The shipped codec, measured on `fields_min5m/field_PCC0`

41.8 M cells, 0 .. 13.4 MEUR, 14.8 % unreachable. Baseline is **deflated
float32** (121.2 MB), not the uncompressed save (167.2 MB) — quoting against
the raw file is what turned a 6.2x codec into a "9.1x" headline.

| guaranteed understatement | size | vs deflated float32 | worst observed |
|---|---|---|---|
| 100 EUR | 19.9 MB | 0.16x | 99.98 EUR |
| 10 EUR | 25.3 MB | 0.21x | 9.99 EUR |
| **1 EUR** | **31.4 MB** | **0.26x** | 0.996 EUR |
| 0.1 EUR | 35.2 MB | 0.29x | 0.099 EUR |

The 1 EUR row reproduces the plan's hand-rolled 31.9 MB. The bound holds
one-sidedly at every level, the reachability mask round-trips exactly, and
decode is **3.0x slower** than reading deflated float32 (2.4 s vs 0.8 s) —
the codec buys size, not speed, and the earlier "2.7x faster" claim came
from timing `zlib.decompress` alone.

Error is identical at every tile size (64 to 512), which is the fixed
quantum working as advertised: there is no outlier tile to promote, so that
lever — real against a RANGE-based codec — is deliberately not implemented.

The format also carries what the plan's review said it must: the
reachability mask (without it unreachable cells decode to a finite filler
and every bound over them is false), float64 decode nudged one ulp outward
(Neumaier & Shcherbina safe bounds, because `k * quantum` is itself a
rounded product), tile-major ordering, and per-tile-row framing so a
partial read is possible. Dispatch is on a `codec` metadata key, never on
`_FIELD_FORMAT_VERSION`, which is a strict-equality check that would reject
every field already on disk. `CostFieldSet` refuses a lossy codec outright:
it picks resident-vs-spilled from available memory, so a lossy spill would
make precision machine-dependent.

One thing the implementation added that the probes did not have: the
residual width is chosen **per frame** from what that frame needs (1, 2, 4
or 8 bytes). A fixed 32-bit residual overflows on a field whose range
divided by the quantum exceeds 2^32, which the probe would have hit at
1 EUR on a 1e13-scale field.

## The probes

Each script is standalone and prints a table; none of them writes to the
cache it reads.

They read the CIRED 2026 field cache under
`TopoMILP/data/cired2026/raster/`, which lives outside this repo. Point the
`FIELD` / `ROOT` constants elsewhere to run them on another study.

| script | answers |
|---|---|
| `probe_encodings.py` | How small can one field get, and what does each encoding cost to decode? Byte shuffle, tile quantisation, delta coding, LZMA ceiling. |
| `probe_bit_depth_sweep.py` | 8 / 12 / 16 bit against every field kind in the cache, with the worst one-sided error each one admits, and the whole-cache total. |
| `probe_large_field_streamed.py` | Does the ratio hold on the 240 M-cell field that dominates the cache? Streamed in row stripes to keep the peak near 1 GB. |
| `probe_descent_reconstruction.py` | Can the predecessor plane be dropped and the route re-derived from `dist` alone? Builds a real field from a 1600² window, then compares 300 descent-reconstructed routes against the stored plane. |

Two things the numbers depend on, both easy to get wrong:

- **Unreachable cells arrive as `+inf`, and `nanmin`/`nanmax` do not filter
  those.** One `inf` blows its tile's span to infinity and destroys the scale
  for every other cell in the tile. The first run of `probe_encodings.py` did
  exactly this and reported a 2.9 M EUR quantisation error that was pure
  artefact.
- **The quantiser rounds DOWN**, so a decoded value is always ≤ the true one
  and a screening bound built on it stays a bound. Rounding to nearest would
  be smaller on average and unsound.

`probe_descent_reconstruction.py` needs the correct descent rule: plain
`argmin over u of (dist[u] + w(u,v))`, which by Dijkstra optimality equals
`dist[v]`. Minimising the *residual* `|dist[u] + w - dist[v]|` instead looks
equivalent and is not — it is biased under quantisation and needs a tolerance
that has no principled value.

## Negative results — three levers that do NOT work

Added by the plan's self-review. Each of these looks like it should shrink the
16-bit error floor (45.89 EUR per lookup on `fields_min5m`). None does. They
are kept so nobody spends another afternoon on them.

| script | lever | result |
|---|---|---|
| `probe_tile_size_and_framing.py` | smaller tiles | 45.9 → 43.3 EUR for **+44 % size** (256 → 32). The error is set by local cost heterogeneity, not by tile extent. |
| `probe_outlier_tiles.py` | store the worst tiles exactly | **RETRACTED 2026-09-20 — this probe has an accounting bug and its conclusion was wrong.** `extra = k * TILE * TILE * 4` charges promoted tiles as UNCOMPRESSED float32 while the 121 MB baseline is compressed, and it never removes those tiles from the quantised stream. Corrected: 512 promoted = 98.7 MB at 7.27 EUR (**below** the 121.2 MB baseline), and 128 promoted = 42.0 MB at 10.91 EUR — a 4.2× error reduction for +20 MB. The lever WORKS. Only the distribution statistics in this row were right (median 9.04, p99 25.39, max 45.89, 99.7 % of live tiles > 1 EUR). |
| `probe_residual_trend.py` | quantise `dist − c·euclid` instead of `dist` | **bigger and no better**: 32.3 MB (+19 %) at 43.4 EUR. The per-tile `lo` already removes the local offset, and subtracting a radial cone from a field that is not radially symmetric adds structure. |

**The "intrinsic floor" conclusion is itself retracted.** It is a property of
quantising the VALUE onto a fixed bit budget, which is what all three levers
above attack. Fixing the QUANTUM instead (bin width 2·eb, as SZ-class
compressors do) makes the error a knob independent of dynamic range entirely:
measured on the same field, a guaranteed 1 EUR bound costs 31.9 MB and a
guaranteed 100 EUR bound costs 18.9 MB — 30 % SMALLER than the tiled codec at a
comparable error. See `probe_fixed_quantum.py`.

Two further corrections to the numbers above: delta-coding along full raster
rows crosses ~35 tiles with different `lo`/`scale`, so every tile boundary
emits garbage residuals — tile-major ordering gives −19.8 % for free (27.1 →
21.7 MB). And decode is **slower**, not faster: timing only `zlib.decompress`
omits unshuffle, cumulative-sum and dequantise, which together make the full
read ~1.6× slower than deflated float32 (2.93 s vs 1.88 s), not 2.7× faster.

Quantised fields remain a screening tier — but **the role-based rule stated
here originally was wrong**, and `pyorps.io.field_codec` implements the
corrected one. "Safe for lower bounds, unsafe for incumbents" is not the
distinction: `certify_k_truncation` puts a field value on the *left* of a
must-not-exceed test and SUBTRACTS it, so understating makes that certificate
falsely pass even though the value is playing the part of a bound. The rule is
**directional**: any quantity that must not be UNDERSTATED needs the upper end
of the interval, wherever it sits in the inequality, and any quantity that must
not be OVERSTATED needs the lower end. `DecodedField.choose("lower"|"upper")`
makes the caller say which, and refuses a third option — "the value" is exactly
the ambiguity the type exists to remove.

`probe_tile_size_and_framing.py` also answers whether the stream can be decoded
lazily. As first written it could not: the delta ran across full rows and the
whole plane was one deflate stream. Framing each tile-row independently costs
**0.0–0.4 %**, so random access is essentially free — but it has to be built
in, not assumed.
