# ArcGIS Pro Distance Accumulation — comparison procedure

Manual procedure for the ArcGIS side of the eikonal accuracy comparison
(plan `2026-08-06-eikonal-fim-gpu-backend.md` §6.4). For the four
analytic cases the referee is the **analytic closed form**, not either
solver. The two §6.2 cases without a closed form (`smooth_random`,
`real_raster`) are a **cross-check**: ArcGIS is compared against our
FIM field — agreement within discretization error is the expected
outcome; a systematic gap flags an algorithmic difference, not an error
of either side. ArcGIS runs are manual (license); the pyorps side is
fully scripted.

## Prerequisites

- ArcGIS Pro ≥ 2.6 (Distance Accumulation has been the eikonal-based
  solver since 2020) with the **Spatial Analyst** extension.
- Generated inputs (run once):

  ```
  .venv/Scripts/python.exe benchmarks/arcgis_comparison/generate_cases.py
  ```

  This writes per case (`uniform`, `radial`, `snell`, `barrier`,
  `smooth_random`, `real_raster`):
  - `cases/<name>_cost.tif` — float32 cost raster (cost per metre,
    1 m cells, EPSG:32632; NoData = impassable barrier),
  - `cases/<name>_source.gpkg` — the source point (layer `source`),
  - `fim/<name>_fim.tif` — our block-FIM accumulation field.

  Case-specific notes:
  - `smooth_random` — deterministic Gaussian-blurred noise (σ=16,
    fixed seed), 400²; no barriers.
  - `real_raster` — the repository's real land-use cost raster
    (`examples/data/raster/small_raster.tiff`, 3278×5364, 10 cost
    classes, 51 % NoData), EPSG:25832 at its true location. The
    declared grid is idealized to exact 1 m square cells (the raw
    cells are 0.99999 × 1.00017 m; data unchanged) so the map-unit
    accumulation is exactly comparable with the cell-unit T field.
    Expect a noticeably longer ArcGIS run than the synthetic cases.

## Per-case steps in ArcGIS Pro

1. **Insert ▸ New Map**, then add `cases/<name>_cost.tif` and the
   `source` layer from `cases/<name>_source.gpkg`.
2. Open **Geoprocessing ▸ Distance Accumulation** (Spatial Analyst).
3. Set exactly:
   - *Input raster or feature source data*: the `source` point layer.
   - *Input cost raster*: `<name>_cost.tif`.
   - **Leave every other parameter empty/default** — no surface raster,
     no vertical/horizontal factors, no source characteristics. The
     comparison targets the plain isotropic cost accumulation.
   - *Output distance accumulation raster*: `<name>_arcgis`.
4. Run, then **export the result**: right-click the output layer ▸
   Data ▸ Export Raster — format TIFF, 32-bit float, output location
   `benchmarks/arcgis_comparison/results_arcgis/`, name
   `<name>_arcgis.tif`. Do not change cell size, extent or CRS.

Notes:
- Environments must stay untouched (snap raster/extent default to the
  input cost raster automatically). If prompted about parallel
  processing, defaults are fine — the tool is deterministic.
- The cost rasters carry cost **per metre** on 1 m cells, so the ArcGIS
  accumulation values are directly comparable with the analytic fields
  and `fim/<name>_fim.tif` — no unit conversion.
- NoData cells are barriers in Distance Accumulation — this matches the
  pyorps exclusion semantics (65535 / +inf).

## Evaluate

```
.venv/Scripts/python.exe benchmarks/arcgis_comparison/compare_results.py
```

The script computes the same error metrics (vs the analytic reference)
for every `results_arcgis/<name>_arcgis.tif` it finds, alongside the FIM
column, and writes `comparison.json` + a markdown table. Cases without a
dropped ArcGIS raster are reported as *pending* — the pyorps side of the
table is valid without them. For `smooth_random`/`real_raster` the
metrics are the ArcGIS-vs-FIM relative difference (plus a
coverage-mismatch count: cells reached by exactly one solver would
expose differing barrier semantics); the FIM row shows *reference*.

Paste the resulting table into `benchmarks/EIKONAL_FINDINGS.md` §"ArcGIS
comparison" when the ArcGIS column is filled.
