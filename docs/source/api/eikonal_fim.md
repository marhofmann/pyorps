---
title: "Eikonal FIM Backend"
summary: "GPU eikonal solver raster_fim and the raw .gpur raster format."
status: unreleased
since: "unreleased"
available_in: source
module: "pyorps.graph.api.raster_fim_api"
api:
  - pyorps.graph.api.raster_fim_api.RasterFIMAPI
  - pyorps.io.gpu_raster.save_gpu_raster
  - pyorps.io.gpu_raster.load_gpu_raster
---
# Eikonal FIM Backend

`graph_api="raster_fim"` solves the continuous eikonal equation `|grad T| = c(x)` on the cost raster with a GPU fast-iterative method (CuPy kernels) instead of searching a graph. Routes are traced by steepest descent on the arrival-time field `T`. One solve from a source serves all of its targets, and the result has no directional bias from a discrete neighbourhood.

(eikonal-usage)=
## Usage

```python
from pyorps import PathFinder

pf = PathFinder(
    dataset_source="cost_raster.tiff",
    source_coords=source,
    target_coords=target,
    graph_api="raster_fim",
)
result = pf.find_route(algorithm="fim")   # "fim", "eikonal" or "dijkstra" (alias)
```

The backend class is `RasterFIMAPI` in `pyorps.graph.api.raster_fim_api`. It needs an NVIDIA GPU and CuPy (`pip install pyorps[gpu]`); both are optional for PYORPS as a whole.

(eikonal-semantics)=
## What to expect

- The route cost is `T[target]`, a continuous value. It is never bit-equal to the Dijkstra cost on the same raster, and the cost that `PathFinder` recomputes over the returned cell path differs slightly.
- `steps` (the neighbourhood) is accepted and ignored.
- `max_gradient_pct` is only screened by a solve, check, mask and re-solve loop. A returned route satisfies the limit; a reported failure does not prove that no route exists. Use a discrete backend as the authority.
- Any algorithm name other than `"fim"`, `"eikonal"` or `"dijkstra"` raises `AlgorithmNotImplementedError`.
- Keep GPU chunks below about 2 GB and leave headroom for other GPU work.

See {doc}`../concepts/known_limitations` for the cost-unit and infinity conventions.

(eikonal-gpur)=
## The `.gpur` raw raster format

`pyorps.io.gpu_raster` converts a GeoTIFF into an uncompressed row-major binary file (`.gpur`) plus a JSON sidecar (`.gpur.json`) holding shape, dtype, CRS, transform and nodata. Loading is a raw copy into GPU (or NumPy) memory.

```python
from pyorps.io.gpu_raster import save_gpu_raster, load_gpu_raster

meta = save_gpu_raster("cost_raster.tiff", "cost_raster.gpur", band_index=0)
raster, metadata = load_gpu_raster("cost_raster.gpur", device=True)
```

`load_gpu_raster` needs the sidecar, accepts only `uint16`, `float32` and `float64` rasters, and raises `ValueError` if the file size does not match the sidecar. Both backends `raster_gpu` and `raster_fim` can use it; see {doc}`graph_backends`.
