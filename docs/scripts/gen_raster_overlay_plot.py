"""Generate raster overlay comparison: base land use vs modified (with overlays).

Uses the existing raster files from the distribution grid planning example:
- raster_for_distribution_grid_planning.tiff (base land use only)
- modified_raster_for_distribution_grid_planning.tiff (+ water, soil, nature protection)
"""

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import rasterio
from rasterio.windows import from_bounds

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

OUTPUT = os.path.join(
    os.path.dirname(__file__), "../source/_static/generated/raster_overlay.png"
)
BASE_RASTER = os.path.join(
    os.path.dirname(__file__),
    "../../examples/data/raster/raster_for_distribution_grid_planning.tiff",
)
MODIFIED_RASTER = os.path.join(
    os.path.dirname(__file__),
    "../../examples/data/raster/modified_raster_for_distribution_grid_planning.tiff",
)

# Use the constrained example area for a zoomed-in view
minx, miny, maxx, maxy = 473000, 5606000, 475500, 5608000

fig, axes = plt.subplots(1, 2, figsize=(18, 9))

rasters = [
    (BASE_RASTER, "Step 1: Base Land Use\n(23 cost categories)"),
    (MODIFIED_RASTER, "Step 2: + Water Protection, Soil, Nature Reserves\n(299 cost categories)"),
]

for ax, (raster_path, title) in zip(axes, rasters):
    with rasterio.open(raster_path) as ds:
        window = from_bounds(minx, miny, maxx, maxy, ds.transform)
        data = ds.read(1, window=window)
        t = ds.window_transform(window)

    extent = [t.c, t.c + data.shape[1] * t.a,
              t.f + data.shape[0] * t.e, t.f]

    display = data.astype(float)
    valid = data[data < 65535]
    norm = mcolors.PowerNorm(gamma=0.3, vmin=valid.min(), vmax=valid.max())
    im = ax.imshow(display, extent=extent, cmap="Greys", interpolation="nearest",
                   norm=norm)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")

fig.colorbar(im, ax=axes, shrink=0.6, label="Cost Value", pad=0.02)
fig.suptitle("Building a Cost Raster: Layered Data Overlays",
             fontsize=14, fontweight="bold")
fig.tight_layout()

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUTPUT}")
