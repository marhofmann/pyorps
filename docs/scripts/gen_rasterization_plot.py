"""Generate rasterization pipeline plot: cost raster with colorbar."""

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pyorps import PathFinder

OUTPUT = os.path.join(
    os.path.dirname(__file__), "../source/_static/generated/rasterization_pipeline.png"
)
RASTER = os.path.join(
    os.path.dirname(__file__), "../../examples/data/raster/sample_raster.tiff"
)

source = (472000, 5593400)
target = (472800, 5594000)

pf = PathFinder(
    dataset_source=RASTER,
    source_coords=source,
    target_coords=target,
    search_space_buffer_m=800,
)

raster_data = np.squeeze(pf.raster_handler.data)
t = pf.raster_handler.window_transform
extent = [t.c, t.c + raster_data.shape[1] * t.a,
          t.f + raster_data.shape[0] * t.e, t.f]

fig, ax = plt.subplots(figsize=(10, 8))
display = raster_data.astype(float)
valid = raster_data[raster_data < 65535]
norm = mcolors.PowerNorm(gamma=0.3, vmin=valid.min(), vmax=valid.max())
im = ax.imshow(display, extent=extent, cmap="Greys", interpolation="nearest", norm=norm)
cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Cell Cost Value")
ax.plot(*source, "o", color="#2ecc71", markersize=10, zorder=5, label="Source")
ax.plot(*target, "X", color="#3498db", markersize=10, zorder=5, label="Target")
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
ax.set_title("Cost Raster: Cell Values Drive Routing Decisions")
ax.legend(loc="upper left")
fig.tight_layout()

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUTPUT}")
