"""Generate search space plot: 3 panels with different buffer sizes."""

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pyorps import PathFinder

OUTPUT = os.path.join(
    os.path.dirname(__file__), "../source/_static/generated/search_space_buffers.png"
)
RASTER = os.path.join(
    os.path.dirname(__file__), "../../examples/data/raster/sample_raster.tiff"
)

source = (472000, 5593400)
target = (472800, 5594000)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
buffers = [200, 600, 1200]

for ax, buf in zip(axes, buffers):
    pf = PathFinder(
        dataset_source=RASTER,
        source_coords=source,
        target_coords=target,
        search_space_buffer_m=buf,
    )
    pf.find_route()

    raster_data = np.squeeze(pf.raster_handler.data)
    t = pf.raster_handler.window_transform
    extent = [t.c, t.c + raster_data.shape[1] * t.a,
              t.f + raster_data.shape[0] * t.e, t.f]

    display = raster_data.astype(float)
    valid = raster_data[raster_data < 65535]
    norm = mcolors.PowerNorm(gamma=0.3, vmin=valid.min(), vmax=valid.max())
    ax.imshow(display, extent=extent, cmap="Greys", interpolation="nearest", norm=norm)

    path = pf.paths.all[0]
    coords = np.array(path.path_coords)
    ax.plot(coords[:, 0], coords[:, 1], color="#e74c3c", linewidth=2)
    ax.plot(*source, "o", color="#2ecc71", markersize=8, zorder=5)
    ax.plot(*target, "X", color="#3498db", markersize=8, zorder=5)
    cells = raster_data.shape[0] * raster_data.shape[1]
    ax.set_title(f"Buffer = {buf}m\n({cells:,} cells)")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

fig.suptitle("Search Space Buffer: Smaller = Faster", fontsize=14, fontweight="bold")
fig.tight_layout()

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUTPUT}")
