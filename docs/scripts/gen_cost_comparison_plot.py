"""Generate cost comparison plot: same raster, two neighborhoods showing path quality."""

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pyorps import PathFinder

OUTPUT = os.path.join(
    os.path.dirname(__file__), "../source/_static/generated/cost_comparison.png"
)
RASTER = os.path.join(
    os.path.dirname(__file__), "../../examples/data/raster/sample_raster.tiff"
)

source = (472000, 5593400)
target = (472800, 5594000)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

neighborhoods = ["r1", "r3"]
titles = ["Neighborhood R1 (8 connections)", "Neighborhood R3 (32 connections)"]

for ax, nb, title in zip(axes, neighborhoods, titles):
    pf = PathFinder(
        dataset_source=RASTER,
        source_coords=source,
        target_coords=target,
        search_space_buffer_m=600,
        neighborhood_str=nb,
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
    ax.plot(coords[:, 0], coords[:, 1], color="#e74c3c", linewidth=2.5)
    ax.plot(*source, "o", color="#2ecc71", markersize=10, zorder=5)
    ax.plot(*target, "X", color="#3498db", markersize=10, zorder=5)
    ax.set_title(title)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

fig.suptitle("Path Quality vs. Neighborhood Connectivity", fontsize=14, fontweight="bold")
fig.tight_layout()

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUTPUT}")
