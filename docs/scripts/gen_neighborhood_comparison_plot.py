"""Generate neighborhood comparison: all paths overlaid in a single plot with legend."""

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pyorps import PathFinder

OUTPUT = os.path.join(
    os.path.dirname(__file__),
    "../source/_static/generated/neighborhood_comparison.png",
)
RASTER = os.path.join(
    os.path.dirname(__file__), "../../examples/data/raster/sample_raster.tiff"
)

source = (472000, 5593400)
target = (472800, 5594000)
BUFFER = 600

neighborhoods = [
    ("r0", "R0 — 4 neighbors",  "#e74c3c"),
    ("r1", "R1 — 8 neighbors",  "#f39c12"),
    ("r2", "R2 — 16 neighbors", "#2ecc71"),
    ("r3", "R3 — 32 neighbors", "#3498db"),
]

# Compute all paths first, reuse raster from first run
raster_bg = None
extent = None
path_data = []

for nb, label, color in neighborhoods:
    pf = PathFinder(
        dataset_source=RASTER,
        source_coords=source,
        target_coords=target,
        search_space_buffer_m=BUFFER,
        neighborhood_str=nb,
    )
    pf.find_route()

    if raster_bg is None:
        raster_bg = np.squeeze(pf.raster_handler.data)
        t = pf.raster_handler.window_transform
        extent = [t.c, t.c + raster_bg.shape[1] * t.a,
                  t.f + raster_bg.shape[0] * t.e, t.f]

    path = pf.paths.all[0]
    coords = np.array(path.path_coords)
    cost = path.total_cost
    path_data.append((coords, label, color, cost))

# Single figure with all paths overlaid
fig, ax = plt.subplots(figsize=(12, 9))

display = raster_bg.astype(float)
valid = raster_bg[raster_bg < 65535]
norm = mcolors.PowerNorm(gamma=0.3, vmin=valid.min(), vmax=valid.max())
ax.imshow(display, extent=extent, cmap="Greys", interpolation="nearest", norm=norm)

# Plot paths from coarsest to finest so the finer ones draw on top
for coords, label, color, cost in path_data:
    cost_str = f" (cost: {cost:,.0f})" if cost else ""
    ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=2.5,
            label=f"{label}{cost_str}", zorder=3)

ax.plot(*source, "o", color="white", markersize=14, zorder=5,
        markeredgecolor="black", markeredgewidth=1.5, label="Source")
ax.plot(*target, "X", color="white", markersize=14, zorder=5,
        markeredgecolor="black", markeredgewidth=1.5, label="Target")

ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
ax.set_title("Neighborhood Comparison: Higher Connectivity → Smoother Paths",
             fontsize=14, fontweight="bold")
ax.legend(loc="upper left", fontsize=11, framealpha=0.9)

fig.tight_layout()

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUTPUT}")
