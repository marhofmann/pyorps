"""Generate path finding modes plot: S2S, S2M, M2M, M2M pairwise on real GIS raster."""

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pyorps import PathFinder

OUTPUT = os.path.join(
    os.path.dirname(__file__), "../source/_static/generated/routing_modes.png"
)
RASTER = os.path.join(
    os.path.dirname(__file__), "../../examples/data/raster/sample_raster.tiff"
)

s1 = (472000, 5593400)
s2 = (471800, 5594200)
t1 = (472800, 5594000)
t2 = (473200, 5593500)

BUFFER = 800
path_colors = ["#e74c3c", "#3498db", "#f39c12", "#9b59b6"]


def plot_raster_bg(ax, pf):
    """Plot raster background with white-to-black cost colormap."""
    raster_data = np.squeeze(pf.raster_handler.data)
    t = pf.raster_handler.window_transform
    extent = [t.c, t.c + raster_data.shape[1] * t.a,
              t.f + raster_data.shape[0] * t.e, t.f]
    display = raster_data.astype(float)
    valid = raster_data[raster_data < 65535]
    norm = mcolors.PowerNorm(gamma=0.3, vmin=valid.min(), vmax=valid.max())
    ax.imshow(display, extent=extent, cmap="Greys", interpolation="nearest", norm=norm)
    return extent


fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# --- S2S ---
pf = PathFinder(dataset_source=RASTER, source_coords=s1, target_coords=t1,
                search_space_buffer_m=BUFFER)
pf.find_route()
ax = axes[0, 0]
plot_raster_bg(ax, pf)
coords = np.array(pf.paths.all[0].path_coords)
ax.plot(coords[:, 0], coords[:, 1], color=path_colors[0], linewidth=2)
ax.plot(*s1, "o", color="#2ecc71", markersize=10, zorder=5)
ax.plot(*t1, "X", color="#e74c3c", markersize=10, zorder=5)
ax.set_title("Single to Single", fontsize=13, fontweight="bold")

# --- S2M ---
pf2 = PathFinder(dataset_source=RASTER, source_coords=s1, target_coords=[t1, t2],
                 search_space_buffer_m=BUFFER)
pf2.find_route()
ax = axes[0, 1]
plot_raster_bg(ax, pf2)
for i, path in enumerate(pf2.paths.all):
    coords = np.array(path.path_coords)
    ax.plot(coords[:, 0], coords[:, 1], color=path_colors[i], linewidth=2)
ax.plot(*s1, "o", color="#2ecc71", markersize=10, zorder=5)
ax.plot(*t1, "X", color="#e74c3c", markersize=10, zorder=5)
ax.plot(*t2, "X", color="#e74c3c", markersize=10, zorder=5)
ax.set_title("Single to Multi", fontsize=13, fontweight="bold")

# --- M2M (all combinations) ---
pf3 = PathFinder(dataset_source=RASTER, source_coords=[s1, s2], target_coords=[t1, t2],
                 search_space_buffer_m=BUFFER)
pf3.find_route(pairwise=False)
ax = axes[1, 0]
plot_raster_bg(ax, pf3)
for i, path in enumerate(pf3.paths.all):
    coords = np.array(path.path_coords)
    ax.plot(coords[:, 0], coords[:, 1], color=path_colors[i], linewidth=2)
ax.plot(*s1, "o", color="#2ecc71", markersize=10, zorder=5)
ax.plot(*s2, "o", color="#2ecc71", markersize=10, zorder=5)
ax.plot(*t1, "X", color="#e74c3c", markersize=10, zorder=5)
ax.plot(*t2, "X", color="#e74c3c", markersize=10, zorder=5)
ax.set_title("Multi to Multi (all combinations)", fontsize=13, fontweight="bold")

# --- M2M (pairwise) ---
pf4 = PathFinder(dataset_source=RASTER, source_coords=[s1, s2], target_coords=[t1, t2],
                 search_space_buffer_m=BUFFER)
pf4.find_route(pairwise=True)
ax = axes[1, 1]
plot_raster_bg(ax, pf4)
for i, path in enumerate(pf4.paths.all):
    coords = np.array(path.path_coords)
    ax.plot(coords[:, 0], coords[:, 1], color=path_colors[i], linewidth=2)
ax.plot(*s1, "o", color="#2ecc71", markersize=10, zorder=5)
ax.plot(*s2, "o", color="#2ecc71", markersize=10, zorder=5)
ax.plot(*t1, "X", color="#e74c3c", markersize=10, zorder=5)
ax.plot(*t2, "X", color="#e74c3c", markersize=10, zorder=5)
ax.set_title("Multi to Multi (pairwise)", fontsize=13, fontweight="bold")

for ax in axes.flat:
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

fig.suptitle("PYORPS Routing Modes", fontsize=15, fontweight="bold")
fig.tight_layout()

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUTPUT}")
