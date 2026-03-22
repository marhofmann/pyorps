"""Generate neighborhood sweep animation: R0 through R5 paths on same raster."""

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pyorps import PathFinder

OUTPUT = os.path.join(
    os.path.dirname(__file__), "../source/_static/generated/neighborhood_sweep.gif"
)
RASTER = os.path.join(
    os.path.dirname(__file__), "../../examples/data/raster/sample_raster.tiff"
)

source = (472000, 5593400)
target = (472800, 5594000)
BUFFER = 600

neighborhoods = [
    ("r0", "R0 — 4 neighbors"),
    ("r1", "R1 — 8 neighbors"),
    ("r2", "R2 — 16 neighbors (default)"),
    ("r3", "R3 — 32 neighbors"),
]

# Pre-compute all paths
paths_data = []
raster_data = None
extent = None

for nb, label in neighborhoods:
    pf = PathFinder(
        dataset_source=RASTER,
        source_coords=source,
        target_coords=target,
        search_space_buffer_m=BUFFER,
        neighborhood_str=nb,
    )
    pf.find_route()

    if raster_data is None:
        raster_data = np.squeeze(pf.raster_handler.data)
        extent = [
            pf.raster_handler.window_transform.c,
            pf.raster_handler.window_transform.c
            + raster_data.shape[1] * pf.raster_handler.window_transform.a,
            pf.raster_handler.window_transform.f
            + raster_data.shape[0] * pf.raster_handler.window_transform.e,
            pf.raster_handler.window_transform.f,
        ]

    path = pf.paths.all[0]
    coords = np.array(path.path_coords)
    cost = path.total_cost
    paths_data.append((coords, label, cost))

display = raster_data.astype(float)
valid = raster_data[raster_data < 65535]
vmin_val, vmax_val = valid.min(), valid.max()
colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

# Create animation
fig, ax = plt.subplots(figsize=(10, 7))


def update(frame):
    ax.clear()
    norm = mcolors.PowerNorm(gamma=0.3, vmin=vmin_val, vmax=vmax_val)
    ax.imshow(display, extent=extent, cmap="Greys", interpolation="nearest", norm=norm)

    idx = frame % len(paths_data)
    coords, label, cost = paths_data[idx]
    color = colors[idx]

    ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=3)
    ax.plot(*source, "o", color="#2ecc71", markersize=12, zorder=5)
    ax.plot(*target, "X", color="#3498db", markersize=12, zorder=5)

    cost_str = f" | Cost: {cost:,.0f}" if cost else ""
    ax.set_title(f"{label}{cost_str}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")


# Each neighborhood shown for 2 seconds (12 frames at 6 fps)
frames_per_nb = 12
total_frames = len(paths_data) * frames_per_nb

ani = animation.FuncAnimation(
    fig, lambda f: update(f // frames_per_nb),
    frames=total_frames, interval=167,
)

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
ani.save(OUTPUT, writer="pillow", fps=6, dpi=100)
plt.close(fig)
print(f"Saved: {OUTPUT}")
