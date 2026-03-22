"""Generate Dijkstra wavefront expansion animation on real cost raster.

Uses the same source/target as the S2S routing example on sample_raster.tiff.
Runs a manual Dijkstra on the windowed raster to record the expansion frontier.
"""

import os
import sys
import heapq

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pyorps import PathFinder

OUTPUT = os.path.join(
    os.path.dirname(__file__), "../source/_static/generated/dijkstra_wavefront.gif"
)
RASTER = os.path.join(
    os.path.dirname(__file__), "../../examples/data/raster/sample_raster.tiff"
)

source = (472000, 5593400)
target = (472800, 5594000)
BUFFER = 600

# Load raster via PathFinder to get the windowed data + coordinate transforms
pf = PathFinder(
    dataset_source=RASTER,
    source_coords=source,
    target_coords=target,
    search_space_buffer_m=BUFFER,
)

raster = np.squeeze(pf.raster_handler.data).astype(np.float64)
rows, cols = raster.shape
t = pf.raster_handler.window_transform
extent = [t.c, t.c + cols * t.a, t.f + rows * t.e, t.f]

# Convert source/target to pixel indices
src_idx = pf.raster_handler.coords_to_indices([source])
tgt_idx = pf.raster_handler.coords_to_indices([target])
sr, sc = int(src_idx[0][0]), int(src_idx[0][1])
tr, tc = int(tgt_idx[0][0]), int(tgt_idx[0][1])

# Run Dijkstra manually, recording visited frontier at intervals
INF = float("inf")
dist = np.full((rows, cols), INF)
dist[sr, sc] = 0
visited = np.zeros((rows, cols), dtype=bool)
prev = np.full((rows, cols, 2), -1, dtype=int)

heap = [(0.0, sr, sc)]
step = 0
RECORD_INTERVAL = max(1, (rows * cols) // 80)  # ~80 frames

snapshots = []
neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
sqrt2 = np.sqrt(2)

while heap:
    d, r, c = heapq.heappop(heap)
    if visited[r, c]:
        continue
    visited[r, c] = True
    step += 1

    if step % RECORD_INTERVAL == 0:
        snapshots.append(visited.copy())

    if (r, c) == (tr, tc):
        break

    for dr, dc in neighbors:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
            cell_cost = raster[nr, nc]
            if cell_cost >= 65535:
                continue
            edge_len = sqrt2 if (dr != 0 and dc != 0) else 1.0
            nd = d + cell_cost * edge_len
            if nd < dist[nr, nc]:
                dist[nr, nc] = nd
                prev[nr, nc] = [r, c]
                heapq.heappush(heap, (nd, nr, nc))

snapshots.append(visited.copy())

# Trace optimal path
path = []
r, c = tr, tc
while (r, c) != (sr, sc) and prev[r, c, 0] != -1:
    path.append((r, c))
    r, c = int(prev[r, c, 0]), int(prev[r, c, 1])
path.append((sr, sc))
path.reverse()
path_arr = np.array(path)

# Convert path pixel coords to map coords
path_x = t.c + (path_arr[:, 1] + 0.5) * t.a
path_y = t.f + (path_arr[:, 0] + 0.5) * t.e

# Subsample to ~40 frames
max_frames = 40
if len(snapshots) > max_frames:
    indices = np.linspace(0, len(snapshots) - 1, max_frames, dtype=int)
    snapshots = [snapshots[i] for i in indices]

# Prepare raster background
display = raster.copy()
valid = raster[raster < 65535]
norm = mcolors.PowerNorm(gamma=0.3, vmin=valid.min(), vmax=valid.max())

# Create animation
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title("Dijkstra's Algorithm: Wavefront Expansion", fontsize=13, fontweight="bold")

# Raster background
ax.imshow(display, extent=extent, cmap="Greys", interpolation="nearest",
          norm=norm, alpha=0.5)

# Frontier overlay
frontier_img = ax.imshow(
    np.zeros((rows, cols, 4)), extent=extent, interpolation="nearest",
)

# Source/target markers
ax.plot(*source, "o", color="#2ecc71", markersize=14, zorder=10)
ax.plot(*target, "X", color="#3498db", markersize=14, zorder=10)
path_line, = ax.plot([], [], color="#e74c3c", linewidth=3, zorder=9)
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")

progress_text = ax.text(
    0.02, 0.98, "", transform=ax.transAxes, fontsize=11,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
)


def update(frame):
    if frame < len(snapshots):
        frontier = snapshots[frame]
        # Create RGBA overlay: visited = semi-transparent blue
        rgba = np.zeros((rows, cols, 4))
        rgba[frontier, 0] = 0.2   # R
        rgba[frontier, 1] = 0.4   # G
        rgba[frontier, 2] = 0.9   # B
        rgba[frontier, 3] = 0.45  # A
        frontier_img.set_data(rgba)
        pct = frontier.sum() / (rows * cols) * 100
        progress_text.set_text(f"Explored: {pct:.0f}%")
        path_line.set_data([], [])
    else:
        # Final frames: show path
        frontier = snapshots[-1]
        rgba = np.zeros((rows, cols, 4))
        rgba[frontier, 0] = 0.2
        rgba[frontier, 1] = 0.4
        rgba[frontier, 2] = 0.9
        rgba[frontier, 3] = 0.35
        frontier_img.set_data(rgba)
        path_line.set_data(path_x, path_y)
        progress_text.set_text(f"Optimal path found!\n{len(path_arr)} cells")
    return frontier_img, path_line, progress_text


total_frames = len(snapshots) + 10  # Extra frames to hold the final path
ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=200, blit=True)

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
fig.tight_layout()
ani.save(OUTPUT, writer="pillow", fps=5, dpi=100)
plt.close(fig)
print(f"Saved: {OUTPUT}")
