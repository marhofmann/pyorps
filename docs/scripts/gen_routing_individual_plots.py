"""Generate individual routing mode plots: one image per mode."""

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pyorps import PathFinder

OUT_DIR = os.path.join(os.path.dirname(__file__), "../source/_static/generated")
RASTER = os.path.join(
    os.path.dirname(__file__), "../../examples/data/raster/sample_raster.tiff"
)

s1 = (472000, 5593400)
s2 = (471800, 5594200)
t1 = (472800, 5594000)
t2 = (473200, 5593500)
BUFFER = 800
path_colors = ["#e74c3c", "#3498db", "#f39c12", "#9b59b6"]


def plot_single(pf, title, fname, sources, targets):
    """Plot a single routing result with Greys PowerNorm background."""
    fig, ax = plt.subplots(figsize=(12, 9))
    raster_data = np.squeeze(pf.raster_handler.data)
    t = pf.raster_handler.window_transform
    extent = [t.c, t.c + raster_data.shape[1] * t.a,
              t.f + raster_data.shape[0] * t.e, t.f]
    display = raster_data.astype(float)
    valid = raster_data[raster_data < 65535]
    norm = mcolors.PowerNorm(gamma=0.3, vmin=valid.min(), vmax=valid.max())
    ax.imshow(display, extent=extent, cmap="Greys", interpolation="nearest", norm=norm)

    for i, path in enumerate(pf.paths.all):
        coords = np.array(path.path_coords)
        ax.plot(coords[:, 0], coords[:, 1], color=path_colors[i % len(path_colors)],
                linewidth=2.5, zorder=3)

    for src in sources:
        ax.plot(*src, "o", color="#2ecc71", markersize=12, zorder=5)
    for tgt in targets:
        ax.plot(*tgt, "X", color="#e74c3c", markersize=12, zorder=5)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, fname)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


os.makedirs(OUT_DIR, exist_ok=True)

# S2S
pf = PathFinder(dataset_source=RASTER, source_coords=s1, target_coords=t1,
                search_space_buffer_m=BUFFER)
pf.find_route()
plot_single(pf, "Single to Single", "route_s2s.png", [s1], [t1])

# S2M
pf2 = PathFinder(dataset_source=RASTER, source_coords=s1, target_coords=[t1, t2],
                 search_space_buffer_m=BUFFER)
pf2.find_route()
plot_single(pf2, "Single to Multi", "route_s2m.png", [s1], [t1, t2])

# M2M all combinations
pf3 = PathFinder(dataset_source=RASTER, source_coords=[s1, s2], target_coords=[t1, t2],
                 search_space_buffer_m=BUFFER)
pf3.find_route(pairwise=False)
plot_single(pf3, "Multi to Multi (all combinations)", "route_m2m.png", [s1, s2], [t1, t2])

# M2M pairwise
pf4 = PathFinder(dataset_source=RASTER, source_coords=[s1, s2], target_coords=[t1, t2],
                 search_space_buffer_m=BUFFER)
pf4.find_route(pairwise=True)
plot_single(pf4, "Multi to Multi (pairwise)", "route_pairwise.png", [s1, s2], [t1, t2])

print("All routing mode plots generated.")
