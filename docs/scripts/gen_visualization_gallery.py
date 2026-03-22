"""Generate visualization gallery: 4 different plot styling options."""

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pyorps import PathFinder

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "../source/_static/generated"
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
    search_space_buffer_m=600,
)
pf.find_route()

os.makedirs(OUTPUT_DIR, exist_ok=True)

raster_data = np.squeeze(pf.raster_handler.data)
extent = [
    pf.raster_handler.window_transform.c,
    pf.raster_handler.window_transform.c
    + raster_data.shape[1] * pf.raster_handler.window_transform.a,
    pf.raster_handler.window_transform.f
    + raster_data.shape[0] * pf.raster_handler.window_transform.e,
    pf.raster_handler.window_transform.f,
]
display = raster_data.astype(float)
valid = raster_data[raster_data < 65535]
vmin_val, vmax_val = valid.min(), valid.max()
path = pf.paths.all[0]
coords = np.array(path.path_coords)

styles = [
    {
        "cmap": "Greys",
        "path_color": "#e74c3c",
        "src_color": "#2ecc71",
        "tgt_color": "#3498db",
        "title": "Grayscale (Default)",
        "fname": "viz_gallery_1.png",
    },
    {
        "cmap": "RdYlGn_r",
        "path_color": "#2c3e50",
        "src_color": "#e67e22",
        "tgt_color": "#8e44ad",
        "title": "Cost Heatmap Style",
        "fname": "viz_gallery_2.png",
    },
    {
        "cmap": "Greys",
        "path_color": "#e74c3c",
        "src_color": "#27ae60",
        "tgt_color": "#2980b9",
        "title": "Grayscale Background",
        "fname": "viz_gallery_3.png",
    },
    {
        "cmap": "viridis",
        "path_color": "#ffffff",
        "src_color": "#ff6b6b",
        "tgt_color": "#ffd93d",
        "title": "Viridis + White Path",
        "fname": "viz_gallery_4.png",
    },
]

for style in styles:
    fig, ax = plt.subplots(figsize=(10, 7))
    norm = mcolors.PowerNorm(gamma=0.3, vmin=vmin_val, vmax=vmax_val)
    ax.imshow(display, extent=extent, cmap=style["cmap"], interpolation="nearest", norm=norm)
    ax.plot(coords[:, 0], coords[:, 1], color=style["path_color"], linewidth=2.5)
    ax.plot(*source, "o", color=style["src_color"], markersize=12, zorder=5)
    ax.plot(*target, "X", color=style["tgt_color"], markersize=12, zorder=5)
    ax.set_title(style["title"], fontsize=13, fontweight="bold")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, style["fname"])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
