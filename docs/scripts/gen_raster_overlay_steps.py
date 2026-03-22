"""Generate individual raster overlay step plots from existing raster files."""

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import rasterio
from rasterio.windows import from_bounds

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

OUT_DIR = os.path.join(os.path.dirname(__file__), "../source/_static/generated")
BASE_RASTER = os.path.join(
    os.path.dirname(__file__),
    "../../examples/data/raster/raster_for_distribution_grid_planning.tiff",
)
MODIFIED_RASTER = os.path.join(
    os.path.dirname(__file__),
    "../../examples/data/raster/modified_raster_for_distribution_grid_planning.tiff",
)

# Zoomed-in view around the constrained example area
minx, miny, maxx, maxy = 472500, 5605500, 476000, 5608500


def plot_raster(raster_path, title, fname):
    with rasterio.open(raster_path) as ds:
        window = from_bounds(minx, miny, maxx, maxy, ds.transform)
        data = ds.read(1, window=window)
        t = ds.window_transform(window)

    extent = [t.c, t.c + data.shape[1] * t.a,
              t.f + data.shape[0] * t.e, t.f]

    fig, ax = plt.subplots(figsize=(12, 9))
    display = data.astype(float)
    valid = data[data < 65535]
    norm = mcolors.PowerNorm(gamma=0.3, vmin=valid.min(), vmax=valid.max())
    im = ax.imshow(display, extent=extent, cmap="Greys", interpolation="nearest",
                   norm=norm)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Cost Value")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, fname)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


os.makedirs(OUT_DIR, exist_ok=True)

plot_raster(BASE_RASTER,
            "Step 1: Base Land Use Raster\n(23 cost categories from ALKIS land use data)",
            "overlay_step1_base.png")

plot_raster(MODIFIED_RASTER,
            "Step 2: Final Cost Raster\n(+ water protection, soil conditions, nature reserves)",
            "overlay_step2_modified.png")

print("All overlay step plots generated.")
