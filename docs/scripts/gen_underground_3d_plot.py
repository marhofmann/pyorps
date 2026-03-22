"""Generate underground cable 3D routing comparison: with and without DEM."""

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import rasterio
from rasterio.windows import from_bounds

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pyorps import PathFinder

OUTPUT = os.path.join(
    os.path.dirname(__file__), "../source/_static/generated/underground_3d.png"
)
COST_RASTER = os.path.join(
    os.path.dirname(__file__),
    "../../examples/data/raster/modified_raster_for_distribution_grid_planning.tiff",
)
DEM_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../case_studies/3d/Biebertal - DGM1/dgm1_merged.tif",
)

source = (473609, 5607305)
target = (474443, 5606872)
BUFFER = 500

# --- 2D route (no DEM) ---
pf_2d = PathFinder(
    dataset_source=COST_RASTER,
    source_coords=source,
    target_coords=target,
    search_space_buffer_m=BUFFER,
    neighborhood_str="r2",
)
pf_2d.find_route()
path_2d = pf_2d.paths.all[0]
coords_2d = np.array(path_2d.path_coords)

# --- 3D route (with DEM) ---
pf_3d = PathFinder(
    dataset_source=COST_RASTER,
    source_coords=source,
    target_coords=target,
    search_space_buffer_m=BUFFER,
    neighborhood_str="r2",
    dem=DEM_PATH,
)
pf_3d.find_route()
path_3d = pf_3d.paths.all[0]
coords_3d = np.array(path_3d.path_coords)

# Determine plot bounds
all_x = np.concatenate([coords_2d[:, 0], coords_3d[:, 0]])
all_y = np.concatenate([coords_2d[:, 1], coords_3d[:, 1]])
pad = 200
minx, maxx = all_x.min() - pad, all_x.max() + pad
miny, maxy = all_y.min() - pad, all_y.max() + pad

# Load DEM window
with rasterio.open(DEM_PATH) as dem_ds:
    window = from_bounds(minx, miny, maxx, maxy, dem_ds.transform)
    dem_data = dem_ds.read(1, window=window)
    dem_transform = dem_ds.window_transform(window)

dem_extent = [
    dem_transform.c,
    dem_transform.c + dem_data.shape[1] * dem_transform.a,
    dem_transform.f + dem_data.shape[0] * dem_transform.e,
    dem_transform.f,
]

# Load cost raster window
raster_data = np.squeeze(pf_2d.raster_handler.data)
t = pf_2d.raster_handler.window_transform
cost_extent = [t.c, t.c + raster_data.shape[1] * t.a,
               t.f + raster_data.shape[0] * t.e, t.f]

fig, axes = plt.subplots(1, 2, figsize=(18, 9))

# --- Left: 2D on cost raster ---
ax = axes[0]
display = raster_data.astype(float)
valid = raster_data[raster_data < 65535]
norm = mcolors.PowerNorm(gamma=0.3, vmin=valid.min(), vmax=valid.max())
ax.imshow(display, extent=cost_extent, cmap="Greys", interpolation="nearest", norm=norm)
ax.plot(coords_2d[:, 0], coords_2d[:, 1], color="#e74c3c", linewidth=2.5, zorder=3)
ax.plot(*source, "o", color="#2ecc71", markersize=12, zorder=5)
ax.plot(*target, "X", color="#3498db", markersize=12, zorder=5)
cost_str = f"cost: {path_2d.total_cost:,.0f}" if path_2d.total_cost else ""
ax.set_title(f"Underground Cable — 2D (cost only)\n{cost_str}",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
ax.set_aspect("equal")

# --- Right: 3D on DEM heatmap ---
ax = axes[1]
im = ax.imshow(dem_data, extent=dem_extent, cmap="terrain", interpolation="bilinear",
               alpha=0.85)
ax.plot(coords_3d[:, 0], coords_3d[:, 1], color="#e74c3c", linewidth=2.5, zorder=3)
ax.plot(*source, "o", color="#2ecc71", markersize=12, zorder=5)
ax.plot(*target, "X", color="#3498db", markersize=12, zorder=5)
cost_str = f"cost: {path_3d.total_cost:,.0f}" if path_3d.total_cost else ""
ax.set_title(f"Underground Cable — 3D (DEM gradient penalty)\n{cost_str}",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
ax.set_aspect("equal")

fig.colorbar(im, ax=axes[1], shrink=0.7, label="Elevation (m)")
fig.suptitle("Underground Cable Routing: 2D vs 3D (DEM-Aware)",
             fontsize=14, fontweight="bold")
fig.tight_layout()

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUTPUT}")
