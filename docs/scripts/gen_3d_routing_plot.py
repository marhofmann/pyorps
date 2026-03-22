"""Generate 3D routing comparison: 2D vs 3D constrained paths on DEM heatmap."""

import os
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import rasterio
from rasterio.windows import from_bounds

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

OUTPUT = os.path.join(
    os.path.dirname(__file__), "../source/_static/generated/3d_routing.png"
)
RESULTS = os.path.join(os.path.dirname(__file__), "../../examples/data/results")
DEM_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../case_studies/3d/Biebertal - DGM1/dgm1_merged.tif",
)
COST_RASTER = os.path.join(
    os.path.dirname(__file__),
    "../../examples/data/raster/modified_raster_for_distribution_grid_planning.tiff",
)

# Load 2D result (no DEM) and 3D result (with DEM+DOM), both R5
route_2d = gpd.read_file(os.path.join(RESULTS, "route_constrained_380kv_r5.geojson"))
towers_2d = gpd.read_file(os.path.join(RESULTS, "towers_constrained_380kv_r5.geojson"))
route_3d = gpd.read_file(os.path.join(RESULTS, "route_constrained_380kv_3d_r7_DOM_DGM.geojson"))
towers_3d = gpd.read_file(os.path.join(RESULTS, "towers_constrained_380kv_3d_r7_DOM_DGM.geojson"))

# Determine plot bounds from both routes
all_bounds = np.array([route_2d.total_bounds, route_3d.total_bounds])
pad = 200
minx = all_bounds[:, 0].min() - pad
miny = all_bounds[:, 1].min() - pad
maxx = all_bounds[:, 2].max() + pad
maxy = all_bounds[:, 3].max() + pad

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
with rasterio.open(COST_RASTER) as cost_ds:
    window_c = from_bounds(minx, miny, maxx, maxy, cost_ds.transform)
    cost_data = cost_ds.read(1, window=window_c)
    cost_transform = cost_ds.window_transform(window_c)

cost_extent = [
    cost_transform.c,
    cost_transform.c + cost_data.shape[1] * cost_transform.a,
    cost_transform.f + cost_data.shape[0] * cost_transform.e,
    cost_transform.f,
]

tower_styles = {
    "terminal":    ("D", 11),
    "suspension":  ("o", 8),
    "light_angle": ("s", 10),
    "heavy_angle": ("^", 11),
}

fig, axes = plt.subplots(1, 2, figsize=(18, 9))

# --- Left: 2D routing on cost raster ---
ax = axes[0]
cost_display = cost_data.astype(float)
valid_cost = cost_data[cost_data < 65535]
norm = mcolors.PowerNorm(gamma=0.3, vmin=valid_cost.min(), vmax=valid_cost.max())
ax.imshow(cost_display, extent=cost_extent, cmap="Greys", interpolation="nearest", norm=norm)

for _, row in route_2d.iterrows():
    xs, ys = row.geometry.xy
    ax.plot(xs, ys, color="#e74c3c", linewidth=2.5, zorder=3)

for _, t in towers_2d.iterrows():
    marker, ms = tower_styles.get(t.tower_type, ("o", 8))
    ax.plot(t.geometry.x, t.geometry.y, marker, color="#e74c3c", markersize=ms,
            zorder=5, markeredgecolor="white", markeredgewidth=1)

cost_2d = route_2d.iloc[0].get("total_terrain_cost", 0) + route_2d.iloc[0].get("total_tower_cost", 0)
ax.set_title(f"2D Routing (cost raster only)\n{len(towers_2d)} towers — cost: {cost_2d:,.0f}",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
ax.set_aspect("equal")

# --- Right: 3D routing on DEM heatmap ---
ax = axes[1]
im = ax.imshow(dem_data, extent=dem_extent, cmap="terrain", interpolation="bilinear", alpha=0.85)

for _, row in route_3d.iterrows():
    xs, ys = row.geometry.xy
    ax.plot(xs, ys, color="#e74c3c", linewidth=2.5, zorder=3)

for _, t in towers_3d.iterrows():
    marker, ms = tower_styles.get(t.tower_type, ("o", 8))
    ax.plot(t.geometry.x, t.geometry.y, marker, color="#e74c3c", markersize=ms,
            zorder=5, markeredgecolor="white", markeredgewidth=1)

cost_3d = route_3d.iloc[0].get("total_terrain_cost", 0) + route_3d.iloc[0].get("total_tower_cost", 0)
ax.set_title(f"3D Routing (DEM + DSM aware)\n{len(towers_3d)} towers — cost: {cost_3d:,.0f}",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
ax.set_aspect("equal")

fig.colorbar(im, ax=axes[1], shrink=0.7, label="Elevation (m)")
fig.suptitle("2D vs 3D Constrained Routing (380kV, R7)", fontsize=14, fontweight="bold")
fig.tight_layout()

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUTPUT}")
