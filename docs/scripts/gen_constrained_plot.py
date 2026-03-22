"""Generate constrained path plot from existing 380kV 3D results with DEM heatmap."""

import os
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import from_bounds

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

OUTPUT = os.path.join(
    os.path.dirname(__file__), "../source/_static/generated/constrained_towers.png"
)
RESULTS = os.path.join(os.path.dirname(__file__), "../../examples/data/results")
DEM_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../case_studies/3d/Biebertal - DGM1/dgm1_merged.tif",
)

# Load existing results: 380kV 3D with DEM+DOM, R5
route = gpd.read_file(os.path.join(RESULTS, "route_constrained_380kv_3d_r7_DOM_DGM.geojson"))
towers = gpd.read_file(os.path.join(RESULTS, "towers_constrained_380kv_3d_r7_DOM_DGM.geojson"))

# Determine plot bounds from route with padding
bounds = route.total_bounds  # [minx, miny, maxx, maxy]
pad = 200
minx, miny, maxx, maxy = bounds[0] - pad, bounds[1] - pad, bounds[2] + pad, bounds[3] + pad

# Load DEM window for the route area
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

# Plot
fig, ax = plt.subplots(figsize=(14, 10))

# DEM heatmap background
im = ax.imshow(dem_data, extent=dem_extent, cmap="terrain", interpolation="bilinear",
               alpha=0.8)
cbar = fig.colorbar(im, ax=ax, shrink=0.7, label="Elevation (m)")

# Route path
for _, row in route.iterrows():
    xs, ys = row.geometry.xy
    ax.plot(xs, ys, color="#2c3e50", linewidth=2, zorder=3, label="Route")

# Tower markers by type
tower_styles = {
    "terminal":    ("D", "#e74c3c", 13, "Terminal"),
    "suspension":  ("o", "#3498db", 10, "Suspension"),
    "light_angle": ("s", "#f39c12", 12, "Light Angle"),
    "heavy_angle": ("^", "#9b59b6", 13, "Heavy Angle"),
}
plotted = set()
for _, t in towers.iterrows():
    ttype = t.tower_type
    marker, color, ms, label = tower_styles.get(ttype, ("o", "gray", 8, ttype))
    ax.plot(t.geometry.x, t.geometry.y, marker, color=color, markersize=ms,
            zorder=5, markeredgecolor="white", markeredgewidth=1.5,
            label=label if ttype not in plotted else None)
    plotted.add(ttype)

# Span lines
tower_pts = [(t.geometry.x, t.geometry.y) for _, t in towers.iterrows()]
for i in range(len(tower_pts) - 1):
    ax.plot([tower_pts[i][0], tower_pts[i + 1][0]],
            [tower_pts[i][1], tower_pts[i + 1][1]],
            "--", color="#7f8c8d", linewidth=0.8, alpha=0.5)

n_towers = len(towers)
terrain_cost = route.iloc[0].get("total_terrain_cost", 0)
tower_cost = route.iloc[0].get("total_tower_cost", 0)
total = terrain_cost + tower_cost

ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
ax.set_title(
    f"380kV Constrained Path with Tower Placement (R7, DEM+DSM)\n"
    f"{n_towers} towers — total cost: {total:,.0f}",
    fontsize=13, fontweight="bold",
)
ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
ax.set_aspect("equal")
fig.tight_layout()

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUTPUT}")
