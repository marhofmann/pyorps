"""Generate architecture data flow diagram using matplotlib."""

import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUTPUT = os.path.join(
    os.path.dirname(__file__), "../source/_static/generated/architecture_flow.png"
)

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.set_aspect("equal")
ax.axis("off")

box_style = dict(boxstyle="round,pad=0.4", facecolor="#ecf0f1", edgecolor="#2c3e50",
                 linewidth=2)
highlight_style = dict(boxstyle="round,pad=0.4", facecolor="#3498db", edgecolor="#2c3e50",
                       linewidth=2)
result_style = dict(boxstyle="round,pad=0.4", facecolor="#2ecc71", edgecolor="#2c3e50",
                    linewidth=2)

# Input layer
inputs = [
    ("GeoTIFF\nRaster", 1, 7),
    ("Shapefile\nVector", 3.5, 7),
    ("WFS\nRemote", 6, 7),
    ("NumPy\nIn-Memory", 8.5, 7),
]
for label, x, y in inputs:
    ax.text(x, y, label, ha="center", va="center", fontsize=9, bbox=box_style)

# GeoDataset
ax.text(4.5, 5.5, "GeoDataset\n(normalize CRS/resolution)", ha="center", va="center",
        fontsize=11, fontweight="bold", bbox=highlight_style, color="white")

# Arrows from inputs to GeoDataset
for label, x, y in inputs:
    ax.annotate("", xy=(4.5, 5.9), xytext=(x, y - 0.35),
                arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.5))

# GeoRasterizer (optional)
ax.text(2, 4, "GeoRasterizer\n(vector -> raster)", ha="center", va="center",
        fontsize=10, bbox=box_style)
ax.annotate("", xy=(2, 4.4), xytext=(4.5, 5.1),
            arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.5))
ax.annotate("", xy=(5, 3.4), xytext=(2, 3.6),
            arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.5))

# PathFinder
ax.text(7.5, 4, "PathFinder\n(main interface)", ha="center", va="center",
        fontsize=11, fontweight="bold", bbox=highlight_style, color="white")
ax.annotate("", xy=(7.5, 4.4), xytext=(4.5, 5.1),
            arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.5))

# Internal components
internals = [
    ("RasterHandler\n(windowing)", 5, 2.5),
    ("GraphAPI\n(backend)", 7.5, 2.5),
    ("Algorithm\n(Dijkstra/Delta)", 10, 2.5),
]
for label, x, y in internals:
    ax.text(x, y, label, ha="center", va="center", fontsize=9, bbox=box_style)
    ax.annotate("", xy=(x, y + 0.35), xytext=(7.5, 3.6),
                arrowprops=dict(arrowstyle="->", color="#95a5a6", lw=1.2))

# Results
ax.text(7.5, 1, "Path / PathCollection\n(GeoJSON, Shapefile, GeoPackage)", ha="center",
        va="center", fontsize=11, fontweight="bold", bbox=result_style, color="white")
ax.annotate("", xy=(7.5, 1.4), xytext=(7.5, 2.1),
            arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.5))

# Backend options
backends = ["Cython", "NetworKit", "Rustworkx", "NetworkX", "iGraph", "GPU"]
for i, name in enumerate(backends):
    bx = 11.5
    by = 5.5 - i * 0.6
    ax.text(bx, by, name, ha="center", va="center", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffeaa7",
                      edgecolor="#fdcb6e", linewidth=1))

ax.text(11.5, 6.0, "Backends", ha="center", va="center", fontsize=10, fontweight="bold")

ax.set_title("PYORPS Architecture: Data Flow Pipeline", fontsize=14, fontweight="bold",
             pad=20)

fig.tight_layout()

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUTPUT}")
