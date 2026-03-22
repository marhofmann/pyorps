"""Generate GPU performance chart: log-scale comparison of backends by raster size."""

import os

import matplotlib.pyplot as plt
import numpy as np

OUTPUT = os.path.join(
    os.path.dirname(__file__), "../source/_static/generated/gpu_performance.png"
)

# Pre-computed benchmark data from MEMORY.md
sizes = [500, 1000, 2000, 3000]
size_labels = ["500x500", "1000x1000", "2000x2000", "3000x3000"]

gpu_times = [0.026, 0.054, 0.133, 0.235]
cython_times = [0.068, 0.285, 1.158, 2.672]
networkit_times = [0.140, 0.744, 3.136, 8.800]

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(sizes, gpu_times, "o-", color="#e74c3c", linewidth=2.5, markersize=8,
        label="GPU (RasterGPU)")
ax.plot(sizes, cython_times, "s-", color="#3498db", linewidth=2.5, markersize=8,
        label="Cython (CPU)")
ax.plot(sizes, networkit_times, "^-", color="#2ecc71", linewidth=2.5, markersize=8,
        label="NetworKit")

ax.set_yscale("log")
ax.set_xlabel("Raster Size (side length)")
ax.set_ylabel("Runtime (seconds, log scale)")
ax.set_title("GPU vs CPU Performance Scaling")
ax.set_xticks(sizes)
ax.set_xticklabels(size_labels)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Add speedup annotations
for i, size in enumerate(sizes):
    speedup = cython_times[i] / gpu_times[i]
    ax.annotate(
        f"{speedup:.1f}x",
        xy=(size, gpu_times[i]),
        xytext=(0, -20),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        color="#e74c3c",
        fontweight="bold",
    )

fig.tight_layout()

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUTPUT}")
