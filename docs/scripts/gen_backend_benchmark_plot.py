"""Generate backend benchmark bar chart using pre-computed data."""

import os

import matplotlib.pyplot as plt
import numpy as np

OUTPUT = os.path.join(
    os.path.dirname(__file__), "../source/_static/generated/backend_benchmark.png"
)

# Pre-computed benchmark data (representative values)
sizes = ["500x500", "1000x1000", "2000x2000", "3000x3000"]
data = {
    "Cython (default)": [0.068, 0.285, 1.158, 2.672],
    "NetworKit": [0.140, 0.744, 3.136, 8.800],
    "GPU": [0.026, 0.054, 0.133, 0.235],
}

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(sizes))
width = 0.25
colors = ["#3498db", "#2ecc71", "#e74c3c"]

for i, (name, times) in enumerate(data.items()):
    ax.bar(x + i * width, times, width, label=name, color=colors[i])

ax.set_xlabel("Raster Size")
ax.set_ylabel("Runtime (seconds)")
ax.set_title("Backend Performance Comparison")
ax.set_xticks(x + width)
ax.set_xticklabels(sizes)
ax.legend()
ax.set_yscale("log")
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUTPUT}")
