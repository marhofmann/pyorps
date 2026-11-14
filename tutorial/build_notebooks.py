"""
Build the OSMSES 2026 tutorial notebooks for PYORPS.

This script is the single source of both notebooks. Edit the cells below, then run it. Do not edit the notebooks by
hand: the next rebuild would revert the change, and ``--check`` (run by ``tests/test_tutorial_generator.py``) fails
until the script and the notebooks agree again.

It writes two notebooks:

1. ``OSMSES_2026_Tutorial_PYORPS_full.ipynb``: all code and explanations (the solutions);
2. ``OSMSES_2026_Tutorial_PYORPS_exercises.ipynb``: the same cells, with gaps for participants to fill in. A cell
   defined with ``exercise_source`` carries that text here; the cell at the same position in the full notebook is its
   solution.

Usage (paths are anchored on this file, so the working directory does not matter)::

    python tutorial/build_notebooks.py                 # rebuild both notebooks next to this script
    python tutorial/build_notebooks.py --out-dir DIR   # write them to DIR instead
    python tutorial/build_notebooks.py --check         # write nothing; exit 1 if the notebooks in the target
                                                       # folder differ from what this script generates

A rebuild keeps the outputs, execution counts and cell metadata of every cell whose source did not change, so the
rendered results survive; a changed cell starts empty. ``--check`` compares the number, order, type, id and source text
of the cells and the notebook metadata that matters for validity; outputs and execution counts are not compared.
"""

import argparse
import copy
import difflib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FULL_NAME = "OSMSES_2026_Tutorial_PYORPS_full.ipynb"
EXERCISES_NAME = "OSMSES_2026_Tutorial_PYORPS_exercises.ipynb"

#: The oldest Python a notebook may claim in its metadata (the project's ``requires-python``).
MIN_PYTHON = (3, 12)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def md(source, exercise_source=None):
    """Create a markdown cell. If exercise_source is given, it will be used in the exercise notebook."""
    cell = {"cell_type": "markdown", "metadata": {}, "source": source}
    if exercise_source is not None:
        cell["_exercise_source"] = exercise_source
    return cell

def code(source, exercise_source=None):
    """Create a code cell. If exercise_source is given, it will be used in the exercise notebook."""
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }
    if exercise_source is not None:
        cell["_exercise_source"] = exercise_source
    return cell


# language_info.version is left at the project minimum: Jupyter replaces it with the version of the kernel that runs
# the notebook, and a rebuild keeps that value as long as it is not older than MIN_PYTHON.
NOTEBOOK_METADATA = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.12",
    },
}

# Cell ids (nbformat 4.5 needs one per cell), in cell order, as they are in the committed notebooks. The two notebooks
# have different ids. When you insert a cell, insert its id at the same position in both tuples; positions beyond the
# end of a tuple get "cell-NNN" from their index.
CELL_IDS = {
    "full": (
        "8883a960014a7a4f", "2654e8e6166182e1", "7034daff4af1aa3d", "2034948c5046a405",  # cells 0-3
        "c760cf2e67c60d56", "eead340015f814a1", "e10c0e9c4154e1c6", "73522a4005d3599",  # cells 4-7
        "83123dd677d0b4a6", "684b2b60f2938496", "92f98ff1de78cfb5", "de6f8ac943567ab0",  # cells 8-11
        "3884aae745190fb3", "b25054898d0dadf2", "924ae98fc477caef", "e73655128eb37013",  # cells 12-15
        "9fb286db6f13b47f", "28fe908865562911", "7c170b6b56654626", "1e5983cd3fb59e7e",  # cells 16-19
        "acd48bf79504905c", "ca51049c32f9c33c", "b9513bc1405d4a28", "0pzs2u06znb",  # cells 20-23
        "j1kmkb2gar", "3tkkkeuxfck", "n6yqpbgrjlo", "cq2388inoga",  # cells 24-27
        "426c733f968963ac", "a87d40c58cddf268", "1bcdebe116e1b5de", "cf2bd2d99df72796",  # cells 28-31
        "b2f9705e3a21aee3", "8fe79dbbe098b5b0", "6f85018c1e877d32", "b52dd86b833116f6",  # cells 32-35
        "3676e6c42eb1273f", "993d9cc04b51d8c3", "c2abc76184c7ce6f", "64f747a6ae394b45",  # cells 36-39
        "340ab6aee57e62fd", "5700b7cf750aac28", "a367adac6245327f", "efa26c96fc9b5801",  # cells 40-43
        "a522f540c109e2f6", "5f70970b5543d49c", "c919d3fb32ab0f08", "b064e623ac98082a",  # cells 44-47
        "38eebbfe23f484a2", "817b245defb62e9d", "928040db89e38ad4", "568ce57235d77de5",  # cells 48-51
        "2381b9d79e1edb9f", "f8ed0310ca0f719b", "eb46f6dc6230a6c8", "3798255c30e6eea6",  # cells 52-55
        "91eb7e5074f499fe", "c40012b102a3b994", "9d9f7f3756a17a10", "ad1cdfac602149a1",  # cells 56-59
        "fd1c0bbd18179a0a", "e254cd11d464ef07",  # cells 60-61
    ),
    "exercises": (
        "048eba2c08b6", "ac29f3d31b79", "0f8e0781056a", "adaa52607cbf",  # cells 0-3
        "a9c5ce74534f", "1e4866275c79", "4b691605e4a3", "d6e38399d643",  # cells 4-7
        "f31dbb155be1", "efba3954aca5", "0d1dc752b3dc", "3a557a17a7c5",  # cells 8-11
        "752003a7d3bb", "c57bdcbfdda2", "6808629fb64d", "f2f6a7e899d9",  # cells 12-15
        "f4f38dd897a9", "6b9457552145", "6c2e298fa210", "05a188d7b010",  # cells 16-19
        "84c69cc99ad2", "467a84f3a046", "1f3dbb183cc6", "eec7210f92fa",  # cells 20-23
        "203ef467de59", "01dbd70814d2", "7ef752640d8a", "476fa11b3e85",  # cells 24-27
        "464c8f73d430", "674a689f8f2a", "72eeb9158892", "9492c2dfa5d9",  # cells 28-31
        "edbd2bc7309e", "ed6b8897b64d", "54a3f9e03df9", "a13ca32612af",  # cells 32-35
        "1e53f747a173", "d3a55288a20a", "35643ef20b11", "32dd9d061e0c",  # cells 36-39
        "ec0713018982", "8ded859fc54e", "ff2e10ddeb53", "0bdd0dfebb11",  # cells 40-43
        "8a2c3a877d75", "4624ed63f5ce", "49d3fecd39da", "5b38e50eccbf",  # cells 44-47
        "272c46506d2c", "fff03dd62757", "e6ab7f70dbe7", "9e1febb2dbd4",  # cells 48-51
        "e474c2823818", "31ff69346b55", "c5e2a3371995", "1cf87edab281",  # cells 52-55
        "302c19b67d47", "cc8974fe0358", "1bc862c2a09b", "ab0b8619e83e",  # cells 56-59
        "bca72572c7be", "253e3142966a",  # cells 60-61
    ),
}

# ---------------------------------------------------------------------------
# Cell definitions
# ---------------------------------------------------------------------------

cells = []

# ==========================================================================
# PYORPS Tutorial: Automated Underground Cable Routing for Distribution Grid Planning
# ==========================================================================

cells.append(md(
"""# PYORPS Tutorial: Automated Underground Cable Routing for Distribution Grid Planning

**OSMSES 2026 &mdash; 4th International Workshop on Open Source Modelling and Simulation of Energy Systems**
*March 23&ndash;25, 2026 &mdash; Karlsruhe, Germany*

**Speaker:** Martin Hofmann M.Sc.

---

## Tutorial Overview

This hands-on tutorial introduces **PYORPS** (*Python for Optimal Routes in Power Systems*), an open-source tool for automated power line routing using least-cost path analysis on high-resolution raster geodata.

**Duration:** 1.5 hours

### What You Will Learn

| # | Topic | Time |
|---|-------|------|
| 1 | PYORPS Fundamentals &mdash; raster costs, neighborhoods, pathfinding | ~15 min |
| 2 | **Creating a Cost Raster** &mdash; from open geodata (WFS) with multiple layers | ~20 min |
| 3 | **Case Study 1** &mdash; PCC optimization for a utility-scale PV system | ~25 min |
| 4 | **Case Study 2** &mdash; Automated grid extension with Delaunay triangulation & ring opening | ~35 min |
| 5 | Summary & next steps | ~10 min |

### Prerequisites
- Basic knowledge of power distribution systems
- Python programming (intermediate level)
- `pyorps`, `pandapower`, `scipy`, `geopandas`, `matplotlib` installed"""
))

# 0. Setup & Installation

cells.append(md(
"""## 0. Setup & Installation

Install PYORPS with all optional dependencies (includes pandapower, geopandas, etc.):

```bash
pip install pyorps[full]
```

If you only need the core routing functionality:
```bash
pip install pyorps
```"""
))

cells.append(code(
"""# Verify installation
import pyorps
print(f"PYORPS version: {pyorps.__version__}")"""
))

cells.append(code(
"""# Standard imports used throughout the tutorial
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm, PowerNorm
from shapely.geometry import Point, LineString, MultiPoint
from scipy.spatial import Delaunay

import pandapower as pp
from pandapower.topology import create_nxgraph
from networkx import cycle_basis

from pyorps import (PathFinder, PathCollection, initialize_geo_dataset,
                    GeoRasterizer, CostAssumptions, detect_feature_columns)
import rasterio
from rasterio.plot import show as rshow


def plot_cost_raster(ax, data, title=None, alpha=0.7):
    ""\"Plot cost raster in greyscale (dark = low cost, white = forbidden).""\"
    display = np.where(data >= 65535, np.nan, data.astype(float))
    ax.imshow(display, cmap='gray', norm=PowerNorm(gamma=0.5, vmin=0, vmax=500),
              origin='upper', interpolation='nearest', alpha=alpha)
    if title:
        ax.set_title(title, fontsize=13)""",

exercise_source=\
"""# Standard imports used throughout the tutorial
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
from shapely.geometry import Point, LineString, MultiPoint
from scipy.spatial import Delaunay

import pandapower as pp
from pandapower.topology import create_nxgraph
from networkx import cycle_basis

from pyorps import (PathFinder, PathCollection, initialize_geo_dataset,
                    GeoRasterizer, CostAssumptions, detect_feature_columns)
import rasterio
from rasterio.plot import show as rshow

# Plotting defaults
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

print("All imports successful!")"""
))

# ==========================================================================
# Introduction: How PYORPS Works
# ==========================================================================

cells.append(md(
"""---
# Introduction: How PYORPS Works

PYORPS converts the routing problem into a **graph-based shortest-path problem**:

1. **Raster input** &mdash; Each pixel represents a terrain type with an associated construction cost (EUR/m)
2. **Graph creation** &mdash; Pixels become nodes; edges connect neighboring pixels weighted by cost &times; distance
3. **Shortest path** &mdash; Dijkstra's algorithm finds the minimum-cost route
4. **Result** &mdash; A georeferenced path with length, cost breakdown, and exportable geometry

### Cost Values

| Value | Meaning |
|-------|---------|
| 0 | Free passage (e.g., existing duct) |
| 90&ndash;500 | Typical construction costs in EUR/m |
| 65535 | **Forbidden / impassable** (residential areas, protected zones) |

Values are stored as `uint16` (0&ndash;65535).""",

exercise_source=\
"""---
# Introduction: How PYORPS Works

PYORPS converts the routing problem into a **graph-based shortest-path problem**:

1. **Raster input** &mdash; Each pixel represents a terrain type with an associated construction cost (EUR/m)
2. **Graph creation** &mdash; Pixels become nodes; edges connect neighboring pixels weighted by cost &times; distance
3. **Shortest path** &mdash; Dijkstra's algorithm finds the minimum-cost route
4. **Result** &mdash; A georeferenced path with length, cost breakdown, and exportable geometry

### Cost Values

| Value | Meaning |
|-------|--------|
| 0 | Free passage (e.g., existing duct) |
| 90&ndash;500 | Typical construction costs in EUR/m |
| 65535 | **Forbidden / impassable** (residential areas, protected zones) |

Values are stored as `uint16` (0&ndash;65535)."""
))

# Minimal Example

cells.append(md(
"""## Minimal Example

Three lines are enough: create a `PathFinder`, call `find_route()`, visualize."""
))

cells.append(code(
"""source = (472000, 5593400)
target = (472800, 5594000)
raster_path = "../examples/data/raster/sample_raster.tiff"

pf = PathFinder(dataset_source=raster_path, source_coords=source,
                target_coords=target, neighborhood_str="r2")

path = pf.find_route()
print(path)
print(f"\\nLength: {path.total_length:.0f} m | Cost: {path.total_cost:.0f} EUR "
      f"| Detour: {path.total_length / path.euclidean_distance:.2f}x")

pf.plot_paths()
plt.title("Minimal Example: Optimal Cable Route")
plt.tight_layout()
plt.show()""",

exercise_source=\
"""source = (472000, 5593400)
target = (472800, 5594000)
raster_path = "../examples/data/raster/sample_raster.tiff"

# TODO: Create a PathFinder and call find_route().
# pf = PathFinder(dataset_source=..., source_coords=...,
#                 target_coords=..., neighborhood_str="r2")
# path = pf.find_route()
# print(path)

# YOUR CODE HERE"""
))

# ==========================================================================
# Part 1: Creating a Cost Raster from Open Geodata
# ==========================================================================

cells.append(md(
"""---
# Part 1: Creating a Cost Raster from Open Geodata

In practice you build your cost raster from publicly available geodata.
This part shows how &mdash; step by step:

1. **Base layer** &mdash; Land-use data from the official cadastre (ALKIS)
2. **Protection layers** &mdash; Naturschutz, Vogelschutz, Wasserschutz, Landschaftsschutz

Each layer is fetched live from an official **WFS** (Web Feature Service) provided by
the state of Baden-W&uuml;rttemberg."""
))

# 1.1 Study Area &amp; Base Land-Use Data

cells.append(md(
"""## 1.1 Study Area &amp; Base Land-Use Data

We pick a **5 &times; 5&nbsp;km** window inside the MV Oberrhein region and
load **ALKIS** land-use polygons via WFS.""",

exercise_source=\
"""## 1.1 Study Area &amp; Base Land-Use Data

We pick a **5 &times; 5&nbsp;km** window inside the MV Oberrhein region and load
the ALKIS land-use data via the LGL-BW WFS."""
))

cells.append(code(
"""# 5 x 5 km bounding box in the MV Oberrhein area (EPSG:25832)
bbox_tutorial = (410000, 5358000, 415000, 5363000)

# WFS for Baden-Wuerttemberg ALKIS land-use data
alkis_wfs = {
    'url': r"https://owsproxy.lgl-bw.de/owsproxy/wfs/WFS_LGL-BW_ALKIS?version=2.0.0",
    'layer': "Tatsaechliche Nutzung"
}

# Load and inspect
base_dataset = initialize_geo_dataset(alkis_wfs, bbox=bbox_tutorial)
base_dataset.load_data()
print(f"Features loaded: {len(base_dataset.data):,}")

# detect_feature_columns() suggests the best column for cost categorisation
main_feature, _ = detect_feature_columns(base_dataset.data, max_features_per_column=100)
print(f"Feature column: '{main_feature}'")
print(base_dataset.data[main_feature].value_counts().head(10).to_string())""",

exercise_source=\
"""# 5 x 5 km bounding box in the MV Oberrhein area (EPSG:25832)
bbox_tutorial = (410000, 5358000, 415000, 5363000)

# TODO: Define a WFS request dict with 'url' and 'layer' keys for BW ALKIS data.
# Hint: URL = https://owsproxy.lgl-bw.de/owsproxy/wfs/WFS_LGL-BW_ALKIS?version=2.0.0
#       Layer = "Tatsaechliche Nutzung"
alkis_wfs = {
    # YOUR CODE HERE
}

base_dataset = initialize_geo_dataset(alkis_wfs, bbox=bbox_tutorial)
base_dataset.load_data()
print(f"Features loaded: {len(base_dataset.data):,}")

# detect_feature_columns() suggests the best column for cost categorisation
main_feature, _ = detect_feature_columns(base_dataset.data, max_features_per_column=100)
print(f"Feature column: '{main_feature}'")
print(base_dataset.data[main_feature].value_counts().head(10).to_string())"""
))

# 1.2 Define Cost Assumptions

cells.append(md(
"""## 1.2 Define Cost Assumptions

Each land-use category gets a construction cost in **EUR/m**.
The special value **65535** marks areas as **impassable** (e.g., residential zones).""",

exercise_source=\
"""## 1.2 Define Cost Assumptions

Each land-use category gets a construction cost in **EUR/m**.
The special value **65535** marks a cell as *forbidden*."""
))

cells.append(code(
"""# Cost assumptions for BW ALKIS land-use categories (EUR/m)
base_cost_assumptions = {
    main_feature: {
        # --- Forbidden areas (65535 = impassable) ---
        'Wohnbaufläche': 65535,
        'Gebäude- und Freifläche Industrie und Gewerbe': 65535,
        'Gebäude- und Freifläche Land- und Forstwirtschaft': 65535,
        'Handel und Dienstleistung': 65535,
        'Friedhof': 65535,
        'Bergbau': 65535,
        'Sumpf': 65535,
        'Flugverkehr': 65535,

        # --- Passable areas (sorted by increasing cost) ---
        'Weg': 97,
        'Graben': 120,
        'Grünland': 200,
        'Ackerland': 250,
        'Garten': 300,
        'Platz': 152,
        'Fließgewässer': 155,
        'Stehendes Gewässer': 155,
        'Straßenverkehr': 178,
        'Laubholz': 365,
        'Nadelholz': 365,
        'Laub- und Nadelholz': 365,
        'Bahnverkehr': 415,
        'Sport-, Freizeit- und Erholungsfläche': 107,
        'Heide': 92,
        'Moor': 92,
        'Unland/Vegetationslose Fläche': 92,
    }
}""",

exercise_source=\
"""# TODO: Define cost assumptions mapping the land-use categories (the values of main_feature) to EUR/m costs.
# Use 65535 for forbidden areas, realistic values for passable terrain.
base_cost_assumptions = {
    main_feature: {
        'Wohnbaufläche': 65535,
        # ... add more forbidden categories ...
        'Weg': 97,
        'Ackerland': 250,
        # ... add more passable categories ...
        # YOUR CODE HERE
    }
}"""
))

# 1.3 Rasterize the Base Layer

cells.append(md(
"""## 1.3 Rasterize the Base Layer

`GeoRasterizer` converts the vector polygons into a raster where each pixel holds
the cost value of its land-use category."""
))

cells.append(code(
"""import os
os.makedirs("../tutorial/output", exist_ok=True)
base_raster_path = "../tutorial/output/base_cost_raster.tiff"

geo_rasterizer = GeoRasterizer(base_dataset, CostAssumptions(base_cost_assumptions))
geo_rasterizer.rasterize(save_path=base_raster_path)

with rasterio.open(base_raster_path) as src:
    base_data = src.read(1)
    print(f"Raster: {base_data.shape}, {abs(src.transform.a):.1f} m/pixel")

fig, ax = plt.subplots(figsize=(10, 8))
plot_cost_raster(ax, base_data, title="Step 1: Base Land-Use Cost Raster")
plt.tight_layout()
plt.show()""",

exercise_source=\
"""import os
os.makedirs("../tutorial/output", exist_ok=True)
base_raster_path = "../tutorial/output/base_cost_raster.tiff"

# TODO: Create a GeoRasterizer and call rasterize().
# geo_rasterizer = GeoRasterizer(base_dataset, CostAssumptions(base_cost_assumptions))
# geo_rasterizer.rasterize(save_path=base_raster_path)

# YOUR CODE HERE"""
))

# 1.4 Define Modification Layers (Protection Zones)

cells.append(md(
"""## 1.4 Define Modification Layers (Protection Zones)

Each additional dataset modifies the base raster:

| Layer | WFS source | Effect |
|-------|-----------|--------|
| Naturschutzgebiete | LUBW | Override &rarr; 65535 (forbidden) |
| Vogelschutzgebiete | LUBW | Multiply &times; 1.75 |
| Wasserschutzgebiete | LUBW | Multiply &times; 1.5 |
| Landschaftsschutzgebiete | LUBW | Multiply &times; 1.25 |""",

exercise_source=\
"""## 1.4 Define Modification Layers (Protection Zones)

Each additional dataset modifies the base raster:

| Layer | WFS source (LUBW) | Effect |
|-------|-------------------|--------|
| Naturschutzgebiete | Forbidden (&rarr; 65535) | `multiply=False` |
| Vogelschutzgebiete (SPA) | &times;1.75 | `multiply=True` |
| Wasserschutzgebiete | &times;1.5 | `multiply=True` |
| Landschaftsschutzgebiete | &times;1.25 | `multiply=True` |"""
))

cells.append(code(
"""# LUBW base URL for WFS services
lubw = "https://rips-gdi.lubw.baden-wuerttemberg.de/arcgis/services/wfs"

# WFS request dicts
nature_wfs      = {'url': f"{lubw}/Naturschutzgebiet/MapServer/WFSServer",
                   'layer': "Naturschutzgebiet"}
bird_wfs        = {'url': f"{lubw}/Vogelschutzgebiet_SPA/MapServer/WFSServer",
                   'layer': "Vogelschutzgebiet_SPA"}
water_wfs       = {'url': f"{lubw}/Wasserschutzgebiet/MapServer/WFSServer",
                   'layer': "Wasserschutzgebiet"}
landscape_wfs   = {'url': f"{lubw}/Landschaftsschutzgebiet/MapServer/WFSServer",
                   'layer': "Landschaftsschutzgebiet"}

# Modification list: each dict is passed to geo_rasterizer.modify_raster_from_dataset()
datasets_to_modify = [
    {"input_data": nature_wfs,    "cost_assumptions": 65535, "multiply": False},
    {"input_data": bird_wfs,      "cost_assumptions": 1.75,  "multiply": True},
    {"input_data": water_wfs,     "cost_assumptions": 1.5,   "multiply": True},
    {"input_data": landscape_wfs, "cost_assumptions": 1.25,  "multiply": True},
]

for d in datasets_to_modify:
    action = "multiply" if d["multiply"] else "override"
    print(f"  {d['input_data']['layer']:30s}  ({action}, {d['cost_assumptions']})")""",

exercise_source=\
"""# LUBW base URL
lubw = "https://rips-gdi.lubw.baden-wuerttemberg.de/arcgis/services/wfs"

# TODO: Define WFS request dicts for each protection layer and build the
# datasets_to_modify list. Each entry needs: "input_data", "cost_assumptions", "multiply".

datasets_to_modify = [
    # {"input_data": ..., "cost_assumptions": 65535, "multiply": False},  # Naturschutz
    # {"input_data": ..., "cost_assumptions": 1.75,  "multiply": True},   # Vogelschutz
    # {"input_data": ..., "cost_assumptions": 1.5,   "multiply": True},   # Wasserschutz
    # {"input_data": ..., "cost_assumptions": 1.25,  "multiply": True},   # Landschaftsschutz
    # YOUR CODE HERE
]"""
))

# 1.5 Apply All Layers &amp; Compare

cells.append(md(
"""## 1.5 Apply All Layers &amp; Compare

We apply the modifications and compare the base raster against the final result."""
))

cells.append(code(
"""# Apply each protection layer to the base raster
for mod in datasets_to_modify:
    geo_rasterizer.modify_raster_from_dataset(**mod)

final_raster_path = "../tutorial/output/modified_cost_raster.tiff"
geo_rasterizer.save_raster(save_path=final_raster_path)

# Compare side by side
with rasterio.open(final_raster_path) as src:
    mod_data = src.read(1)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for ax, data, title in zip(axes, [base_data, mod_data],
                            ["Base (land use only)", "Modified (+ protection zones)"]):
    plot_cost_raster(ax, data, title=title)
plt.tight_layout()
plt.show()

print(f"Forbidden cells:  {(base_data == 65535).sum():>8,d}  ->  {(mod_data == 65535).sum():>8,d}")""",

exercise_source=\
"""# TODO: Apply all modifications and save the result.
# Hint: loop over datasets_to_modify, call geo_rasterizer.modify_raster_from_dataset(**mod)

# YOUR CODE HERE

final_raster_path = "../tutorial/output/modified_cost_raster.tiff"
geo_rasterizer.save_raster(save_path=final_raster_path)"""
))

# 1.6 Shortcut: All-in-One with PathFinder

cells.append(md(
"""## 1.6 Shortcut: All-in-One with PathFinder

`PathFinder` can handle rasterisation **and** modification internally &mdash; no need
to create `GeoRasterizer` manually. Pass the WFS dict as `dataset_source`, the cost
assumptions, and the `datasets_to_modify` list directly:"""
))

cells.append(code(
"""# Define source and target on passable land (farmland area, away from villages)
source_custom = (411200, 5361300)
target_custom = (414500, 5362500)

pf_onestep = PathFinder(
    dataset_source=alkis_wfs,
    cost_assumptions=base_cost_assumptions,
    datasets_to_modify=datasets_to_modify,
    bbox=bbox_tutorial,
    source_coords=source_custom,
    target_coords=target_custom,
    raster_save_path="../tutorial/output/onestep_raster.tiff",
)
path_onestep = pf_onestep.find_route()
print(path_onestep)

pf_onestep.plot_paths()
plt.title("All-in-One: WFS -> Raster -> Route in a single call")
plt.tight_layout()
plt.show()""",

exercise_source=\
"""# Define source and target on passable land (farmland area, away from villages)
source_custom = (411200, 5361300)
target_custom = (414500, 5362500)

# TODO: Use PathFinder with dataset_source=alkis_wfs, cost_assumptions,
# datasets_to_modify, and bbox to do everything in one step.

# pf_onestep = PathFinder(
#     dataset_source=alkis_wfs,
#     cost_assumptions=...,
#     datasets_to_modify=...,
#     bbox=...,
#     source_coords=source_custom,
#     target_coords=target_custom,
#     raster_save_path="../tutorial/output/onestep_raster.tiff",
# )
# path_onestep = pf_onestep.find_route()

# YOUR CODE HERE"""
))

# Part 1 Summary

cells.append(md(
"""### Part 1 Summary

| Step | Data source | Effect |
|------|------------|--------|
| Base | ALKIS land use (LGL BW) | Construction costs per terrain |
| +1 | Naturschutzgebiete (LUBW) | Forbidden zones |
| +2 | Vogelschutzgebiete (LUBW) | &times;1.75 surcharge |
| +3 | Wasserschutzgebiete (LUBW) | &times;1.5 surcharge |
| +4 | Landschaftsschutzgebiete (LUBW) | &times;1.25 surcharge |

> **Tip:** Add as many layers as needed (soil conditions, flood zones,
> infrastructure buffers, &hellip;). `modify_raster_from_dataset()` accepts any WFS
> or local vector/raster source. For quick workflows, pass everything to
> `PathFinder` directly.""",

exercise_source=\
"""### Part 1 Summary

| Step | Data source | Effect |
|------|------------|--------|
| Base | ALKIS land use (LGL BW) | Construction costs per terrain |
| +1 | Naturschutzgebiete (LUBW) | Forbidden zones |
| +2 | Vogelschutzgebiete (LUBW) | &times;1.75 surcharge |
| +3 | Wasserschutzgebiete (LUBW) | &times;1.5 surcharge |
| +4 | Landschaftsschutzgebiete (LUBW) | &times;1.25 surcharge |

> **Tip:** Add as many layers as needed. `modify_raster_from_dataset()` accepts any WFS
> or local vector/raster source. For quick workflows, pass everything to
> `PathFinder` directly."""
))

# ==========================================================================
# Part 2: Path Finding with Different Neighborhoods &amp; Search Spaces
# ==========================================================================

cells.append(md(
"""---
# Part 2: Path Finding with Different Neighborhoods &amp; Search Spaces

Now that we have a cost raster, we explore how **neighborhood degree** and
**search space** affect routing results.

## 2.1 Default Route (R2, Full Search Space)

We start with the default settings: **R2** neighborhood (16 directions) and
no search-space restriction."""
))

cells.append(code(
"""pf_default = PathFinder(
    dataset_source=final_raster_path,
    source_coords=source_custom,
    target_coords=target_custom,
)
path_default = pf_default.find_route()
print(path_default)
print(f"\\nLength: {path_default.total_length:.0f} m | Cost: {path_default.total_cost:.0f} EUR "
      f"| Detour: {path_default.total_length / path_default.euclidean_distance:.2f}x")

pf_default.plot_paths()
plt.title("Default Route (R2, full search space)")
plt.tight_layout()
plt.show()""",

exercise_source=\
"""# TODO: Create a PathFinder with the modified raster (final_raster_path)
# and find a route using default settings.

# pf_default = PathFinder(
#     dataset_source=final_raster_path,
#     source_coords=source_custom,
#     target_coords=target_custom,
# )
# path_default = pf_default.find_route()
# print(path_default)

# YOUR CODE HERE"""
))

# 2.2 Neighborhood Strategies (R0&ndash;R3)

cells.append(md(
"""## 2.2 Neighborhood Strategies (R0&ndash;R3)

The **neighborhood degree** controls how many directions the algorithm considers
at each pixel:

| Neighborhood | Directions | Max Angular Error | Description |
|:---:|:---:|:---:|---|
| R0 | 4 | 41.4% | Cardinal only (N, S, E, W) |
| R1 | 8 | 7.6% | + diagonals |
| **R2** | **16** | **1.8%** | **+ knight's moves (default)** |
| R3 | 32 | 1.1% | + extended knight's moves |

Higher neighborhoods produce smoother, more accurate paths but require more computation.
Let's compare all four on our custom raster:""",

exercise_source=\
"""## 2.2 Neighborhood Strategies (R0&ndash;R3)

The **neighborhood degree** controls how many directions the algorithm considers
at each pixel:

| Neighborhood | Directions | Max Angular Error | Description |
|:---:|:---:|:---:|---|
| R0 | 4 | 41.4% | Cardinal only (N, S, E, W) |
| R1 | 8 | 7.6% | + diagonals |
| **R2** | **16** | **1.8%** | **+ knight's moves (default)** |
| R3 | 32 | 1.1% | + extended knight's moves |

Higher neighborhoods produce smoother, more accurate paths but require more computation.
Compare all four on the custom raster:"""
))

cells.append(code(
"""results_nbh = {}
for r in ["r0", "r1", "r2", "r3"]:
    pf_nbh = PathFinder(
        dataset_source=final_raster_path,
        source_coords=source_custom,
        target_coords=target_custom,
        neighborhood_str=r,
    )
    p = pf_nbh.find_route()
    results_nbh[r] = p
    pf_nbh.plot_paths()
    plt.title(f"{r.upper()}: {p.total_length:.0f} m, {p.total_cost:.0f} EUR")
    plt.tight_layout()
    plt.show()

# Summary table
print(f"\\n{'Nbh':<6} {'Directions':<12} {'Length (m)':<12} {'Cost (EUR)':<14}")
print("-" * 44)
for r, p in results_nbh.items():
    n_dirs = {"r0": 4, "r1": 8, "r2": 16, "r3": 32}[r]
    print(f"{r.upper():<6} {n_dirs:<12} {p.total_length:<12.1f} {p.total_cost:<14.1f}")""",

exercise_source=\
"""results_nbh = {}
# TODO: Loop over ["r0", "r1", "r2", "r3"], create a PathFinder for each
# using final_raster_path, source_custom, target_custom.
# Find route, store result, plot.

for r in ["r0", "r1", "r2", "r3"]:
    # pf_nbh = PathFinder(
    #     dataset_source=final_raster_path,
    #     source_coords=source_custom,
    #     target_coords=target_custom,
    #     neighborhood_str=r,
    # )
    # p = pf_nbh.find_route()
    # results_nbh[r] = p
    # pf_nbh.plot_paths()
    # plt.title(f"{r.upper()}: {p.total_length:.0f} m, {p.total_cost:.0f} EUR")
    # plt.tight_layout()
    # plt.show()
    pass  # YOUR CODE HERE"""
))

# Key Takeaway

cells.append(md(
"""### Key Takeaway

- **R0** produces staircase-like paths (only horizontal/vertical movement)
- **R2** (default) offers excellent accuracy with moderate cost
- **R3** provides marginal improvement over R2 at higher computational cost
- For most applications, **R2 is the recommended choice**"""
))

# 2.3 Varying the Search Space

cells.append(md(
"""## 2.3 Varying the Search Space

By default, PYORPS searches the **entire raster**. The `search_space_buffer_m` parameter
restricts the search to a rectangular buffer around source and target. A tighter buffer
speeds up computation but may miss detours around large obstacles.

| Buffer | Effect |
|--------|--------|
| `None` (default) | Full raster &mdash; guaranteed global optimum |
| Large (e.g. 2000&nbsp;m) | Fast, still finds most detours |
| Small (e.g. 500&nbsp;m) | Very fast, but may miss routes around wide obstacles |"""
))

cells.append(code(
"""buffers = [None, 2000, 1000, 500]
results_buf = {}

for buf in buffers:
    label = "Full raster" if buf is None else f"{buf} m buffer"
    pf_buf = PathFinder(
        dataset_source=final_raster_path,
        source_coords=source_custom,
        target_coords=target_custom,
        search_space_buffer_m=buf,
    )
    try:
        p = pf_buf.find_route()
        results_buf[buf] = p
        pf_buf.plot_paths()
        plt.title(f"{label}: {p.total_length:.0f} m, {p.total_cost:.0f} EUR")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        results_buf[buf] = None
        print(f"{label}: {e}")

# Summary table
print(f"\\n{'Buffer':<14} {'Length (m)':<12} {'Cost (EUR)':<14}")
print("-" * 40)
for buf, p in results_buf.items():
    label = "Full raster" if buf is None else f"{buf} m"
    if p is not None:
        print(f"{label:<14} {p.total_length:<12.1f} {p.total_cost:<14.1f}")
    else:
        print(f"{label:<14} {'No path found'}")""",

exercise_source=\
"""buffers = [None, 2000, 1000, 500]
results_buf = {}

# TODO: Loop over buffer values, create PathFinder with search_space_buffer_m=buf,
# find route, plot results. Use try/except to handle cases where no path is found.

for buf in buffers:
    # label = "Full raster" if buf is None else f"{buf} m buffer"
    # pf_buf = PathFinder(
    #     dataset_source=final_raster_path,
    #     source_coords=source_custom,
    #     target_coords=target_custom,
    #     search_space_buffer_m=buf,
    # )
    # p = pf_buf.find_route()
    # results_buf[buf] = p
    pass  # YOUR CODE HERE"""
))

# ==========================================================================
# Part 3: Case Study 1 &mdash; PCC Optimization for Utility-Scale PV Systems
# ==========================================================================

cells.append(md(
"""---
# Part 3: Case Study 1 &mdash; PCC Optimization for Utility-Scale PV Systems

## Scenario

A **1 MVA photovoltaic (PV) system** needs to be connected to the medium-voltage (MV) grid. We need to determine the **optimal point of common coupling (PCC)** considering:

- **Construction costs** (terrain-dependent cable routing)
- **Technical feasibility** (voltage limits, equipment loading)
- **Path quality** (via different neighborhood strategies)

We use the **MV Oberrhein** benchmark network from pandapower."""
))

# 3.1 Load Network &amp; Define PV Site

cells.append(md(
"""## 3.1 Load Network &amp; Define PV Site"""
))

cells.append(code(
"""# Load the MV Oberrhein network
net = pp.networks.mv_oberrhein(
    scenario='generation', separation_by_sub=True, include_substations=True
)[1]

net_crs = "EPSG:4326"
gis_crs = "EPSG:25832"

def get_bus_geoseries(net, net_crs, buses=None):
    if buses is None:
        buses = net.bus.index
    return net.bus.loc[buses].geo.geojson.as_geoseries.to_crs(crs=net_crs)

all_bus_geo = get_bus_geoseries(net, net_crs).to_crs(gis_crs)
trafo_hv_buses = net.bus.loc[
    net.bus.index.isin(net.trafo.hv_bus) & (net.bus.vn_kv == 20)
].index.tolist()
trafo_bus_geo = get_bus_geoseries(net, net_crs, trafo_hv_buses).to_crs(gis_crs)

pv_location = Point(412874, 5362043)
pv_geoseries = gpd.GeoSeries([pv_location], crs=gis_crs)

print(f"Buses: {len(net.bus)} | Trafo stations: {len(trafo_bus_geo)} | "
      f"Lines: {len(net.line)} | Generators: {len(net.sgen)}")

fig, ax = plt.subplots(figsize=(12, 10))
all_bus_geo.plot(ax=ax, color='lightblue', markersize=5, label='MV buses')
trafo_bus_geo.plot(ax=ax, color='blue', markersize=30, marker='s', label='Trafo stations')
pv_geoseries.plot(ax=ax, color='red', markersize=100, marker='*', zorder=5, label='PV plant')
ax.legend(fontsize=11)
ax.set_title("MV Oberrhein: Bus Locations & PV Site")
plt.tight_layout()
plt.show()"""
))

# 3.2 Find Candidate PCCs

cells.append(md(
"""## 3.2 Find Candidate PCCs

Identify the **8 nearest transformer stations** to the PV plant."""
))

cells.append(code(
"""n_candidates = 8
distances = trafo_bus_geo.distance(pv_location).sort_values()
closest_buses = distances.head(n_candidates).index.tolist()
closest_geo = trafo_bus_geo.loc[closest_buses]

for bus_idx in closest_buses:
    print(f"  Bus {bus_idx}: {distances.loc[bus_idx]:.0f} m")""",

exercise_source=\
"""n_candidates = 8
# TODO: Find the 8 closest transformer stations.
# Hint: trafo_bus_geo.distance(pv_location).sort_values().head(n)

# YOUR CODE HERE"""
))

# 3.3 Load the Cost Raster

cells.append(md(
"""## 3.3 Load the Cost Raster

We use a pre-computed raster for the full MV Oberrhein area (same cost assumptions
as Part 1, but covering the larger network footprint)."""
))

cells.append(code(
"""# Pre-computed raster covering the full MV Oberrhein network area
raster_path_oberrhein = "../case_studies/mv_oberrhein/data/raster/mv_oberrhein_new.tiff"

with rasterio.open(raster_path_oberrhein) as src:
    raster_data = src.read(1)
    print(f"Raster: {raster_data.shape}, {abs(src.transform.a):.1f} m/pixel")

step = max(1, max(raster_data.shape) // 2500)
fig, ax = plt.subplots(figsize=(14, 10))
plot_cost_raster(ax, raster_data[::step, ::step],
                 title=f"Cost Raster: MV Oberrhein (downsampled {step}x)")
plt.tight_layout()
plt.show()""",

exercise_source=\
"""# Pre-computed raster covering the full MV Oberrhein network area
raster_path_oberrhein = "../case_studies/mv_oberrhein/data/raster/mv_oberrhein_new.tiff"

with rasterio.open(raster_path_oberrhein) as src:
    raster_data = src.read(1)
    print(f"Raster: {raster_data.shape}, {abs(src.transform.a):.1f} m/pixel")

step = max(1, max(raster_data.shape) // 2500)
fig, ax = plt.subplots(figsize=(14, 10))
ax.imshow(raster_data[::step, ::step], cmap='terrain_r', origin='upper', vmin=0, vmax=500)
ax.set_title(f"Cost Raster: MV Oberrhein (downsampled {step}x)")
plt.tight_layout()
plt.show()"""
))

# 3.4 Find Optimal Routes to All Candidate PCCs

cells.append(md(
"""## 3.4 Find Optimal Routes to All Candidate PCCs

PYORPS supports **single-source, multi-target** routing &mdash; one Dijkstra run
finds paths to all candidates at once."""
))

cells.append(code(
"""pf = PathFinder(
    dataset_source=raster_path_oberrhein,
    source_coords=pv_geoseries,
    target_coords=closest_geo,
    neighborhood_str="r2",
    ignore_max_cost=True,
)
pf.find_route()
all_paths = pf.paths  # PathCollection with all results

for p in all_paths:
    print(f"  Path {p.path_id}: {p.total_length:.0f} m, {p.total_cost:.0f} EUR")

pf.plot_paths()
plt.suptitle("Candidate Routes: PV Plant to Transformer Stations")
plt.tight_layout()
plt.show()""",

exercise_source=\
"""# TODO: Create a PathFinder from PV to all candidate PCCs (single-source, multi-target).
# pf = PathFinder(dataset_source=raster_path_oberrhein, source_coords=pv_geoseries,
#                 target_coords=closest_geo, ...)
# pf.find_route()
# all_paths = pf.paths

# YOUR CODE HERE"""
))

# 3.5 Technical Validation with Power Flow Analysis

cells.append(md(
"""## 3.5 Technical Validation with Power Flow Analysis

For each candidate route: add a cable to the network, run power flow, check voltage
limits and equipment loading."""
))

cells.append(code(
"""# Fresh network copy for PV connection analysis
net_pcc = pp.networks.mv_oberrhein(
    scenario='generation', separation_by_sub=True, include_substations=True
)[1]
net_pcc.ext_grid.vm_pu = 1.04
p_mw, q_mvar = pp.pq_from_cosphi(
    net_pcc.sgen.sn_mva, cosphi=0.90, qmode="underexcited", pmode="gen")
net_pcc.sgen.p_mw = p_mw
net_pcc.sgen.q_mvar = q_mvar
net_pcc.sgen.scaling = 0.7

pv_bus = pp.create_bus(net_pcc, vn_kv=20, name='PV-Site')
pp.create_sgen_from_cosphi(net_pcc, bus=pv_bus, sn_mva=1.0,
                            mode="underexcited", cos_phi=0.90)
net_pcc.sgen.iloc[-1, net_pcc.sgen.columns.get_loc('scaling')] = 0.7
original_sgen_p = net_pcc.sgen.p_mw.values.copy()
line_type = "NA2XS2Y 1x95 RM/25 12/20 kV\""""
))

cells.append(code(
"""# Match each path target to its nearest bus using vectorised distance
target_points = gpd.GeoSeries([Point(p.target) for p in all_paths], crs=gis_crs)
target_buses = [trafo_bus_geo.distance(pt).idxmin() for pt in target_points]
lengths_km = np.array([p.total_length for p in all_paths]) / 1000

# Add ALL candidate lines at once (out of service) using batch API
candidate_line_ids = pp.create_lines(
    net_pcc,
    from_buses=np.repeat(pv_bus, len(target_buses)),
    to_buses=target_buses,
    length_km=lengths_km,
    std_type=line_type,
    in_service=False,  # All out of service initially
    name=[f"PCC_{tb}" for tb in target_buses],
)

# Evaluate each candidate: enable one line at a time, run power flow, record metrics
pcc_results = []

for i, line_id in enumerate(candidate_line_ids):
    net_pcc.line.at[line_id, 'in_service'] = True
    pp.runpp(net_pcc)

    # Recompute mask after runpp (bus count may change with new line)
    sgen_bus_mask = net_pcc.res_bus.index.isin(net_pcc.sgen.bus)

    result = {
        'Target Bus': target_buses[i],
        'Line length [km]': lengths_km[i],
        'V at PV [p.u.]': net_pcc.res_bus.at[pv_bus, 'vm_pu'],
        'Min V [p.u.]': net_pcc.res_bus.vm_pu.min(),
        'Max V [p.u.]': net_pcc.res_bus.vm_pu.max(),
        'Line loading [%]': net_pcc.res_line.at[line_id, 'loading_percent'],
        'Max line loading [%]': net_pcc.res_line.loading_percent.max(),
    }

    # Voltage rise: full generation vs. zero generation
    v_with_gen = net_pcc.res_bus.loc[sgen_bus_mask, 'vm_pu'].values.copy()
    net_pcc.sgen.p_mw = 0
    pp.runpp(net_pcc)
    v_without_gen = net_pcc.res_bus.loc[sgen_bus_mask, 'vm_pu'].values
    result['Voltage rise [p.u.]'] = (v_with_gen - v_without_gen).max()

    # Restore state
    net_pcc.sgen.p_mw = original_sgen_p
    net_pcc.line.at[line_id, 'in_service'] = False

    p = all_paths.all[i]
    result['Path cost [EUR]'] = p.total_cost
    result['Path length [m]'] = p.total_length
    result['Path ID'] = p.path_id
    pcc_results.append(result)

# Clean up: remove all candidate lines
net_pcc.line.drop(candidate_line_ids, inplace=True)

results_df = pd.DataFrame(pcc_results)
print("\\nPCC Evaluation Results:")
print(results_df.to_string(index=False))""",

exercise_source=\
"""# Match each path target to its nearest bus
target_points = gpd.GeoSeries([Point(p.target) for p in all_paths], crs=gis_crs)
target_buses = [trafo_bus_geo.distance(pt).idxmin() for pt in target_points]
lengths_km = np.array([p.total_length for p in all_paths]) / 1000

# TODO: Add ALL candidate lines at once using pp.create_lines() with in_service=False.
# Then iterate: enable one line at a time, run power flow, record metrics, disable it.

# candidate_line_ids = pp.create_lines(
#     net_pcc,
#     from_buses=...,
#     to_buses=...,
#     length_km=...,
#     std_type=...,
#     in_service=False,
# )

# YOUR CODE HERE: Loop over candidate_line_ids,
# enable each, run pp.runpp(), collect results, disable again.

pcc_results = []
# YOUR CODE HERE

if pcc_results:
    results_df = pd.DataFrame(pcc_results)
    print(results_df.to_string(index=False))"""
))

# 3.6 Select the Optimal PCC

cells.append(md(
"""## 3.6 Select the Optimal PCC

Filter for technically valid options (voltage rise &lt; 0.02 p.u., bus voltages
0.95&ndash;1.05 p.u., loading &lt; 50%) and pick the cheapest."""
))

cells.append(code(
"""valid = results_df[
    (results_df['Voltage rise [p.u.]'] < 0.02) &
    (results_df['Max V [p.u.]'] < 1.05) &
    (results_df['Min V [p.u.]'] > 0.95) &
    (results_df['Max line loading [%]'] < 50)
].sort_values('Path cost [EUR]')

print(f"Valid: {len(valid)} / {len(results_df)}")
if len(valid):
    best = valid.iloc[0]
    print(f"\\nBest PCC: Bus {int(best['Target Bus'])}, "
          f"{best['Path length [m]']:.0f} m, {best['Path cost [EUR]']:.0f} EUR")
    print(valid[['Target Bus', 'Path length [m]', 'Path cost [EUR]',
                  'Voltage rise [p.u.]']].to_string(index=False))"""
))

# Case Study 1 Summary

cells.append(md(
"""### Case Study 1 Summary

We demonstrated a complete PCC optimization workflow:
1. **GIS data** provides terrain-aware construction costs
2. **PYORPS** finds optimal cable routes respecting terrain constraints
3. **Power flow analysis** validates technical feasibility
4. The cheapest *technically valid* option is selected

This process replaces manual route planning with data-driven, reproducible analysis."""
))

# ==========================================================================
# Part 4: Case Study 2 &mdash; Automated Grid Extension Planning
# ==========================================================================

cells.append(md(
"""---
# Part 4: Case Study 2 &mdash; Automated Grid Extension Planning

## Scenario

We demonstrate how PYORPS supports **automated grid extension planning** by:

1. **Identifying candidate connections** between MV stations using Delaunay triangulation
2. **Finding optimal cable routes** for each connection using PYORPS
3. **Building a network model** in pandapower
4. **Optimizing the topology** through iterative ring opening
5. **Evaluating cost vs. technical benefit** of each new cable

This addresses a critical gap: early-stage grid planning often neglects specific cable routing, leading to inaccurate cost estimates and suboptimal decisions."""
))

# 4.1 Load Network &amp; Extract Station Locations

cells.append(md(
"""## 4.1 Load Network &amp; Extract Station Locations"""
))

cells.append(code(
"""net_ext = pp.networks.mv_oberrhein(
    scenario='generation', separation_by_sub=True, include_substations=True
)[1]
pp.runpp(net_ext)

trafo_hv_buses_ext = net_ext.bus.loc[
    net_ext.bus.index.isin(net_ext.trafo.hv_bus) & (net_ext.bus.vn_kv == 20)
].index.tolist()
ext_grid_buses = net_ext.ext_grid.bus.tolist()

station_geo = get_bus_geoseries(net_ext, net_crs, trafo_hv_buses_ext).to_crs(gis_crs)
ext_grid_geo = get_bus_geoseries(net_ext, net_crs, ext_grid_buses).to_crs(gis_crs)

print(f"MV stations: {len(station_geo)} | "
      f"Max loading: {net_ext.res_line.loading_percent.max():.1f}% | "
      f"V: [{net_ext.res_bus.vm_pu.min():.4f}, {net_ext.res_bus.vm_pu.max():.4f}] p.u.")"""
))

# 4.2 Apply Delaunay Triangulation

cells.append(md(
"""## 4.2 Apply Delaunay Triangulation

**Why Delaunay?** Connecting every pair of N stations would require N(N-1)/2 cables &mdash; far too many. Delaunay triangulation provides a **natural meshed topology** where each station connects to its geometric neighbors.

Properties of Delaunay triangulation:
- Maximizes the minimum angle of triangles (avoids very acute triangles)
- Every station connects to ~6 neighbors on average
- The resulting graph contains all possible minimum spanning trees
- Produces a reasonable meshed network that can be optimized further"""
))

cells.append(code(
"""# Extract coordinates for triangulation
coords = np.column_stack([station_geo.geometry.x, station_geo.geometry.y])
bus_indices = station_geo.index.values

print(f"Triangulating {len(coords)} station locations...")

# Apply Delaunay triangulation
tri = Delaunay(coords)

# Extract unique edges from the triangulation
edges = set()
for simplex in tri.simplices:
    for i in range(3):
        edge = tuple(sorted([simplex[i], simplex[(i + 1) % 3]]))
        edges.add(edge)

# Convert to bus index pairs
delaunay_connections = [
    (bus_indices[e[0]], bus_indices[e[1]]) for e in edges
]

print(f"Delaunay edges: {len(delaunay_connections)}")
print(f"Average connections per station: {2 * len(delaunay_connections) / len(coords):.1f}")

# Filter out very long connections (optional, e.g., > 5 km)
max_distance = 5000  # meters
filtered_connections = []
for fb, tb in delaunay_connections:
    dist = station_geo.loc[fb].distance(station_geo.loc[tb])
    if dist <= max_distance:
        filtered_connections.append((fb, tb))

print(f"After filtering (< {max_distance/1000:.0f} km): {len(filtered_connections)} connections")""",

exercise_source=\
"""# Extract coordinates for triangulation
coords = np.column_stack([station_geo.geometry.x, station_geo.geometry.y])
bus_indices = station_geo.index.values

print(f"Triangulating {len(coords)} station locations...")

# TODO: Apply Delaunay triangulation using scipy.spatial.Delaunay
# Then extract unique edges from tri.simplices

# tri = Delaunay(...)

# edges = set()
# for simplex in tri.simplices:
#     # Extract edges from each triangle
#     # YOUR CODE HERE
#     pass

# Convert to bus index pairs
# delaunay_connections = [(bus_indices[e[0]], bus_indices[e[1]]) for e in edges]

# YOUR CODE HERE

# Optional: Filter connections longer than 5 km
max_distance = 5000
# YOUR CODE HERE"""
))

cells.append(code(
"""# Visualize filtered Delaunay connections
fig, ax = plt.subplots(figsize=(14, 10))
edge_lines = [LineString([station_geo.loc[fb], station_geo.loc[tb]])
              for fb, tb in filtered_connections]
gpd.GeoSeries(edge_lines, crs=gis_crs).plot(ax=ax, color='steelblue', linewidth=1, alpha=0.6)
station_geo.plot(ax=ax, color='blue', markersize=30, marker='s', zorder=4)
ext_grid_geo.plot(ax=ax, color='red', markersize=80, marker='^', zorder=5)
ax.set_title(f"Delaunay Connections < {max_distance/1000:.0f} km ({len(filtered_connections)} edges)")
plt.tight_layout()
plt.show()"""
))

# 4.3 Find Optimal Cable Routes with PYORPS

cells.append(md(
"""## 4.3 Find Optimal Cable Routes with PYORPS

For each Delaunay edge, we find the **terrain-aware optimal route** using PYORPS. This replaces straight-line distance estimates with realistic, routed cable paths."""
))

cells.append(code(
"""# Build source/target coordinate arrays from filtered connections
source_coords = station_geo.loc[[fb for fb, tb in filtered_connections]]
target_coords = station_geo.loc[[tb for fb, tb in filtered_connections]]

print(f"Routing {len(filtered_connections)} connections in a single PathFinder call...")

pf_route = PathFinder(
    dataset_source=raster_path_oberrhein,
    source_coords=source_coords,
    target_coords=target_coords,
    neighborhood_str="r2",
    search_space_buffer_m=1500,
    ignore_max_cost=True,
)
route_collection = pf_route.find_route(pairwise=True)  # one route per (source, target) pair, not all pairs
routes_gdf = pf_route.create_path_geodataframe()

# Add bus IDs for downstream use
routes_gdf['from_bus'] = [fb for fb, tb in filtered_connections]
routes_gdf['to_bus'] = [tb for fb, tb in filtered_connections]

print(f"Routed: {len(routes_gdf)} / {len(filtered_connections)}")
routes_gdf.head()""",

exercise_source=\
"""# TODO: Create a single PathFinder with all source/target pairs and use pairwise=True (one route per pair).
# Then use create_path_geodataframe() to get a GeoDataFrame of all routes.

# source_coords = station_geo.loc[[fb for fb, tb in filtered_connections]]
# target_coords = station_geo.loc[[tb for fb, tb in filtered_connections]]
#
# pf_route = PathFinder(
#     dataset_source=raster_path_oberrhein,
#     source_coords=source_coords,
#     target_coords=target_coords,
#     neighborhood_str="r2",
#     search_space_buffer_m=1500,
#     ignore_max_cost=True,
# )
# route_collection = pf_route.find_route(pairwise=True)
# routes_gdf = pf_route.create_path_geodataframe()
# routes_gdf['from_bus'] = [fb for fb, tb in filtered_connections]
# routes_gdf['to_bus'] = [tb for fb, tb in filtered_connections]

# YOUR CODE HERE"""
))

cells.append(code(
"""pf_route.plot_paths()
plt.title("PYORPS Cable Routes (Terrain-Aware)")
plt.tight_layout()
plt.show()"""
))

# 4.4 Build a pandapower Network from Routed Cables

cells.append(md(
"""## 4.4 Build a pandapower Network from Routed Cables

We create a new pandapower network where each Delaunay connection is a cable with the **actual routed length** (not straight-line distance)."""
))

cells.append(code(
"""net_new = pp.networks.mv_oberrhein(
    scenario='generation', separation_by_sub=True, include_substations=True
)[1]
new_line_type = "NA2XS2Y 1x240 RM/25 12/20 kV"

# Filter out already-existing connections
existing_edges = set(zip(net_new.line.from_bus, net_new.line.to_bus))
existing_edges |= set(zip(net_new.line.to_bus, net_new.line.from_bus))
new_mask = [(int(r['from_bus']), int(r['to_bus'])) not in existing_edges
            for _, r in routes_gdf.iterrows()]
new_routes = routes_gdf.loc[new_mask].copy()
new_routes['length_km'] = new_routes['path_length'] / 1000

# Batch-create lines + switches
new_line_ids = pp.create_lines(
    net_new, from_buses=new_routes['from_bus'].astype(int).values,
    to_buses=new_routes['to_bus'].astype(int).values,
    length_km=new_routes['length_km'].values, std_type=new_line_type, in_service=True,
    name=[f"New: {fb}-{tb}" for fb, tb in
          zip(new_routes['from_bus'].astype(int), new_routes['to_bus'].astype(int))])
pp.create_switches(net_new, buses=new_routes['from_bus'].astype(int).values,
                   elements=new_line_ids, et='l', closed=True)
new_routes['line_idx'] = new_line_ids
new_lines_df = new_routes

pp.runpp(net_new)
print(f"Added {len(new_lines_df)} cables ({new_lines_df['length_km'].sum():.1f} km, "
      f"{new_lines_df['path_cost'].sum():.0f} EUR)")
print(f"Meshed: max loading {net_new.res_line.loading_percent.max():.1f}%, "
      f"V: [{net_new.res_bus.vm_pu.min():.4f}, {net_new.res_bus.vm_pu.max():.4f}]")"""
))

# 4.5 Ring Opening at Smallest Current

cells.append(md(
"""## 4.5 Ring Opening at Smallest Current

MV distribution networks typically operate in **radial (open-ring) topology**. When new cables create loops, we need to **open the ring** by opening the switch with the **smallest current flow**.

### Algorithm
```
while network has cycles:
    1. Find all cycles in the network
    2. Identify switches within the cycles
    3. Open the switch carrying the smallest current
    4. Re-run power flow
    5. Record metrics (loading, voltage, losses)
```

This heuristic ensures we remove the least-utilized connection in each loop, preserving the most important power flow paths."""
))

cells.append(code(
"""def open_rings_smallest_current(net, verbose=True):
    ""\"
    Iteratively open network rings by opening the switch with the smallest current.

    Returns a list of dictionaries with metrics at each iteration.
    ""\"
    history = []
    iteration = 0

    # Build NetworkX graph from closed switches
    nxg = create_nxgraph(net, multi=False)

    while True:
        cycles = cycle_basis(nxg)
        if not cycles:
            break

        # Get all nodes in cycles
        all_cycle_nodes = list(set(node for cycle in cycles for node in cycle))

        # Find lines in cycles
        cycle_lines = net.line.loc[
            net.line.from_bus.isin(all_cycle_nodes) &
            net.line.to_bus.isin(all_cycle_nodes)
        ].index.values

        # Find closed switches on these lines
        switches = net.switch.loc[
            net.switch.element.isin(cycle_lines) &
            net.switch.bus.isin(all_cycle_nodes) &
            net.switch.closed &
            (net.switch.et == 'l')
        ]

        if switches.empty:
            break

        # Find switch with smallest current
        switch_currents = net.res_switch.loc[switches.index, 'i_ka']
        min_switch = switch_currents.idxmin()
        opened_line = net.switch.at[min_switch, 'element']
        opened_current = switch_currents[min_switch]

        # Open the switch
        net.switch.at[min_switch, 'closed'] = False
        pp.runpp(net)

        # Record metrics
        history.append({
            'iteration': iteration,
            'switch_opened': min_switch,
            'line_opened': opened_line,
            'opened_current_ka': opened_current,
            'cycles_remaining': len(cycles) - 1,
            'max_loading': net.res_line.loading_percent.max(),
            'min_voltage': net.res_bus.vm_pu.min(),
            'max_voltage': net.res_bus.vm_pu.max(),
            'total_losses_mw': net.res_line.pl_mw.sum(),
        })

        if verbose:
            print(f"  Iter {iteration:2d}: Opened switch {min_switch} "
                  f"(line {opened_line}, I={opened_current:.4f} kA) "
                  f"-> max loading {net.res_line.loading_percent.max():.1f}%, "
                  f"V: [{net.res_bus.vm_pu.min():.4f}, {net.res_bus.vm_pu.max():.4f}] p.u.")

        # Rebuild graph
        nxg = create_nxgraph(net, multi=False)
        iteration += 1

    if verbose:
        print(f"\\nRing opening complete after {iteration} iterations. Network is now radial.")

    return pd.DataFrame(history)

# Run ring opening
print("Opening rings at smallest current...")
ring_history = open_rings_smallest_current(net_new)""",

exercise_source=\
"""def open_rings_smallest_current(net, verbose=True):
    ""\"
    Iteratively open network rings by opening the switch with the smallest current.

    TODO: Implement the ring opening algorithm:
    1. Build a NetworkX graph from the network (create_nxgraph)
    2. Find cycles (cycle_basis)
    3. While cycles exist:
       a. Find switches in cycle nodes
       b. Open the switch with smallest current (net.res_switch.i_ka)
       c. Re-run power flow
       d. Record metrics
    ""\"
    history = []
    iteration = 0

    # Build NetworkX graph
    nxg = create_nxgraph(net, multi=False)

    while True:
        cycles = cycle_basis(nxg)
        if not cycles:
            break

        # YOUR CODE HERE:
        # 1. Find all nodes in cycles
        # 2. Find lines connecting those nodes
        # 3. Find closed switches on those lines
        # 4. Open the switch with smallest current
        # 5. Run power flow
        # 6. Record metrics
        # 7. Rebuild graph

        iteration += 1
        break  # Remove this break when implemented

    return pd.DataFrame(history)

# Run ring opening
print("Opening rings at smallest current...")
ring_history = open_rings_smallest_current(net_new)"""
))

# 4.6 Evaluate Cost vs. Technical Benefit

cells.append(md(
"""## 4.6 Evaluate Cost vs. Technical Benefit

Now we compare the **baseline network** (without new cables) against the **optimized network** (with new cables and optimized ring topology)."""
))

cells.append(code(
"""# Compare baseline vs. optimized
baseline_net = pp.networks.mv_oberrhein(
    scenario='generation', separation_by_sub=True, include_substations=True
)[1]
pp.runpp(baseline_net)

print("=" * 70)
print("COMPARISON: Baseline vs. Optimized Network")
print("=" * 70)
print(f"{'Metric':<30} {'Baseline':<20} {'Optimized':<20} {'Change':<15}")
print("-" * 70)

metrics = [
    ('Max line loading [%]',
     baseline_net.res_line.loading_percent.max(),
     net_new.res_line.loading_percent.max()),
    ('Min voltage [p.u.]',
     baseline_net.res_bus.vm_pu.min(),
     net_new.res_bus.vm_pu.min()),
    ('Max voltage [p.u.]',
     baseline_net.res_bus.vm_pu.max(),
     net_new.res_bus.vm_pu.max()),
    ('Voltage range [p.u.]',
     baseline_net.res_bus.vm_pu.max() - baseline_net.res_bus.vm_pu.min(),
     net_new.res_bus.vm_pu.max() - net_new.res_bus.vm_pu.min()),
    ('Total losses [kW]',
     baseline_net.res_line.pl_mw.sum() * 1000,
     net_new.res_line.pl_mw.sum() * 1000),
]

for name, base_val, opt_val in metrics:
    change = opt_val - base_val
    pct = (change / base_val * 100) if base_val != 0 else 0
    sign = "+" if change > 0 else ""
    print(f"{name:<30} {base_val:<20.4f} {opt_val:<20.4f} {sign}{change:.4f} ({sign}{pct:.1f}%)")

print(f"\\nNew infrastructure cost: {new_lines_df['path_cost'].sum():.0f} EUR")
print(f"New cable length:        {new_lines_df['length_km'].sum():.1f} km")"""
))

cells.append(code(
"""if len(ring_history) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].plot(ring_history['iteration'], ring_history['max_loading'], 'b-o', ms=4)
    axes[0].axhline(y=baseline_net.res_line.loading_percent.max(), color='r', ls='--', label='Baseline')
    axes[0].set(xlabel="Iteration", ylabel="Max Loading [%]", title="Line Loading")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].fill_between(ring_history['iteration'], ring_history['min_voltage'],
                          ring_history['max_voltage'], alpha=0.3)
    axes[1].axhline(y=0.95, color='grey', ls=':'); axes[1].axhline(y=1.05, color='grey', ls=':')
    axes[1].set(xlabel="Iteration", ylabel="Voltage [p.u.]", title="Voltage Band")
    axes[1].grid(True, alpha=0.3)
    plt.suptitle("Ring Opening Progress")
    plt.tight_layout()
    plt.show()"""
))

cells.append(code(
"""# Per-cable cost-benefit analysis
line_ids = new_lines_df['line_idx'].values
sw_mask = net_new.switch.element.isin(line_ids) & (net_new.switch.et == 'l')
sw_status = net_new.switch.loc[sw_mask].set_index('element')['closed']

analysis = new_lines_df[['from_bus', 'to_bus', 'length_km', 'path_cost']].copy()
analysis.columns = ['From', 'To', 'Length [km]', 'Cost [EUR]']
analysis['Active'] = [sw_status.get(new_lines_df.at[i, 'line_idx'], False) for i in analysis.index]

active_cost = analysis.loc[analysis['Active'], 'Cost [EUR]'].sum()
total_cost = analysis['Cost [EUR]'].sum()
print(f"Active cables: {analysis['Active'].sum()} | "
      f"Opened during ring opening: {(~analysis['Active']).sum()}")
print(f"Active cable cost: {active_cost:.0f} EUR | Total: {total_cost:.0f} EUR | "
      f"Savings: {total_cost - active_cost:.0f} EUR")"""
))

# ==========================================================================
# Part 5: Summary & Next Steps
# ==========================================================================

cells.append(md(
"""---
# Part 5: Summary & Next Steps

## What We Learned

| Topic | Key Takeaway |
|-------|-------------|
| **PYORPS fundamentals** | Raster-based least-cost path analysis with configurable neighborhoods (R0&ndash;R3) |
| **Cost raster creation** | Build multi-layer cost rasters from open WFS geodata (land use, protection zones, soil) |
| **Cost assumptions** | Land-use categories mapped to construction costs; 65535 = forbidden |
| **PCC optimization** | Combine PYORPS routing with power flow analysis for technically valid, cost-optimal connections |
| **Grid extension** | Delaunay triangulation + PYORPS + ring opening = automated meshed-to-radial planning |
| **Routing accuracy** | R2 neighborhood offers excellent accuracy with moderate computation |
| **Cost vs. straight-line** | Routed paths are typically 1.1&ndash;1.5x longer than straight-line estimates |

## Advanced Features (Not Covered Today)

- **3D routing** with Digital Elevation Models (DEM)
- **Constrained routing** with turn angles and tower placement (overhead lines)
- **GPU acceleration** for large-scale problems
- **Custom preprocessing** for refined terrain classification (road buffers, soil conditions)

## Resources

- **GitHub**: [github.com/<user>/pyorps](https://github.com/<user>/pyorps)
- **Documentation**: [pyorps.readthedocs.io](https://pyorps.readthedocs.io)
- **Publication**: *PYORPS: an open-source tool for automated power line routing* (CIRED 2025)

## Contact

**Martin Hofmann M.Sc.**
martin.hofmann-3@ei.thm.de
THM &mdash; University of Applied Sciences, Giessen, Germany"""
))

cells.append(md(
"""---
*This tutorial notebook was created for the 4th International Workshop on Open Source Modelling and Simulation of Energy Systems (OSMSES 2026), Karlsruhe, Germany.*"""
))


# ---------------------------------------------------------------------------
# Build notebooks
# ---------------------------------------------------------------------------

def cell_text(cell):
    """The source of a cell as one string (a notebook stores it as a string or as a list of lines)."""
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def build_notebook(cells_list, is_exercise=False):
    """Build a notebook dict from cell definitions (outputs empty, execution counts None)."""
    ids = CELL_IDS["exercises" if is_exercise else "full"]
    nb_cells = []
    for index, cell in enumerate(cells_list):
        c = copy.deepcopy(cell)
        exercise_source = c.pop("_exercise_source", None)
        text = exercise_source if is_exercise and exercise_source is not None else c["source"]
        # nbformat stores the source as one string per line, each keeping its newline.
        c["source"] = text.splitlines(keepends=True)
        c["id"] = ids[index] if index < len(ids) else f"cell-{index:03d}"
        nb_cells.append(c)

    all_ids = [c["id"] for c in nb_cells]
    if len(set(all_ids)) != len(all_ids):
        duplicates = sorted({i for i in all_ids if all_ids.count(i) > 1})
        raise ValueError(f"cell ids must be unique, repeated: {duplicates}")

    return {
        "cells": nb_cells,
        "metadata": copy.deepcopy(NOTEBOOK_METADATA),
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def render_notebook(nb):
    """The JSON text nbformat writes: sorted keys, indent 1, non-ASCII characters kept, final newline."""
    return json.dumps(nb, indent=1, sort_keys=True, ensure_ascii=False, separators=(",", ": ")) + "\n"


def claimed_python(metadata):
    """(major, minor) of the Python that ``language_info`` claims, or None when it names no version."""
    version = (metadata.get("language_info") or {}).get("version")
    if not version:
        return None
    try:
        return tuple(int(part) for part in str(version).split(".")[:2])
    except ValueError:
        return (0, 0)


def keep_execution_state(nb, previous):
    """Copy outputs, execution counts and cell metadata of unchanged cells from the notebook being replaced.

    A cell is unchanged when a cell of the same type with the same source exists in ``previous``: the one at the same
    position if it matches, otherwise the first unused one that does. The notebook metadata is taken over when it
    comes from a real kernel (a Python version not older than MIN_PYTHON). Returns the number of code cells whose
    outputs were kept.
    """
    old_cells = previous.get("cells", [])
    taken = {}
    for index, cell in enumerate(nb["cells"]):
        if index < len(old_cells) and (old_cells[index].get("cell_type"), cell_text(old_cells[index])) == (
                cell["cell_type"], cell_text(cell)):
            taken[index] = index
    used = set(taken.values())
    for index, cell in enumerate(nb["cells"]):
        if index in taken:
            continue
        for old_index, old in enumerate(old_cells):
            if old_index not in used and (old.get("cell_type"), cell_text(old)) == (cell["cell_type"], cell_text(cell)):
                taken[index] = old_index
                used.add(old_index)
                break

    kept = 0
    for index, old_index in taken.items():
        cell, old = nb["cells"][index], old_cells[old_index]
        cell["metadata"] = copy.deepcopy(old.get("metadata", {}))
        if cell["cell_type"] == "code":
            cell["outputs"] = copy.deepcopy(old.get("outputs", []))
            cell["execution_count"] = old.get("execution_count")
            kept += bool(cell["outputs"])

    old_language = (previous.get("metadata") or {}).get("language_info")
    old_version = claimed_python(previous.get("metadata") or {})
    if old_language and old_version is not None and old_version >= MIN_PYTHON:
        nb["metadata"]["language_info"] = copy.deepcopy(old_language)
    return kept


def compare_notebooks(label, generated, existing):
    """Readable differences between the generated notebook and an existing one (an empty list: they agree).

    Compared: nbformat version, kernelspec, the Python version in language_info (it must not be older than MIN_PYTHON),
    and per cell the type, the id and the source text. Outputs and execution counts are not compared.
    """
    found = []
    for key in ("nbformat", "nbformat_minor"):
        if existing.get(key) != generated[key]:
            found.append(f"{label}: {key} is {existing.get(key)!r}, the script generates {generated[key]!r}")
    metadata = existing.get("metadata") or {}
    if metadata.get("kernelspec") != generated["metadata"]["kernelspec"]:
        found.append(f"{label}: kernelspec is {metadata.get('kernelspec')!r}, "
                     f"the script generates {generated['metadata']['kernelspec']!r}")
    claimed = claimed_python(metadata)
    if claimed is not None and claimed < MIN_PYTHON:
        version = metadata["language_info"]["version"]
        found.append(f"{label}: language_info.version is {version!r}, older than the project minimum "
                     f"{'.'.join(map(str, MIN_PYTHON))}")

    old_cells, new_cells = existing.get("cells", []), generated["cells"]
    if len(old_cells) != len(new_cells):
        found.append(f"{label}: the notebook has {len(old_cells)} cells, the script generates {len(new_cells)}")
    for index, (old, new) in enumerate(zip(old_cells, new_cells)):
        problems = []
        if old.get("cell_type") != new["cell_type"]:
            problems.append(f"type {old.get('cell_type')!r}, generated {new['cell_type']!r}")
        if old.get("id") != new["id"]:
            problems.append(f"id {old.get('id')!r}, generated {new['id']!r}")
        old_text, new_text = cell_text(old), cell_text(new)
        if old_text != new_text:
            problems.append("source differs")
        if not problems:
            continue
        found.append(f"{label}: cell {index} ({new['cell_type']}, id {old.get('id')}): " + "; ".join(problems))
        if old_text != new_text:
            diff = [line for line in difflib.unified_diff(old_text.splitlines(), new_text.splitlines(),
                                                           "notebook", "generated", lineterm="", n=0)
                    if not line.startswith(("---", "+++"))]
            shown = diff[:8]
            found.extend(f"      {line[:110]}" for line in shown)
            if len(diff) > len(shown):
                found.append(f"      ... {len(diff) - len(shown)} more diff lines")
            if not diff:
                found.append("      (the text differs in line endings or trailing newlines only)")
    return found


def read_notebook(path):
    """The parsed notebook at ``path``, or None when the file is missing or is not JSON."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build the two OSMSES 2026 tutorial notebooks (full and exercises) from this script.")
    parser.add_argument("--out-dir", type=Path, default=HERE, metavar="DIR",
                        help="folder to write the notebooks to (and, with --check, to compare against); relative "
                             "paths are relative to the current directory; default: next to this script")
    parser.add_argument("--check", action="store_true",
                        help="write nothing; compare what would be generated with the notebooks in the folder, print "
                             "the differences per cell and exit 1 if there are any")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = args.out_dir
    targets = (("Full notebook", FULL_NAME, False), ("Exercise notebook", EXERCISES_NAME, True))

    if args.check:
        differences = []
        for label, name, is_exercise in targets:
            existing = read_notebook(out_dir / name)
            if existing is None:
                differences.append(f"{label}: {out_dir / name} is missing or is not valid JSON")
                continue
            differences += compare_notebooks(label, build_notebook(cells, is_exercise), existing)
        if differences:
            print("\n".join(differences))
            print(f"The notebooks in {out_dir} differ from {Path(__file__).name}.")
            return 1
        print(f"OK: both notebooks in {out_dir} match {Path(__file__).name} (outputs are not compared).")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    for label, name, is_exercise in targets:
        path = out_dir / name
        nb = build_notebook(cells, is_exercise)
        kept = 0
        newline = "\n"
        if path.exists():
            old_bytes = path.read_bytes()
            newline = "\r\n" if b"\r\n" in old_bytes[:4096] else "\n"
            previous = read_notebook(path)
            if previous is None:
                print(f"Warning: {path} is not valid JSON; it is replaced without keeping outputs.", file=sys.stderr)
            else:
                kept = keep_execution_state(nb, previous)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(render_notebook(nb).replace("\n", newline))
        print(f"{label + ':':<19}{path}  (kept the outputs of {kept} code cells)")

    print("Done!")
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):  # a diff may hold characters that the console encoding lacks
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    sys.exit(main())
