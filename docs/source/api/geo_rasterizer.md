---
title: "GeoRasterizer"
summary: "Turn vector layers and cost assumptions into a cost raster."
status: stable
since: "0.2.1"
available_in: pypi
module: "pyorps.raster.rasterizer"
api:
  - pyorps.GeoRasterizer
---
# 🗺️ Rasterization

Rasterization converts vector geodata (polygons, lines) into a cost raster that the routing algorithms operate on. The `GeoRasterizer` class handles this conversion, applying cost assumptions to map feature attributes to numeric cell values.

```{image} ../_static/generated/rasterization_pipeline.png
:alt: Rasterization pipeline
:class: only-light
:width: 100%
```

---

(geo-rasterizer-basic-workflow)=
## Basic Workflow

```python
from pyorps.io.geo_dataset import initialize_geo_dataset
from pyorps import GeoRasterizer, CostAssumptions

# 1. Load vector data
dataset = initialize_geo_dataset("landuse.shp")
dataset.load_data()

# 2. Create rasterizer with cost assumptions
rasterizer = GeoRasterizer(
    input_data=dataset,
    cost_assumptions="costs.csv",
)

# 3. Rasterize
rasterizer.rasterize(resolution_in_m=1.0)

# 4. Save the result
rasterizer.save_raster("cost_raster.tiff")
```

:::{note}
When you pass vector data directly to `PathFinder`, rasterization happens automatically behind the scenes. Use `GeoRasterizer` directly when you need fine-grained control over the rasterization process or want to reuse the same raster across multiple routing runs.
:::

---

(geo-rasterizer-resolution-control)=
## Resolution Control

The `resolution_in_m` parameter controls the spatial resolution of the output raster in meters per pixel. Lower values produce finer-grained rasters but increase memory usage and computation time.

```python
# 1m resolution (high detail, large raster)
rasterizer.rasterize(resolution_in_m=1.0)

# 5m resolution (lower detail, smaller raster)
rasterizer.rasterize(resolution_in_m=5.0)
```

If the input CRS uses geographic coordinates (degrees), the rasterizer auto-reprojects to a suitable UTM CRS for accurate metric calculations.

---

(geo-rasterizer-geometry-buffering)=
## Geometry Buffering

Vector geometries can be expanded before rasterization using the `geometry_buffer_m` parameter. This is useful for linear features (roads, rivers) that need a spatial extent.

```python
rasterizer.rasterize(
    resolution_in_m=1.0,
    geometry_buffer_m=10,  # Expand all geometries by 10 meters
)
```

The buffer is applied in the dataset's CRS units (typically meters for projected CRS).

---

(geo-rasterizer-forbidden-features-narrower-than-a-cell)=
## Forbidden Features Narrower Than a Cell

A cell is normally burned only if its **centre** falls inside the polygon. A
forbidden feature narrower than one cell can therefore slip between the cell
centres and disappear from the raster entirely — measured: a 0.8 m barrier on a
1 m grid burned 60 cells at one sub-pixel alignment and **zero cells at 3 of 7**
alignments tested.

**By default `rasterize()` repairs nothing.** The burned array is bit-identical
to a plain GDAL burn, and PYORPS *warns* that a forbidden feature did not
survive. The fix that is correct rather than a trade is **a finer cell size** —
`suggest_resolution()` computes it, and reports what it would cost in cells.

```python
rasterizer.rasterize(
    resolution_in_m=1.0,
    widen_thin_forbidden=False,   # default: opt-in repair, widens sub-cell forbidden geometry
    all_touched=False,            # default: opt-in repair, forbidden features grow to every touched cell
    on_thin_features="warn",      # 'warn' | 'raise' | 'ignore'
)
```

:::{danger}
Both repairs also **seal legitimate sub-cell openings** — a gate, a culvert, a
gap between parcels — because at a fixed cell size a sub-cell barrier and a
sub-cell gap are the same geometry seen from opposite sides. Swept over 8 gate
widths × 8 sub-pixel offsets under **four different gate-width sets**, counting
only configurations the plain rule leaves passable: in a 0.4 m wall
`widen_thin_forbidden=True` sealed **38–100 %** of them and `all_touched=True`
**31–100 %** — indistinguishable. In a 2 m wall widening seals **0 %** under
every set (it never touches a fat feature) and `all_touched=True` **17–100 %**.
The percentages move with the gate widths swept, so only the structure
transfers; the table and the exact widths are in
{doc}`thin_forbidden_features`. Turning one on can turn a solvable problem into
`NoPathFoundError`, so PYORPS emits a `SealedOpeningWarning` when a repair
disconnects free space that the plain burn left connected.
:::

See {doc}`thin_forbidden_features` for the full explanation, the five available
options and their effects, and the measured overhead.

---

(geo-rasterizer-fill-value)=
## Fill Value

Cells that fall outside all vector geometries receive the `fill_value`. By default, this is `65535` (impassable), meaning areas not covered by any input feature are treated as forbidden.

```python
rasterizer.rasterize(
    resolution_in_m=1.0,
    fill_value=65535,  # Default: uncovered areas are impassable
)
```

---

(geo-rasterizer-multi-layer-rasterization)=
## Multi-Layer Rasterization

After creating the base raster, additional vector datasets can overlay or modify cell values using `modify_raster_from_dataset()`. This allows you to build up a cost raster from multiple data sources.

```python
# Start with base land-use raster
rasterizer.rasterize(resolution_in_m=1.0)

# Overlay nature reserves (set to fixed cost)
rasterizer.modify_raster_from_dataset(
    input_data="nature_reserves.gpkg",
    cost_assumptions=1200,
    geometry_buffer_m=20,
)

# Overlay landscape protection (multiply existing costs)
rasterizer.modify_raster_from_dataset(
    input_data="landscape_protection.shp",
    cost_assumptions=1.1,
    multiply=True,
)

# Overlay water protection zones (per-zone costs via dict)
rasterizer.modify_raster_from_dataset(
    input_data="water_protection.shp",
    cost_assumptions={"ZONE": {"Zone I": 65535, "Zone II": 5000}},
)

# Save the combined raster
rasterizer.save_raster("combined_cost_raster.tiff")
```

:::{tip}
The `ignore_value` parameter (default: `65535`) prevents modification of cells that are already marked as impassable. Set `ignore_value=None` to modify all cells regardless of their current value.
:::

:::{warning}
Overlays applied with `modify_raster_from_dataset()` use the pixel-centre rule
and are **not** covered by the all-touched treatment that `rasterize()` gives
forbidden features. A forbidden zone narrower than one cell that is stamped in
this way can still vanish silently. Either include such a layer in the base
burn, or give it a `geometry_buffer_m` of at least
`safe_forbidden_width_m(resolution) / 2`. See
{doc}`thin_forbidden_features`.
:::

---

(geo-rasterizer-clipping)=
## Clipping

Restrict the raster to a specific geometry using `clip_to_area()`:

```python
from shapely.geometry import box

study_area = box(460000, 5580000, 480000, 5600000)
rasterizer.clip_to_area(study_area)
```

The `shrink_raster()` method removes outer rows and columns that contain only a specific value (e.g., impassable border cells), reducing the raster size:

```python
rasterizer.shrink_raster(exclude_value=65535)
```

---

(geo-rasterizer-full-pipeline-example)=
## Full Pipeline Example

A complete workflow from WFS download through rasterization to routing:

```python
from pyorps import PathFinder, GeoRasterizer
from pyorps.io.geo_dataset import initialize_geo_dataset
from pyorps.core.cost_assumptions import (
    detect_feature_columns, save_empty_cost_assumptions,
)

# 1. Download vector data from WFS
wfs_source = {
    "url": "https://example.com/wfs",
    "layer": "landuse_layer",
}
dataset = initialize_geo_dataset(
    wfs_source,
    bbox=(460000, 5580000, 480000, 5600000),
)
dataset.load_data()

# 2. Detect feature columns and create a cost template
main_feature, side_features = detect_feature_columns(dataset.data)
save_empty_cost_assumptions(dataset, "cost_template.csv", file_type="csv")
# >>> Edit cost_template.csv to assign cost values, then reload <<<

# 3. Rasterize with the filled-in cost assumptions
rasterizer = GeoRasterizer(
    input_data=dataset,
    cost_assumptions="cost_template.csv",
)
rasterizer.rasterize(resolution_in_m=1.0, save_path="cost_raster.tiff")

# 4. Route on the raster
pf = PathFinder(
    dataset_source="cost_raster.tiff",
    source_coords=(465000, 5585000),
    target_coords=(475000, 5595000),
    search_space_buffer_m=600,
)
result = pf.find_route()
```

---

(geo-rasterizer-building-comprehensive-cost-rasters-from-multipl)=
## Building Comprehensive Cost Rasters from Multiple Data Sources

Real-world infrastructure planning requires integrating multiple geospatial datasets into a single cost raster. Each layer adds domain-specific information — land use categories, environmental protection zones, soil conditions, and regulatory constraints — to produce a comprehensive cost surface.

The example below follows the workflow from `examples/prepare_data_for_distribution_grid_planning.py`, which creates a cost raster for distribution grid planning in Hessen, Germany.

### Step 1: Base Land Use Raster

Start with a land use dataset (here from a WFS service) and assign construction cost values per land use category:

```python
from pyorps import (initialize_geo_dataset, GeoRasterizer,
                    CostAssumptions, detect_feature_columns)

# Download land use data from WFS
base_wfs = {
    "url": "https://www.gds.hessen.de/wfs2/aaa-suite/cgi-bin/alkis/vereinf/wfs",
    "layer": "ave_Nutzung",
}
bbox = (460000, 5590000, 490000, 5610000)
dataset = initialize_geo_dataset(base_wfs, bbox=bbox)
dataset.load_data()

# Define cost assumptions per land use category (EUR/m)
land_use_costs = {
    ('nutzart', 'bez'): {
        "Wald":          {"Nadelholz": 405, "Laubholz": 475, "": 405},
        "Landwirtschaft": {"Ackerland": 437, "Grünland": 437, "": 437},
        "Straßenverkehr": {"Autobahn": 750, "Bundesstr.": 500, "": 340},
        "Weg":            {"": 300},
        "Bahnverkehr":    {"": 800},
        "Wohnbaufläche":  {"": 65535},  # Forbidden
        "Friedhof":       {"": 65535},
        # ... more categories
    }
}

# Rasterize
rasterizer = GeoRasterizer(dataset, CostAssumptions(land_use_costs))
rasterizer.rasterize(save_path="base_raster.tiff")
```

This produces a raster with ~23 distinct cost values representing different terrain types.

```{image} ../_static/generated/overlay_step1_base.png
:width: 100%
:alt: Step 1 — Base land use cost raster
```

### Step 2: Overlay Drinking Water Protection Zones

Water protection zones require special construction techniques. Apply zone-specific **multipliers** to increase costs in protected areas:

```python
water_protection_wfs = {
    "url": "https://geodienste-umwelt.hessen.de/.../WFSServer",
    "layer": "TWS_HQS_TK25",
}

water_multipliers = {
    "ZONE": {
        1: 100,   # Core zone: effectively forbidden (cost x100)
        2: 2.0,   # Inner zone: double the base cost
        3: 1.5,   # Outer zone: +50%
        4: 1.2,   # Extended zone: +20%
    }
}

rasterizer.modify_raster_from_dataset(
    input_data=water_protection_wfs,
    cost_assumptions=CostAssumptions(water_multipliers),
    multiply=True,  # Multiply with existing cell values
)
```

### Step 3: Overlay Soil Conditions

Different soil types affect excavation difficulty. Apply cost factors based on DIN 18300 soil classification:

```python
soil_wfs = {
    "url": "https://geodienste-umwelt.hessen.de/.../WFSServer",
    "layer": "Bodeneinheiten_Bodenuebersicht_500000",
}

soil_factors = {
    "AUSGANGSGESTEIN": {
        "Lösslehm, Löss": 1.0,             # Easy excavation
        "Schluff- und Tonsteine": 1.05,      # Standard + effort
        "Sandsteine": 1.15,                  # Soft rock
        "Basalt, Basalttuff": 1.3,           # Hard rock — special equipment
    }
}

rasterizer.modify_raster_from_dataset(
    input_data=soil_wfs,
    cost_assumptions=CostAssumptions(soil_factors),
    multiply=True,
)
```

### Step 4: Overlay Nature and Landscape Protection

- **Nature reserves**: mark as forbidden (override with 65535)
- **Landscape protection**: moderate cost increase (multiply by 1.25)

```python
# Nature reserves — forbidden
rasterizer.modify_raster_from_dataset(
    input_data={"url": "...", "layer": "Naturschutzgebiete"},
    cost_assumptions=65535,
    multiply=False,  # Override cell values
)

# Landscape protection — +25%
rasterizer.modify_raster_from_dataset(
    input_data={"url": "...", "layer": "Landschaftsschutzgebiete"},
    cost_assumptions=1.25,
    multiply=True,
)
```

```{image} ../_static/generated/overlay_step2_modified.png
:width: 100%
:alt: Step 2 — Final cost raster with all overlays applied
```

### Step 5: Save and Route

```python
rasterizer.save_raster("comprehensive_cost_raster.tiff")

pf = PathFinder(
    dataset_source="comprehensive_cost_raster.tiff",
    source_coords=source, target_coords=target,
)
pf.find_route()
```

### How Overlays Combine

When `multiply=True`, cost factors stack multiplicatively. A cell in a water protection zone 2 (`x2.0`) on hard rock (`x1.3`) in a landscape protection area (`x1.25`) gets:

```
Final cost = Base cost × 2.0 × 1.3 × 1.25
```

For a base cost of 437 (agriculture): `437 × 2.0 × 1.3 × 1.25 = 1,420 EUR/m`

When `multiply=False`, the value is overridden entirely — used for forbidden zones like nature reserves (`65535`).
