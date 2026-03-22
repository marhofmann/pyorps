# 💰 Cost Assumptions

Cost assumptions map geospatial features (land use categories, protection zones, etc.) to numeric cost values. These values drive the least-cost path algorithm: lower cost means the route prefers that area, higher cost means it avoids it, and `65535` marks a cell as forbidden/impassable.

:::{note}
Cost values use `uint16` representation, so valid values range from `0` to `65535`. The value `65535` is reserved for impassable cells.
:::

---

## Dictionary Input

The most direct way to define costs is with a Python dictionary. The top-level key is the feature column name; the values map feature categories to costs.

**Simple mapping** (one level):

```python
costs = {
    "land_use": {
        "Forest": 365,
        "Agriculture": 107,
        "Residential": 65535,  # Forbidden
        "Water": 65535,        # Forbidden
        "Industrial": 800,
    }
}

pf = PathFinder(
    dataset_source="landuse.shp",
    cost_assumptions=costs,
    source_coords=source,
    target_coords=target,
)
```

**Nested mapping** (main feature + side feature):

```python
costs = {
    "land_use": {
        "Forest": {"Coniferous": 365, "Mixed Deciduous": 402, "Broadleaf": 420},
        "Agriculture": {"Arable land": 107, "Grassland": 130},
        "Residential": 65535,
    }
}
```

When side features are present, an empty string `""` acts as a wildcard default for unmatched sub-categories.

---

## File-Based Input

Cost assumptions can be loaded from CSV, Excel, or JSON files.

### CSV

```python
pf = PathFinder(
    dataset_source="landuse.shp",
    cost_assumptions="costs.csv",
    source_coords=source,
    target_coords=target,
)
```

The CSV file should contain feature columns and one numeric cost column. PYORPS auto-detects the delimiter, decimal separator, and encoding. Example CSV structure:

| land_use | category | cost |
|----------|----------|------|
| Forest | Coniferous | 365 |
| Forest | Mixed Deciduous | 402 |
| Agriculture | Arable land | 107 |
| Agriculture | Grassland | 130 |
| Residential | | 65535 |

The first non-numeric column becomes the `main_feature`, additional non-numeric columns become `side_features`, and the first numeric column is used as the cost.

### Excel

```python
pf = PathFinder(
    dataset_source="landuse.shp",
    cost_assumptions="costs.xlsx",
    ...
)
```

Same tabular structure as CSV. The `.xlsx` and `.xls` formats are supported.

### JSON

```python
pf = PathFinder(
    dataset_source="landuse.shp",
    cost_assumptions="costs.json",
    ...
)
```

JSON files can use a structured format with metadata:

```json
{
  "metadata": {
    "main_feature": "land_use",
    "side_features": ["category"]
  },
  "cost_assumptions": {
    "Forest__Coniferous": 365,
    "Forest__Mixed Deciduous": 402,
    "Agriculture__Arable land": 107
  }
}
```

Or a plain dictionary (legacy format) without the metadata wrapper.

---

## Auto-Detection and Template Generation

When working with unfamiliar vector data, PYORPS can analyze the attribute columns and suggest which ones to use as features. It can also generate a template file with zero costs that you fill in.

```python
from pyorps import detect_feature_columns, save_empty_cost_assumptions
from pyorps.io.geo_dataset import initialize_geo_dataset

# Load the dataset
dataset = initialize_geo_dataset("landuse.shp")
dataset.load_data()

# Detect suitable feature columns
main_feature, side_features = detect_feature_columns(dataset.data)
print(f"Main feature: {main_feature}")
print(f"Side features: {side_features}")

# Generate a template with zero costs for all feature combinations
save_empty_cost_assumptions(dataset, "template.csv", file_type="csv")
```

The `detect_feature_columns()` function uses statistical analysis (Shannon entropy, area coverage, cross-tabulation) to rank candidate columns and select the most informative ones.

---

## Cost Modifiers (`datasets_to_modify`)

After the base raster is created, additional datasets can overlay or modify cost values. This is useful for protection zones, buffer areas, or other constraints that add cost on top of the base land-use costs.

Pass a list of modifier dictionaries to `PathFinder`:

```python
datasets_to_modify = [
    # Fixed value: set all cells inside nature reserves to 1200
    {
        "input_data": "nature_reserve.shp",
        "cost_assumptions": 1200,
    },
    # Multiplier: multiply existing costs by 1.1 in landscape protection areas
    {
        "input_data": "landscape_protection.shp",
        "cost_assumptions": 1.1,
        "multiply": True,
    },
    # Per-zone costs: different values for different zones
    {
        "input_data": "water_protection.shp",
        "cost_assumptions": {"ZONE": {"Zone I": 65535, "Zone II": 5000}},
    },
    # With geometry buffer: expand protection zones by 20m before applying
    {
        "input_data": "nature_reserves.gpkg",
        "cost_assumptions": 1200,
        "geometry_buffer_m": 20,
    },
]

pf = PathFinder(
    dataset_source="landuse.shp",
    cost_assumptions="base_costs.csv",
    datasets_to_modify=datasets_to_modify,
    source_coords=source,
    target_coords=target,
)
```

Each modifier dictionary supports the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `input_data` | str, GeoDataFrame | Path to vector file or in-memory GeoDataFrame |
| `cost_assumptions` | int, float, str, dict | Cost value, multiplier, file path, or dictionary |
| `multiply` | bool | If `True`, multiply existing costs instead of replacing |
| `geometry_buffer_m` | float | Buffer geometries by this distance (in CRS units) before applying |
| `ignore_value` | int | Raster value to skip during modification (default: `65535`) |

:::{tip}
Modifiers are applied in order. Place broad, low-impact modifiers first and specific, high-impact modifiers last so that the final raster reflects the strictest constraints.
:::

---

## Saving and Exporting Cost Assumptions

The `CostAssumptions` class can export its contents to different formats:

```python
from pyorps import CostAssumptions

ca = CostAssumptions(source="costs.csv")

# Export to different formats
ca.to_csv("costs_export.csv", separator=";", decimal=".")
ca.to_json("costs_export.json", indent=2)
ca.to_excel("costs_export.xlsx", sheet_name="Costs")
```
