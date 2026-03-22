# 🎨 Visualization

PYORPS provides built-in plotting capabilities to visualize computed routes overlaid on the cost raster. The `plot_paths()` method supports customization of colors, markers, layout, and more.

## Quick Plot

After computing a route, call `plot_paths()` to display it:

```{code-block} python
path_finder.find_route()
path_finder.plot_paths()
```

This renders the route on top of the cost raster with default styling: green source marker, red target marker, and an automatically chosen path color.

## Customization

Adjust colors, markers, line width, and other visual properties:

```{code-block} python
path_finder.plot_paths(
    source_color='blue',
    target_color='red',
    path_colors=['green', 'orange'],
    path_line_width=3,
    source_marker='*',
    target_marker='D',
    title="My Route",
    show_raster=True,
    reverse_colors=True,
)
```

## `plot_paths()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `paths` | Path/PathCollection/list | None | Paths to plot (all stored paths if None) |
| `plot_all` | bool | True | Plot all stored paths |
| `subplots` | bool | True | Separate subplot per path |
| `subplot_size` | tuple | (10, 8) | Size per subplot in inches |
| `source_color` | str | 'green' | Source marker color |
| `target_color` | str | 'red' | Target marker color |
| `path_colors` | str/list | None | Path line color(s) |
| `path_line_width` | int | 2 | Path line width |
| `show_raster` | bool | True | Show cost raster background |
| `title` | str/list | None | Plot title(s) |
| `sup_title` | str | None | Overall figure title |
| `reverse_colors` | bool | False | Reverse raster colormap |

## Multiple Paths

When working with multiple routes (from multi-source/target routing), you can choose between subplots and overlay display.

### Subplots (One Per Path)

Each path is displayed in its own subplot for easy comparison:

```{code-block} python
path_finder.plot_paths(subplots=True)
```

### Overlay

All paths are drawn on a single plot, which is useful for comparing routes visually:

```{code-block} python
path_finder.plot_paths(subplots=False)
```

:::{tip}
When overlaying multiple paths, use distinct `path_colors` to distinguish them:

```python
path_finder.plot_paths(
    subplots=False,
    path_colors=['blue', 'green', 'orange', 'red'],
)
```
:::

## Plotting Specific Paths

Select individual paths by ID instead of plotting all stored results:

```{code-block} python
# Single path by ID
path_finder.plot_paths(path_id=0)

# Multiple specific paths
path_finder.plot_paths(path_id=[0, 2])
```

## Example Gallery

```{image} ../_static/generated/viz_gallery_1.png
:alt: Example visualization showing routes on a cost raster
:width: 100%
:align: center
```
