import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import gridspec
from rasterio.windows import bounds as window_bounds
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from typing import List, Tuple, Dict, Union, Optional, Any

# Project imports
from ..raster.handler import RasterHandler
from ..core.path import PathCollection, Path


# Separate class instead of nested class for raster visualization data
class RasterVizData:
    """Container for raster visualization data used for plotting."""

    def __init__(self) -> None:
        """Initialize empty visualization data containers."""
        self.unique_values: np.ndarray = np.array([])
        self.gray_colors: List[Tuple[float, float, float]] = []
        self.value_to_index: Dict[float, int] = {}


class PathPlotter:
    """
    A class for visualizing paths from a PathCollection.

    This class provides functionality to plot paths with options to:
    - Display all paths or individual paths
    - Show or hide the raster data as background
    - Customize markers, colors, and other visual elements
    - Create individual subplots for each path or combine them
    """

    def __init__(self, paths: PathCollection, raster_handler: RasterHandler) -> None:
        """
        Initialize the PathPlotter with a path collection and raster handler.

        Args:
            paths: Collection of Path objects to plot
            raster_handler: RasterHandler object containing the raster data
        """
        self.paths = paths
        self.raster_handler = raster_handler

    def plot_paths(self,
                   plot_all: bool = True,
                   subplots: bool = True,
                   subplotsize: Tuple[int, int] = (10, 8),
                   source_color: str = 'green',
                   target_color: str = 'red',
                   path_colors: Optional[Union[str, List[str]]] = None,
                   source_marker: str = 'o',
                   target_marker: str = 'x',
                   path_linewidth: int = 2,
                   show_raster: bool = True,
                   title: Optional[Union[str, List[str]]] = None,
                   suptitle: Optional[str] = None,
                   path_id: Optional[int | list[int]] = None,
                   reverse_colors: bool = True) -> Union[Axes, List[Axes]]:
        """
        Plot paths with options to display all paths or individual paths.

        Args:
            plot_all: If True, plot all paths. If False, plot only the path with path_id.
            subplots: If True and plot_all is True, create subplots for each path.
            subplotsize: Size of each individual subplot in inches. Defaults to (10, 8).
            source_color: Color for source marker. Defaults to 'green'.
            target_color: Color for target marker. Defaults to 'red'.
            path_colors: Colors for paths. If None, uses default colors.
            source_marker: Marker style for source. Defaults to 'o'.
            target_marker: Marker style for target. Defaults to 'x'.
            path_linewidth: Line width for paths. Defaults to 2.
            show_raster: Whether to show raster data as background. Defaults to True.
            title: Subplot title(s). If None, default titles are created.
            suptitle: The subtitle of the entire figure. Defaults to None.
            path_id: ID of specific path to plot when plot_all is False.
            reverse_colors: Whether to reverse the color scheme (low=dark, high=bright)

        Returns:
            The axes object(s) with the plot. If multiple subplots are created, returns a list.
        """
        # Check if we have any paths
        if not self.paths:
            raise ValueError("No paths found in PathCollection.")

        # Setup path colors and determine which paths to plot
        path_colors = self._setup_path_colors(path_colors, plot_all)
        paths_to_plot = self._determine_paths_to_plot(plot_all, path_id)

        # Create figure, axes, and legend area
        fig, axes, legend_ax = self._create_figure_and_axes(
            paths_to_plot, plot_all, subplots, subplotsize)
        if suptitle:
            fig.suptitle(suptitle, fontsize=16)

        # Initialize data for legend
        legend_handles: List[Any] = []
        legend_labels: List[str] = []
        raster_viz_data = None

        # Plot each path
        for i, (path, ax) in enumerate(zip(paths_to_plot, axes)):
            # Get the title for this plot
            plot_title = self._get_plot_title(title, i, path)

            # Plot raster background if requested
            if show_raster and self.raster_handler is not None:
                raster_viz_data = self._plot_raster_background(ax, raster_viz_data, reverse_colors=reverse_colors)

            # Plot the path and add to legend
            path_color = path_colors[i] if isinstance(path_colors, list) else path_colors
            new_handles, new_labels = self._plot_path(
                ax, path, path_color, path_linewidth,
                source_color, target_color, source_marker, target_marker)

            # Update legend data
            for handle, label in zip(new_handles, new_labels):
                if label not in legend_labels:
                    legend_handles.append(handle)
                    legend_labels.append(label)

            # Format axes
            self._format_axes(ax, plot_title, i, len(axes))

        # Setup the legend
        if show_raster and raster_viz_data and raster_viz_data.unique_values.size > 0:
            self._add_raster_legend(
                legend_handles, legend_labels,
                raster_viz_data.unique_values,
                raster_viz_data.value_to_index,
                raster_viz_data.gray_colors)

        # Create the legend and finalize the plot
        self._create_legend(legend_ax, legend_handles, legend_labels)
        plt.show()

        # Return the axes objects
        return axes[0] if len(axes) == 1 else axes

    def _setup_path_colors(self,
                           path_colors: Optional[Union[str, List[str]]],
                           plot_all: bool) -> Union[str, List[str]]:
        """
        Set up colors for paths based on input parameters.

        Args:
            path_colors: Colors specified by the user, can be a single color or list of colors
            plot_all: Whether all paths will be plotted

        Returns:
            Either a single color string or a list of color values for each path
        """
        # Default path colors
        if path_colors is None:
            # Use colormap for multiple paths
            if plot_all and len(self.paths) > 1:
                # Create colors from the viridis colormap based on number of paths
                cmap = plt.get_cmap('hsv')
                path_colors = [cmap(i / len(self.paths)) for i in range(len(self.paths))]
            else:
                path_colors = 'blue'  # Default color for single path

        # If a single color is given but we need multiple, repeat it
        if isinstance(path_colors, str) and plot_all and len(self.paths) > 1:
            path_colors = [path_colors] * len(self.paths)

        return path_colors

    def _determine_paths_to_plot(self,
                                 plot_all: bool,
                                 path_id: Optional[int | list[int]]) -> List[Path]:
        """
        Determine which paths to plot based on user inputs.

        Args:
            plot_all: Whether to plot all paths
            path_id: ID of specific path to plot when plot_all is False

        Returns:
            List of Path objects to be plotted

        Raises:
            ValueError: If path_id is specified but no matching path is found
        """
        if plot_all:
            # Return all paths in the collection
            return self.paths.all
        elif path_id is not None:
            if isinstance(path_id, int):
                # Find specific path by ID
                path = self.paths.get(path_id=path_id)
                if path is None:
                    raise ValueError(f"No path found with ID {path_id}")
                return [path]
            else:
                return [self.paths.get(path_id=pid) for pid in path_id]
        else:
            # Default to the last path if no specific path is requested
            return [self.paths.all[-1]]

    @staticmethod
    def _create_figure_and_axes(paths_to_plot: List[Path],
                                plot_all: bool,
                                subplots: bool,
                                subplotsize: Tuple[int, int]) -> Tuple[Figure, List[Axes], Axes]:
        """
        Create the figure, grid, and axes for plotting.

        Args:
            paths_to_plot: List of Path objects to be plotted
            plot_all: Whether all paths will be plotted
            subplots: Whether to create individual subplots for each path
            subplotsize: Size of each individual subplot in inches

        Returns:
            Tuple containing:
            - Figure object
            - List of axes for each path
            - Legend axis
        """
        # Calculate grid dimensions
        n_paths = len(paths_to_plot)
        # Use max 3 columns, only if we're plotting multiple paths with subplots
        n_cols = min(3, n_paths) if (plot_all or subplots) and n_paths > 1 else 1
        # Calculate required number of rows
        n_rows = int(np.ceil(n_paths / n_cols)) if plot_all and subplots and n_paths > 1 else 1

        # Calculate figure size based on subplot size and layout with extra space for legend
        figsize = (subplotsize[0] * n_cols + 1.5, subplotsize[1] * n_rows)

        # Create figure
        fig = plt.figure(figsize=figsize)

        # Create main grid with plots and legend area (9:1 ratio)
        outer_gs = gridspec.GridSpec(1, 2, width_ratios=[9, 1], figure=fig)

        # Create grid for plots
        plot_gs = gridspec.GridSpecFromSubplotSpec(
            n_rows, n_cols, subplot_spec=outer_gs[0],
            wspace=0.05, hspace=0.15)  # Minimal spacing between plots

        # Create legend area
        legend_gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_gs[1])
        legend_rect = legend_gs[0].get_position(fig)
        legend_ax = fig.add_axes((legend_rect.x0, legend_rect.y0, legend_rect.width, legend_rect.height))
        legend_ax.axis('off')  # Hide axis for the legend

        # Create axes for plots
        axes: List[Axes] = []
        for i in range(n_paths):
            if plot_all and subplots and n_paths > 1:
                # Multiple subplots case
                row, col = divmod(i, n_cols)

                # Share y axis for plots in the same row for consistency
                if col > 0:
                    share_y_with = axes[row * n_cols]
                    ax = fig.add_subplot(plot_gs[row, col], sharey=share_y_with)
                    plt.setp(ax.get_yticklabels(), visible=False)  # Hide y-ticks for shared axes
                else:
                    ax = fig.add_subplot(plot_gs[row, col])
            else:
                # Single plot case
                ax = fig.add_subplot(plot_gs[0, 0])

            axes.append(ax)

        # Hide unused subplots (when n_paths doesn't fill the grid)
        if plot_all and subplots and n_paths > 1:
            for i in range(n_paths, n_rows * n_cols):
                row, col = divmod(i, n_cols)
                ax = fig.add_subplot(plot_gs[row, col])
                ax.axis('off')  # Turn off unused axes

        return fig, axes, legend_ax

    @staticmethod
    def _get_plot_title(title: Optional[Union[str, List[str]]],
                        index: int,
                        path: Path) -> str:
        """
        Get the title for a specific plot.

        Args:
            title: User-provided title or list of titles
            index: Index of the current path in the list of paths to plot
            path: Current Path object being plotted

        Returns:
            Title string for the current plot
        """
        # If title is a list and we have enough elements, use the corresponding title
        if isinstance(title, list) and index < len(title):
            return title[index]
        # If a single title is provided, use it for all plots
        elif title is not None:
            return title
        # Otherwise create a default title using path information
        elif hasattr(path, 'total_length') and path.total_length is not None:
            return f"Path {path.path_id} (length: {path.total_length:.2f} units)"
        else:
            return f"Path {path.path_id} from Source to Target"

    def _plot_raster_background(self,
                                ax: Axes,
                                viz_data: Optional[RasterVizData] = None,
                                reverse_colors: bool = True) -> RasterVizData:
        """
        Plot the raster data as background for a path.

        Args:
            ax: Matplotlib axes to plot on
            viz_data: Optional visualization data from previous plots
            reverse_colors: Whether to reverse the color scheme (low=dark, high=bright)

        Returns:
            Object containing visualization data for raster legend
        """
        # Create or use existing visualization data
        if viz_data is None:
            viz_data = RasterVizData()

        # Get raster data and create mask for NaN values
        raster_data = self.raster_handler.data[0].copy()
        mask = np.isnan(raster_data)

        # Get bounds for plotting from the raster window
        bounds = window_bounds(
            self.raster_handler.window,
            self.raster_handler.raster_dataset.transform)

        # Apply coordinate correction used in indices_to_coords
        # This ensures the raster is properly aligned with vector data
        pixel_width = abs(self.raster_handler.raster_dataset.transform.a)
        pixel_height = abs(self.raster_handler.raster_dataset.transform.e)

        # Define the extent of the raster plot (left, right, bottom, top)
        extent = (
            bounds[0] - pixel_width,  # left
            bounds[2] - pixel_width,  # right
            bounds[1] + pixel_height,  # bottom
            bounds[3] + pixel_height  # top
        )

        # Identify unique values in the raster (ignoring NaNs)
        valid_data = raster_data[~mask]

        # Only compute unique values and colormap once for efficiency
        if viz_data.unique_values.size == 0:
            # Get and sort the unique values
            viz_data.unique_values = np.unique(valid_data)
            viz_data.unique_values = np.sort(viz_data.unique_values)  # Sort values
            n_values = len(viz_data.unique_values)

            # Create visualization data array
            visualization_data = np.zeros_like(raster_data, dtype=float)

            # Create mapping from raster values to colormap indices
            viz_data.value_to_index = {val: i for i, val in enumerate(viz_data.unique_values)}

            # Create grayscale colormap with sorted values
            if reverse_colors:
                # Reversed: low values (dark/black) to high values (bright/white)
                gray_values = np.linspace(0.05, 0.95, n_values)
            else:
                # Original: low values (bright/white) to high values (dark/black)
                gray_values = np.linspace(0.95, 0.05, n_values)

            viz_data.gray_colors = [(v, v, v) for v in gray_values]
            custom_cmap = colors.ListedColormap(viz_data.gray_colors)
        else:
            # Reuse previously computed data
            visualization_data = np.zeros_like(raster_data, dtype=float)
            custom_cmap = colors.ListedColormap(viz_data.gray_colors)

        # Fill in the visualization data with indices based on the mapping
        for val in viz_data.unique_values:
            visualization_data[raster_data == val] = viz_data.value_to_index[val]

        # Set NaN values in visualization data
        visualization_data[mask] = np.nan

        # Plot the raster with proper extent and colormap
        ax.imshow(
            visualization_data,
            extent=extent,
            cmap=custom_cmap,
            interpolation='nearest',  # No interpolation for accurate representation
            alpha=0.7,  # Partial transparency to see path on top
            vmin=-0.5,  # Offset for proper color mapping
            vmax=len(viz_data.unique_values) - 0.5
        )

        return viz_data

    @staticmethod
    def _plot_path(ax: Axes,
                   path: Path,
                   path_color: Union[str, Tuple[float, float, float, float]],
                   path_linewidth: int,
                   source_color: str,
                   target_color: str,
                   source_marker: str,
                   target_marker: str) -> Tuple[List[Any], List[str]]:
        """
        Plot a single path with its source and target markers.

        Args:
            ax: Matplotlib axes to plot on
            path: Path object to plot
            path_color: Color to use for this path
            path_linewidth: Width of the path line
            source_color: Color for source marker
            target_color: Color for target marker
            source_marker: Marker style for source
            target_marker: Marker style for target

        Returns:
            Tuple containing:
            - List of plot handles for the legend
            - List of labels for the legend
        """
        # Initialize legend items for this path
        legend_handles: List[Any] = []
        legend_labels: List[str] = []

        # Get path coordinates
        path_coords = path.path_coords
        path_x = [coord[0] for coord in path_coords]
        path_y = [coord[1] for coord in path_coords]

        # Plot the path line
        path_line = ax.plot(path_x, path_y, color=path_color, linewidth=path_linewidth)[0]

        # Add path to legend handles
        path_label = f'Path {path.path_id}'
        legend_handles.append(path_line)
        legend_labels.append(path_label)

        # Get source and target coordinates
        source_coords = path.source
        target_coords = path.target

        # Handle both single coordinates and lists of coordinates
        if not isinstance(source_coords, list):
            source_coords = [source_coords]
        if not isinstance(target_coords, list):
            target_coords = [target_coords]

        # Plot each source point
        for coord in source_coords:
            source_point = ax.plot(coord[0], coord[1], marker=source_marker,
                                   color=source_color, markersize=10)[0]
            legend_handles.append(source_point)
            legend_labels.append('Source')

        # Plot each target point
        for coord in target_coords:
            target_point = ax.plot(coord[0], coord[1], marker=target_marker,
                                   color=target_color, markersize=10)[0]
            legend_handles.append(target_point)
            legend_labels.append('Target')

        return legend_handles, legend_labels

    @staticmethod
    def _format_axes(ax: Axes,
                     title: str,
                     index: int,
                     n_cols: int) -> None:
        """
        Format axis with title, labels and proper aspect ratio.

        Args:
            ax: Matplotlib axes to format
            title: Title for the plot
            index: Index of the current plot
            n_cols: Number of columns in the plot grid
        """
        # Set title
        ax.set_title(title)

        # Always add x-axis label
        ax.set_xlabel('X Coordinate')

        # Only set Y label for leftmost subplot in each row to avoid redundancy
        if index % n_cols == 0:
            ax.set_ylabel('Y Coordinate')

        # Set aspect ratio to equal for proper spatial representation
        # This ensures that distances in x and y directions are scaled equally
        ax.set_aspect('equal')

    @staticmethod
    def _add_raster_legend(legend_handles: List[Any],
                           legend_labels: List[str],
                           unique_values: np.ndarray,
                           value_to_index: Dict[float, int],
                           gray_colors: List[Tuple[float, float, float]]) -> None:
        """
        Add raster color value information to the legend.

        Args:
            legend_handles: List of plot handles for the legend
            legend_labels: List of labels for the legend
            unique_values: Array of unique values in the raster
            value_to_index: Dictionary mapping values to colormap indices
            gray_colors: List of grayscale colors for the colormap
        """
        # Add title/separator for the cost section
        legend_handles.append(Line2D([0], [0], color='none'))
        legend_labels.append('Raster Value (Cost)')

        # Limit the number of color entries for readability
        max_colors_to_show = min(12, len(unique_values))

        # Choose which values to show in the legend
        if len(unique_values) > max_colors_to_show:
            # Choose a subset of values evenly spaced across the range
            indices_to_show = np.linspace(0, len(unique_values) - 1, max_colors_to_show).astype(int)
            values_to_show = unique_values[indices_to_show]
        else:
            # Show all values if there aren't too many
            values_to_show = unique_values

        # Create color patches for the legend
        for val in values_to_show:
            idx = value_to_index[val]
            color = gray_colors[idx]
            # Create a rectangle patch with the appropriate color
            patch = Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.7)
            legend_handles.append(patch)
            # Format as integer if the value is a whole number
            legend_labels.append(f'{int(val)}' if val == int(val) else f'{val}')

    @staticmethod
    def _create_legend(legend_ax: Axes,
                       legend_handles: List[Any],
                       legend_labels: List[str]) -> None:
        """
        Create the legend in the designated legend area.

        Args:
            legend_ax: Matplotlib axes for the legend
            legend_handles: List of plot handles for the legend
            legend_labels: List of labels for the legend
        """
        # Only create legend if we have items to show
        if legend_handles:
            legend_ax.legend(
                legend_handles,
                legend_labels,
                loc='center left',  # Position legend on left side
                fontsize='small',  # Use small font to fit more items
                title="Legend",  # Add title to the legend
                frameon=True  # Show a frame around the legend
            )


