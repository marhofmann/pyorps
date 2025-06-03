import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString

from pyorps.utils.plotting import RasterVizData, PathPlotter
from pyorps.core.path import Path, PathCollection
from pyorps.raster.handler import RasterHandler


class TestRasterVizData(unittest.TestCase):
    def test_initialization(self):
        """Test the initialization of RasterVizData."""
        viz_data = RasterVizData()
        self.assertTrue(hasattr(viz_data, 'unique_values'))
        self.assertTrue(hasattr(viz_data, 'gray_colors'))
        self.assertTrue(hasattr(viz_data, 'value_to_index'))
        self.assertTrue(len(viz_data.unique_values) == 0)
        self.assertTrue(len(viz_data.gray_colors) == 0)
        self.assertTrue(len(viz_data.value_to_index) == 0)


class TestPathPlotter(unittest.TestCase):
    def setUp(self):
        """Set up test environment with mocked dependencies."""
        # Create mock PathCollection
        self.path_collection = MagicMock(spec=PathCollection)

        # Create mock Path objects
        self.path1 = MagicMock(spec=Path)
        self.path1.path_id = 1
        self.path1.source = (0, 0)
        self.path1.target = (10, 10)
        self.path1.path_coords = np.array([(0, 0), (5, 5), (10, 10)])
        self.path1.path_geometry = LineString([(0, 0), (5, 5), (10, 10)])
        self.path1.total_length = 14.14
        self.path1.total_cost = 25.0

        self.path2 = MagicMock(spec=Path)
        self.path2.path_id = 2
        self.path2.source = (0, 10)
        self.path2.target = (10, 0)
        self.path2.path_coords = np.array([(0, 10), (5, 5), (10, 0)])
        self.path2.path_geometry = LineString([(0, 10), (5, 5), (10, 0)])
        self.path2.total_length = 14.14
        self.path2.total_cost = 30.0

        # Setup the path collection's all property and __len__ method
        self.path_collection.all = [self.path1, self.path2]
        self.path_collection.__len__.return_value = 2

        # Mock the get method to return paths by ID
        def get_path_by_id(path_id=None, source=None, target=None):
            if path_id == 1:
                return self.path1
            elif path_id == 2:
                return self.path2
            return None

        self.path_collection.get.side_effect = get_path_by_id

        # Create mock RasterHandler
        self.raster_handler = MagicMock(spec=RasterHandler)
        self.raster_handler.data = [np.array([[1, 2], [3, 4]])]

        # Properly mock the window object to work with window_bounds
        window_mock = MagicMock()
        window_mock.__iter__.return_value = [(0, 10), (0, 10)]  # Structure as expected by window_bounds
        self.raster_handler.window = window_mock

        # THIS IS THE FIX: Create the raster_dataset attribute first
        self.raster_handler.raster_dataset = MagicMock()

        # Create a proper mock for the transform that handles multiplication
        transform_mock = MagicMock()
        # When multiplied with any coordinates, return a tuple with two values
        transform_mock.__mul__ = MagicMock(return_value=(0, 0))
        self.raster_handler.raster_dataset.transform = transform_mock

        # Make sure transform has attributes needed by the function
        transform_mock.a = 1.0
        transform_mock.e = 1.0

        # Add other attributes that might be needed
        self.raster_handler.raster_dataset.count = 1
        self.raster_handler.raster_dataset.shape = (100, 100)

        # Initialize PathPlotter with mocks
        self.plotter = PathPlotter(self.path_collection, self.raster_handler)

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.show')
    @patch('pyorps.utils.plotting.window_bounds')  # Update the patch path to be exact
    def test_plot_paths_basic(self, mock_window_bounds, mock_show, mock_figure):
        """Test the basic functionality of plot_paths."""
        # Set the return value for window_bounds
        mock_window_bounds.return_value = (0, 0, 10, 10)

        # Mock figure and axes
        mock_fig = MagicMock()
        mock_axes = [MagicMock()]
        mock_fig.add_subplot.return_value = mock_axes[0]
        mock_figure.return_value = mock_fig

        # Mock gridspec and add_subplot behavior
        with patch('matplotlib.gridspec.GridSpec'):
            with patch('matplotlib.gridspec.GridSpecFromSubplotSpec'):
                # Call the method
                result = self.plotter.plot_paths(plot_all=False, path_id=1)

                # Check basic interactions
                self.assertEqual(result, mock_axes[0])
                mock_show.assert_called_once()
                mock_figure.assert_called_once()

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.show')
    @patch('pyorps.utils.plotting.window_bounds')  # Update the patch path
    def test_plot_paths_with_all_options(self, mock_window_bounds, mock_show, mock_figure):
        """Test plot_paths with various configuration options."""
        # Set the return value for window_bounds
        mock_window_bounds.return_value = (0, 0, 10, 10)

        # Mock figure and axes
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_legend_ax = MagicMock()

        mock_fig.add_subplot.side_effect = [mock_ax1, mock_legend_ax, mock_ax2]
        mock_figure.return_value = mock_fig

        # Setup all the complex mocks for gridspec
        with patch('matplotlib.gridspec.GridSpec'):
            with patch('matplotlib.gridspec.GridSpecFromSubplotSpec'):
                # Call with all paths and subplots
                result = self.plotter.plot_paths(
                    plot_all=True,
                    subplots=True,
                    source_color='cyan',
                    target_color='magenta',
                    path_colors=['yellow', 'orange'],
                    source_marker='s',
                    target_marker='d',
                    path_linewidth=3,
                    show_raster=True,
                    title=["Path 1", "Path 2"],
                    suptitle="All Paths",
                )

                # Check that all options were applied appropriately
                mock_fig.suptitle.assert_called_with("All Paths", fontsize=16)
                self.assertIsInstance(result, list)
                self.assertEqual(len(result), 2)

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.show')
    def test_plot_paths_no_paths(self, mock_show, mock_figure):
        """Test plot_paths with an empty path collection."""
        # Create empty path collection
        empty_collection = MagicMock(spec=PathCollection)
        empty_collection.all = []
        empty_collection.__len__.return_value = 0

        # Create plotter with empty collection
        plotter = PathPlotter(empty_collection, self.raster_handler)

        # Should raise ValueError when no paths are available
        with self.assertRaises(ValueError):
            plotter.plot_paths()

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.show')
    def test_plot_paths_no_raster(self, mock_show, mock_figure):
        """Test plot_paths with no raster data."""
        # Mock figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()

        # Configure mock to always return the same ax for any call to add_subplot
        mock_fig.add_subplot.return_value = mock_ax
        mock_figure.return_value = mock_fig

        # Create plotter with no raster handler
        plotter = PathPlotter(self.path_collection, None)

        with patch('matplotlib.gridspec.GridSpec'):
            with patch('matplotlib.gridspec.GridSpecFromSubplotSpec'):
                result = plotter.plot_paths(path_id=1)
                self.assertEqual(result[0] if isinstance(result, list) else result, mock_ax)

    def test_setup_path_colors(self):
        """Test the _setup_path_colors method."""
        # Test with default colors for single path
        colors = self.plotter._setup_path_colors(None, False)
        self.assertEqual(colors, 'blue')

        # Test with default colors for all paths
        colors = self.plotter._setup_path_colors(None, True)
        self.assertIsInstance(colors, list)
        self.assertEqual(len(colors), 2)  # One color per path

        # Test with custom single color
        colors = self.plotter._setup_path_colors('red', False)
        self.assertEqual(colors, 'red')

        # Test with custom list of colors
        colors = self.plotter._setup_path_colors(['red', 'green'], True)
        self.assertEqual(colors, ['red', 'green'])

        # Test with single color expanded to all paths
        colors = self.plotter._setup_path_colors('purple', True)
        self.assertEqual(colors, ['purple', 'purple'])

    def test_determine_paths_to_plot(self):
        """Test the _determine_paths_to_plot method."""
        # Test with plot_all=True
        paths = self.plotter._determine_paths_to_plot(True, None)
        self.assertEqual(paths, self.path_collection.all)

        # Test with specific path_id
        paths = self.plotter._determine_paths_to_plot(False, 1)
        self.assertEqual(paths, [self.path1])

        # Test with invalid path_id
        with self.assertRaises(ValueError):
            self.plotter._determine_paths_to_plot(False, 999)

        # Test with multiple path_ids
        paths = self.plotter._determine_paths_to_plot(False, [1, 2])
        self.assertEqual(paths, [self.path1, self.path2])

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.gridspec.GridSpec')
    @patch('matplotlib.gridspec.GridSpecFromSubplotSpec')
    def test_create_figure_and_axes(self, mock_gs_from_subplot, mock_gridspec, mock_figure):
        """Test the _create_figure_and_axes method."""
        # Create mock figure and axes
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig

        # Create mock axes
        mock_ax = MagicMock()
        mock_legend_ax = MagicMock()

        # Set up gridspec mocks
        mock_outer_gs = MagicMock()
        mock_plot_gs = MagicMock()
        mock_legend_gs = MagicMock()

        mock_gridspec.return_value = mock_outer_gs
        mock_gs_from_subplot.side_effect = [mock_plot_gs, mock_legend_gs]

        # Set up indexing for gridspec
        mock_outer_gs.__getitem__.side_effect = [0, 1]
        mock_plot_gs.__getitem__.return_value = MagicMock()

        # Create a proper mock for legend_gs[0] that has a get_position method
        gs_item_mock = MagicMock()
        mock_rect = MagicMock()
        mock_rect.x0 = 0.8
        mock_rect.y0 = 0.1
        mock_rect.width = 0.15
        mock_rect.height = 0.8
        gs_item_mock.get_position.return_value = mock_rect
        mock_legend_gs.__getitem__.return_value = gs_item_mock

        # Set up subplot returns
        mock_fig.add_subplot.side_effect = [mock_ax, mock_legend_ax]
        mock_fig.add_axes.return_value = mock_legend_ax

        # Run the method
        fig, axes, legend_ax = self.plotter._create_figure_and_axes(
            [self.path1], True, True, (10, 8))

        # Check results
        self.assertEqual(fig, mock_fig)
        self.assertIsInstance(axes, list)
        self.assertEqual(legend_ax, mock_legend_ax)

    def test_get_plot_title(self):
        """Test the _get_plot_title method."""
        # Test with specific title
        title = self.plotter._get_plot_title("My Title", 0, self.path1)
        self.assertEqual(title, "My Title")

        # Test with list of titles
        title = self.plotter._get_plot_title(["Title 1", "Title 2"], 0, self.path1)
        self.assertEqual(title, "Title 1")

        # Test with default title (using path info)
        self.path1.total_length = 15.0
        title = self.plotter._get_plot_title(None, 0, self.path1)
        self.assertEqual(title, "Path 1 (length: 15.00 units)")

        # Test with default title (no path length)
        self.path1.total_length = None
        title = self.plotter._get_plot_title(None, 0, self.path1)
        self.assertEqual(title, "Path 1 from Source to Target")

    @patch('matplotlib.axes.Axes')
    def test_plot_raster_background(self, mock_ax):
        """Test the _plot_raster_background method."""
        # Setup raster handler with necessary attributes
        self.raster_handler.data = [np.array([[1, 2], [3, np.nan]])]
        self.raster_handler.window = ((0, 10), (0, 10))
        self.raster_handler.raster_dataset = MagicMock()
        self.raster_handler.raster_dataset.transform = MagicMock()

        # Mock window_bounds
        with patch('pyorps.utils.plotting.window_bounds', return_value=(0, 0, 10, 10)):
            # Test with no existing viz_data
            result = self.plotter._plot_raster_background(mock_ax)

            # Check that visualization data was created
            self.assertIsInstance(result, RasterVizData)
            self.assertTrue(len(result.unique_values) > 0)
            self.assertTrue(len(result.gray_colors) > 0)

            # Test with existing viz_data
            viz_data = RasterVizData()
            viz_data.unique_values = np.array([1, 2, 3])
            viz_data.value_to_index = {1: 0, 2: 1, 3: 2}
            viz_data.gray_colors = [(0.1, 0.1, 0.1), (0.5, 0.5, 0.5), (0.9, 0.9, 0.9)]

            result = self.plotter._plot_raster_background(mock_ax, viz_data)

            # Should reuse the existing viz_data
            self.assertEqual(result, viz_data)

            # Test with reverse_colors=False
            result = self.plotter._plot_raster_background(mock_ax, None, False)

            # Check that visualization data was created with reversed color scheme
            self.assertIsInstance(result, RasterVizData)
            self.assertTrue(len(result.unique_values) > 0)
            self.assertTrue(len(result.gray_colors) > 0)

    @patch('matplotlib.axes.Axes')
    def test_plot_path(self, mock_ax):
        """Test the _plot_path method."""
        # Setup axes mock
        mock_ax.plot.return_value = [MagicMock()]

        # Call method
        handles, labels = self.plotter._plot_path(
            mock_ax, self.path1, 'blue', 2, 'green', 'red', 'o', 'x')

        # Check basic outputs
        self.assertIsInstance(handles, list)
        self.assertIsInstance(labels, list)
        self.assertTrue(len(handles) > 0)
        self.assertTrue(len(labels) > 0)

        # Verify that ax.plot was called multiple times (path + markers)
        self.assertTrue(mock_ax.plot.call_count >= 3)

        # Test with list of source/target coordinates
        self.path1.source = [(0, 0), (1, 1)]
        self.path1.target = [(10, 10), (11, 11)]

        handles, labels = self.plotter._plot_path(
            mock_ax, self.path1, 'blue', 2, 'green', 'red', 'o', 'x')

        # Should have more handles/labels for multiple markers
        self.assertTrue(len(handles) > 3)
        self.assertTrue(len(labels) > 3)

    @patch('matplotlib.axes.Axes')
    def test_format_axes(self, mock_ax):
        """Test the _format_axes method."""
        # Test with first plot
        self.plotter._format_axes(mock_ax, "Test Title", 0, 2)

        # Check that title was set
        mock_ax.set_title.assert_called_with("Test Title")

        # Check that labels were set appropriately
        mock_ax.set_xlabel.assert_called_with('X Coordinate')
        mock_ax.set_ylabel.assert_called_with('Y Coordinate')

        # Test with second plot (should not have y-axis label)
        mock_ax.reset_mock()
        self.plotter._format_axes(mock_ax, "Test Title 2", 1, 2)

        mock_ax.set_xlabel.assert_called_with('X Coordinate')
        self.assertFalse(mock_ax.set_ylabel.called)

    def test_add_raster_legend(self):
        """Test the _add_raster_legend method."""
        # Setup test data
        handles = []
        labels = []
        unique_values = np.array([1, 2, 3, 4, 5])
        value_to_index = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
        gray_colors = [(0.1, 0.1, 0.1), (0.3, 0.3, 0.3), (0.5, 0.5, 0.5),
                       (0.7, 0.7, 0.7), (0.9, 0.9, 0.9)]

        # Call method
        self.plotter._add_raster_legend(
            handles, labels, unique_values, value_to_index, gray_colors)

        # Check that items were added to handles and labels
        self.assertTrue(len(handles) > 0)
        self.assertTrue(len(labels) > 0)

        # Check that the raster title was added
        self.assertIn('Raster Value (Cost)', labels)

        # Test with a large number of values (should limit legend entries)
        handles = []
        labels = []
        unique_values = np.arange(1, 50)  # 49 values
        value_to_index = {v: i for i, v in enumerate(unique_values)}
        gray_colors = [(i / 49, i / 49, i / 49) for i in range(49)]

        self.plotter._add_raster_legend(
            handles, labels, unique_values, value_to_index, gray_colors)

        # Should limit number of color entries to 12 (plus title entry)
        self.assertEqual(len(handles), 13)
        self.assertEqual(len(labels), 13)

    @patch('matplotlib.axes.Axes')
    def test_create_legend(self, mock_ax):
        """Test the _create_legend method."""
        # Setup test data
        handles = [MagicMock(), MagicMock()]
        labels = ['Item 1', 'Item 2']

        # Call method
        self.plotter._create_legend(mock_ax, handles, labels)

        # Check that legend was created
        mock_ax.legend.assert_called_once()

        # Test with empty handles/labels (should not create legend)
        mock_ax.reset_mock()
        self.plotter._create_legend(mock_ax, [], [])

        self.assertFalse(mock_ax.legend.called)


if __name__ == '__main__':
    unittest.main()
