"""
PYORPS: An Open-Source Tool for Automated Power Line Routing

Reference:
[1] Hofmann, M., Stetz, T., Kammer, F., Repo, S.: 'PYORPS: An Open-Source Tool for
    Automated Power Line Routing', CIRED 2025 - 28th Conference and Exhibition on
    Electricity Distribution, 16 - 19 June 2025, Geneva, Switzerland
"""
import warnings
from typing import Any

import numpy as np
from pyproj import Transformer
from rasterio import open as rio_open
from rasterio.features import rasterize
from rasterio.transform import Affine, from_origin, rowcol
from rasterio.transform import xy as transform_xy
from rasterio.windows import Window
from rasterio.windows import transform as transform_window
from shapely.geometry import LineString, MultiPoint, Polygon

from pyorps.core.exceptions import RasterShapeError
from pyorps.core.types import CoordinateList, CoordinateTuple
from pyorps.io.geo_dataset import RasterDataset


class RasterHandler:
    """
    Class for efficiently working with raster data while preserving
    geographic transformation information. Can be initialized with either a file path
    or directly with raster data, CRS, and transform.
    """
    raster_dataset: RasterDataset
    search_space_buffer_m: float
    buffer_geometry: Polygon
    window: Window
    window_transform: Affine
    data: np.ndarray
    windowed_source_read: bool

    def __init__(self,
                 raster_source: RasterDataset,
                 source_coords: tuple[float, float] | list[tuple[float, float]],
                 target_coords: tuple[float, float] | list[tuple[float, float]],
                 search_space_buffer_m: float | None = None,
                 input_crs: str | None = None,
                 apply_mask: bool = True,
                 outside_value: Any | None = None,
                 bands: list[int] | None = None,
                 windowed_read: bool = True,
                 copy_window: bool = True):
        """
        Initialize a RasterHandler for working with raster data and coordinate
        transformations.

        Creates a window and buffer geometry based on source and target coordinates:
        - If source and target are single coordinates: creates a line buffer
        - If source and/or target are lists of coordinates: creates a polygon buffer

        Parameters:
            raster_source: Either:
                          - Path to the raster file (str), or
                          - Tuple of (data_array, crs, transform)
            source_coords: Source point(s) as (x, y) tuple or list of tuples
            target_coords: Target point(s) as (x, y) tuple or list of tuples
            search_space_buffer_m: Buffer distance in map units (typically meters)
            input_crs: CRS of the input coordinates (e.g., 'EPSG:4326'). If None,
                assumes same as raster
            apply_mask: If True, apply the buffer mask after loading data
            outside_value: Value to set for pixels outside the buffer (defaults to max
                value of the data type)
            bands: List of bands to modify if apply_mask is True (1-based). If None, all
                bands are modified
            windowed_read: If True (default) and ``raster_source`` is a file-backed
                dataset whose data has **not** been loaded yet, read only the search
                window from the file instead of the whole raster (plan item 2.3).
                Has no effect once ``raster_source.data`` is populated — an already
                loaded dataset is always windowed by slicing, exactly as before.
            copy_window: If True (default) ``self.data`` owns its memory, so the
                handler never writes into the dataset it was given (plan item 2.1).
                Setting it False restores the historical zero-copy view and is only
                allowed together with ``apply_mask=False``.
        """
        # Determine the type of input we're working with
        self.raster_dataset = raster_source
        self.windowed_source_read = False
        self._init_from_metadata(
                source_coords,
                target_coords,
                search_space_buffer_m,
                input_crs,
                apply_mask,
                outside_value,
                bands,
                windowed_read,
                copy_window
            )

    def _init_from_metadata(
            self,
            source_coords: CoordinateTuple | CoordinateList,
            target_coords: CoordinateTuple | CoordinateList,
            search_space_buffer_m: float | None = None,
            input_crs: str | None = None,
            apply_mask: bool = True,
            outside_value: Any | None = None,
            bands: list[int] | None = None,
            windowed_read: bool = True,
            copy_window: bool = True
    ):
        """
        Initialize using metadata and raster data.

        This method contains the common initialization code used regardless of
        whether the input is a path or direct data components.

        Parameters:
            source_coords: Source point(s) as (x, y) tuple or list of tuples
            target_coords: Target point(s) as (x, y) tuple or list of tuples
            search_space_buffer_m: Buffer distance in map units (typically meters)
            input_crs: CRS of the input coordinates (e.g., 'EPSG:4326'). If None,
                assumes same as raster
            apply_mask: If True, apply the buffer mask after loading data
            outside_value: Value to set for pixels outside the buffer (defaults to max
                value of the data type)
            bands: List of bands to modify if apply_mask is True (1-based). If None, all
                bands are modified
            windowed_read: Read only the search window from a not-yet-loaded
                file-backed dataset (see :meth:`__init__`)
            copy_window: Give ``self.data`` its own memory (see :meth:`__init__`)
        """
        if apply_mask and not copy_window:
            raise ValueError(
                "copy_window=False is only allowed with apply_mask=False: masking "
                "a view would write the outside_value sentinel through into the "
                "source dataset and permanently corrupt it.")

        # Decide whether the source raster can be read window-only. This must
        # happen before anything touches .crs/.transform/.shape, because for a
        # not-yet-loaded file dataset those come from the header read here.
        read_from_file = self._prepare_windowed_source(windowed_read,
                                                       search_space_buffer_m)

        # Transform coordinates if needed
        raster_crs = self.raster_dataset.crs
        transformed_source_coords = self._transform_coords(source_coords, input_crs,
                                                           raster_crs)
        transformed_target_coords = self._transform_coords(target_coords, input_crs,
                                                           raster_crs)

        # Determine if we're working with single coordinates or multiple coordinates
        is_single_source = isinstance(transformed_source_coords, tuple) or (
                isinstance(transformed_source_coords, list) and
                len(transformed_source_coords) == 2 and
                not isinstance(transformed_source_coords[0], (list, tuple))
        )

        is_single_target = isinstance(transformed_target_coords, tuple) or (
                isinstance(transformed_target_coords, list) and
                len(transformed_target_coords) == 2 and
                not isinstance(transformed_target_coords[0], (list, tuple))
        )

        # Create appropriate geometry and buffer it
        if is_single_source and is_single_target:
            # Single pair of coordinates - create a line buffer
            buffer_geom = LineString([transformed_source_coords,
                                      transformed_target_coords])
        else:
            # Multiple coordinates - create a polygon buffer
            all_points = []
            if is_single_source:
                all_points.append(transformed_source_coords)
            else:
                all_points.extend(transformed_source_coords)

            if is_single_target:
                all_points.append(transformed_target_coords)
            else:
                all_points.extend(transformed_target_coords)

            # Create a convex hull from all points and buffer it
            multi_point = MultiPoint(all_points)
            buffer_geom = multi_point.convex_hull

        if search_space_buffer_m is None:
            self.search_space_buffer_m = self.estimate_buffer_width(source_coords,
                                                                    target_coords)
        else:
            self.search_space_buffer_m = search_space_buffer_m
        self.buffer_geometry = buffer_geom.buffer(distance=self.search_space_buffer_m,
                                                  quad_segs=32)

        # Calculate pixel bounds for the buffered geometry
        transform = self.raster_dataset.transform

        # Create window
        self.window = self.window_from_bounds(self.buffer_geometry.bounds,
                                              transform,
                                              self.raster_dataset.shape)

        # Get window-specific transform (crucial for correct coordinate transformations)
        self.window_transform = transform_window(self.window, transform)

        # Extract the windowed data
        if read_from_file:
            # Item 2.3: read only the window from the file. A fresh read always
            # owns its memory, so the copy_window guarantee holds by construction.
            self.data = self._read_window_from_source()
            self.windowed_source_read = True
        elif isinstance(self.raster_dataset.data, np.ndarray):
            min_row = int(self.window.row_off)
            min_col = int(self.window.col_off)
            max_row = min_row + int(self.window.height)
            max_col = min_col + int(self.window.width)
            # Handle different dimensions
            if len(self.raster_dataset.data.shape) == 3:  # (bands, height, width)
                self.data = self.raster_dataset.data[:,
                                                     min_row:max_row,
                                                     min_col:max_col]
            elif len(self.raster_dataset.data.shape) == 2:  # (height, width)
                self.data = self.raster_dataset.data[min_row:max_row, min_col:max_col]
                # Ensure data has shape (bands, height, width)
                self.data = np.expand_dims(self.data, axis=0)
            # Item 2.1: detach the window from the parent array. The values are
            # copied verbatim, so everything inside the window is bit-identical;
            # what changes is only that masking can no longer write through into
            # the dataset the handler was handed.
            if copy_window:
                self.data = self.data.copy()
        else:
            # This shouldn't happen with current implementation
            raise ValueError("Data must be a numpy array")

        # Apply mask if requested
        if apply_mask:
            self.apply_geometry_mask(self.buffer_geometry, outside_value, bands)

    @staticmethod
    def window_from_bounds(
            bounds: tuple[float, float, float, float],
            transform: Affine,
            shape: tuple[int, int]
    ) -> Window:
        """
        Pixel window of ``bounds`` in a raster of ``shape`` under ``transform``.

        This is the single definition of "the search window": it needs nothing
        but the raster header, which is what allows the data to be read window
        first (item 2.3). The arithmetic is unchanged from the original inline
        version, so the window is identical to the historical one.

        Parameters:
            bounds: (minx, miny, maxx, maxy) in the raster CRS
            transform: Affine transform of the raster the window indexes into
            shape: (height, width) of that raster

        Returns:
            rasterio Window clipped to the raster extent
        """
        # Convert bounds to pixel coordinates (top-left and bottom-right)
        min_row, min_col = rowcol(transform, bounds[0], bounds[3])
        max_row, max_col = rowcol(transform, bounds[2], bounds[1])

        # Ensure bounds are within the raster
        min_row = max(0, min_row)
        min_col = max(0, min_col)
        max_row = min(shape[0], max_row)
        max_col = min(shape[1], max_col)

        return Window(min_col, min_row, max_col - min_col, max_row - min_row)

    def _prepare_windowed_source(
            self,
            windowed_read: bool,
            search_space_buffer_m: float | None
    ) -> bool:
        """
        Decide whether the source raster is read window-only, and load its header.

        Returns True when the window must be read straight from the file. In that
        case the dataset's header metadata (crs/transform/shape/count/dtype) is
        populated but ``dataset.data`` stays untouched — callers that need the
        whole raster keep using the explicit full ``load_data()`` path.
        """
        dataset = self.raster_dataset
        if not windowed_read:
            return False
        if getattr(dataset, "data", None) is not None:
            # Already loaded: window by slicing, exactly as before.
            return False
        if not (callable(getattr(dataset, "load_metadata", None)) and
                callable(getattr(dataset, "read_window", None))):
            return False

        dataset.load_metadata()

        if search_space_buffer_m is None:
            # estimate_buffer_width() samples the raster to pick a buffer, and
            # the buffer is what defines the window - a genuine chicken-and-egg.
            # Fall back to the historical full read rather than silently routing
            # on a different search space.
            warnings.warn(
                "windowed_read requires an explicit search_space_buffer_m "
                "(the buffer estimator samples the raster, which is what the "
                "window is derived from). Falling back to a full-raster read.",
                UserWarning, stacklevel=3)
            dataset.load_data()
            return False
        return True

    def _read_window_from_source(self) -> np.ndarray:
        """Read ``self.window`` from the file-backed dataset as (bands, h, w)."""
        data = self.raster_dataset.read_window(self.window)
        if data.ndim == 2:
            data = np.expand_dims(data, axis=0)
        return data

    @staticmethod
    def _transform_coords(
            coords: CoordinateTuple | CoordinateList,
            input_crs: str,
            target_crs: str
    ):
        """
        Transform coordinates from input_crs to target_crs. Handles both single
        coordinates and lists of coordinates.

        Parameters:
            coords: Coordinates to transform from input_crs to target_crs
            input_crs: Coordinate reference system of the input coordinates
            target_crs: Coordinate reference system of the target coordinates

        Returns:
            The transformed coordinates
        """
        if input_crs is None or input_crs == target_crs:
            return coords

        transformer = Transformer.from_crs(input_crs, target_crs, always_xy=True)
        is_coord_list = (isinstance(coords, list) and
                         len(coords) == 2 and
                         not isinstance(coords[0], (list, tuple)))
        if isinstance(coords, tuple) or is_coord_list:
            # Single coordinate pair
            x, y = transformer.transform(coords[0], coords[1])
            return x, y
        # List of coordinates
        result = []
        for coord in coords:
            x, y = transformer.transform(coord[0], coord[1])
            result.append((x, y))
        return result

    def estimate_buffer_width(
            self,
            source_coords: CoordinateTuple | CoordinateList,
            target_coords: CoordinateTuple | CoordinateList,
            min_buffer: float = 200,
            max_buffer: float = 4000,
            sample_radius: float = 50
    ):
        """
        Estimate an appropriate buffer width for path finding based on terrain
        characteristics.

        Parameters:
            source_coords: (x, y) coordinates of the source point
            target_coords: (x, y) coordinates of the target point
            min_buffer: Minimum buffer width to consider (meters)
            max_buffer: Maximum buffer width to consider (meters)
            sample_radius: Radius for sampling around the straight line to assess
                terrain complexity

        Returns:
            Estimated optimal buffer width in meters
        """
        forbidden_value = np.iinfo(self.raster_dataset.dtype).max
        points, euclidean_dist = RasterHandler.max_distance_pair(source_coords,
                                                                 target_coords)
        s, t = points

        # Sample points along the straight line path
        num_samples = min(int(euclidean_dist), 1000)  # Cap at 1000 samples
        x_samples = np.linspace(s[0], t[0], num_samples).astype(int)
        y_samples = np.linspace(s[1], t[1], num_samples).astype(int)

        rows, cols = rowcol(self.raster_dataset.transform,
                            list(x_samples), list(y_samples))

        # Convert bounds to pixel coordinates
        height, width = self.raster_dataset.shape
        x_samples = np.clip(rows, 0, height - 1)
        y_samples = np.clip(cols, 0, width - 1)

        if len(self.raster_dataset.data.shape) == 3:
            raster_array = self.raster_dataset.data[0]
        elif len(self.raster_dataset.data.shape) == 2:
            raster_array = self.raster_dataset.data
        else:
            raise RasterShapeError(self.raster_dataset.data.shape)
        # Sample costs along the line
        line_costs = raster_array[y_samples, x_samples]

        # Count obstacles along the direct path
        obstacle_count = np.sum(line_costs == forbidden_value)
        obstacle_ratio = obstacle_count / len(line_costs)

        # Calculate terrain complexity by examining cost variance in wider area
        complexity_samples = []
        # Limit to 100 sample points for efficiency
        for i in range(min(1000, num_samples)):
            idx = i * (num_samples // min(1000, num_samples))
            x, y = x_samples[idx], y_samples[idx]

            # Define sample region around this point
            x_min = max(0, int(x - sample_radius))
            y_min = max(0, int(y - sample_radius))
            x_max = min(width - 1, int(x + sample_radius))
            y_max = min(height - 1, int(y + sample_radius))

            # Sample the region
            region = raster_array[y_min:y_max, x_min:x_max]
            valid_costs = region[region != forbidden_value]

            if len(valid_costs) > 0:
                # Calculate coefficient of variation to measure complexity
                mean_cost = np.mean(valid_costs)
                if mean_cost > 0:
                    std_cost = np.std(valid_costs)
                    complexity_samples.append(std_cost / mean_cost)

        # Average complexity (coefficient of variation)
        terrain_complexity = np.mean(complexity_samples) if complexity_samples else 0.5

        # Base buffer width on distance
        distance_factor = min(1.0, euclidean_dist / 10000)
        base_buffer = min_buffer + distance_factor * (max_buffer - min_buffer) * 0.5

        # Adjust for terrain complexity and obstacles
        complexity_factor = min(1.0, terrain_complexity * 2)
        obstacle_factor = min(1.0, obstacle_ratio * 5)

        # Final buffer estimation
        buffer_width = base_buffer * (1 + complexity_factor * 0.5 + obstacle_factor)

        # Ensure we stay within bounds
        buffer_width = min(max(buffer_width, min_buffer), max_buffer)

        return int(buffer_width)

    @staticmethod
    def max_distance_pair(
            coords1: CoordinateTuple | CoordinateList,
            coords2: CoordinateTuple | CoordinateList
    ):
        """
        Find the pair of coordinates (one from coords1, one from coords2) with the
        highest Euclidean distance.

        Parameters:
            coords1: Either a single coordinate tuple (x, y, ...) or a list of
                coordinate tuples
            coords2: Either a single coordinate tuple (x, y, ...) or a list of
                coordinate tuples

        Returns:
            A tuple containing the two points with the maximum distance (point1, point2)
        """

        # Normalize inputs to lists of tuples
        def normalize_coords(coords):
            is_coord_list = (len(coords) == 0 or not isinstance(coords[0], tuple))
            if isinstance(coords, tuple) and is_coord_list:
                return [coords]
            return coords

        coords1_list = normalize_coords(coords1)
        coords2_list = normalize_coords(coords2)

        if not coords1_list or not coords2_list:
            return None  # Handle empty inputs

        max_distance = -1
        max_pair = None

        for point1 in coords1_list:
            for point2 in coords2_list:
                # Calculate Euclidean distance
                distance = np.sqrt(np.sum((a - b) ** 2 for a, b in zip(point1, point2)))

                if distance > max_distance:
                    max_distance = distance
                    max_pair = (point1, point2)

        return max_pair, max_distance

    def apply_geometry_mask(
            self,
            geometry: Polygon,
            outside_value: int | None = None,
            bands: list[int] | int | None = None
    ):
        """
        Set pixel values outside the given geometry to the specified value.

        The write always lands in memory owned by this handler: if ``self.data``
        is still a view into the source dataset (only reachable via
        ``copy_window=False``), it is detached first. Masking must never write
        the sentinel back through into the raster it was windowed from — a
        second handler on the same dataset would otherwise inherit the first
        one's buffer, and nothing may cache a buffer that changes behind it
        (plan item 2.1).

        Parameters:
            geometry: A shapely geometry object (Polygon)
            outside_value: Value to set for pixels outside the geometry
            bands: List of bands to modify (1-based). If None, all bands are modified.
        """
        if not self.data.flags.owndata:
            self.data = self.data.copy()

        # Set default outside value if needed
        if outside_value is None:
            outside_value = np.iinfo(self.data.dtype).max

        # Create a mask using rasterization
        mask = rasterize(
            [(geometry, 1)],
            out_shape=(self.window.height, self.window.width),
            transform=self.window_transform,
            fill=0,
            dtype=np.uint8
        )

        # Determine which bands to modify
        if bands is None:
            bands = range(self.data.shape[0])
        else:
            # Convert to 0-based indices for array access
            bands = [b - 1 for b in bands]

        # Apply the mask to selected bands
        for b in bands:
            # Set all pixels outside the buffer (where mask == 0) to outside_value
            self.data[b][mask == 0] = outside_value

        return self.data

    def coords_to_indices(
            self,
            coords: CoordinateTuple | CoordinateList
    ) -> np.ndarray:
        """
        Convert geographic coordinates to pixel row/column indices within this raster
        section.

        Parameters:
            coords: List of (x, y) coordinate tuples or a single coordinate tuple

        Returns:
            numpy.ndarray: Array of (row, col) pixel indices
        """
        transform = self.raster_dataset.transform
        # Use rasterio's rowcol function with the window-specific transform
        if isinstance(coords[0], tuple):
            xs, ys = zip(*coords)
            rows, cols = rowcol(transform, xs, ys)
        else:
            # Single coordinate
            rows, cols = rowcol(transform, coords[0], coords[1])

        # Adjust indices to the window's local coordinate system
        rows = np.array(rows) - self.window.row_off
        cols = np.array(cols) - self.window.col_off

        return np.array(list(zip(rows, cols)))

    def indices_to_coords(self, indices: list[tuple[int, int]]) -> np.ndarray:
        """
        Convert pixel indices to geographic coordinates.

        Indices are window-local (row, col) pairs. The returned coordinates
        are pixel centers in the raster's CRS.

        Parameters:
            indices: List of (row, col) pixel indices

        Returns:
            numpy.ndarray: Array of (x, y) coordinates
        """
        # Convert indices to numpy array if needed
        indices_array = np.atleast_2d(np.array(indices))

        # Extract rows and cols correctly
        if len(indices_array.shape) == 2 and indices_array.shape[1] == 2:
            rows = indices_array[:, 0]
            cols = indices_array[:, 1]
        else:
            rows = indices[0]
            cols = indices[1]

        # Use window_transform with local indices. rasterio's transform_xy
        # uses offset='center' by default, which adds 0.5 to get pixel centers.
        xs, ys = transform_xy(self.window_transform, rows, cols)

        return np.array(list(zip(xs, ys)))

    def save_section_as_raster(self, output_path: str):
        """
        Save the section as a new raster file with proper geo referencing.

        Parameters:
            output_path: Path for the output raster file
        """
        # Create a new raster file with the same properties as the section
        with rio_open(
                output_path,
                'w',
                driver='GTiff',
                height=self.window.height,
                width=self.window.width,
                count=self.raster_dataset.count,
                dtype=self.data.dtype,
                crs=self.raster_dataset.crs,
                transform=self.window_transform  # Use the section-specific transform
        ) as dst:
            dst.write(self.data)


def create_test_tiff(
        output_path: str,
        width: int = 100,
        height: int = 100,
        transform: Affine | None = None,
        crs="EPSG:32632",
        pattern: str = "random",
        bands: int = 1,
        nodata: int | None = None
):
    """
    Creates a synthetic GeoTIFF file for testing with different patterns.

    Parameters:
        output_path: Path to save the test GeoTIFF file
        width: Width of the raster in pixels
        height: Height of the raster in pixels
        transform: Affine transformation for the raster
        crs: Coordinate reference system
        pattern: Data pattern - "random", "gradient", or "checkerboard"
        bands: Number of bands to create
        nodata: No data value

    Returns:
        An array which can be used as a test raster
    """
    if transform is None:
        transform = from_origin(500000, 5600000, 1, 1)

    dtype = np.uint16
    np.random.seed(1234)
    # Create synthetic data based on the specified pattern
    if pattern == "random":
        data = np.random.randint(1, 10, size=(bands, height, width), dtype=dtype)
    elif pattern == "gradient":
        # Create a gradient from top-left to bottom-right
        y, x = np.mgrid[0:height, 0:width]
        base = (x + y) / (width + height) * 100
        data = np.zeros((bands, height, width), dtype=dtype)
        for i in range(bands):
            data[i] = base + i * 10
    elif pattern == "checkerboard":
        # Create a checkerboard pattern
        y, x = np.mgrid[0:height, 0:width]
        data = np.zeros((bands, height, width), dtype=dtype)
        for i in range(bands):
            data[i] = ((x + y + i) % 2) * 50 + 25
    else:
        raise ValueError(f"Pattern {pattern} not recognized!")

    # Write the GeoTIFF
    with rio_open(
            output_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=bands,
            dtype=dtype,
            crs=crs,
            transform=transform,
            nodata=nodata
    ) as dst:
        dst.write(data)

    return data
