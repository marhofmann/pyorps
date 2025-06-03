from typing import Tuple, List, Union, Optional, Any

import numpy as np
from shapely.geometry import LineString, MultiPoint, Polygon
from rasterio import open as rio_open
from rasterio.features import rasterize
from rasterio.windows import Window, transform as transform_window
from rasterio.transform import from_origin, rowcol, Affine, xy as transform_xy
from pyproj import Transformer

from pyorps.io.geo_dataset import RasterDataset
from pyorps.core.types import CoordinateTuple, CoordinateList


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

    def __init__(self,
                 raster_source: RasterDataset,
                 source_coords: Union[Tuple[float, float], List[Tuple[float, float]]],
                 target_coords: Union[Tuple[float, float], List[Tuple[float, float]]],
                 search_space_buffer_m: Optional[float] = None,
                 input_crs: Optional[str] = None,
                 apply_mask: bool = True,
                 outside_value: Optional[Any] = None,
                 bands: Optional[List[int]] = None):
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
        """
        # Determine the type of input we're working with
        self.raster_dataset = raster_source
        self._init_from_metadata(
                source_coords,
                target_coords,
                search_space_buffer_m,
                input_crs,
                apply_mask,
                outside_value,
                bands
            )

    def _init_from_metadata(
            self,
            source_coords: Union[CoordinateTuple, CoordinateList],
            target_coords: Union[CoordinateTuple, CoordinateList],
            search_space_buffer_m: Optional[float] = None,
            input_crs: Optional[str] = None,
            apply_mask: bool = True,
            outside_value: Optional[Any] = None,
            bands: Optional[List[int]] = None
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
        """

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
        bounds = self.buffer_geometry.bounds  # (minx, miny, maxx, maxy)
        transform = self.raster_dataset.transform

        # Convert bounds to pixel coordinates (top-left and bottom-right)
        min_row, min_col = rowcol(transform, bounds[0], bounds[3])
        max_row, max_col = rowcol(transform, bounds[2], bounds[1])

        # Ensure bounds are within the raster
        min_row = max(0, min_row)
        min_col = max(0, min_col)
        max_row = min(self.raster_dataset.shape[0], max_row)
        max_col = min(self.raster_dataset.shape[1], max_col)

        window_width = max_col - min_col
        window_height = max_row - min_row

        # Create window
        self.window = Window(min_col, min_row, window_width, window_height)

        # Get window-specific transform (crucial for correct coordinate transformations)
        self.window_transform = transform_window(self.window, transform)

        # Extract the windowed data
        # For file input, data was already read with window
        # For direct data input, we need to slice the array
        if isinstance(self.raster_dataset.data, np.ndarray):
            # Handle different dimensions
            if len(self.raster_dataset.data.shape) == 3:  # (bands, height, width)
                self.data = self.raster_dataset.data[:,
                                                     min_row:max_row,
                                                     min_col:max_col]
            elif len(self.raster_dataset.data.shape) == 2:  # (height, width)
                self.data = self.raster_dataset.data[min_row:max_row, min_col:max_col]
                # Ensure data has shape (bands, height, width)
                self.data = np.expand_dims(self.data, axis=0)
        else:
            # This shouldn't happen with current implementation
            raise ValueError("Data must be a numpy array")

        # Apply mask if requested
        if apply_mask:
            self.apply_geometry_mask(self.buffer_geometry, outside_value, bands)

    @staticmethod
    def _transform_coords(
            coords: Union[CoordinateTuple, CoordinateList],
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
        else:
            # List of coordinates
            result = []
            for coord in coords:
                x, y = transformer.transform(coord[0], coord[1])
                result.append((x, y))
            return result

    def estimate_buffer_width(
            self,
            source_coords: Union[CoordinateTuple, CoordinateList],
            target_coords: Union[CoordinateTuple, CoordinateList],
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
        x_samples = np.clip(rows, 0, width - 1)
        y_samples = np.clip(cols, 0, height - 1)

        raster_array = self.raster_dataset.data[self.raster_dataset.count - 1]
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
            coords1: Union[CoordinateTuple, CoordinateList],
            coords2: Union[CoordinateTuple, CoordinateList]
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
            outside_value: Optional[int] = None,
            bands: Optional[Union[list[int], int]] = None
    ):
        """
        Set pixel values outside the given geometry to the specified value.

        Parameters:
            geometry: A shapely geometry object (Polygon)
            outside_value: Value to set for pixels outside the geometry
            bands: List of bands to modify (1-based). If None, all bands are modified.
        """
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
            coords: Union[CoordinateTuple, CoordinateList]
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

    def indices_to_coords(self, indices: List[Tuple[int, int]]) -> np.ndarray:
        """
        Convert pixel indices to geographic coordinates.

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

        # Adjust indices to the full raster coordinate system
        rows = np.array(rows) + self.window.row_off
        cols = np.array(cols) + self.window.col_off

        # Add 0.5 to convert integer indices to pixel centers
        rows = rows + 0.5
        cols = cols + 0.5

        # Get coordinates using rasterio's transform
        xs, ys = transform_xy(self.raster_dataset.transform, rows, cols)

        # Apply correction based on the observed pixel offset
        pixel_width = abs(self.raster_dataset.transform.a)
        pixel_height = abs(self.raster_dataset.transform.e)

        xs_corrected = np.array(xs) - pixel_width
        ys_corrected = np.array(ys) + pixel_height

        return np.array(list(zip(xs_corrected, ys_corrected)))

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
        transform: Optional[Affine] = None,
        crs="EPSG:32632",
        pattern: str = "random",
        bands: int = 1,
        nodata: Optional[int] = None
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
