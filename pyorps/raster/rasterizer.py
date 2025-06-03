
from typing import Union, Optional
from copy import deepcopy

import numpy as np
from geopandas import GeoDataFrame
from rasterio import open as rio_open
from rasterio.features import rasterize, geometry_mask
from rasterio.transform import Affine, from_bounds
from shapely.geometry import Polygon, box

# Changed to relative imports from other modules
from pyorps.io.geo_dataset import (initialize_geo_dataset, VectorDataset, 
                                   RasterDataset, InMemoryRasterDataset, GeoDataset)
from pyorps.core.types import (InputDataType, CostAssumptionsType, BboxType, 
                               GeometryMaskType)
from pyorps.core.cost_assumptions import CostAssumptions


class GeoRasterizer:
    """
    A class for preparing and rasterizing geospatial data with cost assumptions.

    This class integrates:
        - GeoDataset for representing datasets with metadata
        - CostAssumptions for handling cost mappings
        - Rasterization functionality for converting vector data to rasters
    """

    def __init__(
            self,
            input_data: GeoDataset,
            cost_assumptions: CostAssumptionsType,
            bbox: Optional[BboxType] = None,
            mask: Optional[GeometryMaskType] = None,
            default_crs: Optional[str] = None,
            **kwargs
    ):
        """
        Initialize the GeoRasterizer with a base dataset and optional parameters.

        Parameters:
            input_data: The base dataset to rasterize (file path, GeoDataFrame, dict
                with web params, or GeoDataset)
            mask: Window or polygon mask to limit data reading
            cost_assumptions: Cost values for rasterization (dict, file path, or
                CostAssumptions object)
            default_crs: Default coordinate reference system to use
            **kwargs: Additional parameters passed to load function of the GeoDataset
                if base_dataset is not a GeoDataset
        """
        self.base_dataset = input_data
        self.additional_datasets = []
        self.default_crs = default_crs
        self.bbox = bbox
        self.mask = mask

        if isinstance(cost_assumptions, CostAssumptions):
            self.cost_manager = cost_assumptions
        else:
            self.cost_manager = CostAssumptions(cost_assumptions)

        if self.base_dataset.data is None:
            self.base_dataset.load_data(**kwargs)

        if isinstance(self.base_dataset, RasterDataset):
            self.raster = self.base_dataset.data
            self.transform = self.base_dataset.transform
            self.raster_dataset = self.base_dataset
        else:
            self.raster = None
            self.transform = None
            self.raster_dataset = None

    @property
    def base_data(self) -> GeoDataFrame:
        """
        Property to directly access the data attribute of the base_dataset.

        Returns:
            The base dataset (GeoDataFrame
        """
        return self.base_dataset.data

    def clip_to_area(
            self,
            clip_geometry: Union[GeoDataFrame, Polygon]
    ) -> GeoDataset:
        """
        Clip the base dataset to a specific area.

        Parameters:
            clip_geometry: The geometry to clip by

        Returns:
            The clipped base dataset
        """
        if self.base_data is None:
            raise ValueError("No base data loaded to clip")

        self.base_dataset.data = self.base_data.clip(clip_geometry)
        return self.base_dataset

    @staticmethod
    def create_buffer(
            dataset: Union[VectorDataset, GeoDataFrame],
            geometry_buffer_m: float,
            inplace: bool = True
    ) -> Union[VectorDataset, GeoDataFrame]:
        """
        Add a buffer to geometries in a dataset.

        Parameters:
            dataset: The dataset to buffer (GeoDataset or GeoDataFrame)
            geometry_buffer_m: Distance to buffer in dataset's CRS units
            inplace: If True, modify the dataset in place

        Returns:
            The buffered dataset
        """
        if isinstance(dataset, VectorDataset):
            data = dataset.data
        else:
            data = dataset

        if data is None:
            raise ValueError("Dataset has no data to buffer")

        if geometry_buffer_m <= 0:
            return dataset if isinstance(dataset, VectorDataset) else data

        if inplace:
            data['geometry'] = data.buffer(geometry_buffer_m)
            return dataset if isinstance(dataset, VectorDataset) else data
        else:
            buffered_data = data.copy()
            buffered_data['geometry'] = buffered_data.buffer(geometry_buffer_m)

            if isinstance(dataset, VectorDataset):
                # Create a new GeoDataset with the buffered data
                buffered_dataset = deepcopy(dataset)
                buffered_dataset.data = buffered_data
                return buffered_dataset
            else:
                return buffered_data

    def create_bounds_geodataframe(
            self,
            target_crs: Optional[str] = None
    ) -> GeoDataFrame:
        """
        Creates a GeoDataFrame from the bounds of the base data in a specified CRS.

        Parameters:
            target_crs: The desired CRS for the new GeoDataFrame

        Returns:
            A new GeoDataFrame containing the bounds of the base data
        """
        if self.base_dataset is None or self.base_dataset.data is None:
            raise ValueError("No base data loaded to create bounds from")

        # Calculate the bounds of the source GeoDataFrame
        minx, miny, maxx, maxy = self.base_dataset.data.total_bounds

        # Create a bounding box geometry
        bounding_box = box(minx, miny, maxx, maxy)

        # Create a new GeoDataFrame with the bounding box
        bounds_gdf = GeoDataFrame(geometry=[bounding_box], crs=self.crs)

        # Set the CRS of the new GeoDataFrame to the target CRS if specified
        if target_crs:
            bounds_gdf = bounds_gdf.to_crs(target_crs)

        return bounds_gdf

    @property
    def crs(self):
        """
        Passing crs property of base_dataset.

        Returns:
            The desired CRS of the base dataset
        """
        return self.base_dataset.crs

    def rasterize(
            self,
            field_name: str = 'cost',
            resolution_in_m: float = 1.0,
            fill_value: int = 65535,
            save_path: Optional[str] = None,
            dtype: str = "uint16",
            geometry_buffer_m: float = 0,
            bounding_box: Optional[Polygon] = None
    ) -> RasterDataset:
        """
        Rasterize the base dataset based on a specified field.

        Parameters:
            field_name: The field to use for rasterization values
            resolution_in_m: The resolution of the output raster in meters
            fill_value: Value to use for areas with no data
            save_path: Path to save the rasterized output
            dtype: Data type for the output raster
            geometry_buffer_m: Buffer to apply to the dataset geometries
            bounding_box: Bounding box to define the rasterization extent

        Returns:
            tuple of (raster_data, transform)
        """
        if self.base_data.shape[0] == 0:
            raise ValueError("Base data is empty - nothing to rasterize!")

        if self.base_dataset is None or self.base_dataset.data is None:
            raise ValueError("No base dataset loaded to rasterize")

        # Add cost field
        if field_name == 'cost':
            self.cost_manager.apply_to_geodataframe(self.base_data)

        # Fill NA values in the field
        if field_name not in self.base_data.columns:
            raise ValueError(f"Field '{field_name}' not found in the dataset")

        if self.base_data[field_name].isna().any():
            self.base_data[field_name] = self.base_data[field_name].fillna(fill_value)

        # Round the values in the specified field and convert to the desired data type
        self.base_data[field_name] = self.base_data[field_name].round().astype(dtype)

        # Sort values by field to ensure higher cost values have higher priority
        data = self.base_data.sort_values(by=field_name, ascending=True)

        # Apply buffer if needed
        if geometry_buffer_m > 0:
            buffered = self.create_buffer(data, geometry_buffer_m, inplace=False)
        else:
            buffered = data

        if bounding_box is None:
            # Calculate the output shape based on the GeoDataFrame's bounds and the
            # specified resolution
            out_shape = self._calculate_out_shape_from_geodataframe(buffered,
                                                                    resolution_in_m)

            # Create a transformation object to convert between coordinate systems
            self.transform = from_bounds(*buffered.total_bounds, *out_shape[::-1])

            # Create a generator of shapes (geometry, value) pairs for rasterization
            geometry_field_name = zip(buffered['geometry'], buffered[field_name])
            shapes = ((geom, value) for geom, value in geometry_field_name)

            # Rasterize the shapes into a 2D array
            self.raster = rasterize(
                shapes,
                out_shape=out_shape,
                fill=fill_value,
                dtype=dtype,
                transform=self.transform
            )
        else:
            # Calculate the output shape based on the bounding box
            out_shape = self._calculate_out_shape_from_bounding_box(bounding_box,
                                                                    resolution_in_m)

            # Create a transformation object
            self.transform = from_bounds(*bounding_box.bounds, *out_shape[::-1])

            # Create initial raster with fill value
            self.raster = rasterize(
                [(bounding_box, fill_value)],
                out_shape=out_shape,
                fill=fill_value,
                dtype=dtype,
                transform=self.transform
            )

            # Override with dataset values
            for unique_value in buffered[field_name].unique():
                value_geoms = buffered.loc[buffered[field_name] == unique_value]
                geoms_field_name = zip(value_geoms['geometry'], value_geoms[field_name])
                shapes = ((geom, value) for geom, value in geoms_field_name)
                rasterize(
                    shapes,
                    out_shape=out_shape,
                    fill=fill_value,
                    out=self.raster,
                    dtype=dtype,
                    transform=self.transform
                )

        self.raster_dataset = InMemoryRasterDataset(self.raster,
                                                    self.crs,
                                                    self.transform)
        # Write the rasterized data to a new raster file if a save path is provided
        if save_path is not None:
            self.save_raster(save_path)
        return self.raster_dataset

    def _calculate_out_shape_from_bounding_box(
            self,
            bounding_box: Polygon,
            resolution_m2: float = 1.0
    ) -> tuple[int, int]:
        """
        Calculate the output shape (rows, columns) based on a bounding box and
        resolution.

        Parameters:
            bounding_box: The bounding box defining the output shape in a planar CRS
            resolution_m2: The resolution in square meters

        Returns:
            tuple of (rows, columns) representing the output shape
        """
        # Calculate the bounding box dimensions
        bounds = bounding_box.bounds  # (minx, miny, maxx, maxy)
        width = bounds[2] - bounds[0]  # maxx - minx
        height = bounds[3] - bounds[1]  # maxy - miny

        # Calculate the total area of the bounding box in square meters
        total_area_m2 = width * height

        return self._get_rows_and_columns(width, height, resolution_m2, total_area_m2)

    def _calculate_out_shape_from_geodataframe(
            self,
            gdf: GeoDataFrame,
            resolution_m2: float = 1.0,
            bounding_box: Optional[Polygon] = None
    ) -> tuple[int, int]:
        """
        Calculate the output shape (rows, columns) based on a GeoDataFrame and
        resolution.

        Parameters:
            gdf: The GeoDataFrame containing the geometries to cover
            resolution_m2: The resolution in square meters
            bounding_box: Optional bounding box defining the output shape

        Returns:
            tuple of (rows, columns) representing the output shape
        """
        # Ensure the GeoDataFrame is in a projected CRS that uses meters
        if gdf.crs.is_geographic:
            # Transform to a planar CRS (e.g., Web Mercator)
            gdf = gdf.to_crs(epsg=3857)

        # Calculate the bounding box of the GeoDataFrame
        bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
        width = bounds[2] - bounds[0]  # maxx - minx
        height = bounds[3] - bounds[1]  # maxy - miny

        # Calculate the total area of the GeoDataFrame in square meters
        if bounding_box is None:
            total_area_m2 = width * height
        else:
            bounds_bbox = bounding_box.bounds
            bbox_width = bounds_bbox[2] - bounds_bbox[0]
            bbox_height = bounds_bbox[3] - bounds_bbox[1]
            total_area_m2 = bbox_width * bbox_height

        return self._get_rows_and_columns(width, height, resolution_m2, total_area_m2)

    @staticmethod
    def _get_rows_and_columns(width, height, resolution_m2, total_area_m2):
        """
        Calculate rows and columns based on width, height, and resolution.

        Parameters:
            width: Width of the area
            height: Height of the area
            resolution_m2: Resolution in square meters
            total_area_m2: Total area in square meters

        Returns:
            tuple of (rows, columns)
        """
        # Calculate the aspect ratio
        aspect_ratio = width / height if height != 0 else 1.0
        # Calculate the area of each pixel
        pixel_area = resolution_m2
        # Calculate the total number of pixels needed
        total_pixels = total_area_m2 / pixel_area
        # Calculate the height and width based on the aspect ratio
        calculated_height = (total_pixels / aspect_ratio) ** 0.5
        calculated_width = calculated_height * aspect_ratio
        # Convert to integers for output shape
        rows = int(calculated_height)
        columns = int(calculated_width)
        # Adjusting to ensure the total area is covered
        if rows * columns < total_pixels:
            # Increase columns if needed
            if (calculated_width - columns) > (calculated_height - rows):
                columns += 1
            else:
                rows += 1
        return rows, columns

    def modify_raster_with_geodataframe(
            self,
            gdf: GeoDataFrame,
            value: float,
            ignore_value: Optional[float] = 65535,
            multiply: bool = False) -> np.ndarray:
        """
        Modifies the raster cells inside the polygons of a GeoDataFrame.

        Parameters:
            gdf: The GeoDataFrame containing polygons to use for masking
            value: The value to set for the raster cells inside the polygons
            ignore_value: Value in the raster to ignore during modification
            multiply: If True, multiply the raster values by the given value

        Returns:
            The modified raster
        """
        if self.raster is None or self.transform is None:
            raise ValueError("No raster data available to modify")

        # Create a mask from the geometries in the GeoDataFrame
        mask_array = geometry_mask(
            gdf['geometry'].values,
            transform=self.transform,
            invert=True,  # Invert the mask to keep the area inside the polygons
            out_shape=self.raster.shape
        )

        if ignore_value is None:
            ignore_value_mask = np.ones_like(self.raster, dtype=bool)
        else:
            ignore_value_mask = self.raster != ignore_value

        mask = mask_array & ignore_value_mask

        # Modify the raster values based on the specified parameters
        if multiply:
            # Set the raster cells to a multiple of the existing values
            self.raster[mask] = self.raster[mask] * value
        else:
            # Set the raster cells to the new value
            self.raster[mask] = value

        return self.raster

    def modify_raster_from_dataset(
            self,
            input_data: InputDataType,
            cost_assumptions: Optional[Union[CostAssumptionsType, int, float]] = None,
            bbox: Optional[BboxType] = None,
            mask: Optional[GeometryMaskType] = None,
            transform: Optional[Affine] = None,
            geometry_buffer_m: float = 0,
            ignore_value: Optional[float] = 65535,
            multiply: bool = False,
            zone_field: Optional[str] = None,
            forbidden_zone: Optional[str] = None,
            forbidden_value: int = 65535,
            **kwargs
    ) -> np.ndarray:
        """
        Modify the raster with an additional dataset.

        Parameters:
            input_data: Path to the additional dataset file
            cost_assumptions: The CostAssumptionsType or numeric to apply as cost
                values to the base_dataset
            bbox: The bounding box to apply to the input data
            mask: The geometry mask to apply to the input data
            transform: The transform describing the input data
            geometry_buffer_m: Buffer to apply to the dataset geometries
            ignore_value: Value in the raster to ignore
            multiply: If True, multiply the raster values by the given value
                (in cost_assumptions)
            zone_field: Field name for zones in the dataset
            forbidden_zone: Zone value that should be treated as forbidden
            forbidden_value: Value to use for forbidden areas
            **kwargs: Additional keyword arguments, passed to the loading function
                of the GeoDataset

        Returns:
            The modified raster
        """
        if self.raster is None or self.transform is None:
            msg = "No raster data available to modify. Call rasterize() first."
            raise ValueError(msg)

        # Create bounds for data reading
        if bbox is None:
            bbox = self.create_bounds_geodataframe()
        if mask is None:
            mask = self.mask

        dataset = initialize_geo_dataset(input_data, crs=self.crs, bbox=bbox,
                                         mask=mask,
                                         transform=transform)
        dataset.load_data(**kwargs)
        gdf = dataset.data

        # Apply buffer if needed
        gdf = self.create_buffer(gdf, geometry_buffer_m)
        if isinstance(cost_assumptions, float) or isinstance(cost_assumptions, int):
            self._modify_raster_from_dataset_simple_cost_assumptions(gdf,
                                                                     cost_assumptions,
                                                                     ignore_value,
                                                                     multiply,
                                                                     zone_field,
                                                                     forbidden_zone,
                                                                     forbidden_value)
        else:
            if isinstance(cost_assumptions, str) or isinstance(cost_assumptions, dict):
                ca = CostAssumptions(source=cost_assumptions)
            else:
                ca = cost_assumptions

            ca.apply_to_geodataframe(gdf)
            for unique_value in gdf['cost'].unique():
                value_geoms = gdf.loc[gdf['cost'] == unique_value]
                if value_geoms.empty:
                    continue
                geometry_field_names = zip(value_geoms['geometry'], value_geoms['cost'])
                shapes = ((geom, value) for geom, value in geometry_field_names)

                # Create a mask from the geometries in the GeoDataFrame
                mask_array = geometry_mask(
                    shapes,
                    transform=self.transform,
                    invert=True,  # Invert the mask to keep the area inside the polygons
                    out_shape=self.raster.shape
                )
                if ignore_value is None:
                    ignore_value_mask = np.ones_like(self.raster, dtype=bool)
                else:
                    ignore_value_mask = self.raster != ignore_value

                mask = mask_array & ignore_value_mask
                self.raster[mask] = unique_value
        return self.raster

    def _modify_raster_from_dataset_simple_cost_assumptions(
            self,
            gdf: GeoDataFrame,
            cost_assumptions: Optional[Union[CostAssumptionsType, int, float]] = None,
            ignore_value: Optional[float] = 65535,
            multiply: bool = False,
            zone_field: Optional[str] = None,
            forbidden_zone: Optional[str] = None,
            forbidden_value: int = 65535,
    ):
        """
        Modify the raster with an additional GeoDataFrame.

        Parameters:
            gdf: GeoDataFrame, used to modify the raster dataset
            cost_assumptions: The CostAssumptionsType or numeric to apply as cost
                values to the base_dataset
            ignore_value: Value in the raster to ignore
            multiply: If True, multiply the raster values by the given value
                (in cost_assumptions)
            zone_field: Field name for zones in the dataset
            forbidden_zone: Zone value that should be treated as forbidden
            forbidden_value: Value to use for forbidden areas

        Returns:
            The modified raster
        """
        # Handle zoning if specified
        if zone_field and forbidden_zone:
            forbidden_areas = gdf.loc[gdf[zone_field] == forbidden_zone]
            other_areas = gdf.loc[gdf[zone_field] != forbidden_zone]

            # Apply multiplication factor to non-forbidden zones
            if not other_areas.empty:
                self.modify_raster_with_geodataframe(
                    gdf=other_areas,
                    value=cost_assumptions,
                    ignore_value=ignore_value,
                    multiply=multiply
                )

            # Set forbidden zones to forbidden value
            if not forbidden_areas.empty:
                self.modify_raster_with_geodataframe(
                    gdf=forbidden_areas,
                    value=forbidden_value,
                    ignore_value=ignore_value,
                    multiply=False
                )
        else:
            # Standard modification for the entire dataset
            self.modify_raster_with_geodataframe(
                gdf=gdf,
                value=cost_assumptions,
                ignore_value=ignore_value,
                multiply=multiply
            )

    def save_raster(self, save_path: str) -> None:
        """
        Save the rasterized data to a file.

        Parameters:
            save_path: Path to save the raster file
        """
        if self.raster is None or self.transform is None:
            msg = "No raster data available to save. Call rasterize() first."
            raise ValueError(msg)

        with rio_open(
                save_path,
                'w',
                # Specify the output format as GeoTIFF
                driver='GTiff',
                # Height of the raster
                height=self.raster_dataset.shape[0],
                # Width of the raster
                width=self.raster_dataset.shape[1],
                # Number of bands in the output raster
                count=1,
                # Data type of the raster
                dtype=self.raster_dataset.dtype,
                # Coordinate reference system of the raster
                crs=self.raster_dataset.crs,
                # Transformation for the raster
                transform=self.raster_dataset.transform
        ) as dst:
            # Write the raster data to the first band
            dst.write(self.raster_dataset.data, 1)

    def shrink_raster(self, exclude_value: int) -> np.ndarray:
        """
        Shrink the raster by removing outer bounds with a specific value.

        Parameters:
            exclude_value: Value to exclude from the outer bounds

        Returns:
            The shrunk raster
        """
        if self.raster is None:
            msg = "No raster data available to shrink. Call rasterize() first."
            raise ValueError(msg)

        # Create a mask where the raster does not equal the exclude_value
        mask = self.raster != exclude_value

        # Find the first and last rows and columns that contain non-excluded values
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not np.any(rows) or not np.any(cols):
            return self.raster  # Return original if no non-excluded values

        # Determine the indices for the outer bounds to be excluded
        # First row with a non-excluded value
        first_row = np.argmax(rows)
        # Last row with a non-excluded value
        last_row = len(rows) - np.argmax(rows[::-1])
        # First column with a non-excluded value
        first_col = np.argmax(cols)
        # Last column with a non-excluded value
        last_col = len(cols) - np.argmax(cols[::-1])

        # Use the indices to slice the array and return the shrunk raster
        self.raster = self.raster[first_row:last_row, first_col:last_col]

        # Update the transform to account for the change in origin
        self.transform = Affine(
            self.transform.a,
            self.transform.b,
            self.transform.c + first_col * self.transform.a,
            self.transform.d,
            self.transform.e,
            self.transform.f + first_row * self.transform.e
        )
        return self.raster
