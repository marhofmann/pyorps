from abc import ABC, abstractmethod
from os.path import splitext, isfile
from typing import Union, Optional, Any

import geopandas as gpd
from numpy import ndarray, dtype
from rasterio.transform import Affine
from rasterio import open as rio_open

# Changed from flat import to relative import from the same io module
from .vector_loader import load_from_wfs
# Changed to relative import from the core module
from ..core.types import BboxType, InputDataType, GeometryMaskType


class GeoDataset(ABC):
    file_source: Any
    crs: Optional[str] = None

    data: Optional[Union[gpd.GeoDataFrame, ndarray]] = None

    def __init__(self,
                 file_source: Any,
                 crs: Optional[str] = None):
        self.file_source = file_source
        self.crs = crs

    @abstractmethod
    def load_data(self, **kwargs):
        pass


class VectorDataset(GeoDataset, ABC):
    bbox: Optional[BboxType] = None,
    mask: Optional[GeometryMaskType] = None,

    def __init__(self,
                 file_source: Any,
                 crs: Optional[str] = None,
                 bbox: Optional[BboxType] = None,
                 mask: Optional[GeometryMaskType] = None):
        super().__init__(file_source, crs)
        self.bbox = bbox
        self.mask = mask

    @abstractmethod
    def correct_crs(self):
        pass

    @abstractmethod
    def apply_mask(self):
        pass

    @abstractmethod
    def apply_bbox(self):
        pass

    @abstractmethod
    def post_loading(self):
        pass


class InMemoryVectorDataset(VectorDataset):
    def load_data(self, **kwargs):
        self.data = self.file_source
        self.post_loading()

    def correct_crs(self):
        if self.crs is not None:
            if self.crs != self.data.crs:
                self.data = self.data.to_crs(self.crs)
        else:
            self.crs = self.data.crs

    # noinspection PyUnresolvedReferences
    def apply_bbox(self):
        if self.bbox is not None:
            if hasattr(self.bbox, 'crs') and self.bbox.crs != self.data.crs:
                self.bbox = self.bbox.to_crs(self.data.crs)
            self.data = self.data.clip(self.bbox, keep_geom_type=True)

    # noinspection PyUnresolvedReferences
    def apply_mask(self):
        if self.mask is not None:
            if hasattr(self.mask, 'crs') and self.mask.crs != self.data.crs:
                self.mask = self.mask.to_crs(self.data.crs)
            self.data = self.data.clip(self.mask, keep_geom_type=True)

    def post_loading(self):
        self.correct_crs()
        self.apply_bbox()
        self.apply_mask()


class LocalVectorDataset(InMemoryVectorDataset):
    def load_data(self, **kwargs):
        if self.bbox is not None:
            self.data = gpd.read_file(self.file_source, bbox=self.bbox, **kwargs)
        elif self.mask is not None:
            self.data = gpd.read_file(self.file_source, mask=self.mask, **kwargs)
        else:
            self.data = gpd.read_file(self.file_source, **kwargs)
        self.post_loading()

    # noinspection PyUnresolvedReferences
    def apply_bbox(self):
        if self.bbox is not None:
            if hasattr(self.bbox, 'crs') and self.bbox.crs != self.data.crs:
                raise ValueError(f"CRS-Missmatch: The CRS of the vector source and the "
                                 f"bbox are different!\n"
                                 f"CRS of vector source:\n{self.data.crs}\n"
                                 f"CRS of bbox:\n{self.bbox.crs}\n"
                                 f"\nWhen reading a {self.__class__.__name__} "
                                 f"'bbox' needs to have  the same CRS then "
                                 f"the Vector file! A CRS-Missmatch may lead to empty "
                                 f"datasets!")

    # noinspection PyUnresolvedReferences
    def apply_mask(self):
        if self.bbox is None and self.mask is not None:
            if hasattr(self.mask, 'crs') and self.mask.crs != self.data.crs:
                raise ValueError(f"CRS-Missmatch: The CRS of the vector source and the "
                                 f"mask are different!\n"
                                 f"CRS of vector source:\n{self.data.crs}\n"
                                 f"CRS of mask:\n{self.mask.crs}\n"
                                 f"\nWhen reading a {self.__class__.__name__} "
                                 f"'mask' needs to have  the same CRS then "
                                 f"the Vector file! A CRS-Missmatch may lead to empty "
                                 f"datasets!")
        else:
            if self.mask is not None:
                if hasattr(self.mask, 'crs') and self.mask.crs != self.data.crs:
                    self.mask = self.mask.to_crs(self.data.crs)
                self.data = self.data.clip(self.mask, keep_geom_type=True)


class WFSVectorDataset(LocalVectorDataset):
    def load_data(self, **kwargs):
        if "url" not in self.file_source or "layer" not in self.file_source:
            raise ValueError(f"Unsupported dataset source for WFSVectorDataset: "
                             f"{self.file_source}!"
                             f"\nPlease provide a dictionary with a valid 'url' and "
                             f"'layer' key-value pairs!")
        else:

            self.data = load_from_wfs(
                url=self.file_source["url"],
                layer=self.file_source["layer"],
                bbox=self.bbox,
                filter_params=kwargs.get("filter_params"),
                auto_match=kwargs.get("auto_match", True)
            )
        self.post_loading()

    # noinspection PyUnresolvedReferences
    def apply_mask(self):
        if self.mask is not None:
            if hasattr(self.mask, 'crs') and self.mask.crs != self.data.crs:
                self.mask = self.mask.to_crs(self.data.crs)
            self.data = self.data.clip(self.mask, keep_geom_type=True)


class RasterDataset(GeoDataset, ABC):
    crs: str
    transform: Affine
    shape: tuple[int, int]
    count: int
    dtype: dtype


class LocalRasterDataset(RasterDataset):
    def load_data(self, **kwargs):
        with rio_open(self.file_source) as src:
            self.data = src.read(**kwargs)
            self.crs = src.crs
            self.transform = src.transform
            self.count = self.data.shape[0] if len(self.data.shape) > 2 else 1
            if len(self.data.shape) > 2:
                height = self.data.shape[1]
            else:
                height = self.data.shape[0]
            if len(self.data.shape) > 2:
                width = self.data.shape[2]
            else:
                width = self.data.shape[1]
            self.shape = (height, width)
            self.dtype = self.data.dtype


class InMemoryRasterDataset(RasterDataset):
    def __init__(self,
                 file_source: Any,
                 crs: str,
                 transform: Affine):
        super().__init__(file_source, crs)
        self.transform = transform
        self.data = self.file_source
        self.count = self.data.shape[0] if len(self.data.shape) > 2 else 1
        height = self.data.shape[1] if len(self.data.shape) > 2 else self.data.shape[0]
        width = self.data.shape[2] if len(self.data.shape) > 2 else self.data.shape[1]
        self.shape = (height, width)
        self.dtype = self.data.dtype

    def load_data(self, **kwargs):
        pass


def initialize_geo_dataset(file_source: InputDataType,
                           crs: Optional[str] = None,
                           bbox: Optional[BboxType] = None,
                           mask: Optional[GeometryMaskType] = None,
                           transform: Optional[Affine] = None) -> GeoDataset:
    """
    Factory function to create the appropriate GeoDataset instance based on the
    provided input.

    Args:
        file_source: Source data (file path, GeoDataFrame, URL dict, numpy array, etc.)
        crs: Coordinate reference system
        bbox: Bounding box for vector datasets
        mask: Mask for vector datasets
        transform: Affine transform for in-memory raster datasets

    Returns:
        An appropriate GeoDataset subclass instance

    Examples:
        # From local vector file
        vector_dataset = create_geo_dataset("path/to/shapefile.shp", crs="EPSG:4326")

        # From GeoDataFrame
        vector_dataset = create_geo_dataset(gdf, bbox=(x1, y1, x2, y2))

        # From WFS source
        wfs_dataset = create_geo_dataset({"url": "https://example.com/wfs",
                                          "layer": "layer1"})

        # From local raster file
        raster_dataset = create_geo_dataset("path/to/dem.tif")

        # From numpy array
        raster_dataset = create_geo_dataset(array_data, transform=transform,
                                            crs="EPSG:4326")
    """
    # Determine data type (vector or raster)
    data_type = _determine_data_type(file_source)

    # Create appropriate dataset based on type
    if data_type == "vector":
        geodataset = _create_vector_dataset(file_source, crs, bbox, mask)
    elif data_type == "raster":
        geodataset = _create_raster_dataset(file_source, crs, transform)
    else:
        raise ValueError(f"Unable to determine appropriate dataset type for: "
                         f"{file_source}")
    return geodataset


def _determine_data_type(file_source: Any) -> str:
    """Determine whether the input is for a vector or raster dataset."""
    # Check if it's already a GeoDataset subclass
    if isinstance(file_source, GeoDataset):
        if isinstance(file_source, VectorDataset):
            return "vector"
        elif isinstance(file_source, RasterDataset):
            return "raster"

    # Check for in-memory vector data
    if isinstance(file_source, (gpd.GeoDataFrame, gpd.GeoSeries)):
        return "vector"

    # Check for in-memory raster data
    if isinstance(file_source, ndarray):
        return "raster"

    # Check for WFS data
    if (isinstance(file_source, dict) and "url" in file_source
            and "layer" in file_source):
        return "vector"

    # Check file extension for local files
    if isinstance(file_source, str):
        if isfile(file_source):
            ext = splitext(file_source)[1].lower()
            # Vector file extensions
            if ext in [".shp", ".geojson", ".json", ".gpkg", ".gml", ".kml"]:
                return "vector"
            # Raster file extensions
            elif ext in [".tif", ".tiff", ".jp2", ".img", ".bil", ".dem"]:
                return "raster"
        else:
            raise FileNotFoundError(f"File {file_source} not found.")

    return "unknown"


def _create_vector_dataset(file_source: Any,
                           crs: Optional[str] = None,
                           bbox: Optional[BboxType] = None,
                           mask: Optional[GeometryMaskType] = None) -> VectorDataset:
    """Create the appropriate vector dataset based on the input type."""
    # In-memory GeoDataFrame or GeoSeries
    if isinstance(file_source, (gpd.GeoDataFrame, gpd.GeoSeries)):
        return InMemoryVectorDataset(file_source, crs, bbox=bbox, mask=mask)

    # WFS data source
    elif (isinstance(file_source, dict) and "url" in file_source and
          "layer" in file_source):
        return WFSVectorDataset(file_source, crs, bbox=bbox, mask=mask)

    # Local file
    elif isinstance(file_source, str):
        return LocalVectorDataset(file_source, crs, bbox=bbox, mask=mask)

    raise ValueError(f"Unsupported vector data source: {file_source}")


def _create_raster_dataset(file_source: Any,
                           crs: Optional[str] = None,
                           transform: Optional[Affine] = None
                           ) -> RasterDataset:
    """Create the appropriate raster dataset based on the input type."""
    # In-memory numpy array
    if isinstance(file_source, ndarray):
        if not transform or not isinstance(transform, Affine):
            raise ValueError("A valid 'transform' of type Affine is required for "
                             "InMemoryRasterDataset")
        return InMemoryRasterDataset(file_source, crs, transform=transform)

    # Local file
    elif isinstance(file_source, str):
        return LocalRasterDataset(file_source, crs)

    raise ValueError(f"Unsupported raster data source: {file_source}")
