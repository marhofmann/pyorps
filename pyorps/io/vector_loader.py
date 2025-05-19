from typing import Optional
from pathlib import Path
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from defusedxml import ElementTree as et
import tempfile

import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union

from ..core.types import BboxType, GeometryMaskType
from ..core.exceptions import (WFSLayerNotFoundError, WFSConnectionError,
                               WFSResponseParsingError, WFSError)


def load_from_wfs(
        url: str,
        layer: str,
        bbox: Optional[BboxType] = None,
        mask: Optional[GeometryMaskType] = None,
        filter_params: Optional[dict] = None,
        auto_match: bool = True,
        max_workers: int = 4
) -> Optional[gpd.GeoDataFrame]:
    """
    Load data from a Web Feature Service (WFS) using chunked loading.

    Args:
        url: The base URL of the WFS service
        layer: Name of the layer to retrieve
        bbox: Optional bounding box to limit the query extent (minx, miny, maxx, maxy)
        mask: Optional geometry mask to limit the query (Shapely Polygon, GeoDataFrame,
                or GeoSeries)
        filter_params: Additional WFS parameters to filter results
        auto_match: Whether to attempt finding similar layer names if exact match not
                found
        max_workers: Maximum number of parallel threads to use

    Returns:
        Loaded GeoDataFrame or None if no data could be loaded

    Raises:
        WFSLayerNotFoundError: If the layer cannot be found and auto_match is False
    """
    # Find the correct layer name
    if auto_match:
        layer = _resolve_layer(url, layer)

    # If mask is provided but no bbox, get bbox from mask
    if bbox is None and mask is not None:
        bbox = _get_bbox_from_mask(mask)

    # If no bounding box is provided, try to load the entire dataset directly
    if bbox is None:
        # Try to load the entire dataset first
        gdf, limit_reached = _try_direct_load(url, layer, filter_params, mask)

        # If we successfully loaded the entire dataset without hitting limits
        if gdf is not None and not limit_reached:
            return gdf

        # If we hit a limit or failed, try to get a bounding box and use chunked loading
        bbox = _get_extent_from_capabilities(url, layer)

        # If we still don't have a bbox but got some data, use the data's extent
        if bbox is None and gdf is not None and not gdf.empty:
            bbox = _add_buffer_to_bbox(gdf.total_bounds)

        # If we still can't get a bbox, we can't proceed
        if bbox is None:
            raise WFSError("Could not determine data extent for chunked loading.")

    # Load data using parallel chunked approach
    return _load_data_in_parallel(url, layer, bbox, filter_params, max_workers, mask)


def _get_bbox_from_mask(mask) -> tuple[float, float, float, float]:
    """
    Extract a bounding box from a geometry mask.

    Args:
        mask: A Shapely geometry, GeoDataFrame, or GeoSeries

    Returns:
        Bounding box as (minx, miny, maxx, maxy)

    Raises:
        ValueError: If the mask is not a supported type
    """
    # For a Shapely geometry
    if hasattr(mask, 'bounds'):
        return mask.bounds
    # For GeoDataFrame or GeoSeries
    elif hasattr(mask, 'total_bounds'):
        return mask.total_bounds
    # For list of geometries
    elif isinstance(mask, list) and all(hasattr(item, 'bounds') for item in mask):
        bounds_list = [geom.bounds for geom in mask]
        min_x = min(b[0] for b in bounds_list)
        min_y = min(b[1] for b in bounds_list)
        max_x = max(b[2] for b in bounds_list)
        max_y = max(b[3] for b in bounds_list)
        return min_x, min_y, max_x, max_y
    else:
        raise ValueError("Mask must be a Shapely geometry, GeoDataFrame, or GeoSeries")


def _chunk_intersects_mask(chunk: tuple[float, float, float, float], mask) -> bool:
    """
    Check if a chunk intersects with a mask.

    Args:
        chunk: Bounding box as (minx, miny, maxx, maxy)
        mask: A Shapely geometry, GeoDataFrame, or GeoSeries

    Returns:
        True if the chunk intersects the mask, False otherwise
    """
    chunk_box = box(*chunk)

    # For a Shapely geometry
    if hasattr(mask, 'intersects'):
        return mask.intersects(chunk_box)
    # For GeoDataFrame or GeoSeries with multiple geometries
    elif hasattr(mask, 'geometry'):
        return any(geom.intersects(chunk_box) for geom in mask.geometry)
    # For list of geometries
    elif isinstance(mask, list):
        return any(geom.intersects(chunk_box) for geom in mask)
    else:
        # Default to True if we can't determine intersection
        return True


def _clip_data_by_mask(gdf: gpd.GeoDataFrame, mask) -> Optional[gpd.GeoDataFrame]:
    """
    Clip a GeoDataFrame by a geometry mask.

    Args:
        gdf: GeoDataFrame to clip
        mask: A Shapely geometry, GeoDataFrame, or GeoSeries

    Returns:
        Clipped GeoDataFrame
    """
    if gdf is None or gdf.empty:
        return gdf

    # For a Shapely geometry
    if hasattr(mask, 'intersects'):
        return gdf[gdf.geometry.intersects(mask)]
    # For GeoDataFrame or GeoSeries
    elif hasattr(mask, 'geometry'):
        # Convert to a single unary_union if it's a multi-geometry mask
        combined_geom = unary_union(list(mask.geometry))
        return gdf[gdf.geometry.intersects(combined_geom)]
    # For list of geometries
    elif isinstance(mask, list):
        combined_geom = unary_union(mask)
        return gdf[gdf.geometry.intersects(combined_geom)]
    else:
        return gdf


def _try_direct_load(
        url: str,
        layer: str,
        filter_params: Optional[dict] = None,
        mask=None
) -> tuple[Optional[gpd.GeoDataFrame], bool]:
    """
    Try to load the entire dataset directly without chunking.

    Args:
        url: The base URL of the WFS service
        layer: Name of the layer to retrieve
        filter_params: Additional WFS parameters to filter results
        mask: Optional geometry mask to limit the query

    Returns:
        tuple of (GeoDataFrame or None, boolean indicating if a server limit was
        likely reached)
    """
    # Extract namespace if present
    namespace = None
    if ':' in layer:
        namespace, _ = layer.split(':', 1)

    # Try different WFS versions
    for version in ["2.0.0", "1.1.0", "1.0.0"]:
        # Set version-specific parameters
        type_param = "TYPENAMES" if version == "2.0.0" else "TYPENAME"

        params = {
            'SERVICE': 'WFS',
            'VERSION': version,
            'REQUEST': 'GetFeature',
            type_param: layer,
            'SRSNAME': 'EPSG:25832'
        }

        # Add namespace parameter if needed
        if namespace and version == "2.0.0":
            base = 'https://www.adv-online.de/namespaces/adv/gid'
            params['NAMESPACES'] = f'xmlns({namespace}={base}/{namespace})'

        # Add any additional filter parameters
        if filter_params:
            params.update(filter_params)

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            content_type = response.headers.get('Content-Type', '').lower()

            gdf = None
            if 'json' in content_type:
                gdf = _parse_geojson_response(response)
            elif 'xml' in content_type or 'gml' in content_type:
                gdf = _parse_xml_response(response)

            if gdf is not None:
                # Apply mask if provided
                if mask is not None:
                    gdf = _clip_data_by_mask(gdf, mask)

                    # If the mask filtered out all data, consider it empty but not
                    # limited
                    if gdf.empty:
                        return gdf, False

                # Check if we likely hit a server limit (common limits are 10,000 or
                # 100,000)
                limit_reached = len(gdf) in (10_000, 100_000, 1_000, 5_000, 50_000)
                return gdf, limit_reached

        except requests.RequestException:
            continue

    return None, False


def _resolve_layer(url: str, requested_layer: str) -> str:
    """
    Find the correct layer name, using fuzzy matching if necessary.

    Args:
        url: The base URL of the WFS service
        requested_layer: The layer name to find or match

    Returns:
        The exact layer name if found, or the best matching layer name

    Raises:
        WFSLayerNotFoundError: If no matching layer can be found
    """
    available_layers = _get_available_layers(url)

    if not available_layers:
        raise WFSLayerNotFoundError("No layers found in WFS service")

    if requested_layer in available_layers:
        return requested_layer

    # Try to find the best match
    best_match = _find_best_matching_layer(requested_layer, available_layers)

    if best_match:
        return best_match

    raise WFSLayerNotFoundError(f"Layer '{requested_layer}' not found and no similar "
                                f"layers available.")


def _get_available_layers(url: str) -> list[str]:
    """
    Get available layers from a WFS service.

    Args:
        url: The base URL of the WFS service

    Returns:
        list of available layer names from the WFS service

    Raises:
        WFSConnectionError: If connection to the WFS service fails
        WFSResponseParsingError: If the WFS response cannot be parsed correctly
    """
    capabilities_params = {
        'SERVICE': 'WFS',
        'VERSION': '2.0.0',
        'REQUEST': 'GetCapabilities'
    }

    try:
        response = requests.get(url, params=capabilities_params, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        raise WFSConnectionError(f"Failed to connect to WFS service: {e}")

    try:
        # Parse the XML response
        root = et.fromstring(response.content)

        # Handle different namespace possibilities
        namespaces = {
            'wfs': 'http://www.opengis.net/wfs/2.0',
            'wfs1': 'http://www.opengis.net/wfs'
        }

        # Try different paths to find feature types
        for namespace_prefix in ['wfs:', 'wfs1:', '']:
            feature_types = root.findall(f'.//{namespace_prefix}FeatureType',
                                         namespaces)
            if feature_types:
                break

        # Extract layer names from feature types
        layers = []
        for feature_type in feature_types:
            for namespace_prefix in ['wfs:', 'wfs1:', '']:
                name_elem = feature_type.find(f'.//{namespace_prefix}Name',
                                              namespaces)
                if name_elem is not None and name_elem.text:
                    layers.append(name_elem.text)
                    break

        return layers

    except et.ParseError as e:
        raise WFSResponseParsingError(f"Failed to parse WFS capabilities: {e}")
    except Exception as e:
        raise WFSResponseParsingError(f"Unexpected error parsing WFS capabilities: "
                                      f"{str(e)}")


def _find_best_matching_layer(target_name: str,
                              available_layers: list[str]) -> Optional[str]:
    """
    Find the layer name with highest similarity to the target name.

    Args:
        target_name: The layer name to search for
        available_layers: list of available layer names

    Returns:
        Best matching layer name or None if no suitable match found
    """
    if not available_layers:
        return None

    # Calculate similarity scores for all available layers
    similarity_scores = [
        (layer, SequenceMatcher(None, target_name.lower(), layer.lower()).ratio())
        for layer in available_layers
    ]

    # Sort by similarity score (highest first)
    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    best_match, score = similarity_scores[0]

    # Only return if similarity is reasonable
    return best_match if score > 0.3 else None


def _get_extent_from_capabilities(url: str,
                                  layer: str) -> Optional[tuple[float,
                                                                float,
                                                                float,
                                                                float]]:
    """
    Extract layer extent from WFS GetCapabilities response.

    Args:
        url: The base URL of the WFS service
        layer: Name of the layer

    Returns:
        Bounding box as (minx, miny, maxx, maxy) or None if extent not found

    Raises:
        WFSConnectionError: If connection to the WFS service fails
        WFSResponseParsingError: If the WFS response cannot be parsed correctly
    """
    capabilities_params = {
        'SERVICE': 'WFS',
        'VERSION': '2.0.0',
        'REQUEST': 'GetCapabilities'
    }

    try:
        response = requests.get(url, params=capabilities_params, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        raise WFSConnectionError(f"Failed to connect to WFS service: {e}")

    try:
        # Parse the XML response
        root = et.fromstring(response.content)

        # Define namespaces
        namespaces = {
            'wfs': 'https://www.opengis.net/wfs/2.0',
            'wfs1': 'https://www.opengis.net/wfs',
            'ows': 'https://www.opengis.net/ows/1.1'
        }

        # Find feature types with different namespace options
        for ns_prefix in ['wfs:', 'wfs1:', '']:
            feature_types = root.findall(f'.//{ns_prefix}FeatureType', namespaces)
            if feature_types:
                break

        # Iterate through feature types to find the one matching our layer
        for feature_type in feature_types:
            # Get the name using different namespace possibilities
            name = None
            for ns_prefix in ['wfs:', 'wfs1:', '']:
                name_elem = feature_type.find(f'.//{ns_prefix}Name', namespaces)
                if name_elem is not None:
                    name = name_elem.text
                    break

            if name and name == layer:
                # Try to find WGS 84 bounding box
                for bbox_path in ['./ows:WGS84BoundingBox', './WGS84BoundingBox',
                                  './BoundingBox']:
                    bbox_elem = feature_type.find(bbox_path, namespaces)
                    if bbox_elem is not None:
                        break

                if bbox_elem is not None:
                    # Get lower and upper corners
                    lower_corner = (bbox_elem.find('./ows:LowerCorner', namespaces)
                                    or bbox_elem.find('./LowerCorner'))
                    upper_corner = (bbox_elem.find('./ows:UpperCorner', namespaces)
                                    or bbox_elem.find('./UpperCorner'))

                    if lower_corner is not None and upper_corner is not None:
                        # Parse coordinates
                        min_lon, min_lat = map(float, lower_corner.text.split())
                        max_lon, max_lat = map(float, upper_corner.text.split())
                        return min_lon, min_lat, max_lon, max_lat

    except et.ParseError as e:
        raise WFSResponseParsingError(f"Failed to parse WFS capabilities: {e}")

    return None


def _add_buffer_to_bbox(
        bounds: tuple[float, float, float, float],
        buffer_factor: float = 0.1
) -> tuple[float, float, float, float]:
    """
    Add a buffer around a bounding box.

    Args:
        bounds: Original bounding box as (minx, miny, maxx, maxy)
        buffer_factor: Fraction of width/height to add as buffer (default: 0.1 or 10%)

    Returns:
        Expanded bounding box with buffer added
    """
    minx, miny, maxx, maxy = bounds
    buffer_x = (maxx - minx) * buffer_factor
    buffer_y = (maxy - miny) * buffer_factor

    return (
        minx - buffer_x,
        miny - buffer_y,
        maxx + buffer_x,
        maxy + buffer_y
    )


def _create_grid(
        bbox: BboxType,
        x_divisions: int,
        y_divisions: int
) -> list[tuple[float, float, float, float]]:
    """
    Divide a bounding box into a grid of smaller chunks.

    Args:
        bbox: Original bounding box as (minx, miny, maxx, maxy)
        x_divisions: Number of divisions along the x-axis
        y_divisions: Number of divisions along the y-axis

    Returns:
        list of bounding boxes representing grid cells
    """
    if isinstance(bbox, tuple):
        minx, miny, maxx, maxy = bbox
    else:
        minx, miny, maxx, maxy = bbox.total_bounds
    width = (maxx - minx) / x_divisions
    height = (maxy - miny) / y_divisions

    chunks = []
    for i in range(x_divisions):
        for j in range(y_divisions):
            chunk = (
                minx + i * width,
                miny + j * height,
                minx + (i + 1) * width,
                miny + (j + 1) * height
            )
            chunks.append(chunk)
    return chunks


def _load_data_in_parallel(
        url: str,
        layer: str,
        bbox: tuple[float, float, float, float],
        filter_params: Optional[dict] = None,
        max_workers: int = 4,
        mask=None
) -> Optional[gpd.GeoDataFrame]:
    """
    Load WFS data in chunks using parallel processing.

    Args:
        url: The base URL of the WFS service
        layer: Name of the layer
        bbox: Bounding box to divide into chunks as (minx, miny, maxx, maxy)
        filter_params: Additional WFS parameters to filter results
        max_workers: Maximum number of parallel threads to use
        mask: Optional geometry mask to limit the query

    Returns:
        Combined GeoDataFrame with all data or None if no data found
    """
    all_gdfs = []

    # Start with a 2x2 grid of chunks
    initial_chunks = _create_grid(bbox, 2, 2)

    # Filter chunks by mask if provided
    if mask is not None:
        initial_chunks = [chunk for chunk in initial_chunks
                          if _chunk_intersects_mask(chunk, mask)]

    # Track chunks to process and processed chunks
    chunks_to_process = [(chunk, 2, 2) for chunk in initial_chunks]
    processed_chunks = set()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while chunks_to_process:
            # Take a batch of chunks for parallel processing
            current_batch = chunks_to_process[:max_workers]
            chunks_to_process = chunks_to_process[max_workers:]

            # Skip any chunks that have been processed before
            filtered_batch = [
                chunk_info for chunk_info in current_batch
                if _chunk_to_key(chunk_info[0]) not in processed_chunks
            ]

            if not filtered_batch:
                continue

            # Mark chunks as processed
            for chunk_info in filtered_batch:
                processed_chunks.add(_chunk_to_key(chunk_info[0]))

            # Create a dictionary mapping futures to their chunk info
            future_to_chunk_info = {}
            for chunk_info in filtered_batch:
                chunk = chunk_info[0]
                future = executor.submit(_fetch_wfs_data, url, layer,
                                         chunk, filter_params)
                future_to_chunk_info[future] = chunk_info

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_chunk_info):
                chunk_info = future_to_chunk_info[future]
                chunk, x_div, y_div = chunk_info

                try:
                    gdf = future.result()

                    if gdf is None or gdf.empty:
                        continue

                    # Apply mask if provided
                    if mask is not None:
                        gdf = _clip_data_by_mask(gdf, mask)
                        if gdf.empty:
                            continue

                    # Add successful results
                    all_gdfs.append(gdf)

                    # Check if we likely hit a feature limit
                    if len(gdf) in (10_000, 100_000, 1_000, 5_000, 50_000):
                        # Create subchunks
                        new_x_div, new_y_div = x_div * 2, y_div * 2
                        sub_chunks = _create_grid(chunk, 2, 2)

                        # Filter sub-chunks by mask if provided
                        if mask is not None:
                            sub_chunks = [
                                sub_chunk for sub_chunk in sub_chunks
                                if _chunk_intersects_mask(sub_chunk, mask)
                            ]

                        # Add new sub-chunks to queue
                        chunks_to_process.extend(
                            [(sub_chunk, new_x_div, new_y_div)
                             for sub_chunk in sub_chunks]
                        )

                except (WFSError, requests.RequestException):
                    # If a chunk fails, try to subdivide it
                    sub_chunks = _create_grid(chunk, 2, 2)

                    # Filter sub-chunks by mask if provided
                    if mask is not None:
                        sub_chunks = [
                            sub_chunk for sub_chunk in sub_chunks
                            if _chunk_intersects_mask(sub_chunk, mask)
                        ]

                    chunks_to_process.extend(
                        [(sub_chunk, x_div * 2, y_div * 2)
                         for sub_chunk in sub_chunks]
                    )

    # Combine all collected data
    return _combine_geodataframes(all_gdfs)


def _chunk_to_key(chunk: tuple[float, float, float, float]) -> str:
    """
    Convert a chunk (bbox tuple) to a string key for deduplication.

    Args:
        chunk: Bounding box as (minx, miny, maxx, maxy)

    Returns:
        String representation of the bounding box with fixed precision
    """
    return ",".join(f"{coord:.6f}" for coord in chunk)


def _fetch_wfs_data(
        url: str,
        layer: str,
        bbox: tuple[float, float, float, float],
        filter_params: Optional[dict] = None
) -> Optional[gpd.GeoDataFrame]:
    """
    Fetch WFS data for a specific bounding box.

    Args:
        url: The base URL of the WFS service
        layer: Name of the layer
        bbox: Bounding box to query as (minx, miny, maxx, maxy)
        filter_params: Additional WFS parameters to filter results

    Returns:
        GeoDataFrame with data or None if no data found or error occurred
    """
    # Extract namespace if present
    namespace = None
    if ':' in layer:
        namespace, _ = layer.split(':', 1)

    # Try different WFS versions
    for version in ["2.0.0", "1.1.0", "1.0.0"]:
        # Set version-specific parameters
        type_param = "TYPENAMES" if version == "2.0.0" else "TYPENAME"

        params = {
            'SERVICE': 'WFS',
            'VERSION': version,
            'REQUEST': 'GetFeature',
            type_param: layer,
            'SRSNAME': 'EPSG:25832',
            'BBOX': f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},EPSG:25832"
        }

        # Add namespace parameter if needed
        if namespace and version == "2.0.0":
            base = 'https://www.adv-online.de/namespaces/adv/gid'
            params['NAMESPACES'] = f'xmlns({namespace}={base}/{namespace})'

        # Add any additional filter parameters
        if filter_params:
            params.update(filter_params)

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            content_type = response.headers.get('Content-Type', '').lower()

            if 'json' in content_type:
                return _parse_geojson_response(response)
            elif 'xml' in content_type or 'gml' in content_type:
                return _parse_xml_response(response)

        except requests.RequestException:
            continue
    return None


def _parse_geojson_response(response: requests.Response) -> Optional[gpd.GeoDataFrame]:
    """
    Parse a GeoJSON response into a GeoDataFrame.

    Args:
        response: HTTP response object containing GeoJSON data

    Returns:
        GeoDataFrame created from GeoJSON features or None if parsing fails
    """
    try:
        geojson_data = response.json()
        if 'features' in geojson_data and geojson_data['features']:
            return gpd.GeoDataFrame.from_features(geojson_data['features'])
    except ValueError:
        return None


def _parse_xml_response(response: requests.Response) -> Optional[gpd.GeoDataFrame]:
    """
    Parse an XML/GML response into a GeoDataFrame.

    Args:
        response: HTTP response object containing XML/GML data

    Returns:
        GeoDataFrame created from XML/GML data or None if parsing fails
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir) / "wfs_response.xml"
            temp_file.write_bytes(response.content)
            return gpd.read_file(temp_file)
    except IOError:
        return None


def _combine_geodataframes(gdfs: list[gpd.GeoDataFrame]) -> Optional[gpd.GeoDataFrame]:
    """
    Combine multiple GeoDataFrames and remove duplicates.

    Args:
        gdfs: list of GeoDataFrames to combine

    Returns:
        Combined GeoDataFrame with duplicates removed or None if input list is empty
    """
    if not gdfs:
        return None

    # Concatenate all GeoDataFrames
    combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))

    # Remove duplicates by geometry
    combined_gdf = combined_gdf.drop_duplicates(subset=['geometry'])

    return combined_gdf
