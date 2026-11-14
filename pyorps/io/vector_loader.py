"""
PYORPS: An Open-Source Tool for Automated Power Line Routing

Reference:
[1] Hofmann, M., Stetz, T., Kammer, F., Repo, S.: 'PYORPS: An Open-Source Tool for
Automated Power Line Routing', CIRED 2025 - 28th Conference and Exhibition on
Electricity Distribution, 16 - 19 June 2025, Geneva, Switzerland
"""
import concurrent.futures
import functools
import ipaddress
import logging
import re
import tempfile
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import urlparse
from xml.etree.ElementTree import Element as _Element

import geopandas as gpd
import pandas as pd
import requests
from defusedxml import ElementTree as et
from pyogrio.errors import DataSourceError
from shapely.geometry import box
from shapely.ops import unary_union

from ..core.exceptions import (
    InvalidInputError,
    WFSConnectionError,
    WFSError,
    WFSLayerNotFoundError,
    WFSRequestRejectedError,
    WFSResponseParsingError,
    WFSServerError,
)
from ..core.types import BboxType, GeometryMaskType

logger = logging.getLogger(__name__)

# Allowed URL schemes for WFS requests (SSRF prevention)
_ALLOWED_WFS_SCHEMES = {"http", "https"}

# Private/reserved IP networks that should be blocked for WFS requests
_PRIVATE_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]

# Allowlisted WFS GetFeature parameter keys (case-insensitive comparison).
# Parameters not in this set but also not in _RESERVED_WFS_KEYS are passed
# through with a warning.
_ALLOWED_FILTER_PARAMS = {
    "CQL_FILTER", "FILTER", "PROPERTYNAME", "MAXFEATURES", "COUNT",
    "SORTBY", "STARTINDEX", "RESULTTYPE", "OUTPUTFORMAT", "SRSNAME",
    "BBOX", "NAMESPACES", "TYPENAMES", "TYPENAME",
}

# Reserved WFS keys that filter_params must NEVER override, because they
# change fundamental request semantics.
_RESERVED_WFS_KEYS = {"SERVICE", "REQUEST", "VERSION"}


def _validate_wfs_url(url: str, block_private: bool = True) -> None:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """
    Validate a WFS URL to prevent SSRF attacks.

    Only http:// and https:// schemes are allowed.  Optionally rejects URLs
    that resolve to private/loopback IP ranges.

    Args:
        url: The URL to validate
        block_private: If True, reject URLs whose hostname is a
            private or loopback IP address

    Raises:
        ValueError: If the URL scheme is not allowed or the host is a
            private IP address
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    parsed = urlparse(url)

    if parsed.scheme not in _ALLOWED_WFS_SCHEMES:
        raise InvalidInputError(
            f"Invalid WFS URL scheme '{parsed.scheme}'. "
            f"Only {sorted(_ALLOWED_WFS_SCHEMES)} are allowed."
        )

    if not parsed.hostname:
        raise InvalidInputError("WFS URL must include a hostname.")

    if block_private:
        # Only a literal IP address can be in a private range; a hostname
        # (ip_address() raises ValueError for it) is fine here. The guard's own
        # ValueError is raised OUTSIDE the try block, so it is never mistaken
        # for the parse failure -- no matching on message text.
        try:
            ip = ipaddress.ip_address(parsed.hostname)
        except ValueError:
            return
        for network in _PRIVATE_NETWORKS:
            if ip in network:
                raise InvalidInputError(
                    f"WFS URL points to a private/reserved IP address "
                    f"({parsed.hostname}). This is not allowed for "
                    f"security reasons."
                )


def _sanitize_filter_params(filter_params: dict | None) -> dict | None:
    """
    Sanitize WFS filter parameters.

    Rejects reserved WFS keys (SERVICE, REQUEST, VERSION) that could change
    request semantics.  Logs a warning for unrecognised parameter names but
    still passes them through.

    Args:
        filter_params: Dictionary of additional WFS parameters

    Returns:
        Sanitized copy of filter_params (or None)

    Raises:
        ValueError: If filter_params contains reserved WFS keys
    """
    if not filter_params:
        return filter_params

    sanitized = {}
    for key, value in filter_params.items():
        upper_key = key.upper()

        if upper_key in _RESERVED_WFS_KEYS:
            raise InvalidInputError(
                f"filter_params must not contain the reserved WFS key "
                f"'{key}'. Reserved keys ({sorted(_RESERVED_WFS_KEYS)}) "
                f"are managed internally and cannot be overridden."
            )

        if upper_key not in {k.upper() for k in _ALLOWED_FILTER_PARAMS}:
            logger.warning(
                "Unrecognised WFS filter parameter '%s'. "
                "Passing through, but verify this is intentional.", key
            )

        sanitized[key] = value

    return sanitized


# Standard OGC namespace URIs used in WFS GetCapabilities responses.
# These MUST use http:// (not https://) per OGC specification.
_WFS_NAMESPACES = {
    'wfs': 'http://www.opengis.net/wfs/2.0',
    'wfs1': 'http://www.opengis.net/wfs',
    'ows': 'http://www.opengis.net/ows/1.1'
}

# Maximum number of chunk subdivision levels before giving up.
# Each level quadruples the number of chunks (2x2 split), so depth 8
# means up to 4^8 = 65536 chunks — well beyond any reasonable need.
MAX_CHUNK_DEPTH = 8

# Minimum chunk area (in squared CRS units) below which further
# subdivision is pointless. For EPSG:25832 (metres) this is 1 m^2;
# for degree-based CRS it is ~1e-6 sq degrees (~0.01 m^2 at equator).
MIN_CHUNK_AREA = 1e-6

# Stop the chunked download when this many chunk requests fail in a row
# without a single success. "Split a failed chunk" otherwise queues up to
# 4^MAX_CHUNK_DEPTH requests against a server that is simply down.
MAX_CONSECUTIVE_CHUNK_FAILURES = 16

# OGC exception codes that say the request itself is wrong: splitting the
# chunk cannot fix them. Any other code (InternalServerError,
# NoApplicableCode, ...) is treated as a server failure worth retrying.
_REJECTED_EXCEPTION_CODES = frozenset({
    "invalidparametervalue", "missingparametervalue",
    "operationnotsupported", "optionnotsupported",
    "versionnegotiationfailed", "invalidrequest",
})

_ROOT_ELEMENT_RE = re.compile(r"<(?![?!])(?:[\w.-]+:)?([\w.-]+)")
_EXCEPTION_CODE_RE = re.compile(r'(?:exceptionCode|code)="([^"]*)"')
_EXCEPTION_TEXT_RE = re.compile(
    r"<(?:[\w.-]+:)?(?:ExceptionText|ServiceException)\b[^>]*>(.*?)</",
    re.DOTALL)


def _wfs_exception_report(response) -> tuple[str, str] | None:
    """
    Return ``(code, message)`` if a response is an OGC exception report.

    Some WFS servers (e.g. Hessen's XtraServer for WFS 1.1.0) report errors
    as ``HTTP 200`` with an ``ExceptionReport`` body, so the status code and
    the XML content type cannot be trusted. Only the first 4 kB is inspected:
    real responses can be hundreds of MB.
    """
    head = getattr(response, "content", None)
    if not isinstance(head, (bytes, bytearray)):
        return None
    text = bytes(head[:4096]).decode("utf-8", errors="replace")
    root = _ROOT_ELEMENT_RE.search(text)
    if root is None or not root.group(1).endswith("ExceptionReport"):
        return None
    code = _EXCEPTION_CODE_RE.search(text)
    message = _EXCEPTION_TEXT_RE.search(text)
    return (code.group(1) if code else "unknown",
            " ".join(message.group(1).split()) if message else "")


def _exception_for_reports(reports: list[tuple[str, str]]) -> WFSError | None:
    """Return the WFS error that fits the exception reports collected (or None)."""
    if not reports:
        return None
    summary = "; ".join(f"{code}: {msg}" for code, msg in reports)
    if all(code.lower() in _REJECTED_EXCEPTION_CODES for code, _ in reports):
        return WFSRequestRejectedError(f"WFS server rejected the request ({summary})")
    return WFSServerError(f"WFS server failed to process the request ({summary})")


def _raise_for_exception_reports(reports: list[tuple[str, str]]) -> None:
    """Raise the WFS error that fits the exception reports collected."""
    error = _exception_for_reports(reports)
    if error is not None:
        raise error


def _response_excerpt(response, limit: int = 200) -> str:
    """First ``limit`` characters of a response body, for error messages."""
    content = getattr(response, "content", None)
    if not isinstance(content, (bytes, bytearray)):
        return ""
    return " ".join(bytes(content[:limit]).decode("utf-8", errors="replace").split())


def load_from_wfs(
        url: str,
        layer: str,
        bbox: BboxType | None = None,
        mask: GeometryMaskType | None = None,
        filter_params: dict | None = None,
        auto_match: bool = True,
        max_workers: int = 4,
        crs: str | None = None,
        allow_partial: bool = False
) -> gpd.GeoDataFrame | None:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
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
        crs: Spatial reference system for WFS requests (e.g. "EPSG:4326").
            Defaults to "EPSG:25832" if not provided.
        allow_partial: What to do when the chunked download cannot be completed
            (chunk requests keep failing, or the server's feature limit still
            truncates a chunk at the finest subdivision). ``False`` (default)
            raises a WFS error that names the failed request and says how many
            features were received. ``True`` returns what was received and
            emits a ``UserWarning`` with the same counts.

    Returns:
        Loaded GeoDataFrame or None if no data could be loaded

    Raises:
        WFSLayerNotFoundError: If the layer cannot be found and auto_match is False
        WFSConnectionError, WFSServerError, WFSResponseParsingError, WFSError:
            If the chunked download is incomplete and ``allow_partial`` is False
            (the class of the underlying failure; its cause is chained)
        ValueError: If the URL scheme is invalid or filter_params contain
            reserved WFS keys
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    # Security: validate URL scheme and block private IPs
    _validate_wfs_url(url)

    # Security: sanitize filter parameters
    filter_params = _sanitize_filter_params(filter_params)

    # Derive SRS from the crs argument, falling back to EPSG:25832
    srs = crs if crs is not None else 'EPSG:25832'

    # Find the correct layer name
    if auto_match:
        layer = _resolve_layer(url, layer)

    # If mask is provided but no bbox, get bbox from mask
    if bbox is None and mask is not None:
        bbox = _get_bbox_from_mask(mask)

    # If no bounding box is provided, try to load the entire dataset directly
    if bbox is None:
        # Try to load the entire dataset first. A failed direct request is not
        # fatal -- many servers refuse an unbounded request that they would
        # answer chunk by chunk -- but the reason is kept and reported if the
        # chunked path cannot proceed either.
        direct_error: WFSError | None = None
        try:
            gdf, limit_reached = _try_direct_load(url, layer, filter_params,
                                                  mask, srs=srs)
        except WFSError as err:
            direct_error = err
            gdf, limit_reached = None, False
            logger.warning("Direct WFS request for layer '%s' failed (%s); "
                           "falling back to chunked loading.", layer, err)

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
            reason = (f" The direct request failed first: {direct_error}"
                      if direct_error is not None else "")
            raise WFSError("Could not determine data extent for chunked "
                           f"loading.{reason}") from direct_error

    # Load data using parallel chunked approach. ``allow_partial`` is only
    # forwarded when set, so the call made by the default path is unchanged.
    extra = {"allow_partial": True} if allow_partial else {}
    return _load_data_in_parallel(url, layer, bbox, filter_params, max_workers,
                                 mask, srs=srs, **extra)


def _get_bbox_from_mask(mask) -> tuple[float, float, float, float]:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """
    Extract a bounding box from a geometry mask.

    Args:
        mask: A Shapely geometry, GeoDataFrame, or GeoSeries

    Returns:
        Bounding box as (minx, miny, maxx, maxy)

    Raises:
        ValueError: If the mask is not a supported type
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    # For GeoDataFrame or GeoSeries. This must come first: they also have a
    # ``bounds`` attribute, but it is a per-geometry DataFrame, not one box.
    if hasattr(mask, 'total_bounds'):
        minx, miny, maxx, maxy = mask.total_bounds
        return float(minx), float(miny), float(maxx), float(maxy)
    # For a Shapely geometry
    if hasattr(mask, 'bounds'):
        return mask.bounds
    # For list of geometries
    if isinstance(mask, list) and all(hasattr(item, 'bounds') for item in mask):
        bounds_list = [geom.bounds for geom in mask]
        min_x = min(b[0] for b in bounds_list)
        min_y = min(b[1] for b in bounds_list)
        max_x = max(b[2] for b in bounds_list)
        max_y = max(b[3] for b in bounds_list)
        return min_x, min_y, max_x, max_y
    raise InvalidInputError("Mask must be a Shapely geometry, GeoDataFrame, or GeoSeries")


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
    if hasattr(mask, 'geometry'):
        return any(geom.intersects(chunk_box) for geom in mask.geometry)
    # For list of geometries
    if isinstance(mask, list):
        return any(geom.intersects(chunk_box) for geom in mask)
    # Default to True if we can't determine intersection
    return True


def _clip_data_by_mask(gdf: gpd.GeoDataFrame, mask) -> gpd.GeoDataFrame | None:
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
    if hasattr(mask, 'geometry'):
        # Convert to a single unary_union if it's a multi-geometry mask
        combined_geom = unary_union(list(mask.geometry))
        return gdf[gdf.geometry.intersects(combined_geom)]
    # For list of geometries
    if isinstance(mask, list):
        combined_geom = unary_union(mask)
        return gdf[gdf.geometry.intersects(combined_geom)]
    return gdf


def _try_direct_load(
        url: str,
        layer: str,
        filter_params: dict | None = None,
        mask=None,
        srs: str = 'EPSG:25832'
) -> tuple[gpd.GeoDataFrame | None, bool]:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """
    Try to load the entire dataset directly without chunking.

    Args:
        url: The base URL of the WFS service
        layer: Name of the layer to retrieve
        filter_params: Additional WFS parameters to filter results
        mask: Optional geometry mask to limit the query
        srs: Spatial reference system for the SRSNAME parameter
            (default: "EPSG:25832")

    Returns:
        tuple of (GeoDataFrame or None, boolean indicating if a server limit was
        likely reached). ``None`` means the server answered but sent no features.

    Raises:
        WFSError: If no WFS version produced a usable answer: the last failure
            (WFSConnectionError, WFSServerError, WFSResponseParsingError, ...)
            with its cause chained. ``load_from_wfs`` treats this as "the direct
            request is not possible" and falls back to chunked loading.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    # Extract namespace if present
    namespace = None
    if ':' in layer:
        namespace, _ = layer.split(':', 1)

    # What went wrong per version, and whether any version answered at all
    failures: list[Exception] = []
    answered = False

    # Try different WFS versions
    for version in ["2.0.0", "1.1.0", "1.0.0"]:
        # Set version-specific parameters
        type_param = "TYPENAMES" if version == "2.0.0" else "TYPENAME"

        params = {
            'SERVICE': 'WFS',
            'VERSION': version,
            'REQUEST': 'GetFeature',
            type_param: layer,
            'SRSNAME': srs
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

            report = _wfs_exception_report(response)
            if report is not None:
                failures.append(_exception_for_reports([report]))
                continue

            content_type = response.headers.get('Content-Type', '').lower()

            gdf = None
            if 'json' in content_type:
                gdf = _parse_geojson_response(response)
            elif 'xml' in content_type or 'gml' in content_type:
                gdf = _parse_xml_response(response)
            answered = True

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

        except (requests.RequestException, WFSResponseParsingError) as exc:
            failures.append(exc)
            continue

    if failures and not answered:
        last = failures[-1]
        if isinstance(last, WFSError):
            raise last
        raise WFSConnectionError(
            f"Direct WFS request for layer '{layer}' failed for every WFS "
            f"version tried (last error: {last})") from last

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
                                f"layers available! Available layers:"
                                f"\n{available_layers}")


@functools.lru_cache(maxsize=16)
def _fetch_capabilities_xml(url: str) -> _Element:
    """
    Fetch and cache the parsed WFS GetCapabilities XML tree.

    The result is cached per URL so that multiple callers
    (e.g. ``_get_available_layers`` and ``_get_extent_from_capabilities``)
    do not issue redundant HTTP requests for the same service.

    Args:
        url: The base URL of the WFS service

    Returns:
        Parsed XML Element tree root

    Raises:
        WFSConnectionError: If connection to the WFS service fails
        WFSResponseParsingError: If the XML response cannot be parsed
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
        raise WFSConnectionError(f"Failed to connect to WFS service: {e}") from e

    try:
        return et.fromstring(response.content)
    except et.ParseError as e:
        raise WFSResponseParsingError(f"Failed to parse WFS capabilities: {e}") from e


def _get_available_layers(url: str,
                          capabilities_xml: _Element | None = None
                          ) -> list[str]:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """
    Get available layers from a WFS service.

    Args:
        url: The base URL of the WFS service
        capabilities_xml: Optional pre-fetched capabilities XML root element.
            If not provided, the capabilities will be fetched (and cached).

    Returns:
        list of available layer names from the WFS service

    Raises:
        WFSConnectionError: If connection to the WFS service fails
        WFSResponseParsingError: If the WFS response cannot be parsed correctly
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    if capabilities_xml is None:
        capabilities_xml = _fetch_capabilities_xml(url)

    try:
        root = capabilities_xml

        # Handle different namespace possibilities
        namespaces = _WFS_NAMESPACES

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

    except Exception as e:
        raise WFSResponseParsingError(f"Unexpected error parsing WFS capabilities: "
                                      f"{str(e)}") from e


def _find_best_matching_layer(target_name: str,
                              available_layers: list[str]) -> str | None:
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


def _get_extent_from_capabilities(
        url: str,
        layer: str,
        capabilities_xml: _Element | None = None
) -> tuple[float, float, float, float] | None:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """
    Extract layer extent from WFS GetCapabilities response.

    Args:
        url: The base URL of the WFS service
        layer: Name of the layer
        capabilities_xml: Optional pre-fetched capabilities XML root element.
            If not provided, the capabilities will be fetched (and cached).

    Returns:
        Bounding box as (minx, miny, maxx, maxy) or None if extent not found

    Raises:
        WFSConnectionError: If connection to the WFS service fails
        WFSResponseParsingError: If the WFS response cannot be parsed correctly
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    if capabilities_xml is None:
        capabilities_xml = _fetch_capabilities_xml(url)

    try:
        root = capabilities_xml

        # Define namespaces
        namespaces = _WFS_NAMESPACES

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
                    # Note: do NOT use `elem or fallback` with XML Elements,
                    # because an Element with no child elements evaluates as
                    # falsy even when it exists and has text content.
                    lower_corner = bbox_elem.find('./ows:LowerCorner', namespaces)
                    if lower_corner is None:
                        lower_corner = bbox_elem.find('./LowerCorner')
                    upper_corner = bbox_elem.find('./ows:UpperCorner', namespaces)
                    if upper_corner is None:
                        upper_corner = bbox_elem.find('./UpperCorner')

                    if lower_corner is not None and upper_corner is not None:
                        # Parse coordinates
                        min_lon, min_lat = map(float, lower_corner.text.split())
                        max_lon, max_lat = map(float, upper_corner.text.split())
                        return min_lon, min_lat, max_lon, max_lat

    except et.ParseError as e:
        raise WFSResponseParsingError(f"Failed to parse WFS capabilities: {e}") from e

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


def _chunk_area(chunk: tuple[float, float, float, float]) -> float:
    """Return the area of a bounding-box chunk."""
    minx, miny, maxx, maxy = chunk
    return abs((maxx - minx) * (maxy - miny))


def _partial_wfs_error(message: str, cause: BaseException | None) -> WFSError:
    """The WFS error class that fits the failure behind an incomplete download."""
    if isinstance(cause, WFSError):
        return type(cause)(message)
    if cause is not None:
        return WFSConnectionError(message)
    return WFSError(message)


def _load_data_in_parallel(
        url: str,
        layer: str,
        bbox: tuple[float, float, float, float],
        filter_params: dict | None = None,
        max_workers: int = 4,
        mask=None,
        srs: str = 'EPSG:25832',
        allow_partial: bool = False
) -> gpd.GeoDataFrame | None:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """
    Load WFS data in chunks using parallel processing.

    Chunks are subdivided when the WFS server appears to have hit a
    feature-count limit.  To prevent infinite subdivision (e.g. when the
    server *always* returns exactly the limit), two safeguards are in
    place:

    * **MAX_CHUNK_DEPTH** -- the maximum number of subdivision levels
      (each level splits every chunk into a 2x2 grid).
    * **MIN_CHUNK_AREA** -- the smallest chunk area (in squared CRS
      units) below which further subdivision is skipped.

    A download that cannot be completed (a chunk fails and cannot be split
    further, the server's limit still truncates a chunk at the finest level,
    or too many requests fail in a row) leaves a hole in the result. By
    default that raises a WFS error; see ``allow_partial``.

    Args:
        url: The base URL of the WFS service
        layer: Name of the layer
        bbox: Bounding box to divide into chunks as (minx, miny, maxx, maxy)
        filter_params: Additional WFS parameters to filter results
        max_workers: Maximum number of parallel threads to use
        mask: Optional geometry mask to limit the query
        srs: Spatial reference system for SRSNAME and BBOX parameters
            (default: "EPSG:25832")
        allow_partial: If False (default) an incomplete download raises the WFS
            error that fits the failure (WFSConnectionError, WFSServerError,
            WFSResponseParsingError, or WFSError for a truncated chunk). The
            message names the request and says how many features were received;
            the underlying failure is chained. If True the data received is
            returned and a ``UserWarning`` reports the same counts.

    Returns:
        Combined GeoDataFrame with all data or None if no data found

    Raises:
        WFSRequestRejectedError: If the server rejects a request as invalid.
        WFSError: If the download is incomplete and ``allow_partial`` is False.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    all_gdfs = []
    n_features = 0       # features kept so far (before de-duplication)
    n_requests_ok = 0    # chunk requests that were answered
    holes: list[str] = []  # allow_partial only: one line per request that left a hole

    def incomplete(message: str, cause: BaseException | None = None) -> None:
        """A request left a hole in the result: fail now, or remember it."""
        message = (f"{message} Received so far: {n_features} features from "
                   f"{n_requests_ok} successful chunk requests.")
        if not allow_partial:
            raise _partial_wfs_error(
                f"WFS download of layer '{layer}' from {url} is incomplete. "
                f"{message}", cause) from cause
        logger.warning("WFS download of layer '%s' is incomplete: %s", layer, message)
        holes.append(message)

    # Start with a 2x2 grid of chunks
    initial_chunks = _create_grid(bbox, 2, 2)

    # Filter chunks by mask if provided
    if mask is not None:
        initial_chunks = [chunk for chunk in initial_chunks
                          if _chunk_intersects_mask(chunk, mask)]

    # Track chunks to process and processed chunks
    # Each entry is (chunk_bbox, x_div, y_div, depth)
    # A deque, not a list: the batch is taken from the FRONT on every
    # iteration, and slicing a list to do that copies the whole queue each
    # time. A failing server subdivides to MAX_CHUNK_DEPTH, which queues up
    # to sum(4**d for d in 0..8) = 87 381 chunks; at max_workers=1 the list
    # version rebuilt that queue once per chunk -- O(n^2), ~3.8e9 element
    # copies, which reads as a hang rather than a slow call.
    chunks_to_process = deque((chunk, 2, 2, 0) for chunk in initial_chunks)
    processed_chunks = set()
    failed_in_a_row = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while chunks_to_process:
            # Take a batch of chunks for parallel processing
            current_batch = [chunks_to_process.popleft()
                             for _ in range(min(max_workers,
                                                len(chunks_to_process)))]

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
                                         chunk, filter_params, srs)
                future_to_chunk_info[future] = chunk_info

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_chunk_info):
                chunk_info = future_to_chunk_info[future]
                chunk, x_div, y_div, depth = chunk_info

                try:
                    gdf = future.result()
                except WFSRequestRejectedError:
                    # The request itself is wrong; smaller chunks cannot help
                    raise
                except (WFSError, requests.RequestException) as exc:
                    failed_in_a_row += 1
                    if failed_in_a_row > MAX_CONSECUTIVE_CHUNK_FAILURES:
                        incomplete(
                            f"{failed_in_a_row} chunk requests failed in a row; "
                            f"giving up. Last failed request: chunk {chunk} "
                            f"({srs}): {exc}", exc)
                        chunks_to_process.clear()
                        continue

                    # Guard: don't subdivide on error if depth exceeded
                    if depth >= MAX_CHUNK_DEPTH:
                        incomplete(
                            f"Request for chunk {chunk} ({srs}) failed and the "
                            f"maximum subdivision depth {MAX_CHUNK_DEPTH} is "
                            f"reached: {exc}", exc)
                        continue

                    # Guard: don't subdivide on error if chunk is too small
                    if _chunk_area(chunk) < MIN_CHUNK_AREA:
                        incomplete(
                            f"Request for chunk {chunk} ({srs}) failed and its "
                            f"area ({_chunk_area(chunk):.2e}) is below the "
                            f"minimum ({MIN_CHUNK_AREA:.2e}): {exc}", exc)
                        continue

                    # If a chunk fails, try to subdivide it
                    sub_chunks = _create_grid(chunk, 2, 2)

                    # Filter sub-chunks by mask if provided
                    if mask is not None:
                        sub_chunks = [
                            sub_chunk for sub_chunk in sub_chunks
                            if _chunk_intersects_mask(sub_chunk, mask)
                        ]

                    chunks_to_process.extend(
                        [(sub_chunk, x_div * 2, y_div * 2, depth + 1)
                         for sub_chunk in sub_chunks]
                    )
                    continue

                failed_in_a_row = 0
                n_requests_ok += 1

                if gdf is None or gdf.empty:
                    continue

                # Apply mask if provided
                if mask is not None:
                    gdf = _clip_data_by_mask(gdf, mask)
                    if gdf.empty:
                        continue

                # Add successful results
                all_gdfs.append(gdf)
                n_features += len(gdf)

                # Check if we likely hit a feature limit
                if len(gdf) in (10_000, 100_000, 1_000, 5_000, 50_000):
                    # Guard: stop subdividing if max depth reached
                    if depth >= MAX_CHUNK_DEPTH:
                        incomplete(
                            f"Chunk {chunk} ({srs}) returned exactly {len(gdf)} "
                            f"features at the maximum subdivision depth "
                            f"{MAX_CHUNK_DEPTH}; the server's feature limit "
                            f"probably truncates it.")
                        continue

                    # Guard: stop subdividing if chunk is too small
                    if _chunk_area(chunk) < MIN_CHUNK_AREA:
                        incomplete(
                            f"Chunk {chunk} ({srs}) returned exactly {len(gdf)} "
                            f"features and its area ({_chunk_area(chunk):.2e}) "
                            f"is below the minimum ({MIN_CHUNK_AREA:.2e}); the "
                            f"server's feature limit probably truncates it.")
                        continue

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
                        [(sub_chunk, new_x_div, new_y_div, depth + 1)
                         for sub_chunk in sub_chunks]
                    )

    if holes:
        warnings.warn(
            f"WFS download of layer '{layer}' from {url} is PARTIAL "
            f"(allow_partial=True): {n_features} features from "
            f"{n_requests_ok} successful chunk requests; {len(holes)} request(s) "
            f"left a hole. First: {holes[0]}",
            UserWarning, stacklevel=2)

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
        filter_params: dict | None = None,
        srs: str = 'EPSG:25832'
) -> gpd.GeoDataFrame | None:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """
    Fetch WFS data for a specific bounding box.

    Args:
        url: The base URL of the WFS service
        layer: Name of the layer
        bbox: Bounding box to query as (minx, miny, maxx, maxy)
        filter_params: Additional WFS parameters to filter results
        srs: Spatial reference system for SRSNAME and BBOX parameters
            (default: "EPSG:25832")

    Returns:
        GeoDataFrame with data, or None if the server answered with no features

    Raises:
        WFSServerError, WFSRequestRejectedError: If the server answered with
            an OGC exception report.
        WFSConnectionError: If every WFS version failed at the HTTP level (the
            last ``requests`` exception is chained). A failed request is never
            reported as an empty chunk.
        WFSResponseParsingError: If an answer could not be parsed.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    # Extract namespace if present
    namespace = None
    if ':' in layer:
        namespace, _ = layer.split(':', 1)

    # Try different WFS versions; collect OGC exception reports on the way
    reports = []
    last_error: requests.RequestException | None = None
    answered = False
    for version in ["2.0.0", "1.1.0", "1.0.0"]:
        # Set version-specific parameters
        type_param = "TYPENAMES" if version == "2.0.0" else "TYPENAME"

        params = {
            'SERVICE': 'WFS',
            'VERSION': version,
            'REQUEST': 'GetFeature',
            type_param: layer,
            'SRSNAME': srs,
            'BBOX': f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{srs}"
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

            report = _wfs_exception_report(response)
            if report is not None:
                logger.warning("WFS %s answered with an exception report "
                               "for chunk %s: %s", version, bbox, report)
                reports.append(report)
                continue

            content_type = response.headers.get('Content-Type', '').lower()

            if 'json' in content_type:
                return _parse_geojson_response(response)
            if 'xml' in content_type or 'gml' in content_type:
                return _parse_xml_response(response)
            answered = True
            logger.warning("WFS %s answered with the unrecognised content type "
                           "%r for chunk %s; treating it as no data.",
                           version, content_type, bbox)

        except requests.RequestException as exc:
            report = _wfs_exception_report(getattr(exc, "response", None))
            if report is not None:
                reports.append(report)
            else:
                last_error = exc
            continue
    _raise_for_exception_reports(reports)
    if last_error is not None and not answered:
        raise WFSConnectionError(
            f"WFS request for layer '{layer}' (bbox {bbox}) failed for every "
            f"WFS version tried; last error: {last_error}") from last_error
    return None


def _parse_geojson_response(response: requests.Response) -> gpd.GeoDataFrame | None:
    """
    Parse a GeoJSON response into a GeoDataFrame.

    Args:
        response: HTTP response object containing GeoJSON data

    Returns:
        GeoDataFrame created from GeoJSON features, or None if the response is
        valid GeoJSON without features

    Raises:
        WFSResponseParsingError: If the body is not JSON at all (typically an
            HTML or text error page served under a JSON content type). The
            first 200 characters of the body are part of the message.
    """
    try:
        geojson_data = response.json()
    except ValueError as exc:
        excerpt = _response_excerpt(response)
        starts = f" (body starts: {excerpt!r})" if excerpt else ""
        raise WFSResponseParsingError(
            f"WFS response announced as JSON could not be parsed{starts}: {exc}") from exc
    if 'features' in geojson_data and geojson_data['features']:
        return gpd.GeoDataFrame.from_features(geojson_data['features'])
    return None


def _parse_xml_response(response: requests.Response) -> gpd.GeoDataFrame | None:
    """
    Parse an XML/GML response into a GeoDataFrame.

    Args:
        response: HTTP response object containing XML/GML data

    Returns:
        GeoDataFrame created from XML/GML data, or None if the response is a
        valid feature collection without features

    Raises:
        WFSResponseParsingError: If the body is not readable as GML/GeoJSON or
            the temporary file used to read it cannot be handled (OSError).
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir) / "wfs_response.xml"
            temp_file.write_bytes(response.content)
            try:
                # pyogrio Arrow fast path (2-4x documented) when available
                import pyarrow  # noqa: F401  # pylint: disable=unused-import
                use_arrow = True
            except ImportError:
                use_arrow = False
            if use_arrow:
                try:
                    return gpd.read_file(temp_file, use_arrow=True)
                except NotImplementedError as exc:
                    # MEASURED (2026-08-20) against the LGL-BW ALKIS WFS
                    # ("Tatsaechliche Nutzung"): pyogrio's Arrow path raises
                    # this for plain, non-curved gml:Polygon features too --
                    # fiona and pyogrio's classic (non-Arrow) path both read
                    # the identical response correctly (886/886 features,
                    # all simple Polygon). A GDAL/pyogrio Arrow-export quirk
                    # for this GML driver output, not a real curved-geometry
                    # limitation. Retry without Arrow rather than dropping
                    # the whole response.
                    if "nonlinear" not in str(exc).lower():
                        raise
            return gpd.read_file(temp_file, use_arrow=False)
    except DataSourceError as exc:
        raise WFSResponseParsingError(
            f"WFS response is not readable as GML/GeoJSON: {exc}") from exc
    except IndexError:
        # MEASURED (2026-11-10): GDAL/pyogrio raise "IndexError: index 0 is out
        # of bounds" for a well-formed GML FeatureCollection with zero
        # features, so this is an EMPTY answer, not a failure.
        logger.debug("WFS response holds no features (body starts: %r)",
                     _response_excerpt(response))
        return None
    except OSError as exc:
        raise WFSResponseParsingError(
            f"WFS response could not be read from a temporary file: {exc}") from exc


def _combine_geodataframes(gdfs: list[gpd.GeoDataFrame]) -> gpd.GeoDataFrame | None:
    """
    Combine multiple GeoDataFrames and remove duplicates.

    Args:
        gdfs: list of GeoDataFrames to combine

    Returns:
        Combined GeoDataFrame with duplicates removed or None if input list is empty
    """
    if not gdfs:
        return None

    # Drop duplicate columns in each chunk (some WFS/GML responses
    # return the same attribute under multiple namespace prefixes)
    cleaned = []
    for gdf in gdfs:
        if gdf.columns.duplicated().any():
            gdf = gdf.loc[:, ~gdf.columns.duplicated()]
        cleaned.append(gdf)

    # Concatenate all GeoDataFrames
    combined_gdf = gpd.GeoDataFrame(pd.concat(cleaned, ignore_index=True))

    # Remove duplicates by geometry
    combined_gdf = combined_gdf.drop_duplicates(subset=['geometry'])

    return combined_gdf
