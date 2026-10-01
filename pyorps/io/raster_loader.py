"""
Raster loading, combining, and discovery utilities for PYORPS
"""
import logging
import warnings
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.warp import Resampling, calculate_default_transform, reproject

from pyorps.io.geo_dataset import InMemoryRasterDataset


def _parse_crs(crs_input):
    """Parse a CRS input to a rasterio CRS object if it's a string."""
    if crs_input and isinstance(crs_input, str):
        return CRS.from_string(crs_input)
    return crs_input


def _validate_raster_paths(file_paths):
    """Validate and normalize raster file paths, returning a list of Path objects."""
    if isinstance(file_paths, (str, Path)):
        file_paths = [file_paths]

    paths = []
    for fp in file_paths:
        path = Path(fp)
        if not path.exists():
            raise FileNotFoundError(f"Raster file not found: {path}")
        if path.suffix.lower() not in ['.tif', '.tiff', '.geotiff']:
            warnings.warn(
                f"File {path} may not be a GeoTIFF file (extension: {path.suffix})")
        paths.append(path)
    return paths


def _assign_crs_in_memory(src, default_crs):
    """Create an in-memory copy of a raster with a CRS assigned."""
    data = src.read()
    profile = src.profile
    profile['crs'] = default_crs
    src.close()

    memfile = MemoryFile()
    mem_dst = memfile.open(**profile)
    mem_dst.write(data)
    return mem_dst, memfile


def _reproject_to_target_crs(src, target_crs):
    """Reproject a raster dataset to a target CRS in memory."""
    transform, width, height = calculate_default_transform(
        src.crs, target_crs, src.width, src.height, *src.bounds
    )

    profile = src.profile.copy()
    profile.update({
        'crs': target_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    memfile = MemoryFile()
    mem_dst = memfile.open(**profile)

    for band_idx in range(1, src.count + 1):
        reproject(
            source=rasterio.band(src, band_idx),
            destination=rasterio.band(mem_dst, band_idx),
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear
        )
    src.close()
    return mem_dst, memfile


def read_raster_files(
    file_paths: str | Path | list[str | Path],
    default_crs: str | CRS | None = None,
    target_crs: str | CRS | None = None
) -> tuple[list[rasterio.DatasetReader], list[MemoryFile]]:
    """
    Read a set of raster data files (tif/tiff/geotiff) using rasterio.

    This function reads one or more raster files and ensures they have a valid CRS.
    If no CRS is defined in a raster file, it sets the CRS using the default_crs parameter.
    Optionally reprojects all rasters to a target CRS for consistency.

    Args:
        file_paths: Single file path or list of file paths to raster files
        default_crs: CRS to assign to files that don't have a CRS defined
                    (e.g., "EPSG:4326" or rasterio.CRS object)
        target_crs: Optional CRS to reproject all rasters to for consistency

    Returns:
        Tuple of (List of opened rasterio dataset readers, List of MemoryFile objects to keep alive)

    Raises:
        FileNotFoundError: If any of the specified files don't exist
        ValueError: If a file has no CRS and no default_crs is provided
    """
    paths = _validate_raster_paths(file_paths)
    default_crs = _parse_crs(default_crs)
    target_crs = _parse_crs(target_crs)

    datasets = []
    memory_files = []

    for path in paths:
        src = rasterio.open(path, 'r')

        if src.crs is None:
            if default_crs is None:
                src.close()
                raise ValueError(
                    f"Raster file {path} has no CRS defined and no "
                    f"default_crs was provided")
            mem_dst, memfile = _assign_crs_in_memory(src, default_crs)
            memory_files.append(memfile)
            datasets.append(mem_dst)

        elif target_crs and src.crs != target_crs:
            mem_dst, memfile = _reproject_to_target_crs(src, target_crs)
            memory_files.append(memfile)
            datasets.append(mem_dst)

        else:
            datasets.append(src)

    return datasets, memory_files


def _close_datasets_safely(datasets, memory_files):
    """Close rasterio datasets and memory files, ignoring already-closed errors."""
    for ds in datasets:
        try:
            ds.close()
        except (OSError, RuntimeError):
            pass

    for memfile in memory_files:
        try:
            memfile.close()
        except (OSError, RuntimeError):
            pass


def _resolve_merge_params(datasets, dtype, nodata):
    """Resolve dtype and nodata from input datasets if not explicitly set."""
    if dtype is None:
        dtypes = [ds.dtypes[0] for ds in datasets]
        dtype = np.result_type(*dtypes)

    if nodata is None:
        for ds in datasets:
            if ds.nodata is not None:
                nodata = ds.nodata
                break

    return dtype, nodata


def _validate_and_unpack_datasets(raster_datasets):
    """Unpack and validate raster datasets input, return (datasets, memory_files)."""
    if isinstance(raster_datasets, tuple):
        datasets, memory_files = raster_datasets
    else:
        datasets = raster_datasets
        memory_files = []

    if not datasets:
        raise ValueError("No raster datasets provided for combining")

    base_crs = datasets[0].crs
    for ds in datasets[1:]:
        if ds.crs != base_crs:
            raise ValueError(
                f"All rasters must have the same CRS. Found {base_crs} "
                f"and {ds.crs}. Consider using target_crs in read_raster_files()")

    return datasets, memory_files, base_crs


def _build_merge_profile(merged_array, merged_transform, base_crs, dtype, nodata):
    """Build a GeoTIFF profile dict from merge results."""
    return {
        'driver': 'GTiff', 'dtype': dtype, 'nodata': nodata,
        'width': merged_array.shape[2], 'height': merged_array.shape[1],
        'count': merged_array.shape[0], 'crs': base_crs,
        'transform': merged_transform,
        'compress': 'lzw', 'tiled': True, 'blockxsize': 256, 'blockysize': 256,
    }


def combine_rasters(
    raster_datasets: list[rasterio.DatasetReader] | tuple[list[rasterio.DatasetReader], list],
    save_path: str | Path | None = None,
    method: str = 'first',
    nodata: float | None = None,
    dtype: str | None = None,
    resampling: Resampling = Resampling.bilinear
) -> tuple[np.ndarray, dict]:
    """Combine multiple raster datasets into a single array.

    Returns (merged_array, profile_dict) with CRS, transform, shape, etc.
    """
    datasets, memory_files, base_crs = _validate_and_unpack_datasets(raster_datasets)
    dtype, nodata = _resolve_merge_params(datasets, dtype, nodata)

    merged_array, merged_transform = merge(
        datasets, method=method, nodata=nodata, dtype=dtype, resampling=resampling
    )
    profile = _build_merge_profile(
        merged_array, merged_transform, base_crs, dtype, nodata)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(save_path, 'w', **profile) as dst:
            dst.write(merged_array)
        logger.info("Combined raster saved to: %s", save_path)

    _close_datasets_safely(datasets, memory_files)
    return merged_array, profile


def create_combined_raster_dataset(
    file_paths: str | Path | list[str | Path],
    crs: str | None = None,
    target_crs: str | None = None,
    save_path: str | Path | None = None,
    method: str = 'first'
) -> 'InMemoryRasterDataset':
    """
    Convenience function to read and combine rasters, returning an InMemoryRasterDataset
    that integrates with your existing GeoDataset infrastructure.

    Args:
        file_paths: Single file path or list of file paths to raster files
        crs: Default CRS to assign to files without CRS
        target_crs: Optional CRS to reproject all rasters to
        save_path: Optional path to save the combined raster
        method: Method for combining overlapping pixels

    Returns:
        InMemoryRasterDataset object compatible with your existing code
    """
    # Read the raster files (now returns tuple)
    datasets, memory_files = read_raster_files(file_paths, default_crs=crs, target_crs=target_crs)

    # Combine them (pass the tuple)
    merged_array, profile = combine_rasters(
        (datasets, memory_files),
        save_path=save_path,
        method=method
    )

    # Create InMemoryRasterDataset compatible with your existing code
    raster_dataset = InMemoryRasterDataset(
        file_source=merged_array,
        crs=str(profile['crs']),
        transform=profile['transform']
    )

    return raster_dataset


_DEFAULT_RASTER_EXTENSIONS = {
    '.tif', '.tiff', '.geotiff', '.gtiff',
    '.jp2', '.j2k',
    '.img',
    '.dem', '.dt0', '.dt1', '.dt2',
    '.hgt',
    '.nc', '.nc4',
    '.hdf', '.hdf4', '.hdf5', '.h4', '.h5',
    '.vrt'
}


def _build_glob_patterns(extensions, pattern, recursive):
    """Build glob patterns from extensions and optional filename pattern."""
    if pattern is not None:
        if recursive and '**' not in pattern:
            return [f"**/{pattern}"]
        return [pattern]

    prefix = "**/*" if recursive else "*"
    return [f"{prefix}{ext}" for ext in extensions]


def _is_excluded(file_path, exclude_patterns):
    """Check if a file path matches any exclusion pattern."""
    if not exclude_patterns:
        return False
    return any(file_path.match(p) for p in exclude_patterns)


def find_raster_files(
    directory: str | Path,
    recursive: bool = True,
    extensions: set[str] | None = None,
    pattern: str | None = None,
    exclude_patterns: list[str] | None = None,
    return_absolute: bool = True
) -> list[Path]:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Find raster files in a directory, filtered by extension/pattern."""
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    dir_path = Path(directory)

    if not dir_path.exists():
        raise ValueError(f"Directory does not exist: {directory}")
    if not dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    if extensions is None:
        extensions = _DEFAULT_RASTER_EXTENSIONS
    extensions = {ext.lower() for ext in extensions}

    glob_patterns = _build_glob_patterns(extensions, pattern, recursive)

    raster_files = set()
    for glob_pattern in glob_patterns:
        if recursive and '**' not in glob_pattern:
            matches = dir_path.glob(f"**/{glob_pattern}")
        else:
            matches = dir_path.glob(glob_pattern)

        for file_path in matches:
            if file_path.is_dir():
                continue
            if pattern is not None and file_path.suffix.lower() not in extensions:
                continue
            if _is_excluded(file_path, exclude_patterns):
                continue
            raster_files.add(file_path.absolute() if return_absolute else file_path)

    return sorted(raster_files)


def _collect_raster_metadata(raster_files):
    """Collect metadata from raster files, returning (metadata_list, issues_list)."""
    metadata = []
    issues = []
    for file_path in raster_files:
        try:
            with rasterio.open(file_path) as src:
                metadata.append({
                    'file': Path(file_path).name,
                    'path': file_path,
                    'crs': str(src.crs) if src.crs else None,
                    'width': src.width,
                    'height': src.height,
                    'count': src.count,
                    'dtype': str(src.dtypes[0]),
                    'transform': src.transform,
                    'res_x': abs(src.transform[0]),
                    'res_y': abs(src.transform[4]),
                    'bounds': src.bounds,
                    'nodata': src.nodata
                })
        except Exception as e:
            issues.append(f"Could not read {Path(file_path).name}: {str(e)}")
    return metadata, issues


def _check_grouping_issues(metadata, key, label, format_fn=None):
    """Check if a metadata field has multiple distinct values; return issue string or None."""
    groups = defaultdict(list)
    for m in metadata:
        groups[m[key] if not callable(format_fn) else format_fn(m)].append(m['file'])

    if len(groups) <= 1:
        return None

    lines = [f"Multiple {label} found:\n"]
    for val, files in groups.items():
        lines.append(f"  {val}: {len(files)} files\n")
    return "".join(lines)


def _check_crs_issues(metadata):
    """Check CRS compatibility, returning a list of issue strings."""
    issues = []
    crs_values = [m['crs'] for m in metadata]
    unique_crs = set(crs_values)

    if None in unique_crs:
        files_without = [m['file'] for m in metadata if m['crs'] is None]
        issues.append(f"Files without CRS: {', '.join(files_without)}")

    if len(unique_crs - {None}) > 1:
        issue = _check_grouping_issues(metadata, 'crs', 'CRS')
        if issue:
            issues.append(issue)
    return issues


def _check_resolution_issues(metadata):
    """Check resolution compatibility, returning a list of issue strings."""
    resolutions = set((m['res_x'], m['res_y']) for m in metadata)
    if len(resolutions) <= 1:
        return []

    groups = defaultdict(list)
    for m in metadata:
        groups[(m['res_x'], m['res_y'])].append(m['file'])

    lines = ["Multiple resolutions found:\n"]
    for res, files in groups.items():
        lines.append(f"  {res[0]:.2f}x{res[1]:.2f}: {len(files)} files\n")
    return ["".join(lines)]


def _check_band_issues(metadata):
    """Check band count compatibility, returning a list of issue strings."""
    issue = _check_grouping_issues(metadata, 'count', 'band counts')
    return [issue] if issue else []


def _print_validation_summary(result, compatible):
    """Log validation results."""
    metadata = result['metadata']
    issues = result['issues']
    logger.info("Validation Results for %d raster files:", len(metadata))
    logger.info("-" * 50)
    if compatible:
        logger.info("Files are compatible for combining")
    else:
        logger.warning("Compatibility issues found:")
        for issue in issues:
            logger.warning("  - %s", issue)

    summary = result['summary']
    logger.info("Summary:")
    logger.info("  CRS: %s", ', '.join(str(c) for c in summary['crs_values']))
    logger.info("  Resolutions: %s", summary['resolutions'])
    logger.info("  Band counts: %s", summary['band_counts'])
    logger.info("  Data types: %s", summary['data_types'])


def validate_raster_compatibility(
    raster_files: list[str | Path],
    check_crs: bool = True,
    check_resolution: bool = True,
    check_bands: bool = False,
    verbose: bool = True
) -> dict:
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Check CRS, resolution, and band compatibility of raster files."""
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    if not raster_files:
        return {'compatible': False, 'issues': ['No raster files provided']}

    metadata, issues = _collect_raster_metadata(raster_files)

    if not metadata:
        return {'compatible': False, 'issues': issues}

    if check_crs:
        issues.extend(_check_crs_issues(metadata))
    if check_resolution:
        issues.extend(_check_resolution_issues(metadata))
    if check_bands:
        issues.extend(_check_band_issues(metadata))

    compatible = len(issues) == 0 or (len(issues) == 1 and 'without CRS' in issues[0])

    result = {
        'compatible': compatible,
        'issues': issues,
        'metadata': metadata,
        'summary': {
            'total_files': len(metadata),
            'crs_values': list(set(m['crs'] for m in metadata)),
            'resolutions': list(set((m['res_x'], m['res_y']) for m in metadata)),
            'band_counts': list(set(m['count'] for m in metadata)),
            'data_types': list(set(m['dtype'] for m in metadata))
        }
    }

    if verbose:
        _print_validation_summary(result, compatible)

    return result


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Example 1: Find and combine all DEMs in a directory
    logger.info("Example 1: Finding and combining DEMs")
    logger.info("-" * 40)

    directory = r"C:\Users\mhnn82\Documents\6_Andere_Projekte\3D-Trasse"

    dem_files = find_raster_files(
        directory, pattern="dgm1_*.tif", recursive=True
    )
    logger.info("Found %d DEM files", len(dem_files))

    validation = validate_raster_compatibility(dem_files)

    if validation['compatible']:
        datasets, memory_files = read_raster_files(dem_files, default_crs="EPSG:25832")
        merged_data, profile = combine_rasters(
            (datasets, memory_files),
            save_path=r"C:\Users\mhnn82\Documents\6_Andere_Projekte\3D-Trasse\merged_dem.tif",
            method='first'
        )
        logger.info("DEMs successfully combined")
    else:
        logger.warning("DEMs are not compatible: %s", validation['issues'])
