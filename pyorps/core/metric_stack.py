"""
PYORPS: An Open-Source Tool for Automated Power Line Routing

MetricStack: aligned per-cell metric layers for feasibility routing.

A :class:`MetricStack` holds K named float32 metric layers (cost, landscape,
permit, ...) plus a shared forbidden mask, an optional category band (feature
class ids for reporting breakdowns), and an optional DEM band — all on one
grid with one transform. Alignment is a structural invariant: bands can only
be windowed together.

Forbidden semantics: a cell that is forbidden in ANY layer (value == 65535
exactly, inf or NaN at ingestion) is forbidden for the whole stack,
regardless of objective weights. Layer arrays themselves are kept finite;
finite values above 65535 are legitimate (float layers are not bound by
the uint16 range).

Phase 2 of docs/superpowers/plans/2026-08-04-feasibility-multi-objective-routing.md.
The objective combination (combine/quantize) arrives with Phase 3.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass

import numpy as np
from rasterio.features import rasterize as rio_rasterize
from rasterio.transform import Affine
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

from .exceptions import MetricStackError
from .objective import RESERVED_LAYERS, Objective

#: Sentinel marking forbidden cells (mirrors IMPASSABLE_CELL_COST; not
#: imported from .types to avoid a circular import).
FORBIDDEN_VALUE = 65535.0

#: Reserved band names used in the multi-band GeoTIFF layout.
_BAND_FORBIDDEN = "__forbidden__"
_BAND_CATEGORY = "__category__"
_BAND_DEM = "__dem__"
_RESERVED_BAND_NAMES = {_BAND_FORBIDDEN, _BAND_CATEGORY, _BAND_DEM}

_TAG_KEY = "pyorps_metric_stack"


def reproject_to_grid(source: np.ndarray, src_transform: Affine, src_crs,
                      dst_shape: tuple[int, int], dst_transform: Affine,
                      dst_crs) -> np.ndarray:
    """Reproject a raster onto a target grid (bilinear, NaN where uncovered).

    Transform-correct for any source resolution or extent — unlike
    shape-based zooming, which silently assumes identical extents.
    """
    from rasterio.warp import Resampling, reproject

    destination = np.full(dst_shape, np.nan, dtype=np.float32)
    reproject(
        source=np.ascontiguousarray(source, dtype=np.float32),
        destination=destination,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
    )
    return destination

#: Largest usable quantized weight (65535 is the forbidden sentinel).
_MAX_WEIGHT = 65534.0


@dataclass
class CombineResult:
    """Result of combining a stack under an objective (plan section 5).

    Attributes:
        weights: uint16 search raster (65535 = forbidden). On the legacy
            zero-copy path this IS the original raster object.
        scale: quantization scale — ``weights ≈ feasibility * scale``.
            Divide kernel distances by it (× cell length) to approximate
            the achieved objective; the evaluator recomputes exactly.
        feasibility: float32 combined surface F (None on the zero-copy
            path; values on forbidden cells are meaningless).
        resolution: user-units per quantization level (``F_max / 65534``;
            0.0 on the zero-copy path — no quantization happened).
        legacy_passthrough: True when weights is the untouched original
            uint16 raster (byte-identical legacy behavior).
    """

    weights: np.ndarray
    scale: float
    feasibility: np.ndarray | None
    resolution: float
    legacy_passthrough: bool


class MetricStack:
    """Aligned float32 metric layers sharing one grid and forbidden mask.

    Parameters:
        transform: Affine transform of the grid.
        crs: Coordinate reference system (any rasterio-compatible value).
    """

    def __init__(self, transform: Affine, crs):
        if transform is None:
            raise MetricStackError("MetricStack requires a transform")
        self.transform = transform
        self.crs = crs
        self._layers: dict[str, np.ndarray] = {}
        self.forbidden_mask: np.ndarray | None = None
        self.category: np.ndarray | None = None
        self.category_labels: dict[int, str] | None = None
        self.dem: np.ndarray | None = None
        self._legacy_raster: np.ndarray | None = None

        # 0.1 % tolerance: real-world rasters are slightly anisotropic
        # after reprojection; only warn on genuinely rectangular cells.
        if abs(abs(transform.a) - abs(transform.e)) > 1e-3 * abs(transform.a):
            warnings.warn(
                f"MetricStack grid is anisotropic "
                f"({transform.a} x {transform.e}); pyorps assumes square "
                f"cells — distances use |transform.a|.",
                UserWarning, stacklevel=2)

    # ------------------------------------------------------------ properties

    @property
    def shape(self) -> tuple[int, int] | None:
        if self.forbidden_mask is not None:
            return self.forbidden_mask.shape
        if self._legacy_raster is not None:
            return self._legacy_raster.shape
        return None

    @property
    def cell_size(self) -> float:
        return float(abs(self.transform.a))

    @property
    def layer_names(self) -> list[str]:
        self._materialize_legacy()
        return list(self._layers)

    @property
    def is_legacy_alias(self) -> bool:
        """True while the stack merely aliases a single uint16 raster."""
        return self._legacy_raster is not None and not self._layers

    @property
    def legacy_raster(self) -> np.ndarray | None:
        """The aliased uint16 raster (legacy fast path), if any."""
        return self._legacy_raster

    # -------------------------------------------------------------- building

    @classmethod
    def from_single_raster(cls, raster: np.ndarray, transform: Affine,
                           crs) -> "MetricStack":
        """Zero-copy legacy alias around a single uint16 cost raster.

        The raster is NOT copied or converted; the float 'cost' layer and
        the forbidden mask materialize lazily on first layer access. Until
        then :attr:`legacy_raster` is the byte-identical original.
        """
        if raster.ndim != 2:
            raise MetricStackError(
                f"from_single_raster expects a 2D raster, "
                f"got shape {raster.shape}")
        stack = cls(transform, crs)
        stack._legacy_raster = raster
        return stack

    def _materialize_legacy(self) -> None:
        if self._legacy_raster is None or self._layers:
            return
        raster = self._legacy_raster
        forbidden = raster >= FORBIDDEN_VALUE
        values = raster.astype(np.float32)
        values[forbidden] = 0.0
        self._layers["cost"] = values
        self._merge_forbidden(forbidden)

    def _merge_forbidden(self, forbidden: np.ndarray) -> None:
        if self.forbidden_mask is None:
            self.forbidden_mask = forbidden.astype(bool)
        else:
            self.forbidden_mask |= forbidden

    def _check_shape(self, array: np.ndarray, what: str) -> None:
        if array.ndim != 2:
            raise MetricStackError(
                f"{what} must be 2D, got shape {array.shape}")
        shape = self.shape
        if shape is not None and array.shape != shape:
            raise MetricStackError(
                f"{what} shape {array.shape} does not match the stack "
                f"shape {shape} — all bands must share one grid. Resample "
                f"before adding.")

    def add_layer(self, name: str, values: np.ndarray,
                  hard_max: float | None = None,
                  hard_min: float | None = None) -> None:
        """Add a metric layer (float32; forbidden values join the mask).

        The sentinel value 65535 (exactly), inf and NaN mark the cell
        forbidden for the WHOLE stack; the stored layer is finite
        (forbidden cells hold 0). Finite values above 65535 are legitimate
        metric values — float layers are not bound by the uint16 range.

        Parameters:
            hard_max / hard_min: Optional hard constraints — cells whose
                value exceeds/undershoots the bound become forbidden
                (e.g. ``terrain_slope`` with ``hard_max=45``).
        """
        if not isinstance(name, str) or not name:
            raise MetricStackError(
                f"Layer name must be a non-empty string, got {name!r}")
        if name in RESERVED_LAYERS:
            raise MetricStackError(
                f"'{name}' is a reserved objective layer "
                f"({list(RESERVED_LAYERS)}) and cannot be a raster layer.")
        if name in _RESERVED_BAND_NAMES:
            raise MetricStackError(
                f"'{name}' is a reserved band name.")
        self._materialize_legacy()
        if name in self._layers:
            raise MetricStackError(
                f"Layer '{name}' already exists in the stack.")
        self._check_shape(values, f"Layer '{name}'")

        values = np.asarray(values, dtype=np.float32).copy()
        with np.errstate(invalid="ignore"):
            forbidden = (~np.isfinite(values)) | (values == FORBIDDEN_VALUE)
        if np.any(values[np.isfinite(values)] < 0):
            raise MetricStackError(
                f"Layer '{name}' contains negative values — metric layers "
                f"are per-meter intensities and must be >= 0.")
        if hard_max is not None:
            forbidden |= values > hard_max
        if hard_min is not None:
            forbidden |= values < hard_min
        values[forbidden] = 0.0
        self._layers[name] = values
        self._merge_forbidden(forbidden)

    def __contains__(self, name: str) -> bool:
        self._materialize_legacy()
        return name in self._layers

    def __getitem__(self, name: str) -> np.ndarray:
        self._materialize_legacy()
        try:
            return self._layers[name]
        except KeyError:
            raise MetricStackError(
                f"No metric layer '{name}' in the stack. Available: "
                f"{list(self._layers)}") from None

    def ensure_layers(self, names) -> None:
        """Raise when any of the given layer names is missing."""
        self._materialize_legacy()
        missing = [n for n in names if n not in self._layers]
        if missing:
            raise MetricStackError(
                f"Missing metric layer(s) {missing}. Available: "
                f"{list(self._layers)}")

    def attach_category(self, category: np.ndarray,
                        labels: dict[int, str] | None = None) -> None:
        """Attach the feature-class id band (uint16; 0 = no class)."""
        self._check_shape(category, "Category band")
        category = np.asarray(category)
        if category.dtype != np.uint16:
            if category.min() < 0 or category.max() > 65535:
                raise MetricStackError(
                    "Category ids must fit uint16 (0..65535)")
            category = category.astype(np.uint16)
        self.category = category
        self.category_labels = dict(labels) if labels else None

    def attach_dem(self, dem: np.ndarray,
                   resample: bool = True) -> None:
        """Attach the DEM band, resampling and sanitizing it.

        A DEM of a different shape is resampled onto the stack grid
        (bilinear, the proven pattern from ConstrainedPathFinder). NaN/inf
        cells (nodata voids) mark the cell forbidden and are filled with
        the mean finite height so no non-finite value can ever reach a
        kernel ((int)(NaN * x) is undefined on GPU).
        """
        dem = np.asarray(dem, dtype=np.float32)
        if dem.ndim == 3:
            dem = dem[0]
        if dem.ndim != 2:
            raise MetricStackError(
                f"DEM must be 2D, got shape {dem.shape}")
        shape = self.shape
        if shape is None:
            raise MetricStackError(
                "Attach at least one layer (or use from_single_raster) "
                "before attaching a DEM — the stack grid is undefined.")
        if dem.shape != shape:
            if not resample:
                raise MetricStackError(
                    f"DEM shape {dem.shape} does not match stack shape "
                    f"{shape} and resample=False")
            from scipy.ndimage import zoom
            factors = (shape[0] / dem.shape[0], shape[1] / dem.shape[1])
            dem = zoom(dem, factors, order=1).astype(np.float32)
            if dem.shape != shape:  # guard against rounding artifacts
                dem = dem[:shape[0], :shape[1]]

        invalid = ~np.isfinite(dem)
        if np.any(invalid):
            finite = dem[~invalid]
            fill = float(finite.mean()) if finite.size else 0.0
            dem = dem.copy()
            dem[invalid] = fill
            self._merge_forbidden(invalid)
            warnings.warn(
                f"DEM contains {int(invalid.sum())} non-finite cell(s) — "
                f"marked forbidden and filled with the mean height "
                f"({fill:.1f} m) for kernel safety.",
                UserWarning, stacklevel=2)
        self.dem = dem

    def derive_terrain_slope(self) -> np.ndarray:
        """Per-cell terrain slope magnitude ``|∇DEM|`` in percent.

        This is the direction-independent CROSS-SLOPE constructability
        measure (a side-hill trench is hard to build even when traversed
        dead-level). The along-route grade is the reserved edge metric
        ``gradient`` computed inside the kernels — a cell layer cannot
        represent it (plan section 0).
        """
        if self.dem is None:
            raise MetricStackError(
                "derive_terrain_slope requires a DEM — attach_dem first.")
        dy, dx = np.gradient(self.dem.astype(np.float64), self.cell_size)
        return (np.sqrt(dx * dx + dy * dy) * 100.0).astype(np.float32)

    # ----------------------------------------------------------- windowing

    def window(self, window: Window) -> "MetricStack":
        """Return a stack windowed to the given rasterio Window.

        One slice operation applied to EVERY band — alignment can not
        drift. Band arrays are views where numpy allows it.
        """
        self._materialize_legacy()
        rows, cols = window.toslices()
        rows = slice(int(rows.start), int(rows.stop))
        cols = slice(int(cols.start), int(cols.stop))

        sub = MetricStack(window_transform(window, self.transform), self.crs)
        sub._layers = {name: arr[rows, cols]
                       for name, arr in self._layers.items()}
        if self.forbidden_mask is not None:
            sub.forbidden_mask = self.forbidden_mask[rows, cols]
        if self.category is not None:
            sub.category = self.category[rows, cols]
            sub.category_labels = self.category_labels
        if self.dem is not None:
            sub.dem = self.dem[rows, cols]
        return sub

    def mask_outside(self, geometry) -> None:
        """Mark every cell outside the geometry forbidden.

        Equivalent to RasterHandler.apply_geometry_mask setting the
        outside to the max/sentinel value — expressed on the shared mask.
        """
        shape = self.shape
        if shape is None:
            raise MetricStackError("Cannot mask an empty stack")
        self._materialize_legacy()
        inside = rio_rasterize(
            [(geometry, 1)],
            out_shape=shape,
            transform=self.transform,
            fill=0,
            dtype=np.uint8,
        )
        self._merge_forbidden(inside == 0)

    # ---------------------------------------------------------- combination

    def combine(self, objective: Objective,
                quantize: bool = True) -> CombineResult:
        """Combine the stack under an objective into the search raster.

        ``F[cell] = sum_k w_k * layer_k[cell] (+ w_length)`` — exact linear
        scalarization — then **pure-scaling** quantization into uint16
        [1, 65534] (an offset would inject a hidden length term and change
        the optimum); forbidden cells become 65535.

        Fast path: a pure-cost objective on a legacy single-raster alias
        returns the ORIGINAL uint16 raster, zero copy, zero quantization —
        byte-identical to today's behavior (scale = 1 / w_cost).

        Emits diagnostics (plan section 5.2): a warning when quantization
        collapses distinct feature classes or when the median cell value
        quantizes below 8 levels (one layer dominating the range).

        Parameters:
            quantize: True (default) produces the uint16 search raster.
                False produces LOSSLESS float32 weights (forbidden cells
                carry +inf, scale = 1.0, no diagnostics needed) for the
                float-capable backends — Phase 9 of the feasibility plan.
        """
        if not isinstance(objective, Objective):
            raise MetricStackError(
                f"combine() expects an Objective, got {type(objective)}")

        weights = objective.weights
        w_length = weights.get("length", 0.0)
        cell_weights = {name: w for name, w in weights.items()
                       if name not in RESERVED_LAYERS}

        # Zero-copy legacy fast path: only 'cost' weighted, no length term.
        if (quantize and self.is_legacy_alias and w_length == 0.0
                and set(cell_weights) == {"cost"} and cell_weights["cost"] > 0):
            return CombineResult(
                weights=self._legacy_raster,
                scale=1.0 / cell_weights["cost"],
                feasibility=None,
                resolution=0.0,
                legacy_passthrough=True,
            )

        self._materialize_legacy()
        if not self._layers:
            raise MetricStackError("Cannot combine an empty stack")
        objective.validate_layers(list(self._layers))

        shape = self.shape
        feasibility = np.zeros(shape, dtype=np.float32)
        for name, weight in cell_weights.items():
            if weight > 0:
                feasibility += np.float32(weight) * self._layers[name]
        if w_length > 0:
            # Implicit constant layer: contributes w_length per meter,
            # i.e. exactly w_length * path_length (plan section 4.3).
            feasibility += np.float32(w_length)

        traversable = ~self.forbidden_mask
        if not traversable.any():
            raise MetricStackError(
                "Every cell of the stack is forbidden — nothing to route.")

        f_valid = feasibility[traversable]
        f_max = float(f_valid.max())

        if not quantize:
            # Lossless float32 weights: forbidden = +inf, nothing rounded.
            float_weights = feasibility.copy()
            float_weights[self.forbidden_mask] = np.float32(np.inf)
            return CombineResult(
                weights=float_weights,
                scale=1.0,
                feasibility=feasibility,
                resolution=0.0,
                legacy_passthrough=False,
            )

        if f_max <= 0.0:
            warnings.warn(
                "The objective evaluates to 0 on every traversable cell "
                "(no positively weighted layer varies here). Using a "
                "uniform minimal weight — the search degenerates to a "
                "shortest-step path. Add a small 'length' weight to make "
                "this explicit.",
                UserWarning, stacklevel=2)
            weight_raster = np.ones(shape, dtype=np.uint16)
            scale = 1.0
            resolution = 0.0
        else:
            scale = _MAX_WEIGHT / f_max
            weight_raster = np.clip(
                np.rint(feasibility * np.float32(scale)),
                1.0, _MAX_WEIGHT).astype(np.uint16)
            resolution = f_max / _MAX_WEIGHT
            self._quantization_diagnostics(f_valid, scale)

        weight_raster[self.forbidden_mask] = int(FORBIDDEN_VALUE)

        return CombineResult(
            weights=weight_raster,
            scale=scale,
            feasibility=feasibility,
            resolution=resolution,
            legacy_passthrough=False,
        )

    @staticmethod
    def _quantization_diagnostics(f_valid: np.ndarray,
                                  scale: float) -> None:
        """Warn when uint16 quantization visibly degrades the objective."""
        # (b) median cell quantizes below 8 levels: one layer dominates.
        median_levels = float(np.median(f_valid)) * scale
        if median_levels < 8.0:
            warnings.warn(
                f"Quantization: the median traversable cell maps to "
                f"{median_levels:.1f} of 65534 levels — a large-magnitude "
                f"term dominates the objective and fine differences of "
                f"the other layers are lost. Rescale the weights (only "
                f"ratios matter) or reconsider the layer scales.",
                UserWarning, stacklevel=3)
            return

        # (a) distinct feature-class values collapsing onto one level.
        # Only meaningful for class-like surfaces; skip continuous ones.
        unique_f, counts = np.unique(f_valid, return_counts=True)
        if unique_f.size > 4096 or unique_f.size < 2:
            return
        quantized = np.clip(np.rint(unique_f * scale), 1.0, _MAX_WEIGHT)
        collapsed_cells = 0
        _, inverse, group_sizes = np.unique(
            quantized, return_inverse=True, return_counts=True)
        for group_index, size in enumerate(group_sizes):
            if size > 1:
                collapsed_cells += counts[inverse == group_index].sum()
        fraction = collapsed_cells / counts.sum()
        if fraction > 0.01:
            warnings.warn(
                f"Quantization collapsed distinct feature-class values on "
                f"{fraction:.1%} of traversable cells — differences that "
                f"exist in the float objective are invisible to the "
                f"search. Reduce the dynamic range of the weights.",
                UserWarning, stacklevel=3)

    # ------------------------------------------------------------------ I/O

    def save(self, path: str) -> None:
        """Persist the stack as a multi-band float32 GeoTIFF.

        Band layout: metric layers (band descriptions carry the names),
        then '__forbidden__', optional '__category__', optional '__dem__'.
        Stack metadata (layer names, category labels) goes into the
        dataset tags under 'pyorps_metric_stack'.
        """
        from rasterio import open as rio_open

        self._materialize_legacy()
        if not self._layers:
            raise MetricStackError("Cannot save an empty stack")
        shape = self.shape

        band_names = list(self._layers) + [_BAND_FORBIDDEN]
        bands = [self._layers[n] for n in self._layers]
        bands.append(self.forbidden_mask.astype(np.float32))
        if self.category is not None:
            band_names.append(_BAND_CATEGORY)
            bands.append(self.category.astype(np.float32))
        if self.dem is not None:
            band_names.append(_BAND_DEM)
            bands.append(self.dem)

        meta = {
            "version": 1,
            "layers": list(self._layers),
            "category_labels": ({str(k): v for k, v in
                                 self.category_labels.items()}
                                if self.category_labels else None),
        }

        with rio_open(
                path, "w",
                driver="GTiff",
                height=shape[0], width=shape[1],
                count=len(bands),
                dtype="float32",
                crs=self.crs,
                transform=self.transform,
        ) as dst:
            for index, (name, band) in enumerate(zip(band_names, bands),
                                                 start=1):
                dst.write(band.astype(np.float32), index)
                dst.set_band_description(index, name)
            dst.update_tags(**{_TAG_KEY: json.dumps(meta)})

    @classmethod
    def load(cls, path: str) -> "MetricStack":
        """Load a stack persisted by :meth:`save`."""
        from rasterio import open as rio_open

        with rio_open(path) as src:
            tags = src.tags()
            if _TAG_KEY not in tags:
                raise MetricStackError(
                    f"{path} is not a pyorps metric stack (missing "
                    f"'{_TAG_KEY}' tag). Single-band rasters go through "
                    f"MetricStack.from_single_raster instead.")
            meta = json.loads(tags[_TAG_KEY])
            descriptions = list(src.descriptions)
            data = src.read()
            transform = src.transform
            crs = src.crs

        stack = cls(transform, crs)
        by_name = {name: data[i] for i, name in enumerate(descriptions)}

        missing = [n for n in meta["layers"] + [_BAND_FORBIDDEN]
                   if n not in by_name]
        if missing:
            raise MetricStackError(
                f"Metric stack file {path} is missing band(s) {missing}")

        for name in meta["layers"]:
            stack._layers[name] = by_name[name].astype(np.float32)
        stack.forbidden_mask = by_name[_BAND_FORBIDDEN] > 0.5
        if _BAND_CATEGORY in by_name:
            labels = meta.get("category_labels")
            stack.attach_category(
                by_name[_BAND_CATEGORY].astype(np.uint16),
                {int(k): v for k, v in labels.items()} if labels else None)
        if _BAND_DEM in by_name:
            stack.dem = by_name[_BAND_DEM].astype(np.float32)
        return stack

    # -------------------------------------------------------------- dunders

    def __repr__(self) -> str:
        shape = self.shape
        if self.is_legacy_alias:
            return (f"MetricStack(legacy alias, shape={shape}, "
                    f"cell_size={self.cell_size:g})")
        extras = []
        if self.category is not None:
            extras.append("category")
        if self.dem is not None:
            extras.append("dem")
        extra = f", +{'+'.join(extras)}" if extras else ""
        return (f"MetricStack(layers={list(self._layers)}, shape={shape}, "
                f"cell_size={self.cell_size:g}{extra})")
