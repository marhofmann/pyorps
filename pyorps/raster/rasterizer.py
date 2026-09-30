"""
PYORPS: An Open-Source Tool for Automated Power Line Routing

Reference:
[1] Hofmann, M., Stetz, T., Kammer, F., Repo, S.: 'PYORPS: An Open-Source Tool for
    Automated Power Line Routing', CIRED 2025 - 28th Conference and Exhibition on
    Electricity Distribution, 16 - 19 June 2025, Geneva, Switzerland
"""
import warnings
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import numpy as np
from geopandas import GeoDataFrame
from rasterio import open as rio_open
from rasterio.features import geometry_mask, rasterize
from rasterio.transform import Affine, from_bounds
from shapely.geometry import Polygon, box

from pyorps.core.cost_assumptions import CostAssumptions
from pyorps.core.metric_stack import MetricStack
from pyorps.core.types import (
    IMPASSABLE_CELL_COST,
    BboxType,
    CostAssumptionsType,
    GeometryMaskType,
    InputDataType,
)

# Changed to relative imports from other modules
from pyorps.io.geo_dataset import (
    GeoDataset,
    InMemoryRasterDataset,
    RasterDataset,
    VectorDataset,
    initialize_geo_dataset,
)
from pyorps.raster._geojson import geojson_shapes
from pyorps.raster.thinness import (
    ForbiddenBurnReport,
    is_thin,
    report_forbidden_burn_defects,
    report_repair_seals,
    widen_thin_features,
)


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
            bbox: BboxType | None = None,
            mask: GeometryMaskType | None = None,
            default_crs: str | None = None,
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
        self.metric_stack = None
        # Phase 2.5: burned class-id band reused across cost-table edits.
        self._class_band_cache: dict[str, Any] | None = None
        self.last_forbidden_burn_report: ForbiddenBurnReport | None = None

        if isinstance(cost_assumptions, CostAssumptions):
            self.cost_manager = cost_assumptions
        else:
            self.cost_manager = CostAssumptions(cost_assumptions)

        if self.base_dataset.data is None:
            self.base_dataset.load_data(**kwargs)

        if isinstance(self.base_dataset, RasterDataset):
            self.raster = self.base_dataset.data
            # Normalize 3D raster (bands, height, width) to 2D (first band)
            if self.raster is not None and self.raster.ndim == 3:
                self.raster = self.raster[0]
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
            clip_geometry: GeoDataFrame | Polygon
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
        self.invalidate_class_cache()
        return self.base_dataset

    @staticmethod
    def create_buffer(
            dataset: VectorDataset | GeoDataFrame,
            geometry_buffer_m: float,
            inplace: bool = True
    ) -> VectorDataset | GeoDataFrame:
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
        buffered_data = data.copy()
        buffered_data['geometry'] = buffered_data.buffer(geometry_buffer_m)

        if isinstance(dataset, VectorDataset):
            # Create a new GeoDataset with the buffered data
            buffered_dataset = deepcopy(dataset)
            buffered_dataset.data = buffered_data
            return buffered_dataset
        return buffered_data

    def create_bounds_geodataframe(
            self,
            target_crs: str | None = None
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

    # ------------------------------------------------------------------
    # Burn once, gather K times (performance plan items 2.4/2.5/2.7)
    #
    # Every band this class produces is a *scan conversion of the same
    # geometry sequence* — only the burned value differs. rasterio burns
    # with ``merge_alg=MergeAlg.replace``, so the feature that owns a cell
    # is the LAST one in the sequence that covers it, and that choice does
    # not depend on the value being burned. One index burn therefore
    # determines the winner for every band at once, and each band is a
    # numpy gather ``lut[index_band]`` over that single answer. Painting
    # order is preserved verbatim, so every derived band is bit-identical
    # to burning it separately.
    # ------------------------------------------------------------------

    #: Index/class-band sentinel: "no feature covers this cell".
    _NO_FEATURE = 0

    @staticmethod
    def _index_band_dtype(n_ids: int) -> str:
        """Smallest rasterio-supported unsigned dtype holding 0..n_ids."""
        if n_ids <= np.iinfo(np.uint16).max:
            return "uint16"
        if n_ids <= np.iinfo(np.uint32).max:
            return "uint32"
        raise ValueError(
            f"{n_ids} burn ids exceed the uint32 index band capacity.")

    @classmethod
    def _burn_index_band(
            cls,
            geometries,
            ids,
            n_ids: int,
            out_shape: tuple[int, int],
            transform: Affine,
    ) -> np.ndarray:
        """Burn ONE band of 1-based ids (0 = covered by no feature).

        ``geometries`` and ``ids`` are zipped in the given order, which IS
        the painting order: later entries overwrite earlier ones.
        """
        # Bulk GeoJSON conversion instead of rasterio's per-geometry
        # ``__geo_interface__`` call; order preserving, so the painting
        # order above is untouched (see ``pyorps.raster._geojson``).
        return rasterize(
            ((geom, int(value)) for geom, value
             in zip(geojson_shapes(geometries), ids)),
            out_shape=out_shape,
            fill=cls._NO_FEATURE,
            dtype=cls._index_band_dtype(n_ids),
            transform=transform,
        )

    @staticmethod
    def _gather_band(
            index_band: np.ndarray,
            values: np.ndarray,
            fill,
            dtype,
    ) -> np.ndarray:
        """Derive a value band from an index band by a LUT gather.

        ``values[i]`` belongs to burn id ``i + 1``; id 0 (no feature) picks
        up ``fill``. The gather reproduces a separate rasterize pass of the
        same values exactly, because the ids encode the winning feature.
        """
        lut = np.empty(len(values) + 1, dtype=dtype)
        lut[0] = fill
        lut[1:] = values
        return lut[index_band]

    # ------------------------------------------------------------------
    # Sub-cell forbidden features (options A/C/D/E; see raster.thinness)
    #
    # GDAL burns a cell iff its CENTRE falls inside the polygon, so a
    # forbidden feature narrower than one cell burns 60 cells at one
    # sub-pixel alignment and ZERO at the next. The route stays optimal for
    # the raster that was burned; the raster is what is wrong.
    #
    # NEITHER REPAIR IS THE DEFAULT. C widens the forbidden features that are
    # ACTUALLY thinner than sqrt(2) cells before the burn; D is a SECOND,
    # ALL_TOUCHED scan conversion of the whole forbidden subset painted over
    # the finished raster. Both make sub-cell barriers survive, and both close
    # sub-cell OPENINGS while doing it, because at a fixed cell size a
    # sub-cell barrier and a sub-cell gap are the same geometry seen from
    # opposite sides. Swept over 8 gate widths x 8 sub-pixel offsets and
    # counting only the configurations the plain rule leaves passable, under
    # four different gate-width sets (raster.thinness carries the table;
    # tests/test_raster/sweep_forbidden_repair_tradeoff.py reproduces it): on a
    # FAT 2 m wall C seals 0 % under every set while D seals 17-100 %; on a
    # THIN 0.4 m wall C seals 38-100 % and D 31-100 %, i.e. the two are not
    # distinguishable there. A single percentage is NOT a property of the
    # repair - it moves with the gate widths swept. So the default DETECTS and
    # changes nothing, the two repairs are opt-in with that range on the
    # parameter, and the only correct fix — a finer cell size — is what
    # suggest_resolution() computes.
    #
    # Neither touches ordinary land-use classes: passing all_touched=True to
    # the MAIN burn would fatten every class boundary, and widening an
    # ordinary class would move its boundary for no reason. Painting
    # forbidden last is exactly the existing contract: features are burned
    # in ASCENDING cost order, so the most expensive one already wins an
    # overlap, and IMPASSABLE_CELL_COST is the most expensive value there is.
    # ------------------------------------------------------------------

    @staticmethod
    def _forbidden_rows(values: np.ndarray) -> np.ndarray:
        """Row mask of the features that burn as impassable.

        The burned VALUE defines forbidden-ness, not the feature class: a
        cost table edit can make any class impassable and any impassable
        class ordinary. Note that a dtype too narrow to hold
        IMPASSABLE_CELL_COST (uint8, say) simply yields an empty subset,
        which is correct — nothing in such a raster is forbidden.
        """
        return np.asarray(values) == IMPASSABLE_CELL_COST

    @staticmethod
    def _overlay_forbidden(raster, geometries, out_shape, transform, value):
        """Paint every cell the forbidden geometries TOUCH (option D).

        Grows a forbidden zone by at most one cell, and only there: a
        boolean hit mask is burned separately and used as an index, so no
        cell outside a forbidden feature is written at all.
        """
        if len(geometries) == 0:
            return raster
        hit = rasterize(
            ((geom, 1) for geom in geojson_shapes(geometries)),
            out_shape=out_shape,
            fill=0,
            dtype="uint8",
            transform=transform,
            all_touched=True,
        )
        raster[hit.astype(bool)] = value
        return raster

    @staticmethod
    def _widen_forbidden_rows(data: GeoDataFrame, forbidden: np.ndarray,
                              resolution_in_m: float) -> GeoDataFrame:
        """Return ``data`` with thin forbidden geometries widened (option C).

        Only forbidden rows are ever touched, and among them only the ones
        :func:`is_thin` actually flags — a fat forbidden feature keeps its
        geometry, which is why it seals fewer legitimate sub-cell openings than
        an all-touched overlay of the whole forbidden subset. That advantage is
        real only where the difference lies: on a FAT (2 m) wall this option
        sealed 0 % of the gates the plain rule leaves passable under every
        gate-width set swept, against 17-100 % for the overlay. On a THIN
        (0.4 m) wall — the only case where it acts at all — it sealed 38-100 %
        against the overlay's 31-100 %, so there it is no better. Fewer is not
        none, which is why it is opt-in; see raster.thinness for the sweep.

        Returns the SAME object when nothing needed widening. Callers use that
        identity to decide whether the class-band cache survives; a copy every
        time would throw the cache away on every burn of a layer that merely
        HAS forbidden features.

        The caller MUST already have computed out_shape/transform from the
        unwidened frame: widening a forbidden feature that sits on the data's
        outer edge would otherwise enlarge total_bounds, shift the origin and
        change every cell of the output — a worse silent failure than the one
        being fixed.
        """
        if not forbidden.any():
            return data
        geometries = data['geometry'].to_numpy()
        subset = geometries[forbidden]
        thin = is_thin(subset, resolution_in_m)
        if not thin.any():
            return data
        widened = data.copy()
        new_geometries = widened['geometry'].to_numpy()
        new_geometries[forbidden] = widen_thin_features(
            subset, resolution_in_m, thin=thin)
        widened['geometry'] = new_geometries
        return widened

    @staticmethod
    def _burn_plain_passable(geometries, forbidden, out_shape, transform):
        """The PLAIN burn's passable mask, from the un-repaired geometries.

        The seal check needs to know which cells the pixel-centre rule left
        walkable, and that cannot be read off the finished raster: the fill
        value IS ``IMPASSABLE_CELL_COST``, so a cell no feature covers and a
        cell a forbidden feature covers are the same number (see
        :func:`~pyorps.raster.thinness.detect_repair_seals`). So the plain burn
        is really re-done — but only its passable/impassable verdict is needed,
        so it burns a 1-byte band over the SAME ascending-cost sequence:
        1 for an ordinary feature, 2 for a forbidden one, 0 for no feature.
        Same order, same replace semantics, so the winner per cell is the one
        the plain cost burn would have picked, and ``band == 1`` is exactly
        ``plain_raster != IMPASSABLE_CELL_COST``.

        Only runs when a repair is EXPLICITLY enabled, and only for widening —
        the all-touched overlay hands over the pre-overlay raster instead,
        which is the plain burn already.

        It is not even a net cost. Measured on a synthetic layer of 4401 / 8801
        features (10 % of them forbidden fences, half sub-cell) at 3000x3000 and
        6000x6000 cells, best of 3:

            9.0 M cells   this re-burn 40.8 ms | the two forbidden id-band
                          burns the withdrawn reconstruction needed 58.0 ms
                          | the whole plain cost burn 60.3 ms
            36.0 M cells  155.4 ms | 193.3 ms | 156.6 ms

        so replacing the reconstruction with a real burn made the seal check
        CHEAPER, and what dominates it is neither: the two
        ``scipy.ndimage.label`` passes cost 302 ms and 1157 ms respectively.
        """
        values = np.where(np.asarray(forbidden), 2, 1).astype(np.uint8)
        band = rasterize(
            ((geom, int(v)) for geom, v
             in zip(geojson_shapes(geometries), values)),
            out_shape=out_shape,
            fill=0,
            dtype="uint8",
            transform=transform,
        )
        return band == 1

    def _finish_forbidden(self, raster, data, field_name, out_shape,
                          transform, resolution_in_m, all_touched,
                          on_thin_features, value=IMPASSABLE_CELL_COST,
                          plain_geometries=None):
        """Apply option D, report what the plain burn did (A), and what a
        repair did to free space.

        Two detectors, because a repair is a trade and each of them can only
        see one side of it:

        * Option A asks whether the burn represents every forbidden feature.
          Under the default (no repair) that is the whole story. Under option C
          it inspects the WIDENED geometry, which is defect-free by
          construction — it can confirm the widening worked, never that it went
          too far. Under option D it has nothing left to find at all and is
          skipped.
        * The seal check asks the opposite question: did the repair disconnect
          free space that the plain pixel-centre burn left connected? It runs
          exactly when a repair is active, since with no repair the answer is
          trivially no.

        ``plain_geometries`` carries the forbidden geometries as DIGITIZED,
        which the caller must hand over when it widened them — ``data`` by then
        holds the repaired ones, and the seal check needs both.

        The seal check also needs the PLAIN burn's passable mask, which is NOT
        recoverable from the finished raster (fill == IMPASSABLE_CELL_COST ==
        a forbidden burn). It is obtained the only unambiguous way:

        * option D — the raster as it stands BEFORE the overlay pass IS the
          plain burn, so the mask is one comparison and costs no burn at all;
        * option C — the plain sequence is re-burned as a 1-byte
          passable/forbidden band (see :meth:`_burn_plain_passable`).
        """
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        if on_thin_features not in ("warn", "raise", "ignore"):
            # Validated even when nothing is forbidden, so a typo cannot lurk
            # until the day a cost table first marks something impassable.
            raise ValueError(
                "on_thin_features must be 'warn', 'raise' or 'ignore', got "
                f"{on_thin_features!r}")
        forbidden = self._forbidden_rows(data[field_name].to_numpy())
        if not forbidden.any():
            return raster
        geometries = data['geometry'].to_numpy()[forbidden]
        # Both reports return immediately under 'ignore'. Checking here as
        # well keeps a silenced caller from paying for the masks and the
        # re-burn they would have been handed.
        detect = on_thin_features != "ignore"
        plain_passable = None
        if all_touched:
            if detect and plain_geometries is None:
                # Snapshot BEFORE the overlay: this is the plain burn itself.
                # With BOTH repairs on it is not — the burn underneath is
                # already widened — so that case falls through to the re-burn.
                plain_passable = raster != value
            raster = self._overlay_forbidden(raster, geometries, out_shape,
                                             transform, value)
        elif detect:
            self.last_forbidden_burn_report = report_forbidden_burn_defects(
                geometries, out_shape, transform,
                resolution_in_m=resolution_in_m,
                bounds=tuple(data.total_bounds),
                on_thin_features=on_thin_features,
                stacklevel=4,
                # A sub-cell arm can only hole a barrier where the raster is
                # walkable; without this, an arm running through NODATA reads
                # as a hole because nodata burns IMPASSABLE_CELL_COST too.
                passable=raster != value,
            )
        if detect and plain_geometries is not None and plain_passable is None:
            # Option C. The widened geometries live in `data`; putting the
            # digitized ones back in their rows reproduces the exact sequence
            # the plain burn would have scan-converted.
            plain_sequence = data['geometry'].to_numpy().copy()
            plain_sequence[forbidden] = plain_geometries
            plain_passable = self._burn_plain_passable(
                plain_sequence, forbidden, out_shape, transform)
        if all_touched or plain_geometries is not None:
            report_repair_seals(
                raster,
                geometries if plain_geometries is None else plain_geometries,
                geometries, out_shape, transform,
                all_touched=all_touched,
                impassable=value,
                resolution_in_m=resolution_in_m,
                on_thin_features=on_thin_features,
                stacklevel=4,
                plain_passable=plain_passable,
            )
        return raster

    def invalidate_class_cache(self) -> None:
        """Drop the cached class-id band (item 2.5).

        Call this after mutating the base geometries in place — the cache
        key can only see the GeoDataFrame's identity, length and bounds.
        """
        self._class_band_cache = None

    def rasterize(
            self,
            field_name: str = 'cost',
            resolution_in_m: float = 1.0,
            fill_value: int = IMPASSABLE_CELL_COST,
            save_path: str | None = None,
            dtype: str = "uint16",
            geometry_buffer_m: float = 0,
            bounding_box: Polygon | None = None,
            preprocessing_function: Callable | None = None,
            preprocessing_kwargs: dict[str, Any] | None = None,
            *,
            use_class_cache: bool = True,
            all_touched: bool = False,
            widen_thin_forbidden: bool = False,
            on_thin_features: str = "warn",
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
            preprocessing_function: A function that takes the base dataset as a first
            argument and other arguments defined in preprocessing_kwargs which will
            be called before rasterization
            preprocessing_kwargs: The keyword arguments passed to preprocessing_function
            use_class_cache: Reuse a burned class-id band across calls that
                only change the cost table (item 2.5). The band is re-burned
                whenever the geometry set, the extent or the cost ORDER of
                the classes changes; the produced raster is bit-identical to
                a full re-burn either way. Set False to force a re-burn.
            all_touched: OPT-IN (option D). Burn FORBIDDEN features (those
                whose value is IMPASSABLE_CELL_COST) into every cell they
                intersect, via a second overlay pass after the ordinary burn.
                Ordinary classes always keep GDAL's pixel-centre rule and stay
                bit-identical; only cells a forbidden feature touches can
                change, so a forbidden zone grows by at most one cell.
                TRADE-OFF, and the reason this is not the default: it fattens
                EVERY forbidden feature, fat ones included, so it also SEALS
                legitimate sub-cell OPENINGS — a gate, a culvert, a gap
                between parcels. Swept over 8 gate widths x 8 sub-pixel
                offsets under four different gate-width sets, counting only the
                configurations the plain rule leaves passable, this option
                SEALED 31-100 % of them in a 0.4 m wall and 17-100 % in a 2 m
                wall (widen_thin_forbidden: 38-100 % and 0 %; the default: 0 %
                and 0 %, by construction). The FAT-wall column is the robust
                difference between the two repairs; on a thin wall they are
                indistinguishable. See ``pyorps.raster.thinness`` for the table
                and the exact gate widths — a bare percentage from one sweep
                does not transfer. Enable this only when you want maximum
                conservatism and every opening narrower than a cell may be
                treated as closed.
            widen_thin_forbidden: OPT-IN (option C). Before burning, buffer
                forbidden features narrower than sqrt(2) cells up to that
                width. It repairs exactly the features that the pixel-centre
                rule cannot represent and touches no other forbidden feature,
                which makes it the better of the two repairs on FAT features —
                but not a safe one: in the sweep above it still sealed 38-100 %
                of the gates in a 0.4 m wall, no better than the overlay there,
                because a sub-cell barrier and a sub-cell gap are the same
                geometry seen from opposite sides. The only fix that is correct
                rather than a trade is a finer cell size; see
                ``suggest_resolution``. When this actually widens something it
                disables the class-band cache for that call: which features are
                forbidden depends on the cost table, so the burned geometry
                would too.
            on_thin_features: 'warn' (default), 'raise' or 'ignore' — what to
                do when the burn does not faithfully represent the vector data.
                That covers both directions: a forbidden feature dropped, holed
                or fragmented by the pixel-centre rule (checked whenever
                ``all_touched`` is off), and a repair that SEALED passable free
                space the plain burn left connected (checked whenever a repair
                is on). Under the default — no repair — the raster is
                bit-identical to a plain GDAL burn and only the first check can
                fire.
        Returns:
            tuple of (raster_data, transform)
        """
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        if self.base_data.shape[0] == 0:
            raise ValueError("Base data is empty - nothing to rasterize!")

        if self.base_dataset is None or self.base_dataset.data is None:
            raise ValueError("No base dataset loaded to rasterize")

        if preprocessing_function is not None:
            if preprocessing_kwargs is None:
                preprocessing_kwargs = dict()
            preprocessing_function(self.base_dataset.data, **preprocessing_kwargs)

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

        # Apply buffer if needed. Buffering is strictly row-wise, so doing
        # it BEFORE the sort yields exactly the frame the previous
        # sort-then-buffer order produced — but it also leaves an
        # unsorted, cost-independent row order for the class-band cache
        # (item 2.5): the sort permutation itself depends on the cost
        # table, and a cache keyed on it could never hit after a re-cost.
        if geometry_buffer_m > 0:
            unsorted = self.create_buffer(self.base_data, geometry_buffer_m,
                                          inplace=False)
        else:
            unsorted = self.base_data

        # Sort values by field to ensure higher cost values have higher priority.
        # kind='stable': pandas defaults to an UNSTABLE quicksort, which leaves
        # the paint order among equal-cost features at the mercy of the input
        # row order. See the note on the same sort in rasterize_metrics.
        buffered = unsorted.sort_values(by=field_name, ascending=True,
                                        kind='stable')

        if bounding_box is None:
            # Calculate the output shape based on the GeoDataFrame's bounds and the
            # specified resolution
            out_shape = self._calculate_out_shape_from_geodataframe(buffered,
                                                                    resolution_in_m)

            # Create a transformation object to convert between coordinate systems
            self.transform = from_bounds(*buffered.total_bounds, *out_shape[::-1])
        else:
            # Calculate the output shape based on the bounding box
            out_shape = self._calculate_out_shape_from_bounding_box(bounding_box,
                                                                    resolution_in_m)

            # Create a transformation object
            self.transform = from_bounds(*bounding_box.bounds, *out_shape[::-1])

            # rasterio.features.rasterize refuses a degenerate output; the
            # dropped bbox pre-burn (item 2.7b) used to raise this for us.
            if min(out_shape) == 0:
                raise ValueError("width and height must be > 0")

        # Option C. Strictly AFTER out_shape/self.transform above: widening
        # a forbidden feature on the outer edge would otherwise enlarge
        # total_bounds and move every cell of the raster.
        plain_geometries = None
        if widen_thin_forbidden:
            forbidden = self._forbidden_rows(buffered[field_name].to_numpy())
            widened = self._widen_forbidden_rows(buffered, forbidden,
                                                 resolution_in_m)
            if widened is not buffered:
                # Kept for the seal check: once `buffered` is the widened
                # frame the digitized geometry is gone, and the check needs
                # both to say what the repair COST.
                plain_geometries = buffered['geometry'].to_numpy()[forbidden]
                buffered = widened
                # The class band caches GEOMETRY, and widening depends on
                # which features the current cost table calls forbidden — so
                # a cached band from a different cost table would be wrong.
                # Only a burn that actually widened something pays this;
                # otherwise the geometry is the one the cache was burned from.
                use_class_cache = False

        # Item 2.5: reuse a burned class-id band whenever only the cost
        # table changed. Returns None when no valid cache applies, in which
        # case we fall through to the single-pass burn below.
        cache_key = None
        if use_class_cache and preprocessing_function is None:
            cache_key = self._class_cache_key(unsorted, field_name, dtype,
                                              resolution_in_m,
                                              geometry_buffer_m, bounding_box,
                                              out_shape)
        else:
            self.invalidate_class_cache()

        raster = None
        if cache_key is not None:
            raster = self._rasterize_via_class_lut(
                unsorted, field_name, fill_value, dtype, out_shape,
                self.transform, cache_key)

        if raster is None:
            # ONE scan conversion over the ascending-sorted sequence. The
            # former bounding-box branch pre-burned the bbox polygon with
            # fill_value into an array already filled with fill_value (a
            # wasted O(N) pass, item 2.7b) and then looped once per unique
            # value; both produce the same winner-per-cell as this single
            # pass, because sorting ascending makes the last feature that
            # covers a cell the most expensive one either way.
            shapes = ((geom, value) for geom, value
                      in zip(geojson_shapes(buffered['geometry'].to_numpy()),
                             buffered[field_name]))
            raster = rasterize(
                shapes,
                out_shape=out_shape,
                fill=fill_value,
                dtype=dtype,
                transform=self.transform
            )

        # Hook AFTER the raster is final, so the class-LUT path above and the
        # plain burn are covered by the same call.
        raster = self._finish_forbidden(raster, buffered, field_name,
                                        out_shape, self.transform,
                                        resolution_in_m, all_touched,
                                        on_thin_features,
                                        plain_geometries=plain_geometries)
        self.raster = raster

        self.raster_dataset = InMemoryRasterDataset(self.raster,
                                                    self.crs,
                                                    self.transform)
        # Write the rasterized data to a new raster file if a save path is provided
        if save_path is not None:
            self.save_raster(save_path)
        return self.raster_dataset

    # ------------------------------------------------------------------
    # Item 2.5 — class-id band + LUT re-cost
    #
    # A cost-table edit changes what a class costs, never which cells a
    # class occupies. So the scan conversion is burned ONCE as class ids
    # and every subsequent cost table is applied as ``lut[class_band]``.
    #
    # The one way this can go wrong is the legacy paint order: features are
    # painted by ASCENDING cost, so the most expensive feature wins an
    # overlap. The burned band froze one particular class ranking; if a new
    # cost table reorders the classes, the frozen band's winner is no longer
    # the most expensive one. The validity condition is therefore exactly
    # "the new cost is non-decreasing along the burned paint order" — which
    # permits arbitrary value changes (and new ties) but rejects every rank
    # swap. It is checked in O(number of classes) on every call.
    # ------------------------------------------------------------------

    def _class_key_columns(self, data: GeoDataFrame) -> list[str]:
        """Feature columns that determine the cost of a row."""
        main = self.cost_manager.main_feature
        side = self.cost_manager.side_features or []
        return [c for c in [main, *side] if c and c in data.columns]

    def _class_cache_key(self, data, field_name, dtype, resolution_in_m,
                         geometry_buffer_m, bounding_box, out_shape):
        """Identity of the burn: everything but the burned VALUES.

        Returns None when the class-band path does not apply at all.
        """
        if field_name != 'cost':
            # Only the cost column is a pure function of the class columns.
            return None
        if not self._class_key_columns(data):
            return None
        bbox_bounds = (None if bounding_box is None
                       else tuple(round(v, 9) for v in bounding_box.bounds))
        return (
            id(self.base_dataset.data),
            len(data),
            field_name,
            np.dtype(dtype).name,
            float(resolution_in_m),
            float(geometry_buffer_m),
            tuple(round(float(v), 9) for v in data.total_bounds),
            bbox_bounds,
            tuple(out_shape),
        )

    def _row_class_codes(self, data: GeoDataFrame):
        """Per-row class code (0..C-1) and the class count, or None."""
        columns = self._class_key_columns(data)
        if not columns:
            return None
        try:
            codes = data.groupby(columns, sort=False,
                                 dropna=False).ngroup().to_numpy()
        except (TypeError, ValueError):
            # Unhashable feature values (lists from JSON attributes, ...)
            # cannot be grouped — fall back to a full burn.
            return None
        if codes.size == 0 or codes.min() < 0:
            return None
        n_classes = int(codes.max()) + 1
        # Ids burned into the band are ranks 1..C; 0 means "no feature".
        if n_classes > np.iinfo(np.uint16).max:
            return None
        return np.ascontiguousarray(codes, dtype=np.int64), n_classes

    def _rasterize_via_class_lut(self, data, field_name, fill_value,
                                 dtype, out_shape, transform, cache_key):
        """Produce the cost raster as ``lut[class_band]``, or None.

        ``data`` must be in the cost-INDEPENDENT base row order (buffered
        but not sorted), so the per-row codes are stable across cost-table
        edits; the burn itself is re-sorted by class rank below.

        None means "the class path does not apply here" and the caller must
        fall back to a full burn. The returned raster is bit-identical to
        that full burn in every case where this does return an array.
        """
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        grouping = self._row_class_codes(data)
        if grouping is None:
            return None
        codes, n_classes = grouping

        values = np.ascontiguousarray(data[field_name].to_numpy())
        # A class must be cost-homogeneous, otherwise it is not a class.
        # (Scatter the last value per class, then verify every row agrees.)
        class_values = np.zeros(n_classes, dtype=values.dtype)
        class_values[codes] = values
        if not np.array_equal(class_values[codes], values):
            return None

        cache = self._class_band_cache
        reusable = (
            cache is not None
            and cache['key'] == cache_key
            and cache['n_classes'] == n_classes
            and cache['band'].shape == tuple(out_shape)
            and np.array_equal(cache['codes'], codes)
        )
        if reusable:
            # THE invalidation rule: the burned paint order must still be a
            # non-decreasing cost order, else a re-ranked class would keep
            # losing (or start winning) an overlap it no longer should.
            # float64 (not the band's own unsigned dtype) so the difference
            # cannot wrap, and not int64 so a float cost column cannot be
            # truncated into a spurious tie.
            ordered = class_values[cache['order']].astype(np.float64)
            reusable = bool(np.all(np.diff(ordered) >= 0))

        if not reusable:
            # Rank classes by ascending cost — the legacy paint order — and
            # burn the RANKS, so overlaps resolve exactly as before.
            order = np.argsort(class_values, kind='stable')
            rank = np.empty(n_classes, dtype=np.int64)
            rank[order] = np.arange(n_classes, dtype=np.int64)
            row_ids = rank[codes] + 1
            # Paint by rank, not by raw value: two classes that merely TIE
            # today must still be painted in their ranked order, or a later
            # re-cost that separates them would pick the wrong winner.
            paint = np.argsort(row_ids, kind='stable')
            geometries = data['geometry'].to_numpy()[paint]
            band = self._burn_index_band(geometries, row_ids[paint],
                                         n_classes, out_shape, transform)
            cache = {
                'key': cache_key,
                'codes': codes,
                'order': order,
                'n_classes': n_classes,
                'band': band,
            }
            self._class_band_cache = cache

        return self._gather_band(cache['band'],
                                 class_values[cache['order']],
                                 fill_value, dtype)

    def rasterize_metrics(
            self,
            resolution_in_m: float = 1.0,
            geometry_buffer_m: float = 0,
            include_category: bool = True,
            preprocessing_function: Callable | None = None,
            preprocessing_kwargs: dict[str, Any] | None = None,
            *,
            all_touched: bool = False,
            widen_thin_forbidden: bool = False,
            on_thin_features: str = "warn",
    ) -> MetricStack:
        """Rasterize the base dataset into a multi-band :class:`MetricStack`.

        ONE geometry pass, K value bindings: the sorted geometry sequence is
        scan-converted a single time into a band of row indices (item 2.4 of
        the 2026-08-07 performance plan), and every metric column written by
        the cost manager becomes one float32 band — plus an optional
        feature-class category band for reporting breakdowns — by gathering
        that index band through a lookup table. Because the index band
        already encodes the winning feature of the SAME sorted sequence, the
        winner on overlaps is identical in every band and every band is
        bit-identical to a separate rasterize pass — the alignment invariant
        of the feasibility plan (section 6.4).

        Cells outside all features are forbidden (as in :meth:`rasterize`,
        where they receive the 65535 fill). Metric bands are NOT rounded —
        float values like landscape indices survive as-is.

        Parameters:
            resolution_in_m: Output resolution in meters per pixel.
            geometry_buffer_m: Optional buffer applied to the geometries.
            include_category: Also rasterize the feature-class id band.
            preprocessing_function: Optional hook called with the base
                GeoDataFrame before applying cost assumptions.
            preprocessing_kwargs: Keyword arguments for the hook.
            all_touched: OPT-IN. Overlay forbidden features (cost ==
                IMPASSABLE_CELL_COST) onto the COST band with an all-touched
                pass. It sealed 31-100 % of the measured sub-cell openings in a
                thin wall and 17-100 % in a fat one, the range being over four
                gate-width sets — see the trade-off documented on
                :meth:`rasterize`. The other metric bands are deliberately
                left alone: forbidden-ness rides the cost band alone, and
                repainting e.g. a gradient band on the extra rim would invent
                values no feature ever supplied.
            widen_thin_forbidden: OPT-IN. Widen the forbidden features that
                are actually narrower than sqrt(2) cells before the burn
                (option C), which repairs a vanishing barrier without fattening
                the fat ones — and still sealed 38-100 % of those thin-wall
                openings. Not the default for that reason; see
                :meth:`rasterize`.
            on_thin_features: 'warn' (default), 'raise' or 'ignore' when the
                burn does not faithfully represent the vector data — a
                forbidden feature dropped, holed or fragmented by the
                pixel-centre rule, or a repair that sealed passable free space.

        Returns:
            The rasterized :class:`MetricStack` (also stored as
            ``self.metric_stack``).
        """
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        if self.base_dataset is None or self.base_dataset.data is None:
            raise ValueError("No base dataset loaded to rasterize")
        if self.base_data.shape[0] == 0:
            raise ValueError("Base data is empty - nothing to rasterize!")

        if preprocessing_function is not None:
            preprocessing_function(self.base_dataset.data,
                                   **(preprocessing_kwargs or {}))

        # Apply cost assumptions: one column per metric ('cost' first)
        self.cost_manager.apply_to_geodataframe(self.base_data)
        metric_names = self.cost_manager.metric_names
        missing = [m for m in metric_names if m not in self.base_data.columns]
        if missing:
            raise ValueError(
                f"Cost manager did not produce metric column(s) {missing}")

        data = self.base_data
        # Unmatched features: forbidden in cost => forbidden in the stack;
        # other metrics default to 0 (their forbidden state rides the mask).
        data['cost'] = data['cost'].fillna(IMPASSABLE_CELL_COST)
        for name in metric_names:
            if name != 'cost':
                data[name] = data[name].fillna(0.0)

        # ONE ordering for every band: ascending cost, so on overlaps the
        # more expensive feature wins — exactly as in rasterize().
        #
        # kind='stable' is load-bearing here, not tidiness. Ties are common
        # (many classes share a cost) and the band below burns ROW INDICES, so
        # which of several equal-cost features wins a cell decides that cell's
        # value in every OTHER metric band. Under pandas' default quicksort
        # that winner depends on the input row order, so a reader change
        # (pyogrio vs fiona) or a geopandas upgrade silently alters the
        # non-cost bands — measured at 8.1 % of covered cells. The cost band
        # itself is unaffected either way: equal cost, equal burn value.
        data = data.sort_values(by='cost', ascending=True, kind='stable')

        if geometry_buffer_m > 0:
            data = self.create_buffer(data, geometry_buffer_m, inplace=False)

        out_shape = self._calculate_out_shape_from_geodataframe(
            data, resolution_in_m)
        transform = from_bounds(*data.total_bounds, *out_shape[::-1])

        # Option C, strictly after the frame is fixed — see the note in
        # _widen_forbidden_rows on why the order matters.
        plain_geometries = None
        if widen_thin_forbidden:
            forbidden = self._forbidden_rows(data['cost'].to_numpy())
            widened = self._widen_forbidden_rows(data, forbidden,
                                                 resolution_in_m)
            if widened is not data:
                # See rasterize(): the seal check needs the digitized geometry.
                plain_geometries = data['geometry'].to_numpy()[forbidden]
                data = widened

        stack = MetricStack(transform, self.crs)

        # Item 2.4: ONE scan conversion of the sorted geometry sequence,
        # burning 1-based ROW indices. Every band below is then a numpy
        # gather over that single answer instead of another full-extent
        # pass that re-converts all F shapely geometries through the
        # Python geo-interface. The winning feature per cell is decided by
        # the paint order alone — identical sequence, identical replace
        # semantics — so each gathered band is bit-identical to burning it
        # on its own.
        n_rows = len(data)
        index_band = self._burn_index_band(
            data['geometry'],
            np.arange(1, n_rows + 1, dtype=np.int64),
            n_rows,
            out_shape,
            transform,
        )

        # Cost band first: its 65535 fill defines the outside-features
        # forbidden area for the whole stack.
        cost_band = self._gather_band(
            index_band,
            data['cost'].astype(np.float32).to_numpy(),
            np.float32(IMPASSABLE_CELL_COST),
            np.float32,
        )
        cost_band = self._finish_forbidden(
            cost_band, data, 'cost', out_shape, transform, resolution_in_m,
            all_touched, on_thin_features,
            value=np.float32(IMPASSABLE_CELL_COST),
            plain_geometries=plain_geometries)
        stack.add_layer('cost', cost_band)

        for name in metric_names:
            if name == 'cost':
                continue
            band = self._gather_band(
                index_band,
                data[name].astype(np.float32).to_numpy(),
                np.float32(0.0),
                np.float32,
            )
            stack.add_layer(name, band)

        if include_category:
            ids, labels = self._build_category_ids(data)
            category_band = self._gather_band(
                index_band,
                np.asarray(ids, dtype=np.uint16),
                np.uint16(0),
                np.uint16,
            )
            stack.attach_category(category_band, labels)

        self.metric_stack = stack
        return stack

    def _build_category_ids(self, data):
        """Feature-class ids (1-based, 0 = no class) from the feature columns.

        Ids follow the same row order as the value bindings, so the
        category band picks the same winning feature as the metric bands.
        """
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        main = self.cost_manager.main_feature
        side = self.cost_manager.side_features or []
        columns = [c for c in [main, *side] if c in data.columns]
        if not columns:
            # No feature columns detected — one class per unique cost value
            keys = data['cost'].astype(str)
        else:
            keys = data[columns].astype(str).agg(
                lambda row: " > ".join(v for v in row if v), axis=1)

        unique_keys = list(dict.fromkeys(keys))
        if len(unique_keys) > 65534:
            raise ValueError(
                f"{len(unique_keys)} feature classes exceed the uint16 "
                f"category band capacity (65534).")
        id_by_key = {key: i + 1 for i, key in enumerate(unique_keys)}
        ids = keys.map(id_by_key).astype(np.uint16)
        labels = {i + 1: key for i, key in enumerate(unique_keys)}
        return ids, labels

    def _calculate_out_shape_from_bounding_box(
            self,
            bounding_box: Polygon,
            resolution_in_m: float = 1.0
    ) -> tuple[int, int]:
        """
        Calculate the output shape (rows, columns) based on a bounding box and
        resolution.

        Parameters:
            bounding_box: The bounding box defining the output shape in a planar CRS
            resolution_in_m: The linear resolution per pixel in meters

        Returns:
            tuple of (rows, columns) representing the output shape
        """
        # Calculate the bounding box dimensions
        bounds = bounding_box.bounds  # (minx, miny, maxx, maxy)
        width = bounds[2] - bounds[0]  # maxx - minx
        height = bounds[3] - bounds[1]  # maxy - miny

        # Calculate the total area of the bounding box in square meters
        total_area_m2 = width * height

        return self._get_rows_and_columns(width, height, resolution_in_m,
                                          total_area_m2)

    def _calculate_out_shape_from_geodataframe(
            self,
            gdf: GeoDataFrame,
            resolution_in_m: float = 1.0,
            bounding_box: Polygon | None = None
    ) -> tuple[int, int]:
        """
        Calculate the output shape (rows, columns) based on a GeoDataFrame and
        resolution.

        Parameters:
            gdf: The GeoDataFrame containing the geometries to cover
            resolution_in_m: The linear resolution per pixel in meters
            bounding_box: Optional bounding box defining the output shape

        Returns:
            tuple of (rows, columns) representing the output shape
        """
        # Ensure the GeoDataFrame is in a projected CRS that uses meters
        if gdf.crs.is_geographic:
            utm_crs = gdf.estimate_utm_crs()
            warnings.warn(
                f"Geographic CRS ({gdf.crs}) detected. Auto-reprojecting to "
                f"{utm_crs} for accurate metric calculations.",
                UserWarning, stacklevel=2
            )
            gdf = gdf.to_crs(utm_crs)

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

        return self._get_rows_and_columns(width, height, resolution_in_m,
                                          total_area_m2)

    @staticmethod
    def _get_rows_and_columns(width, height, resolution_in_m, total_area_m2):
        """
        Calculate rows and columns based on width, height, and resolution.

        Parameters:
            width: Width of the area in meters
            height: Height of the area in meters
            resolution_in_m: Linear resolution per pixel in meters
            total_area_m2: Total area in square meters

        Returns:
            tuple of (rows, columns)
        """
        # Calculate the aspect ratio
        aspect_ratio = width / height if height != 0 else 1.0
        # Calculate the area of each pixel (linear resolution squared)
        pixel_area = resolution_in_m ** 2
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
            ignore_value: float | None = IMPASSABLE_CELL_COST,
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

        # Create a mask from the geometries in the GeoDataFrame.
        # ``geometry_mask`` takes BARE geometries (no (geom, value) tuples),
        # so the bulk conversion feeds it bare GeoJSON dicts.
        mask_array = geometry_mask(
            geojson_shapes(gdf['geometry'].values),
            transform=self.transform,
            invert=True,  # Invert the mask to keep the area inside the polygons
            out_shape=self.raster.shape
        )

        # ``mask_array`` is a freshly allocated boolean band; narrow it in
        # place instead of allocating an all-True band and a combined one
        # (item 2.7a — ``x & ones`` is a no-op by construction).
        mask = mask_array
        if ignore_value is not None:
            mask &= self.raster != ignore_value

        # Modify the raster values based on the specified parameters
        if multiply:
            # Widen to float to prevent uint16 overflow, then round and clip.
            # The FACTOR must not be cast to an integer. Cost factors are
            # routinely fractional -- 1.15 for a drinking-water zone, 1.8 for
            # a soil class -- and ``np.uint32(1.15) == 1`` silently turned
            # every one of them into a no-op, while 2.5 became 2. Only the
            # product ever needed widening; the factor never did.
            # One float64 buffer, then multiply/round/clip in place. The
            # chained-expression form allocates a fresh temporary per step:
            # measured over 70 M selected cells that was 1.52 s and 1120 MB
            # against 1.06 s and 700 MB here, for a bit-identical result.
            # Peak memory is what rules the full 1.045e9-cell raster out, so
            # it is the number worth spending a line on.
            buf = self.raster[mask].astype(np.float64)
            buf *= float(value)
            np.rint(buf, out=buf)
            np.clip(buf, 0, np.iinfo(np.uint16).max, out=buf)
            self.raster[mask] = buf.astype(self.raster.dtype)
        else:
            # Set the raster cells to the new value
            self.raster[mask] = value

        return self.raster

    def modify_raster_from_dataset(
            self,
            input_data: InputDataType,
            cost_assumptions: CostAssumptionsType | int | float | None = None,
            bbox: BboxType | None = None,
            mask: GeometryMaskType | None = None,
            transform: Affine | None = None,
            geometry_buffer_m: float = 0,
            ignore_value: float | None = IMPASSABLE_CELL_COST,
            multiply: bool = False,
            zone_field: str | None = None,
            forbidden_zone: str | None = None,
            forbidden_value: int = IMPASSABLE_CELL_COST,
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
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
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
            self._apply_cost_groups(gdf, ignore_value, multiply)
        return self.raster

    def _apply_cost_groups(
            self,
            gdf: GeoDataFrame,
            ignore_value: float | None,
            multiply: bool,
    ) -> None:
        """Apply one cost value per group of overlay geometries (item 2.7a).

        The legacy implementation ran a full ``geometry_mask`` plus three
        further full-raster passes per unique cost value. In *replace* mode
        that whole loop collapses into ONE scan conversion of a group-id
        band plus one vectorized apply, with the paint order chosen so the
        outcome is identical (see ``_group_paint_order``).

        *Multiply* mode is genuinely sequential — a cell covered by two
        groups is multiplied twice, and a cell clipped up to ``ignore_value``
        is frozen for later groups — so it keeps the loop and only sheds the
        redundant allocations.
        """
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        raster = self.raster
        groups = []
        for unique_value in gdf['cost'].unique():
            value_geoms = gdf.loc[gdf['cost'] == unique_value]
            if value_geoms.empty:
                continue
            groups.append((unique_value, value_geoms))
        if not groups:
            return

        if multiply:
            self._apply_cost_groups_multiply(groups, ignore_value)
            return

        # Cast the group values exactly as ``raster[mask] = value`` would,
        # so both the LUT and the ignore_value comparison see the values
        # that actually land in the raster.
        lut = np.zeros(len(groups) + 1, dtype=raster.dtype)
        for index, (unique_value, _) in enumerate(groups):
            lut[index + 1] = unique_value

        order = self._group_paint_order(lut, ignore_value)
        # Flatten the per-group geometries into ONE array in exactly
        # ``order`` before converting: the bulk GeoJSON path is order
        # preserving, so materializing the paint sequence up front is what
        # keeps the burn identical (a permutation here would silently move
        # which group wins an overlap).
        per_group = [groups[group_index][1]['geometry'].to_numpy()
                     for group_index in order]
        geometries = (np.concatenate(per_group) if len(per_group) > 1
                      else per_group[0])
        ids = np.repeat(order + 1, [len(part) for part in per_group])
        shapes = (
            (geom, int(value))
            for geom, value in zip(geojson_shapes(geometries), ids)
        )
        band = rasterize(
            shapes,
            out_shape=raster.shape,
            fill=self._NO_FEATURE,
            dtype=self._index_band_dtype(len(groups)),
            transform=self.transform,
        )
        mask = band != self._NO_FEATURE
        if ignore_value is not None:
            mask &= raster != ignore_value
        raster[mask] = lut[band[mask]]

    @classmethod
    def _group_paint_order(cls, lut: np.ndarray,
                           ignore_value: float | None) -> np.ndarray:
        """Paint order reproducing the legacy sequential overwrite.

        Legacy semantics, per cell, in *replace* mode:

        * a cell already holding ``ignore_value`` is never touched;
        * otherwise groups are applied in first-appearance order, later
          ones overwriting earlier ones — EXCEPT that the first group whose
          value equals ``ignore_value`` freezes the cell, because the very
          next iteration recomputes ``raster != ignore_value`` and excludes
          it from then on.

        Moving every ignore-valued group to the end of the paint order makes
        a single replace burn produce exactly that: an ignore-valued group
        wins whenever one covers the cell (they all carry the same value, so
        which one wins is immaterial), and otherwise the last ordinary group
        wins — which is precisely the legacy outcome.
        """
        group_count = len(lut) - 1
        if ignore_value is None:
            return np.arange(group_count, dtype=np.int64)
        frozen = lut[1:] == ignore_value
        return np.concatenate([np.flatnonzero(~frozen),
                               np.flatnonzero(frozen)])

    def _apply_cost_groups_multiply(self, groups, ignore_value) -> None:
        """Sequential multiply-mode application (compounding, per legacy)."""
        raster = self.raster
        scratch = None
        for unique_value, value_geoms in groups:
            mask = geometry_mask(
                geojson_shapes(value_geoms['geometry'].values),
                transform=self.transform,
                invert=True,  # keep the area inside the polygons
                out_shape=raster.shape,
            )
            if ignore_value is not None:
                if scratch is None:
                    scratch = np.empty(raster.shape, dtype=bool)
                np.not_equal(raster, ignore_value, out=scratch)
                mask &= scratch
            # Widen to float, not to an integer: casting the factor truncated
            # every fractional cost factor away. See modify_raster_with_
            # geodataframe for the full note.
            # In-place on one float64 buffer; see modify_raster_with_
            # geodataframe for the measurement behind this shape.
            buf = raster[mask].astype(np.float64)
            buf *= float(unique_value)
            np.rint(buf, out=buf)
            np.clip(buf, 0, np.iinfo(np.uint16).max, out=buf)
            raster[mask] = buf.astype(raster.dtype)

    def _modify_raster_from_dataset_simple_cost_assumptions(
            self,
            gdf: GeoDataFrame,
            cost_assumptions: CostAssumptionsType | int | float | None = None,
            ignore_value: float | None = IMPASSABLE_CELL_COST,
            multiply: bool = False,
            zone_field: str | None = None,
            forbidden_zone: str | None = None,
            forbidden_value: int = IMPASSABLE_CELL_COST,
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

        # Update raster_dataset to reflect the shrunk raster
        self.raster_dataset = InMemoryRasterDataset(
            self.raster, self.crs, self.transform
        )

        return self.raster
