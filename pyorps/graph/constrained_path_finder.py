"""Constrained path finder with infrastructure-aware routing."""

import math
import os
import time
import warnings
from bisect import bisect_right
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LineString, Point

from pyorps.core.constrained_path import ConstrainedPath, Tower
from pyorps.core.exceptions import (
    FeatureNotImplementedError,
    InvalidInputError,
    InvalidTypeError,
    NoPathFoundError,
    NumericalPrecisionError,
)
from pyorps.core.infrastructure_profile import InfrastructureProfile
from pyorps.core.types import IMPASSABLE_CELL_COST
from pyorps.core.warning_types import PyorpsWarning, SpanRuleWarning
from pyorps.graph.api.graph_library_api import reject_unknown_options
from pyorps.graph.path_finder import _INTERNAL_FILE_PREFIXES, PathFinder
from pyorps.graph.span_clearance import path_span_clearance
from pyorps.utils.constrained_sssp_gpu import CONSTRAINED_GPU_CHECKED_SETTINGS, check_constrained_gpu_settings
from pyorps.utils.traversal_gpu import describe_gpu_error, gpu_out_of_memory_error, is_gpu_out_of_memory

#: A merged end span within this relative distance of ``max_span_m`` counts as reaching it (the search keeps every
#: span strictly below the longest span; distances computed from cell centres carry rounding of this size).
_SPAN_REL_TOL = 1e-9

class _FindersSetting:
    """Default of the ``find_route`` connection-height keywords: "the finder's own setting" (None = not set)."""

    def __repr__(self):
        return "<the finder's setting>"


_UNSET = _FindersSetting()

#: A relative difference between the x and the y cell size above which the cells count as not square.
_SQUARE_CELL_TOL = 1e-6

#: The CPU kernels switch from their dense state array to a sparse (hash map) form above this many states
#: (``constrained_dijkstra_2d`` and ``constrained_delta_stepping_height_2d`` do it in the kernel, the finder hands a
#: clearance search above it to ``constrained_delta_stepping_lazy``). Mirrors the literal in the kernels.
_KERNEL_DENSE_MAX_STATES = 500_000_000

#: Bytes the dense kernels hold per search state (``StateData``: double distance, int64 predecessor, float span, two
#: flag bytes, padding).
_DENSE_BYTES_PER_STATE = 24
#: Bytes per (cell, direction) of the intermediate-cell caches every kernel builds (status byte and float cost).
_CACHE_BYTES_PER_CELL_DIRECTION = 5
#: Bytes per (cell, direction) of the gradient-penalty cache (kernels that get a DEM).
_GRADIENT_BYTES_PER_CELL_DIRECTION = 4
#: Bytes per REACHED state of the sparse forms: four node-based hash containers (distance, predecessor, span,
#: visited) of about 40 to 50 bytes an entry and the heap, in ``constrained_dijkstra_2d``.
_DIJKSTRA_SPARSE_BYTES_PER_STATE = 200
#: The same for ``constrained_delta_stepping_lazy`` (one node-based hash map entry holding a 24-byte state).
_LAZY_BYTES_PER_STATE = 64

#: Most failing spans a ``SpanRuleWarning`` lists; ``ConstrainedPath.span_notes`` has them all.
_MAX_LISTED_SPAN_ISSUES = 6

#: Budget (loop steps) of the search that matches the kernel's tower cells to positions along the route.
_TOWER_MATCH_STEPS = 200_000

#: The keyword arguments of the kernels that carry the line-end settings (the kernels reject a bool or a str).
_LINE_END_KERNEL_ARGUMENTS = ("target_min_span", "source_attach_height", "target_attach_height")


@dataclass(frozen=True)
class ExactBinsMemory:
    """What the CPU kernel of a run with exact span bins allocates (see ``ConstrainedPathFinder.exact_bins_memory``).

    Attributes:
        kernel: Name of the kernel the finder calls for this run.
        form: ``"dense"`` (arrays sized by the state space) or ``"sparse"`` (hash maps that grow with the states
            the search reaches).
        states: Cells x directions x span bins x height classes.
        nbytes: The estimate in bytes; for the sparse form the worst case, every state reached.
        detail: One sentence on what the estimate assumes.
    """

    kernel: str
    form: str
    states: int
    nbytes: int
    detail: str

#: Under ``terminal_span_rule="enforce"`` an end span this far (relative) below ``min_span_m`` is a span of exactly
#: ``min_span_m`` whose distance was rounded down, not a violation.
_ENFORCE_ROUNDING = 1e-6


@dataclass
class _EndSpanOutcome:
    """What the end-span rule did to one route (private to :class:`ConstrainedPathFinder`).

    Everything is keyed by POSITION in ``path_indices`` (the index along the route), never by cell: a route that
    crosses itself passes a cell twice and can carry a tower at each passage.

    Attributes:
        positions: Positions of the towers the report keeps, ascending: the line towers and the two terminals
            (positions 0 and ``len(path) - 1``).
        terminals: The positions of the two terminal towers.
        search_chain: Positions of the start, every search tower and the end (before any removal).
        removed: Position of every search tower the rule removed, in path order.
        kept_notes: Note for a kept line tower (by position) next to an end point.
        removed_notes: Note for a removed tower (by position).
        issues: ``(text, warn)`` per flagged or broken span: ``warn`` is True where a ``SpanRuleWarning`` is due.
        failed_spans: ``(first position, second position)`` of the spans already reported as failing the clearance.
        blind_spots: Number of failing spans that only the strict check found (the search does not test the first
            cell after a tower, nor a span of one step).
    """

    positions: list = field(default_factory=list)
    terminals: set = field(default_factory=set)
    search_chain: list = field(default_factory=list)
    removed: list = field(default_factory=list)
    kept_notes: dict = field(default_factory=dict)
    removed_notes: dict = field(default_factory=dict)
    issues: list = field(default_factory=list)
    failed_spans: set = field(default_factory=set)
    blind_spots: int = 0


@dataclass
class _EndSpanGeometry:
    """Ground data for the Python clearance check of a route's end spans (None when the search checked none).

    The heights a check uses are the ones the report states: each line tower's own height (the kernels report it
    for every tower, also with several tower heights) and, for the two ends, the connection height, else (source
    with several tower heights) the height class the search charged, else the profile's terminal height, else the
    line height.
    """

    dem: np.ndarray
    obstacle: np.ndarray | None
    rows: np.ndarray
    cols: np.ndarray


@dataclass
class _LineEnds:
    """The line-end settings of one run and what the search did with them (private to the finder).

    Attributes:
        source_height_m, target_height_m: The connection heights of this run (None = not set). They are reported as
            the end towers' heights whether or not the search could use them.
        target_min_span: Metres, 0 = off. ``min_span_m`` for ``terminal_span_rule="enforce"`` (with span limits):
            handed to the kernels as ``target_min_span``.
        source_attach, target_attach: The heights the search got (``source_attach_height`` /
            ``target_attach_height``); None when not set or when the clearance check is off.
        span_bin_size: Span-bin size of the search in metres (the finder's bins, or the exact bins of a run with
            the new settings).
        exact_bins: True when ``span_bin_size`` was shortened for the new settings.
        n_span_bins: Number of bins of the search.
        base_height_shortcut: The lowest tower height (metres) when a default run with several tower heights
            returned the route of its lowest-height-only search (see ``_run_cython_backend``), else None.
        source_search_height: The height the search gave the structure at the source (the height class it
            charged); None when the question does not arise (see ``_source_height_class``).
    """

    source_height_m: float | None = None
    target_height_m: float | None = None
    target_min_span: float = 0.0
    source_attach: float | None = None
    target_attach: float | None = None
    span_bin_size: float | None = None
    exact_bins: bool = False
    n_span_bins: int | None = None
    base_height_shortcut: float | None = None
    source_search_height: float | None = None

    @property
    def in_search(self) -> bool:
        """True when a new setting changed the search itself (an empty result then has a reason to name)."""
        return self.target_min_span > 0.0 or self.source_attach is not None or self.target_attach is not None


class ConstrainedPathFinder(PathFinder):
    """PathFinder extension for infrastructure-constrained routing.

    Supports turn-angle limits and intermediate structure (tower) placement
    via coupled extended-state Dijkstra. Inherits all PathFinder parameters
    and capabilities.

    Uses exact span tracking (float) to avoid quantization drift when
    span_bin_size > step_distance, enabling correct coupled route+tower
    optimization at any raster resolution.

    Objective weights (``objective=``, 'cython' backend only): the search
    minimises ``J = sum_k w_k * M_k(route) + w_tower * T + w_smooth * A``,
    where ``M_k`` are the cell layers of the objective (with ``"length"``),
    ``T`` the profile's tower costs (foundation by land use plus tower type)
    and ``A`` the profile's turn penalties. ``"tower"`` and ``"smoothness"``
    default to 1, every other missing weight is 0. The kernel adds all of
    these terms in ONE unit: the search raster is the objective times the
    metric stack's quantisation scale ``s`` (``CombineResult.scale``:
    ``65534 / max(F)``, or ``1 / w_cost`` for a cost-only objective on a
    single raster), and the turn and tower tables are multiplied by the
    same ``s`` before the search. Reported costs are in the user's units
    (see ``_build_constrained_path``). Without an objective, or with
    ``{"cost": 1}`` and the reserved weights at 1, ``s`` is 1 and nothing
    is rescaled.
    """

    # raster_gpu_v3 was removed 2026-08-11: it returned an empty path on an
    # obstacle-free raster at both production tuning values, and v4 does not
    # build on it. The unconstrained SSSP "V3" kernel in sssp_gpu.py is a
    # different lineage and is unaffected.
    SUPPORTED_BACKENDS = ("cython", "cython_parallel", "raster_gpu",
                          "raster_gpu_v4")

    #: The graph_api values that run a constrained GPU solver (keys of
    #: ``pyorps.utils.constrained_sssp_gpu.CONSTRAINED_GPU_SOLVERS``).
    GPU_BACKENDS = ("raster_gpu", "raster_gpu_v4")

    #: Largest search state space (cells x directions x span bins x height classes) a run with exact span bins
    #: (``terminal_span_rule="enforce"`` or a connection height) may have. Above it the run is refused with the
    #: numbers instead of starting a search that would not fit in memory. A judgement value: the kernels switch to
    #: their sparse (hash map) form above 5e8 states and then use memory in proportion to the states they reach.
    #: The states are only half of the bound: ``EXACT_BINS_MAX_BYTES`` limits the memory the kernel allocates for
    #: them, which depends on the kernel (see ``exact_bins_memory``).
    EXACT_BINS_MAX_STATES = 2_000_000_000

    #: Largest memory (bytes) the CPU kernel of a run with exact span bins may allocate, by the estimate of
    #: ``exact_bins_memory``. Default 4 GiB. Above it the run is refused with the estimate, the limit and the state
    #: count. Raise it (and ``EXACT_BINS_MAX_STATES``) on a machine that has the memory.
    EXACT_BINS_MAX_BYTES = 4 * 1024 ** 3

    #: The kernels hold a span bin in 16 bits.
    EXACT_BINS_MAX_COUNT = 65535

    #: Most steps of the shortest step that the longest span may be for the bins to stay exact. The kernels add a span
    #: up in single precision; a float32 sum of K steps is off by at most about K * 2**-24 of its length, and the
    #: exact bins keep a margin of 0.25 / K of a bin from the edge of the bin a span of K steps belongs to. In the
    #: worst case (every rounding in the same direction) a span can therefore cross a bin edge from K = 2048 on
    #: (K * K * 2**-24 = 0.25). A float32 simulation of the kernel's own arithmetic (every direction of r1 to r3,
    #: 306 cell sizes between 0.003 and 10 m) found no merged bins up to K = 4096 and the first ones near K = 4500;
    #: 2048 is the provable bound and sits well below that.
    EXACT_BINS_MAX_STEPS = 2048

    #: Keyword arguments find_route() reads besides its named parameters.
    _FIND_ROUTE_KEYWORDS = ("source", "target", "source_connection_height_m", "target_connection_height_m")
    #: Keyword arguments find_route() accepts although nothing reads them.
    #: Empty on purpose: no caller in the package (the GUI service
    #: ``pyorps.gui.services.constrained`` included) passes one. Add a name
    #: here, with the caller that needs it, rather than widening the check.
    _FIND_ROUTE_IGNORED_KEYWORDS = ()

    def __init__(self, dataset_source, source_coords, target_coords,
                 profile, graph_api="cython", neighborhood_str="r2",
                 dsm=None, source_connection_height_m=None,
                 target_connection_height_m=None, **kwargs):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        """Initialize constrained path finder.

        Parameters:
            dataset_source: Raster or vector data source (same as PathFinder).
            source_coords: Source coordinates (same as PathFinder).
            target_coords: Target coordinates (same as PathFinder).
            profile: InfrastructureProfile, dict, or path (str or
                os.PathLike, e.g. pathlib.Path) to a YAML/JSON profile.
            graph_api: One of ``SUPPORTED_BACKENDS``. The GPU backends do
                not implement every constraint: "raster_gpu_v4" has no DEM
                gradient, clearance or exact tower footprint, and
                "raster_gpu" enforces max_span_m only on its span-bin grid.
                find_route() raises AlgorithmNotImplementedError when the
                profile and data ask for something the backend would
                ignore (see ``pyorps.utils.constrained_sssp_gpu``).
            neighborhood_str: Neighborhood string (e.g. "r1", "r2").
            dsm: Digital Surface Model (includes trees/buildings). When both
                 dem and dsm are provided, obstacle_heights = dsm - dem is
                 computed automatically for clearance checking.
            source_connection_height_m: Height (metres above the ground at
                 the source cell) at which the conductor leaves the
                 structure at the source, e.g. a substation gantry. None
                 (default) = not set. With a DEM and a profile that has
                 clearance data the search uses it for the first span (the
                 first tower's position and height are chosen for it; the
                 source then pays no height premium); it is also the reported
                 height of the source terminal tower. The profile's
                 ``terminal_tower_height_m`` stays a report-only value. A
                 number above 0, else InvalidTypeError/InvalidInputError.
                 ``find_route`` can override it per call.
            target_connection_height_m: The same for the target: the last
                 span must clear the ground down to this height at the
                 target, so the last tower's position and height are chosen
                 for it.
            **kwargs: All other PathFinder parameters (search_space_buffer_m,
                      cost_assumptions, dem, etc.). They are checked by
                      PathFinder: an unknown name raises TypeError.

        Raises:
            TypeError: profile is none of the accepted types.
            InvalidTypeError: a connection height is not a number.
            InvalidInputError: a connection height is not finite or not
                above 0.

        Exact span bins: the search only merges states that share a span
        bin, and with the coarse bins of a default run (``min_span_m``) it
        can miss the cheapest route, or report no route at all when an
        end-span rule asks for a span the merged states no longer tell
        apart. A run with ``terminal_span_rule="enforce"`` or with a
        connection height the search can use therefore takes exact bins
        (no longer than the shortest step) and says so in
        ``ConstrainedPath.span_bin_size_m`` and ``search_notes``. It is
        refused with InvalidInputError (and the numbers) when the state
        space is above ``EXACT_BINS_MAX_STATES``, when the memory the
        kernel would allocate for it (``exact_bins_memory``) is above
        ``EXACT_BINS_MAX_BYTES``, or when the longest span is more than
        ``EXACT_BINS_MAX_STEPS`` steps long (single-precision sums would
        merge neighbouring bins). Default runs keep their bins and results.

        ``"enforce"`` (exact bins) can therefore return a CHEAPER route than
        ``"keep"`` or ``"remove"`` on the same problem. Those rules keep the
        coarse default bins, which can miss the optimum.

        With several tower heights, a DEM and clearance data, a default run
        first searches with the lowest height alone and returns that route
        if one exists, without comparing it with routes that use taller
        towers at their premium (a known, separately tracked limitation: the
        result says so in ``search_notes``). A run that uses a line-end
        setting (``"enforce"`` or a connection height the search uses) skips
        this shortcut and always runs the full search over all heights.
        """
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        if graph_api not in self.SUPPORTED_BACKENDS:
            raise InvalidInputError(
                f"ConstrainedPathFinder requires graph_api in "
                f"{self.SUPPORTED_BACKENDS}, got '{graph_api}'"
            )

        # Load profile before super().__init__ so it's available if needed
        if isinstance(profile, (str, os.PathLike)):
            self._profile = InfrastructureProfile.load(os.fsdecode(profile))
        elif isinstance(profile, dict):
            self._profile = InfrastructureProfile.from_dict(profile)
        elif isinstance(profile, InfrastructureProfile):
            self._profile = profile
        else:
            raise InvalidTypeError(
                "profile must be an InfrastructureProfile, a dict or a path "
                "(str or os.PathLike) to a YAML/JSON profile file, got "
                f"{type(profile).__name__}")

        # Store the requested backend for constrained routing
        self._constrained_backend = graph_api
        self._dsm_source = dsm

        # Connection-point heights of the two line ends (validated here, so a wrong value fails at construction)
        self._source_connection_height_m = self._validate_connection_height(
            source_connection_height_m, "source_connection_height_m")
        self._target_connection_height_m = self._validate_connection_height(
            target_connection_height_m, "target_connection_height_m")
        self._warned_once = set()

        # Initialize parent — sets up raster_handler, steps, etc.
        # Both "cython" and "raster_gpu" produce directed steps in parent
        super().__init__(
            dataset_source=dataset_source,
            source_coords=source_coords,
            target_coords=target_coords,
            graph_api=graph_api,
            neighborhood_str=neighborhood_str,
            **kwargs,
        )

        # Load DSM and compute obstacle heights (DSM - DEM)
        self._obstacle_data = None
        self._load_obstacle_data()

        # Feasibility objective integration (plan Phase 8): the metric
        # pipeline already produced the combined weight raster through the
        # parent; here we validate what the constrained kernels support
        # and apply the reserved "smoothness"/"tower" weights host-side.
        self._tower_cost_raster = None
        if self.objective is not None:
            self._check_objective_support(self.objective)
            if self.dem_dataset is not None:
                warnings.warn(
                    "Constrained routing with a DEM uses the profile's "
                    "gradient model, not the objective's 3D stretch / "
                    "response LUTs.", UserWarning, stacklevel=2)

        # Precompute LUTs using parent's steps
        self._angle_cost_lut, self._angle_valid_lut = (
            self._profile.precompute_angle_lut(self.steps)
        )
        self._assert_neighborhood_allows_turns()
        cell_size = abs(self.raster_handler.window_transform.a)
        self._cell_size = cell_size
        self._warn_if_cells_not_square()
        self._step_distances = self._profile.compute_step_distances(
            self.steps, cell_size
        )

        # Coarsen span bins for coupled Dijkstra performance.
        # Since exact span is tracked via float (span_dist), bins are only
        # used for state deduplication — not for span enforcement.
        # Using min_span as bin size gives just ceil(max/min) bins (typically
        # 2) instead of ceil(max/bin_size) (typically 40). This reduces the
        # state space ~20x, often the difference between sparse and dense.
        if self._profile.has_span_constraints:
            effective_bin = self._profile.min_span_m
            self._effective_span_bin_size = effective_bin
            self._effective_n_span_bins = max(
                2, math.ceil(self._profile.max_span_m / effective_bin)
            )
        else:
            self._effective_span_bin_size = 1e6
            self._effective_n_span_bins = 1
        # The line-end settings of the current run (find_route fills them per call)
        self._ends = self._new_line_ends(self._source_connection_height_m, self._target_connection_height_m)

        # Rotated square offsets for "exact" area cost mode (set for all
        # profiles so backends can rely on the attributes existing)
        self._area_offsets = None
        self._area_offset_starts = None
        self._area_offset_counts = None

        if self._profile.has_span_constraints:
            self._tower_terrain_costs = (
                self._profile.precompute_tower_terrain_costs()
            )
            self._tower_angle_costs = (
                self._profile.precompute_tower_angle_costs(self.steps)
            )
            if (self._profile.tower_area_cost_mode == "exact"
                    and self._profile.tower_ground_area_m2 > 1.0):
                self._precompute_area_offsets()
        else:
            self._tower_terrain_costs = np.zeros(65536, dtype=np.float64)
            self._tower_angle_costs = np.zeros(
                (len(self.steps), len(self.steps)), dtype=np.float64
            )

        # The profile's own (unweighted) tables: the source of the weighted
        # tables (_apply_objective_weights) and of the report's "profile_*"
        # entries. They are never modified.
        self._profile_angle_cost_lut = self._angle_cost_lut
        self._profile_tower_terrain_costs = self._tower_terrain_costs
        self._profile_tower_angle_costs = self._tower_angle_costs
        self._w_smooth = 1.0
        self._w_tower = 1.0
        # Search units per user unit (set per find_route, see
        # _objective_search_scale) and the cost-layer index of every cell
        # for the tower foundation report (objective mode only).
        self._search_scale = 1.0
        self._tower_site_index = None

        self._apply_objective_weights()
        if self.objective is not None:
            self._warn_if_ground_unpriced(stacklevel=3)

    @staticmethod
    def _validate_connection_height(value, name):
        """A connection height as a float, or None (not set).

        Raises:
            InvalidTypeError: ``value`` is not a number (a bool is not).
            InvalidInputError: ``value`` is not finite or not above 0.
        """
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise InvalidTypeError(
                f"{name} must be a number (metres above the ground at the end cell) or None, "
                f"got {type(value).__name__}")
        height = float(value)
        if not math.isfinite(height) or height <= 0.0 or height > float(np.finfo(np.float32).max):
            raise InvalidInputError(
                f"{name} must be a finite number above 0 (metres above the ground at the end cell), got {value!r}")
        return height

    def _new_line_ends(self, source_height_m, target_height_m):
        """The line-end settings of a run that has not started: the heights and the finder's own span bins.

        ``span_bin_size`` is None for a profile without span limits (the kernels get a placeholder bin then, which is
        not a span-bin size of the search).
        """
        spans = self._profile.has_span_constraints
        return _LineEnds(
            source_height_m=source_height_m, target_height_m=target_height_m,
            span_bin_size=self._effective_span_bin_size if spans else None,
            n_span_bins=self._effective_n_span_bins)

    @property
    def source_connection_height_m(self):
        """Height of the conductor attachment at the source (metres above the ground), or None (not set)."""
        return self._source_connection_height_m

    @source_connection_height_m.setter
    def source_connection_height_m(self, value):
        self._source_connection_height_m = self._validate_connection_height(value, "source_connection_height_m")

    @property
    def target_connection_height_m(self):
        """Height of the conductor attachment at the target (metres above the ground), or None (not set)."""
        return self._target_connection_height_m

    @target_connection_height_m.setter
    def target_connection_height_m(self, value):
        self._target_connection_height_m = self._validate_connection_height(value, "target_connection_height_m")

    def _warn_once(self, key, message):
        """One ``PyorpsWarning`` per finder and ``key``, attributed to the caller's line."""
        if key in self._warned_once:
            return
        self._warned_once.add(key)
        warnings.warn(message, PyorpsWarning, skip_file_prefixes=_INTERNAL_FILE_PREFIXES)

    def _warn_if_cells_not_square(self) -> None:
        """Warn when the y cell size differs from the x size: every distance of the search uses the x size."""
        transform = self.raster_handler.window_transform
        cell_x, cell_y = abs(transform.a), abs(transform.e)
        if cell_x > 0.0 and abs(cell_y - cell_x) > _SQUARE_CELL_TOL * cell_x:
            self._warn_once(
                "non_square_cells",
                f"The raster cells are not square (x size {cell_x:g}, y size {cell_y:g}). Constrained routing "
                f"measures every distance with the x size ({cell_x:g}): path lengths, spans, step costs and tower "
                f"distances are computed as if a cell were {cell_x:g} wide in both directions, so every length "
                f"along y is off by the factor {cell_y / cell_x:.4g}. Resample the raster to square cells for "
                f"exact lengths.")

    def _check_objective_support(self, objective) -> None:
        """Refuse the objectives the constrained kernels cannot honour.

        Called at construction and by :meth:`set_objective`, so that both
        refuse the same things with the same errors.

        Raises:
            FeatureNotImplementedError: ``objective`` has a gradient term, the
                backend is not 'cython', or the profile has clearance
                constraints.
        """
        if objective.has_gradient_terms:
            raise FeatureNotImplementedError(
                "Constrained routing keeps the PROFILE's gradient "
                "model (gradient_cost_params / max_gradient_percent); "
                "objective gradient terms are not wired into the "
                "constrained kernels yet.")
        if self._constrained_backend != "cython":
            raise FeatureNotImplementedError(
                "Feasibility objectives on constrained routing are "
                "supported on the 'cython' backend (per-cell tower "
                "costs); parallel/GPU constrained variants are "
                "pending.")
        if self._profile.has_clearance_constraints:
            raise FeatureNotImplementedError(
                "Feasibility objectives with clearance constraints "
                "are not wired yet (the clearance kernels lack the "
                "per-cell tower-cost raster).")

    def _apply_objective_weights(self) -> None:
        """Derive the weighted tables and weights from the CURRENT objective.

        Reserved objective weights: "smoothness" scales the turn-angle
        penalty LUT, "tower" scales the tower cost terms. Default 1.0
        (the profile LUTs ARE the steering magnitudes); explicit 0
        disables the term. Hard limits (angle_valid_lut) are
        constraints, never scaled. The weighted tables stay in the
        user's units here; find_route multiplies them by the search
        scale right before the kernel call.

        The single source of truth: always computed from the profile's own
        tables and ``self.objective``, never from an earlier weighting, so
        it is correct after any change of the objective (construction,
        :meth:`set_objective`, the ensemble's restore, ``objective = None``).
        Called at construction, by :meth:`set_objective` and at the start of
        every :meth:`find_route`.
        """
        w_smooth = 1.0
        w_tower = 1.0
        if self.objective is not None:
            w_smooth = self.objective.weights.get("smoothness", 1.0)
            w_tower = self.objective.weights.get("tower", 1.0)
        self._w_smooth = w_smooth
        self._w_tower = w_tower
        self._angle_cost_lut = self._profile_angle_cost_lut
        self._tower_terrain_costs = self._profile_tower_terrain_costs
        self._tower_angle_costs = self._profile_tower_angle_costs
        if w_smooth != 1.0:
            self._angle_cost_lut = self._profile_angle_cost_lut.copy()
            finite = np.isfinite(self._angle_cost_lut)
            self._angle_cost_lut[finite] *= w_smooth
        if w_tower != 1.0:
            self._tower_terrain_costs = (
                self._profile_tower_terrain_costs * w_tower)
            self._tower_angle_costs = (
                self._profile_tower_angle_costs * w_tower)

    def _warn_if_ground_unpriced(self, stacklevel: int) -> None:
        """Warn when the objective prices no traversable cell (see ``_ground_unpriced``)."""
        if not self._ground_unpriced():
            return
        warnings.warn(
            "The objective is 0 on every traversable cell (every weighted "
            "cell layer is 0 there, and there is no 'length' weight), so "
            "constrained routing prices the ground at 0 (a missing weight is 0): "
            "the route minimises only the weighted tower and turn "
            "costs, and routes of equal tower and turn cost are "
            "equally good (the one returned is arbitrary). The "
            "uniform minimal weight the metric stack substitutes is "
            "not used here. Add a 'cost' weight, or a small 'length' "
            "weight, to price the ground.",
            PyorpsWarning, stacklevel=stacklevel)

    def set_objective(self, objective, gradient_options=None) -> None:
        """Swap the feasibility objective; tables and weights follow it.

        Everything :class:`PathFinder` does, plus what constrained routing
        needs: the objective is refused with the same errors as at
        construction (a ``gradient`` weight, a backend other than 'cython',
        clearance constraints) and the weighted turn and tower tables follow
        the new objective. On a refusal the finder keeps its previous
        objective.
        """
        normalized = self._normalize_objective(objective, gradient_options)
        if normalized is not None and self.metric_stack is not None:
            # The refusals of the constructor, in its order (the stack's
            # gradient check first), BEFORE anything is recombined: a refused
            # objective leaves the finder exactly as it was.
            previous = self.objective
            self.objective = normalized
            try:
                self._validate_gradient_support()
                self._check_objective_support(normalized)
            finally:
                self.objective = previous
        super().set_objective(objective, gradient_options)
        self._apply_objective_weights()
        self._warn_if_ground_unpriced(stacklevel=3)

    def _ground_unpriced(self) -> bool:
        """True when the objective prices no cell (and no length).

        That is the metric stack's degenerate branch: the combined surface
        is 0 on every traversable cell, and ``combine`` returns a uniform
        minimal weight of 1 with scale 1 (and warns). A plain search needs
        that fallback to have a minimum at all; constrained routing does not
        (the tower and turn terms still define one), and the uniform weight
        would be a hidden length term of arbitrary size next to the tower
        costs. find_route therefore searches on a ground cost of exactly 0.
        """
        if self.objective is None:
            return False
        result = self._combine_result
        return (result is not None and not result.legacy_passthrough
                and result.feasibility is not None
                and result.resolution == 0.0
                and result.weights.dtype == np.uint16)

    def _objective_search_scale(self) -> float:
        """Search units per user unit of the objective.

        The quantisation scale of the metric stack (``CombineResult.scale``):
        the search raster holds the objective times this scale, so the turn
        and tower tables are multiplied by it before the search and every
        reported cost is divided by it (or, for the profile tables, given in
        the user's units directly). 1.0 without an objective and for a
        cost-only objective with ``cost`` 1 (the zero-copy fast path), so
        those runs are bit-identical to the legacy search.
        """
        if self.objective is None:
            return 1.0
        result = self._combine_result
        if result is None:
            return 1.0
        return float(result.scale)

    def _search_tables(self):
        """The turn and tower tables in search units, as the kernel takes them.

        Returns:
            Tuple ``(angle_cost_lut, tower_terrain_costs, tower_angle_costs)``
            of float32 arrays: the (weighted) tables times the search scale.
            With a scale of 1 these are exactly the legacy arguments.
        """
        angle_lut = self._angle_cost_lut
        tower_terrain = self._tower_terrain_costs
        tower_angle = self._tower_angle_costs
        scale = self._search_scale
        if scale == 1.0:
            return (angle_lut.astype(np.float32),
                    tower_terrain.astype(np.float32),
                    tower_angle.astype(np.float32))
        scaled = []
        for name, table in (("turn penalty", angle_lut),
                            ("tower foundation", tower_terrain),
                            ("tower type", tower_angle)):
            # inf (hard angle limit) stays inf. A finite entry must stay
            # finite in float32: only absurdly small objective weights
            # (scale = 65534 / max(F), or 1 / w_cost) could overflow it.
            with np.errstate(over="ignore"):  # checked right below
                product = (table * scale).astype(np.float32)
            if not np.isfinite(product[np.isfinite(table)]).all():
                raise NumericalPrecisionError(
                    f"The {name} costs times the search scale {scale:.6g} "
                    f"exceed the float32 range of the constrained kernel: "
                    f"the objective's cell weights are too small next to "
                    f"its 'tower' and 'smoothness' weights. Raise the cell "
                    f"weights or lower 'tower' / 'smoothness' (only the "
                    f"ratios of the weights matter).")
            scaled.append(product)
        return tuple(scaled)

    def _assert_neighborhood_allows_turns(self) -> None:
        """Warn when a neighborhood is too coarse for ``hard_angle_limit_deg``.

        ``r0``/``r1`` space their directions 90 deg / 45 deg apart. When
        that spacing exceeds the profile's ``hard_angle_limit_deg``, every
        OFF-axis turn in ``_angle_valid_lut`` is invalid -- the router can
        only continue straight in whatever direction it started. A route
        colinear with one of those directions still finds a route (no
        turn needed, see ``test_end_to_end_with_380kv_profile``'s pure
        diagonal), so this cannot be a hard error; but any OTHER source/
        target pair fails with an opaque ``NoPathFoundError`` far from the
        actual cause. Measured on the shipped 110 kV profile (r1 45 deg
        spacing vs a 40 deg hard limit): every non-straight turn invalid.
        """
        valid = self._angle_valid_lut
        n_dirs = valid.shape[0]
        if n_dirs == 0:
            return
        off_diagonal = valid & ~np.eye(n_dirs, dtype=bool)
        if not off_diagonal.any():
            warnings.warn(
                f"neighborhood_str={self.neighborhood_str!r} ({n_dirs} "
                f"directions) cannot satisfy hard_angle_limit_deg="
                f"{self._profile.hard_angle_limit_deg} -- every turn "
                f"between distinct directions is invalid, so a route "
                f"will only be found if source and target are exactly "
                f"colinear with one direction. Use a finer neighborhood "
                f"(e.g. 'r2') or raise hard_angle_limit_deg in the "
                f"profile if that is not intentional.",
                UserWarning, stacklevel=2)

    def _load_obstacle_data(self):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        """Load DSM and compute obstacle heights (DSM - DEM).

        When both a DSM source and a DEM raster handler are available,
        computes obstacle_heights = dsm - dem (trees, buildings) and
        resamples both to match the cost raster shape.
        """
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        if (self._dsm_source is not None and
                hasattr(self, 'dem_raster_handler') and
                self.dem_raster_handler is not None):
            from pyorps.io.geo_dataset import RasterDataset, initialize_geo_dataset
            from pyorps.raster.handler import RasterHandler
            dsm_dataset = initialize_geo_dataset(
                self._dsm_source, crs=self.dataset.crs)
            if isinstance(dsm_dataset, RasterDataset):
                dsm_dataset.load_data()
                dsm_handler = RasterHandler(
                    dsm_dataset,
                    self.source_coords, self.target_coords,
                    self.search_space_buffer_m, apply_mask=False)
                dsm_raw = dsm_handler.data
                if len(dsm_raw.shape) > 2:
                    dsm_raw = dsm_raw[0]
                dem_raw = self.dem_raster_handler.data
                if len(dem_raw.shape) > 2:
                    dem_raw = dem_raw[0]
                # Resample both to cost raster shape
                target_shape = self.raster_handler.data[0].shape
                if dsm_raw.shape != target_shape:
                    from scipy.ndimage import zoom
                    zf = (target_shape[0] / dsm_raw.shape[0],
                          target_shape[1] / dsm_raw.shape[1])
                    dsm_raw = zoom(dsm_raw, zf, order=1)
                if dem_raw.shape != target_shape:
                    from scipy.ndimage import zoom
                    zf = (target_shape[0] / dem_raw.shape[0],
                          target_shape[1] / dem_raw.shape[1])
                    dem_raw = zoom(dem_raw, zf, order=1)
                # Obstacle heights = surface - ground (trees, buildings)
                obstacle = (dsm_raw.astype(np.float32) -
                            dem_raw.astype(np.float32))
                obstacle[obstacle < 0] = 0  # clamp negatives
                self._obstacle_data = obstacle

    def _precompute_area_offsets(self):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        """Precompute rotated square pixel offsets per direction pair.

        For each (d_in, d_out) pair, computes the set of (dr, dc) pixel
        offsets that fall within a square of side sqrt(ground_area_m2),
        rotated to bisect the incoming and outgoing direction angles.

        Results are packed into flat arrays for Cython consumption:
        - area_offsets: int32 flat [dr0, dc0, dr1, dc1, ...] all pairs
        - area_offset_starts: int32[n_dirs*n_dirs] start index per pair
        - area_offset_counts: int32[n_dirs*n_dirs] offset count per pair
        """
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        side_m = math.sqrt(self._profile.tower_ground_area_m2)
        half_side_px = side_m / (2.0 * self._cell_size)

        n_dirs = len(self.steps)
        angles = [math.atan2(float(dc), float(dr))
                  for dr, dc in self.steps]

        all_offsets = []
        starts = np.zeros(n_dirs * n_dirs, dtype=np.int32)
        counts = np.zeros(n_dirs * n_dirs, dtype=np.int32)

        # Search radius: diagonal of the square in pixels
        r_max = int(math.ceil(half_side_px * math.sqrt(2))) + 1

        for d_in in range(n_dirs):
            for d_out in range(n_dirs):
                # Bisector angle
                bisector = (angles[d_in] + angles[d_out]) / 2.0
                diff = angles[d_out] - angles[d_in]
                if abs(math.atan2(math.sin(diff), math.cos(diff))) > math.pi:
                    bisector += math.pi

                cos_b = math.cos(-bisector)
                sin_b = math.sin(-bisector)
                offsets = []
                for dr in range(-r_max, r_max + 1):
                    for dc in range(-r_max, r_max + 1):
                        # Inverse rotate to check if in unit square
                        local_r = dr * cos_b - dc * sin_b
                        local_c = dr * sin_b + dc * cos_b
                        if (abs(local_r) <= half_side_px
                                and abs(local_c) <= half_side_px):
                            offsets.append((dr, dc))

                pair_idx = d_in * n_dirs + d_out
                starts[pair_idx] = len(all_offsets)
                counts[pair_idx] = len(offsets)
                all_offsets.extend(offsets)

        if all_offsets:
            flat = np.array(all_offsets, dtype=np.int32).flatten()
        else:
            flat = np.zeros(0, dtype=np.int32)
        self._area_offsets = flat
        self._area_offset_starts = starts
        self._area_offset_counts = counts

    @property
    def profile(self):
        """The infrastructure profile used for constrained routing."""
        return self._profile

    def find_route(self, source=None, target=None, source_connection_height_m=_UNSET,  # pylint: disable=arguments-differ
                   target_connection_height_m=_UNSET, **kwargs):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        """Find constrained optimal route with tower placement.

        Parameters:
            source: Override source coordinates (default: use init coords).
            target: Override target coordinates (default: use init coords).
            source_connection_height_m: Override the finder's connection
                height at the source for this call (see ``__init__``). Left
                out = the finder's setting; None = not set for this call.
            target_connection_height_m: The same for the target.
            **kwargs: None are accepted. The parameter only exists so that
                a PathFinder.find_route option (algorithm=, simplify=, ...)
                or a misspelt name is rejected with the valid names instead
                of being dropped: constrained routing has no other per-call
                options, the solver is chosen with graph_api= at
                construction.

        Returns:
            ConstrainedPath with towers, cost breakdown, and geometry. It
            records what the run used: ``terminal_span_rule``,
            ``source_connection_height_m``, ``target_connection_height_m``,
            ``span_bin_size_m`` and ``search_notes``.

        Raises:
            TypeError: An unknown keyword argument was passed.
            InvalidTypeError, InvalidInputError: A connection height is not a
                number, or not finite and above 0.
            PyorpsError: source or target is NaN/infinite or lies outside
                the search window (it would otherwise reach a kernel as an
                out-of-raster index).
            AlgorithmNotImplementedError: A GPU backend was chosen and the
                profile/data set a constraint it does not enforce (this
                includes ``terminal_span_rule="enforce"`` and a connection
                height the search would use: no GPU solver has them).
            InvalidInputError: The exact span bins of a run with
                ``terminal_span_rule="enforce"`` or a connection height make
                the search state space larger than
                ``EXACT_BINS_MAX_STATES`` (or the bins do not fit in 16 bits).
            NoPathFoundError: No route satisfies ``terminal_span_rule=
                "enforce"`` or a connection height the search used. A run
                with none of these settings returns an empty
                ConstrainedPath instead, as before.
        """
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        reject_unknown_options(
            kwargs, self._FIND_ROUTE_KEYWORDS, "ConstrainedPathFinder.find_route()",
            ignored=self._FIND_ROUTE_IGNORED_KEYWORDS,
            hint="Constrained routing has no other per-call options; the solver is chosen with graph_api= at "
                 "construction.")

        t_start = time.time()
        self._ends = self._new_line_ends(
            (self._source_connection_height_m if source_connection_height_m is _UNSET
             else self._validate_connection_height(source_connection_height_m, "source_connection_height_m")),
            (self._target_connection_height_m if target_connection_height_m is _UNSET
             else self._validate_connection_height(target_connection_height_m, "target_connection_height_m")))

        # Resolve coordinates (allow overrides like parent)
        src = source if source is not None else self.source_coords
        tgt = target if target is not None else self.target_coords

        if src is None or tgt is None:
            raise InvalidInputError("Source and target coordinates must not be None!")

        src = PathFinder.normalize_coordinates(src)
        tgt = PathFinder.normalize_coordinates(tgt)

        # Convert CRS coordinates to local raster row/col. check_bounds: a
        # point outside the search window raises here, with the window
        # bounds, instead of reaching a kernel as an out-of-raster index.
        src_indices = self.raster_handler.coords_to_indices([src], check_bounds=True)
        tgt_indices = self.raster_handler.coords_to_indices([tgt], check_bounds=True)
        source_row, source_col = int(src_indices[0][0]), int(src_indices[0][1])
        target_row, target_col = int(tgt_indices[0][0]), int(tgt_indices[0][1])

        # Get raster data (first band, windowed section)
        raster = self.raster_handler.data[0]

        backend = self._constrained_backend

        # One unit for everything the kernel adds up: the search raster is
        # the objective times this scale, so the turn and tower tables are
        # multiplied by it too (_search_tables, _build_tower_cost_raster),
        # and the report divides by it. 1.0 without an objective. The
        # weighted tables are derived from the CURRENT objective on every
        # run, so a set_objective(), the ensemble's restore or an assigned
        # objective can never leave stale tables behind.
        self._apply_objective_weights()
        self._search_scale = self._objective_search_scale()
        if self._ground_unpriced():
            # No cell is priced: search on a ground cost of exactly 0
            # instead of the stack's uniform minimal weight of 1.
            raster = np.where(raster == IMPASSABLE_CELL_COST, raster,
                              0).astype(np.uint16)

        # Per-cell tower-cost raster (objective mode): tower foundation
        # costs come from the LAND USE (the stack's cost layer), not from
        # the combined feasibility values in the search raster.
        self._tower_cost_raster = self._build_tower_cost_raster()

        # Coupled approach: full extended-state Dijkstra
        route_result = self._find_route_coupled(
            raster, source_row, source_col, target_row, target_col,
            backend,
        )

        t_pathfinding = time.time() - t_start

        # Handle 3-tuple (variable height) or 2-tuple (fixed height) returns
        if len(route_result) == 3:
            path_indices, tower_cell_indices, tower_heights_arr = route_result
        else:
            path_indices, tower_cell_indices = route_result
            tower_heights_arr = None

        if len(path_indices) == 0 and self._ends.in_search:
            # a default run keeps its empty result; a run the new settings restricted names them as the reason
            raise NoPathFoundError(src, tgt, complete_message=self._no_route_message(src, tgt))

        result = self._build_constrained_path(
            path_indices, tower_cell_indices, raster, src, tgt,
            t_pathfinding, tower_heights_arr=tower_heights_arr,
        )

        # Honest per-metric reporting (Phase 6 evaluator) in objective mode
        if (self.objective is not None and self.metric_stack is not None
                and len(result.path_indices) > 1):
            idx = np.asarray(result.path_indices, dtype=np.int64)
            cols_n = raster.shape[1]
            self._evaluate_objective_metrics(idx // cols_n, idx % cols_n,
                                             result)

        # Store in parent's PathCollection so save_paths() works. Drop the
        # cached GeoDataFrame first, as PathFinder.find_route does: it was
        # built from the routes before this one.
        self.path_gdf = None
        self.paths.add(result)

        return result

    def _build_tower_cost_raster(self):
        """Per-cell tower foundation costs from the stack's cost layer.

        ``tower_cost[cell] = LUT[round(cost_layer[cell])]`` (the LUT is
        already "tower"-weight-scaled, and multiplied by the search scale so
        it is in the unit of the search raster); forbidden cells carry
        INFINITY. The cost-layer index of every cell is kept in
        ``_tower_site_index`` for the report. Without an objective (or
        without span constraints) returns None — the kernels then use the
        legacy value-keyed LUT, bit-identically.
        """
        self._tower_site_index = None
        if self.objective is None or self.metric_stack is None:
            return None
        if not self._profile.has_span_constraints:
            return None
        sub = self.metric_stack.window(self.raster_handler.window)
        cost_layer = sub["cost"]
        indices = np.clip(np.rint(cost_layer), 0, 65535).astype(np.int64)
        self._tower_site_index = indices
        tower = self._search_tables()[1][indices]
        tower[sub.forbidden_mask] = np.inf
        return tower

    def _prepare_dem_data(self, raster):
        """Extract and resample DEM data, build dem_kwargs.

        Returns:
            Tuple of (dem_data, dem_kwargs) where dem_data is the resampled
            DEM array (or None) and dem_kwargs is a dict of gradient params.
        """
        dem_data = None
        if hasattr(self, 'dem_raster_handler') and self.dem_raster_handler is not None:
            dem_raw = self.dem_raster_handler.data
            if len(dem_raw.shape) > 2:
                dem_data = dem_raw[0].astype(np.float32)
            else:
                dem_data = dem_raw.astype(np.float32)
            # Resample DEM to match cost raster shape (resolutions may differ)
            target_shape = (raster.shape[0], raster.shape[1])
            if dem_data.shape != target_shape:
                from scipy.ndimage import zoom
                zoom_factors = (target_shape[0] / dem_data.shape[0],
                                target_shape[1] / dem_data.shape[1])
                dem_data = zoom(dem_data, zoom_factors, order=1).astype(
                    np.float32)

        # Gradient parameters from profile
        dem_kwargs = {}
        if dem_data is not None:
            dem_kwargs['dem_data'] = dem_data
            dem_kwargs['cell_size'] = self._cell_size
            dem_kwargs['max_gradient_pct'] = (
                self._profile.max_gradient_percent
                if self._profile.max_gradient_percent is not None
                else 100.0
            )
            dem_kwargs['gradient_scale'] = (
                self._profile.gradient_cost_params.get('scale', 2.0)
                if self._profile.gradient_cost_params is not None
                else 2.0
            )

        return dem_data, dem_kwargs

    def _build_height_arrays(self):
        """Build heights and premiums arrays from profile.

        Returns:
            Tuple of (heights, premiums) as float32 numpy arrays.
        """
        if self._profile.has_variable_height:
            heights = np.array(
                self._profile.effective_tower_heights_m,
                dtype=np.float32)
            premiums = self._profile.precompute_height_premium()
        else:
            h = self._profile.tower_height_m or 25.0
            heights = np.array([h], dtype=np.float32)
            premiums = np.array([0.0], dtype=np.float32)
        return heights, premiums

    def _build_gpu_kwargs(self, raster, source_row, source_col,
                          target_row, target_col, n_span_bins,
                          span_bin_size, min_span, max_span,
                          heights, premiums):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        """Build the common gpu_kwargs dict shared by GPU backends.

        Returns:
            Dict of keyword arguments for GPU constrained SSSP functions.
        """
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        return {
            "raster": raster,
            "source_row": source_row, "source_col": source_col,
            "target_row": target_row, "target_col": target_col,
            "steps": self.steps,
            "angle_cost_lut": self._angle_cost_lut.astype(np.float32),
            "angle_valid_lut": self._angle_valid_lut.astype(np.uint8),
            "step_distances": self._step_distances.astype(np.float32),
            "tower_terrain_costs": self._tower_terrain_costs.astype(
                np.float32),
            "tower_angle_costs": self._tower_angle_costs.astype(
                np.float32),
            "n_span_bins": n_span_bins, "span_bin_size": span_bin_size,
            "min_span": min_span, "max_span": max_span,
            "tower_heights": heights,
            "height_premiums": premiums,
            "n_heights": len(heights),
        }

    def _add_dem_to_kwargs(self, gpu_kwargs, dem_data, dem_kwargs):
        """Add DEM/clearance/gradient/obstacle params to gpu_kwargs.

        Mutates and returns gpu_kwargs.
        """
        if dem_data is not None:
            gpu_kwargs.update(
                dem=dem_data,
                cell_size=self._cell_size,
                conductor_weight_per_m=(
                    self._profile.conductor_weight_per_m or 0.0),
                conductor_tension=(
                    self._profile.conductor_tension_n or 1.0),
                min_clearance=(
                    self._profile.min_clearance_m or 0.0),
                max_gradient_pct=dem_kwargs.get(
                    'max_gradient_pct', 100.0),
                gradient_scale=dem_kwargs.get(
                    'gradient_scale', 2.0),
            )
            if self._obstacle_data is not None:
                gpu_kwargs['obstacle_heights'] = self._obstacle_data
        return gpu_kwargs

    def _add_area_to_kwargs(self, gpu_kwargs):
        """Add area_offsets if present to gpu_kwargs.

        Mutates and returns gpu_kwargs.
        """
        if self._area_offsets is not None:
            gpu_kwargs.update(
                area_offsets=self._area_offsets,
                area_offset_starts=self._area_offset_starts,
                area_offset_counts=self._area_offset_counts,
            )
        return gpu_kwargs

    def _run_gpu_backend(self, gpu_func, raster, source_row, source_col,
                         target_row, target_col, n_span_bins, span_bin_size,
                         min_span, max_span, dem_data, dem_kwargs,
                         extra_kwargs=None):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        """Run a GPU backend with standard kwargs setup.

        Returns:
            Result from the GPU function call.
        """
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        heights, premiums = self._build_height_arrays()
        gpu_kwargs = self._build_gpu_kwargs(
            raster, source_row, source_col, target_row, target_col,
            n_span_bins, span_bin_size, min_span, max_span,
            heights, premiums,
        )
        if extra_kwargs:
            gpu_kwargs.update(extra_kwargs)
        self._add_dem_to_kwargs(gpu_kwargs, dem_data, dem_kwargs)
        self._add_area_to_kwargs(gpu_kwargs)
        return gpu_func(**gpu_kwargs)

    def _run_cython_backend(self, raster, source_row, source_col,
                            target_row, target_col, n_span_bins,
                            span_bin_size, min_span, max_span,
                            dem_data, dem_kwargs, backend):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        """Run the Cython backend for constrained routing.

        Dispatches to clearance-aware or basic algorithm depending on
        whether DEM data and clearance constraints are present.

        Returns:
            Result tuple from the Cython algorithm.
        """
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        from pyorps.utils.constrained_path_algorithms import (
            constrained_delta_stepping_2d,
            constrained_delta_stepping_height_2d,
            constrained_delta_stepping_lazy,
            constrained_dijkstra_2d,
        )

        # Helper: build clearance kwargs and dispatch to dense or lazy.
        # Uses constrained_delta_stepping_height_2d (which auto-selects
        # dense vs sparse based on state space size) for all clearance
        # calls, wrapping single-height into 1-element arrays.
        # Area cost offset arrays (None when not using "exact" mode)
        area_kwargs = {}
        if self._area_offsets is not None:
            area_kwargs = {
                "area_offsets": self._area_offsets,
                "area_offset_starts": self._area_offset_starts,
                "area_offset_counts": self._area_offset_counts,
            }

        # The line-end settings of this run, passed only when they are set: a default run calls the kernels exactly
        # as before. The 2-D kernels take the shortest last span, the clearance kernels the connection heights too.
        ends = self._ends
        end_kwargs = {"target_min_span": ends.target_min_span} if ends.target_min_span > 0.0 else {}
        height_kwargs = dict(end_kwargs)
        if ends.source_attach is not None:
            height_kwargs["source_attach_height"] = ends.source_attach
        if ends.target_attach is not None:
            height_kwargs["target_attach_height"] = ends.target_attach

        def _run_clearance(height_arr, premium_arr, obs_data):
            kwargs = {
                "raster": raster,
                "source_row": source_row, "source_col": source_col,
                "target_row": target_row, "target_col": target_col,
                "steps": self.steps,
                "angle_cost_lut": self._angle_cost_lut.astype(np.float32),
                "angle_valid_lut": self._angle_valid_lut.astype(np.uint8),
                "step_distances": self._step_distances.astype(np.float32),
                "tower_terrain_costs": self._tower_terrain_costs.astype(
                    np.float32),
                "tower_angle_costs": self._tower_angle_costs.astype(np.float32),
                "n_span_bins": n_span_bins, "span_bin_size": span_bin_size,
                "min_span": min_span, "max_span": max_span,
                "dem_data": dem_data,
                "cell_size": self._cell_size,
                "tower_heights": height_arr,
                "height_premiums": premium_arr,
                "conductor_weight_per_m": self._profile.conductor_weight_per_m,
                "conductor_tension": self._profile.conductor_tension_n,
                "min_clearance_val": self._profile.min_clearance_m,
                "max_gradient_pct": dem_kwargs['max_gradient_pct'],
                "gradient_scale": dem_kwargs['gradient_scale'],
                **area_kwargs,
                **height_kwargs,
            }
            if obs_data is not None:
                kwargs['obstacle_heights'] = obs_data

            n_h = len(height_arr)
            total_states = (raster.shape[0] * raster.shape[1]
                            * len(self.steps) * n_span_bins * n_h)
            if total_states > _KERNEL_DENSE_MAX_STATES:
                return self._call_kernel(constrained_delta_stepping_lazy, **kwargs)
            return self._call_kernel(constrained_delta_stepping_height_2d, **kwargs)

        # Variable-height clearance. A default run first searches with the
        # lowest height alone and returns that route if one exists; only when
        # it finds none does it run the full variable-height search. That
        # shortcut is NOT optimal: a route with some taller towers (at their
        # premium) can be cheaper than the lowest-height route, because a
        # taller tower lets a span clear an obstacle and saves towers. It is
        # kept for default runs (a known, separately tracked limitation, and
        # the result says so in search_notes). A run with a line-end setting
        # in the search ("enforce" or a connection height the search uses)
        # must find the optimum under that setting, so it skips the shortcut.
        if (dem_data is not None and self._profile.has_clearance_constraints
                and self._profile.has_variable_height):
            base_height = min(self._profile.effective_tower_heights_m)
            if not ends.in_search:
                fixed_result = _run_clearance(
                    np.array([base_height], dtype=np.float32),
                    np.array([0.0], dtype=np.float32),
                    self._obstacle_data,
                )
                if len(fixed_result[0]) > 0:
                    ends.base_height_shortcut = float(base_height)
                    return fixed_result

            # Run the variable-height search
            heights = np.array(
                self._profile.effective_tower_heights_m, dtype=np.float32)
            premiums = self._profile.precompute_height_premium()
            return _run_clearance(heights, premiums, self._obstacle_data)

        # Fixed-height clearance algorithm
        if (dem_data is not None and self._profile.has_clearance_constraints):
            effective_height = self._profile.tower_height_m
            if (self._profile.tower_heights_m is not None and
                    len(self._profile.tower_heights_m) == 1):
                effective_height = self._profile.tower_heights_m[0]
            return _run_clearance(
                np.array([effective_height], dtype=np.float32),
                np.array([0.0], dtype=np.float32),
                self._obstacle_data,
            )

        algo_func = (constrained_delta_stepping_2d if backend == "cython_parallel"
                     else constrained_dijkstra_2d)

        tower_kwargs = {}
        if (self._tower_cost_raster is not None
                and backend != "cython_parallel"):
            tower_kwargs["tower_cost_raster"] = self._tower_cost_raster

        # In the unit of the search raster (identical to the plain float32
        # casts when the search scale is 1, i.e. always without objective).
        angle_lut, tower_terrain, tower_angle = self._search_tables()
        # The kernels price the terrain per METRE only when they get the
        # cell size (cost x cost factor x cell_size per step); without it
        # they price it per cell, while the tower and turn tables are per
        # tower and the report (_terrain_eur) is per metre. The DEM
        # arguments carry it already; without a DEM it is passed here, and
        # the result is the one a flat DEM gives (its gradient factors are
        # exactly 1). At 1 m this is the kernels' default: nothing changes.
        dem_kwargs = {"cell_size": self._cell_size, **dem_kwargs}
        return self._call_kernel(
            algo_func,
            raster=raster,
            source_row=source_row, source_col=source_col,
            target_row=target_row, target_col=target_col,
            steps=self.steps,
            angle_cost_lut=angle_lut,
            angle_valid_lut=self._angle_valid_lut.astype(np.uint8),
            step_distances=self._step_distances.astype(np.float32),
            tower_terrain_costs=tower_terrain,
            tower_angle_costs=tower_angle,
            n_span_bins=n_span_bins, span_bin_size=span_bin_size,
            min_span=min_span, max_span=max_span,
            **dem_kwargs,
            **area_kwargs,
            **tower_kwargs,
            **end_kwargs,
        )
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite

    def _call_kernel(self, kernel, **kwargs):
        """Call a CPU search kernel; a refusal of the span bins becomes the project's ``InvalidInputError``.

        The kernels check at entry that the span bins are usable (they cover the longest span and, with a line-end
        option on, keep the single-precision spans of one direction in distinct bins) and raise ``ValueError``
        otherwise. That text goes to the caller with the numbers of this run.

        Raises:
            InvalidInputError: The kernel refused the span bins. Any other error of the kernel is passed on.
        """
        try:
            return kernel(**kwargs)
        except TypeError as err:
            named = [name for name in _LINE_END_KERNEL_ARGUMENTS if name in str(err)]
            if isinstance(err, InvalidTypeError) or not named:
                raise
            raise InvalidTypeError(
                f"The search kernel refused {', '.join(named)} of this run ({err}). The finder validates the "
                f"connection heights (a number above 0, not a bool) and takes terminal_span_rule='enforce' from the "
                f"profile's min_span_m, so check that value.") from err
        except ValueError as err:
            if isinstance(err, InvalidInputError) or "bin" not in str(err).lower():
                raise
            raise self._span_bins_error(err) from err

    def _span_bins_error(self, err):
        """The ``InvalidInputError`` for a kernel's refusal ``err`` of this run's span bins, with the numbers."""
        profile, ends = self._profile, self._ends
        shortest = float(np.min(self._step_distances))
        bin_size = ends.span_bin_size if ends.span_bin_size is not None else self._effective_span_bin_size
        longest = (f"the longest span, max_span_m {profile.max_span_m:g} m, is "
                   f"{max(1, math.ceil(profile.max_span_m / shortest))} steps of the shortest step {shortest:g} m"
                   if profile.has_span_constraints else "the profile has no span limits")
        return InvalidInputError(
            f"The search kernel refused the span bins of this run ({str(err).rstrip('. ')}). The run used "
            f"{ends.n_span_bins} span bins of "
            f"{bin_size:.4g} m; {longest}. The exact bins of a run with terminal_span_rule='enforce' or a connection "
            f"height are distinct up to ConstrainedPathFinder.EXACT_BINS_MAX_STEPS = {self.EXACT_BINS_MAX_STEPS} "
            f"steps per longest span; with more steps, or with the coarse bins of a default run, the kernels refuse "
            f"a line-end option. Use a coarser raster or a shorter max_span_m, or terminal_span_rule 'keep' or "
            f"'remove_if_valid' without connection heights (they keep the coarse bins).")

    def _find_route_coupled(self, raster, source_row, source_col,
                            target_row, target_col, backend):
        """Full coupled Dijkstra with span bins in the state."""
        n_span_bins = self._effective_n_span_bins
        span_bin_size = self._effective_span_bin_size
        min_span = (self._profile.min_span_m
                    if self._profile.has_span_constraints else 0.0)
        max_span = (self._profile.max_span_m
                    if self._profile.has_span_constraints else 1e6)

        # Extract DEM data if available (aligned via parent's dem_raster_handler)
        dem_data, dem_kwargs = self._prepare_dem_data(raster)

        # What the end-span rule and the connection heights ask of the search (fills self._ends)
        self._resolve_line_ends(dem_data)

        # The one place that decides whether a GPU solver may run: refuse
        # before any GPU code is imported when the profile/data set a
        # constraint the chosen solver would ignore.
        if backend in self.GPU_BACKENDS:
            self._check_gpu_backend_settings(
                backend, n_span_bins, span_bin_size, max_span,
                dem_data, dem_kwargs)
        elif self._ends.in_search:
            # exact bins: the new settings must never hide a feasible route (CPU kernels only)
            n_span_bins, span_bin_size = self._exact_span_bins(raster, dem_data)
            self._ends.n_span_bins, self._ends.span_bin_size = n_span_bins, span_bin_size
            self._ends.exact_bins = True

        if backend == "raster_gpu":
            try:
                from pyorps.utils.constrained_sssp_gpu_v2 import (
                    constrained_sssp_raster_gpu_v2,
                )
                result = self._run_gpu_backend(
                    constrained_sssp_raster_gpu_v2,
                    raster, source_row, source_col, target_row, target_col,
                    n_span_bins, span_bin_size, min_span, max_span,
                    dem_data, dem_kwargs, extra_kwargs={'sparse': 'auto'},
                )
                self._warn_gpu_cell_size(backend)
                return result
            except (ImportError, RuntimeError) as err:
                self._fall_back_to_cython("GPU v2", err, raster)
                backend = "cython"

        if backend == "raster_gpu_v4":
            try:
                from pyorps.utils.constrained_sssp_gpu_v4 import (
                    constrained_sssp_raster_gpu_v4,
                )
                result = self._run_gpu_backend(
                    constrained_sssp_raster_gpu_v4,
                    raster, source_row, source_col, target_row, target_col,
                    n_span_bins, span_bin_size, min_span, max_span,
                    dem_data, dem_kwargs,
                    extra_kwargs={'max_visited_fraction': 0.4},
                )
                self._warn_gpu_cell_size(backend)
                return result
            except (ImportError, RuntimeError) as err:
                self._fall_back_to_cython("GPU v4", err, raster)
                backend = "cython"

        return self._run_cython_backend(
            raster, source_row, source_col, target_row, target_col,
            n_span_bins, span_bin_size, min_span, max_span,
            dem_data, dem_kwargs, backend,
        )

    def _resolve_line_ends(self, dem_data):
        """Decide what the end-span rule and the connection heights ask of this run's search (fills ``self._ends``).

        - ``terminal_span_rule="enforce"`` (with span limits): ``target_min_span`` = ``min_span_m``.
        - A connection height reaches the search only with the clearance check on: a DEM and a profile with tower
          height, conductor weight, tension, minimum clearance and span limits. Without it the height cannot act on
          the route; one ``PyorpsWarning`` says so (the height is still the reported end height).
        - With the clearance check on, a connection height below the profile's ``min_clearance_m`` is refused: the
          conductor cannot be higher above the ground at its own end point than the structure it hangs from, so no
          span from there can clear.

        Returns:
            True when the clearance check is on.

        Raises:
            InvalidInputError: A connection height the search would use is below ``min_clearance_m``.
        """
        ends, profile = self._ends, self._profile
        has_spans = profile.has_span_constraints
        clearance = dem_data is not None and profile.has_clearance_constraints and has_spans
        if has_spans and profile.terminal_span_rule == "enforce":
            ends.target_min_span = float(profile.min_span_m)
        if clearance:
            for name, value in (("source_connection_height_m", ends.source_height_m),
                                ("target_connection_height_m", ends.target_height_m)):
                if value is not None and value < float(profile.min_clearance_m):
                    raise InvalidInputError(
                        f"{name} {value:g} m is below the profile's min_clearance_m {profile.min_clearance_m:g} m: "
                        f"the conductor is no higher above the ground at the end point than the structure it hangs "
                        f"from, so a line from there cannot clear the minimum clearance at its own end cell. Enter a "
                        f"height of at least {profile.min_clearance_m:g} m, or leave {name} out.")
            ends.source_attach, ends.target_attach = ends.source_height_m, ends.target_height_m
        elif ends.source_height_m is not None or ends.target_height_m is not None:
            lacks = []
            if dem_data is None:
                lacks.append("there is no DEM")
            if not profile.has_clearance_constraints:
                lacks.append("the profile lacks tower height, conductor weight, conductor tension or minimum "
                             "clearance")
            if not has_spans:
                lacks.append("the profile has no span limits")
            named = ", ".join(f"{name}={value:g} m" for name, value in (
                ("source_connection_height_m", ends.source_height_m),
                ("target_connection_height_m", ends.target_height_m)) if value is not None)
            self._warn_once(
                ("unused_connection_height", ends.source_height_m, ends.target_height_m),
                f"{named} has no effect on the route: the search checks the conductor clearance of a span only with "
                f"a DEM and a profile that has tower height, conductor weight, conductor tension, minimum clearance "
                f"and span limits, and here {' and '.join(lacks)}. The height is still reported as the height of "
                f"the end tower.")
        return clearance

    def _exact_span_bins(self, raster, dem_data):
        """Span bins that tell every span apart, for a run with the new line-end settings.

        The search merges the states of one cell and direction that share a span bin, so coarse bins can drop the
        state a later rule needs (the finder's default bins, ``min_span_m`` wide, are coarse). Within one cell and
        direction the spans are whole multiples of that direction's step, which differ by at least the shortest
        step; a bin just under the shortest step therefore holds one span value. The bin is
        ``shortest_step * K / (K + 0.25)`` with ``K = ceil(max_span_m / shortest_step)`` (the span ``k`` steps long
        falls in bin ``k`` for every ``k <= K``, whatever the rounding of the running sum), and there are
        ``K + 1`` bins.

        The bins are exact only while the kernels' single-precision sums keep every k-step span in its own bin:
        above ``EXACT_BINS_MAX_STEPS`` steps per longest span the run is refused (the class attribute documents the
        bound), and the kernels' own check of the bins (``check_exact_span_bins``) is run here too, so that its
        refusal reaches the caller as the finder's error before any kernel call. The run is also refused when the
        state space is above ``EXACT_BINS_MAX_STATES`` and when the memory of the kernel the run would call
        (``exact_bins_memory``) is above ``EXACT_BINS_MAX_BYTES``.

        Returns:
            ``(n_span_bins, span_bin_size)``.

        Raises:
            InvalidInputError: The bins do not fit in the kernels' 16-bit bin index, the longest span is more than
                ``EXACT_BINS_MAX_STEPS`` steps long, the kernels' bin check refuses the bins, the state space is
                larger than ``EXACT_BINS_MAX_STATES`` or the estimated memory is larger than
                ``EXACT_BINS_MAX_BYTES``; the message gives the numbers and what to change.
        """
        profile = self._profile
        shortest = float(np.min(self._step_distances))
        steps_per_span = max(1, math.ceil(profile.max_span_m / shortest))
        bin_size = shortest * steps_per_span / (steps_per_span + 0.25)
        n_bins = steps_per_span + 1
        clearance = dem_data is not None and profile.has_clearance_constraints
        n_heights = 1
        if clearance and profile.has_variable_height:
            n_heights = len(profile.effective_tower_heights_m)
        cells = int(raster.shape[0]) * int(raster.shape[1])
        n_dirs = len(self.steps)
        states = cells * n_dirs * n_bins * n_heights
        reasons = []
        if self._ends.target_min_span > 0.0:
            reasons.append("terminal_span_rule='enforce'")
        if self._ends.source_attach is not None or self._ends.target_attach is not None:
            reasons.append("a connection height")
        why = " and ".join(reasons) or "this run"
        advice = ("Use a coarser raster (the bins are as short as the shortest step), a smaller search window "
                  "(search_space_buffer_m), a shorter max_span_m, or terminal_span_rule 'keep' or 'remove_if_valid' "
                  "without connection heights (they keep the coarse bins). ")
        if n_bins > self.EXACT_BINS_MAX_COUNT:
            raise InvalidInputError(
                f"{why} needs exact span bins, but max_span_m {profile.max_span_m:g} m is {steps_per_span} steps "
                f"of the shortest step {shortest:g} m, so {n_bins} span bins of {bin_size:.4g} m are needed and the "
                f"search holds at most {self.EXACT_BINS_MAX_COUNT}. " + advice)
        if steps_per_span > self.EXACT_BINS_MAX_STEPS:
            raise InvalidInputError(
                f"{why} needs exact span bins, but max_span_m {profile.max_span_m:g} m is {steps_per_span} steps "
                f"of the shortest step {shortest:g} m, and the bins stay exact only up to "
                f"ConstrainedPathFinder.EXACT_BINS_MAX_STEPS = {self.EXACT_BINS_MAX_STEPS} steps per longest span: "
                f"the kernels add a span up in single precision, and above that two neighbouring spans can fall "
                f"into one bin of {bin_size:.4g} m again (the kernels refuse bins that are not distinct). "
                + advice)
        self._check_bins_distinct(why, n_bins, bin_size, steps_per_span, shortest, advice)
        if states > self.EXACT_BINS_MAX_STATES:
            raise InvalidInputError(
                f"{why} needs exact span bins, which make the search state space {states:,} states "
                f"({raster.shape[0]} x {raster.shape[1]} cells x {n_dirs} directions x {n_bins} span bins "
                f"of {bin_size:.4g} m x {n_heights} height classes), above "
                f"ConstrainedPathFinder.EXACT_BINS_MAX_STATES = {self.EXACT_BINS_MAX_STATES:,}. " + advice +
                "Or raise ConstrainedPathFinder.EXACT_BINS_MAX_STATES if the machine has the memory.")
        memory = self.exact_bins_memory(
            cells, n_dirs, n_bins, n_heights, backend=self._constrained_backend, clearance=clearance,
            has_dem=dem_data is not None)
        if memory.nbytes > self.EXACT_BINS_MAX_BYTES:
            gib = 1024 ** 3
            raise InvalidInputError(
                f"{why} needs exact span bins, and the search kernel {memory.kernel} would then need about "
                f"{memory.nbytes / gib:.1f} GiB of memory, above ConstrainedPathFinder.EXACT_BINS_MAX_BYTES = "
                f"{self.EXACT_BINS_MAX_BYTES / gib:.1f} GiB. The search state space is {states:,} states "
                f"({raster.shape[0]} x {raster.shape[1]} cells x {n_dirs} directions x {n_bins} span bins "
                f"of {bin_size:.4g} m x {n_heights} height classes); {memory.detail} " + advice +
                "Or raise ConstrainedPathFinder.EXACT_BINS_MAX_BYTES (and EXACT_BINS_MAX_STATES) if the machine has "
                "the memory.")
        return n_bins, bin_size

    def _check_bins_distinct(self, why, n_bins, bin_size, steps_per_span, shortest, advice):
        """Run the kernels' own span-bin check (``check_exact_span_bins``) on this run's bins, before any kernel call.

        The check simulates, in the kernels' float32 arithmetic and for every direction, the spans of 1 to K steps
        and wants strictly increasing bins below ``n_bins``. Its refusal (a ``ValueError``) becomes an
        ``InvalidInputError`` with the numbers of this run.

        Raises:
            InvalidInputError: Two span lengths of one direction would share a bin.
        """
        from pyorps.utils._constrained_context import check_exact_span_bins

        try:
            check_exact_span_bins(self._step_distances.astype(np.float32), bin_size, self._profile.max_span_m, n_bins)
        except ValueError as err:
            raise InvalidInputError(
                f"{why} needs exact span bins, but the search kernels would refuse the {n_bins} span bins of "
                f"{bin_size:.4g} m for max_span_m {self._profile.max_span_m:g} m ({steps_per_span} steps of the "
                f"shortest step {shortest:g} m): {str(err).rstrip('. ')}. " + advice) from err

    @staticmethod
    def exact_bins_memory(cells, n_dirs, n_span_bins, n_heights=1, *, backend="cython", clearance=False,
                          has_dem=False):
        """Memory the CPU kernel of a run with exact span bins allocates, estimated before the run.

        The state space is ``cells x n_dirs x n_span_bins x n_heights`` states. What a state costs depends on the
        kernel the finder calls for the run (mirror of ``_run_cython_backend``):

        - a run with the clearance check on (``clearance``: a DEM and the profile's clearance data) uses
          ``constrained_delta_stepping_height_2d``, whatever the backend. It is dense up to 5e8 states (24 bytes a
          state) and above that the finder hands the search to ``constrained_delta_stepping_lazy``, a hash map
          (about 64 bytes per REACHED state);
        - without it, ``backend="cython_parallel"`` uses ``constrained_delta_stepping_2d``, which is ALWAYS dense
          (24 bytes a state, however many states there are);
        - ``backend="cython"`` uses ``constrained_dijkstra_2d``: dense (24 bytes a state) up to 5e8 states, sparse
          above (four hash containers and a heap, about 200 bytes per REACHED state).

        Every kernel also builds per-(cell, direction) caches: 5 bytes (intermediate-cell status and cost), and 4
        more when it gets a DEM (``has_dem``; always with the clearance kernels). The queue of the delta-stepping
        and Dijkstra searches (8 bytes an entry, a few entries per reached state) is not counted: it grows with the
        states the search reaches and is small next to the state array.

        The sparse forms need memory in proportion to the number of states the search REACHES, which cannot be
        known before the search. The estimate takes every state as reached (the worst case), so it is an upper
        bound; a search that stays near its route reaches far fewer states. A run is refused on this bound, not on
        a guess: raise ``EXACT_BINS_MAX_BYTES`` if the machine has the memory.

        Returns:
            An :class:`ExactBinsMemory`.
        """
        cells, n_dirs, n_span_bins, n_heights = int(cells), int(n_dirs), int(n_span_bins), int(n_heights)
        states = cells * n_dirs * n_span_bins * n_heights
        if clearance:
            lazy = states > _KERNEL_DENSE_MAX_STATES
            kernel = "constrained_delta_stepping_lazy" if lazy else "constrained_delta_stepping_height_2d"
            per_state = _LAZY_BYTES_PER_STATE if lazy else _DENSE_BYTES_PER_STATE
        elif backend == "cython_parallel":
            lazy = False
            kernel, per_state = "constrained_delta_stepping_2d", _DENSE_BYTES_PER_STATE
        else:
            lazy = states > _KERNEL_DENSE_MAX_STATES
            kernel = "constrained_dijkstra_2d"
            per_state = _DIJKSTRA_SPARSE_BYTES_PER_STATE if lazy else _DENSE_BYTES_PER_STATE
        cache = _CACHE_BYTES_PER_CELL_DIRECTION + (_GRADIENT_BYTES_PER_CELL_DIRECTION if (clearance or has_dem) else 0)
        nbytes = states * per_state + cells * n_dirs * cache
        if lazy:
            detail = (f"above {_KERNEL_DENSE_MAX_STATES:,} states the kernel keeps its states in hash maps that grow "
                      f"with the number of states the search reaches (about {per_state} bytes each); that number is "
                      f"not known before the search, so the estimate takes every state as reached (the worst case).")
        else:
            detail = (f"the kernel allocates {per_state} bytes for every state of its dense state array, plus "
                      f"{cache} bytes per cell and direction for its caches.")
        return ExactBinsMemory(kernel=kernel, form="sparse" if lazy else "dense", states=states, nbytes=int(nbytes),
                               detail=detail)

    def _no_route_message(self, source, target):
        """The complete message of the ``NoPathFoundError`` of a search that the line-end settings restricted.

        It does not start with the base class's hint to increase the buffer: a bigger search window does not help
        when the settings are what excludes every route.
        """
        ends, profile = self._ends, self._profile
        parts = []
        if ends.target_min_span > 0.0:
            parts.append(f"terminal_span_rule='enforce' (every end span at least min_span_m {profile.min_span_m:g} "
                         f"m long, a line without a tower excepted)")
        heights = [f"{name} {value:g} m" for name, value, used in (
            ("source_connection_height_m", ends.source_height_m, ends.source_attach is not None),
            ("target_connection_height_m", ends.target_height_m, ends.target_attach is not None)) if used]
        if heights:
            parts.append("the end span clearance from " + " and ".join(heights))
        return (f"No route from {source} to {target} satisfies the line-end settings of this run: the search was run "
                f"with {' and '.join(parts)} (exact span bins of {ends.span_bin_size:.4g} m), and no route meets "
                f"them. Move the end point, allow shorter end spans (terminal_span_rule 'keep' or "
                f"'remove_if_valid'), or change the connection height. If the same call fails without these "
                f"settings too, no route exists between the points at all (check the forbidden cells and the "
                f"search window, search_space_buffer_m).")

    def _warn_gpu_cell_size(self, backend):
        """Warn once that a constrained GPU solver prices the terrain per cell, not per metre, on a raster whose
        cell size is not 1: the CPU backends price it per metre at every cell size."""
        cell = self._cell_size
        if abs(cell - 1.0) <= 1e-9:
            return
        self._warn_once(
            ("gpu_cell_size", backend),
            f"The raster cell size is {cell:g}, not 1. The constrained GPU solver {backend!r} still prices the "
            f"terrain per cell (its kernels take no cell size for the terrain term), while the CPU backends "
            f"('cython', 'cython_parallel') and the report price it per metre. On this raster the GPU solver "
            f"weighs the terrain against the tower and turn costs differently from the report, so its route can "
            f"differ from the CPU route. Use a CPU backend for cell sizes other than 1.")

    def _check_gpu_backend_settings(self, backend, n_span_bins, span_bin_size,
                                    max_span, dem_data, dem_kwargs):
        """Raise when the GPU solver of ``backend`` would ignore a constraint.

        Looks at exactly the keyword arguments :meth:`_run_gpu_backend`
        hands to the solver (same helpers), so the check cannot drift from
        the call. The line-end settings (``target_min_span`` of
        ``terminal_span_rule="enforce"``, the connection heights the search
        would use) are checked too: no GPU solver has them.

        Raises:
            AlgorithmNotImplementedError: Names the settings, the solver
                and the backends that enforce them.
        """
        settings = {"n_span_bins": n_span_bins, "span_bin_size": span_bin_size,
                    "max_span": max_span,
                    "target_min_span": self._ends.target_min_span,
                    "source_attach_height": self._ends.source_attach,
                    "target_attach_height": self._ends.target_attach}
        self._add_dem_to_kwargs(settings, dem_data, dem_kwargs)
        self._add_area_to_kwargs(settings)
        check_constrained_gpu_settings(
            backend,
            **{name: settings[name]
               for name in CONSTRAINED_GPU_CHECKED_SETTINGS if name in settings})

    @staticmethod
    def _fall_back_to_cython(label, err, raster):
        """Warn that a constrained GPU solver could not run and Cython takes over.

        Called for an ``ImportError`` (no CuPy) or a ``RuntimeError`` (the
        solver's "CUDA GPU not available", or a CUDA runtime/driver error of
        CuPy). Every other exception of the solver propagates to the caller.

        Raises:
            MemoryError: ``err`` is an out-of-memory condition. It is never
                reported as "unavailable" and there is no silent fallback.
        """
        if is_gpu_out_of_memory(err):
            raise gpu_out_of_memory_error(
                err, f"running the constrained {label} solver",
                shape=raster.shape, nbytes=raster.nbytes) from err
        warnings.warn(
            f"{label} unavailable ({describe_gpu_error(err)}), falling back to Cython",
            UserWarning, stacklevel=4)

    @staticmethod
    def _tower_positions(path_arr, tower_cells, ncols):
        """The position along the route of each tower cell the kernel returned.

        The kernels name a tower by its CELL, in route order, and a route that crosses itself passes a cell twice
        (and can carry a tower at each passage, or at one passage only). A tower stands at every direction change
        of the route (the search forces it), so the entries are matched to positions that increase and leave no
        direction change without a tower: for each entry only the passages of its cell between the previous tower
        and the first direction change after it come into question, and a search over these choices (backtracking,
        in route order) finds the assignment that covers every direction change. Without a crossing every
        cell has one passage and nothing is searched. Everything after this works on positions, never on cells.

        An entry whose cell the route does not leave through (such as the end cell) gets None. If no assignment
        covers every direction change (output that did not come from a search) or the search would take too long,
        each entry gets the first passage after the previous entry's position. Position 0 can match (a tower on
        the start cell); the last cell cannot.

        Returns:
            A list with one position (or None) per entry of ``tower_cells``.
        """
        entries = [int(c) for c in tower_cells]
        n_cells = len(path_arr)
        if n_cells < 2 or not entries:
            return [None] * len(entries)
        rows, cols = path_arr // ncols, path_arr % ncols
        d_rows, d_cols = np.diff(rows), np.diff(cols)
        turns = (np.flatnonzero((d_rows[1:] != d_rows[:-1]) | (d_cols[1:] != d_cols[:-1])) + 1).tolist()
        passages = {}
        for position, cell in enumerate(path_arr[:-1].tolist()):
            passages.setdefault(cell, []).append(position)

        def options(k, previous):
            found = passages.get(entries[k])
            if found is None:
                return [None]                                   # off the route: the entry is skipped
            at = bisect_right(turns, previous)
            limit = turns[at] if at < len(turns) else n_cells - 2
            return found[bisect_right(found, previous):bisect_right(found, limit)]

        n_entries = len(entries)
        chosen = [None] * n_entries
        last = [-1] * (n_entries + 1)                           # last[k]: the position of the last tower before k
        frames, k, steps = [], 0, 0
        while steps < _TOWER_MATCH_STEPS:
            steps += 1
            if k == n_entries:
                if bisect_right(turns, last[k]) == len(turns):
                    return chosen                               # every direction change has its tower
                k -= 1
                continue
            if len(frames) == k:
                frames.append([options(k, last[k]), 0])
            choices, index = frames[k]
            if index >= len(choices):
                frames.pop()
                k -= 1
                if k < 0:
                    break
                continue
            frames[k][1] = index + 1
            chosen[k] = choices[index]
            last[k + 1] = last[k] if choices[index] is None else choices[index]
            k += 1
        positions, previous = [], -1                            # no consistent assignment: the first passages
        for cell in entries:
            found = next((p for p in passages.get(cell, ()) if p > previous), None)
            positions.append(found)
            if found is not None:
                previous = found
        return positions

    @staticmethod
    def _close_end_towers(search_positions, path_arr, min_span_m, idx_to_coord):
        """The first and last search tower when they stand closer than ``min_span_m`` to an end point.

        ``search_positions`` are the positions (along the route) of the search towers between the two end points.
        Only the first one (measured from the start) and the last one (measured from the end) are looked at, by the
        straight distance of their cell centres.

        Returns:
            ``{position: [(end, distance_m), ...]}`` in the order source, target; ``end`` is ``"source"`` or
            ``"target"``. One tower can be close to both ends.
        """
        close = {}
        if not len(search_positions):
            return close
        for end, pos, end_pos in (("source", search_positions[0], 0),
                                  ("target", search_positions[-1], len(path_arr) - 1)):
            (x, y), (ex, ey) = idx_to_coord(path_arr[pos]), idx_to_coord(path_arr[end_pos])
            dist = math.sqrt((x - ex)**2 + (y - ey)**2)
            if dist < min_span_m:
                close.setdefault(pos, []).append((end, dist))
        return close

    def _apply_terminal_span_rule(self, path_arr, search_positions, min_span_m, idx_to_coord,
                                  ncols, tower_height_map=None, geometry=None):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        """Apply the profile's ``terminal_span_rule`` to the search towers next to the two end points.

        The search keeps the first span at least ``min_span_m`` long (a tower stands that far from the source, up
        to float rounding) but accepts the target at any span, so the last search tower can stand closer than
        ``min_span_m`` to the target. Only the first and the last search tower are looked at; what becomes of such
        a tower depends on the rule (``InfrastructureProfile.terminal_span_rule``):

        - ``"remove"``: it is removed (the behaviour of every earlier version). A merged end span that reaches
          ``max_span_m``, a removed tower at a bend and a merged span that fails the clearance check are recorded
          as issues to warn about.
        - ``"remove_if_valid"``: removed only when the tower stands on a straight stretch, the merged span stays
          below ``max_span_m`` and (clearance check on) clears the ground; otherwise kept and flagged.
        - ``"keep"``: kept and flagged.
        - ``"enforce"``: the search itself kept every end span at least ``min_span_m`` long (``target_min_span``;
          the first span already is), so no search tower stands closer than that to an end point and nothing is
          removed or flagged here. A tower that is close anyway (a line that has no tower except one next to both
          ends, which the search accepts as one short span) is kept like under ``"keep"``.

        Everything works on POSITIONS along the route (see ``_tower_positions``), so a route that crosses itself
        and has a tower at each passage of a cell is handled tower by tower.

        Parameters:
            path_arr: Cells of the route, start to end (int64 array).
            search_positions: Positions along the route of the kernel's towers, ascending (see
                ``_tower_positions``). The two end positions are ignored: they are the terminal towers.
            min_span_m: The profile's shortest span.
            idx_to_coord: Cell index -> (x, y) of the cell centre.
            ncols: Columns of the search raster (cell index -> row, column).
            tower_height_map: Position -> height the search chose (variable-height runs), else None.
            geometry: :class:`_EndSpanGeometry` of the clearance check, None when the search ran none.

        Returns:
            An :class:`_EndSpanOutcome`; ``positions`` holds the towers the report keeps, the two terminals included.
        """
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        rule = self._profile.terminal_span_rule
        tower_height_map = {} if tower_height_map is None else tower_height_map
        path_arr = np.asarray(path_arr, dtype=np.int64)
        n_cells = len(path_arr)
        outcome = _EndSpanOutcome()

        inner = sorted({int(p) for p in search_positions if 0 < int(p) < n_cells - 1})
        chain = [0] + inner + ([n_cells - 1] if n_cells > 1 else [])
        outcome.search_chain = list(chain)

        work = list(chain)
        removed = []
        for pos, ends in self._close_end_towers(inner, path_arr, min_span_m, idx_to_coord).items():
            if rule == "enforce" and min(d for _, d in ends) >= min_span_m * (1.0 - _ENFORCE_ROUNDING):
                continue            # a span of exactly min_span_m, short by the rounding of the distance only
            if pos not in work or pos in (0, n_cells - 1):
                continue
            problems = []
            if rule == "remove_if_valid":
                problems = self._merge_problems(path_arr, ncols, pos, work, idx_to_coord, tower_height_map, geometry)
            if rule in ("keep", "enforce") or problems:
                self._keep_end_tower(outcome, pos, ends, min_span_m, problems, rule)
                continue
            work.remove(pos)
            removed.append((pos, ends))

        if rule == "remove":
            # judged on the line that is left after both removals
            for pos, ends in removed:
                problems = self._merge_problems(path_arr, ncols, pos, work + [pos], idx_to_coord, tower_height_map,
                                                geometry)
                if problems:
                    outcome.issues.append((self._removed_tower_text(pos, path_arr, ends, min_span_m, problems,
                                                                    idx_to_coord), True))
                    outcome.failed_spans.update(span for code, _, span in problems if code == "clearance")
        for pos, ends in removed:
            outcome.removed.append(pos)
            outcome.removed_notes[pos] = (
                f"removed by terminal_span_rule={rule!r}: {self._end_distance_text(ends)}, "
                f"shorter than min_span_m {min_span_m:g} m")

        outcome.terminals = {0, n_cells - 1}
        outcome.positions = sorted(set(work) | outcome.terminals)
        return outcome

    @staticmethod
    def _end_distance_text(ends):
        """'2.00 m from the target' for the ends a close tower stands next to."""
        dist = min(d for _, d in ends)
        where = f"the {ends[0][0]}" if len(ends) == 1 else "both end points"
        return f"{dist:.2f} m from {where}"

    def _keep_end_tower(self, outcome, pos, ends, min_span_m, problems, rule):
        """Flag a kept line tower that stands closer than ``min_span_m`` to an end point.

        Under ``"enforce"`` the search never produces one: if it is there anyway (output that did not come from a
        search with ``target_min_span``), it is kept and a ``SpanRuleWarning`` is due.
        """
        text = (f"end span {min(d for _, d in ends):.2f} m is shorter than min_span_m {min_span_m:g} m; "
                f"kept by terminal_span_rule={rule!r}")
        if rule == "enforce":
            text += ", although the rule asks the search to keep every end span at least that long"
        if problems:
            text += " because removing the tower would leave " + " and ".join(p[1] for p in problems)
        outcome.kept_notes[pos] = text
        outcome.issues.append((text, rule == "enforce"))

    def _removed_tower_text(self, pos, path_arr, ends, min_span_m, problems, idx_to_coord):
        """One issue line for the tower at position ``pos`` that ``terminal_span_rule='remove'`` removed and that
        broke a rule."""
        x, y = idx_to_coord(path_arr[pos])
        return (f"terminal_span_rule='remove' removed the tower at ({x:.1f}, {y:.1f}), "
                f"{self._end_distance_text(ends)} (min_span_m {min_span_m:g} m); it leaves "
                + " and ".join(p[1] for p in problems))

    def _merge_problems(self, path_arr, ncols, pos, chain, idx_to_coord, tower_height_map, geometry):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        """What breaks when the line tower at path position ``pos`` is taken out of ``chain``.

        The towers next to ``pos`` in ``chain`` (path positions, start to end) become the ends of the merged span.

        The merged span's clearance is judged with each tower's own height (the kernels report it also with several
        tower heights) and the end heights the report states.

        Returns:
            List of ``(code, phrase, span)``: ``code`` is ``"bend"`` (the line bends at the tower), ``"span"`` (the
            merged span reaches ``max_span_m``) or ``"clearance"`` (it fails the clearance check; ``span`` is its
            ``(first, last)`` position); ``phrase`` finishes "... would leave".
        """
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        rows, cols = path_arr // ncols, path_arr % ncols
        ordered = sorted(chain)
        prev = max(q for q in ordered if q < pos)
        nxt = min(q for q in ordered if q > pos)
        problems = []
        bend = (rows[pos] - rows[pos - 1], cols[pos] - cols[pos - 1]) != (
            rows[pos + 1] - rows[pos], cols[pos + 1] - cols[pos])
        if bend:
            problems.append(("bend", "a bend without a tower", None))
        (ax, ay), (bx, by) = idx_to_coord(path_arr[prev]), idx_to_coord(path_arr[nxt])
        merged = math.sqrt((bx - ax)**2 + (by - ay)**2)
        max_span = self._profile.max_span_m
        label = f"from ({ax:.1f}, {ay:.1f}) to ({bx:.1f}, {by:.1f})"
        if merged >= max_span - _SPAN_REL_TOL * max_span:
            problems.append(("span", f"a merged span {label} of {merged:.2f} m, not below max_span_m {max_span:g} m",
                             None))
        if geometry is not None and not bend:
            result = self._python_span_clearance(geometry, prev, nxt, len(path_arr), tower_height_map)
            if result is not None and not result.ok:
                problems.append(("clearance", f"a merged span {label} that fails the clearance check "
                                 f"({self._clearance_detail(result, path_arr, ncols, idx_to_coord)})",
                                 (prev, nxt)))
        return problems

    def _python_span_clearance(self, geometry, pos_a, pos_b, n_cells, tower_height_map, heights=None, strict=False,
                               start_cell=False):
        """Python clearance check of the route between path positions ``pos_a`` and ``pos_b``.

        The attachment heights are the ones the report states for the towers at those positions (a terminal at
        position 0 or ``n_cells - 1``), unless ``heights`` gives them. ``strict`` and ``start_cell`` are passed on
        to ``path_span_clearance`` (False = the search's own test with its blind spots, True = every cell and every
        span, ``start_cell`` also the cell the span starts on). None when there is nothing to check: no height, or
        the stretch is not straight.
        """
        if heights is None:
            heights = tuple(
                self._resolve_tower_height(
                    pos in (0, n_cells - 1), pos, tower_height_map, self._end_of(pos, n_cells))
                for pos in (pos_a, pos_b))
        if heights[0] is None or heights[1] is None:
            return None
        return path_span_clearance(
            geometry.dem, geometry.obstacle, geometry.rows, geometry.cols, pos_a, pos_b, self._cell_size,
            heights[0], heights[1], conductor_weight_per_m=self._profile.conductor_weight_per_m,
            conductor_tension_n=self._profile.conductor_tension_n, min_clearance_m=self._profile.min_clearance_m,
            strict=strict, start_cell=start_cell)

    def _clearance_detail(self, result, path_arr, ncols, idx_to_coord):
        """'conductor 3.20 m above the ground at (x, y), min_clearance_m 7 m' for a failed check."""
        if result.worst_cell is None:
            return result.reason or "not clear"
        row, col = result.worst_cell
        x, y = idx_to_coord(row * ncols + col)
        return (f"the conductor is {result.worst_clearance_m:.2f} m above the ground at ({x:.1f}, {y:.1f}), "
                f"min_clearance_m is {self._profile.min_clearance_m:g} m")

    def _end_span_geometry(self, raster, path_indices):
        """Ground data for the Python clearance check of the end spans, or None.

        None when the search ran no clearance check (no DEM, or the profile lacks tower height, conductor weight,
        tension or clearance). With several tower heights the check uses each tower's own height: the kernels
        report it (they reported the previous tower's height until round 17).
        """
        if not self._profile.has_clearance_constraints or len(path_indices) < 2:
            return None
        dem_data, _ = self._prepare_dem_data(raster)
        if dem_data is None:
            return None
        path_arr = np.asarray(path_indices, dtype=np.int64)
        ncols = raster.shape[1]
        return _EndSpanGeometry(
            dem=dem_data, obstacle=self._obstacle_data, rows=path_arr // ncols, cols=path_arr % ncols)

    @staticmethod
    def _end_of(pos, n_cells):
        """``"source"`` for the first position of a route, ``"target"`` for the last, else None."""
        if pos == 0:
            return "source"
        return "target" if pos == n_cells - 1 else None

    def _check_span_clearance(self, towers, tower_positions, path_arr, geometry, outcome, idx_to_coord, ncols):
        """Check EVERY span of the reported line against the heights the report states, strictly.

        The strict check (``span_clearance(..., strict=True)``) samples every cell of a span, the first one after
        the tower too, and checks a span of one step; for the first span it also samples the cell the line starts
        on. The search does none of this between two line towers (its clearance test skips the first cell after a
        tower and does not look at a span of one step), and it checks the last span only when it has a target
        connection height, so a route can come out of the search with a span that is below the clearance there.
        A span at a connection point (the first span from a source connection height, the last span to a target
        connection height) the search samples in full, like this check. Each failing span is added to
        ``outcome.issues`` (a ``SpanRuleWarning`` is due); one that the search's own test would have passed counts
        in ``outcome.blind_spots``. Spans already reported by the removal check (a merged span) are skipped.
        Routes and numbers do not change.
        """
        if geometry is None or len(towers) < 2:
            return
        ends = self._ends
        n_cells = len(path_arr)
        n_spans = len(towers) - 1
        for i in range(n_spans):
            pos_a, pos_b = tower_positions[i], tower_positions[i + 1]
            if (pos_a, pos_b) in outcome.failed_spans:
                continue
            heights = (towers[i].height_m, towers[i + 1].height_m)
            first = i == 0
            strict = self._python_span_clearance(geometry, pos_a, pos_b, n_cells, {}, heights, strict=True,
                                                 start_cell=first)
            if strict is None or strict.ok:
                continue
            outcome.failed_spans.add((pos_a, pos_b))
            searched_in_full = (first and ends.source_attach is not None) or (
                i == n_spans - 1 and ends.target_attach is not None)
            kernel_view = self._python_span_clearance(geometry, pos_a, pos_b, n_cells, {}, heights)
            blind = not searched_in_full and kernel_view is not None and kernel_view.ok
            if n_spans == 1:
                label = "only span"
            elif i == 0:
                label = "first span"
            elif i == n_spans - 1:
                label = "last span"
            else:
                label = f"span {i + 1} of {n_spans}"
            (ax, ay), (bx, by) = idx_to_coord(path_arr[pos_a]), idx_to_coord(path_arr[pos_b])
            length = math.sqrt((bx - ax)**2 + (by - ay)**2)
            text = (f"the {label} from ({ax:.1f}, {ay:.1f}) to ({bx:.1f}, {by:.1f}) ({length:.1f} m, attachment "
                    f"heights {heights[0]:g} m and {heights[1]:g} m) fails the clearance check: "
                    f"{self._clearance_detail(strict, path_arr, ncols, idx_to_coord)}")
            if blind:
                outcome.blind_spots += 1
                start = (int(geometry.rows[pos_a]), int(geometry.cols[pos_a]))
                if pos_b - pos_a <= 1:
                    what = "a span of one step"
                elif first and strict.worst_cell == start:
                    what = "the cell the line starts on"
                else:
                    what = "the first cell after a tower"
                text += f"; the search accepted it because it does not test {what}"
            outcome.issues.append((text, True))

    def _build_removed_towers(self, outcome, path_indices, tower_height_map, ncols, raster, idx_to_coord):
        """Tower objects for the search towers the rule removed (``tower_id`` -1, with the costs the search charged)."""
        removed = []
        for pos in outcome.removed:
            tower = self._build_tower_object(
                pos, path_indices, outcome.search_chain, tower_height_map, outcome.terminals, ncols, raster,
                idx_to_coord, outcome.search_chain.index(pos))
            tower.tower_id = -1
            tower.note = outcome.removed_notes.get(pos)
            removed.append(tower)
        return removed

    def _fixed_line_height(self):
        """The one tower height of a profile without height classes (``tower_height_m``), else None."""
        profile = self._profile
        if profile.has_variable_height:
            return None
        if profile.tower_heights_m is not None and len(profile.tower_heights_m) == 1:
            return float(profile.tower_heights_m[0])
        return profile.tower_height_m

    def _search_description(self):
        """What the search did about the two end spans of this run, in one sentence (for ``SpanRuleWarning``).

        The source side names the height the search really used: the connection height, the height class it
        charged (several tower heights) or the one line height; the target side says whether the last span was
        checked at all (only with a target connection height).
        """
        ends = self._ends
        if ends.source_attach is not None:
            first = f"the first span from source_connection_height_m {ends.source_attach:g} m"
        elif ends.source_search_height is not None:
            first = f"the first span with the source at the {ends.source_search_height:g} m height class it chose"
        elif self._fixed_line_height() is not None:
            first = f"the first span with the line height {self._fixed_line_height():g} m at the source"
        else:
            first = "the first span with the line height at the source"
        if ends.target_attach is not None:
            return (f"The search checks {first} and the last span to target_connection_height_m "
                    f"{ends.target_attach:g} m.")
        return f"The search does not check the last span, and checks {first}."

    def _warn_span_rule(self, outcome):
        """One ``SpanRuleWarning`` for the route, listing the issues that ask for one.

        At most ``_MAX_LISTED_SPAN_ISSUES`` are listed; the total is given and ``ConstrainedPath.span_notes`` has
        all of them. When the strict check found spans that the search's own test passes, the message says plainly
        that the search does not test the first cell after a tower or a span of one step.
        """
        texts = [text for text, warn in outcome.issues if warn]
        if not texts:
            return
        shown = texts[:_MAX_LISTED_SPAN_ISSUES]
        listing = "; ".join(f"({k}) {text}" for k, text in enumerate(shown, start=1))
        count = f"{len(texts)} case{'s' if len(texts) != 1 else ''}"
        if len(texts) > len(shown):
            count += f", the first {len(shown)} listed here, ConstrainedPath.span_notes has all of them"
        blind = ""
        if outcome.blind_spots:
            blind = (f" The search does not test the first cell after a tower, nor a span of one step, nor the cell a "
                     f"line without a source connection height starts on, so it accepted {outcome.blind_spots} of "
                     f"these spans although the line is below the clearance there.")
        # what the search did about the end spans matters for a clearance failure only
        searched = f" {self._search_description()}" if outcome.failed_spans else ""
        warnings.warn(
            f"The line of profile {self._profile.name!r} breaks a span or clearance rule "
            f"(terminal_span_rule={self._profile.terminal_span_rule!r}); the route and its numbers are returned "
            f"as they are ({count}): {listing}.{searched}{blind}",
            SpanRuleWarning, skip_file_prefixes=_INTERNAL_FILE_PREFIXES)

    def _compute_tower_angle_and_bisector(self, pos, path_indices, x, y,
                                          idx_to_coord):
        """Compute turn angle and bisector at a tower position.

        Returns:
            Tuple of (turn_deg, bisector_rad).
        """
        turn_deg = 0.0
        bisector_rad = 0.0
        if 0 < pos < len(path_indices) - 1:
            prev = idx_to_coord(path_indices[pos - 1])
            curr = (x, y)
            nxt = idx_to_coord(path_indices[pos + 1])
            d_in = math.atan2(curr[1] - prev[1], curr[0] - prev[0])
            d_out = math.atan2(nxt[1] - curr[1], nxt[0] - curr[0])
            turn_deg = abs(math.degrees(
                math.atan2(math.sin(d_out - d_in), math.cos(d_out - d_in))
            ))
            # Bisector angle: halfway between incoming and outgoing
            bisector_rad = (d_in + d_out) / 2.0
            # Ensure bisector is on the correct side (acute bisector)
            if abs(d_out - d_in) > math.pi:
                bisector_rad += math.pi
        elif pos == 0 and len(path_indices) > 1:
            # Terminal start: align with outgoing segment
            nxt = idx_to_coord(path_indices[pos + 1])
            bisector_rad = math.atan2(nxt[1] - y, nxt[0] - x)
        elif pos == len(path_indices) - 1 and len(path_indices) > 1:
            # Terminal end: align with incoming segment
            prev = idx_to_coord(path_indices[pos - 1])
            bisector_rad = math.atan2(y - prev[1], x - prev[0])
        return turn_deg, bisector_rad

    def _classify_tower_type(self, is_terminal, turn_deg):
        """Determine tower type and angle cost from angle.

        Terminal towers are always classified as "terminal".
        Non-terminal towers are classified based on angle_types config. A
        turn sharper than every type gets the type with the largest
        ``max_angle_deg``, as the search does
        (``InfrastructureProfile.precompute_tower_angle_costs`` charges "the
        most expensive" = last sorted type): the same type and cost.

        Returns:
            Tuple of (tower_type, angle_cost), in profile units (the caller
            applies the objective's "tower" weight).
        """
        tower_type = "suspension"
        angle_cost = 0.0
        if is_terminal:
            tower_type = "terminal"
            angle_cost = (self._profile.terminal_tower_cost
                          if self._profile.terminal_tower_cost > 0
                          else 0.0)
        else:
            angle_types = self._profile.tower_cost_params.get(
                "angle_types", {})
            if angle_types:
                sorted_types = sorted(
                    angle_types.items(),
                    key=lambda t: t[1]["max_angle_deg"],
                )
                for tname, tconfig in sorted_types:
                    if turn_deg <= tconfig["max_angle_deg"]:
                        tower_type = tname
                        angle_cost = tconfig["base_cost"]
                        break
                else:
                    # sharper than every type: the search charges the last
                    # (largest max_angle_deg) type
                    tower_type, last = sorted_types[-1]
                    angle_cost = last["base_cost"]
        return tower_type, angle_cost

    def _resolve_tower_height(self, is_terminal, position, tower_height_map, end=None):
        """Resolve the height for the tower at ``position`` (its index along the route).

        ``end`` is ``"source"`` or ``"target"`` for the tower at that end of the route, else None. An end with a
        connection height of this run reports it. Other terminal towers use terminal_tower_height_m if available.
        Intermediate towers use algorithm-selected height (``tower_height_map``: position -> height) or fixed
        height.

        Returns:
            Tower height in meters (float or None).
        """
        if end == "source" and self._ends.source_height_m is not None:
            return self._ends.source_height_m
        if end == "target" and self._ends.target_height_m is not None:
            return self._ends.target_height_m
        if is_terminal and self._profile.terminal_tower_height_m is not None:
            return self._profile.terminal_tower_height_m

        tower_h = tower_height_map.get(position, None)
        # Fallback: use base height from profile
        if tower_h is None and self._profile.has_variable_height:
            tower_h = min(self._profile.effective_tower_heights_m)
        elif tower_h is None and self._profile.tower_height_m is not None:
            tower_h = self._profile.tower_height_m
        return tower_h

    def _build_tower_object(self, pos, path_indices, tower_positions, tower_height_map, terminal_positions,
                            ncols, raster, idx_to_coord, tid):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        """Build a single Tower from position data.

        ``pos`` is the tower's position along the route (never its cell: a route that crosses itself passes a cell
        twice), ``tower_positions`` the positions of all towers of the line (``tid`` indexes it),
        ``tower_height_map`` maps a position to the height the search chose, and ``terminal_positions`` holds the
        positions of the terminal towers.

        Handles angle computation, type classification, span computation,
        and height selection. The costs follow the rule of
        :meth:`_build_constrained_path`: ``terrain_cost`` (foundation) and
        ``angle_cost`` (tower type) are the profile costs times the objective's
        ``"tower"`` weight, in the user's units; without an objective they
        are the profile costs.

        Returns:
            Tower instance.
        """
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        cell_idx = int(path_indices[pos])
        x, y = idx_to_coord(cell_idx)
        r, c = cell_idx // ncols, cell_idx % ncols
        terrain_cost = self._tower_site_cost(r, c, raster,
                                             self._tower_terrain_costs)

        # Compute turn angle and bisector at tower
        turn_deg, bisector_rad = self._compute_tower_angle_and_bisector(
            pos, path_indices, x, y, idx_to_coord)

        # Terminal towers are dead-end structures at start/end
        is_terminal = pos in terminal_positions

        # Determine tower type from angle (terminals are always dead-end)
        tower_type, angle_cost = self._classify_tower_type(
            is_terminal, turn_deg)
        if self._w_tower != 1.0:
            # the foundation LUT is weighted already; the type cost too
            angle_cost = angle_cost * self._w_tower

        # Spans to previous/next tower
        span_prev = None
        if tid > 0:
            prev_pos = tower_positions[tid - 1]
            prev_coord = idx_to_coord(path_indices[prev_pos])
            span_prev = math.sqrt(
                (x - prev_coord[0])**2 + (y - prev_coord[1])**2
            )

        span_next = None
        if tid < len(tower_positions) - 1:
            next_pos = tower_positions[tid + 1]
            next_coord = idx_to_coord(path_indices[next_pos])
            span_next = math.sqrt(
                (x - next_coord[0])**2 + (y - next_coord[1])**2
            )

        # Height: an end with a connection height reports it, other terminal
        # towers use terminal_tower_height_m, intermediate towers use the
        # algorithm-selected height or the fixed height.
        tower_h = self._resolve_tower_height(
            is_terminal, pos, tower_height_map,
            self._end_of(pos, len(path_indices)))
        premium = 0.0 if is_terminal else self._height_premium(pos, tower_height_map)
        ground_area = self._profile.tower_ground_area_m2
        return Tower(
            location=Point(x, y),
            cell_index=cell_idx,
            tower_type=tower_type,
            turn_angle_deg=turn_deg,
            terrain_cost=terrain_cost,
            angle_cost=angle_cost,
            total_cost=terrain_cost + angle_cost + premium,
            height_premium=premium,
            span_to_previous_m=span_prev,
            span_to_next_m=span_next,
            tower_id=tid,
            height_m=tower_h,
            bisector_angle_rad=bisector_rad,
            ground_area_m2=ground_area,
        )

    def _premium_table(self):
        """``{height: premium}`` of the profile's height classes (heights as float32 values, like the kernels)."""
        return {float(np.float32(h)): float(p)
                for h, p in zip(self._profile.effective_tower_heights_m, self._profile.precompute_height_premium())}

    def _height_premium(self, position, tower_height_map):
        """The premium the search charged for the height class of the line tower at route ``position``.

        0.0 with one tower height, for a tower the search gave no height (no clearance check), and for the base
        height.
        """
        if not self._profile.has_variable_height:
            return 0.0
        height = tower_height_map.get(position)
        if height is None:
            return 0.0
        return self._premium_table().get(float(np.float32(height)), 0.0)

    def _source_height_class(self, geometry, outcome, path_arr, tower_height_map):
        """The height class (and premium) the search gave the structure at the source, or None.

        With several tower heights and no source connection height the kernels seed the source with every height
        class and charge the premium of the class the first span needs; they do not report the class. It follows
        from the route: the first span (source to the first search tower, or to the target on a line without a
        tower) is judged by the same clearance test as in the search, with the first tower's own height, and the
        cheapest class (lowest premium, then lowest height) that clears is the one the search used. A first span
        that is not checked (a line without a tower and without a target height) leaves the cheapest class.

        Returns:
            ``(height_m, premium)``, or None when the question does not arise: one tower height, a source
            connection height (the source pays no premium then), no clearance check or no search chain.
        """
        profile = self._profile
        if (geometry is None or outcome is None or len(outcome.search_chain) < 2 or not profile.has_variable_height
                or self._ends.source_height_m is not None):
            return None
        heights = list(profile.effective_tower_heights_m)
        premiums = [float(p) for p in profile.precompute_height_premium()]
        order = sorted(range(len(heights)), key=lambda k: (premiums[k], heights[k]))
        first, second = outcome.search_chain[0], outcome.search_chain[1]
        n_cells = len(path_arr)
        if second == n_cells - 1:
            second_height = self._ends.target_height_m              # no line tower: the span ends at the target
        else:
            second_height = tower_height_map.get(second)
        if second_height is not None:
            for k in order:
                result = path_span_clearance(
                    geometry.dem, geometry.obstacle, geometry.rows, geometry.cols, first, second, self._cell_size,
                    heights[k], second_height, conductor_weight_per_m=profile.conductor_weight_per_m,
                    conductor_tension_n=profile.conductor_tension_n, min_clearance_m=profile.min_clearance_m)
                if result is None or result.ok:
                    return heights[k], premiums[k]
        return heights[order[0]], premiums[order[0]]

    def _charge_source_height_class(self, towers, tower_positions, path_arr, geometry, outcome, tower_height_map,
                                    search_notes):
        """Give the source terminal the height class and premium the search charged (see ``_source_height_class``).

        The premium is part of the source tower's ``total_cost`` (and so of ``total_tower_cost``): the search added
        it to the route's cost. A class the search charged a premium for is the height of the source tower, also
        when the profile states a ``terminal_tower_height_m`` (the search checked the first span from that class,
        so the report, the clearance check of the report and the warning must use it too). Without a premium the
        reported height is the profile's ``terminal_tower_height_m`` when it states one, else the class height; a
        source connection height always wins (this method does nothing then). A premium above 0 is also said in
        ``search_notes``. The height the search used is kept in ``self._ends.source_search_height``.
        """
        if not towers or tower_positions[0] != 0 or towers[0].tower_type != "terminal":
            return
        chosen = self._source_height_class(geometry, outcome, path_arr, tower_height_map)
        if chosen is None:
            return
        height, premium = chosen
        source = towers[0]
        self._ends.source_search_height = height
        if premium > 0.0 or self._profile.terminal_tower_height_m is None:
            source.height_m = height
        if premium > 0.0:
            source.height_premium = premium
            source.total_cost += premium
            search_notes.append(
                f"the search gave the structure at the source the {height:g} m height class (premium {premium:g}): "
                f"the first span needs it; the premium is part of the source tower's cost")

    def _exact_bins_note(self):
        """The result note of a run that used exact span bins."""
        ends = self._ends
        why = []
        if ends.target_min_span > 0.0:
            why.append("terminal_span_rule='enforce'")
        if ends.source_attach is not None or ends.target_attach is not None:
            why.append("a connection height")
        return (f"{' and '.join(why)}: the search used exact span bins of {ends.span_bin_size:.4g} m "
                f"({ends.n_span_bins} bins, no longer than the shortest step) instead of min_span_m "
                f"{self._profile.min_span_m:g} m, so that no feasible route is hidden by merged states")

    def _base_height_shortcut_note(self):
        """The result note of a default run that took the lowest-height-only shortcut (several tower heights)."""
        height = self._ends.base_height_shortcut
        return (f"several tower heights: the search first looked for a route with the lowest height ({height:g} m) "
                f"alone and returned the one it found; it did not compare that route with routes that use taller "
                f"towers at their premium, which can be cheaper where a taller tower lets one span clear an obstacle "
                f"and saves towers (a known limitation of this shortcut)")

    def _tower_site_cost(self, row, col, raster, lut):
        """Foundation cost of a tower at (row, col) from the table ``lut``.

        Keyed like the search: by the cost layer (land use) in objective
        mode, by the search raster value otherwise. ``lut`` is in the
        user's units (the weighted or the profile table), never scaled by
        the search scale.
        """
        if self._tower_site_index is not None:
            return float(lut[self._tower_site_index[row, col]])
        return float(lut[raster[row, col]])

    def _turn_penalty_cost(self, path_indices, ncols, lut):
        """Sum of the turn penalties the search charged along a route.

        See :meth:`_turn_penalty_costs`; this is the one-table form.

        Returns:
            The total, or NaN (with one warning) when a step of the route is
            not a step of the neighbourhood.
        """
        return self._turn_penalty_costs(path_indices, ncols, (lut,))[0]

    def _turn_penalty_costs(self, path_indices, ncols, luts):
        """Sums of the turn penalties the search charged, one per table.

        The kernel adds ``angle_cost_lut[d_in, d_out]`` on every step
        (``d_in`` the direction of the step that reached the cell, ``d_out``
        the next one), so this walks the same steps; the first step counts
        as straight. Direction changes happen only at towers, so this is
        the sum over the towers' turns (plus ``lut[d, d]`` per straight
        step, 0 for every built-in angle cost function).

        A report must not turn a found route into an error: when a step of
        the route is not one of ``self.steps`` (a path that does not consist
        of single neighbourhood steps), the penalties cannot be recomputed,
        and every total is NaN (so ``total_cost`` is NaN too, never a number
        that is too small), with ONE ``PyorpsWarning`` for the route.

        Parameters:
            luts: Tuple of (n_dirs, n_dirs) tables in the user's units: the
                "smoothness"-weighted table and/or the profile's own.

        Returns:
            Tuple with one float per table.
        """
        if len(path_indices) < 2:
            return tuple(0.0 for _ in luts)
        idx = np.asarray(path_indices, dtype=np.int64)
        d_rows = np.diff(idx // ncols)
        d_cols = np.diff(idx % ncols)
        lookup = {(int(dr), int(dc)): k for k, (dr, dc) in enumerate(self.steps)}
        dirs = []
        for dr, dc in zip(d_rows, d_cols):
            step = (int(dr), int(dc))
            if step not in lookup:
                warnings.warn(
                    f"The turn penalties of this route could not be "
                    f"recomputed: it contains the step {step} (rows, "
                    f"columns), which is not a step of the neighbourhood "
                    f"{self.neighborhood_str!r}, so the turn penalty and "
                    f"the total cost are reported as NaN. The route and "
                    f"its towers are unaffected.",
                    PyorpsWarning, skip_file_prefixes=_INTERNAL_FILE_PREFIXES)
                return tuple(math.nan for _ in luts)
            dirs.append(lookup[step])
        totals = []
        for lut in luts:
            total = float(lut[dirs[0], dirs[0]])
            for d_in, d_out in zip(dirs[:-1], dirs[1:]):
                total += float(lut[d_in, d_out])
            totals.append(total)
        return tuple(totals)

    def _profile_tower_cost(self, towers, raster, ncols):
        """Sum of the towers' UNWEIGHTED profile costs (foundation + type)."""
        total = 0.0
        for tower in towers:
            r, c = tower.cell_index // ncols, tower.cell_index % ncols
            _, type_cost = self._classify_tower_type(
                tower.tower_type == "terminal", tower.turn_angle_deg)
            total += self._tower_site_cost(
                r, c, raster, self._profile_tower_terrain_costs) + type_cost
        return total

    def _compute_span_statistics(self, towers):
        """Compute span distances and tower type counts/costs.

        Returns:
            Dict with keys: spans, min_span_actual_m, max_span_actual_m,
            avg_span_m, tower_type_counts, tower_type_costs.
        """
        spans = []
        for i in range(len(towers) - 1):
            dx = towers[i+1].location.x - towers[i].location.x
            dy = towers[i+1].location.y - towers[i].location.y
            spans.append(math.sqrt(dx**2 + dy**2))

        tower_type_counts = {}
        tower_type_costs = {}
        for t in towers:
            tower_type_counts[t.tower_type] = (
                tower_type_counts.get(t.tower_type, 0) + 1
            )
            tower_type_costs[t.tower_type] = (
                tower_type_costs.get(t.tower_type, 0) + t.total_cost
            )

        return {
            "spans": spans,
            "min_span_actual_m": min(spans) if spans else 0.0,
            "max_span_actual_m": max(spans) if spans else 0.0,
            "avg_span_m": sum(spans) / len(spans) if spans else 0.0,
            "tower_type_counts": tower_type_counts,
            "tower_type_costs": tower_type_costs,
        }

    def _terrain_eur(self, raster, path_indices, ignore=(IMPASSABLE_CELL_COST,)):
        """Distance-weighted EUR for a path over this finder's search raster.

        Matches how ``Path.total_cost`` prices an unconstrained route:
        ``category_value (EUR/m) x category_length (m)``, summed over every
        category the path touches -- not a raw per-endpoint cell-value sum
        (which is dimensionless and skips every intermediate cell of a
        multi-cell r2+ step). ``length_by_category`` from
        ``calculate_path_metrics_numba`` is in CELL units, so it is scaled
        by ``cell_size`` here, mirroring ``path_finder.py``'s own scaling.
        """
        from pyorps.utils._traversal import calculate_path_metrics_numba
        if len(path_indices) < 2:
            return 0.0
        cell_size = abs(self.raster_handler.window_transform.a)
        _, categories, lengths = calculate_path_metrics_numba(
            np.ascontiguousarray(raster, dtype=np.uint16),
            np.asarray(path_indices, dtype=np.uint32), None)
        return float(sum(
            float(c) * float(length) * cell_size
            for c, length in zip(categories, lengths) if int(c) not in ignore
        ))

    def _build_constrained_path(self, path_indices, tower_cell_indices,
                                 raster, source, target, t_pathfinding,
                                 tower_heights_arr=None):
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        """Convert raw kernel output into ConstrainedPath with towers.

        Rule for every reported cost: it is the cost the search minimised,
        in the USER'S UNITS, i.e. the weighted objective term (the kernel's
        value divided by the search scale). Without an objective that is
        the profile and raster cost as before; with weights, a weight of 5
        shows 5 times the profile cost.

        - ``total_terrain_cost`` (``cost_breakdown["terrain"]``): the cell
          term ``sum_k w_k * M_k + w_length * length`` as the search priced
          it on its (quantised) raster, divided by the search scale; the
          exact unquantised value is ``Path.feasibility`` (per layer and
          unweighted: ``Path.metrics``). Planar, like ``_terrain_eur``.
        - ``Tower.terrain_cost`` / ``Tower.angle_cost`` / ``Tower.total_cost``
          and ``total_tower_cost``: foundation and tower-type costs times the
          ``"tower"`` weight. Terminal towers are included (with
          ``terminal_tower_cost``), although the search does not charge
          them: they are the same for every route between the two points.
        - ``total_angle_penalty_cost``: the turn penalties the search
          charged on this route (``_turn_penalty_cost``), times the
          ``"smoothness"`` weight. NaN (and so ``total_cost`` NaN), with one
          warning, when a step of the route is not a neighbourhood step.
        - A non-terminal tower sharper than every angle type has the type
          and cost of the last type, like the search.
        - ``total_cost`` = terrain + towers + turn penalties.
        - Several tower heights (DEM and clearance data): ``Tower.height_premium`` is the premium of the tower's
          height class that the search charged; it is part of ``Tower.total_cost`` and so of ``total_tower_cost``
          and ``total_cost``. ``cost_breakdown["height_premiums"]`` is their sum (already inside ``"towers"``,
          not an extra term; present only for such runs). The search also seeds the source with a height class and
          charges its premium; the class follows from the route (``_source_height_class``) and its premium is
          carried by the source terminal tower. A source connection height makes the source pay no premium.
        - With an objective, ``cost_breakdown`` also holds the profile's own
          (unweighted) costs of the route: ``"profile_towers"`` and
          ``"profile_angle_penalties"``.

        So ``total_cost - (terminal tower costs) + cost_breakdown["removed_towers"]``
        equals the kernel's optimal distance divided by the search scale, where
        ``"removed_towers"`` is the summed cost of the search towers that
        ``terminal_span_rule`` took out of the report next to an end point
        (``ConstrainedPath.removed_towers``; the key is there only when a tower
        was removed, and it is not part of ``total_cost``). Without a removed
        tower this is ``total_cost - (terminal tower costs)``, the report
        keeping every tower the search placed. The terminal tower costs are
        ``terrain_cost + angle_cost`` of the terminal towers (the foundation
        and ``terminal_tower_cost``, no height premium: the search charges no
        terminal tower, but it does charge the source's height class). It
        holds while the tower footprint mode is "uniform" and no DEM is given
        (the exact footprint sum and the slope factors of the DEM are charged
        by the search but not recomputed here; a flat DEM has slope factors
        of exactly 1, so it holds there, which is how the premiums are
        checked). The terrain is priced per metre for every cell size, by
        the search (the kernels get ``cell_size`` with or without a DEM) and
        by this report, so the identity holds at any cell size.

        End spans: a search tower closer than ``min_span_m`` to an end point
        is removed, kept or removed only if valid, as the profile's
        ``terminal_span_rule`` says (``_apply_terminal_span_rule``); under
        ``"enforce"`` the search itself keeps every end span at least
        ``min_span_m`` long. Flagged spans are listed in
        ``ConstrainedPath.span_notes`` and, where a rule of the profile is
        broken, in one ``SpanRuleWarning``. A connection height is the
        reported height of its end tower; ``ConstrainedPath`` records both
        heights, the span-bin size the search used (None without span
        limits) and ``search_notes``.

        Clearance report: with the clearance check on, EVERY span of the
        returned line is checked strictly (``_check_span_clearance``): every
        cell of the span, the first one after a tower too, and spans of one
        step, which the search's own test skips. A failing span is one entry
        of ``span_notes`` and of the route's single ``SpanRuleWarning``
        (``_warn_span_rule`` lists a few and gives the total). Routes and
        numbers are not changed by it.

        Position, not cell: the kernels return the cells of the towers, and
        a route can cross itself. Every step after matching the cells to
        positions along the route (``_tower_positions``) works on positions.
        """
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        ncols = raster.shape[1]
        transform = self.raster_handler.window_transform

        def idx_to_coord(idx):
            r, c = int(idx) // ncols, int(idx) % ncols
            x, y = transform @ (c + 0.5, r + 0.5)
            return x, y

        path_coords = np.array([idx_to_coord(i) for i in path_indices])
        path_geometry = LineString(path_coords) if len(path_coords) > 1 else None

        # Build the tower list from the Dijkstra output + start/end anchors.
        # Turn towers are already placed by the Dijkstra (mandatory tower
        # at every direction change), so only start/end need post-processing.
        # The kernels name a tower by its cell; everything below works on the
        # POSITION of a tower along the route (a route that crosses itself
        # passes a cell twice and can have a tower at each passage). The
        # height of a tower (variable-height output) is keyed by position too.
        path_arr = np.asarray(path_indices, dtype=np.int64)
        kernel_positions = self._tower_positions(path_arr, tower_cell_indices, ncols)
        tower_height_map = {}
        if tower_heights_arr is not None and len(tower_heights_arr) > 0:
            for i, pos in enumerate(kernel_positions):
                if pos is not None and i < len(tower_heights_arr):
                    tower_height_map[pos] = float(tower_heights_arr[i])
        search_positions = [pos for pos in kernel_positions if pos is not None]

        # Terminal towers (start/end anchors) connect to substation/transformer.
        # A search tower closer than min_span to an end point is handled by
        # the profile's terminal_span_rule (removed, removed only if valid,
        # or kept and flagged); the start and end cells are added as terminals.
        outcome = None
        geometry = None
        tower_positions = search_positions
        terminal_positions = set()
        min_span_m = (self._profile.min_span_m
                      if self._profile.has_span_constraints else 0.0)
        if len(path_indices) > 0 and self._profile.has_span_constraints:
            geometry = self._end_span_geometry(raster, path_indices)
            outcome = self._apply_terminal_span_rule(
                path_arr, search_positions, min_span_m,
                idx_to_coord, ncols, tower_height_map, geometry,
            )
            tower_positions, terminal_positions = outcome.positions, outcome.terminals

        towers = []
        for tid, pos in enumerate(tower_positions):
            towers.append(self._build_tower_object(
                pos, path_indices, tower_positions, tower_height_map,
                terminal_positions, ncols, raster, idx_to_coord, tid,
            ))

        removed_towers = []
        span_notes = []
        search_notes = []
        if outcome is not None:
            for tower, pos in zip(towers, tower_positions):
                tower.note = outcome.kept_notes.get(pos)
            removed_towers = self._build_removed_towers(
                outcome, path_indices, tower_height_map, ncols, raster,
                idx_to_coord)
            self._charge_source_height_class(
                towers, tower_positions, path_arr, geometry, outcome, tower_height_map, search_notes)
            self._check_span_clearance(
                towers, tower_positions, path_arr, geometry, outcome,
                idx_to_coord, ncols)
            span_notes = [text for text, _ in outcome.issues]
            self._warn_span_rule(outcome)

        # Cost breakdown, in the user's units (see the docstring)
        total_tower_cost = sum(t.total_cost for t in towers)
        total_terrain_cost = self._terrain_eur(raster, path_indices)
        if self._search_scale != 1.0:
            total_terrain_cost = total_terrain_cost / self._search_scale
        # One walk of the route and at most one warning, also when the
        # profile's own table is wanted too (NaN when a step is unmatched).
        luts = (self._angle_cost_lut,)
        if self.objective is not None:
            luts += (self._profile_angle_cost_lut,)
        turn_totals = self._turn_penalty_costs(path_indices, ncols, luts)
        total_angle_penalty_cost = turn_totals[0]
        cost_breakdown = {
            "terrain": total_terrain_cost,
            "towers": total_tower_cost,
            "angle_penalties": total_angle_penalty_cost,
        }
        if removed_towers:
            # informational, not part of total_cost: the search charged these
            # towers, the report no longer carries them (see the docstring)
            cost_breakdown["removed_towers"] = sum(
                t.total_cost for t in removed_towers)
        if tower_heights_arr is not None and self._profile.has_variable_height:
            # the part of "towers" that the search charged for the height classes (already in total_tower_cost)
            cost_breakdown["height_premiums"] = sum(t.height_premium for t in towers)
        if self._ends.exact_bins:
            search_notes.append(self._exact_bins_note())
        if self._ends.base_height_shortcut is not None:
            search_notes.append(self._base_height_shortcut_note())
        if self.objective is not None:
            # the profile's own costs of the same route, before weighting
            cost_breakdown["profile_towers"] = self._profile_tower_cost(
                towers, raster, ncols)
            cost_breakdown["profile_angle_penalties"] = turn_totals[1]

        stats = self._compute_span_statistics(towers)

        return ConstrainedPath(
            source=source,
            target=target,
            algorithm="constrained-dijkstra",
            graph_api=self._constrained_backend,
            path_indices=path_indices,
            path_coords=path_coords,
            path_geometry=path_geometry,
            euclidean_distance=math.sqrt(
                (path_coords[-1][0] - path_coords[0][0])**2 +
                (path_coords[-1][1] - path_coords[0][1])**2
            ) if len(path_coords) > 1 else 0.0,
            runtimes={"pathfinding": t_pathfinding},
            path_id=len(self.paths),
            search_space_buffer_m=self.search_space_buffer_m,
            neighborhood=self.neighborhood_str,
            profile_name=self._profile.name,
            towers=towers,
            n_towers=len(towers),
            total_terrain_cost=total_terrain_cost,
            total_tower_cost=total_tower_cost,
            total_angle_penalty_cost=total_angle_penalty_cost,
            cost_breakdown=cost_breakdown,
            spans=stats["spans"],
            min_span_actual_m=stats["min_span_actual_m"],
            max_span_actual_m=stats["max_span_actual_m"],
            avg_span_m=stats["avg_span_m"],
            turn_angles=[t.turn_angle_deg for t in towers],
            max_turn_angle_deg=(
                max(t.turn_angle_deg for t in towers) if towers else 0.0
            ),
            tower_type_counts=stats["tower_type_counts"],
            tower_type_costs=stats["tower_type_costs"],
            terminal_span_rule=self._profile.terminal_span_rule,
            removed_towers=removed_towers,
            span_notes=span_notes,
            source_connection_height_m=self._ends.source_height_m,
            target_connection_height_m=self._ends.target_height_m,
            span_bin_size_m=self._ends.span_bin_size,
            search_notes=search_notes,
        )
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
