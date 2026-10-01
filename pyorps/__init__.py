"""PYORPS - Python for Optimal Routes in Power Systems."""

__version__ = "0.3.2"

# Suppress third-party deprecation warnings triggered during import
import warnings as _warnings

_warnings.filterwarnings(
    "ignore",
    message="The 'shapely.geos' module is deprecated",
    category=DeprecationWarning,
)
del _warnings

# Import key components for easy access
from .core.cost_assumptions import (
    CostAssumptions,
    detect_feature_columns,
    get_zero_cost_assumptions,
    save_empty_cost_assumptions,
)
from .core.exceptions import (
    AlgorithmNotImplementedError,
    CostAssumptionsError,
    NoPathFoundError,
    PairwiseError,
    PyorpsError,
    RasterShapeError,
    WFSError,
)
from .core.ensemble import RouteEnsemble
from .core.metric_stack import MetricStack
from .core.objective import (
    GradientOptions,
    Objective,
)
from .core.path import (  # Fixed: import from core.path instead of graph
    Path,
    PathCollection,
)
from .graph.path_finder import PathFinder
from .graph.search_session import (
    CostField,
    CostFieldSet,
    Leg,
    SavedCostField,
    SearchSession,
    fields_fit_in_memory,
    full_window_buffer_m,
)
from .graph.tower_field import (
    AngleTables,
    ClearanceModel,
    TowerField,
    TowerFieldModel,
    TowerFieldSolver,
    TowerLattice,
    angle_tables_from_profile,
    check_bounds,
    clearance_from_profile,
    solve_tower_field,
    tower_field_bounds,
    tower_field_from_raster,
)
from .graph.corridor import (
    cell_sharing_profile,
    corridor_graph_from_routes,
    route_metrics,
    supercover_cells,
)
from .io.geo_dataset import (
    GeoDataset,
    InMemoryRasterDataset,
    InMemoryVectorDataset,
    LocalRasterDataset,
    LocalVectorDataset,
    RasterDataset,
    VectorDataset,
    WFSVectorDataset,
    initialize_geo_dataset,
)
from .raster.rasterizer import GeoRasterizer
from .raster.thinness import (
    MIN_FORBIDDEN_WIDTH_CELLS,
    ForbiddenBurnAssessment,
    ForbiddenBurnReport,
    ForbiddenBurnRoutingAssessment,
    ForbiddenBurnSeverity,
    RepairSealReport,
    ResolutionAdvice,
    SealedOpeningError,
    SealedOpeningWarning,
    ThinForbiddenFeatureError,
    ThinForbiddenFeatureWarning,
    detect_forbidden_burn_defects,
    detect_repair_seals,
    is_thin,
    min_feature_width,
    recommended_geometry_buffer_m,
    safe_forbidden_width_m,
    suggest_resolution,
    thin_parts,
    widen_thin_features,
)

__all__ = [
    # Core dataset classes
    "GeoDataset", "VectorDataset", "RasterDataset",
    "InMemoryVectorDataset", "LocalVectorDataset",
    "WFSVectorDataset", "LocalRasterDataset",
    "InMemoryRasterDataset", "initialize_geo_dataset",

    # Rasterization
    "GeoRasterizer",

    # Sub-cell forbidden features (thin barriers)
    "MIN_FORBIDDEN_WIDTH_CELLS", "safe_forbidden_width_m",
    "recommended_geometry_buffer_m",
    "is_thin", "min_feature_width", "thin_parts", "widen_thin_features",
    "detect_forbidden_burn_defects", "ForbiddenBurnReport",
    "ForbiddenBurnAssessment", "ForbiddenBurnSeverity",
    "ForbiddenBurnRoutingAssessment",
    "suggest_resolution", "ResolutionAdvice",
    "ThinForbiddenFeatureWarning", "ThinForbiddenFeatureError",
    "detect_repair_seals", "RepairSealReport",
    "SealedOpeningWarning", "SealedOpeningError",

    # Graph and routing
    "PathFinder", "Path", "PathCollection",

    # Retained search fields (rooted cost fields, incremental editing)
    "CostField", "CostFieldSet", "SavedCostField", "SearchSession",
    "Leg", "full_window_buffer_m", "fields_fit_in_memory",
    "corridor_graph_from_routes", "route_metrics", "cell_sharing_profile",
    "supercover_cells",

    # Precomputed tower fields (overhead siting without a state explosion)
    "TowerField", "TowerFieldModel", "TowerFieldSolver", "TowerLattice",
    "AngleTables", "ClearanceModel", "solve_tower_field",
    "tower_field_from_raster", "tower_field_bounds", "check_bounds",
    "angle_tables_from_profile", "clearance_from_profile",

    # Feasibility objective
    "Objective", "GradientOptions", "MetricStack", "RouteEnsemble",

    # Cost assumptions
    "CostAssumptions", "get_zero_cost_assumptions", "detect_feature_columns",
    "save_empty_cost_assumptions",

    # Exceptions
    "PyorpsError", "NoPathFoundError", "RasterShapeError",
    "AlgorithmNotImplementedError", "PairwiseError",
    "CostAssumptionsError", "WFSError",
]
