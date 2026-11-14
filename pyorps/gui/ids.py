"""
PYORPS GUI: component-id constants.

Every component id that more than one module touches lives here so callbacks,
layout and tests never drift apart on a typo. Pattern-matching (dict) ids keep
their ``type`` strings here as well.
"""

# ----------------------------------------------------------------- map & hosts
MAP = "map"
BASEMAP_TILE = "basemap-tile"             # the single background tile layer (z-ordered)
LAYER_HOST = "layer-host"                 # dl.LayerGroup, direct Map child (C1)
DRAW_CONTROL = "draw-control"             # dl.EditControl inside its FeatureGroup
ACTIVE_ROUTE = "active-route"             # dl.Polyline for the route being edited
CONTROL_MARKERS = "control-markers"       # dl.LayerGroup with control-point markers
BUILDER_MARKERS = "builder-markers"       # dl.LayerGroup with route-builder points
MODE_BADGE = "mode-badge"                 # overlay badge showing the click mode

# pattern-matching id types rendered into the layer host
TYPE_LAYER_GEOJSON = "layer"              # {"type": TYPE_LAYER_GEOJSON, "id": <layer id>}
TYPE_RASTER_TILE = "rasttile"             # {"type": TYPE_RASTER_TILE, "id": <layer id>}

# ---------------------------------------------------------------------- stores
UI_STATE = "ui-state"                     # {active_tab, click_mode, active_route_id}
LAYERS_VIEW = "layers-view"               # ordered [{id,name,kind,visible,z}]
MAP_VIEW = "map-view"                     # {"fit_bounds": [[s,w],[n,e]], "seq": n}
NOTICES = "notices"                       # [notice dict, ...] (services.errors.Notice)
ROUTE_DRAFT = "route-draft"               # {"sources":[], "targets":[], "waypoints":[]}
EDIT_REQUEST = "edit-request"             # {"action","lat","lng","seq"} from dispatcher
COST_GRID_STATE = "cost-grid-state"       # {dataset_id, feature_keys}
FOCUS_FLASH = "focus-flash"               # {"control": id, "seq": n} -> scroll+flash

# --------------------------------------------------------------- notifications
NOTICE_STACK = "notice-stack"
TYPE_NOTICE = "notice"                    # {"type": TYPE_NOTICE, "index": <notice id>}
TYPE_NOTICE_FIX = "notice-fix"            # {"type": ..., "tab": <tab id>, "index": ...}
# persistent log viewer (every notice is also written to a rotating log file)
LOG_VIEW_BTN = "log-view-btn"
LOG_REFRESH_BTN = "log-refresh-btn"
LOG_OFFCANVAS = "log-offcanvas"
LOG_CONTENT = "log-content"
LOG_PATH_INFO = "log-path-info"

# ----------------------------------------------------------------------- tabs
TABS = "sidebar-tabs"
TAB_DATA = "tab-data"
TAB_COST = "tab-cost"
TAB_RASTER = "tab-raster"
TAB_ROUTES = "tab-routes"
TAB_EDIT = "tab-edit"
TAB_LAYERS = "tab-layers"
TAB_ATTRS = "tab-attrs"

# ------------------------------------------------------------------- data tab
STUDY_AREA_CLEAR = "study-area-clear"
STUDY_AREA_INFO = "study-area-info"
PROJECT_CRS = "project-crs"
DATA_SOURCE_TYPE = "data-source-type"     # radio: local | wfs
DATA_LOCAL_PANEL = "data-local-panel"     # shown only when source = local
DATA_WFS_PANEL = "data-wfs-panel"         # shown only when source = wfs
LOCAL_PATH = "local-path"
LOCAL_LAYER = "local-layer"               # optional sub-layer (gpkg)
LOCAL_LOAD_BTN = "local-load-btn"
LOCAL_LOAD_STATUS = "local-load-status"   # spinner target while loading
WFS_PRESET = "wfs-preset"
WFS_CATEGORY = "wfs-category"             # filter presets by dataset category
WFS_IN_VIEW_ONLY = "wfs-in-view-only"    # viewport filter toggle
WFS_URL = "wfs-url"
WFS_LAYER = "wfs-layer"
WFS_CAPS_BTN = "wfs-caps-btn"            # list feature types (GetCapabilities)
WFS_LAYER_SELECT = "wfs-layer-select"    # discovered feature types dropdown
WFS_LOAD_BTN = "wfs-load-btn"
WFS_LOAD_STATUS = "wfs-load-status"      # spinner target while loading
WFS_WHOLE_LAYER = "wfs-whole-layer"      # checkbox: load every feature (no study area)
CLIP_TO_AREA = "clip-to-area"
DATASET_LIST = "dataset-list"
# WMS overlays + DEM (BKG / topographic / infrastructure services)
OVERLAY_PRESET = "overlay-preset"
OVERLAY_LOAD_BTN = "overlay-load-btn"
DEM_LOAD_BTN = "dem-load-btn"            # fetch DGM DEM raster for the area
# OpenStreetMap features (Overpass API) — editable vector / cost source
OSM_PRESET = "osm-preset"
OSM_TAGS = "osm-tags"                    # custom tag filter (key or key=value)
OSM_LOAD_BTN = "osm-load-btn"
# categorized OSM menu: pick a feature column (tag key), then its values
OSM_KEY = "osm-key"                      # tag-key dropdown (searchable)
OSM_VALUES = "osm-values"                # values of the picked key (multi)
OSM_ADD_BTN = "osm-add-btn"              # add the key/value combination
OSM_SELECTIONS = "osm-selections"        # store: [{"key","values"}]
OSM_SELECTION_LIST = "osm-selection-list"  # chip list of added combinations
TYPE_OSM_SEL_REMOVE = "osm-sel-remove"   # {"type": ..., "index": n}
# merge several loaded vector layers into one (cross-state ALKIS projects)
MERGE_SELECT = "merge-select"            # multi-select of vector layers
MERGE_NAME = "merge-name"                # name of the merged layer
MERGE_BTN = "merge-btn"

# ------------------------------------------------------------------- cost tab
COST_DATASET = "cost-dataset"             # dropdown of loaded vector layers
COST_FEATURE_KEYS = "cost-feature-keys"   # ordered multi-select of columns
COST_FEATURE_INFO = "cost-feature-info"   # category counts / combination size
COST_SEED_BTN = "cost-seed-btn"
COST_GRID = "cost-grid"                   # dash-ag-grid cost table
COST_ADD_ROW_BTN = "cost-add-row-btn"
COST_DEL_ROW_BTN = "cost-del-row-btn"
COST_COVERAGE_INFO = "cost-coverage-info"
COST_IMPORT_PATH = "cost-import-path"
COST_IMPORT_BTN = "cost-import-btn"
COST_EXPORT_PATH = "cost-export-path"
COST_EXPORT_BTN = "cost-export-btn"
MODIFIER_GRID = "modifier-grid"           # ordered modifier list (F12)
MODIFIER_ADD_BTN = "modifier-add-btn"
MODIFIER_DEL_BTN = "modifier-del-btn"
PREPROC_SELECT = "preproc-select"
PREPROC_BUF_A = "preproc-buf-a"
PREPROC_BUF_B = "preproc-buf-b"
PREPROC_BUF_L = "preproc-buf-l"
# customizable preprocessing steps (Feature 4)
PREPROC_GRID = "preproc-grid"             # ordered step list (buffer/set/keep/drop)
PREPROC_ADD_BTN = "preproc-add-btn"
PREPROC_DEL_BTN = "preproc-del-btn"
# condition-group step builder: N (column, operator, value) conditions
# combined by a group operator, previewed in python-like plain english
PPB_DATASET = "ppb-dataset"               # dataset the step applies to
PPB_COLUMN = "ppb-column"                 # column dropdown (of the dataset)
PPB_OPERATOR = "ppb-operator"
PPB_VALUE = "ppb-value"                   # values present in the column
PPB_ADD_COND_BTN = "ppb-add-cond-btn"     # + column-value-operator group
PPB_CONDS = "ppb-conds"                   # store: [{"column","operator","value"}]
PPB_COND_LIST = "ppb-cond-list"           # rendered condition chips
TYPE_PPB_COND_REMOVE = "ppb-cond-remove"  # {"type": ..., "index": n}
PPB_COMBINE = "ppb-combine"               # group operator: & (and) | (or)
PPB_OP = "ppb-op"                         # action: buffer / set / keep / drop
PPB_TARGET = "ppb-target"                 # target column (set)
PPB_ARG = "ppb-arg"                       # buffer metres / new value
PPB_PREVIEW = "ppb-preview"               # plain-english combined mask
PPB_ADD_STEP_BTN = "ppb-add-step-btn"     # append the step to PREPROC_GRID
# custom drawn cost layer (Feature 5 / task 33 — editable in the Cost tab)
MANUAL_NAME = "manual-cost-name"
MANUAL_COST = "manual-cost-value"
MANUAL_MODE = "manual-cost-mode"          # base | override
MANUAL_CREATE_BTN = "manual-cost-create-btn"
MANUAL_DEL_BTN = "manual-cost-del-btn"    # remove selected polygon(s) from the list
MANUAL_STATUS = "manual-cost-status"
DRAW_TARGET = "draw-target"               # store: "area" | "cost" (by tab)
MANUAL_GRID = "manual-cost-grid"          # per-polygon name/cost editor

# ----------------------------------------------------------------- raster tab
# explicit dataset + cost-table pairing for rasterization (only compatible
# tables are offered for the picked dataset)
RASTERIZE_DATASET = "rasterize-dataset"
RASTERIZE_TABLE = "rasterize-table"
RES_M = "raster-resolution-m"
RASTER_SIZE_INFO = "raster-size-info"
FILL_VALUE = "raster-fill-value"
RASTER_DTYPE = "raster-dtype"
GEOM_BUFFER = "raster-geom-buffer"
# thin-feature burn controls + the border crop (card "Rasterization" > Advanced)
WIDEN_THIN = "raster-widen-thin"          # switch: widen_thin_forbidden
ALL_TOUCHED = "raster-all-touched"        # switch: all_touched
ON_THIN = "raster-on-thin"                # dropdown: warn | raise | ignore
CROP_BORDER = "raster-crop-border"        # switch: shrink_raster after the burn
RASTER_SETTINGS = "raster-settings"       # store: mirror of the rasterize controls
RASTER_SAVE_PATH = "raster-save-path"
RASTERIZE_BTN = "rasterize-btn"
RASTERIZE_LOG = "rasterize-log"
RASTER_LOAD_PATH = "raster-load-path"
RASTER_LOAD_BTN = "raster-load-btn"
RASTER_OPACITY = "raster-opacity"
RASTER_COLORMAP = "raster-colormap"       # F3: selectable colormap
COLORMAP_LEGEND = "colormap-legend"       # cost <-> colour <-> combination
# graduated rendering (QGIS-style classification of the selected raster)
GRAD_MODE = "grad-mode"                   # continuous | equal | quantile | …
GRAD_CLASSES = "grad-classes"             # number of classes
GRAD_APPLY_BTN = "grad-apply-btn"
GRAD_STATUS = "grad-status"
TYPE_GRAD_COLOR = "grad-color"            # {"type": ..., "layer","index"}
# raster algebra — combine cost rasters (Feature 6)
RASTER_COMBINE_SELECT = "raster-combine-select"   # multi-select of raster layers
RASTER_COMBINE_OP = "raster-combine-op"           # add|multiply|min|max|overlay|merge
RASTER_COMBINE_BTN = "raster-combine-btn"
RASTER_COMBINE_STATUS = "raster-combine-status"

# ----------------------------------------------------------------- routes tab
BUILD_MODE = "build-mode"                 # radio: off / +source / +target / +waypoint
BUILD_POINTS_TABLE = "build-points-table"
BUILD_CLEAR_BTN = "build-clear-btn"
NEIGHBORHOOD = "route-neighborhood"       # number input: the order k of "rK"
ALGORITHM = "route-algorithm"
GRAPH_API = "route-graph-api"             # dropdown "Backend"
SEARCH_BUFFER = "route-search-buffer"
SEARCH_BUFFER_INFO = "route-search-buffer-info"
FULL_WINDOW = "route-full-window"         # switch: the whole raster is the window
DELTA = "route-delta"
NUM_THREADS = "route-num-threads"
MARGIN = "route-margin"                   # delta-stepping termination margin
# wrapper divs carrying the tooltip of the three delta-stepping settings (the
# dbc inputs take no ``title``); the tooltip says why a field is switched off
DELTA_WRAP = "route-delta-wrap"
THREADS_WRAP = "route-num-threads-wrap"
MARGIN_WRAP = "route-margin-wrap"
IGNORE_MAX_COST = "route-ignore-max-cost"
PAIRWISE = "route-pairwise"
SIMPLIFY = "route-simplify"
SIMPLIFY_TOL = "route-simplify-tol"
ROUTE_RASTER = "route-raster"             # dropdown: which raster layer to route on
RUN_ROUTING_BTN = "run-routing-btn"
# busy flags of the three gated buttons: flipped by running= on their callback,
# read by the workflow gating callback (callbacks/guide.py) so gating always wins
RASTERIZE_BUSY = "rasterize-busy"
RUN_ROUTING_BUSY = "run-routing-busy"
# both runners listen to RUN_ROUTING_BTN; each flips its OWN flag through
# running= (a shared one would be cleared by the runner that ends at once) and
# a small callback ORs them into RUN_ROUTING_BUSY
RUN_PLAIN_BUSY = "run-plain-busy"
RUN_OHL_BUSY = "run-ohl-busy"
RASTER_COMBINE_BUSY = "raster-combine-busy"
ROUTING_STATUS = "routing-status"
ROUTING_SETTINGS = "routing-settings"     # store: mirror of the routing controls
# heavy-run confirmation + background job control (accept waiting / stop)
ROUTING_CONFIRM_MODAL = "routing-confirm-modal"
ROUTING_CONFIRM_BODY = "routing-confirm-body"
ROUTING_CONFIRM_OK = "routing-confirm-ok"
ROUTING_CONFIRM_CANCEL = "routing-confirm-cancel"
ROUTING_STOP_BTN = "routing-stop-btn"
ROUTING_POLL = "routing-poll"             # dcc.Interval polling the job
ROUTES_LOAD_PATH = "routes-load-path"
ROUTES_LOAD_BTN = "routes-load-btn"

# ------------------------------------------------- overhead line (constrained)
OHL_ENABLE = "ohl-enable"
OHL_COLLAPSE = "ohl-collapse"
OHL_PROFILE_PRESET = "ohl-profile-preset"
OHL_PROFILE_TEXT = "ohl-profile-text"
OHL_PROFILE_PATH = "ohl-profile-path"
OHL_PROFILE_LOAD_BTN = "ohl-profile-load-btn"
OHL_PROFILE_SAVE_BTN = "ohl-profile-save-btn"
OHL_BACKEND = "ohl-backend"
OHL_EXPERIMENTAL = "ohl-experimental"
OHL_DEM = "ohl-dem"
OHL_DSM = "ohl-dsm"
OHL_RUN_BTN = "ohl-run-btn"
OHL_STATUS = "ohl-status"

# ---------------------------------------------- route selection & editing
EDIT_ROUTE_SELECT = "edit-route-select"
EDIT_ENABLE = "edit-enable"               # switch: map clicks edit the selected route
POINTS_APPLY_BTN = "points-apply-btn"     # grid rows -> variant / draft
POINTS_REMOVE_BTN = "points-remove-btn"   # delete selected grid rows
NEW_ROUTE_BTN = "new-route-btn"           # deselect -> grid shows the draft
EDIT_COST_READOUT = "edit-cost-readout"
AUTO_REFRESH = "edit-auto-refresh"        # recompute on every edit (default on)
REFRESH_ROUTES_BTN = "refresh-routes-btn" # recompute the staged/edited routes
EDIT_UNDO_BTN = "edit-undo-btn"           # bring back the route before the last edit
EDIT_SIMPLIFY_TOL = "edit-simplify-tol"
EDIT_SIMPLIFY_BTN = "edit-simplify-btn"
TYPE_DATASET_REFRESH = "dataset-refresh"  # {"type": ..., "index": layer id}
DELETE_VARIANT_BTN = "delete-variant-btn"
# editable routes grid (grouped by group) + group management (Feature 2)
ROUTES_GRID = "routes-grid"
ROUTE_MOVE_GROUP = "route-move-group"       # target group for "Move to group"
ROUTE_MOVE_BTN = "route-move-btn"
GROUP_SELECT = "group-select"               # pick a group for group-level ops
GROUP_RENAME = "group-rename"
GROUP_RENAME_BTN = "group-rename-btn"
GROUP_COLOR = "group-color"
GROUP_DASH = "group-dash"
GROUP_RESTYLE_BTN = "group-restyle-btn"
GROUP_DELETE_BTN = "group-delete-btn"
GROUP_UP_BTN = "group-up-btn"
GROUP_DOWN_BTN = "group-down-btn"
READD_POINTS_BTN = "readd-points-btn"       # group points -> new-routing draft
EXPORT_ROUTE_PATH = "export-route-path"
EXPORT_ROUTE_BTN = "export-route-btn"
EXPORT_STATUS = "export-status"

# ----------------------------------------------------------------- layers tab
LAYERS_GRID = "layers-grid"
LAYER_REMOVE_BTN = "layer-remove-btn"
LAYER_VIEW_BTN = "layer-view-btn"           # open the selected layer's table
LAYER_TABLE_OFFCANVAS = "layer-table-offcanvas"
LAYER_TABLE_GRID = "layer-table-grid"       # full attribute table (excel-like)
LAYER_TABLE_TITLE = "layer-table-title"
# background map controls (z-order editable alongside overlays)
BASEMAP_SELECT = "basemap-select"
BASEMAP_OPACITY = "basemap-opacity"
BASEMAP_ZORDER = "basemap-zorder"           # below | above the raster overlays
# hierarchical route tree (route layer -> groups -> routes)
ROUTE_TREE = "route-tree"
TYPE_ROUTE_TREE_ITEM = "route-tree-item"    # {"type": ..., "id": <route id>}
# step-by-step workflow guide + gating
WORKFLOW_GUIDE = "workflow-guide"
# feature selection (click a feature of the selected layer -> highlight it)
SELECTED_FEATURE = "selected-feature"       # store: {"layer_id","row","props"}
SELECTED_FEATURE_LAYER = "selected-feature-layer"  # map GeoJSON highlight overlay

# ------------------------------------------------------------------ attrs tab
ATTR_PANEL = "attr-panel"

# ------------------------------------------------------------ app shell / theme
APP_SHELL = "app-shell"                   # root row; collapse class lands here
THEME_TOGGLE = "theme-toggle"             # header light/dark switch
THEME_STORE = "theme-store"               # last applied theme ("light"|"dark")
SIDEBAR_TOGGLE = "sidebar-toggle"         # header button: collapse the sidebar

# ---------------------------------------------------------------- project I/O
SAVE_INCLUDE = "save-include"             # checklist: raster|vector|cost_table
DIRTY_STORE = "dirty-store"               # bool: unsaved work (close guard)
PROJECT_SAVE_PATH = "project-save-path"
PROJECT_SAVE_BTN = "project-save-btn"
PROJECT_OPEN_PATH = "project-open-path"
PROJECT_OPEN_BTN = "project-open-btn"
PROJECT_NEW_BTN = "project-new-btn"
PROJECT_STATUS = "project-status"

# ---------------------------------------------------- confirm before destroying
# ONE modal (title / text / confirm label) and a small store that holds the
# action waiting for the answer. Every destructive button feeds a gate callback
# (callbacks/confirm.py): it asks, or - when nothing would be lost - goes ahead
# at once by writing the action's GO store; the action callback listens on that
# store (a counter) and never on the button itself.
CONFIRM_MODAL = "confirm-modal"
CONFIRM_TITLE = "confirm-title"
CONFIRM_BODY = "confirm-body"
CONFIRM_OK = "confirm-ok"
CONFIRM_CANCEL = "confirm-cancel"
CONFIRM_PENDING = "confirm-pending"       # store: {"action", "label"} or None
GO_NEW_PROJECT = "go-new-project"
GO_OPEN_PROJECT = "go-open-project"
GO_SAVE_PROJECT = "go-save-project"
GO_REMOVE_LAYERS = "go-remove-layers"
GO_DELETE_GROUP = "go-delete-group"
GO_DELETE_VARIANT = "go-delete-variant"
GO_EXPORT_ROUTE = "go-export-route"
GO_RASTERIZE = "go-rasterize"
GO_EXPORT_COSTS = "go-export-costs"
GO_SAVE_PROFILE = "go-save-profile"
GO_SAVE_CORRIDOR = "go-save-corridor"
# every GO store, in the order of callbacks/confirm.py's gate table
GO_STORES = (GO_NEW_PROJECT, GO_OPEN_PROJECT, GO_SAVE_PROJECT, GO_REMOVE_LAYERS,
             GO_DELETE_GROUP, GO_DELETE_VARIANT, GO_EXPORT_ROUTE, GO_RASTERIZE,
             GO_EXPORT_COSTS, GO_SAVE_PROFILE, GO_SAVE_CORRIDOR)

# =============================================================================
# Round 14: every setting reachable in the GUI. Added in one go by the scaffold
# package (P0) so the later packages never edit this file. Each block names the
# package that owns the controls; the prefixes (obj-, cf-, cor-, ohlf-) are
# distinct on purpose: the test harness finds a callback by substring match of
# the ids, so a new id must never contain an older one.
# =============================================================================

# ------------------------------------- objective & terrain (gaps 1, 2, 6; P2 + P5)
OBJ_CARD = "obj-card"
OBJ_PRESET_CLASSIC_BTN = "obj-preset-classic"
OBJ_PRESET_SHORTEST_BTN = "obj-preset-shortest"
OBJ_PRESET_GENTLEST_BTN = "obj-preset-gentlest"
OBJ_W_COST = "obj-w-cost"
OBJ_W_LENGTH = "obj-w-length"
OBJ_W_GRADIENT = "obj-w-gradient"
OBJ_W_GRADIENT_WRAP = "obj-w-gradient-wrap"
OBJ_LAYER_NAME = "obj-mlayer-name"
OBJ_LAYER_SOURCE = "obj-mlayer-source"
OBJ_LAYER_WEIGHT = "obj-mlayer-weight"
OBJ_LAYER_ADD_BTN = "obj-mlayer-add"
OBJ_LAYER_DEL_BTN = "obj-mlayer-del"
# NOT "obj-mlayers-grid": that contains LAYERS_GRID ("layers-grid"), which the
# test harness uses as a fragment to find callbacks (planned value changed)
OBJ_LAYERS_GRID = "obj-mlayer-grid"
OBJ_DEM_LAYER = "obj-dem-layer"
OBJ_SLOPE_WRAP = "obj-slope-wrap"
OBJ_GRAD_MAX_PCT = "obj-grad-max-pct"
OBJ_GRAD_ADDITIVE = "obj-grad-additive"
OBJ_GRAD_ADDITIVE_SCALE = "obj-grad-additive-scale"
OBJ_GRAD_MULTIPLIER = "obj-grad-multiplier"
OBJ_GRAD_MULT_SCALE = "obj-grad-mult-scale"
OBJ_GRAD_SIG_CENTER = "obj-grad-sig-center"
OBJ_GRAD_SIG_STEEPNESS = "obj-grad-sig-steepness"
OBJ_GRAD_SIG_MAX = "obj-grad-sig-max"
OBJ_GRAD_BIN_PCT = "obj-grad-bin-pct"
OBJ_GRAD_SMAX_PCT = "obj-grad-smax-pct"
OBJ_PRECISION = "obj-precision"
OBJ_PRECISION_WRAP = "obj-precision-wrap"
OBJ_STATUS = "obj-status"
OBJ_SETTINGS = "obj-settings"             # store: mirror of the objective card
# wrappers whose title or visibility a callback changes (moved here from sections/objective.py)
OBJ_DEM_WRAP = "obj-dem-box"              # the terrain model dropdown (hover text names why a DEM is refused)
OBJ_ADDITIVE_SCALE_WRAP = "obj-shape-scale-box"   # "Exponential shape scale" (exponential slope shape only)
OBJ_MULT_SCALE_WRAP = "obj-mult-scale-box"        # "Multiplier scale" (exponential, energy, squared multipliers)
OBJ_SIGMOID_WRAP = "obj-sigmoid-box"              # the three sigmoid fields (sigmoid multiplier only)

# ------------------------------------------------------- cost field (gap 3; P11)
CF_CARD = "cf-card"
CF_ORIGINS = "cf-origins"
CF_COMBINE = "cf-combine"
CF_ALGORITHM = "cf-algorithm"
CF_THREADS = "cf-threads"
CF_RUN_BTN = "cf-run-btn"
CF_STOP_BTN = "cf-stop-btn"
CF_POLL = "cf-poll"                       # dcc.Interval polling the job
CF_STATUS = "cf-status"
CF_BEST_INFO = "cf-best-info"
CF_SETTINGS = "cf-settings"               # store: mirror of the cost-field card
CF_RUN_WRAP = "cf-run-wrap"               # wrapper of the Compute button (carries the hover text of a disabled one)

# ------------------------------------------------ corridor graph (gap 4; P12)
COR_CARD = "cor-card"
COR_TERMINALS = "cor-terminals"
COR_K = "cor-k"
COR_MIN_SHARED = "cor-min-shared"
COR_MAX_PAIR_COST = "cor-max-pair-cost"
COR_VALIDATE = "cor-validate"
COR_RUN_BTN = "cor-run-btn"
COR_POLL = "cor-poll"                     # dcc.Interval polling the job
COR_STATUS = "cor-status"
COR_SAVE_PATH = "cor-save-path"
COR_SAVE_BTN = "cor-save-btn"
COR_SETTINGS = "cor-settings"             # store: mirror of the corridor card
COR_RUN_WRAP = "cor-run-wrap"             # wrapper of the Build button (carries the hover text of a disabled one)

# --------------------------------------------- profile form (gap 5; P3 + P6)
# the scalar fields of the constrained-profile form, in form order; the id of
# a field is "ohlf-" + the key with dashes (ohlf-soft-angle-limit-deg ...)
OHLF_KEYS = (
    "name", "description", "soft_angle_limit_deg", "hard_angle_limit_deg",
    "angle_cost_function", "angle_cost_scale", "min_span_m", "max_span_m",
    "span_bin_size_m", "terrain_interpolation", "max_gradient_percent",
    "gradient_scale", "tower_height_m", "tower_heights_m",
    "tower_height_cost_per_increment", "terminal_tower_height_m",
    "terminal_tower_cost", "conductor_weight_per_m", "conductor_tension_n",
    "min_clearance_m", "tower_ground_area_m2", "tower_area_cost_mode",
    "recommended_buffer_m",
    # round 17 (P5): the end-span rule, the 24th field; it sits in the "Line ends" group at the end of the form
    "terminal_span_rule",
)
OHLF_IDS = {key: "ohlf-" + key.replace("_", "-") for key in OHLF_KEYS}
OHLF_ANGLE_GRID = "ohlf-angle-grid"
OHLF_ANGLE_ADD_BTN = "ohlf-angle-add"
OHLF_ANGLE_DEL_BTN = "ohlf-angle-del"
OHLF_TERRAIN_GRID = "ohlf-terrain-costs-grid"
OHLF_TERRAIN_ADD_BTN = "ohlf-terrain-add"
OHLF_TERRAIN_DEL_BTN = "ohlf-terrain-del"
OHLF_TYPES_GRID = "ohlf-types-grid"
OHLF_TYPES_ADD_BTN = "ohlf-types-add"
OHLF_TYPES_DEL_BTN = "ohlf-types-del"
OHLF_FROM_YAML_BTN = "ohlf-from-yaml"
OHLF_STATUS = "ohlf-status"
OHLF_CLEARANCE_INFO = "ohlf-clearance-info"
OHLF_ANGLE_SCALE_WRAP = "ohlf-angle-scale-wrap"   # "Turn cost factor" (linear and quadratic turn cost only)
OHLF_ANGLE_TABLE_WRAP = "ohlf-angle-table-wrap"   # the turn cost table (piecewise turn cost only)
OHLF_BASE = "ohlf-base"                   # store: the last loaded profile dict
OHLF_YAML_DETAILS = "ohlf-yaml-details"   # <details> "Raw YAML (advanced)"
OHL_SETTINGS = "ohlf-settings"            # store: mirror of the constrained settings

# ------------- objective weights of constrained routing (round 16; R3-B), prefix "ohlo-"
# the weight keys of services/constrained_objective, in form order; the id of a
# weight is "ohlo-w-" + the key (ohlo-w-smoothness, ohlo-w-tower)
OHLO_KEYS = ("smoothness", "tower")
OHLO_IDS = {key: "ohlo-w-" + key for key in OHLO_KEYS}
OHLO_BLOCK = "ohlo-block"                 # <details> "Objective weights"
OHLO_FIELDS = "ohlo-fields"               # <fieldset> around the number inputs (disabled with a reason)
OHLO_STATUS = "ohlo-status"               # summary line, or the first problems
OHLO_RESET_BTN = "ohlo-reset"             # "Reset weights": every weight back to its default
OHLO_SETTINGS = "ohlo-settings"           # store: mirror of the weights

# ------------- line ends of constrained routing (round 17; P5), prefix "ohle-"
# the two connection heights of services/constrained_ends (the finder's own keywords), in form order; the id of a
# height is "ohle-" + the key with dashes. The end-span rule is a PROFILE field: its select is
# OHLF_IDS["terminal_span_rule"].
OHLE_KEYS = ("source_connection_height_m", "target_connection_height_m")
OHLE_IDS = {key: "ohle-" + key.replace("_", "-") for key in OHLE_KEYS}
OHLE_BLOCK = "ohle-block"                 # <details> "Line ends" (the rule select and the two heights)
OHLE_FIELDS = "ohle-fields"               # <fieldset> around the two heights (disabled with a reason)
OHLE_STATUS = "ohle-status"               # summary line of the heights, or the first problems
OHLE_RULE_STATUS = "ohle-rule-status"     # line under the rule select: why it is disabled, what a GPU backend refuses
OHLE_CLEAR_BTN = "ohle-clear"             # "Clear heights": both heights back to blank
OHLE_SETTINGS = "ohle-settings"           # store: mirror of the heights

# --------------------------------------- background jobs (gap 7; P1, P7, P10)
RASTERIZE_STOP_BTN = "rasterize-stop-btn"
RASTERIZE_POLL = "rasterize-poll"         # dcc.Interval polling the job
OHL_STOP_BTN = "ohl-stop-btn"
OHL_POLL = "ohl-poll"                     # dcc.Interval polling the job

# ----------------- one-shot stores of the job handover (round 15, jobs.poll_tick)
# The poll tick that hands an ended job's outputs to the page also writes the
# handover's token here; the next tick sends it back, which tells the server that
# the page has the outputs. One store per poll; no other callback writes them.
JOB_ACK_RASTERIZE = "job-ack-rasterize"
JOB_ACK_CONSTRAINED = "job-ack-constrained"
JOB_ACK_COST_FIELD = "job-ack-cost-field"
JOB_ACK_CORRIDOR = "job-ack-corridor"
JOB_ACK_ROUTING = "job-ack-routing"
JOB_ACK_STORES = (JOB_ACK_RASTERIZE, JOB_ACK_CONSTRAINED, JOB_ACK_COST_FIELD,
                  JOB_ACK_CORRIDOR, JOB_ACK_ROUTING)
