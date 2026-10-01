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
DRAG_EVENT = "drag-event"                 # drag-shim events (optional, Phase 8)
ROUTE_DRAFT = "route-draft"               # {"sources":[], "targets":[], "waypoints":[]}
EDIT_REQUEST = "edit-request"             # {"action","lat","lng","seq"} from dispatcher
COST_GRID_STATE = "cost-grid-state"       # {dataset_id, feature_keys}
RECOMPUTE_FLASH = "recompute-flash"       # bumps when a recompute starts (dash feedback)
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
DRAW_TARGET_RADIO = "draw-target-radio"   # (retired) legacy manual toggle
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
NEIGHBORHOOD = "route-neighborhood"
ALGORITHM = "route-algorithm"
HARDWARE = "route-hardware"               # radio cpu / gpu
GRAPH_API = "route-graph-api"
SEARCH_BUFFER = "route-search-buffer"
SEARCH_BUFFER_INFO = "route-search-buffer-info"
IGNORE_MAX_COST = "route-ignore-max-cost"
PAIRWISE = "route-pairwise"
SIMPLIFY = "route-simplify"
SIMPLIFY_TOL = "route-simplify-tol"
ROUTE_RASTER = "route-raster"             # dropdown: which raster layer to route on
RUN_ROUTING_BTN = "run-routing-btn"
ROUTING_STATUS = "routing-status"
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
EDIT_MODE = "edit-mode"                   # (retired) legacy edit-mode radio id
POINTS_APPLY_BTN = "points-apply-btn"     # grid rows -> variant / draft
POINTS_REMOVE_BTN = "points-remove-btn"   # delete selected grid rows
NEW_ROUTE_BTN = "new-route-btn"           # deselect -> grid shows the draft
EDIT_COST_READOUT = "edit-cost-readout"
AUTO_REFRESH = "edit-auto-refresh"        # recompute on every edit (default on)
REFRESH_ROUTES_BTN = "refresh-routes-btn" # recompute the staged/edited routes
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
LAYER_TABLE_HEIGHT = "layer-table-height"   # (retired) legacy resize slider
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
