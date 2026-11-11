"""
PYORPS GUI: server-side project state.

A single :class:`ProjectState` instance is the source of truth for every heavy
object in a GUI session — GeoDataFrames, served rasters, PathFinders, routes.
Dash ``dcc.Store``s only ever hold the small, JSON-serializable *view* of this
state (:meth:`ProjectState.layers_view`); callbacks mutate the state and return
the refreshed view, and render callbacks derive map layers from it.
"""
from __future__ import annotations

import itertools
import logging
import tempfile
import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_LOG = logging.getLogger(__name__)

DEFAULT_PROJECT_CRS = "EPSG:25832"

LAYER_KINDS = ("vector", "raster", "route", "study_area", "wms")

#: default paint-order band per kind (background → foreground): study area,
#: vector datasets, overlays (WMS), raster datasets, routes. A NEW layer is
#: inserted on top of its own band — so a fresh raster never covers the
#: routes, and the study-area frame never covers anything.
KIND_Z_BAND = {"study_area": 0, "vector": 1, "wms": 2, "raster": 3,
               "route": 4}


@dataclass
class Layer:
    """One display layer: a vector dataset, served raster, route, or study area.

    Exactly one payload is set depending on ``kind``: ``gdf`` for vector /
    route / study_area layers (in the routing CRS), ``tile`` for raster layers
    (a served :class:`~pyorps.gui.services.tiles.RasterTileLayer`). ``geojson``
    caches the WGS84 display data so renders don't reproject every time.
    """

    id: str
    name: str
    kind: str                     # "vector" | "raster" | "route" | "study_area"
    visible: bool = True
    z: int = 0                    # draw order (higher = painted on top)
    crs: Any = None
    gdf: Any = None               # GeoDataFrame payload (vector/route/study_area)
    tile: Any = None              # RasterTileLayer payload (raster)
    geojson: dict | None = None   # cached WGS84 GeoJSON for display
    style: dict = field(default_factory=dict)
    opacity: float = 0.7          # raster overlay opacity
    meta: dict = field(default_factory=dict)

    def __setattr__(self, name: str, value: Any) -> None:
        # bump a revision every time ``geojson`` is (re)assigned so URL-served
        # layers get a fresh cache-buster (perf: dl.GeoJSON(url=…)). Runs during
        # __init__ too (geojson defaults to None) → every layer starts at rev 0.
        super().__setattr__(name, value)
        if name == "geojson":
            super().__setattr__("_gj_rev", getattr(self, "_gj_rev", -1) + 1)

    @property
    def geojson_rev(self) -> int:
        return getattr(self, "_gj_rev", 0)

    def to_ui(self) -> dict:
        """The small serializable dict the client's layer panel needs."""
        return {"id": self.id, "name": self.name, "kind": self.kind,
                "visible": self.visible, "z": self.z}


@dataclass
class CostLayerConfig:
    """One entry of the cost model: the base dataset or an ordered modifier."""

    dataset_id: str                          # Layer.id of the vector dataset
    feature_keys: tuple[str, ...] = ()       # e.g. ("nutzart", "bez")
    assumptions: Any = None                  # nested cost dict | scalar (modifier)
    role: str = "base"                       # "base" | "multiply" | "override"
    geometry_buffer_m: float = 1.0
    preprocessing: str | None = None         # named preprocessor id
    preprocessing_params: dict = field(default_factory=dict)
    ignore_value: float | None = None        # modifier: cells to leave untouched
    zone_field: str | None = None            # modifier: per-zone factor column
    forbidden_zone: str | None = None
    forbidden_value: int | None = None

    def to_ui(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "feature_keys": list(self.feature_keys),
            "role": self.role,
            "geometry_buffer_m": self.geometry_buffer_m,
            "preprocessing": self.preprocessing,
            "zone_field": self.zone_field,
        }


class TileManager:
    """Tracks live localtileserver TileClients and shuts them down on removal."""

    def __init__(self) -> None:
        self._clients: dict[str, Any] = {}

    def register(self, layer_id: str, tile_layer: Any) -> None:
        self._clients[layer_id] = tile_layer.tile_client

    @staticmethod
    def shutdown_client(client: Any) -> None:
        """Stop one tile server (best effort: a failed stop never blocks)."""
        shutdown = getattr(client, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:  # pragma: no cover  # nosec B110  # pylint: disable=broad-exception-caught
                # best-effort cleanup
                pass

    def detach(self, layer_id: str) -> Any:
        """Forget a layer's client and hand it back, WITHOUT stopping it."""
        return self._clients.pop(layer_id, None)

    def detach_all(self) -> list[Any]:
        """Forget every client and hand them back, WITHOUT stopping them."""
        return [c for c in (self.detach(lid) for lid in list(self._clients))
                if c is not None]

    def release(self, layer_id: str) -> None:
        client = self.detach(layer_id)
        if client is not None:
            self.shutdown_client(client)

    def shutdown_all(self) -> None:
        for client in self.detach_all():
            self.shutdown_client(client)


class ProjectState:
    """In-memory session state; one instance per app, captured by closures.

    The server runs several request threads against this one object, so the
    methods that change the layer registry (add / remove / reorder / clear ...)
    hold :attr:`lock` (re-entrant) for their short, in-memory work and read
    helpers iterate over a snapshot taken under the same lock. Slow cleanup
    (stopping tile servers, closing search sessions) is queued while the lock
    is held and runs AFTER it is released, so the lock never covers I/O or
    routing. :meth:`locked` is the context manager for a multi-step change
    that has to be atomic (see ``project_io.load_project``).
    """

    def __init__(self, work_dir: str | Path | None = None) -> None:
        self.lock = threading.RLock()
        self._lock_depth = 0
        self._deferred: list[Callable[[], None]] = []
        self.study_area: dict | None = None      # WGS84 GeoJSON polygon geometry
        # WKTs of every shape ever used as a study area — so the shared draw
        # tool's study-area rectangle is never mistaken for a cost polygon.
        self.study_area_geoms: set[str] = set()
        # WKTs of manual cost polygons removed from the list — kept out of the
        # layer even though their leaflet-draw shape lingers on the map.
        self.manual_excluded_geoms: set[str] = set()
        self.project_crs: Any = DEFAULT_PROJECT_CRS
        self.layers: dict[str, Layer] = {}
        self.cost_config: list[CostLayerConfig] = []
        # registered cost tables (seeded / imported) so the Raster tab can
        # pair any compatible vector dataset with any table:
        # {table_id: {"name", "dataset_id", "feature_keys", "rows"}}
        self.cost_tables: dict[str, dict] = {}
        self.active_route_id: str | None = None
        self._host_sig: tuple | None = None      # layer-host render fingerprint
        self.tiles = TileManager()
        # transient caches -------------------------------------------------
        self.finders: dict[str, Any] = {}        # raster layer id -> PathFinder
        self.search_sessions: dict[str, Any] = {}  # route id -> SearchSession
        # one-step undo of a route edit: edited route id -> the route layer it
        # replaced. Memory only (never saved with the project). The snapshot is
        # just the route's Layer (line + metadata); the sessions and finders
        # live in the two dicts above and are NOT kept with it.
        self.route_undo: dict[str, Layer] = {}
        self.route_counter = itertools.count(1)
        self.run_counter = itertools.count(1)    # routing runs -> group labels
        # heavy-run gate: params awaiting the user's "accept waiting", and
        # the in-flight background job (services.routing_job)
        self.pending_routing: dict | None = None
        self.routing_job: Any = None
        # the routing settings of the page (backend, algorithm, k, buffer, ...),
        # mirrored from the controls; saved with the project and put back into
        # the controls when the page is built again (services.routing)
        self.routing_settings: dict[str, Any] = {}
        # the same for the rasterize controls (resolution, fill value, dtype,
        # geometry buffer, thin-feature switches, crop; services.rasterize)
        self.raster_settings: dict[str, Any] = {}
        # set when "Open project" has just written the routing controls: the
        # backend the page was given. The algorithm list callback reads it once
        # so the opened algorithm survives the backend change (callbacks/routes)
        self.routing_restore_backend: str | None = None
        # unsaved work since the last project save (drives the close guard)
        self.dirty: bool = False
        self.work_dir = Path(work_dir) if work_dir else Path(
            tempfile.mkdtemp(prefix="pyorps_gui_"))
        self.work_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------- locking
    @contextmanager
    def locked(self) -> Iterator[ProjectState]:
        """Hold the state lock for a multi-step change (re-entrant).

        Cleanup queued by :meth:`_defer` while the lock was held (tile
        servers to stop, search sessions to close) runs once the OUTERMOST
        ``locked`` block has released it, so the lock itself never covers
        slow work.
        """
        pending: list[Callable[[], None]] = []
        try:
            with self.lock:
                self._lock_depth += 1
                try:
                    yield self
                finally:
                    self._lock_depth -= 1
                    if self._lock_depth == 0:
                        pending, self._deferred = self._deferred, []
        finally:
            for job in pending:           # the lock is released here
                try:
                    job()
                except Exception:  # pylint: disable=broad-exception-caught  # best-effort cleanup, never blocks a change
                    _LOG.warning("project-state cleanup failed", exc_info=True)

    def _defer(self, job: Callable[[], None]) -> None:
        """Queue slow cleanup to run after the lock is released (lock held)."""
        self._deferred.append(job)

    @staticmethod
    def _close_session(session: Any) -> None:
        close = getattr(session, "close", None)
        if close is not None:
            close()

    # ------------------------------------------------------------- layer CRUD
    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:12]

    def add_layer(self, name: str, kind: str, *, gdf: Any = None,
                  tile: Any = None, crs: Any = None, geojson: dict | None = None,
                  style: dict | None = None, meta: dict | None = None,
                  layer_id: str | None = None) -> Layer:
        # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
        if kind not in LAYER_KINDS:
            raise ValueError(f"Unknown layer kind {kind!r}; "
                             f"expected one of {LAYER_KINDS}.")
        layer = Layer(
            id=layer_id or self.new_id(), name=name, kind=kind,
            gdf=gdf, tile=tile, crs=crs, geojson=geojson,
            style=style or {}, meta=meta or {},
        )
        with self.locked():
            # default z: on top of the layer's own kind band (KIND_Z_BAND) but
            # below the first layer of a higher band — manual reorders later
            # are free to break the bands.
            band = KIND_Z_BAND.get(kind, 2)
            order = [ly.id for ly in self.ordered_layers()]
            insert_at = len(order)
            for i, lid in enumerate(order):
                if KIND_Z_BAND.get(self.layers[lid].kind, 2) > band:
                    insert_at = i
                    break
            order.insert(insert_at, layer.id)
            self.layers[layer.id] = layer
            for z, lid in enumerate(order):
                self.layers[lid].z = z
            if tile is not None:
                self.tiles.register(layer.id, tile)
            self.dirty = True
        return layer

    def remove_layer(self, layer_id: str) -> None:
        with self.locked():
            layer = self.layers.pop(layer_id, None)
            if layer is None:
                return
            client = self.tiles.detach(layer_id)
            if client is not None:
                self._defer(lambda: self.tiles.shutdown_client(client))
            self.finders.pop(layer_id, None)
            session = self.search_sessions.pop(layer_id, None)
            if session is not None:
                self._defer(lambda: self._close_session(session))
            self.route_undo.pop(layer_id, None)    # a gone route has no undo
            if self.active_route_id == layer_id:
                self.active_route_id = None
            self.dirty = True

    def keep_undo(self, layer_id: str, replaced: Layer) -> None:
        """Keep ``replaced`` as the one undo step of route ``layer_id``.

        A snapshot already held for ``layer_id`` is dropped: one step only.
        """
        with self.locked():
            self.route_undo[layer_id] = replaced

    def get(self, layer_id: str) -> Layer | None:
        return self.layers.get(layer_id)

    def reorder(self, ordered_ids: list[str]) -> None:
        """Assign z from list order (first = bottom, painted first)."""
        with self.locked():
            for z, layer_id in enumerate(ordered_ids):
                if layer_id in self.layers:
                    self.layers[layer_id].z = z

    def set_visible(self, layer_id: str, visible: bool) -> None:
        with self.locked():
            if layer_id in self.layers:
                self.layers[layer_id].visible = bool(visible)

    def rename(self, layer_id: str, name: str) -> None:
        with self.locked():
            if layer_id in self.layers and name:
                self.layers[layer_id].name = str(name)

    # ------------------------------------------------------------ layer views
    def ordered_layers(self) -> list[Layer]:
        """Layers in paint order (lowest z first).

        Iterates a snapshot taken under the lock, so a concurrent add/remove
        can neither raise "dictionary changed size during iteration" nor be
        seen half-applied (layer registered, z not yet renumbered).
        """
        with self.lock:
            snapshot = list(self.layers.values())
        return sorted(snapshot, key=lambda ly: ly.z)

    def layers_view(self) -> list[dict]:
        """The small serializable list driving the layer panel + renders."""
        return [ly.to_ui() for ly in self.ordered_layers()]

    def layers_of_kind(self, *kinds: str) -> list[Layer]:
        return [ly for ly in self.ordered_layers() if ly.kind in kinds]

    def next_route_name(self) -> str:
        return f"Route {next(self.route_counter)}"

    # ------------------------------------------------------------ route groups
    def route_groups(self) -> list[str]:
        """Distinct non-empty group labels across all route layers (order kept)."""
        seen: list[str] = []
        for ly in self.layers_of_kind("route"):
            group = (ly.meta or {}).get("group") or ""
            if group and group not in seen:
                seen.append(group)
        return seen

    def routes_in_group(self, group: str) -> list[Layer]:
        group = group or ""
        return [ly for ly in self.layers_of_kind("route")
                if ((ly.meta or {}).get("group") or "") == group]

    # -------------------------------------------------------------- lifecycle
    def clear(self) -> None:
        """Reset to a blank project (R1); shuts down all tile servers.

        The registry is emptied under the lock; the tile servers and search
        sessions of the old project are stopped after it is released.
        """
        with self.locked():
            for client in self.tiles.detach_all():
                self._defer(lambda c=client: self.tiles.shutdown_client(c))
            self.layers.clear()
            self.cost_config.clear()
            self.cost_tables.clear()
            self.study_area = None
            self.study_area_geoms.clear()
            self.manual_excluded_geoms.clear()
            self.active_route_id = None
            self._host_sig = None
            self.finders.clear()
            for session in list(self.search_sessions.values()):
                self._defer(lambda s=session: self._close_session(s))
            self.search_sessions.clear()
            self.route_undo.clear()
            self.pending_routing = None
            self.routing_settings = {}
            self.raster_settings = {}
            self.routing_restore_backend = None
            if self.routing_job is not None:      # a running job may not touch
                self.routing_job.cancel.set()     # the cleared state anyway
                self.routing_job = None
            self.dirty = False                    # nothing left to lose
            self.project_crs = DEFAULT_PROJECT_CRS

    def shutdown(self) -> None:
        self.tiles.shutdown_all()
