"""
PYORPS GUI: server-side project state.

A single :class:`ProjectState` instance is the source of truth for every heavy
object in a GUI session — GeoDataFrames, served rasters, PathFinders, routes.
Dash ``dcc.Store``s only ever hold the small, JSON-serializable *view* of this
state (:meth:`ProjectState.layers_view`); callbacks mutate the state and return
the refreshed view, and render callbacks derive map layers from it.
"""
from __future__ import annotations

import contextlib
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


#: how long :meth:`TileManager.stop_server` lets the tile requests in flight finish
STOP_SERVER_WAIT_S = 30.0
#: how long it then waits for the server thread to end
_STOP_SERVER_JOIN_S = 10.0


def _stop_server_thread(server: Any, wait_s: float) -> None:
    """Stop one ``server_thread.ServerThread`` without waiting for ever (see ``TileManager.stop_server``)."""
    import time

    uvicorn_server = getattr(getattr(server, "srv", None), "server", None)
    if uvicorn_server is None or not hasattr(uvicorn_server, "force_exit"):
        server.shutdown()                    # a WSGI server: its own stop does not wait for connections
        return
    tasks = getattr(getattr(uvicorn_server, "server_state", None), "tasks", None)
    deadline = time.monotonic() + max(0.0, float(wait_s))
    while tasks and time.monotonic() < deadline:  # the requests in flight finish first
        time.sleep(0.05)
    uvicorn_server.force_exit = True          # then: no waiting for idle keep-alive connections
    uvicorn_server.should_exit = True
    server.join(_STOP_SERVER_JOIN_S)
    if server.is_alive():
        # its __del__ would join again without a limit (at interpreter exit): make that a no-op
        server._lts_initialized = False  # pylint: disable=protected-access
        _LOG.warning("the shared tile server did not stop within %.0f s", _STOP_SERVER_JOIN_S)


class TileManager:
    """The one owner of the live localtileserver TileClients of a project.

    A client is registered when its raster layer is added and released exactly
    once when the layer goes away (removed, replaced under the same id, the
    project cleared or replaced, the state shut down). Releasing a client does
    two things: ``client.shutdown()`` (a no-op for the shared default server of
    the installed localtileserver, which serves every raster; older versions run
    one server per client) and closing the client's raster reader, which is
    what keeps the ``.tif`` open (and locked on Windows) until then.

    A release that fails never breaks the user's action: it is written to the
    logbook as a warning and kept as a notification box for the callbacks that
    can show one (:meth:`pop_notices`).
    """

    #: failures of releases not yet shown to the user (shared by all instances:
    #: ``shutdown_client`` is static, used by the project loader too)
    _failures: list[dict] = []
    _failures_lock = threading.Lock()
    _MAX_FAILURES = 20

    def __init__(self) -> None:
        self._clients: dict[str, Any] = {}

    def register(self, layer_id: str, tile_layer: Any) -> Any:
        """Own ``tile_layer``'s client under ``layer_id``.

        Returns a DIFFERENT client that was registered under the same id (the
        caller must release it), else ``None``.
        """
        client = tile_layer.tile_client
        previous = self._clients.get(layer_id)
        self._clients[layer_id] = client
        return previous if previous is not client else None

    @staticmethod
    def shutdown_client(client: Any) -> bool:
        """Release one client (server stop + reader close), once.

        Returns False when there was nothing to do (no client, or already
        released). A step that fails is reported (see the class docstring) and
        the remaining steps still run.
        """
        if client is None or getattr(client, "_pyorps_released", False):
            return False
        with contextlib.suppress(AttributeError):    # a slotted stand-in: no flag
            client._pyorps_released = True
        problems: list[tuple[str, BaseException]] = []
        shutdown = getattr(client, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception as exc:  # pylint: disable=broad-exception-caught  # reported below, never raised
                problems.append(("stop the tile server", exc))
        try:
            reader = getattr(client, "reader", None)
            close = getattr(reader, "close", None)
            if callable(close):
                close()
        except Exception as exc:  # pylint: disable=broad-exception-caught  # reported below, never raised
            problems.append(("close the raster reader", exc))
        for step, exc in problems:
            TileManager._report_failure(client, step, exc)
        return True

    @staticmethod
    def _report_failure(client: Any, step: str, exc: BaseException) -> None:
        """Write one failed release to the logbook and keep it for the UI."""
        import traceback

        from .services import errors, logbook

        source = getattr(client, "filename", None) or "a raster"
        notice = errors.Notice(
            severity="warning",
            title="A raster layer's tile server could not be released cleanly",
            meaning=f"The GUI could not {step} for {source} "
                    f"({type(exc).__name__}: {exc}).",
            impact="The raster file may stay open (locked on Windows) until "
                   "the GUI is closed. Nothing else is affected.",
            fix="Close the GUI before you delete or overwrite that file.",
            details="".join(traceback.format_exception(
                type(exc), exc, exc.__traceback__)))
        _LOG.warning("tile client release failed (%s): %s", step, exc,
                     exc_info=exc)
        try:
            logbook.log_notice({**notice.to_dict(), "details": notice.details})
        except Exception:  # pylint: disable=broad-exception-caught  # the log is best effort; the warning above stands
            _LOG.warning("tile client failure could not be written to the logbook",
                         exc_info=True)
        with TileManager._failures_lock:
            TileManager._failures.append(notice.to_dict())
            del TileManager._failures[:-TileManager._MAX_FAILURES]

    @staticmethod
    def pop_notices() -> list[dict]:
        """The failed releases since the last call, as notice dicts (cleared)."""
        with TileManager._failures_lock:
            out, TileManager._failures[:] = list(TileManager._failures), []
        return out

    @staticmethod
    def stop_server(wait_s: float = STOP_SERVER_WAIT_S) -> None:
        """Force-stop the shared default tile server (process exit only).

        ``client.shutdown()`` leaves the shared server running for the other
        layers; this ends it. Safe to call when it is not running.

        The library's own stop (``ServerManager.shutdown_server``) is uvicorn's
        graceful shutdown joined without a limit, and that waits until every
        open connection has closed: a keep-alive connection that is never
        closed made it wait for ever (seen at the end of the browser tests).
        So the requests in flight get up to ``wait_s`` seconds to finish (a
        tile render reads the raster with GDAL in a worker thread; cutting it
        off at process exit can block the exit itself), then the server stops
        without waiting for idle connections, and the join is bounded too.
        """
        try:
            from server_thread import ServerManager
        except ImportError:  # pragma: no cover  # localtileserver absent
            return
        try:
            if ServerManager.is_server_live("default"):
                _stop_server_thread(ServerManager.pop_server("default"), wait_s)
        except Exception as exc:  # pylint: disable=broad-exception-caught  # reported, never raised
            TileManager._report_failure(None, "stop the shared tile server", exc)

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
        # round 14: the settings of the newer cards, one dict per card
        # ("objective", "cost_field", "corridor", "constrained"). Mirrored from
        # the controls, saved raw in the project manifest, put back into the
        # controls on open / reload; each card normalises its own key.
        self.gui_settings: dict[str, dict] = {}
        # background jobs by kind ("rasterize", "constrained", "cost_field",
        # "corridor"; services.jobs). Runtime only, never saved. The worker
        # thread only computes; the poll callback changes the state.
        self.jobs: dict[str, Any] = {}
        # the cost table of an opened project (project.json "cost_table"): the
        # cost editor is client-side, so a first paint reads it from here
        self.loaded_cost_table: dict = {}
        # shared-trench corridor graphs by segments-layer id, kept for saving
        self.corridor_graphs: dict[str, Any] = {}
        # set when "Open project" has just written the routing controls: the
        # backend the page was given. The algorithm list callback reads it once
        # so the opened algorithm survives the backend change (callbacks/routes)
        self.routing_restore_backend: str | None = None
        # unsaved work since the last project save (drives the close guard)
        self.dirty: bool = False
        # the project file the project was opened from or saved to (Open, Save,
        # the ``--project`` flag); None for a blank or new project. A page that
        # loads later names it instead of only counting the layers.
        self.project_path: str | None = None
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
            replaced = self.tiles.detach(layer.id) if tile is None else None
            if tile is not None:
                replaced = self.tiles.register(layer.id, tile)
            if replaced is not None:            # the layer id was in use before
                self._defer(lambda c=replaced: self.tiles.shutdown_client(c))
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
            for job in self.jobs.values():        # same for every background job
                cancel = getattr(job, "cancel", None)
                if cancel is not None:
                    cancel.set()
            self.jobs = {}
            self.gui_settings = {}
            self.loaded_cost_table = {}
            self.corridor_graphs = {}
            self.dirty = False                    # nothing left to lose
            self.project_path = None
            self.project_crs = DEFAULT_PROJECT_CRS

    def shutdown(self) -> None:
        """Release every tile client (the shared server itself stays up: see
        :meth:`TileManager.stop_server`, called once at process exit)."""
        self.tiles.shutdown_all()
