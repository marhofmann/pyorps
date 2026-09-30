"""
PYORPS GUI callbacks: constrained overhead-line routing (R13, Section 20).

The profile is edited as YAML (seeded from the shipped 110/220/380 kV /
rural-road presets), validated live before the run (21.5), and experimental
backends are gated behind an explicit flag (F10). Results land as a route
layer plus a towers layer (rotated-footprint polygons), with the 2-D
terrain-cost caveat (F9) always visible.
"""
from __future__ import annotations

from dash import Input, Output, State, no_update
from dash.exceptions import PreventUpdate

from .. import ids
from ..services import constrained, geo, routing
from ..services.errors import Notice, guard, success
from .routes import add_route_layer


def _parse_profile_text(text: str) -> dict:
    import yaml

    config = yaml.safe_load(text or "")
    if not isinstance(config, dict) or not config:
        raise ValueError("The profile editor is empty or not a YAML/JSON "
                         "mapping.")
    return config


def register(app, state) -> None:
    @app.callback(Output(ids.OHL_COLLAPSE, "is_open"),
                  Output(ids.OHL_PROFILE_PRESET, "options"),
                  Input(ids.OHL_ENABLE, "value"))
    def toggle(enabled):
        options = [{"label": name, "value": path}
                   for name, path in constrained.shipped_profiles().items()]
        return bool(enabled), options

    @app.callback(Output(ids.OHL_PROFILE_TEXT, "value"),
                  Input(ids.OHL_PROFILE_PRESET, "value"),
                  prevent_initial_call=True)
    def load_preset(path):
        if not path:
            raise PreventUpdate
        from pathlib import Path

        return Path(path).read_text(encoding="utf-8")

    @app.callback(Output(ids.OHL_PROFILE_TEXT, "value",
                         allow_duplicate=True),
                  Output(ids.NOTICES, "data", allow_duplicate=True),
                  Input(ids.OHL_PROFILE_LOAD_BTN, "n_clicks"),
                  State(ids.OHL_PROFILE_PATH, "value"),
                  State(ids.NOTICES, "data"),
                  prevent_initial_call=True)
    def load_profile_file(n_clicks, path, notices):
        notices = list(notices or [])
        if not n_clicks:
            raise PreventUpdate
        config, notices = guard(constrained.load_profile_dict, path,
                                notices=notices)
        if config is None:
            return no_update, notices
        import yaml

        return yaml.safe_dump(config, sort_keys=False,
                              allow_unicode=True), notices

    @app.callback(Output(ids.NOTICES, "data", allow_duplicate=True),
                  Input(ids.OHL_PROFILE_SAVE_BTN, "n_clicks"),
                  State(ids.OHL_PROFILE_PATH, "value"),
                  State(ids.OHL_PROFILE_TEXT, "value"),
                  State(ids.NOTICES, "data"),
                  prevent_initial_call=True)
    def save_profile_file(n_clicks, path, text, notices):
        from ..services import project_io

        notices = list(notices or [])
        if not n_clicks:
            raise PreventUpdate
        config, notices = guard(_parse_profile_text, text, notices=notices)
        if config is None:
            return notices
        problem = constrained.validate_profile(config)
        if problem is not None:
            notices.append(problem.to_dict())
            return notices
        result, notices = guard(project_io.save_profile, config, path,
                                notices=notices)
        if result is not None:
            notices.append(success("Profile saved",
                                   meaning=f"Written to {result}."))
        return notices

    # constrained routing shares the main Run button; the toggle decides which
    # runner acts, and waypoints become forced towers (task 46)
    @app.callback(
        Output(ids.LAYERS_VIEW, "data", allow_duplicate=True),
        Output(ids.OHL_STATUS, "children"),
        Output(ids.EDIT_ROUTE_SELECT, "options", allow_duplicate=True),
        Output(ids.EDIT_ROUTE_SELECT, "value", allow_duplicate=True),
        Output(ids.NOTICES, "data", allow_duplicate=True),
        Input(ids.RUN_ROUTING_BTN, "n_clicks"),
        State(ids.OHL_ENABLE, "value"),
        State(ids.ROUTE_DRAFT, "data"),
        State(ids.ROUTE_RASTER, "value"),
        State(ids.OHL_PROFILE_TEXT, "value"),
        State(ids.OHL_BACKEND, "value"),
        State(ids.OHL_EXPERIMENTAL, "value"),
        State(ids.NEIGHBORHOOD, "value"),
        State(ids.SEARCH_BUFFER, "value"),
        State(ids.OHL_DEM, "value"),
        State(ids.OHL_DSM, "value"),
        State(ids.BUILD_POINTS_TABLE, "rowData"),
        State(ids.NOTICES, "data"),
        prevent_initial_call=True)
    def run_constrained(n_clicks, constrained_on, draft, raster_layer_id,
                        profile_text, backend, experimental, neighborhood,
                        search_buffer, dem, dsm, points_rows, notices):
        notices = list(notices or [])
        if not n_clicks or not constrained_on:
            raise PreventUpdate          # the unconstrained runner handles it

        stop = (no_update, no_update, no_update, no_update)
        raster_layer = state.get(raster_layer_id) if raster_layer_id else None
        if raster_layer is None or raster_layer.kind != "raster":
            notices.append(Notice(
                severity="warning", title="Pick a cost raster first",
                meaning="Constrained routing needs a raster layer.",
                impact="No route was computed.",
                fix="Build/load a raster and select it under 'Route on "
                    "raster'.", focus_id=ids.TAB_RASTER).to_dict())
            return (*stop, notices)
        draft = draft or {}
        sources = [(p["x"], p["y"]) for p in draft.get("sources", [])]
        targets = [(p["x"], p["y"]) for p in draft.get("targets", [])]
        waypoints = [(p["x"], p["y"]) for p in draft.get("waypoints", [])]
        if not sources or not targets:
            notices.append(Notice(
                severity="warning", title="Set a source and a target first",
                meaning="Constrained routing uses the same source/target/"
                        "waypoint points; each waypoint forces a tower.",
                impact="There is nothing to route yet.",
                fix="Place points with the +Source / +Target / +Waypoint "
                    "click modes.", focus_id=ids.TAB_ROUTES).to_dict())
            return (*stop, notices)
        points = [sources[0], *waypoints, targets[0]]

        config, notices = guard(_parse_profile_text, profile_text,
                                notices=notices)
        if config is None:
            return (*stop, notices)
        problem = constrained.validate_profile(config)     # 21.5 shift-left
        if problem is not None:
            notices.append(problem.to_dict())
            return (*stop, notices)
        import rasterio
        with rasterio.open(raster_layer.meta.get("source_path")) as src:
            resolution_m = abs(src.transform.a)
        perf = constrained.validate_span_bin_vs_resolution(config,
                                                           resolution_m)
        if perf is not None:
            notices.append(perf.to_dict())
        backend, gate_notice = constrained.resolve_backend(
            backend, bool(experimental))                   # F10
        if gate_notice is not None:
            notices.append(gate_notice.to_dict())

        # up-front effort warning (constrained runs are the heaviest; the
        # extended tower state multiplies the plain-routing estimate)
        try:
            est = routing.estimate_routing(
                raster_layer.meta.get("source_path"), sources=[sources[0]],
                targets=[targets[0]], waypoints=waypoints,
                neighborhood=neighborhood or "r2",
                search_buffer_m=(float(search_buffer)
                                 if search_buffer else None))
        except Exception:
            est = None
        if est is not None and routing.needs_confirmation(est):
            notices.append(Notice(
                severity="warning", title="Heavy constrained run ahead",
                meaning=f"~{est['total_cells']:,.0f} window cells at "
                        f"{est['resolution_m']:g} m; the tower state "
                        "multiplies plain-routing effort "
                        f"(≥ {est['est_seconds']:,.0f} s, "
                        f"≥ {est['est_memory_mb']:,.0f} MB expected).",
                impact="The UI blocks until this run finishes.",
                fix="Smaller search buffer, coarser raster resolution, "
                    "lower neighborhood (r1), or a GPU backend reduce "
                    "the effort.").to_dict())

        result, notices = guard(
            constrained.run_constrained_path,
            raster_layer.meta.get("source_path"), points=points,
            profile=config, backend=backend, neighborhood=neighborhood or "r2",
            search_buffer_m=(float(search_buffer) if search_buffer else None),
            dem=dem, dsm=dsm, notices=notices)
        if result is None:
            notices.append(Notice(
                severity="error", title="No route could be found",
                meaning="A constrained segment found no feasible tower-by-"
                        "tower path.",
                impact="No overhead-line route was produced.",
                fix="Loosen the profile (angles/spans), enlarge the search "
                    "buffer, or move the points.",
                focus_id=ids.TAB_ROUTES).to_dict())
            return (*stop, notices)

        from ..services.cost import RouteCost

        summary = result["summary"]
        line = result["line"]
        route_cost = RouteCost(
            total_length=float(result["total_length"]),
            total_cost=float(result["total_cost"]),
            total_cell_cost=float(result["total_cell_cost"]),
            geodesic_length_m=float(line.length))
        crs = result["crs"]
        wp_names = [str(r.get("name", "")) for r in (points_rows or [])
                    if r.get("kind") == "waypoint"]
        control_points = ([list(sources[0])] + [list(w) for w in waypoints]
                          + [list(targets[0])])
        params = {"constrained": True, "profile": config, "backend": backend,
                  "raster_layer_id": raster_layer.id,
                  "neighborhood": neighborhood or "r2"}
        route_layer = add_route_layer(
            state, line=line, route_cost=route_cost, params=params,
            control_points=control_points, crs=crs,
            name=f"OHL {state.next_route_name()}", origin="built",
            waypoint_names=wp_names)
        route_layer.meta["constrained_summary"] = summary

        if result["towers"] is not None and not result["towers"].empty:
            state.add_layer(
                f"{route_layer.name} towers", "vector", gdf=result["towers"],
                crs=crs, geojson=geo.gdf_to_wgs84_geojson(result["towers"]),
                style={"color": "#7b3294", "weight": 2, "fillOpacity": 0.5},
                meta={"towers_for": route_layer.id})

        notices.append(success(
            f"Overhead line routed: {summary['n_towers']} towers"
            + (f" ({len(waypoints)} forced at waypoints)" if waypoints else ""),
            meaning=(f"Tower cost {summary['total_tower_cost']:,.0f}, angle "
                     f"penalties {summary['total_angle_penalty_cost']:,.0f}, "
                     f"max turn {summary['max_turn_angle_deg']:.1f} deg.")))
        notices.append(Notice(
            severity="info", title="Terrain cost shown is 2-D",
            meaning="The optimizer used 3-D penalties but reports the plain "
                    "2-D terrain cost (known limitation).",
            impact="The terrain-cost number understates the optimized cost "
                   "(F9).",
            fix="Treat it as indicative; compare routes by length + tower "
                "cost too.").to_dict())
        status = (f"OHL: {summary['n_towers']} towers, spans "
                  f"{summary['spans_min_max_avg'][0]:.0f}-"
                  f"{summary['spans_min_max_avg'][1]:.0f} m "
                  f"(avg {summary['spans_min_max_avg'][2]:.0f})")
        route_options = [{"label": ly.name, "value": ly.id}
                         for ly in state.layers_of_kind("route")]
        state.active_route_id = route_layer.id
        return (state.layers_view(), status, route_options,
                route_layer.id, notices)
