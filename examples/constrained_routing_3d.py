"""Constrained overhead-line routing with a DEM and a DSM (3D) for three voltage levels and two routes.

Needs data that is not shipped (all git-ignored):

- the distribution-grid cost raster that ``prepare_data_for_distribution_grid_planning.ipynb`` writes to
  ``data/raster/modified_raster_for_distribution_grid_planning.tiff``,
- the elevation model (DGM1) and the surface model (DOM1) as ``data/shapes/additional_data/dgm1_merged.tif`` and
  ``data/shapes/additional_data/dom1_merged.tif``.

Results (route and tower GeoJSON files) are written to ``examples/output/`` (git-ignored).
"""
from pathlib import Path

from pyorps.core.infrastructure_profile import InfrastructureProfile
from pyorps.graph.constrained_path_finder import ConstrainedPathFinder

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "output"

raster_path = HERE / "data" / "raster" / "modified_raster_for_distribution_grid_planning.tiff"
dem_path = HERE / "data" / "shapes" / "additional_data" / "dgm1_merged.tif"
dsm_path = HERE / "data" / "shapes" / "additional_data" / "dom1_merged.tif"

# --- Profiles ---
profile_configs = {
    "110kV": HERE.parent / "profiles" / "overhead_line_110kv.yaml",
    "220kV": HERE.parent / "profiles" / "overhead_line_220kv.yaml",
    "380kV": HERE.parent / "profiles" / "overhead_line_380kv.yaml",
}

# --- Routes ---
routes = {
    "Route 1 (short)": {
        "source": (473609, 5607305),
        "target": (474443, 5606872),
        "buffer": 500,
        "r_values": [4, 5, 6],
    },
    "Route 2 (forest)": {
        "source": (473000, 5608000),
        "target": (475500, 5606000),
        "buffer": 1000,
        "r_values": [3, 4, 5, 6],
    },
}


def run_route(route_name, route_cfg, profile_name, profile_path):
    profile = InfrastructureProfile.load(str(profile_path))
    profile.tower_area_cost_mode = "exact"

    for r in route_cfg["r_values"]:
        print(f"\n{'='*60}")
        print(f"{route_name} — {profile_name} — R{r}")
        print(f"{'='*60}")

        cpf = ConstrainedPathFinder(
            dataset_source=str(raster_path),
            source_coords=route_cfg["source"],
            target_coords=route_cfg["target"],
            profile=profile,
            graph_api="cython",
            neighborhood_str=f"r{r}",
            search_space_buffer_m=route_cfg["buffer"],
            dem=str(dem_path),
            dsm=str(dsm_path),
        )

        if cpf._obstacle_data is not None and r == route_cfg["r_values"][0]:
            obs = cpf._obstacle_data
            print(f"Obstacles: max={obs.max():.0f}m, mean={obs.mean():.1f}m, "
                  f">{15}m: {(obs > 15).mean() * 100:.1f}%")

        result = cpf.find_route()
        if result.path_geometry is None:
            print("No feasible route found")
            continue

        print(f"Towers: {result.n_towers}")
        print(f"Types: {result.tower_type_counts}")
        print(f"Terrain cost:  {result.total_terrain_cost:>12,.0f} EUR")
        print(f"Tower cost:    {result.total_tower_cost:>12,.0f} EUR")
        print(f"TOTAL COST:    {result.total_cost:>12,.0f} EUR")
        print(f"Spans: {result.min_span_actual_m:.0f}–{result.max_span_actual_m:.0f} m")

        crs = cpf.raster_handler.raster_dataset.crs
        tower_gdf = result.towers_to_geodataframe(crs=crs)
        if "height_m" in tower_gdf.columns:
            heights = tower_gdf["height_m"].dropna()
            print(f"Heights: {sorted(heights.unique())}")

        tag = f"{profile_name}_{route_name.split()[0]}_{route_name.split()[1].strip('()')}_r{r}"
        cpf.save_paths(str(OUTPUT_DIR / f"route_{tag}_ta.geojson"))
        tower_gdf.to_file(str(OUTPUT_DIR / f"towers_{tag}_ta.geojson"), driver="GeoJSON")
        print(tower_gdf[["tower_id", "tower_type", "height_m",
                          "turn_angle_deg", "span_to_previous_m"]].to_string())


def main():
    missing = [str(p) for p in (raster_path, dem_path, dsm_path) if not p.exists()]
    if missing:
        listing = "\n  ".join(missing)
        raise SystemExit(f"Input data missing (see the module docstring):\n  {listing}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Run all combinations ---
    for profile_name, profile_path in profile_configs.items():
        for route_name, route_cfg in routes.items():
            run_route(route_name, route_cfg, profile_name, profile_path)


if __name__ == "__main__":
    main()
