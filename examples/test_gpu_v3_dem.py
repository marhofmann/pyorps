"""Run GPU V3 constrained pathfinding - real-world raster, no DEM.
Tests V3 routing on real data without the memory-intensive DEM/clearance.
"""

import time
from pyorps.graph.constrained_path_finder import ConstrainedPathFinder
from pyorps.core.infrastructure_profile import InfrastructureProfile

raster_path = r"./data/raster/modified_raster_for_distribution_grid_planning.tiff"

profile = InfrastructureProfile.load(r"../profiles/overhead_line_110kv.yaml")

route = {
    "source": (473609, 5607305),
    "target": (474200, 5606700),
    "buffer": 500,
}

for backend, label in [("raster_gpu_v3", "GPU V3"), ("cython", "Cython")]:
    for r in [4]:
        print(f"\n{'='*60}")
        print(f"Route 1 - 110kV - R{r} - {label}")
        print(f"{'='*60}")

        t0 = time.perf_counter()
        try:
            cpf = ConstrainedPathFinder(
                dataset_source=raster_path,
                source_coords=route["source"],
                target_coords=route["target"],
                profile=profile,
                graph_api=backend,
                neighborhood_str=f"r{r}",
                search_space_buffer_m=route["buffer"],
            )
        except Exception as e:
            print(f"Init failed: {e}")
            continue
        t_init = time.perf_counter() - t0

        t0 = time.perf_counter()
        try:
            result = cpf.find_route()
        except Exception as e:
            print(f"Route failed: {e}")
            continue
        t_route = time.perf_counter() - t0

        print(f"Init time:  {t_init:.2f}s")
        print(f"Route time: {t_route:.2f}s")

        if result.path_geometry is None:
            print("No feasible route found")
        else:
            print(f"Towers: {result.n_towers}")
            print(f"Types: {result.tower_type_counts}")
            print(f"Terrain cost:  {result.total_terrain_cost:>12,.0f} EUR")
            print(f"Tower cost:    {result.total_tower_cost:>12,.0f} EUR")
            print(f"TOTAL COST:    {result.total_cost:>12,.0f} EUR")
            print(f"Spans: {result.min_span_actual_m:.0f}-{result.max_span_actual_m:.0f} m")

            crs = cpf.raster_handler.raster_dataset.crs
            tower_gdf = result.towers_to_geodataframe(crs=crs)
            print(tower_gdf[["tower_id", "tower_type",
                              "turn_angle_deg", "span_to_previous_m"]].to_string())
