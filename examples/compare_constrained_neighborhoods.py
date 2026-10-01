"""Compare constrained path results across R1-R5 neighborhoods.

Outputs a deterministic fingerprint per neighborhood for regression testing.
"""
import sys
import time
import hashlib
import numpy as np

sys.path.insert(0, r"<local-path>")

from pyorps.graph.constrained_path_finder import ConstrainedPathFinder

source = (473609, 5607305)
target = (474443, 5606872)
raster_path = r"./data/raster/modified_raster_for_distribution_grid_planning.tiff"
profile_path = r"../profiles/overhead_line_380kv.yaml"

results = {}

for r in range(1, 6):
    nb = f"r{r}"
    t0 = time.perf_counter()
    cpf = ConstrainedPathFinder(
        dataset_source=raster_path,
        source_coords=source,
        target_coords=target,
        profile=profile_path,
        graph_api="cython",
        neighborhood_str=nb,
        search_space_buffer_m=500,
    )
    result = cpf.find_route()
    elapsed = time.perf_counter() - t0

    if result.path_geometry is None:
        print(f"{nb}: NO PATH FOUND ({elapsed:.3f}s)")
        results[nb] = None
        continue

    # Deterministic fingerprint from path cells and tower cells
    path_cells = result._path_cells if hasattr(result, '_path_cells') else None
    tower_cells = result._tower_cells if hasattr(result, '_tower_cells') else None

    # Use tower geodataframe for stable comparison
    tower_gdf = result.towers_to_geodataframe(crs=cpf.raster_handler.raster_dataset.crs)
    tower_coords = [(round(g.x, 3), round(g.y, 3)) for g in tower_gdf.geometry]

    n_path = len(result.path_geometry.coords) if result.path_geometry else 0
    n_towers = result.n_towers
    terrain_cost = result.total_terrain_cost
    tower_cost = result.total_tower_cost
    total_cost = terrain_cost + tower_cost

    # Hash path geometry for exact comparison
    coords = list(result.path_geometry.coords)
    coord_str = "|".join(f"{x:.3f},{y:.3f}" for x, y in coords)
    path_hash = hashlib.md5(coord_str.encode()).hexdigest()[:12]

    results[nb] = {
        "n_path": n_path,
        "n_towers": n_towers,
        "terrain_cost": terrain_cost,
        "tower_cost": tower_cost,
        "total_cost": total_cost,
        "tower_coords": tower_coords,
        "path_hash": path_hash,
        "elapsed": elapsed,
    }

    print(f"{nb}: path={n_path} towers={n_towers} "
          f"terrain={terrain_cost:.0f} tower={tower_cost:.0f} "
          f"total={total_cost:.0f} hash={path_hash} "
          f"time={elapsed:.3f}s")

print("\n=== SUMMARY ===")
for nb, r in results.items():
    if r is None:
        print(f"  {nb}: NO PATH")
    else:
        print(f"  {nb}: path={r['n_path']} towers={r['n_towers']} "
              f"total={r['total_cost']:.0f} hash={r['path_hash']} "
              f"time={r['elapsed']:.3f}s")
