import random
import math
import rasterio as rio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyorps import PathFinder
from pyorps.core.exceptions import NoPathFoundError


def plot_benchmark_results(data_file, max_euclidean_distance=None):
    """
    Plot a grid of four subplots showing runtime results vs euclidean distance
    for different graph libraries and neighborhoods.

    Parameters:
    -----------
    data_file : str
        Path to the benchmark results CSV file

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure with a 2x2 subplot grid
    """
    # Load the data
    data = pd.read_csv(data_file)
    if max_euclidean_distance is not None:
        data = data.loc[data.euclidean_distance < max_euclidean_distance]

    # Create a neighborhood column based on the rank of number_of_edges
    # within each (euclidean_distance, graph_api) group
    data = data.sort_values(['euclidean_distance', 'graph_api', 'number_of_edges'])
    data['neighborhood'] = data.groupby(['euclidean_distance', 'graph_api']).cumcount()

    # Create a figure with 2x2 subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    axes = axes.flatten()  # Flatten for easier indexing

    # For each neighborhood (r0, r1, r2, r3)
    for i in range(4):
        ax = axes[i]
        neighborhood_data = data[data['neighborhood'] == i]

        # Plot each graph API as a separate line
        for api in sorted(data['graph_api'].unique()):
            api_data = neighborhood_data[neighborhood_data['graph_api'] == api]

            if not api_data.empty:
                ax.plot(
                    api_data['euclidean_distance'],
                    api_data['runtime_total'],
                    marker='o',
                    label=api,
                    linewidth=2
                )

        # Set titles and labels
        ax.set_title(f'r{i}-Neighborhood')
        ax.set_xlabel('Euclidean Distance')
        ax.set_ylabel('Runtime (seconds)')
        ax.legend()

        # Set grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    return fig


def random_points_with_distance(raster, target_distance):
    """
    Yields pairs of random points within a raster that are separated by the target Euclidean distance
    and have pixel values lower than 65535.

    Parameters:
    -----------
    raster : object with bounds attribute
        The geospatial raster with bounds attributes (left, right, bottom, top)
    target_distance : float
        The desired Euclidean distance between the points

    Yields:
    -------
    tuple
        A tuple containing two points as ((x1, y1), (x2, y2))
    """
    # Extract raster boundaries
    min_x = raster.bounds.left
    max_x = raster.bounds.right
    min_y = raster.bounds.bottom
    max_y = raster.bounds.top

    # Check if the target distance is feasible within this raster
    max_possible_distance = math.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)
    if target_distance > max_possible_distance:
        raise ValueError(
            f"Target distance {target_distance} exceeds maximum possible distance in raster {max_possible_distance}")

    while True:
        # Generate first random point
        x1 = random.uniform(min_x, max_x)
        y1 = random.uniform(min_y, max_y)

        # Generate random angle (0 to 2π)
        angle = random.uniform(0, 2 * math.pi)

        # Calculate second point at target distance in random direction
        x2 = x1 + target_distance * math.cos(angle)
        y2 = y1 + target_distance * math.sin(angle)

        # Check if second point is within bounds
        if min_x <= x2 <= max_x and min_y <= y2 <= max_y:
            # Sample raster at both points (implementation depends on raster type)
            try:
                value1 = sample_raster_value(raster, x1, y1)
                value2 = sample_raster_value(raster, x2, y2)
            except AttributeError:
                continue

            # Only yield points if both values are below 65535
            if value1 < 65535 and value2 < 65535:
                yield (x1, y1), (x2, y2)


def sample_raster_value(raster, x, y):
    """
    Sample the raster value at the given coordinates.
    Implementation depends on your raster library (rasterio, GDAL, etc.).

    Parameters:
    -----------
    raster : raster object
    x, y : float
        Coordinates to sample

    Returns:
    --------
    float
        Raster value at the given point
    """
    # Convert coordinates to pixel indices
    row, col = raster.index(x, y)
    # Read the value (assuming single band)
    value = raster.read(1, window=((row, row + 1), (col, col + 1)))[0, 0]
    return value


def benchmark_graph_apis(raster_path, euclidean_distances, neighborhoods, graph_apis, repeat=3):
    output_csv = "test.csv"  # benchmark_graph_apis.csv"
    results = []
    with rio.open(raster_path) as raster:
        raster_data = raster.read()
        raster_transform = raster.transform
        raster_crs = raster.crs
        for ed in euclidean_distances:
            count = 0
            while count < repeat + 1:
                source, target = next(random_points_with_distance(raster, ed))
                print("ED = ", np.sqrt(np.power(source[0] - target[0], 2) + np.power(source[1] - target[1], 2)))
                new_point = False
                for graph_api in graph_apis:
                    for neighborhood in neighborhoods:
                        print(f"  Using graph API: {graph_api}")
                        try:
                            # Create PathFinder instance
                            path_finder = PathFinder(
                                dataset_source=raster_data,
                                source_coords=source,
                                target_coords=target,
                                neighborhood_str=neighborhood,
                                graph_api=graph_api,
                                transform=raster_transform,
                                crs=raster_crs,
                                search_space_buffer_m=round(ed * 0.2)
                            )

                            # Find path
                            try:
                                path = path_finder.find_route()
                                count += 1
                            except NoPathFoundError:
                                new_point = True
                                break

                            # Extract results
                            result = {
                                "euclidean_distance": round(ed),
                                "graph_api": graph_api,
                                "source": str(source),
                                "target": str(target),
                                "search_space_buffer_m": path_finder.search_space_buffer_m,
                                "number_of_nodes": path_finder.graph_api.get_number_of_nodes(),
                                "number_of_edges": path_finder.graph_api.get_number_of_edges()
                            }

                            # Add path metrics if path was found
                            if path:
                                result.update({
                                    "path_length": path.total_length,
                                    "path_cost": path.total_cost,
                                })

                                # Add all runtime metrics
                                for key, value in path.runtimes.items():
                                    result[f"runtime_{key}"] = value
                            else:
                                print(f"  No path found for {graph_api}")
                                # Set metrics to None
                                result.update({
                                    "path_length": None,
                                    "path_cost": None,
                                })
                                for key in ["raster_loading", "graph_creation", "shortest_path", "total"]:
                                    result[f"runtime_{key}"] = None

                            # Append to results list
                            results.append(result)

                            # Save current results to CSV
                            pd.DataFrame(results).to_csv(output_csv, index=False)
                            print(f"  Results saved to {output_csv}")

                        except Exception as e:
                            print(f"Error with {graph_api} for pair {count + 1}: {e}")
                            # Add error entry to results
                            result = {
                                "euclidean_distance": round(ed),
                                "graph_api": graph_api,
                                "source": str(source),
                                "target": str(target),
                                "search_space_buffer_m":  path_finder.search_space_buffer_m,
                                "error": str(e),
                                "number_of_nodes": 0,
                                "number_of_edges": 0
                            }
                            results.append(result)

                            # Save current results to CSV
                            pd.DataFrame(results).to_csv(output_csv, index=False)
                    if new_point:
                        break
                if new_point:
                    break


if __name__ == "__main__":

    euclidean_distances = list(range(3000, 21000, 1000)) # list(range(100, 1100, 100)) + list(range(1000, 21000, 1000))
    benchmark_graph_apis(r"C:\Users\mhnn82\Documents\3_python_projects\pyorps\examples\data\raster\big_raster.tiff",
                         euclidean_distances,
                         ['r0', 'r1', 'r2', 'r3'],
                         ['rustworkx', 'networkit', 'igraph', 'networkx'])

