from typing import Dict, Optional, Any
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString


@dataclass
class Path:
    """Dataclass representing a path in a raster graph."""
    source: Any
    target: Any
    algorithm: str
    graph_api: str
    path_indices: np.ndarray
    path_coords: np.ndarray
    path_geometry: LineString
    euclidean_distance: float
    runtimes: Dict[str, float]
    path_id: int
    search_space_buffer_m: float
    neighborhood: str

    # Optional metrics that may be calculated
    total_length: Optional[float] = None
    total_cost: Optional[float] = None
    length_by_category: Optional[Dict[float, float]] = None
    length_by_category_percent: Optional[Dict[float, float]] = None

    def to_geodataframe_dict(self) -> dict:
        """
        Convert Path object to a dictionary suitable for GeoDataFrame creation.

        Returns:
            Dictionary with path data formatted for GeoDataFrame
        """
        # Add runtime information
        result = {f"runtime_{key}": value for key, value in self.runtimes.items()}

        # Basic path information
        result.update({
            "path_id": self.path_id,
            "source": str(self.source),
            "target": str(self.target),
            "algorithm": self.algorithm,
            "graph_api": self.graph_api,
            "geometry": self.path_geometry,
            "search_space_buffer_m": self.search_space_buffer_m,
            "euclidean_distance": self.euclidean_distance,
            "neighborhood": self.neighborhood,
        })

        # Add metrics if they exist
        if self.total_length is not None:
            result["path_length"] = self.total_length
            result["path_cost"] = self.total_cost

            # Add length by category columns if available
            if self.length_by_category:
                for category, length in self.length_by_category.items():
                    result[f"length_cost_{category}"] = length
                    lbc = self.length_by_category_percent[category]
                    result[f"percent_cost_{category}"] = lbc

        return result

    def __str__(self) -> str:
        """
        Return a string representation of the path.
        """
        result = f"Path(id={self.path_id}, source={self.source}, target={self.target}"
        if self.total_length is not None:
            result += f", length={self.total_length:.2f}"
        if self.total_cost is not None:
            result += f", cost={self.total_cost:.2f}"
        result += ")"
        return result

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the path.
        """
        result = (f"Path(path_id={self.path_id}, source={repr(self.source)}, "
                  f"target={repr(self.target)}")
        if self.total_length is not None:
            result += f", total_length={self.total_length:.2f}"
        if self.total_cost is not None:
            result += f", total_cost={self.total_cost:.2f}"
        result += ")"
        return result


class PathCollection:
    """
    Container for Path objects with O(1) retrieval by path ID.
    """

    def __init__(self):
        self._paths = {}  # Dictionary with path_id as keys for O(1) lookup
        self._next_id = 0  # Track the next available path ID

    def add(self, path: Path, replace: bool = False) -> None:
        """
        Add a path to the collection.
        """
        if path.path_id is None or not replace:
            path.path_id = self._next_id
            self._next_id += 1
        else:
            # If an explicit path_id is provided, update _next_id if needed
            self._next_id = max(self._next_id, path.path_id + 1)

        self._paths[path.path_id] = path

    def get(self, path_id: int = None, source: Any = None,
            target: Any = None) -> Optional[Path]:
        """
        Retrieve a stored path by ID, or by source AND target.
        """
        if path_id is not None:
            # O(1) lookup by ID
            return self._paths.get(path_id)

        if source is not None and target is not None:
            # O(n) lookup by source AND target - still need to iterate
            for path in self._paths.values():
                if path.source == source and path.target == target:
                    return path

        # If criteria not met or path not found, return None
        return None

    def to_geodataframe_records(self) -> list:
        """
        Convert all paths to a list of dictionaries suitable for a GeoDataFrame.

        Returns:
            List of dictionaries with path data formatted for a GeoDataFrame
        """
        return [path.to_geodataframe_dict() for path in self._paths.values()]

    def __iter__(self):
        """
        Iterate through paths.
        """
        return iter(self._paths.values())

    def __len__(self):
        """
        Return the number of paths.
        """
        return len(self._paths)

    def __getitem__(self, path_id):
        """
        Get path by ID.
        """
        return self._paths[path_id]

    def __str__(self) -> str:
        """
        Return a string representation of the path collection.
        """
        return f"PathCollection(paths={len(self._paths)})"

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the path collection.
        """
        if len(self._paths) <= 5:
            paths_repr = ", \n".join(repr(path) for path in self._paths.values())
        else:
            # Show first 2 paths and last path for large collections
            paths = list(self._paths.values())
            paths_repr = (f"{repr(paths[0])}, \n{repr(paths[1])}, \n..., "
                          f"\n{repr(paths[-1])}")

        return f"PathCollection(paths=[{paths_repr}], count={len(self._paths)})"

    @property
    def all(self):
        """
        Return all paths as a list.
        """
        return list(self._paths.values())

