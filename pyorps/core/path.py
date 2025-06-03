from typing import Optional, Any
from dataclasses import dataclass

from shapely.geometry import LineString

from pyorps.core.types import CoordinateTuple, NodeList, CoordinateList


@dataclass
class Path:
    """
    Dataclass representing a path in a raster graph. Used as container for all path
    metrics and information.
    """
    source: CoordinateTuple
    target: CoordinateTuple
    algorithm: str
    graph_api: str
    path_indices: NodeList
    path_coords: CoordinateList
    path_geometry: LineString
    euclidean_distance: float
    runtimes: dict[str, float]
    path_id: int
    search_space_buffer_m: float
    neighborhood: str

    # Optional metrics that may be calculated
    total_length: Optional[float] = None
    total_cost: Optional[float] = None
    length_by_category: Optional[dict[float, float]] = None
    length_by_category_percent: Optional[dict[float, float]] = None

    def to_geodataframe_dict(self) -> dict:
        """
        Convert Path object to a dictionary suitable for GeoDataFrame creation.

        Returns:
            dictionary with path data formatted for GeoDataFrame
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
        Return a string representation of the path including the path_id, source and
        target, as well as the path's total length and total cost.

        Returns:
            A string representation of the path.
        """
        result = f"Path(id={self.path_id}, source={self.source}, target={self.target}"
        if self.total_length is not None:
            result += f", length_m={self.total_length:.2f}"
        if self.total_cost is not None:
            result += f", cost={self.total_cost:.2f}"
        result += ")"
        return result

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the path.
        """
        return str(self)

    def __eq__(self, other: Any) -> bool:
        """
        Check for equality between two paths.
        """
        equal = True
        equal &= self.source.__eq__(other.source)
        equal &= self.target.__eq__(other.target)
        equal &= self.algorithm.__eq__(other.algorithm)
        equal &= self.graph_api.__eq__(other.graph_api)
        equal &= all(pi in self.path_indices for pi in other.path_indices)
        equal &= self.euclidean_distance.__eq__(other.euclidean_distance)
        equal &= self.search_space_buffer_m.__eq__(other.search_space_buffer_m)
        equal &= self.neighborhood.__eq__(other.neighborhood)
        return equal


class PathCollection:
    """
    Container for Path objects with O(1) retrieval by path ID and  O(n) lookup for
    source and target information. Paths can be added with new id by replacing a Path
    object with the same ID already existing in th PathCollection.

    """
    _paths: dict[int, Path]
    _next_id: int

    def __init__(self):
        """
        Create an empty PathCollection for collecting Paths with their IDs in a
        dictionary.
        """
        self._paths = {}  # dictionary with path_id as keys for O(1) lookup
        self._next_id = 0  # Track the next available path ID

    def add(self, path: Path, replace: bool = False) -> None:
        """
        Add a path to the PathCollection. If the Path's path_id is None or if replace is
        False, the path_id of the Path object will set to self._next_id and
        self._next_id will be incremented. If the Path's path_id is not None and
        replace is True, a Path with the same path_id (if present) will be replaced
        with the new Path object.

        Parameters:
            path: A Path object which should be added to the PathCollection.
            replace: Whether to replace an existing Path object with the same path_id
                (if present) or not.
        """
        if path.path_id is None or not replace:
            path.path_id = self._next_id
            self._next_id += 1
        else:
            # If an explicit path_id is provided, update _next_id if needed
            self._next_id = max(self._next_id, path.path_id + 1)

        self._paths[path.path_id] = path

    def get(
            self,
            path_id: int = None,
            source: Any = None,
            target: Any = None
    ) -> Optional[Path]:
        """
        Retrieve a stored path by ID, or by source AND target.

        Parameters:
            path_id: The ID of the Path object to retrieve (must be None if path
                should be found by source and target)
            source: The source Path object to retrieve (only used if path_id is None
                and target os set too; neglected otherwise)
            target: The target Path object to retrieve (only used if path_id is None
                and target os set too; neglected otherwise)

        Returns:
            The Path object with the specified ID or source/target pair. None if no
            such path exists.
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
        Iterate through all paths in the PathCollection.
        """
        return iter(self._paths.values())

    def __len__(self):
        """
        Return the number of paths in the PathCollection.
        """
        return len(self._paths)

    def __getitem__(self, path_id):
        """
        Get path by path_id of the Path object from the PathCollection.
        """
        return self._paths[path_id]

    def __str__(self) -> str:
        """
        Return a string representation of the path collection.
        """
        return f"PathCollection(count={len(self._paths)})"

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the path collection.
        """
        if len(self._paths) <= 5:
            paths_repr = ""
            for path in self._paths:
                if paths_repr != "":
                    paths_repr += ",\n"
                paths_repr += repr(path)
        else:
            # Show first 2 paths and last path for large collections
            paths_repr = (f"\n\t{repr(self.all[0])},"
                          f"\n\t{repr(self.all[1])},"
                          f"\n\t ..., "
                          f"\n\t{repr(self.all[-1])}")

        return f"PathCollection(paths=[{paths_repr}], count={len(self._paths)})"

    @property
    def all(self):
        """
        Return all Path objects from the values of the PathCollection's _paths
        dictionary as a list.

        Returns:
            A list of all Path objects in the PathCollection.
        """
        return list(self._paths.values())

    def __eq__(self, other) -> bool:
        """
        Check if PathCollections are equal. They do not have to be in the same order
        to be equal!
        """
        return all(o in self.all for o in other.all)
