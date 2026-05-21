"""
PYORPS: An Open-Source Tool for Automated Power Line Routing

Reference:
[1] Hofmann, M., Stetz, T., Kammer, F., Repo, S.: 'PYORPS: An Open-Source Tool for
    Automated Power Line Routing', CIRED 2025 - 28th Conference and Exhibition on
    Electricity Distribution, 16 - 19 June 2025, Geneva, Switzerland
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from numpy import ndarray
from shapely.geometry import LineString

from pyorps.core.types import CoordinateList, CoordinateTuple, NodeList

if TYPE_CHECKING:
    from pyorps.core.cost_assumptions import CostAssumptions


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
    total_length: float | None = None
    total_cost: float | None = None
    total_cell_cost: float | None = None
    length_by_category: dict[float, float] | None = None
    length_by_category_percent: dict[float, float] | None = None

    # Optional mapping from cost values to terrain type names
    cost_labels: dict[int, str] | None = field(default=None, repr=False)

    def _resolve_labels(
            self,
            cost_assumptions: CostAssumptions | None,
            cost_labels: dict[int, str] | None,
    ) -> dict[int, str]:
        if cost_labels is not None:
            return cost_labels
        if cost_assumptions is not None:
            return cost_assumptions.build_cost_labels()
        if self.cost_labels is not None:
            return self.cost_labels
        return {}

    # Simplification metadata (populated when find_route(simplify=...) is used)
    simplification_method: Optional[str] = None
    simplification_tolerance: Optional[float] = None
    original_path_geometry: Optional[LineString] = field(default=None, repr=False)

    def analyze(
            self,
            cost_assumptions: CostAssumptions | None = None,
            cost_labels: dict[int, str] | None = None,
    ) -> str:
        """
        Return a human-readable breakdown of the path by terrain category.

        Labels are resolved in this order:
        1. Explicit *cost_labels* dict passed to this call
        2. Labels built from a *cost_assumptions* object passed to this call
        3. ``self.cost_labels`` (set automatically when the PathFinder had
           cost assumptions available)
        4. Falls back to the raw cost value

        Parameters:
            cost_assumptions: Optional CostAssumptions instance to derive
                labels from.
            cost_labels: Optional dict mapping cost values to terrain names.

        Returns:
            Formatted string with path analysis.
        """
        labels = self._resolve_labels(cost_assumptions, cost_labels)

        lines = []
        lines.append(f"Path Analysis (ID={self.path_id})")
        lines.append("=" * 60)
        lines.append(f"  Source:             {self.source}")
        lines.append(f"  Target:             {self.target}")
        lines.append(f"  Neighborhood:       {self.neighborhood}")

        if self.total_length is not None:
            lines.append(f"  Total length:       {self.total_length:,.1f} m")
        if self.total_cost is not None:
            lines.append(f"  Total cost:         {self.total_cost:,.1f} EUR")
        if self.euclidean_distance:
            lines.append(
                f"  Euclidean distance: {self.euclidean_distance:,.1f} m")
        if self.total_length and self.euclidean_distance:
            detour = self.total_length / self.euclidean_distance
            lines.append(f"  Detour factor:      {detour:.2f}x")

        if self.simplification_method is not None:
            lines.append("")
            lines.append(
                f"  Simplification:     {self.simplification_method} "
                f"(tolerance={self.simplification_tolerance})"
            )
            if self.original_path_geometry is not None:
                vbefore = len(self.original_path_geometry.coords)
                vafter = len(self.path_geometry.coords)
                lines.append(
                    f"  Vertices:           {vbefore} -> {vafter} "
                    f"({100.0 * (1 - vafter / vbefore):.1f}% removed)"
                )

        if self.length_by_category:
            lines.append("")
            lines.append("  Terrain breakdown:")
            lines.append(
                f"  {'Terrain':<35} {'Length [m]':>10}  {'Share':>6}  "
                f"{'Cost [EUR/m]':>12}")
            lines.append("  " + "-" * 67)

            for cost_val in sorted(self.length_by_category.keys()):
                length_m = self.length_by_category[cost_val]
                pct = self.length_by_category_percent.get(cost_val, 0)
                label = labels.get(int(cost_val), f"Cost {int(cost_val)}")
                if len(label) > 35:
                    label = label[:32] + "..."
                lines.append(
                    f"  {label:<35} {length_m:>10,.1f}  {pct:>5.1f}%  "
                    f"{int(cost_val):>12}")

        result = "\n".join(lines)
        print(result)
        return result

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
            "simplification_method": self.simplification_method,
            "simplification_tolerance": self.simplification_tolerance,
        })

        # Add metrics if they exist
        if self.total_length is not None:
            result["path_length"] = self.total_length
            result["path_cost"] = self.total_cost
            result["path_cell_cost"] = self.total_cell_cost

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
        if "runtime_total" in self.runtimes:
            result += f", runtime_total={self.runtimes['runtime_total']:.2f}"

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
        same_source = self.source.__eq__(other.source)

        if isinstance(same_source, ndarray):
            equal &= same_source.all()
        else:
            equal &= same_source

        same_target = self.target.__eq__(other.target)
        if isinstance(same_target, ndarray):
            equal &= same_target.all()
        else:
            equal &= same_target
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
    ) -> Path | None:
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
        if len(self._paths) <= 5:
            paths_str = ""
            for path in self._paths.values():
                if paths_str != "":
                    paths_str += ",\n"
                paths_str += str(path)
        else:
            # Show first 2 paths and last path for large collections
            paths_str = (f"\n\t{str(self.all[0])},"
                          f"\n\t{str(self.all[1])},"
                          f"\n\t ..., "
                          f"\n\t{str(self.all[-1])}")

        return f"PathCollection(paths=[{paths_str}], count={len(self._paths)})"

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the path collection.
        """
        if len(self._paths) <= 5:
            paths_repr = ""
            for path in self._paths.values():
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
        return all(any(o == p for p in self.all) for o in other.all)
