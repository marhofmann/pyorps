"""
PYORPS: An Open-Source Tool for Automated Power Line Routing

Reference:
[1] Hofmann, M., Stetz, T., Kammer, F., Repo, S.: 'PYORPS: An Open-Source Tool for
    Automated Power Line Routing', CIRED 2025 - 28th Conference and Exhibition on
    Electricity Distribution, 16 - 19 June 2025, Geneva, Switzerland
"""
from abc import ABC, abstractmethod
from typing import Union, Optional

from numpy import ndarray

from pyorps.core.types import NodeList, NodePathList


class GraphAPI(ABC):
    """Base class for all graph APIs defining the minimal required interface."""

    def __init__(
            self,
            raster_data: ndarray[int],
            steps: ndarray[int],
            ignore_max: Optional[bool] = True,
    ):
        """
        Initialize the base graph API with raster data and neighborhood steps.

        Args:
            raster_data: 2D numpy array representing the raster costs
            steps: Array defining the neighborhood connections
            ignore_max: Ignore edges whose weights are greater or equal to the maximum
            value in the raster data
        """
        self.raster_data = raster_data
        self.steps = steps
        self.ignore_max = ignore_max

    @abstractmethod
    def shortest_path(
            self,
            source_indices: Union[int, list[int], ndarray[int], tuple[int, int]],
            target_indices: Union[int, list[int], ndarray[int], tuple[int, int]],
            algorithm: str = "dijkstra",
            **kwargs
    ) -> Union[NodeList, NodePathList]:
        """
        Find the shortest path(s) between source and target indices.

        Args:
            source_indices: Source node indices
            target_indices: Target node indices
            algorithm: Algorithm name (e.g., "dijkstra", "astar")
            **kwargs:
                pairwise : bool
                    If True, compute pairwise shortest paths between source_indices and
                    target_indices.
                    Only allowed if len(source_indices) == len(target_indices)
                heuristic : callable, optional
                    A function that takes two node indices (u, target) and returns an
                    estimate of the distance between them. Only used when
                    algorithm="astar".


        Returns:
            list of path indices for each source-target pair
        """
