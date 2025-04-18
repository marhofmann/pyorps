
from abc import ABC, abstractmethod
from typing import Any, Union

from numpy import ndarray


class GraphAPI(ABC):
    """Base class for all graph APIs defining the minimal required interface."""

    def __init__(self, raster_data: ndarray[int], steps: ndarray[int]):
        """
        Initialize the base graph API with raster data and neighborhood steps.

        Args:
            raster_data: 2D numpy array representing the raster costs
            steps: Array defining the neighborhood connections
        """
        self.raster_data = raster_data
        self.steps = steps

    @abstractmethod
    def shortest_path(self,
                      source_indices: Union[int, list[int], ndarray[int]],
                      target_indices: Union[int, list[int], ndarray[int]],
                      algorithm: str = "dijkstra",
                      pairwise: bool = False) -> list[list[int]]:
        """
        Find shortest path(s) between source and target indices.

        Args:
            source_indices: Source node indices
            target_indices: Target node indices
            algorithm: Algorithm name (e.g., "dijkstra", "astar")
            pairwise: Whether to compute paths pairwise (for multiple sources/targets)

        Returns:
            list of path indices for each source-target pair
        """
        pass

