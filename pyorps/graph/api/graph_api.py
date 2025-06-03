
from abc import ABC, abstractmethod
from typing import Union

from numpy import ndarray

from pyorps.core.types import NodeList, NodePathList


class GraphAPI(ABC):
    """Base class for all graph APIs defining the minimal required interface."""

    def __init__(
            self,
            raster_data: ndarray[int],
            steps: ndarray[int]
    ):
        """
        Initialize the base graph API with raster data and neighborhood steps.

        Args:
            raster_data: 2D numpy array representing the raster costs
            steps: Array defining the neighborhood connections
        """
        self.raster_data = raster_data
        self.steps = steps

    @abstractmethod
    def shortest_path(
            self,
            source_indices: Union[int, list[int], ndarray[int]],
            target_indices: Union[int, list[int], ndarray[int]],
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
