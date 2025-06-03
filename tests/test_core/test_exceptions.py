import unittest
from pyorps.core.exceptions import (
    CostAssumptionsError, FileLoadError, InvalidSourceError, FormatError,
    FeatureColumnError, NoSuitableColumnsError, ColumnAnalysisError,
    WFSError, WFSConnectionError, WFSResponseParsingError, WFSLayerNotFoundError,
    RasterShapeError, NoPathFoundError, AlgorthmNotImplementedError
)


class TestExceptions(unittest.TestCase):
    def test_exception_hierarchy(self):
        """Test the inheritance of exceptions."""
        # Cost assumption exceptions
        self.assertTrue(issubclass(FileLoadError, CostAssumptionsError))
        self.assertTrue(issubclass(InvalidSourceError, CostAssumptionsError))
        self.assertTrue(issubclass(FormatError, CostAssumptionsError))

        # Feature column exceptions
        self.assertTrue(issubclass(NoSuitableColumnsError, FeatureColumnError))
        self.assertTrue(issubclass(ColumnAnalysisError, FeatureColumnError))

        # WFS exceptions
        self.assertTrue(issubclass(WFSConnectionError, WFSError))
        self.assertTrue(issubclass(WFSResponseParsingError, WFSError))
        self.assertTrue(issubclass(WFSLayerNotFoundError, WFSError))

    def test_exception_messages(self):
        """Test that exceptions format their messages correctly."""
        # RasterShapeError
        error = RasterShapeError((4, 5, 3))
        self.assertIn("Raster shape of (4, 5, 3) not supported", str(error))

        # NoPathFoundError
        error = NoPathFoundError(10, 20)
        self.assertIn("No path found from 10 to 20", str(error))

        # AlgorthmNotImplementedError
        error = AlgorthmNotImplementedError("dijkstra", "networkx")
        self.assertIn("Algorithm dijkstra for networkx not supported", str(error))
