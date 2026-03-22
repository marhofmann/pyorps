import unittest
from pyorps.core.exceptions import (
    PyorpsError,
    CostAssumptionsError, FileLoadError, InvalidSourceError, FormatError,
    FeatureColumnError, NoSuitableColumnsError, ColumnAnalysisError,
    WFSError, WFSConnectionError, WFSResponseParsingError, WFSLayerNotFoundError,
    RasterShapeError, NoPathFoundError, AlgorithmNotImplementedError,
    PairwiseError
)


class TestExceptions(unittest.TestCase):
    def test_pyorps_error_base_class(self):
        """Test that PyorpsError is the base for all custom exceptions."""
        # All top-level exception classes should inherit from PyorpsError
        self.assertTrue(issubclass(CostAssumptionsError, PyorpsError))
        self.assertTrue(issubclass(FeatureColumnError, PyorpsError))
        self.assertTrue(issubclass(WFSError, PyorpsError))
        self.assertTrue(issubclass(RasterShapeError, PyorpsError))
        self.assertTrue(issubclass(NoPathFoundError, PyorpsError))
        self.assertTrue(issubclass(AlgorithmNotImplementedError, PyorpsError))
        self.assertTrue(issubclass(PairwiseError, PyorpsError))

        # Derived exceptions should also be catchable as PyorpsError
        self.assertTrue(issubclass(FileLoadError, PyorpsError))
        self.assertTrue(issubclass(WFSConnectionError, PyorpsError))
        self.assertTrue(issubclass(NoSuitableColumnsError, PyorpsError))

        # Instance checks
        self.assertIsInstance(NoPathFoundError(0, 0), PyorpsError)
        self.assertIsInstance(PairwiseError(), PyorpsError)
        self.assertIsInstance(RasterShapeError((1, 2, 3)), PyorpsError)

        # PyorpsError itself should be an Exception
        self.assertTrue(issubclass(PyorpsError, Exception))

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

        # AlgorithmNotImplementedError
        error = AlgorithmNotImplementedError("dijkstra", "networkx")
        self.assertIn("Algorithm dijkstra for networkx not supported", str(error))

    def test_pairwise_error(self):
        """Test PairwiseError instantiation and message."""
        error = PairwiseError()
        msg = str(error)
        self.assertIn("Pairwise computation failed", msg)
        self.assertIn("same length", msg)
        self.assertIsInstance(error, Exception)

    def test_column_analysis_error(self):
        """Test ColumnAnalysisError instantiation and hierarchy."""
        error = ColumnAnalysisError("analysis went wrong")
        self.assertIn("analysis went wrong", str(error))
        self.assertIsInstance(error, FeatureColumnError)
        self.assertIsInstance(error, Exception)

    def test_no_path_found_error_with_add_message(self):
        """Test NoPathFoundError with additional message."""
        error = NoPathFoundError(5, 10, add_message=" Extra info.")
        msg = str(error)
        self.assertIn("No path found from 5 to 10", msg)
        self.assertIn("Extra info.", msg)
