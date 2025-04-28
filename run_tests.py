# run_tests.py
import unittest
import sys
import os

if __name__ == "__main__":
    # Make sure we're in the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print(f"Looking for tests in: {os.path.join(script_dir, 'tests')}")

    # Set up test discovery
    loader = unittest.TestLoader()

    # Specify the start directory for test discovery
    start_dir = 'tests'

    # Use the default pattern for test files
    pattern = 'test_*.py'

    # Discover and build the test suite
    suite = loader.discover(start_dir, pattern=pattern)

    # Count tests
    test_count = suite.countTestCases()
    print(f"Found {test_count} tests")

    # Run the tests with more verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with appropriate code
    sys.exit(not result.wasSuccessful())
