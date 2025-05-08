import unittest

if __name__ == '__main__':
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('', pattern='test_*.py')
    test_runner = unittest.TextTestRunner(verbosity=3, warnings="ignore")
    test_runner.run(test_suite)
