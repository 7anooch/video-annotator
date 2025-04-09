#!/usr/bin/env python3
"""
Test runner for the analysis module.
"""

import unittest
import sys
import os

# Add the parent directory to the path so that the tests can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the test modules
from tests.analysis.test_data_model import TestAnnotationData
from tests.analysis.test_statistics import TestStatisticalAnalysis
from tests.analysis.test_utils import TestUtils
from tests.analysis.test_adapters import TestAnalyzeAdapter, TestPlotAdapter
from tests.analysis.test_visualization import TestVisualizationManager
from tests.analysis.test_integration import TestIntegration

def run_tests():
    """Run all the tests."""
    # Create a test suite
    test_suite = unittest.TestSuite()

    # Add the test cases
    test_suite.addTest(unittest.makeSuite(TestAnnotationData))
    test_suite.addTest(unittest.makeSuite(TestStatisticalAnalysis))
    test_suite.addTest(unittest.makeSuite(TestUtils))
    test_suite.addTest(unittest.makeSuite(TestAnalyzeAdapter))
    test_suite.addTest(unittest.makeSuite(TestPlotAdapter))
    test_suite.addTest(unittest.makeSuite(TestVisualizationManager))
    test_suite.addTest(unittest.makeSuite(TestIntegration))

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Return the result
    return result

if __name__ == '__main__':
    result = run_tests()

    # Exit with a non-zero code if there were failures
    sys.exit(not result.wasSuccessful())
