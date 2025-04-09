#!/usr/bin/env python3
"""
Test runner for the Video Annotator project.

This script discovers and runs all tests in the tests directory.
"""

import unittest
import sys
import os
import argparse

def run_tests(test_pattern=None, verbosity=2):
    """
    Run all tests in the tests directory.
    
    Args:
        test_pattern (str, optional): Pattern to match test files. Defaults to None.
        verbosity (int, optional): Verbosity level. Defaults to 2.
        
    Returns:
        bool: True if all tests passed, False otherwise.
    """
    # Add the parent directory to the path so that imports work
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    # Discover and run tests
    if test_pattern:
        test_suite = unittest.defaultTestLoader.discover('.', pattern=test_pattern)
    else:
        test_suite = unittest.defaultTestLoader.discover('.')
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

def main():
    """Main function to run the tests."""
    parser = argparse.ArgumentParser(description="Run tests for the Video Annotator project")
    parser.add_argument('--pattern', type=str, default='test_*.py',
                        help="Pattern to match test files (default: test_*.py)")
    parser.add_argument('--verbosity', type=int, default=2,
                        help="Verbosity level (default: 2)")
    args = parser.parse_args()
    
    success = run_tests(args.pattern, args.verbosity)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
