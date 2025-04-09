#!/bin/bash
# Run tests for the Video Annotator project

# Change to the project root directory
cd "$(dirname "$0")/.."

# Run the tests
python -m tests.run_tests "$@"

# Exit with the same status as the tests
exit $?
