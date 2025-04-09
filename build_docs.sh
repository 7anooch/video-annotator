#!/bin/bash

set -e  # Exit on error

# Check if requirements file exists
if [ ! -f "docs/requirements_docs.txt" ]; then
    echo "Error: docs/requirements_docs.txt not found"
    exit 1
fi

# Install MkDocs and required packages
echo "Installing MkDocs and required packages..."
pip install -r docs/requirements_docs.txt

# Build the documentation
echo "Building documentation..."
mkdocs build

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Documentation built successfully!"
    echo "The documentation is available in the 'site' directory."
    echo "To serve the documentation, run: ./serve_docs.sh"
else
    echo "Error: Documentation build failed"
    exit 1
fi
