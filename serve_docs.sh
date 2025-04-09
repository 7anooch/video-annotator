#!/bin/bash

set -e  # Exit on error

# Check if requirements file exists
if [ ! -f "docs/requirements_docs.txt" ]; then
    echo "Error: docs/requirements_docs.txt not found"
    exit 1
fi

# Install MkDocs and required packages if not already installed
echo "Installing MkDocs and required packages..."
pip install -r docs/requirements_docs.txt

# Serve the documentation
echo "Starting MkDocs server..."
echo "The documentation will be available at http://localhost:8000"
echo "Press Ctrl+C to stop the server"
mkdocs serve
