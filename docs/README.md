# Video Annotator Enhanced Analysis Tools Documentation

This directory contains the documentation for the enhanced analysis tools for the Video Annotator application.

## Building the Documentation

To build the documentation, run:

```bash
./build_docs.sh
```

This will install the required dependencies and build the documentation.

## Serving the Documentation

To serve the documentation locally, run:

```bash
./serve_docs.sh
```

This will start a local web server at http://localhost:8000 where you can view the documentation.

## Documentation Structure

The documentation is organized as follows:

- **Home**: Overview of the enhanced analysis tools.
- **Getting Started**: Guide to get started with the enhanced analysis tools.
- **User Guide**: Detailed information on how to use the enhanced analysis tools.
- **Components**: Documentation for each component of the enhanced analysis tools.
- **API Reference**: Detailed API documentation for each module.
- **Examples**: Examples of how to use the enhanced analysis tools.
- **Development**: Information for developers.
- **About**: License and release notes.

## Contributing to the Documentation

To contribute to the documentation, edit the Markdown files in the `docs/docs` directory. The documentation is built using MkDocs with the Material theme.

## Dependencies

The documentation requires the following dependencies:

- MkDocs
- MkDocs Material theme
- PyMdown Extensions
- MkDocstrings
- MkDocs Awesome Pages Plugin

These dependencies are listed in the `requirements_docs.txt` file and will be installed automatically when running the build or serve scripts.
