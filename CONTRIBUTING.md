# Contributing to Video Annotator

Thank you for your interest in contributing to the Video Annotator project! This document provides a quick overview of the contribution process.

## How to Contribute

1. **Find or create an issue**: Before starting work, check if there's an existing issue or create a new one to discuss your proposed changes.

2. **Fork and clone the repository**: Fork the repository to your GitHub account and clone it to your local machine.

3. **Set up the development environment**: Follow the [Development Setup](docs/developer_guide/development_setup.md) guide to set up your environment.

4. **Create a branch**: Create a new branch for your changes with a descriptive name.

5. **Make your changes**: Implement your changes, following the project's [code style](#code-style).

6. **Write tests**: Add tests for your changes to ensure they work correctly.

7. **Update documentation**: Update the documentation to reflect your changes.

8. **Submit a pull request**: Push your changes to your fork and submit a pull request to the main repository.

## Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with a line length of 88 characters (Black default)
- Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Format your code with Black: `black src`
- Check your code with Flake8: `flake8 src`

## Testing

- Write tests for all new functionality
- Run tests with pytest: `pytest`
- Check test coverage: `pytest --cov=src`

## Documentation

- Update documentation for any changes to functionality
- Add docstrings to all functions, classes, and modules
- Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for docstrings

## Pull Request Process

1. Update the README.md and documentation with details of changes
2. Update the CHANGELOG.md with a description of your changes
3. Ensure all tests pass and the code follows the style guide
4. Submit a pull request to the `main` branch
5. Request a review from a maintainer

## Detailed Guidelines

For more detailed guidelines, please see the [Contributing Guidelines](docs/developer_guide/contributing.md) in the developer documentation.

## Getting Help

If you need help with contributing:

- Ask questions in the issue you're working on
- Contact the maintainers directly

Thank you for contributing to the Video Annotator project!
