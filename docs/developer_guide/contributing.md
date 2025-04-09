# Contributing Guidelines

Thank you for your interest in contributing to the Video Annotator project! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please read and follow our Code of Conduct to foster an inclusive and respectful community.

## Getting Started

### Issues

- Check existing issues to see if your problem or idea has already been reported.
- If not, create a new issue to discuss your proposed changes before starting work.
- For bug reports, include:
  - Steps to reproduce the bug
  - Expected behavior
  - Actual behavior
  - Screenshots or error messages
  - System information (OS, Python version, etc.)

### Development Environment

Set up your development environment following the [Development Setup](development_setup.md) guide.

## Making Changes

### Branching Strategy

We follow a simplified Git workflow:

1. Fork the repository (if you're an external contributor)
2. Create a new branch from `main` with a descriptive name:
   - `feature/your-feature-name` for new features
   - `bugfix/issue-description` for bug fixes
   - `docs/what-you-documented` for documentation changes
3. Make your changes in the new branch
4. Submit a pull request to the `main` branch

### Commit Messages

Write clear and descriptive commit messages:

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters
- Reference issues and pull requests after the first line

Example:
```
Add keyboard shortcuts for annotation

- Add 's' shortcut for Stop
- Add 'r' shortcut for Run
- Add 't' shortcut for Turn

Fixes #123
```

### Code Style

Follow the project's code style:

- Use [PEP 8](https://www.python.org/dev/peps/pep-0008/) with a line length of 88 characters (Black default)
- Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Use type hints for function parameters and return values
- Format your code with Black:
  ```bash
  black src
  ```
- Check your code with Flake8:
  ```bash
  flake8 src
  ```

### Documentation

- Update documentation for any changes to functionality
- Add docstrings to all functions, classes, and modules
- Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for docstrings

### Testing

- Write tests for all new functionality
- Ensure all tests pass before submitting a pull request:
  ```bash
  pytest
  ```
- Aim for high test coverage:
  ```bash
  pytest --cov=src
  ```

## Pull Request Process

1. Update the README.md and documentation with details of changes
2. Update the CHANGELOG.md with a description of your changes
3. Ensure all tests pass and the code follows the style guide
4. Submit a pull request to the `main` branch
5. Request a review from a maintainer
6. Address any feedback from the review
7. Once approved, a maintainer will merge your pull request

## Review Process

Pull requests are reviewed by maintainers who will check:

- Code quality and style
- Test coverage
- Documentation
- Functionality

Feedback will be provided as comments on the pull request. Please address all feedback before the pull request can be merged.

## Release Process

Releases are managed by the maintainers:

1. Maintainers will periodically create release branches
2. Release branches will be tested thoroughly
3. Once tested, a new version will be tagged and released
4. The CHANGELOG.md will be updated with the release notes

## Contribution Types

### Bug Fixes

1. Find or create an issue describing the bug
2. Create a branch for your fix
3. Write a test that reproduces the bug
4. Fix the bug
5. Ensure the test passes
6. Submit a pull request

### Features

1. Discuss the feature in an issue first
2. Create a branch for your feature
3. Implement the feature with tests
4. Update documentation
5. Submit a pull request

### Documentation

1. Create a branch for your documentation changes
2. Make your changes
3. Submit a pull request

### Performance Improvements

1. Create a branch for your performance improvements
2. Implement the improvements
3. Provide benchmarks showing the improvement
4. Submit a pull request

## Getting Help

If you need help with contributing:

- Ask questions in the issue you're working on
- Contact the maintainers directly
- Join the community chat (if available)

## Recognition

All contributors will be recognized in the CONTRIBUTORS.md file and in the release notes.

Thank you for contributing to the Video Annotator project!
