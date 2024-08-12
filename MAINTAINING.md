# DEHB Package Maintaining

This document serves as a comprehensive guide for maintaining the DEHB package. It outlines the essential workflows, configurations, and processes required to ensure the package remains functional and up-to-date.

## Package-related Information

All package-related information can be found in the `pyproject.toml` file. In the following, we list specific, important parts of `pyproject.toml`.

### Dependencies

The mandatory dependencies can be found in the `[project]` section under `dependencies`. Optional dependencies are listed under `[project.optional-dependencies]`. When updating or adding dependencies, ensure they are compatible with the existing codebase by running the full test suite.

### Python Version Requirement

The python version requirement is specfied in the `[project]` section as `requires-python`.

### Package Version

The package version is also specified in the `[project]` section as `version`.

## Workflows

All workflows can be found in the `.github/workflows` folder. As the syntax is realtively straight-forward, we refer to the [github docs](https://docs.github.com/en/actions/using-workflows) for more information.

## Documentation

Our documentation is built using `mike` in order to allow for versioning and we use [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) to allow for material design. Both packages are based on [MkDocs](https://www.mkdocs.org/) and are thus configured via the `mkdocs.yml` file. The docs are automatically built and deployed when pushing to main. If you want to locally build the documentation, please checkout the documentation section in `CONTRIBUTING.md`.

## Tests

We use `pytest` for our unit tests, which are specified in the `tests` folder. We structure our tests in the following way:

- Each python file has their own respective test file (e.g., `test_dehb.py`).
- Inside these test files, we group tests regarding a specific functionality in a class (e.g., `TestAskTell`).
- Specific test cases are then defined as functions inside these classes (e.g., `test_ask_multiple`).

Test can be run via `pytest .` and are automatically run for any push or PR on the development/main branches. When making changes to the codebase, ensure that relevant tests are updated or added to maintain coverage and correctness.

## Releasing a New Version

Given that all changes are pushed to the `development` branch, we suggest following this procedure:

1. Adjust the `version` field in `pyproject.toml` according to [semantic versioning](https://semver.org/).
2. Adjust `CHANGELOG.md` to feature all changes for the new release.
3. Create PR targetting main.
4. Make sure all tests pass before merging.
5. Merge PR.
6. Generate distribution archives (see [here](https://packaging.python.org/en/latest/tutorials/packaging-projects/#generating-distribution-archives)).
7. Upload distribution archives to PyPI (see [here](https://packaging.python.org/en/latest/tutorials/packaging-projects/#uploading-the-distribution-archives)).
8. Create GitHub tag for new release.
