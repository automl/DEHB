# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2023-08-03

### Added

- Unittest and documentation setup via pytest and mkdocs respectively
- Pre-commit pipeline for style-checking via ruff, mypy and black
- Better logging for min_budget >= max_budget (#33)
- CONTRIBUTING.md for future contributions

### Fixed

- Use of deprecated numpy method ```np.int``` in dehb setup (#41)
- If condition for proper Client cleanup, since it was never ```True``` (#45)

### Changed

- README.md to feature badges

[unreleased]: https://github.com/automl/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/automl/releases/tag/v0.1.0