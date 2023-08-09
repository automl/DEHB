# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.6] - 2023-08-04

### Added

- Unittest and documentation setup via pytest and mkdocs respectively
- Pre-commit pipeline for style-checking via ruff, mypy and black
- Better logging for min_budget >= max_budget (#33)
- CONTRIBUTING.md for future contributions

### Fixed

- Use of deprecated numpy method ```np.int``` in dehb setup (#41)
- If condition for proper Client cleanup, since it was never ```True``` (#45)
- Data leak in example ```01_Optimizing_RandomForest_using_DEHB``` (#23)

### Changed

- README.md to feature badges
- Folder structure to have package fully contained in ```src/```

> **_NOTE:_**  For versions <= 0.0.5 we do not provide release tags, since we changed our release pipeline from version 0.0.6 onwards.

## 0.0.5 - 2023-02-19

### Fixed

- Add ```dtype``` option to history ```np.array```

## 0.0.4 - 2022-09-18

### Changed

- Update default hyperparameters to match the hyperparameters in the DEHB paper.

## 0.0.3 - 2022-08-17

### Added

- Add option to append custom name to output file name
## 0.0.1 & 0.0.2 - 2022-06-13

### Added
- Initial project release and push to PyPI

[unreleased]: https://github.com/automl/compare/v0.0.6...HEAD
[0.0.6]: https://github.com/automl/releases/tag/v0.0.6