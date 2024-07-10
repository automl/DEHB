# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2024-07-10

### Added
- Improved logging by making log level parameterizable (#85)
- Improved respecting of runtime budget (#30)
- Improved seeding/rng generation + seeding config space (#83)
- Add warning messages when using deprecated `run` parameters
- Add benchmarking suite + instructions

### Changes
- Add requirement for numpy<2.0 as ConfigSpace does not support numpy 2.0 yet

## [0.1.1] - 2024-04-01

### Added
- Improved logging and state saving
- Checkpointing and restarting an optimization run (#31)
- Clear communication, that warmstarting via tell is currently not supported
- Add class specific random number generators for better reproducibility

### Changes
- Interface changes for run, removing unnecessary logging frequency parameters, since they have been moved to the constructor

## [0.1.0] - 2024-02-15

### Added
- Configuration IDs to improve logging and reproducability (#62)
- Ask and Tell interface (#36)
- Examples for Ask and Tell interface

### Changes
- Interface changes (renamed budget to fidelity) for clearer interface

## [0.0.7] - 2023-08-23

### Added
- Support for `Constant` hyperparameters (#52)
- Unittests to test conversion from vector to configuration
- Landing page of documentation

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

[unreleased]: https://github.com/automl/DEHB/compare/v0.1.2...master
[0.1.2]: https://github.com/automl/DEHB/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/automl/DEHB/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/automl/DEHB/compare/v0.0.7...v0.1.0
[0.0.7]: https://github.com/automl/DEHB/compare/v0.0.6...v0.0.7
[0.0.6]: https://github.com/automl/DEHB/releases/tag/v0.0.6