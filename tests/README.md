# Tests for the `dehb` Package

Welcome to the test suite for the `dehb` package! This directory contains unit tests designed to ensure the correctness and functionality of the `dehb` package. These tests are implemented using the `pytest` testing framework.

## Table of Contents

- [Folder and File Structure](#folder-and-file-structure)
- [Running the Tests](#running-the-tests)
- [Contributing](#contributing)

## Folder and File Structure

All of our tests are locateted in the `tests` folder. Our strategy for test development is the following:
- For each source file in the package we aim to have a test file consisting of unit tests for this specific file (e.g. `test_dehb`)
- For other functionalities, that are not entirely related to a specific source file, we also add a test file (e.g. `test_imports`)
- The test files itself are grouped with the help of classes. Each class represents a specific use/test case of the module, e.g. `TestBudgetExhaustion` which tests whether our runtime budget exhaustion checks are still working correctly. Since we have 3 possible ways of specifying our runtime budget (function evaluations, number of brackets, runtime in seconds), each case is covered with one test function in the Class.

## Running the Tests
To run the tests, ensure you're in the root directory of the project and that pytest is installed on your system/virtual environment. Then, execute the following command:
```sh
pytest .
```
If you only want to run specific test cases, checkout the [pytest documentation](https://docs.pytest.org/en/7.1.x/how-to/usage.html).
## Contributing

We welcome contributions to improve the quality of this package. If you'd like to contribute, please take a moment to review our [Contribution Guidelines](../CONTRIBUTING.md) and adhere to the folder and file structure mentioned above.