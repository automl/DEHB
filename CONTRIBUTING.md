# Contributing to DEHB

Thank you for considering contributing to DEHB! We welcome contributions from the community to help improve our project. Please take a moment to review the guidelines below before getting started.

## Table of Contents

- [How to Contribute](#how-to-contribute)
- [Bug Reports](#bug-reports)
- [Feature Requests](#feature-requests)
- [Code Contributions](#code-contributions)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Code Style and Guidelines](#code-style-and-guidelines)
- [Documentation](#documentation)
- [Community Guidelines](#community-guidelines)

## How to Contribute

There are several ways you can contribute to DEHB:

- Reporting bugs
- Requesting new features
- Improving documentation
- Fixing issues or enhancing existing features
- Writing tests
- Providing feedback and suggestions
- Spreading the word

## Bug Reports

If you encounter a bug while using DEHB, please help us by submitting a detailed bug report. Include the following information in your bug report:

- A clear and descriptive title
- A step-by-step description of how to reproduce the issue
- Details about your environment (e.g., operating system, Python version)
- Any relevant error messages or stack traces

## Feature Requests

We appreciate your ideas and feedback for improving DEHB. If you have a feature request, please follow these guidelines:

- Provide a clear and concise description of the feature you would like to see
- Explain why this feature would be beneficial to DEHB
- If possible, provide examples or use cases to illustrate the feature

## Code Contributions

We welcome code contributions to DEHB! To contribute code, please follow these steps:

1. Fork the repository on GitHub.
2. Clone your forked repository to your local machine.
3. Create a new branch for your changes.
4. Make your modifications or additions.
5. Ensure that your code adheres to the code style and guidelines (see next section).
6. Write tests to cover your changes if applicable.
7. Commit your changes with a clear and descriptive commit message.
8. Push your branch to your forked repository on GitHub.
9. Submit a pull request to the main repository targeting the ```development``` branch.

## Submitting a Pull Request

When submitting a pull request, please ensure the following:

- Provide a clear and descriptive title for your pull request.
- Reference any related issues or pull requests in the description.
- Include a summary of the changes made and the motivation behind them.
- Make sure that all the tests pass.
- Ensure your code follows the project's code style and guidelines.
- Be responsive to any feedback or questions during the review process.

Additonally, we ask you to run specific benchmarks, depending on the depth of your changes:

1. Style changes.

    If your changes only consist of style modifications, such as renaming or adding docstrings, and do not interfere with DEHB's interface, functionality, or algorithm, it is sufficient for all test cases to pass.

2. Changes to DEHB's interface and functionality or the algorithm itself.

    If your changes affect the interface, functionality, or algorithm of DEHB, please also run the synthetic benchmarks (MFH3, MFH6 of MFPBench, and the CountingOnes benchmark). This will help determine whether any changes introduced bugs or significantly altered DEHB's performance. However, at the reviewer's discretion, you may also be asked to run your changes on real-world benchmarks if deemed necessary. For instructions on how to install and run the benchmarks, please have a look at our [benchmarking instructions](./benchmarking/BENCHMARKING.md). Please use the same budget for your benchmark runs as we specified in the instructions.

## Code Style and Guidelines

To maintain consistency and readability, we follow a set of code style and guidelines. Please make sure that your code adheres to these standards:

- Use meaningful variable and function names.
- Write clear and concise comments to explain your code.
- Write docstrings in [Google style](https://google.github.io/styleguide/pyguide.html).
- Follow the project's indentation and formatting conventions. This can be checked by using pre-commit (```pre-commit run --all-files```).
- Keep lines of code within a reasonable length (recommended maximum: 100 characters).
- Write comprehensive and meaningful commit messages.
- Write unit tests for new features and ensure existing tests pass.

## Documentation
Proper documentation is crucial for the maintainability and usability of the DEHB project. Here are the guidelines for documenting your code:

### General Guidelines

- **New Features:** All new features must include documentation.
- **Docstrings:** All public functions must include docstrings that follow the [Google style guide](https://google.github.io/styleguide/pyguide.html).
- **Comments:** Use comments to explain the logic behind complex code, special cases, or non-obvious implementations.
- **Clarity:** Ensure that your comments and docstrings are clear, concise, and informative.

### Docstring Requirements

For each public function, the docstring should include:

1. **Summary:** A brief description of the function's purpose.
2. **Parameters:** A list of all parameters with descriptions, including types and any default values.
3. **Returns:** A description of the return values, including types.
4. **Raises:** A list of any exceptions that the function might raise.

### Example Docstring

```python
def example_function(param1: int, param2: str = "default") -> bool:
    """
    This is an example function that demonstrates how to write a proper docstring.

    Args:
        param1 (int): The first parameter, an integer.
        param2 (str, optional): The second parameter, a string. Defaults to "default".

    Returns:
        bool: The return value. True if successful, False otherwise.

    Raises:
        ValueError: If `param1` is negative.
    """
    if param1 < 0:
        raise ValueError("param1 must be non-negative")
    return True
```

### Rendering Documentation Locally

To render the documentation locally for debugging and review:

1. Install the required `dev` dependencies:

    ```bash
    pip install -e .[dev]
    ```

2. Use `mike` to deploy and serve the documentation locally:

    ```bash
    mike deploy --update-aliases 2.0.0 latest --ignore
    mike serve
    ```

3. The docs should now be viewable on http://localhost:8000/. If not, check your command prompt for any errors (or different local server adress).

## Community Guidelines

When participating in the DEHB community, please adhere to the following guidelines:

- Be respectful and considerate of others' opinions and ideas.
- Avoid offensive, derogatory, or inappropriate language or behavior.
- Help create a welcoming and inclusive environment for all participants.
- Provide constructive feedback and suggestions.
- Report any issues or concerns to the project maintainers.

Thank you for taking the time to read and understand our contribution guidelines. We appreciate your support and look forward to your contributions to DEHB!

