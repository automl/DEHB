## Pull Request Checklist

Thank you for your contribution! Before submitting this PR, please make sure you have completed the following steps:

### 1. Unit Tests / Normal PR Workflow

- [ ] Ensure all existing unit tests pass.
- [ ] Add new unit tests to cover the changes.
- [ ] Verify that your code follows the project's coding standards.
- [ ] Add documentation for your code if necessary.
- [ ] Check below, if your changes require you to run benchmarks.

#### When Do I Need To Run Benchmarks?

Depending on your changes, we ask you to run some benchmarks:

1. Style changes.

    If your changes only consist of style modifications, such as renaming or adding docstrings, and do not interfere with DEHB's interface, functionality, or algorithm, it is sufficient for all test cases to pass.

2. Changes to DEHB's interface and functionality or the algorithm itself.

    If your changes affect the interface, functionality, or algorithm of DEHB, please also run the synthetic benchmarks (MFH3, MFH6 of MFPBench, and the CountingOnes benchmark). This will help determine whether any changes introduced bugs or significantly altered DEHB's performance. However, at the reviewer's discretion, you may also be asked to run your changes on real-world benchmarks if deemed necessary. For instructions on how to install and run the benchmarks, please have a look at our [benchmarking instructions](../benchmarking/BENCHMARKING.md). Please use the same budget for your benchmark runs as we specified in the instructions.
