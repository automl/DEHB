# Welcome to DEHB's documentation!

## Introduction
DEHB was designed to be an algorithm for Hyper Parameter Optimization (HPO). DEHB uses Differential Evolution (DE) under-the-hood as an Evolutionary Algorithm to power the black-box optimization that HPO problems pose. DE is a black-box optimization algorithm that generates candidate configurations $x$, to the black-box function $f(x)$, that is being optimized. The $x$ is evaluated by the black-box and the corresponding response $y$ is made available to the DE algorithm, which can then use this observation ($x$, $y$), along with previous such observations, to suggest a new candidate $x$ for the next evaluation. DEHB also uses Hyperband along with DE, to allow for cheaper approximations of the actual evaluations of $x$.

`dehb` is a python package implementing the [DEHB](https://arxiv.org/abs/2105.09821) algorithm. It offers an intuitive interface to optimize user-defined problems using DEHB.

This documentation explains how to use `dehb` and demonstrates its features. In the following section you will be guided how to install the `dehb` package and how to use it in your own projects. Examples with more hands-on material can be found in the [examples folder](https://github.com/automl/DEHB/tree/master/examples).

## Installation

To start using the `dehb` package, you can install it via pip. You can either install the package right from git or install it as an editable package to modify the code and rerun your experiments:

```bash
# Install from pypi
pip install dehb
```

!!! note "From Source"

    To install directly from from source

    ```bash
    git clone https://github.com/automl/DEHB.git
    pip install -e DEHB  # -e stands for editable, lets you modify the code and rerun things
    ```

## Using DEHB
DEHB allows users to either utilize the Ask & Tell interface for manual task distribution or leverage the built-in functionality (`run`) to set up a Dask cluster autonomously. Please refer to our [Getting Started](getting_started/single_worker.md) examples.

## Contributing
Please have a look at our [contributing guidelines](https://github.com/automl/DEHB/blob/master/CONTRIBUTING.md).

## To cite the paper or code
If you use DEHB in one of your research projects, please cite our paper(s):
```bibtex
@inproceedings{awad-ijcai21,
  author    = {N. Awad and N. Mallik and F. Hutter},
  title     = {{DEHB}: Evolutionary Hyberband for Scalable, Robust and Efficient Hyperparameter Optimization},
  pages     = {2147--2153},
  booktitle = {Proceedings of the Thirtieth International Joint Conference on
               Artificial Intelligence, {IJCAI-21}},
  publisher = {ijcai.org},
  editor    = {Z. Zhou},
  year      = {2021}
}
```