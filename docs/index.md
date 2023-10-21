# Welcome to DEHB's documentation!

## Introduction

`dehb` is a python package implementing the [DEHB](https://arxiv.org/abs/2105.09821) algorithm. It offers an intuitive interface to optimize user-defined problems using DEHB.

This documentation explains how to use `dehb` and demonstrates its features. In the following section you will be guided how to install the `dehb` package and how to use it in your own projects. Examples with more hands-on material can be found in the [examples folder](../examples/).

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

@online{Awad-arXiv-2023,
    title       = {MO-DEHB: Evolutionary-based Hyperband for Multi-Objective Optimization},
    author      = {Noor Awad and Ayushi Sharma and Frank Hutter},
    year        = {2023},
    keywords    = {}
}
```