# DEHB: Evolutionary Hyperband for Scalable, Robust and Efficient Hyperparameter Optimization
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://github.com/automl/DEHB/actions/workflows/pytest.yml/badge.svg)](https://github.com/automl/DEHB/actions/workflows/pytest.yml)
[![docs](https://github.com/automl/DEHB/actions/workflows/docs.yml/badge.svg)](https://automl.github.io/DEHB/)
[![Coverage Status](https://coveralls.io/repos/github/automl/DEHB/badge.svg)](https://coveralls.io/github/automl/DEHB)
[![PyPI](https://img.shields.io/pypi/v/dehb)](https://pypi.org/project/dehb/)
[![Static Badge](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20-blue)](https://pypi.org/project/dehb/)
[![arXiv](https://img.shields.io/badge/arXiv-2105.09821-b31b1b.svg)](https://arxiv.org/abs/2105.09821)

Welcome to DEHB, an algorithm for Hyperparameter Optimization (HPO). DEHB uses Differential Evolution (DE) under-the-hood as an Evolutionary Algorithm to power the black-box optimization that HPO problems pose.

`dehb` is a python package implementing the [DEHB](https://arxiv.org/abs/2105.09821) algorithm. It offers an intuitive interface to optimize user-defined problems using DEHB.

### Getting Started
#### Installation
```bash
pip install dehb
```
#### Using DEHB
DEHB allows users to either utilize the Ask & Tell interface for manual task distribution or leverage the built-in functionality (`run`) to set up a Dask cluster autonomously. The following snippet offers a small look in to how to use DEHB. For further information, please refer to our [getting started examples](https://automl.github.io/DEHB/latest/getting_started/single_worker/) in our documentation.
```python
optimizer = DEHB(
    f=your_target_function,
    cs=config_space, 
    dimensions=dimensions, 
    min_fidelity=min_fidelity, 
    max_fidelity=max_fidelity)

##### Using Ask & Tell
# Ask for next configuration to run
job_info = optimizer.ask()

# Run the configuration for the given fidelity. Here you can freely distribute the computation to any worker you'd like.
result = your_target_function(config=job_info["config"], fidelity=job_info["fidelity"])

# When you received the result, feed them back to the optimizer
optimizer.tell(job_info, result)

##### Using run()
# Run optimization for 1 bracket. Output files will be saved to ./logs
traj, runtime, history = optimizer.run(brackets=1)
```

#### Running DEHB in a parallel setting
For a more in-depth look in how-to run DEHB in a parallel setting, please have a look at our [documentation](https://automl.github.io/DEHB/latest/getting_started/parallel/).

### Tutorials/Example notebooks

* [00 - A generic template to use DEHB for multi-fidelity Hyperparameter Optimization](examples/00_interfacing_DEHB.ipynb)
* [01.1 - Using DEHB to optimize 4 hyperparameters of a Scikit-learn's Random Forest on a classification dataset](examples/01.1_Optimizing_RandomForest_using_DEHB.ipynb)
* [01.2 - Using DEHB to optimize 4 hyperparameters of a Scikit-learn's Random Forest on a classification dataset using Ask & Tell interface](examples/01.2_Optimizing_RandomForest_using_Ask_Tell.ipynb)
* [02 - Optimizing Scikit-learn's Random Forest without using ConfigSpace to represent the hyperparameter space](examples/02_using%20DEHB_without_ConfigSpace.ipynb)
* [03 - Hyperparameter Optimization for MNIST in PyTorch](examples/03_pytorch_mnist_hpo.py)

To run PyTorch example: (*note additional requirements*) 
```bash
python examples/03_pytorch_mnist_hpo.py \
    --min_fidelity 1 \
    --max_fidelity 3 \
    --runtime 60 \
    --verbose
```
### Documentation
For more details and features, please have a look at our [documentation](https://automl.github.io/DEHB/latest/).

### Contributing
Any contribution is greaty appreciated! Please take the time to check out our [contributing guidelines](./CONTRIBUTING.md)

---

### To cite the paper or code

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
