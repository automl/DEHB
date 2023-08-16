# Welcome to DEHB's documentation!

## Introduction

`dehb` is a python package implementing the [DEHB](https://arxiv.org/abs/2105.09821) algorithm. It offers an intuitive interface to optimize user-defined problems using DEHB.

This documentation explains how to use `dehb` and demonstrates its features. In the following section you will be guided how to install the `dehb` package and how to use it in your own projects. Examples with more hands-on material can be found in the [examples folder](../examples/).

## Installation

To start using the `dehb` package, you can install it via pip. You can either install the package right from git or install it as an editable package to modify the code and rerun your experiments:

```bash
# Install from pypi
pip install dehb

# Install as editable from github
git clone https://github.com/automl/DEHB.git
pip install -e DEHB  # -e stands for editable, lets you modify the code and rerun things
```

## Getting Started

In the following sections we provide some basic examplatory setup for running DEHB with a single worker or in a multi-worker setup.

### Basic single worker setup
A basic setup for optimizing can be done as follows. Please note, that this is example should solely show a simple setup of `dehb`. More in-depth examples can be found in the [examples folder](../examples/). First we need to setup a `ConfigurationSpace`, from which Configurations will be sampled:

```python
import ConfigSpace

cs = ConfigSpace.ConfigurationSpace()
cs.add_hyperparameter(ConfigSpace.UniformFloatHyperparameter("x0", lower=3, upper=10, log=False))
```

Next, we need an `object_function`, which we are aiming to optimize:
```python
import numpy as np
def objective_function(x, budget, **kwargs):
    """Toy objective function.

    Args:
        x (ConfigSpace.Configuration): Configuration to evaluate
        budget (float): Budget to evaluate x on

    Returns:
        dict: Result dictionary
    """
    # This obviously does not make sense in a real world example. Replace this with your actual objective value (y) and cost.
    y = np.random.uniform()
    cost = 5
    result = {
        "fitness": y,
        "cost": cost
    }
    return result
```

Finally, we can setup our optimizer and run DEHB:

```python
from dehb import DEHB

dim = len(cs.get_hyperparameters())
optimizer = DEHB(f=objective_function, cs=cs, dimensions=dim, min_budget=3, output_path="./logs",
                max_budget=27, eta=3, n_workers=1)

# Run optimization for 10 brackets. Output files will be save to ./logs
traj, runtime, history = opt.run(brackets=10, verbose=True)
```

### Running DEHB in a parallel setting

DEHB has been designed to interface a [Dask client](https://distributed.dask.org/en/latest/api.html#distributed.Client).
DEHB can either create a Dask client during instantiation and close/kill the client during garbage collection. 
Or a client can be passed as an argument during instantiation.

* Setting `n_workers` during instantiation \
    If set to `1` (default) then the entire process is a sequential run without invoking Dask. \
    If set to `>1` then a Dask Client is initialized with as many workers as `n_workers`. \
    This parameter is ignored if `client` is not None.
* Setting `client` during instantiation \
    When `None` (default), a Dask client is created using `n_workers` specified. \
    Else, any custom-configured Dask Client can be created and passed as the `client` argument to DEHB.
  
#### Using GPUs in a parallel run

Certain target function evaluations (especially for Deep Learning) require computations to be 
carried out on GPUs. The GPU devices are often ordered by device ID and if not configured, all 
spawned worker processes access these devices in the same order and can either run out of memory or
not exhibit parallelism.

For `n_workers>1` and when running on a single node (or local), the `single_node_with_gpus` can be 
passed to the `run()` call to DEHB. Setting it to `False` (default) has no effect on the default setup 
of the machine. Setting it to `True` will reorder the GPU device IDs dynamically by setting the environment 
variable `CUDA_VISIBLE_DEVICES` for each worker process executing a target function evaluation. The re-ordering 
is done in a manner that the first priority device is the one with the least number of active jobs assigned 
to it by that DEHB run.

To run the PyTorch MNIST example on a single node using 2 workers:  
```bash
python examples/03_pytorch_mnist_hpo.py --min_budget 1 --max_budget 3 \
  --verbose --runtime 60 --n_workers 2 --single_node_with_gpus
```

#### Multi-node runs

Multi-node parallelism is often contingent on the cluster setup to be deployed on. Dask provides useful 
frameworks to interface various cluster designs. As long as the `client` passed to DEHB during 
instantiation is of type `dask.distributed.Client`, DEHB can interact with this client and 
distribute its optimization process in a parallel manner. 

For instance, `Dask-CLI` can be used to create a `dask-scheduler` which can dump its connection 
details to a file on a cluster node accessible to all processes. Multiple `dask-worker` can then be
created to interface the `dask-scheduler` by connecting to the details read from the file dumped. Each
dask-worker can be triggered on any remote machine. Each worker can be configured as required, 
including mapping to specific GPU devices. 

Some helper scripts can be found [here](../utils/), that can be used as a reference to run DEHB in a multi-node 
manner on clusters managed by SLURM. (*not expected to work off-the-shelf*)

To run the PyTorch MNIST example on a multi-node setup using 4 workers:
```bash
bash utils/run_dask_setup.sh -f dask_dump/scheduler.json -e env_name -n 4
sleep 5
python examples/03_pytorch_mnist_hpo.py --min_budget 1 --max_budget 3 \
  --verbose --runtime 60 --scheduler_file dask_dump/scheduler.json 
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
title = {MO-DEHB: Evolutionary-based Hyperband for Multi-Objective Optimization},
author = {Noor Awad and Ayushi Sharma and Frank Hutter},
year = {2023},
keywords = {}
}