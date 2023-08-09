# DEHB: Evolutionary Hyperband for Scalable, Robust and Efficient Hyperparameter Optimization
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://github.com/automl/DEHB/actions/workflows/pytest.yml/badge.svg)](https://github.com/automl/DEHB/actions/workflows/pytest.yml)
[![docs](https://github.com/automl/DEHB/actions/workflows/docs.yml/badge.svg)](https://automl.github.io/DEHB/)
[![Coverage Status](https://coveralls.io/repos/github/automl/DEHB/badge.svg)](https://coveralls.io/github/automl/DEHB)
[![PyPI](https://img.shields.io/pypi/v/dehb)](https://pypi.org/project/dehb/)
[![Static Badge](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20-blue)](https://pypi.org/project/dehb/)
### Installation
```bash
# from pypi
pip install dehb

# to run examples, install from github
git clone https://github.com/automl/DEHB.git
pip install -e DEHB  # -e stands for editable, lets you modify the code and rerun things
```

### Tutorials/Example notebooks

* [00 - A generic template to use DEHB for multi-fidelity Hyperparameter Optimization](examples/00_interfacing_DEHB.ipynb)
* [01 - Using DEHB to optimize 4 hyperparameters of a Scikit-learn's Random Forest on a classification dataset](examples/01_Optimizing_RandomForest_using_DEHB.ipynb)
* [02 - Optimizing Scikit-learn's Random Forest without using ConfigSpace to represent the hyperparameter space](examples/02_using%20DEHB_without_ConfigSpace.ipynb)
* [03 - Hyperparameter Optimization for MNIST in PyTorch](examples/03_pytorch_mnist_hpo.py)
* [04 - A generic template to use MODEHB for multi-objectives Hyperparameter Optimization](examples/04_mo_pytorch_mnist_hpo.py)

To run PyTorch example: (*note additional requirements*) 
```bash
python examples/03_pytorch_mnist_hpo.py \
     --min_budget 1 --max_budget 3 --verbose --runtime 60
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

Some helper scripts can be found [here](utils/), that can be used as a reference to run DEHB in a multi-node 
manner on clusters managed by SLURM. (*not expected to work off-the-shelf*)

To run the PyTorch MNIST example on a multi-node setup using 4 workers:
```bash
bash utils/run_dask_setup.sh -f dask_dump/scheduler.json -e env_name -n 4
sleep 5
python examples/03_pytorch_mnist_hpo.py --min_budget 1 --max_budget 3 \
  --verbose --runtime 60 --scheduler_file dask_dump/scheduler.json 
```

### Running DEHB to optimize multiple objectives
To run multi-objective optimization we require 1 extra parameter mo_strategy: we provide MO-optimization using Non-dominated sorted (NDS) with crowding distance (NSGA-II) and NDS with eps-net(EPSNET). Find 04_mo_pytorch_mnist_hpo.py example to help you to get started

### DEHB Hyperparameters

*We recommend the default settings*.
The default settings were chosen based on ablation studies over a collection of diverse problems 
and were found to be *generally* useful across all cases tested. 
However, the parameters are still available for tuning to a specific problem.

The Hyperband components:
* *min\_budget*: Needs to be specified for every DEHB instantiation and is used in determining 
the budget spacing for the problem at hand.
* *max\_budget*: Needs to be specified for every DEHB instantiation. Represents the full-budget 
evaluation or the actual black-box setting.
* *eta*: (default=3) Sets the aggressiveness of Hyperband's aggressive early stopping by retaining
1/eta configurations every round
  
The DE components:
* *strategy*: (default=`rand1_bin`) Chooses the mutation and crossover strategies for DE. `rand1` 
represents the *mutation* strategy while `bin` represents the *binomial crossover* strategy. \
  Other mutation strategies include: {`rand2`, `rand2dir`, `best`, `best2`, `currenttobest1`, `randtobest1`}\
  Other crossover strategies include: {`exp`}\
  Mutation and crossover strategies can be combined with a `_` separator, for e.g.: `rand2dir_exp`.
* *mutation_factor*: (default=0.5) A fraction within [0, 1] weighing the difference operation in DE
* *crossover_prob*: (default=0.5) A probability within [0, 1] weighing the traits from a parent or the mutant

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

@online{Awad-arXiv-2023,
title = {MO-DEHB: Evolutionary-based Hyperband for Multi-Objective Optimization},
author = {Noor Awad and Ayushi Sharma and Frank Hutter},
year = {2023},
keywords = {}
}
