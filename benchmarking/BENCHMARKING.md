# Benchmarking DEHB

Benchmarking DEHB is crucial for ensuring consistent performance across different setups and configurations. We aim to benchmark DEHB on multiple HPOBench-benchmarks and MFPBench-benchmarks with different run setups:

1. Using `dehb.run`,
2. Using the Ask & Tell interface and
3. Restarting the optimization run after half the budget.

In the end, the results for the 3 different execution setups should be the same. With this setup guide, we encourage the developers of DEHB to continually benchmark their changes in order to ensure, that

- the inner workings of DEHB are not corrupted by checking the different execution setup results and
- that overall performance either remains the same, if no algortihmic changes have been made or is still comparable/better, if algorithmic changes have been made.

Please follow the installtion guide below, to benchmark your changes.

## Installation Guide HPOBench

The following guide walks you through installing hpobench and running the benchmarking script. Here, we assume that you execute the commands in your cloned DEHB repository.

### Create Virtual Environment

Before starting, please make sure you have clean virtual environment using python 3.8 ready. The following commands walk you through on how to do this with conda.

```shell
conda create --name dehb_hpo python=3.8
conda activate dehb_hpo
```

### Installing HPOBench

```shell
git clone https://github.com/automl/HPOBench.git
cd HPOBench
git checkout 47bf141 # Checkout specific commit
pip install .[ml_tabular_benchmarks]
cd ..
```

### Installing DEHB

There are some additional dependencies needed for plotting and table generation, therefore please install DEHB with the benchmarking options:

```shell
cd DEHB
pip install -e .[benchmarking,hpobench_benchmark]
```

### Running the Benchmarking Script

The benchmarking script is highly configurable and lets you choose between the budget types (`fevals`, `brackets` and `total_cost`), the execution setup (`run`(default), `ask_tell` and `restart`), the benchmarks used (`tab_nn`, `tab_rf`, `tab_svm`, `tab_lr`, `surrogate`, `nasbench201`) and the seeds used for each benchmark run (default: [0]).

```shell
python3.8 benchmarking/hpobench_benchmark.py --fevals 300 --ask_tell --restart --benchmarks tab_nn tab_rf tab_svm tab_lr surrogate nasbench201 --seed 0 --n_seeds 10 --output_path logs/hpobench_benchmarking
```

## Installation Guide MFPBench

The following guide walks you trough instaling mfpbench and running the benchmarking script. Here, we assume that you execute the commands in your cloned DEHB repository. Depending on the choice of benchmark, different requirements have to be installed, which are not compatible with one another. Thus we divide the setup into two sections, one for installing the JAHS benchmark and one for the PD1 benchmark. The MFHartmann benchmarks are work with both installations.

## JAHS Benchmark

### Create Virtual Environment

Before starting, please make sure you have clean virtual environment using python 3.8 ready. The following commands walk you through on how to do this with conda.

```shell
conda create --name dehb_jahs python=3.8
conda activate dehb_jahs
```

### Installing DEHB with MFPBench

There are some additional dependencies needed for plotting and table generation, therefore please install DEHB with the benchmarking options:

```shell
pip install -e .[benchmarking,jahs_benchmark]
```

### Downloading Benchmark Data

In order to run the benchmark, first we need to download the benchmark data:

```shell
python -m mfpbench download --benchmark jahs
```

### Running the Benchmarking Script

The setup is similar as in the HPOBench section, however under this installation only the `jahs` (joint architecture and hyperparameter search), `mfh3` and `mfh6` benchmarks are available.

```shell
python3.8 benchmarking/mfpbench_benchmark.py --fevals 300 --ask_tell --restart --benchmarks jahs mfh3 mfh6 --seed 0 --n_seeds 10 --output_path logs/jahs_benchmarking
```

## PD1 Benchmark

### Create Virtual Environment

Before starting, please make sure you have clean virtual environment using python 3.8 ready. The following commands walk you through on how to do this with conda.

```shell
conda create --name dehb_pd1 python=3.8
conda activate dehb_pd1
```

### Installing DEHB with MFPBench

There are some additional dependencies needed for plotting and table generation, therefore please install DEHB with the benchmarking options:

```shell
pip install -e .[benchmarking,pd1_benchmark]
```

### Downloading Benchmark Data

In order to run the benchmark, first we need to download the benchmark data:

```shell
python -m mfpbench download --benchmark pd1
```

### Running the Benchmarking Script

We currently support and use the PD1 benchmarks `cifar100_wideresnet_2048`, `imagenet_resnet_512`, `lm1b_transformer_2048` and `translatewmt_xformer_64`. Moreover, the `mfh3` and `mfh6` benchmarks are available.

```shell
python3.8 benchmarking/mfpbench_benchmark.py --fevals 300 --ask_tell --restart --benchmarks cifar100_wideresnet_2048 imagenet_resnet_512 lm1b_transformer_2048 translatewmt_xformer_64 mfh3 mfh6 --seed 0 --n_seeds 10 --output_path logs/pd1_benchmarks
```

## CountingOnes Benchmark

The CountingOnes benchmark is a synthetical benchmark and only depends on numpy, thus it can be used directly without any special setup.

### Running the Benchmarking Script

```shell
python benchmarking/countingones_benchmark.py --seed 0 --n_seeds 10 --fevals 300 --output_path logs/countingones --n_continuous 50 --n_categorical 50
```
