# Benchmarking DEHB
Benchmarking DEHB is crucial for ensuring consistent performance across different setups and configurations. We aim to benchmark DEHB on multiple HPOBench-benchmarks and MFPBench-benchmarks with different run setups:

1. Using `dehb.run`,
2. Using the Ask & Tell interface and
3. Restarting the optimization run after half the budget.

In the end, the results for the 3 different execution setups should be the same. With this setup guide, we encourage the developers of DEHB to continually benchmark their changes in order ensure, that 

- the inner workings of DEHB are not corrupted by checking the different execution setup results and
- that overall performance either remains the same, if no algortihmic changes have been made or is still comparable/better, if algorithmic changes have been made.

Please follow the installtion guide below, to benchmark your changes.

## Installation Guide HPOBench
The following guide walks you throuh installing hpobench and running the benchmarking script. Here, we assume that you execute the commands in your cloned DEHB repository and you have a clean (virtual) python 3.8 environment.
### Installing HPOBench
```
git clone https://github.com/automl/HPOBench.git
cd HPOBench 
pip install .[ml_tabular_benchmarks]
cd ..
```
### Installing DEHB
There are some additional dependencies needed for plotting and table generation, therefore please install DEHB with the benchmarking options:
```
pip install -e .[benchmarking,hpobench_benchmark]
```
### Running the Benchmarking Script
The benchmarking script is highly configurable and lets you choose between the budget types (`fevals`, `brackets` and `total_cost`), the execution setup (`run`(default), `ask_tell` and `restart`), the benchmarks used (`tab_nn`, `tab_rf`, `tab_svm`, `tab_lr`, `surrogate`, `nas`) and the seeds used for each benchmark run (default: [0]). 
```
python3.8 benchmarking/hpobench_benchmark.py --fevals 300 --ask_tell --restart --benchmarks tab_nn --seeds 1 2 3 4 5 --output_path logs/hpobench_benchmarking
```

## Installation Guide MFPBench
The following guide walks you trough instaling mfpbench and running the benchmarking script. Here, we assume that you execute the commands in your cloned DEHB repository and you have a clean (virtual) python 3.8 environment.

### Installing DEHB with MFPBench
There are some additional dependencies needed for plotting and table generation, therefore please install DEHB with the benchmarking options:
```
pip install -e .[benchmarking,mfpbench_benchmark]
```

### Downloading Benchmark Data
In order to run the benchmark, first we need to download the benchmark data:
```
python -m mfpbench download --benchmark jahs
```
### Running the Benchmarking Script
The setup is similar as in the HPOBench section, however currently the only available benchmark is `jahs` (joint architecture and hyperparameter search).
```
python3.8 benchmarking/mfpbench_benchmark.py --fevals 300 --ask_tell --restart --benchmarks jahs --seeds 1 2 3 4 5 --output_path logs/mfpbench_benchmarking
```