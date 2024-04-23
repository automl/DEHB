# Benchmarking DEHB
Benchmarking DEHB is crucial for ensuring consistent performance across different setups and configurations. We aim to benchmark DEHB on multiple HPOBench-benchmarks with different run setups:

1. Using `dehb.run`,
2. Using the Ask & Tell interface and
3. Restarting the optimization run after half the budget.

In the end, the results for the 3 different execution setups should be the same. With this setup guide, we encourage the developers of DEHB to continually benchmark their changes in order ensure, that 

- the inner workings of DEHB are not corrupted by checking the different execution setup results and
- that overall performance either remains the same, if no algortihmic changes have been made or is still comparable/better, if algorithmic changes have been made.

Please follow the installtion guide below, to benchmark your changes.

## Installation Guide
The following guide walks you throuh on how to install hpobench and run the benchmarking script. Here, we assume that you execute the commands in your cloned DEHB repository.
### Installing HPOBench
```
git clone https://github.com/automl/HPOBench.git
cd HPOBench 
pip install .[ml_tabular_benchmarks]
cd ..
```
### Installing DEHB
There are some additional dependencies needed for plotting and table generation, therefore please install DEHB with the benchmarking option:
```
pip install -e .[benchmarks]
```
### Running the Benchmarking Script
The benchmarking script is highly configurable and lets you choose between the budget types (`fevals`, `brackets` and `total_cost`), the execution setup (`run`(default), `ask_tell` and `restart`), the benchmarks used (`ml`, `nas`) and the seeds used for each benchmark run (default: [0]). 
```
python3.9 hpobench_benchmark.py --fevals 300 --ask_tell --restart --benchmarks ml nas --seeds 1 2 3 4 5 --output_path logs/hpobench_benchmarking
```