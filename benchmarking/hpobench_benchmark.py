import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from hpobench.benchmarks.ml.tabular_benchmark import TabularBenchmark
from hpobench.benchmarks.nas.nasbench_201 import Cifar10ValidNasBench201BenchmarkOriginal
from hpobench.benchmarks.surrogates.paramnet_benchmark import ParamNetReducedAdultOnStepsBenchmark
from utils import DEHBOptimizerBase


class DEHBOptimizerHPOBench(DEHBOptimizerBase):
    def _objective_function(self, config, fidelity):
        res = self.benchmark.objective_function(config,
                                                fidelity=
                                                {
                                                    self.fidelity_name: self.fidelity_type(fidelity),
                                                })
        return {
            "fitness": res["function_value"],
            "cost": res["cost"],
            "info": {"fidelity": fidelity},
        }

    def _get_config_space(self, seed):
        return self.benchmark.get_configuration_space(seed=seed)

    def _get_fidelity_range(self, benchmark, fidelity_name, seed, tabular):
        fidelity_space = benchmark.get_fidelity_space(seed)
        fidelity_param = fidelity_space.get_hyperparameter(fidelity_name)
        if tabular:
            min_fidelity = fidelity_param.sequence[0]
            max_fidelity = fidelity_param.sequence[-1]
        else:
            min_fidelity = fidelity_param.lower
            max_fidelity = fidelity_param.upper
        return min_fidelity, max_fidelity

    def _get_benchmark_and_fidelities(self, benchmark_name, seed):
        tabular = True
        if benchmark_name == "tab_nn":
            benchmark = TabularBenchmark(model="nn", rng=seed, task_id=31)
            fidelity_name = "iter"
            fidelity_type = int
        elif benchmark_name == "tab_lr":
            benchmark = TabularBenchmark(model="lr", rng=seed, task_id=31)
            fidelity_name = "iter"
            fidelity_type = int
        elif benchmark_name == "tab_rf":
            benchmark = TabularBenchmark(model="rf", rng=seed, task_id=31)
            fidelity_name = "n_estimators"
            fidelity_type = int
        elif benchmark_name == "tab_svm":
            benchmark = TabularBenchmark(model="svm", rng=seed, task_id=31)
            fidelity_name = "subsample"
            fidelity_type = float
        elif benchmark_name == "nasbench201":
            benchmark = Cifar10ValidNasBench201BenchmarkOriginal(rng=seed)
            fidelity_name = "epoch"
            fidelity_type = int
            tabular = False
        elif benchmark_name == "surrogate":
            benchmark = ParamNetReducedAdultOnStepsBenchmark(rng=seed)
            fidelity_name = "step"
            fidelity_type = int
            tabular = False
        else:
            raise ValueError(f"No benchmark '{benchmark_name}' found.")

        min_fidelity, max_fidelity = self._get_fidelity_range(benchmark, fidelity_name, seed, tabular)
        return benchmark, fidelity_name, (min_fidelity, max_fidelity), fidelity_type

def input_arguments():
    parser = argparse.ArgumentParser(description="Optimizing HPOBench using DEHB.")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used to create random seeds for experiments to run. Defaults to 0.",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=5,
        help="Number of random seeds to run. Defaults to 5.",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="*",
        default=["tab_nn", "tab_lr", "tab_rf", "tab_svm", "surrogate", "nasbench201"],
        help="Benchmarks to run DEHB on.",
        choices=["tab_nn", "tab_lr", "tab_rf", "tab_svm", "surrogate", "nasbench201"],
    )
    parser.add_argument(
        "--eta",
        type=int,
        default=3,
        help="Parameter for Hyperband controlling early stopping aggressiveness",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./hpobench_dehb",
        help="Directory for DEHB to write logs and outputs.",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="Number of CPU workers for DEHB to distribute function evaluations to.",
    ),
    parser.add_argument(
        "--brackets",
        type=int,
        default=None,
        help="Total number of brackets as budget to run DEHB.",
    )
    parser.add_argument(
        "--fevals",
        type=int,
        default=None,
        help="Total number of fevals as budget to run DEHB.",
    )
    parser.add_argument(
        "--walltime",
        type=int,
        default=None,
        help="Maxmimum walltime budget to run DEHB.",
    )
    parser.add_argument(
        "--ask_tell",
        action="store_true",
        default=False,
        help="Use the ask and tell interface.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        default=False,
        help="Restart the training and run again for the same budget.",
    )
    return parser.parse_args()

def main():
    args = input_arguments()
    dehb_params = {
        "eta": args.eta,
        "output_path": args.output_path,
        "n_workers": args.n_workers,
    }

    # Create random seeds from original seed
    rng = np.random.default_rng(args.seed)
    seeds = rng.integers(0, 2**32 - 1, size=args.n_seeds)

    results = {}
    for benchmark_name in args.benchmarks:
        trajectories = []
        for seed in seeds:
            print(f"Running benchmark {benchmark_name} on seed {seed}")
            np.random.seed(seed)
            random.seed(seed)
            dehb_params["seed"] = int(seed)
            dehb_optimizer = DEHBOptimizerHPOBench(
                dehb_params=dehb_params,
                fevals=args.fevals,
                brackets=args.brackets,
                walltime=args.walltime,
                use_ask_tell=args.ask_tell,
                use_restart=args.restart,
                benchmark_name=benchmark_name,
            )
            traj = dehb_optimizer.run()
            trajectories.append(traj)

            # Explicitly delete dehb_optimizer to free space
            del dehb_optimizer

        trajectories = np.array(trajectories)
        mean_trajectory = np.mean(trajectories, axis=0)
        std_trajectory = np.std(trajectories, axis=0)

        results[benchmark_name] = {
            "mean_trajectory": mean_trajectory,
            "std_trajectory": std_trajectory,
        }
    # Save benchmarking results to disc
    results_path = Path("benchmarking", "results", "current")
    for benchmark_name, result in results.items():
        base_path = results_path / benchmark_name
        base_path.mkdir(parents=True, exist_ok=True)

        traj_save_path = base_path / "traj.parquet.gzip"
        df_traj = pd.DataFrame.from_dict(result)
        df_traj.to_parquet(traj_save_path, compression="gzip")

if __name__ == "__main__":
    main()
