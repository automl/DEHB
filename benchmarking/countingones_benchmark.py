import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from counting_ones.counting_ones import CountingOnes
from utils import DEHBOptimizerBase


class DEHBOptimizerCountingOnes(DEHBOptimizerBase):
    def __init__(self, dehb_params, fevals, brackets, walltime, use_ask_tell, use_restart,
                 benchmark_name, n_continuous, n_categorical) -> None:
        self.n_continuous = n_continuous
        self.n_categorical = n_categorical
        super().__init__(dehb_params=dehb_params, fevals=fevals, brackets=brackets,
                         walltime=walltime, use_ask_tell=use_ask_tell, use_restart=use_restart,
                         benchmark_name=benchmark_name)

    def _objective_function(self, config, fidelity):
        res = self.benchmark.objective_function(config,
                                                budget=self.fidelity_type(fidelity))
        return {
            "fitness": res["function_value"],
            "cost": fidelity,
            "info": {"fidelity": fidelity},
        }

    def _get_config_space(self, seed):
        return self.benchmark.get_configuration_space(n_continuous=self.n_continuous,
                                                      n_categorical=self.n_categorical,
                                                      seed=seed)

    def _get_benchmark_and_fidelities(self, benchmark_name, seed):
        self.benchmark = CountingOnes()
        cs = self._get_config_space(seed)
        dimensions = len(cs.get_hyperparameters())
        min_fidelity = 576 / dimensions
        max_fidelity = 93312 / dimensions
        return self.benchmark, None, (min_fidelity, max_fidelity), int

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
        "--n_continuous",
        type=int,
        default=100,
        help="Number of continuous hyperparameters. Defaults to 100.",
    )
    parser.add_argument(
        "--n_categorical",
        type=int,
        default=100,
        help="Number of categorical hyperparameters. Defaults to 100.",
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
        default="./countingones",
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

    benchmark_name = "counting_ones"
    results = {}
    trajectories = []
    for seed in seeds:
        print(f"Running benchmark {benchmark_name} on seed {seed}")
        np.random.seed(seed)
        random.seed(seed)
        dehb_params["seed"] = int(seed)
        dehb_optimizer = DEHBOptimizerCountingOnes(
            dehb_params=dehb_params,
            fevals=args.fevals,
            brackets=args.brackets,
            walltime=args.walltime,
            use_ask_tell=args.ask_tell,
            use_restart=args.restart,
            benchmark_name=benchmark_name,
            n_continuous=args.n_continuous,
            n_categorical=args.n_categorical,
        )
        traj = dehb_optimizer.run()
        trajectories.append(traj)

        # Explicitly delete dehb_optimizer to free space
        del dehb_optimizer

    trajectories = np.array(trajectories)
    # Calculate normalized regret
    mean_trajectory = np.mean(trajectories, axis=0) / (args.n_continuous + args.n_categorical)
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
