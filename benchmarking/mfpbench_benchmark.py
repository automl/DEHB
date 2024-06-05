import argparse
from pathlib import Path

import mfpbench
import numpy as np
import pandas as pd
from utils import DEHBOptimizerBase


class DEHBOptimizerMFPBench(DEHBOptimizerBase):
    def _objective_function(self, config, fidelity):
        res = self.benchmark.query(config, at=self.fidelity_type(fidelity))
        return {
            "fitness": res.error,
            "cost": res.cost,
        }

    def _get_config_space(self, seed):
        return self.benchmark.space
    def _get_fidelity_range(self, benchmark):
        return benchmark.start, benchmark.end

    def _get_benchmark_and_fidelities(self, benchmark_name, seed):
        if benchmark_name == "jahs":
            benchmark = mfpbench.get(name="jahs", task_id="CIFAR10", seed=seed)
        else:
            benchmark = mfpbench.get(name=benchmark_name, seed=seed)

        fidelity_name = benchmark.fidelity_name
        fidelity_type = int

        min_fidelity, max_fidelity = self._get_fidelity_range(benchmark)
        return benchmark, fidelity_name, (min_fidelity, max_fidelity), fidelity_type

def input_arguments():
    parser = argparse.ArgumentParser(description="Optimizing MFPBench using DEHB.")
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
        default=["jahs"],
        help="Benchmarks to run DEHB on.",
        choices=["jahs", "mfh3", "mfh6", "cifar100_wideresnet_2048", "imagenet_resnet_512",
                 "lm1b_transformer_2048", "translatewmt_xformer_64"],
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
        default="./mfpbench_dehb",
        help="Directory for DEHB to write logs and outputs.",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="Number of CPU workers for DEHB to distribute function evaluations to.",
    ),
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Decides verbosity of DEHB optimization.",
    )
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
            dehb_params["seed"] = seed
            dehb_optimizer = DEHBOptimizerMFPBench(
                dehb_params=dehb_params,
                fevals=args.fevals,
                brackets=args.brackets,
                walltime=args.walltime,
                use_ask_tell=args.ask_tell,
                use_restart=args.restart,
                benchmark_name=benchmark_name,
                verbose=args.verbose,
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
