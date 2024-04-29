import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from hpobench.benchmarks.ml.lr_benchmark import LRBenchmark
from hpobench.benchmarks.ml.nn_benchmark import NNBenchmark
from hpobench.benchmarks.ml.rf_benchmark import RandomForestBenchmark
from hpobench.benchmarks.ml.svm_benchmark import SVMBenchmark
from hpobench.benchmarks.nas.nasbench_201 import Cifar10ValidNasBench201BenchmarkOriginal
from hpobench.benchmarks.surrogates.paramnet_benchmark import ParamNetReducedAdultOnStepsBenchmark
from markdown_table_generator import generate_markdown, table_from_string_list
from src.dehb import DEHB


def get_fidelity_range(benchmark, fidelity_name, seed):
    fidelity_space = benchmark.get_fidelity_space(seed)
    fidelity_param = fidelity_space.get_hyperparameter(fidelity_name)
    min_fidelity = fidelity_param.lower
    max_fidelity = fidelity_param.upper
    return min_fidelity, max_fidelity

def get_benchmark_and_fidelities(benchmark_name, seed):
    if benchmark_name == "tab_nn":
        benchmark = NNBenchmark(rng=seed, task_id=31)
        fidelity_name = "iter"
        fidelity_type = int
    elif benchmark_name == "tab_lr":
        benchmark = LRBenchmark(rng=seed, task_id=31)
        fidelity_name = "iter"
        fidelity_type = int
    elif benchmark_name == "tab_rf":
        benchmark = RandomForestBenchmark(rng=seed, task_id=31)
        fidelity_name = "n_estimators"
        fidelity_type = int
    elif benchmark_name == "tab_svm":
        benchmark = SVMBenchmark(rng=seed, task_id=31)
        fidelity_name = "subsample"
        fidelity_type = float
    elif benchmark_name == "nas":
        benchmark = Cifar10ValidNasBench201BenchmarkOriginal(rng=seed)
        fidelity_name = "epoch"
        fidelity_type = int
    elif benchmark_name == "surrogate":
        benchmark = ParamNetReducedAdultOnStepsBenchmark(rng=seed)
        fidelity_name = "step"
        fidelity_type = int
    else:
        raise ValueError(f"No benchmark '{benchmark_name}' found.")

    min_fidelity, max_fidelity = get_fidelity_range(benchmark, fidelity_name, seed)
    return benchmark, fidelity_name, (min_fidelity, max_fidelity), fidelity_type

class DEHBOptimizer():
    def __init__(self, dehb_params, fevals, brackets, walltime, use_ask_tell, use_restart,
                 benchmark_name, verbose) -> None:
        self.verbose = verbose
        self.fevals = fevals
        self.brackets = brackets
        self.walltime = walltime
        self.use_ask_tell = use_ask_tell
        self.use_restart = use_restart
        b, fid_name, (min_fid, max_fid), fid_type = get_benchmark_and_fidelities(benchmark_name,
                                                                                 dehb_params["seed"])
        self.benchmark = b
        self.config_space = self.benchmark.get_configuration_space(seed=dehb_params["seed"])
        dehb_params["dimensions"] = len(self.config_space.get_hyperparameters())
        self.fidelity_name = fid_name
        self.fidelity_type = fid_type
        dehb_params["min_fidelity"] = min_fid
        dehb_params["max_fidelity"] = max_fid
        self.dehb_params = dehb_params
        self.dehb = self._initialize_dehb()

    def _initialize_dehb(self, resume=False):
        return DEHB(
            f=self._objective_function,
            cs=self.config_space,
            dimensions=self.dehb_params["dimensions"],
            min_fidelity=self.dehb_params["min_fidelity"],
            max_fidelity=self.dehb_params["max_fidelity"],
            eta=self.dehb_params["eta"],
            output_path=self.dehb_params["output_path"],
            n_workers=self.dehb_params["n_workers"],
            save_freq="incumbent",
            seed=self.dehb_params["seed"],
            resume=resume,
        )

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

    def _run_for(self, fevals=None, brackets=None, walltime=None):
        if self.use_ask_tell:
            for _i in range(fevals):
                job_info = self.dehb.ask()
                res = self._objective_function(job_info["config"], job_info["fidelity"])
                self.dehb.tell(job_info, res)
            return self.dehb.traj

        traj,_ ,_ = self.dehb.run(fevals=fevals, brackets=brackets,
                                  total_cost=walltime, verbose=self.verbose)
        return traj
    def _run_with_restart(self):
        pre_restart_brackets, post_restart_brackets = None, None
        pre_restart_fevals, post_restart_fevals = None, None
        pre_restart_walltime, post_restart_walltime = None, None
        if self.brackets:
                pre_restart_brackets = self.brackets // 2
                post_restart_brackets = self.brackets - pre_restart_brackets
        elif self.fevals:
            pre_restart_fevals = self.fevals // 2
            post_restart_fevals = self.fevals - pre_restart_fevals
        elif self.walltime:
            pre_restart_walltime = self.walltime / 2
            post_restart_walltime = self.walltime / 2

        # Run for half the budget
        self._run_for(fevals=pre_restart_fevals, brackets=pre_restart_brackets,
                     walltime=pre_restart_walltime)
        # Reinitialize DEHB (resume)
        self.dehb = self._initialize_dehb(resume=True)
        # Run for remaining budget
        return self._run_for(fevals=post_restart_fevals, brackets=post_restart_brackets,
                            walltime=post_restart_walltime)

    def run(self):
        if self.use_restart:
            return self._run_with_restart()

        return self._run_for(self.fevals, self.brackets, self.walltime)


def input_arguments():
    parser = argparse.ArgumentParser(description="Optimizing HPOBench using DEHB.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=[1],
        metavar="S",
        help="Random seeds to benchmark DEHB on (default: 1)",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="*",
        default=["tab_nn"],
        help="Benchmarks to run DEHB on.",
        choices=["tab_nn", "tab_lr", "tab_rf", "tab_svm", "surrogate", "nas"],
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
        "--verbose",
        action="store_true",
        default=True,
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
    scores = {}
    table = [["Benchmark", "Score (mean ± std)"]]
    for benchmark_name in args.benchmarks:
        scores[benchmark_name] = []
        for seed in args.seeds:
            print(f"Running benchmark {benchmark_name} on seed {seed}")
            dehb_params["seed"] = seed
            dehb_optimizer = DEHBOptimizer(
                dehb_params=dehb_params,
                fevals=args.fevals,
                brackets=args.brackets,
                walltime=args.walltime,
                use_ask_tell=args.ask_tell,
                use_restart=args.restart,
                benchmark_name=benchmark_name,
                verbose=args.verbose
            )
            traj = dehb_optimizer.run()
            inc_config, inc_value = dehb_optimizer.dehb.get_incumbents()
            scores[benchmark_name].append(inc_value)
        mean_score = np.mean(scores[benchmark_name])
        std_score = np.std(scores[benchmark_name])
        table.append([benchmark_name, f"{mean_score:.3e} ± {std_score:.3e}"])

    markdown_path = Path(args.output_path) / "benchmark_results.md"
    md_table = table_from_string_list(table)
    markdown = generate_markdown(md_table)
    with markdown_path.open("w") as f:
         f.write(markdown)

    # Plot incumbent trajectory
    fig, ax = plt.subplots()
    ax.plot(range(len(traj)), traj)
    ax.set(ylabel="Incumbent score", xlabel="Step")

    plt.savefig("hpobench_optim.png")

if __name__ == "__main__":
    main()
