import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from hpobench.benchmarks.rl.cartpole import CartpoleReduced
from hpobench.benchmarks.ml.tabular_benchmark import TabularBenchmark
from hpobench.benchmarks.nas.nasbench_201 import Cifar10ValidNasBench201BenchmarkOriginal
from markdown_table_generator import generate_markdown, table_from_string_list
from src.dehb import DEHB


def objective_function(config, fidelity, b):
    result_dict = b.objective_function(config, fidelity={"iter": int(fidelity)})
    return {
        "fitness": result_dict["function_value"],
        "cost": result_dict["cost"],
        "info": {"fidelity": fidelity},
    }

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
        default=["ml"],
        help="Benchmarks to run DEHB on.",
        choices=["ml", "rl", "nas"],
    )
    parser.add_argument(
        "--min_fidelity",
        type=float,
        default=3,
        help="Minimum fidelity (epoch length)",
    )
    parser.add_argument(
        "--max_fidelity",
        type=float,
        default=27,
        help="Maximum fidelity (epoch length)",
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
        help="Directory for DEHB to write logs and outputs",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="Number of CPU workers for DEHB to distribute function evaluations to",
    ),
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Decides verbosity of DEHB optimization",
    )
    parser.add_argument(
        "--brackets",
        type=int,
        default=None,
        help="Total number of brackets as fidelity to run DEHB",
    )
    parser.add_argument(
        "--fevals",
        type=int,
        default=None,
        help="Total number of fevals as fidelity to run DEHB",
    )
    parser.add_argument(
        "--ask_tell",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use ask and tell interface",
    )
    parser.add_argument(
        "--restart",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use ask and tell interface",
    )
    return parser.parse_args()

def get_benchmark(benchmark, seed):
    if benchmark == "ml":
        return TabularBenchmark(rng=seed, task_id=31, model="nn")
    elif benchmark == "rl":
        return CartpoleReduced(rng=seed)
    elif benchmark == "nas":
        return Cifar10ValidNasBench201BenchmarkOriginal(rng=seed)
    raise ValueError(f"No benchmark '{benchmark}' found.")

def run_for(dehb, benchmark, brackets=None, fevals=None, ask_tell=False, verbose=True):
    if brackets:
        traj, runtime, history = dehb.run(brackets=brackets, verbose=verbose, b=benchmark)
    elif ask_tell:
        for _i in range(fevals):
            job_info = dehb.ask()
            res = objective_function(job_info["config"], job_info["fidelity"], benchmark)
            dehb.tell(job_info, res)
        traj = dehb.traj
        # Log the incumbent
        config = dehb.vector_to_configspace(dehb.inc_config)
        dehb.logger.info("Incumbent score: {}".format(dehb.inc_score))
        for k, v in config.get_dictionary().items():
                    dehb.logger.info("{}: {}".format(k, v))
    else:
        traj, runtime, history = dehb.run(fevals=fevals, verbose=verbose, b=benchmark)

    return traj

def main():
    args = input_arguments()
    scores = {}
    table = [["Benchmark", "Score (mean ± std)"]]
    for benchmark in args.benchmarks:
        scores[benchmark] = []
        for seed in args.seeds:
            b = get_benchmark(benchmark, seed)
            cs = b.get_configuration_space(seed=seed)
            dimensions = len(cs.get_hyperparameters())

            ###########################
            # DEHB optimization block #
            ###########################
            dehb = DEHB(
                f=objective_function,
                cs=cs,
                dimensions=dimensions,
                min_fidelity=args.min_fidelity,
                max_fidelity=args.max_fidelity,
                eta=args.eta,
                output_path=args.output_path,
                n_workers=args.n_workers,
                save_freq="incumbent",
                seed=seed,
            )

            traj = run_for(dehb, b, args.brackets, args.fevals, args.ask_tell, args.verbose)

            if args.restart:
                dehb = DEHB(
                    f=objective_function,
                    cs=cs,
                    dimensions=dimensions,
                    min_fidelity=args.min_fidelity,
                    max_fidelity=args.max_fidelity,
                    eta=args.eta,
                    output_path=args.output_path,
                    n_workers=args.n_workers,
                    save_freq="incumbent",
                    seed=seed,
                    resume=True,
                )
                traj = run_for(dehb, b, args.brackets, args.fevals, args.ask_tell, args.verbose)

            inc_config, inc_value = dehb.get_incumbents()
            scores[benchmark].append(inc_value)
        mean_score = np.mean(scores[benchmark])
        std_score = np.std(scores[benchmark])
        table.append([benchmark, f"{mean_score:.3e} ± {std_score:.3e}"])

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
