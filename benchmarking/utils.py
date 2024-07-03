from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dehb import DEHB


def create_plot_for_benchmark(results: dict, output_path: Path,
                              benchmark_name: str):
    plt.clf()
    results = dict(reversed(sorted(results.items())))
    for version, data in results.items():
        mean_trajectory = data["mean_trajectory"]
        std_trajectory = data["std_trajectory"]
        traj_length = len(mean_trajectory)
        x_fevals = np.arange(traj_length)
        plt.plot(x_fevals, mean_trajectory, label=version)
        plt.fill_between(x_fevals, mean_trajectory - std_trajectory, mean_trajectory + std_trajectory,
                        alpha=0.3)

    plt.xlabel("Function evaluation")
    plt.ylabel("Incumbent score")
    plt.title(f"Incumbent Trajectories on {benchmark_name}")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig(output_path / f"{benchmark_name}_traj.png")

def create_table_for_benchmark(results: dict) -> list:
    table = []
    header = ["DEHB Version"]
    for budget in [.2, .4, .6, .8, 1]:
        arbitrary_key = list(results.keys())[0]
        traj_length = len(results[arbitrary_key]["mean_trajectory"])
        header.append(str(int(budget * traj_length)))
    table.append(header)
    results = dict(reversed(sorted(results.items())))
    for version in results:
        row = [version]
        mean_traj = results[version]["mean_trajectory"]
        std_traj = results[version]["std_trajectory"]
        for budget in [.2, .4, .6, .8, 1]:
            traj_idx = int(budget * len(mean_traj)) - 1
            row.append(f"{mean_traj[traj_idx]:.3e} ± {std_traj[traj_idx]:.3e}")
        table.append(row)

    return table

class DEHBOptimizerBase():
    def __init__(self, dehb_params, fevals, brackets, walltime, use_ask_tell, use_restart,
                 benchmark_name, verbose) -> None:
        self.verbose = verbose
        self.fevals = fevals
        self.brackets = brackets
        self.walltime = walltime
        self.use_ask_tell = use_ask_tell
        self.use_restart = use_restart
        b, fid_name, (min_fid, max_fid), fid_type = self._get_benchmark_and_fidelities(benchmark_name,
                                                                                 dehb_params["seed"])
        self.benchmark = b
        self.config_space = self._get_config_space(dehb_params["seed"])
        dehb_params["dimensions"] = len(self.config_space.get_hyperparameters())
        self.fidelity_name = fid_name
        self.fidelity_type = fid_type
        dehb_params["min_fidelity"] = min_fid
        dehb_params["max_fidelity"] = max_fid
        self.dehb_params = dehb_params
        self.dehb = self._initialize_dehb()

    def _get_benchmark_and_fidelities(self, benchmark_name, seed):
        # Implement in subclass
        raise NotImplementedError()

    def _get_config_space(self, seed):
        # Implement in subclass
        raise NotImplementedError()

    def _objective_function(self, config, fidelity):
        # Implement in subclass
        raise NotImplementedError()

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

    def _run_for(self, fevals=None, brackets=None, walltime=None):
        if self.use_ask_tell:
            for _i in range(fevals):
                job_info = self.dehb.ask()
                res = self._objective_function(job_info["config"], job_info["fidelity"])
                self.dehb.tell(job_info, res)
            self.dehb.save()
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