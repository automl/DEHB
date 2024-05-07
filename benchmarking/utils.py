from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dehb import DEHB


def plot_incumbent_traj(trajectories: list, output_path: Path, benchmark_name: str):
    plt.clf()
    trajectories = np.array(trajectories)
    traj_length = len(trajectories[0])
    mean_trajectory = np.mean(trajectories, axis=0)
    std_trajectory = np.std(trajectories, axis=0)
    x_fevals = np.arange(traj_length)
    plt.plot(x_fevals, mean_trajectory, label="Mean incumbent trajectory")
    plt.fill_between(x_fevals, mean_trajectory - std_trajectory, mean_trajectory + std_trajectory, alpha=0.3)

    plt.xlabel("Function evaluation")
    plt.ylabel("Incumbent score")
    plt.title(f"Mean and Standard Deviation of Incumbent Trajectories on {benchmark_name}")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig(output_path)

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
        # Implement int subclass
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