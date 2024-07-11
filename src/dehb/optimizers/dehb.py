import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from threading import Timer
from typing import List, Tuple, Union

import ConfigSpace
import numpy as np
import pandas as pd
from distributed import Client
from loguru import logger

from ..utils import ConfigRepository, SHBracketManager
from .de import AsyncDE

_logger_props = {
    "format": "{time} {level} {message}",
    "enqueue": True,
    "rotation": "500 MB",
}


class DEHBBase:
    def __init__(self, cs=None, f=None, dimensions=None, mutation_factor=None,
                 crossover_prob=None, strategy=None, min_fidelity=None,
                 max_fidelity=None, eta=None, min_clip=None, max_clip=None, seed=None,
                 boundary_fix_type="random", max_age=np.inf, resume=False, **kwargs):
        # Check for deprecated parameters
        if "max_budget" in kwargs or "min_budget" in kwargs:
            raise TypeError("Parameters min_budget and max_budget have been deprecated since " \
                            "v0.1.0. Please use the new parameters min_fidelity and max_fidelity " \
                            "or downgrade to a version prior to v0.1.0")
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**32 - 1))
        elif isinstance(seed, np.random.Generator):
            seed = int(seed.integers(0, 2**32 - 1))

        assert isinstance(seed, int)
        self._original_seed = seed
        self.rng = np.random.default_rng(self._original_seed)

        # Miscellaneous
        self._setup_logger(resume, kwargs)
        self.config_repository = ConfigRepository()

        # Benchmark related variables
        self.cs = cs
        self.use_configspace = True if isinstance(self.cs, ConfigSpace.ConfigurationSpace) else False
        if self.use_configspace:
            self.cs.seed(self._original_seed)
            self.dimensions = len(self.cs.get_hyperparameters())
        elif dimensions is None or not isinstance(dimensions, (int, np.integer)):
            assert "Need to specify `dimensions` as an int when `cs` is not available/specified!"
        else:
            self.dimensions = dimensions
        self.f = f

        # DE related variables
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.strategy = strategy
        self.fix_type = boundary_fix_type
        self.max_age = max_age
        self.de_params = {
            "mutation_factor": self.mutation_factor,
            "crossover_prob": self.crossover_prob,
            "strategy": self.strategy,
            "configspace": self.use_configspace,
            "boundary_fix_type": self.fix_type,
            "max_age": self.max_age,
            "cs": self.cs,
            "dimensions": self.dimensions,
            "f": f,
        }

        # Hyperband related variables
        self.min_fidelity = min_fidelity
        self.max_fidelity = max_fidelity
        if self.max_fidelity <= self.min_fidelity:
            self.logger.error("Only (Max Fidelity > Min Fidelity) is supported for DEHB.")
            if self.max_fidelity == self.min_fidelity:
                self.logger.error(
                    "If you have a fixed fidelity, " \
                    "you can instead run DE. For more information checkout: " \
                    "https://automl.github.io/DEHB/references/de")
            raise AssertionError()
        self.eta = eta
        self.min_clip = min_clip
        self.max_clip = max_clip

        # Precomputing fidelity spacing and number of configurations for HB iterations
        self._pre_compute_fidelity_spacing()

        # Updating DE parameter list
        self.de_params.update({"output_path": self.output_path})

        # Global trackers
        self.population = None
        self.fitness = None
        self.inc_score = np.inf
        self.inc_config = None
        self.history = []

    def _setup_logger(self, resume, kwargs):
        """Sets up the logger."""
        log_level = kwargs["log_level"] if "log_level" in kwargs else "WARNING"
        _logger_props["level"] = log_level
        logger.configure(handlers=[{"sink": sys.stdout, "level": log_level}])
        self.output_path = Path(kwargs["output_path"]) if "output_path" in kwargs else Path("./")
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        # Only append to log if resuming an optimization run, else overwrite
        _logger_props["mode"] = "a" if resume else "w"
        self.log_filename = f"{self.output_path}/dehb.log"
        self.logger.add(
            self.log_filename,
            **_logger_props,
        )

    def _pre_compute_fidelity_spacing(self):
        self.max_SH_iter = None
        self.fidelities = None
        if self.min_fidelity is not None and \
           self.max_fidelity is not None and \
           self.eta is not None:
            self.max_SH_iter = -int(np.log(self.min_fidelity / self.max_fidelity) / np.log(self.eta)) + 1
            self.fidelities = self.max_fidelity * np.power(self.eta,
                                                     -np.linspace(start=self.max_SH_iter - 1,
                                                                  stop=0, num=self.max_SH_iter))

    def reset(self, *, reset_seeds: bool = True):
        self.inc_score = np.inf
        self.inc_config = None
        self.population = None
        self.fitness = None
        self.traj = []
        self.runtime = []
        self.history = []
        if reset_seeds:
            if isinstance(self.cs, ConfigSpace.ConfigurationSpace):
                self.cs.seed(self._original_seed)
            self.rng = np.random.default_rng(self._original_seed)
        self.logger.info("\n\nRESET at {}\n\n".format(time.strftime("%x %X %Z")))

    def _init_population(self):
        raise NotImplementedError("Redefine!")

    def _get_next_iteration(self, iteration: int) -> Tuple[np.array, np.array]:
        """Computes the Successive Halving spacing.

        Given the iteration index, computes the fidelity spacing to be used and
        the number of configurations to be used for the SH iterations.

        Args:
            iteration (int): Iteration index.

        Returns:
            A tuple containing number of configurations in the bracket
            and the respective fidelities
        """
        # number of 'SH runs'
        s = self.max_SH_iter - 1 - (iteration % self.max_SH_iter)
        # fidelity spacing for this iteration
        fidelities = self.fidelities[(-s-1):]
        # number of configurations in that bracket
        n0 = int(np.floor((self.max_SH_iter)/(s+1)) * self.eta**s)
        ns = [max(int(n0*(self.eta**(-i))), 1) for i in range(s+1)]
        if self.min_clip is not None and self.max_clip is not None:
            ns = np.clip(ns, a_min=self.min_clip, a_max=self.max_clip)
        elif self.min_clip is not None:
            ns = np.clip(ns, a_min=self.min_clip, a_max=np.max(ns))

        return ns, fidelities

    def get_incumbents(self) -> Tuple[Union[dict, ConfigSpace.Configuration], float]:
        """Retrieve current incumbent configuration and score.
        
        Returns:
            Tuple containing incumbent configuration and score.
        """
        if self.use_configspace:
            return self.vector_to_configspace(self.inc_config), self.inc_score
        return self.inc_config, self.inc_score

    def _f_objective(self):
        raise NotImplementedError("The function needs to be defined in the sub class.")

    def run(self):
        raise NotImplementedError("The function needs to be defined in the sub class.")


class DEHB(DEHBBase):
    def __init__(self, cs=None, f=None, dimensions=None, mutation_factor=0.5,
                 crossover_prob=0.5, strategy="rand1_bin", min_fidelity=None,
                 max_fidelity=None, eta=3, min_clip=None, max_clip=None, seed=None,
                 configspace=True, boundary_fix_type="random", max_age=np.inf, n_workers=None,
                 client=None, async_strategy="immediate", save_freq="incumbent", resume=False,
                 **kwargs):
        super().__init__(cs=cs, f=f, dimensions=dimensions, mutation_factor=mutation_factor,
                         crossover_prob=crossover_prob, strategy=strategy, min_fidelity=min_fidelity,
                         max_fidelity=max_fidelity, eta=eta, min_clip=min_clip, max_clip=max_clip, 
                         seed=seed, configspace=configspace, boundary_fix_type=boundary_fix_type,
                         max_age=max_age, resume=resume, **kwargs)
        self.de_params.update({"async_strategy": async_strategy})
        self.iteration_counter = -1
        self.de = {}
        self._max_pop_size = None
        self.active_brackets = []  # list of SHBracketManager objects
        self.traj = []
        self.runtime = []
        self.history = []
        self._ask_counter = 0
        self._tell_counter = 0
        self.start = None
        if save_freq not in ["incumbent", "step", "end"] and save_freq is not None:
            self.logger.warning(f"Save frequency {save_freq} unknown. Resorting to using 'end'.")
            save_freq = "end"
        self.save_freq = "end" if save_freq is None else save_freq

        # Dask variables
        if n_workers is None and client is None:
            raise ValueError("Need to specify either 'n_workers'(>0) or 'client' (a Dask client)!")
        if client is not None and isinstance(client, Client):
            self.client = client
            self.n_workers = len(client.ncores())
        else:
            self.n_workers = n_workers
            if self.n_workers > 1:
                self.client = Client(
                    n_workers=self.n_workers, processes=True, threads_per_worker=1, scheduler_port=0
                )  # port 0 makes Dask select a random free port
            else:
                self.client = None
        self.futures = []
        self.shared_data = None

        # Initializing DE subpopulations
        self._get_pop_sizes()
        self._init_subpop()
        self.config_repository.initial_configs = self.config_repository.configs.copy()

        # Misc.
        self.available_gpus = None
        self.gpu_usage = None
        self.single_node_with_gpus = None

        self._time_budget_exhausted = False
        self._runtime_budget_timer = None

        # Setup logging and potentially reload state
        if resume:
            self.logger.info("Loading checkpoint...")
            success = self._load_checkpoint(self.output_path)
            if not success:
                self.logger.error("Checkpoint could not be loaded. " \
                                  "Please refer to the prior warning in order to " \
                                  "identifiy the problem.")
                raise AttributeError("Checkpoint could not be loaded. Check the logs" \
                                     "for more information")
        elif (self.output_path / "dehb_state.json").exists():
            self.logger.warning("A checkpoint already exists, " \
                                "results could potentially be overwritten.")

    def __getstate__(self):
        """Allows the object to picklable while having Dask client as a class attribute."""
        d = dict(self.__dict__)
        d["client"] = None  # hack to allow Dask client to be a class attribute
        d["logger"] = None  # hack to allow logger object to be a class attribute
        d["_runtime_budget_timer"] = None # hack to allow timer object to be a class attribute
        return d

    def __del__(self):
        """Ensures a clean kill of the Dask client and frees up a port."""
        if hasattr(self, "client") and isinstance(self, Client):
            self.client.close()

    def _f_objective(self, job_info):
        """Wrapper to call DE's objective function."""
        # check if job_info appended during job submission self.submit_job() includes "gpu_devices"
        if "gpu_devices" in job_info and self.single_node_with_gpus:
            # should set the environment variable for the spawned worker process
            # reprioritising a CUDA device order specific to this worker process
            os.environ.update({"CUDA_VISIBLE_DEVICES": job_info["gpu_devices"]})

        config, config_id = job_info["config"], job_info["config_id"]
        fidelity, parent_id = job_info["fidelity"], job_info["parent_id"]
        bracket_id = job_info["bracket_id"]
        kwargs = job_info["kwargs"]
        res = self.de[fidelity].f_objective(config, fidelity, **kwargs)
        info = res["info"] if "info" in res else {}
        run_info = {
            "job_info": {
                "config": config,
                "config_id": config_id,
                "fidelity": fidelity,
                "parent_id": parent_id,
                "bracket_id": bracket_id,
            },
            "result": {
                "fitness": res["fitness"],
                "cost": res["cost"],
                "info": info,
            },
        }

        if "gpu_devices" in job_info:
            # important for GPU usage tracking if single_node_with_gpus=True
            device_id = int(job_info["gpu_devices"].strip().split(",")[0])
            run_info.update({"device_id": device_id})
        return run_info

    def _create_cuda_visible_devices(self, available_gpus: List[int], start_id: int) -> str:
        """Generates a string to set the CUDA_VISIBLE_DEVICES environment variable.

        Given a list of available GPU device IDs and a preferred ID (start_id), the environment
        variable is created by putting the start_id device first, followed by the remaining devices
        arranged randomly. The worker that uses this string to set the environment variable uses
        the start_id GPU device primarily now.
        """
        assert start_id in available_gpus
        available_gpus = deepcopy(available_gpus)
        available_gpus.remove(start_id)
        self.rng.shuffle(available_gpus)
        final_variable = [str(start_id)] + [str(_id) for _id in available_gpus]
        final_variable = ",".join(final_variable)
        return final_variable

    def _distribute_gpus(self):
        """Function to create a GPU usage tracker dict.

        The idea is to extract the exact GPU device IDs available. During job submission, each
        submitted job is given a preference of a GPU device ID based on the GPU device with the
        least number of active running jobs. On retrieval of the result, this gpu usage dict is
        updated for the device ID that the finished job was mapped to.
        """
        try:
            available_gpus = os.environ["CUDA_VISIBLE_DEVICES"]
            available_gpus = available_gpus.strip().split(",")
            self.available_gpus = [int(_id) for _id in available_gpus]
        except KeyError as e:
            print("Unable to find valid GPU devices. "
                  f"Environment variable {str(e)} not visible!")
            self.available_gpus = []
        self.gpu_usage = dict()
        for _id in self.available_gpus:
            self.gpu_usage[_id] = 0

    def _timeout_handler(self) -> None:
        self.logger.warning("Runtime budget exhausted. Saving optimization checkpoint now.")
        self.save()
        # Important to set this flag to true after saving
        self._time_budget_exhausted = True

    def vector_to_configspace(self, config: np.array) -> ConfigSpace.Configuration:
        """Converts numpy representation to `Configuration`.

        Args:
            config (np.array): Configuration to convert.

        Returns:
            ConfigSpace.Configuration: Converted configuration
        """
        assert hasattr(self, "de")
        assert len(self.fidelities) > 0
        return self.de[self.fidelities[0]].vector_to_configspace(config)

    def configspace_to_vector(self, config: ConfigSpace.Configuration) -> np.array:
        """Converts `Configuration` to numpy array.

        Args:
            config (ConfigSpace.Configuration): Configuration to convert

        Returns:
            np.array: Converted configuration
        """
        assert hasattr(self, "de")
        assert len(self.fidelities) > 0
        return self.de[self.fidelities[0]].configspace_to_vector(config)

    def reset(self, *, reset_seeds: bool = True):
        super().reset(reset_seeds=reset_seeds)
        if self.n_workers > 1 and hasattr(self, "client") and isinstance(self.client, Client):
            self.client.restart()
        else:
            self.client = None
        self.futures = []
        self.shared_data = None
        self.iteration_counter = -1
        self.de = {}
        self._max_pop_size = None
        self.start = None
        self.active_brackets = []
        self.traj = []
        self.runtime = []
        self.history = []
        self._ask_counter = 0
        self._tell_counter = 0
        self.config_repository.reset()
        self._get_pop_sizes()
        self._init_subpop()
        self.available_gpus = None
        self.gpu_usage = None
        self._time_budget_exhausted = False
        self._runtime_budget_timer = None

    def _init_population(self, pop_size):
        if self.use_configspace:
            population = self.cs.sample_configuration(size=pop_size)
            population = [self.configspace_to_vector(individual) for individual in population]
        else:
            population = self.rng.uniform(low=0.0, high=1.0, size=(pop_size, self.dimensions))
        return population

    def _clean_inactive_brackets(self):
        """Removes brackets from the active list if it is done as communicated by Bracket Manager."""
        if len(self.active_brackets) == 0:
            return
        self.active_brackets = [
            bracket for bracket in self.active_brackets if ~bracket.is_bracket_done()
        ]
        return

    def _update_trackers(self, traj, runtime, history):
        self.traj.append(traj)
        self.runtime.append(runtime)
        self.history.append(history)

    def _update_incumbents(self, config, score, info):
        self.inc_config = config
        self.inc_score = score
        self.inc_info = info

    def _get_pop_sizes(self):
        """Determines maximum pop size for each fidelity."""
        self._max_pop_size = {}
        for i in range(self.max_SH_iter):
            n, r = self._get_next_iteration(i)
            for j, r_j in enumerate(r):
                self._max_pop_size[r_j] = max(
                    n[j], self._max_pop_size[r_j]
                ) if r_j in self._max_pop_size.keys() else n[j]

    def _init_subpop(self):
        """List of DE objects corresponding to the fidelities."""
        self.de = {}
        seeds = self.rng.integers(0, 2**32 - 1, size=len(self._max_pop_size))
        for (i, f), _seed in zip(enumerate(self._max_pop_size.keys()), seeds):
            self.de[f] = AsyncDE(**self.de_params, pop_size=self._max_pop_size[f],
                                 config_repository=self.config_repository, seed=int(_seed))
            self.de[f].population = self.de[f].init_population(pop_size=self._max_pop_size[f])
            self.de[f].population_ids = self.config_repository.announce_population(self.de[f].population, f)
            self.de[f].fitness = np.array([np.inf] * self._max_pop_size[f])
            # adding attributes to DEHB objects to allow communication across subpopulations
            self.de[f].parent_counter = 0
            self.de[f].promotion_pop = None
            self.de[f].promotion_pop_ids = None
            self.de[f].promotion_fitness = None

    def _concat_pops(self, exclude_fidelity=None):
        """Concatenates all subpopulations."""
        fidelities = list(self.fidelities)
        if exclude_fidelity is not None:
            fidelities.remove(exclude_fidelity)
        pop = []
        for fidelity in fidelities:
            pop.extend(self.de[fidelity].population.tolist())
        return np.array(pop)

    def _start_new_bracket(self):
        """Starts a new bracket based on Hyperband."""
        # start new bracket
        self.iteration_counter += 1  # iteration counter gives the bracket count or bracket ID
        n_configs, fidelities = self._get_next_iteration(self.iteration_counter)
        bracket = SHBracketManager(
            n_configs=n_configs, fidelities=fidelities, bracket_id=self.iteration_counter
        )
        self.active_brackets.append(bracket)
        return bracket

    def _get_worker_count(self):
        if isinstance(self.client, Client):
            return len(self.client.ncores())
        else:
            return 1

    def _is_worker_available(self):
        """Checks if at least one worker is available to run a job."""
        if self.n_workers == 1 or self.client is None or not isinstance(self.client, Client):
            # in the synchronous case, one worker is always available
            return True
        # checks the absolute number of workers mapped to the client scheduler
        # client.ncores() should return a dict with the keys as unique addresses to these workers
        # treating the number of available workers in this manner
        workers = self._get_worker_count()  # len(self.client.ncores())
        if len(self.futures) >= workers:
            # pause/wait if active worker count greater allocated workers
            return False
        return True

    def _get_promotion_candidate(self, low_fidelity, high_fidelity, n_configs):
        """Manages the population to be promoted from the lower to the higher fidelity.

        This is triggered or in action only during the first full HB bracket, which is equivalent
        to the number of brackets <= max_SH_iter.
        """
        # finding the individuals that have been evaluated (fitness < np.inf)
        evaluated_configs = np.where(self.de[low_fidelity].fitness != np.inf)[0]
        promotion_candidate_pop = self.de[low_fidelity].population[evaluated_configs]
        promotion_candidate_pop_ids = self.de[low_fidelity].population_ids[evaluated_configs]
        promotion_candidate_fitness = self.de[low_fidelity].fitness[evaluated_configs]
        # ordering the evaluated individuals based on their fitness values
        pop_idx = np.argsort(promotion_candidate_fitness)

        # creating population for promotion if none promoted yet or nothing to promote
        if self.de[high_fidelity].promotion_pop is None or \
                len(self.de[high_fidelity].promotion_pop) == 0:
            self.de[high_fidelity].promotion_pop = np.empty((0, self.dimensions))
            self.de[high_fidelity].promotion_pop_ids = np.array([], dtype=np.int64)
            self.de[high_fidelity].promotion_fitness = np.array([])

            # iterating over the evaluated individuals from the lower fidelity and including them
            # in the promotion population for the higher fidelity only if it's not in the population
            # this is done to ensure diversity of population and avoid redundant evaluations
            for idx in pop_idx:
                individual = promotion_candidate_pop[idx]
                individual_id = promotion_candidate_pop_ids[idx]
                # checks if the candidate individual already exists in the high fidelity population
                if np.any(np.all(individual == self.de[high_fidelity].population, axis=1)):
                    # skipping already present individual to allow diversity and reduce redundancy
                    continue
                self.de[high_fidelity].promotion_pop = np.append(
                    self.de[high_fidelity].promotion_pop, [individual], axis=0
                )
                self.de[high_fidelity].promotion_pop_ids = np.append(
                    self.de[high_fidelity].promotion_pop_ids, individual_id
                )
                self.de[high_fidelity].promotion_fitness = np.append(
                    self.de[high_fidelity].promotion_pop, promotion_candidate_fitness[pop_idx]
                )
            # retaining only n_configs
            self.de[high_fidelity].promotion_pop = self.de[high_fidelity].promotion_pop[:n_configs]
            self.de[high_fidelity].promotion_pop_ids = self.de[high_fidelity].promotion_pop_ids[:n_configs]
            self.de[high_fidelity].promotion_fitness = \
                self.de[high_fidelity].promotion_fitness[:n_configs]

        if len(self.de[high_fidelity].promotion_pop) > 0:
            config = self.de[high_fidelity].promotion_pop[0]
            config_id = self.de[high_fidelity].promotion_pop_ids[0]
            # removing selected configuration from population
            self.de[high_fidelity].promotion_pop = self.de[high_fidelity].promotion_pop[1:]
            self.de[high_fidelity].promotion_pop_ids = self.de[high_fidelity].promotion_pop_ids[1:]
            self.de[high_fidelity].promotion_fitness = self.de[high_fidelity].promotion_fitness[1:]
        else:
            # in case of an edge failure case where all high fidelity individuals are same
            # just choose the best performing individual from the lower fidelity (again)
            config = self.de[low_fidelity].population[pop_idx[0]]
            config_id = self.de[low_fidelity].population_ids[pop_idx[0]]
        return config, config_id

    def _get_next_parent_for_subpop(self, fidelity):
        """Maintains a looping counter over a subpopulation, to iteratively select a parent."""
        parent_id = self.de[fidelity].parent_counter
        self.de[fidelity].parent_counter += 1
        self.de[fidelity].parent_counter = self.de[fidelity].parent_counter % self._max_pop_size[fidelity]
        return parent_id

    def _acquire_config(self, bracket, fidelity):
        """Generates/chooses a configuration based on the fidelity and iteration number."""
        # select a parent/target
        parent_id = self._get_next_parent_for_subpop(fidelity)
        target = self.de[fidelity].population[parent_id]
        # identify lower fidelity to transfer information from
        lower_fidelity, num_configs = bracket.get_lower_fidelity_promotions(fidelity)

        if self.iteration_counter < self.max_SH_iter:
            # promotions occur only in the first set of SH brackets under Hyperband
            # for the first rung/fidelity in the current bracket, no promotion is possible and
            # evolution can begin straight away
            # for the subsequent rungs, individuals will be promoted from the lower_fidelity
            if fidelity != bracket.fidelities[0]:
                # TODO: check if generalizes to all fidelity spacings
                config, config_id = self._get_promotion_candidate(lower_fidelity, fidelity, num_configs)
                return config, config_id, parent_id

        # DE evolution occurs when either all individuals in the subpopulation have been evaluated
        # at least once, i.e., has fitness < np.inf, which can happen if
        # iteration_counter <= max_SH_iter but certainly never when iteration_counter > max_SH_iter

        # a single DE evolution --- (mutation + crossover) occurs here
        mutation_pop_idx = np.argsort(self.de[lower_fidelity].fitness)[:num_configs]
        mutation_pop = self.de[lower_fidelity].population[mutation_pop_idx]
        # generate mutants from previous fidelity subpopulation or global population
        if len(mutation_pop) < self.de[fidelity]._min_pop_size:
            filler = self.de[fidelity]._min_pop_size - len(mutation_pop) + 1
            new_pop = self.de[fidelity]._init_mutant_population(
                pop_size=filler, population=self._concat_pops(),
                target=target, best=self.inc_config
            )
            mutation_pop = np.concatenate((mutation_pop, new_pop))
        # generate mutant from among individuals in mutation_pop
        mutant = self.de[fidelity].mutation(
            current=target, best=self.inc_config, alt_pop=mutation_pop
        )
        # perform crossover with selected parent
        config = self.de[fidelity].crossover(target=target, mutant=mutant)
        config = self.de[fidelity].boundary_check(config)

        # announce new config
        config_id = self.config_repository.announce_config(config, fidelity)
        return config, config_id, parent_id

    def _get_next_bracket(self, only_id=False):
        """Used to retrieve what bracket the bracket for the next job.

        Optionally, a new bracket is started, if there are no more pending jobs or
        when all active brackets are waiting.

        Args:
            only_id (bool): Only returns the id of the next bracket

        Returns:
            SHBracketmanager or int: bracket or bracket ID of next job
        """
        bracket = None
        start_new_bracket = False
        if len(self.active_brackets) == 0 or \
                np.all([bracket.is_bracket_done() for bracket in self.active_brackets]):
            # start new bracket when no pending jobs from existing brackets or empty bracket list
            start_new_bracket = True
        else:
            for _bracket in self.active_brackets:
                # check if _bracket is not waiting for previous rung results of same bracket
                # _bracket is not waiting on the last rung results
                # these 2 checks allow DEHB to have a "synchronous" Successive Halving
                if not _bracket.previous_rung_waits() and _bracket.is_pending():
                    # bracket eligible for job scheduling
                    bracket = _bracket
                    break
            if bracket is None:
                # start new bracket when existing list has all waiting brackets
                start_new_bracket = True

        if only_id:
            return self.iteration_counter + 1 if start_new_bracket else bracket.bracket_id

        return self._start_new_bracket() if start_new_bracket else bracket

    def _get_next_job(self):
        """Loads a configuration and fidelity to be evaluated next.

        Returns:
            dict: Dicitonary containing all necessary information of the next job.
        """
        bracket = self._get_next_bracket()
        # fidelity that the SH bracket allots
        fidelity = bracket.get_next_job_fidelity()
        config, config_id, parent_id = self._acquire_config(bracket, fidelity)

        # transform config to proper representation
        if self.use_configspace:
            # converts [0, 1] vector to a ConfigSpace object
            config = self.de[fidelity].vector_to_configspace(config)

        # notifies the Bracket Manager that a single config is to run for the fidelity chosen
        job_info = {
            "config": config,
            "config_id": config_id,
            "fidelity": fidelity,
            "parent_id": parent_id,
            "bracket_id": bracket.bracket_id,
        }

        # pass information of job submission to Bracket Manager
        for bracket in self.active_brackets:
            if bracket.bracket_id == job_info["bracket_id"]:
                # registering is IMPORTANT for Bracket Manager to perform SH
                bracket.register_job(job_info["fidelity"])
                break
        return job_info

    def ask(self, n_configs: int=1) -> Union[dict, List[dict]]:
        """Get the next configuration to run from the optimizer.

        The retrieved configuration can then be evaluated by the user.
        After evaluation use `tell` to report the results back to the optimizer.
        For more information, please refer to the description of `tell`.

        Args:
            n_configs (int, optional): Number of configs to ask for. Defaults to 1.

        Returns:
            dict or list of dict: Job info(s) of next configuration to evaluate.
        """
        jobs = []
        if n_configs == 1:
            jobs = self._get_next_job()
            self._ask_counter += 1
        else:
            for _ in range(n_configs):
                jobs.append(self._get_next_job())
                self._ask_counter += 1

        return jobs

    def _get_gpu_id_with_low_load(self):
        candidates = []
        for k, v in self.gpu_usage.items():
            if v == min(self.gpu_usage.values()):
                candidates.append(k)
        device_id = self.rng.choice(candidates)
        # creating string for setting environment variable CUDA_VISIBLE_DEVICES
        gpu_ids = self._create_cuda_visible_devices(
            self.available_gpus, device_id,
        )
        # updating GPU usage
        self.gpu_usage[device_id] += 1
        self.logger.debug(f"GPU device selected: {device_id}")
        self.logger.debug(f"GPU device usage: {self.gpu_usage}")
        return gpu_ids

    def _submit_job(self, job_info, **kwargs):
        """Asks a free worker to run the objective function on config and fidelity."""
        job_info["kwargs"] = self.shared_data if self.shared_data is not None else kwargs
        # submit to Dask client
        if self.n_workers > 1 or isinstance(self.client, Client):
            if self.single_node_with_gpus:
                # managing GPU allocation for the job to be submitted
                job_info.update({"gpu_devices": self._get_gpu_id_with_low_load()})
            self.futures.append(
                self.client.submit(self._f_objective, job_info)
            )
        else:
            # skipping scheduling to Dask worker to avoid added overheads in the synchronous case
            self.futures.append(self._f_objective(job_info))

    def _fetch_results_from_workers(self):
        """Iterate over futures and collect results from finished workers."""
        if self.n_workers > 1 or isinstance(self.client, Client):
            done_list = [(i, future) for i, future in enumerate(self.futures) if future.done()]
        else:
            # Dask not invoked in the synchronous case
            done_list = [(i, future) for i, future in enumerate(self.futures)]
        if len(done_list) > 0:
            self.logger.debug(
                f"Collecting {len(done_list)} of the {len(self.futures)} job(s) active.",
            )
        for _, future in done_list:
            if self.n_workers > 1 or isinstance(self.client, Client):
                run_info = future.result()
                if "device_id" in run_info:
                    # updating GPU usage
                    self.gpu_usage[run_info["device_id"]] -= 1
                    self.logger.debug("GPU device released: {}".format(run_info["device_id"]))
                future.release()
            else:
                # Dask not invoked in the synchronous case
                run_info = future
            # tell result
            self.tell(run_info["job_info"], run_info["result"])
        # remove processed future
        self.futures = np.delete(self.futures, [i for i, _ in done_list]).tolist()

    def _adjust_budgets(self, fevals=None, brackets=None):
        # only update budgets if it is not the first run
        if fevals is not None and len(self.traj) > 0:
            fevals = len(self.traj) + fevals
        elif brackets is not None and self.iteration_counter > -1:
            brackets = self.iteration_counter + brackets + 1

        return fevals, brackets

    def _get_state(self):
        state = {}
        # DE parameters
        serializable_de_params = self.de_params.copy()
        serializable_de_params.pop("cs", None)
        serializable_de_params.pop("rng", None)
        serializable_de_params.pop("f", None)
        serializable_de_params["output_path"] = str(serializable_de_params["output_path"])
        state["DE_params"] = serializable_de_params
        # Hyperband variables
        hb_dict = {}
        hb_dict["min_fidelity"] = self.min_fidelity
        hb_dict["max_fidelity"] = self.max_fidelity
        hb_dict["min_clip"] = self.min_clip
        hb_dict["max_clip"] = self.max_clip
        hb_dict["eta"] = self.eta
        state["HB_params"] = hb_dict
        # Save DEHB interals
        dehb_internals = {}
        dehb_internals["initial_configs"] = self.config_repository.get_serialized_initial_configs()
        state["internals"] = dehb_internals
        return state

    def _save_state(self):
        # Get state
        state = self._get_state()
        # Write state to disk
        try:
            state_path = self.output_path / "dehb_state.json"
            with state_path.open("w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.logger.warning(f"State not saved: {e!r}")


    def _is_run_budget_exhausted(self, fevals=None, brackets=None):
        """Checks if the DEHB run should be terminated or continued."""
        if fevals is not None:
            if len(self.traj) >= fevals:
                return True
        elif brackets is not None:
            future_iteration_counter = self._get_next_bracket(only_id=True)
            if future_iteration_counter >= brackets:
                for bracket in self.active_brackets:
                    # waits for all brackets < iteration_counter to finish by collecting results
                    if bracket.bracket_id < future_iteration_counter and \
                            not bracket.is_bracket_done():
                        return False
                return True
        else:
            return self._time_budget_exhausted
        return False

    def _save_incumbent(self):
        # Return early if there is no incumbent yet
        if self.inc_config is None:
            return
        try:
            res = {}
            if self.use_configspace:
                config = self.vector_to_configspace(self.inc_config)
                res["config"] = config.get_dictionary()
            else:
                res["config"] = self.inc_config.tolist()
            res["score"] = self.inc_score
            res["info"] = self.inc_info
            incumbent_path = self.output_path / "incumbent.json"
            with incumbent_path.open("w") as f:
                json.dump(res, f)
        except Exception as e:
            self.logger.warning(f"Incumbent not saved: {e!r}")

    def _save_history(self, name="history.parquet.gzip"):
        # Return early if there is no history yet
        if self.history is None:
            return
        try:
            history_path = self.output_path / name
            history_df = pd.DataFrame(self.history, columns=["config_id", "config", "fitness",
                                                             "cost", "fidelity", "info"])
            # Check if the 'info' column is empty or contains only None values
            if history_df["info"].apply(lambda x: (isinstance(x, dict) and len(x) == 0)).all():
                # Drop the 'info' column
                history_df = history_df.drop(columns=["info"])
            history_df.to_parquet(history_path, compression="gzip")
        except Exception as e:
            self.logger.warning(f"History not saved: {e!r}")

    def _log_debug(self):
        for bracket in self.active_brackets:
            self.logger.debug(f"Bracket ID {bracket.bracket_id}:\n{bracket!s}")

    def _log_runtime(self, fevals, brackets, total_cost):
        if fevals is not None:
            remaining = (len(self.traj), fevals, "function evaluation(s) done")
        elif brackets is not None:
            _suffix = f"bracket(s) started; # active brackets: {len(self.active_brackets)}"
            remaining = (self.iteration_counter + 1, brackets, _suffix)
        else:
            elapsed = np.format_float_positional(time.time() - self.start, precision=2)
            remaining = (elapsed, total_cost, "seconds elapsed")
        self.logger.info(
            f"{remaining[0]}/{remaining[1]} {remaining[2]}",
        )

    def _log_job_submission(self, job_info: dict):
        fidelity = job_info["fidelity"]
        config_id = job_info["config_id"]
        self.logger.info(
            "Evaluating configuration {} with fidelity {} under "
            "bracket ID {}".format(config_id, fidelity, job_info["bracket_id"]),
        )
        self.logger.info(
            f"Best score seen/Incumbent score: {self.inc_score}",
        )

    def _load_checkpoint(self, run_dir: str):
        # Check if path exists, otherwise give warning
        run_dir = Path(run_dir)
        if not Path.exists(run_dir):
            self.logger.warning("Path to run directory does not exist.")
            return False
        # Load dehb state
        dehb_state_path = run_dir / "dehb_state.json"
        with dehb_state_path.open() as f:
            dehb_state = json.load(f)
        # Convert output_path of checkpoint to Path
        dehb_state["DE_params"]["output_path"] = Path(dehb_state["DE_params"]["output_path"])
        if not all(dehb_state["DE_params"][key] == self.de_params[key]
                   for key in dehb_state["DE_params"]):
            self.logger.warning("Initialized DE parameters do not match saved parameters.")
            return False
        self.de_params.update(dehb_state["DE_params"])

        hb_vars = dehb_state["HB_params"]
        if self.min_fidelity != hb_vars["min_fidelity"]:
            self.logger.warning("Initialized min_fidelity does not match saved parameters.")
            return False
        self.min_fidelity = hb_vars["min_fidelity"]

        if self.max_fidelity != hb_vars["max_fidelity"]:
            self.logger.warning("Initialized max_fidelity does not match saved parameters.")
            return False
        self.max_fidelity = hb_vars["max_fidelity"]

        if self.min_clip != hb_vars["min_clip"]:
            self.logger.warning("Initialized min_clip does not match saved parameters.")
            return False
        self.min_clip = hb_vars["min_clip"]

        if self.max_clip != hb_vars["max_clip"]:
            self.logger.warning("Initialized max_clip does not match saved parameters.")
            return False
        self.max_clip = hb_vars["max_clip"]

        if self.eta != hb_vars["eta"]:
            self.logger.warning("Initialized eta does not match saved parameters.")
            return False
        self.eta = hb_vars["eta"]

        # Load history
        history_path = run_dir / "history.parquet.gzip"
        history = pd.read_parquet(history_path)

        # Replay history
        for _, row in history.iterrows():
            job_info = {
                "fidelity": row["fidelity"],
                "config_id": row["config_id"],
                "config": np.array(row["config"]),
            }
            result = {
                "fitness": row["fitness"],
                "cost": row["cost"],
                "info": row.get("info", {}),
            }

            self.tell(job_info, result, replay=True)
        # Clean inactive brackets
        self._clean_inactive_brackets()
        return True

    def save(self):
        """Saves the current incumbent, history and state to disk."""
        self.logger.info("Saving state to disk...")
        if self._time_budget_exhausted:
            self.logger.info("Runtime budget exhausted. Resorting to only saving overtime history.")
            self._save_history(name="overtime_history.parquet.gzip")
        else:
            self._save_incumbent()
            self._save_history()
            self._save_state()

    def tell(self, job_info: dict, result: dict, replay: bool=False) -> None:
        """Feed a result back to the optimizer.

        In order to correctly interpret the results, the `job_info` dict, retrieved by `ask`,
        has to be given. Moreover, the `result` dict has to contain the keys `fitness` and `cost`.
        `fitness` resembles the objective you are trying to optimize, e.g. validation loss.
        `cost` resembles the computational cost for computing the result, e.g. the wallclock time
        for training and validating a neural network to achieve the validation loss specified in
        `fitness`. It is also possible to add the field `info` to the `result` in order to store
        additional, user-specific information.

        !!! note "User-specific information `info`"

            Please note, that we only support types, that are serializable by `pandas`. If
            non-serializable types are used, DEHB will not be able to save the history.
            If you want to be on the safe side, please use built-in python types.

        Args:
            job_info (dict): Job info returned by ask().
            result (dict): Result dictionary with mandatory keys `fitness` and `cost`.
        """
        if replay:
            # Get job_info container from ask and update fields
            job_info_container = self.ask()
            # Update according to given history
            job_info_container["fidelity"] = job_info["fidelity"]
            job_info_container["config"] = job_info["config"]
            job_info_container["config_id"] = job_info["config_id"]

            # Update entry in ConfigRepository
            self.config_repository.configs[job_info["config_id"]].config = job_info["config"]
            # Replace job_info with container to make sure all fields are given
            job_info = job_info_container

        if self._tell_counter >= self._ask_counter:
            raise NotImplementedError("Called tell() more often than ask(). \
                                      Warmstarting with tell is not supported. ")
        self._tell_counter += 1
        # Update bracket information
        fitness, cost = float(result["fitness"]), float(result["cost"])
        info = result["info"] if "info" in result else {}
        fidelity, parent_id = job_info["fidelity"], job_info["parent_id"]
        config, config_id = job_info["config"], job_info["config_id"]
        bracket_id = job_info["bracket_id"]
        for bracket in self.active_brackets:
            if bracket.bracket_id == bracket_id:
                # bracket job complete
                bracket.complete_job(fidelity)  # IMPORTANT to perform synchronous SH

        self.config_repository.tell_result(config_id, fidelity, fitness, cost, info)

        # get hypercube representation from config repo
        if self.use_configspace:
            config = self.config_repository.get(config_id)

        # carry out DE selection
        if fitness <= self.de[fidelity].fitness[parent_id]:
            self.de[fidelity].population[parent_id] = config
            self.de[fidelity].population_ids[parent_id] = config_id
            self.de[fidelity].fitness[parent_id] = fitness
        # updating incumbents
        inc_changed = False
        if self.de[fidelity].fitness[parent_id] < self.inc_score:
            self._update_incumbents(
                config=self.de[fidelity].population[parent_id],
                score=self.de[fidelity].fitness[parent_id],
                info=info,
            )
            inc_changed = True
        # book-keeping
        self._update_trackers(
            traj=self.inc_score, runtime=cost, history=(
                config_id, config.tolist(), float(fitness), float(cost), float(fidelity), info,
            ),
        )

        if self.save_freq == "step" or (self.save_freq == "incumbent" and inc_changed) and not replay:
            self.save()

    @logger.catch
    def run(self, fevals=None, brackets=None, total_cost=None, single_node_with_gpus=False,
            **kwargs) -> Tuple[np.array, np.array, np.array]:
        """Main interface to run optimization by DEHB.

        This function waits on workers and if a worker is free, asks for a configuration and a
        fidelity to evaluate on and submits it to the worker. In each loop, it checks if a job
        is complete, fetches the results, carries the necessary processing of it asynchronously
        to the worker computations.

        The duration of the DEHB run can be controlled by specifying one of 3 parameters. If more
        than one are specified, DEHB selects only one in the priority order (high to low): <br>
        1) Number of function evaluations (fevals) <br>
        2) Number of Successive Halving brackets run under Hyperband (brackets) <br>
        3) Total computational cost (in seconds) aggregated by all function evaluations (total_cost)

        !!! note "Using `tell` under the hood."

            Please note, that `run` uses `tell` under the hood, therefore please have a
            look at the documentation of `tell` for more information e.g. about the result format.

        !!! note "Adjusting verbosity"

            The verbosity of DEHB logs can be adjusted via adding the `log_level` parameter to DEHBs
            initialization. As we use loguru, the logging levels can be found on [their website](https://loguru.readthedocs.io/en/stable/api/logger.html#levels).

        Args:
            fevals (int, optional): Number of functions evaluations to run. Defaults to None.
            brackets (int, optional): Number of brackets to run. Defaults to None.
            total_cost (int, optional): Wallclock budget in seconds. Defaults to None.
            single_node_with_gpus (bool): Workers get assigned different GPUs. Default to False.

        Returns:
            Trajectory, runtime and optimization history.
        """
        # Warn if users use old state saving frequencies
        if "save_history" in kwargs or "save_intermediate" in kwargs or "name" in kwargs:
            logger.warning("The run parameters 'save_history', 'save_intermediate' and 'name' are "\
                           "deprecated, since the changes in v0.1.1. Please use the 'saving_freq' "\
                           "parameter in the constructor to adjust when to save DEHBs state " \
                           "(including history). Please use the 'output_path' parameter to adjust "\
                           "where the state and logs should be saved.")
            raise TypeError("Used deprecated parameters 'save_history', 'save_intermediate' " \
                            "and/or 'name'. Please check the logs for more information.")
        if "verbose" in kwargs:
            logger.warning("The run parameters 'verbose' is deprecated since the changes in v0.1.2. "\
                           "Please use the 'log_level' parameter when initializing DEHB.")
            raise TypeError("Used deprecated parameter 'verbose'. "\
                            "Please check the logs for more information.")
        # check if run has already been called before
        if self.start is not None:
            logger.warning("DEHB has already been run. Calling 'run' twice could lead to unintended"
                           + " behavior. Please restart DEHB with an increased compute budget"
                           + " instead of calling 'run' twice.")
            self._time_budget_exhausted = False

        # checks if a Dask client exists
        if len(kwargs) > 0 and self.n_workers > 1 and isinstance(self.client, Client):
            # broadcasts all additional data passed as **kwargs to all client workers
            # this reduces overload in the client-worker communication by not having to
            # serialize the redundant data used by all workers for every job
            self.shared_data = self.client.scatter(kwargs, broadcast=True)

        # allows each worker to be mapped to a different GPU when running on a single node
        # where all available GPUs are accessible
        self.single_node_with_gpus = single_node_with_gpus
        if self.single_node_with_gpus:
            self._distribute_gpus()

        self.start = self.start = time.time()
        self.logger.info("\nLogging at {} for optimization starting at {}\n".format(
            Path.cwd() / self.log_filename,
            time.strftime("%x %X %Z", time.localtime(self.start)),
        ))

        delimiters = [fevals, brackets, total_cost]
        delim_sum = sum(x is not None for x in delimiters)
        if delim_sum == 0:
            raise ValueError(
                "Need one of 'fevals', 'brackets' or 'total_cost' as budget for DEHB to run."
            )
        fevals, brackets = self._adjust_budgets(fevals, brackets)
        # Set alarm for specified runtime budget
        if total_cost is not None:
            self._runtime_budget_timer = Timer(total_cost, self._timeout_handler)
            self._runtime_budget_timer.start()
        while True:
            if self._is_run_budget_exhausted(fevals, brackets):
                break
            if self._is_worker_available():
                next_bracket_id = self._get_next_bracket(only_id=True)
                if brackets is not None and next_bracket_id >= brackets:
                    # ignore submission and only collect results
                    # when brackets are chosen as run budget, an extra bracket is created
                    # since iteration_counter is incremented in ask() and then checked
                    # in _is_run_budget_exhausted(), therefore, need to skip suggestions
                    # coming from the extra allocated bracket
                    # _is_run_budget_exhausted() will not return True until all the lower brackets
                    # have finished computation and returned its results
                    pass
                else:
                    if self.n_workers > 1 or isinstance(self.client, Client):
                        self.logger.debug("{}/{} worker(s) available.".format(
                            self._get_worker_count() - len(self.futures), self._get_worker_count(),
                        ))
                    # Ask for new job_info
                    job_info = self.ask()
                    # Submit job_info to a worker for execution
                    self._submit_job(job_info, **kwargs)
                    self._log_runtime(fevals, brackets, total_cost)
                    self._log_job_submission(job_info)
                    self._log_debug()
            self._fetch_results_from_workers()
            self._clean_inactive_brackets()
        # end of while
        time_taken = time.time() - self.start
        self.logger.info("End of optimisation! Total duration: {}; Total fevals: {}\n".format(
            time_taken, len(self.traj),
        ))
        self.logger.info(f"Incumbent score: {self.inc_score}")
        self.logger.info("Incumbent config: ")
        if self.use_configspace:
            config = self.vector_to_configspace(self.inc_config)
            for k, v in config.get_dictionary().items():
                self.logger.info(f"{k}: {v}")
        else:
            self.logger.info(f"{self.inc_config}")

        self.save()
        # cancel timer
        if self._runtime_budget_timer:
            self._runtime_budget_timer.cancel()
        # reset waiting jobs of active bracket to allow for continuation
        self.active_brackets = []
        if len(self.active_brackets) > 0:
            for active_bracket in self.active_brackets:
                active_bracket.reset_waiting_jobs()
        return np.array(self.traj), np.array(self.runtime), np.array(self.history, dtype=object)
