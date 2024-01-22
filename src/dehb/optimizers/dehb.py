import os
import sys
import json
import time
import pickle
import numpy as np
import ConfigSpace
from typing import List
from copy import deepcopy
from loguru import logger
from distributed import Client

from .de import DE, AsyncDE
from ..utils import SHBracketManager
from ..utils import ConfigRepository


logger.configure(handlers=[{"sink": sys.stdout, "level": "INFO"}])
_logger_props = {
    "format": "{time} {level} {message}",
    "enqueue": True,
    "rotation": "500 MB"
}


class DEHBBase:
    def __init__(self, cs=None, f=None, dimensions=None, mutation_factor=None,
                 crossover_prob=None, strategy=None, min_fidelity=None,
                 max_fidelity=None, eta=None, min_clip=None, max_clip=None,
                 boundary_fix_type='random', max_age=np.inf, **kwargs):
        # Miscellaneous
        self._setup_logger(kwargs)
        self.config_repository = ConfigRepository()

        # Benchmark related variables
        self.cs = cs
        self.configspace = True if isinstance(self.cs, ConfigSpace.ConfigurationSpace) else False
        if self.configspace:
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
            "configspace": self.configspace,
            "boundary_fix_type": self.fix_type,
            "max_age": self.max_age,
            "cs": self.cs,
            "dimensions": self.dimensions,
            "f": f
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
        self.max_SH_iter = None
        self.fidelities = None
        if self.min_fidelity is not None and \
           self.max_fidelity is not None and \
           self.eta is not None:
            self.max_SH_iter = -int(np.log(self.min_fidelity / self.max_fidelity) / np.log(self.eta)) + 1
            self.fidelities = self.max_fidelity * np.power(self.eta,
                                                     -np.linspace(start=self.max_SH_iter - 1,
                                                                  stop=0, num=self.max_SH_iter))

        # Updating DE parameter list
        self.de_params.update({"output_path": self.output_path})

        # Global trackers
        self.population = None
        self.fitness = None
        self.inc_score = np.inf
        self.inc_config = None
        self.history = []

    def _setup_logger(self, kwargs):
        """Sets up the logger."""
        self.output_path = kwargs['output_path'] if 'output_path' in kwargs else './'
        os.makedirs(self.output_path, exist_ok=True)
        self.logger = logger
        log_suffix = time.strftime("%x %X %Z")
        log_suffix = log_suffix.replace("/", '-').replace(":", '-').replace(" ", '_')
        self.logger.add(
            "{}/dehb_{}.log".format(self.output_path, log_suffix),
            **_logger_props
        )
        self.log_filename = "{}/dehb_{}.log".format(self.output_path, log_suffix)

    def reset(self):
        self.inc_score = np.inf
        self.inc_config = None
        self.population = None
        self.fitness = None
        self.traj = []
        self.runtime = []
        self.history = []
        self.logger.info("\n\nRESET at {}\n\n".format(time.strftime("%x %X %Z")))

    def init_population(self):
        raise NotImplementedError("Redefine!")

    def get_next_iteration(self, iteration):
        '''Computes the Successive Halving spacing

        Given the iteration index, computes the fidelity spacing to be used and
        the number of configurations to be used for the SH iterations.

        Parameters
        ----------
        iteration : int
            Iteration index
        clip : int, {1, 2, 3, ..., None}
            If not None, clips the minimum number of configurations to 'clip'

        Returns
        -------
        ns : array
        fidelities : array
        '''
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

    def get_incumbents(self):
        """ Returns a tuple of the (incumbent configuration, incumbent score/fitness). """
        if self.configspace:
            return self.vector_to_configspace(self.inc_config), self.inc_score
        return self.inc_config, self.inc_score

    def f_objective(self):
        raise NotImplementedError("The function needs to be defined in the sub class.")

    def run(self):
        raise NotImplementedError("The function needs to be defined in the sub class.")


class DEHB(DEHBBase):
    def __init__(self, cs=None, f=None, dimensions=None, mutation_factor=0.5,
                 crossover_prob=0.5, strategy='rand1_bin', min_fidelity=None,
                 max_fidelity=None, eta=3, min_clip=None, max_clip=None, configspace=True,
                 boundary_fix_type='random', max_age=np.inf, n_workers=None, client=None,
                 async_strategy="immediate", **kwargs):
        super().__init__(cs=cs, f=f, dimensions=dimensions, mutation_factor=mutation_factor,
                         crossover_prob=crossover_prob, strategy=strategy, min_fidelity=min_fidelity,
                         max_fidelity=max_fidelity, eta=eta, min_clip=min_clip, max_clip=max_clip,
                         configspace=configspace, boundary_fix_type=boundary_fix_type,
                         max_age=max_age, **kwargs)
        self.de_params.update({"async_strategy": async_strategy})
        self.iteration_counter = -1
        self.de = {}
        self._max_pop_size = None
        self.active_brackets = []  # list of SHBracketManager objects
        self.traj = []
        self.runtime = []
        self.history = []
        self.start = None

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

        # Misc.
        self.available_gpus = None
        self.gpu_usage = None
        self.single_node_with_gpus = None

    def __getstate__(self):
        """ Allows the object to picklable while having Dask client as a class attribute.
        """
        d = dict(self.__dict__)
        d["client"] = None  # hack to allow Dask client to be a class attribute
        d["logger"] = None  # hack to allow logger object to be a class attribute
        return d

    def __del__(self):
        """ Ensures a clean kill of the Dask client and frees up a port.
        """
        if hasattr(self, "client") and isinstance(self, Client):
            self.client.close()

    def _f_objective(self, job_info):
        """ Wrapper to call DE's objective function.
        """
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
        """ Generates a string to set the CUDA_VISIBLE_DEVICES environment variable.

        Given a list of available GPU device IDs and a preferred ID (start_id), the environment
        variable is created by putting the start_id device first, followed by the remaining devices
        arranged randomly. The worker that uses this string to set the environment variable uses
        the start_id GPU device primarily now.
        """
        assert start_id in available_gpus
        available_gpus = deepcopy(available_gpus)
        available_gpus.remove(start_id)
        np.random.shuffle(available_gpus)
        final_variable = [str(start_id)] + [str(_id) for _id in available_gpus]
        final_variable = ",".join(final_variable)
        return final_variable

    def distribute_gpus(self):
        """ Function to create a GPU usage tracker dict.

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
                  "Environment variable {} not visible!".format(str(e)))
            self.available_gpus = []
        self.gpu_usage = dict()
        for _id in self.available_gpus:
            self.gpu_usage[_id] = 0

    def vector_to_configspace(self, config):
        assert hasattr(self, "de")
        assert len(self.fidelities) > 0
        return self.de[self.fidelities[0]].vector_to_configspace(config)

    def configspace_to_vector(self, config):
        assert hasattr(self, "de")
        assert len(self.fidelities) > 0
        return self.de[self.fidelities[0]].configspace_to_vector(config)

    def reset(self):
        super().reset()
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
        self._get_pop_sizes()
        self._init_subpop()
        self.available_gpus = None
        self.gpu_usage = None

    def init_population(self, pop_size):
        if self.configspace:
            population = self.cs.sample_configuration(size=pop_size)
            population = [self.configspace_to_vector(individual) for individual in population]
        else:
            population = np.random.uniform(low=0.0, high=1.0, size=(pop_size, self.dimensions))
        return population

    def clean_inactive_brackets(self):
        """ Removes brackets from the active list if it is done as communicated by Bracket Manager
        """
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
        """Determines maximum pop size for each fidelity
        """
        self._max_pop_size = {}
        for i in range(self.max_SH_iter):
            n, r = self.get_next_iteration(i)
            for j, r_j in enumerate(r):
                self._max_pop_size[r_j] = max(
                    n[j], self._max_pop_size[r_j]
                ) if r_j in self._max_pop_size.keys() else n[j]

    def _init_subpop(self):
        """ List of DE objects corresponding to the fidelities
        """
        self.de = {}
        for i, f in enumerate(self._max_pop_size.keys()):
            self.de[f] = AsyncDE(**self.de_params, pop_size=self._max_pop_size[f],
                                 config_repository=self.config_repository)
            self.de[f].population = self.de[f].init_population(pop_size=self._max_pop_size[f])
            self.de[f].population_ids = self.config_repository.announce_population(self.de[f].population, f)
            self.de[f].fitness = np.array([np.inf] * self._max_pop_size[f])
            # adding attributes to DEHB objects to allow communication across subpopulations
            self.de[f].parent_counter = 0
            self.de[f].promotion_pop = None
            self.de[f].promotion_pop_ids = None
            self.de[f].promotion_fitness = None

    def _concat_pops(self, exclude_fidelity=None):
        """ Concatenates all subpopulations
        """
        fidelities = list(self.fidelities)
        if exclude_fidelity is not None:
            fidelities.remove(exclude_fidelity)
        pop = []
        for fidelity in fidelities:
            pop.extend(self.de[fidelity].population.tolist())
        return np.array(pop)

    def _start_new_bracket(self):
        """ Starts a new bracket based on Hyperband
        """
        # start new bracket
        self.iteration_counter += 1  # iteration counter gives the bracket count or bracket ID
        n_configs, fidelities = self.get_next_iteration(self.iteration_counter)
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

    def is_worker_available(self, verbose=False):
        """ Checks if at least one worker is available to run a job
        """
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
        """ Manages the population to be promoted from the lower to the higher fidelity.

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
        """ Maintains a looping counter over a subpopulation, to iteratively select a parent
        """
        parent_id = self.de[fidelity].parent_counter
        self.de[fidelity].parent_counter += 1
        self.de[fidelity].parent_counter = self.de[fidelity].parent_counter % self._max_pop_size[fidelity]
        return parent_id

    def _acquire_config(self, bracket, fidelity):
        """ Generates/chooses a configuration based on the fidelity and iteration number
        """
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

    def _get_next_job(self):
        """Loads a configuration and fidelity to be evaluated next.

        Returns:
            dict: Dicitonary containing all necessary information of the next job.
        """
        bracket = None
        if len(self.active_brackets) == 0 or \
                np.all([bracket.is_bracket_done() for bracket in self.active_brackets]):
            # start new bracket when no pending jobs from existing brackets or empty bracket list
            bracket = self._start_new_bracket()
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
                bracket = self._start_new_bracket()
        # fidelity that the SH bracket allots
        fidelity = bracket.get_next_job_fidelity()
        config, config_id, parent_id = self._acquire_config(bracket, fidelity)

        # transform config to proper representation
        if self.configspace:
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
            if bracket.bracket_id == job_info['bracket_id']:
                # registering is IMPORTANT for Bracket Manager to perform SH
                bracket.register_job(job_info['fidelity'])
                break
        return job_info

    def ask(self, n_configs: int=1):
        """Get the next configuration to run from the optimizer.

        The retrieved configuration can then be evaluated by the user.
        After evaluation use `tell` to report the results back to the optimizer.
        For more information, please refer to the description of `tell`.

        Args:
            n_configs (int, optional): Number of configs to ask for. Defaults to 1.

        Returns:
            dict or list of dict: Job info(s) of next configuration to evaluate.
        """
        if n_configs == 1:
            return self._get_next_job()

        jobs = []
        for _ in range(n_configs):
            jobs.append(self._get_next_job())
        return jobs

    def _get_gpu_id_with_low_load(self):
        candidates = []
        for k, v in self.gpu_usage.items():
            if v == min(self.gpu_usage.values()):
                candidates.append(k)
        device_id = np.random.choice(candidates)
        # creating string for setting environment variable CUDA_VISIBLE_DEVICES
        gpu_ids = self._create_cuda_visible_devices(
            self.available_gpus, device_id
        )
        # updating GPU usage
        self.gpu_usage[device_id] += 1
        self.logger.debug("GPU device selected: {}".format(device_id))
        self.logger.debug("GPU device usage: {}".format(self.gpu_usage))
        return gpu_ids

    def submit_job(self, job_info, **kwargs):
        """ Asks a free worker to run the objective function on config and fidelity
        """
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
        """ Iterate over futures and collect results from finished workers
        """
        if self.n_workers > 1 or isinstance(self.client, Client):
            done_list = [(i, future) for i, future in enumerate(self.futures) if future.done()]
        else:
            # Dask not invoked in the synchronous case
            done_list = [(i, future) for i, future in enumerate(self.futures)]
        if len(done_list) > 0:
            self.logger.debug(
                "Collecting {} of the {} job(s) active.".format(len(done_list), len(self.futures))
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
            brackets = self.iteration_counter + brackets

        return fevals, brackets

    def _is_run_budget_exhausted(self, fevals=None, brackets=None, total_cost=None):
        """ Checks if the DEHB run should be terminated or continued
        """
        delimiters = [fevals, brackets, total_cost]
        delim_sum = sum(x is not None for x in delimiters)
        if delim_sum == 0:
            raise ValueError(
                "Need one of 'fevals', 'brackets' or 'total_cost' as budget for DEHB to run."
            )
        if fevals is not None:
            if len(self.traj) >= fevals:
                return True
        elif brackets is not None:
            if self.iteration_counter >= brackets:
                for bracket in self.active_brackets:
                    # waits for all brackets < iteration_counter to finish by collecting results
                    if bracket.bracket_id < self.iteration_counter and \
                            not bracket.is_bracket_done():
                        return False
                return True
        else:
            if time.time() - self.start >= total_cost:
                return True
            if len(self.runtime) > 0 and self.runtime[-1] - self.start >= total_cost:
                return True
        return False

    def _save_incumbent(self, name=None):
        if name is None:
            name = time.strftime("%x %X %Z", time.localtime(self.start))
            name = name.replace("/", '-').replace(":", '-').replace(" ", '_')
        try:
            res = dict()
            if self.configspace:
                config = self.vector_to_configspace(self.inc_config)
                res["config"] = config.get_dictionary()
            else:
                res["config"] = self.inc_config.tolist()
            res["score"] = self.inc_score
            res["info"] = self.inc_info
            with open(os.path.join(self.output_path, "incumbent_{}.json".format(name)), 'w') as f:
                json.dump(res, f)
        except Exception as e:
            self.logger.warning("Incumbent not saved: {}".format(repr(e)))

    def _save_history(self, name=None):
        if name is None:
            name = time.strftime("%x %X %Z", time.localtime(self.start))
            name = name.replace("/", '-').replace(":", '-').replace(" ", '_')
        try:
            with open(os.path.join(self.output_path, "history_{}.pkl".format(name)), 'wb') as f:
                pickle.dump(self.history, f)
        except Exception as e:
            self.logger.warning("History not saved: {}".format(repr(e)))

    def _verbosity_debug(self):
        for bracket in self.active_brackets:
            self.logger.debug("Bracket ID {}:\n{}".format(
                bracket.bracket_id,
                str(bracket)
            ))

    def _verbosity_runtime(self, fevals, brackets, total_cost):
        if fevals is not None:
            remaining = (len(self.traj), fevals, "function evaluation(s) done")
        elif brackets is not None:
            _suffix = "bracket(s) started; # active brackets: {}".format(len(self.active_brackets))
            remaining = (self.iteration_counter + 1, brackets, _suffix)
        else:
            elapsed = np.format_float_positional(time.time() - self.start, precision=2)
            remaining = (elapsed, total_cost, "seconds elapsed")
        self.logger.info(
            "{}/{} {}".format(remaining[0], remaining[1], remaining[2])
        )

    def tell(self, job_info: dict, result: dict):
        """Feed a result back to the optimizer.

        In order to correctly interpret the results, the `job_info` dict, retrieved by `ask`,
        has to be given. Moreover, the `result` dict has to contain the keys `fitness` and `cost`.
        It is also possible to add the field `info` to the `result` in order to store additional,
        user-specific information.

        Args:
            job_info (dict): Job info returned by ask().
            result (dict): Result dictionary with mandatory keys `fitness` and `cost`.
        """
        # update bracket information
        fitness, cost = result["fitness"], result["cost"]
        info = result["info"] if "info" in result else dict()
        fidelity, parent_id = job_info["fidelity"], job_info["parent_id"]
        config, config_id = job_info["config"], job_info["config_id"]
        bracket_id = job_info["bracket_id"]
        for bracket in self.active_brackets:
            if bracket.bracket_id == bracket_id:
                # bracket job complete
                bracket.complete_job(fidelity)  # IMPORTANT to perform synchronous SH

        self.config_repository.tell_result(config_id, fidelity, fitness, cost, info)

        # get hypercube representation from config repo
        if self.configspace:
            config = self.config_repository.get(config_id)

        # carry out DE selection
        if fitness <= self.de[fidelity].fitness[parent_id]:
            self.de[fidelity].population[parent_id] = config
            self.de[fidelity].population_ids[parent_id] = config_id
            self.de[fidelity].fitness[parent_id] = fitness
        # updating incumbents
        if self.de[fidelity].fitness[parent_id] < self.inc_score:
            self._update_incumbents(
                config=self.de[fidelity].population[parent_id],
                score=self.de[fidelity].fitness[parent_id],
                info=info
            )
        # book-keeping
        self._update_trackers(
            traj=self.inc_score, runtime=cost, history=(
                config.tolist(), float(fitness), float(cost), float(fidelity), info
            )
        )

    @logger.catch
    def run(self, fevals=None, brackets=None, total_cost=None, single_node_with_gpus=False,
            verbose=False, debug=False, save_intermediate=True, save_history=True, name=None, **kwargs):
        """ Main interface to run optimization by DEHB

        This function waits on workers and if a worker is free, asks for a configuration and a
        fidelity to evaluate on and submits it to the worker. In each loop, it checks if a job
        is complete, fetches the results, carries the necessary processing of it asynchronously
        to the worker computations.

        The duration of the DEHB run can be controlled by specifying one of 3 parameters. If more
        than one are specified, DEHB selects only one in the priority order (high to low):
        1) Number of function evaluations (fevals)
        2) Number of Successive Halving brackets run under Hyperband (brackets)
        3) Total computational cost (in seconds) aggregated by all function evaluations (total_cost)
        """
        # check if run has already been called before
        if self.start is not None:
            logger.warning("DEHB has already been run. Calling 'run' twice could lead to unintended"
                           + " behavior. Please restart DEHB with an increased compute budget"
                           + " instead of calling 'run' twice.")

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
            self.distribute_gpus()

        self.start = self.start = time.time()
        fevals, brackets = self._adjust_budgets(fevals, brackets)
        if verbose:
            print("\nLogging at {} for optimization starting at {}\n".format(
                os.path.join(os.getcwd(), self.log_filename),
                time.strftime("%x %X %Z", time.localtime(self.start))
            ))
        if debug:
            logger.configure(handlers=[{"sink": sys.stdout}])
        while True:
            if self._is_run_budget_exhausted(fevals, brackets, total_cost):
                break
            if self.is_worker_available():
                job_info = self.ask()
                if brackets is not None and job_info["bracket_id"] >= brackets:
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
                            self._get_worker_count() - len(self.futures), self._get_worker_count()
                        ))
                    # submits job_info to a worker for execution
                    self.submit_job(job_info, **kwargs)
                    if verbose:
                        fidelity = job_info["fidelity"]
                        config_id = job_info["config_id"]
                        self._verbosity_runtime(fevals, brackets, total_cost)
                        self.logger.info(
                            "Evaluating configuration {} with fidelity {} under "
                            "bracket ID {}".format(config_id, fidelity, job_info["bracket_id"])
                        )
                        self.logger.info(
                            "Best score seen/Incumbent score: {}".format(self.inc_score)
                        )
                    self._verbosity_debug()
            self._fetch_results_from_workers()
            if save_intermediate and self.inc_config is not None:
                self._save_incumbent(name)
            if save_history and self.history is not None:
                self._save_history(name)
            self.clean_inactive_brackets()
        # end of while

        if verbose and len(self.futures) > 0:
            self.logger.info(
                "DEHB optimisation over! Waiting to collect results from workers running..."
            )
        while len(self.futures) > 0:
            self._fetch_results_from_workers()
            if save_intermediate and self.inc_config is not None:
                self._save_incumbent(name)
            if save_history and self.history is not None:
                self._save_history(name)
            time.sleep(0.05)  # waiting 50ms

        if verbose:
            time_taken = time.time() - self.start
            self.logger.info("End of optimisation! Total duration: {}; Total fevals: {}\n".format(
                time_taken, len(self.traj)
            ))
            self.logger.info("Incumbent score: {}".format(self.inc_score))
            self.logger.info("Incumbent config: ")
            if self.configspace:
                config = self.vector_to_configspace(self.inc_config)
                for k, v in config.get_dictionary().items():
                    self.logger.info("{}: {}".format(k, v))
            else:
                self.logger.info("{}".format(self.inc_config))
        self._save_incumbent(name)
        self._save_history(name)
        # reset waiting jobs of active bracket to allow for continuation
        if len(self.active_brackets) > 0:
            for active_bracket in self.active_brackets:
                active_bracket.reset_waiting_jobs()
        return np.array(self.traj), np.array(self.runtime), np.array(self.history, dtype=object)
