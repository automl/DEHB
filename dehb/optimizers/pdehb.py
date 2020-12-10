import time
import numpy as np
from distributed import client, Client

from .de import DE, AsyncDE
from .dehb import DEHB, DEHBBase


class SHBracketManager(object):
    """ Synchronous Successive Halving utilities
    """
    def __init__(self, n_configs, budgets, bracket_id=None):
        assert len(n_configs) == len(budgets)
        self.n_configs = n_configs
        self.budgets = budgets
        self.bracket_id = bracket_id
        self.sh_bracket = {}
        self._sh_bracket = {}
        self._config_map = {}
        for i, budget in enumerate(budgets):
            # sh_bracket keeps track of jobs/configs that are still to be scheduled/allocatted
            # _sh_bracket keeps track of jobs/configs that have been run and results retrieved for
            # (sh_bracket[i] + _sh_bracket[i]) == n_configs[i] is when no jobs have been scheduled
            #   or all jobs for that budget/rung are over
            # (sh_bracket[i] + _sh_bracket[i]) < n_configs[i] indicates a job has been scheduled
            #   and is queued/running and the bracket needs to be paused till results are retrieved
            self.sh_bracket[budget] = n_configs[i]  # each scheduled job does -= 1
            self._sh_bracket[budget] = 0  # each retrieved job does +=1
        self.n_rungs = len(budgets)
        self.current_rung = 0

    def get_budget(self, rung=None):
        """ Returns the exact budget that rung is pointing to.

        Returns current rung's budget if no rung is passed.
        """
        if rung is not None:
            return self.budgets[rung]
        return self.budgets[self.current_rung]

    def get_lower_budget_promotions(self, budget):
        """ Returns the immediate lower budget and the number of configs to be promoted from there
        """
        assert budget in self.budgets
        rung = np.where(budget == self.budgets)[0][0]
        prev_rung = np.clip(rung - 1, a_min=0, a_max=self.n_rungs-1)
        lower_budget = self.budgets[prev_rung]
        num_promote_configs = self.n_configs[rung]
        return lower_budget, num_promote_configs

    def get_next_job_budget(self):
        """ Returns the budget that will be selected if current_rung is incremented by 1
        """
        if self.sh_bracket[self.get_budget()] > 0:
            # the current rung still has unallocated jobs (>0)
            return self.get_budget()
        else:
            # the current rung has no more jobs to allocate, increment it
            rung = (self.current_rung + 1) % self.n_rungs
            if self.sh_bracket[self.get_budget(rung)] > 0:
                # the incremented rung has unallocated jobs (>0)
                return self.get_budget(rung)
            else:
                # all jobs for this bracket has been allocated/bracket is complete
                # no more budgets to evaluate and can return None
                pass
            return None

    def register_job(self, budget):
        """ Registers the allocation of a configuration for the budget and updates current rung

        This function must be called when scheduling a job in order to allow the bracket manager
        to continue job and budget allocation without waiting for jobs to finish and return
        results necessarily. This feature can be leveraged to run brackets asynchronously.
        """
        assert budget in self.budgets
        assert self.sh_bracket[budget] > 0
        self.sh_bracket[budget] -= 1
        if not self._is_rung_pending(self.current_rung):
            # increment current rung if no jobs left in the rung
            self.current_rung = (self.current_rung + 1) % self.n_rungs

    def complete_job(self, budget):
        """ Notifies the bracket that a job for a budget has been completed

        This function must be called when a config for a budget has finished evaluation to inform
        the Bracket Manager that no job needs to be waited for and the next rung can begin for the
        synchronous Successive Halving case.
        """
        assert budget in self.budgets
        _max_configs = self.n_configs[list(self.budgets).index(budget)]
        assert self._sh_bracket[budget] < _max_configs
        self._sh_bracket[budget] += 1

    def _is_rung_waiting(self, rung):
        """ Returns True if at least one job is still pending/running and waits for results
        """
        job_count = self._sh_bracket[self.budgets[rung]] + self.sh_bracket[self.budgets[rung]]
        if job_count < self.n_configs[rung]:
            return True
        return False

    def _is_rung_pending(self, rung):
        """ Returns True if at least one job pending to be allocatted in the rung
        """
        if self.sh_bracket[self.budgets[rung]] > 0:
            return True
        return False

    def previous_rung_waits(self):
        """ Returns True if none of the rungs < current rung is waiting for results
        """
        for rung in range(self.current_rung):
            if self._is_rung_waiting(rung) and not self._is_rung_pending(rung):
                return True
        return False

    def is_bracket_done(self):
        """ Returns True if all configs in all rungs in the bracket have been allocated
        """
        return ~self.is_pending() and ~self.is_waiting()

    def is_pending(self):
        """ Returns True if any of the rungs/budgets have still a configuration to submit
        """
        return np.any([self._is_rung_pending(i) > 0 for i, _ in enumerate(self.budgets)])

    def is_waiting(self):
        """ Returns True if any of the rungs/budgets have a configuration pending/running
        """
        return np.any([self._is_rung_waiting(i) > 0 for i, _ in enumerate(self.budgets)])


class PDEHB(DEHBBase):
    def __init__(self, cs=None, f=None, dimensions=None, mutation_factor=0.5,
                 crossover_prob=0.5, strategy='rand1_bin', min_budget=None,
                 max_budget=None, eta=3, min_clip=None, max_clip=None, configspace=True,
                 boundary_fix_type='random', max_age=np.inf, n_workers=1, **kwargs):
        super().__init__(cs=cs, f=f, dimensions=dimensions, mutation_factor=mutation_factor,
                         crossover_prob=crossover_prob, strategy=strategy, min_budget=min_budget,
                         max_budget=max_budget, eta=eta, min_clip=min_clip, max_clip=max_clip,
                         configspace=configspace, boundary_fix_type=boundary_fix_type,
                         max_age=max_age, n_workers=1, **kwargs)
        self.iteration_counter = -1
        self.de = {}
        self._max_pop_size = None
        self.active_brackets = []  # list of SHBracketManager objects
        self.traj = []
        self.runtime = []
        self.history = []

        # Dask variables
        self.n_workers = n_workers
        if self.n_workers > 1:
            self.client = Client(
                n_workers=self.n_workers, processes=True, threads_per_worker=1, scheduler_port=0
            )  # port 0 makes Dask select a random free port
        else:
            self.client = None
        self.futures = []
        self.stop_scheduling = False  # used only when DEHB runs delimited by number of brackets

        # Initializing DE subpopulations
        self._get_pop_sizes()
        self._init_subpop()

    def __getstate__(self):
        """ Allows the object to picklable while having Dask client as a class attribute.
        """
        d = dict(self.__dict__)
        d["client"] = None  # hack to allow Dask client to be a class attribute
        return d

    def __del__(self):
        """ Ensures a clean kill of the Dask client and frees up a port.
        """
        if hasattr(self, "client") and isinstance(self, Client):
            self.client.close()

    def _f_objective(self, job_info):
        """ Wrapper to call DE's objective function.
        """
        config, budget, parent_id = job_info['config'], job_info['budget'], job_info['parent_id']
        bracket_id = job_info['bracket_id']
        fitness, cost = self.de[budget].f_objective(config, budget)
        run_info = {
            'fitness': fitness,
            'cost': cost,
            'config': config,
            'budget': budget,
            'parent_id': parent_id,
            'bracket_id': bracket_id
        }
        return run_info

    def reset(self):
        super().reset()
        if hasattr(self, "client") and isinstance(self.client, Client):
            self.client.close()
        if self.n_workers > 1:
            self.client = Client(
                n_workers=self.n_workers, processes=True, threads_per_worker=1, scheduler_port=0
            )
        else:
            self.client = None
        self.futures = []
        self.stop_scheduling = False
        self.iteration_counter = -1
        self.de = {}
        self._max_pop_size = None
        self.active_brackets = []
        self.traj = []
        self.runtime = []
        self.history = []
        self._get_pop_sizes()
        self._init_subpop()

    def clean_inactive_brackets(self):
        """ Removes brackets from the active list if it is done as communicated by Bracket Manager
        """
        if len(self.active_brackets) == 0:
            return
        self.active_brackets = [
            bracket for bracket in self.active_brackets if ~bracket.is_bracket_done()
        ]
        return

    def _update_trackers(self, traj, runtime, history, budget):
        self.traj.append(traj)
        self.runtime.append(runtime)
        self.history.append(history)

    def _get_pop_sizes(self):
        """Determines maximum pop size for each budget
        """
        self._max_pop_size = {}
        for i in range(self.max_SH_iter):
            n, r = self.get_next_iteration(i)
            for j, r_j in enumerate(r):
                self._max_pop_size[r_j] = max(
                    n[j], self._max_pop_size[r_j]
                ) if r_j in self._max_pop_size.keys() else n[j]

    def _init_subpop(self):
        """ List of DE objects corresponding to the budgets (fidelities)
        """
        self.de = {}
        for i, b in enumerate(self._max_pop_size.keys()):
            self.de[b] = AsyncDE(**self.de_params, budget=b, pop_size=self._max_pop_size[b])
            self.de[b].population = self.de[b].init_population(pop_size=self._max_pop_size[b])
            self.de[b].fitness = np.array([np.inf] * self._max_pop_size[b])
            # adding attributes to DEHB objects to allow communication across subpopulations
            self.de[b].parent_counter = 0
            self.de[b].promotion_pop = None
            self.de[b].promotion_fitness = None

    def _concat_pops(self, exclude_budget=None):
        """ Concatenates all subpopulations
        """
        budgets = list(self.budgets)
        if exclude_budget is not None:
            budgets.remove(exclude_budget)
        pop = []
        for budget in budgets:
            pop.extend(self.de[budget].population.tolist())
        return np.array(pop)

    def _start_new_bracket(self):
        """ Starts a new bracket based on Hyperband
        """
        # start new bracket
        self.iteration_counter += 1  # iteration counter gives the bracket count or bracket ID
        n_configs, budgets = self.get_next_iteration(self.iteration_counter)
        bracket = SHBracketManager(
            n_configs=n_configs, budgets=budgets, bracket_id=self.iteration_counter
        )
        self.active_brackets.append(bracket)
        return bracket

    def is_worker_available(self):
        """ Checks if at least one worker is available to run a job
        """
        if self.stop_scheduling:
            # disables job allocation if number of brackets allocated exceeds the run argument
            return False
        if self.n_workers == 1:
            # in the synchronous case, one worker is always available
            return True
        if len(self.futures) >= sum(self.client.nthreads().values()):
            # pause/wait if active worker count greater allocated workers
            return False
        return True

    def _get_promotion_candidate(self, low_budget, high_budget, n_configs):
        """ Manages the population to be promoted from the lower to the higher budget.

        This is triggered or in action only during the first full HB bracket, which is equivalent
        to the number of brackets <= max_SH_iter.
        """
        # finding the individuals that have been evaluated (fitness < np.inf)
        evaluated_configs = np.where(self.de[low_budget].fitness != np.inf)[0]
        promotion_candidate_pop = self.de[low_budget].population[evaluated_configs]
        promotion_candidate_fitness = self.de[low_budget].fitness[evaluated_configs]
        # ordering the evaluated individuals based on their fitness values
        pop_idx = np.argsort(promotion_candidate_fitness)

        # creating population for promotion if none promoted yet or nothing to promote
        if self.de[high_budget].promotion_pop is None or \
                len(self.de[high_budget].promotion_pop) == 0:
            self.de[high_budget].promotion_pop = np.empty((0, self.dimensions))
            self.de[high_budget].promotion_fitness = np.array([])

            # iterating over the evaluated individuals from the lower budget and including them
            # in the promotion population for the higher budget only if it's not in the population
            # this is done to ensure diversity of population and avoid redundant evaluations
            for idx in pop_idx:
                individual = promotion_candidate_pop[idx]
                # checks if the candidate individual already exists in the high budget population
                if np.any(np.all(individual == self.de[high_budget].population, axis=1)):
                    # skipping already present individual to allow diversity and reduce redundancy
                    continue
                self.de[high_budget].promotion_pop = np.append(
                    self.de[high_budget].promotion_pop, [individual], axis=0
                )
                self.de[high_budget].promotion_fitness = np.append(
                    self.de[high_budget].promotion_pop, promotion_candidate_fitness[pop_idx]
                )
            # retaining only n_configs
            self.de[high_budget].promotion_pop = self.de[high_budget].promotion_pop[:n_configs]
            self.de[high_budget].promotion_fitness = self.de[high_budget].promotion_fitness[:n_configs]

        if len(self.de[high_budget].promotion_pop) > 0:
            config = self.de[high_budget].promotion_pop[0]
            # removing selected configuration from population
            self.de[high_budget].promotion_pop = self.de[high_budget].promotion_pop[1:]
            self.de[high_budget].promotion_fitness = self.de[high_budget].promotion_fitness[1:]
        else:
            # in case of an edge failure case where all high budget individuals are same
            # just choose the best performing individual from the lower budget (again)
            config = self.de[low_budget].population[pop_idx[0]]
        return config

    def _get_next_parent_for_subpop(self, budget):
        """ Maintains a looping counter over a subpopulation, to iteratively select a parent
        """
        parent_id = self.de[budget].parent_counter
        self.de[budget].parent_counter += 1
        self.de[budget].parent_counter = self.de[budget].parent_counter % self._max_pop_size[budget]
        return parent_id

    def _acquire_config(self, bracket, budget):
        """ Generates/chooses a configuration based on the budget and iteration number
        """
        # select a parent/target
        parent_id = self._get_next_parent_for_subpop(budget)
        target = self.de[budget].population[parent_id]
        # identify lower budget/fidelity to transfer information from
        lower_budget, num_configs = bracket.get_lower_budget_promotions(budget)

        if self.iteration_counter < self.max_SH_iter:
            # promotions occur only in the first set of SH brackets under Hyperband
            # for the first rung/budget in the current bracket, no promotion is possible and
            # evolution can begin straight away
            # for the subsequent rungs, individuals will be promoted from the lower_budget
            if budget != bracket.budgets[0]:
                # TODO: check if generalizes to all budget spacings
                config = self._get_promotion_candidate(lower_budget, budget, num_configs)
                return config, parent_id

        # DE evolution occurs when either all individuals in the subpopulation have been evaluated
        # at least once, i.e., has fitness < np.inf, which can happen if
        # iteration_counter <= max_SH_iter but certainly never when iteration_counter > max_SH_iter

        # a single DE evolution --- (mutation + crossover) occurs here
        mutation_pop_idx = np.argsort(self.de[lower_budget].fitness)[:num_configs]
        mutation_pop = self.de[lower_budget].population[mutation_pop_idx]
        # TODO: make global pop smarter --- select top configs from subpop?
        # generate mutants from previous budget subpopulation or global population
        if len(mutation_pop) < self.de[budget]._min_pop_size:
            filler = self.de[budget]._min_pop_size - len(mutation_pop) + 1
            new_pop = self.de[budget]._init_mutant_population(
                pop_size=filler, population=self._concat_pops(),
                target=None, best=self.inc_config
            )
            mutation_pop = np.concatenate((mutation_pop, new_pop))
        # generate mutant from among individuals in mutation_pop
        mutant = self.de[budget].mutation(
            current=target, best=self.inc_config, alt_pop=mutation_pop
        )
        # perform crossover with selected parent
        config = self.de[budget].crossover(target=target, mutant=mutant)
        config = self.de[budget].boundary_check(config)
        return config, parent_id

    def _get_next_job(self):
        """ Loads a configuration and budget to be evaluated next by a free worker
        """
        bracket = None
        if len(self.active_brackets) == 0 or \
                np.all([bracket.is_bracket_done() for bracket in self.active_brackets]):
            # start new bracket when list no pending jobs from existing brackets or empty list
            bracket = self._start_new_bracket()
        else:
            for _bracket in self.active_brackets:
                if not _bracket.previous_rung_waits() and _bracket.is_pending():
                    # bracket eligible for job scheduling
                    bracket = _bracket
                    break
            if bracket is None:
                # start new bracket when existing list has all waiting brackets
                bracket = self._start_new_bracket()
        # budget that the SH bracket allots
        budget = bracket.get_next_job_budget()
        config, parent_id = self._acquire_config(bracket, budget)
        # notifies the Bracket Manager that a single config is to run for the budget chosen
        job_info = {
            "config": config,
            "budget": budget,
            "parent_id": parent_id,
            "bracket_id": bracket.bracket_id
        }
        return job_info

    def submit_job(self, job_info):
        """ Asks a free worker to run the objective function on config and budget
        """
        # submit to to Dask client
        if self.n_workers > 1:
            self.futures.append(
                self.client.submit(self._f_objective, job_info)
            )
        else:
            # skipping scheduling to Dask worker to avoid added overheads in the synchronous case
            self.futures.append(self._f_objective(job_info))

        # pass information of job submission to Bracket Manager
        for bracket in self.active_brackets:
            if bracket.bracket_id == job_info['bracket_id']:
                # IMPORTANT for Bracket Manager to perform SH
                bracket.register_job(job_info['budget'])
                break

    def _fetch_results_from_workers(self):
        """ Iterate over futures and collect results from finished workers
        """
        if self.n_workers > 1:
            done_list = [(i, future) for i, future in enumerate(self.futures) if future.done()]
        else:
            # Dask not invoked in the synchronous case
            done_list = [(i, future) for i, future in enumerate(self.futures)]
        for _, future in done_list:
            if self.n_workers > 1:
                run_info = future.result()
            else:
                # Dask not invoked in the synchronous case
                run_info = future
            # update bracket information
            fitness, cost = run_info["fitness"], run_info["cost"]
            budget, parent_id = run_info["budget"], run_info["parent_id"]
            config = run_info["config"]
            bracket_id = run_info["bracket_id"]
            for bracket in self.active_brackets:
                if bracket.bracket_id == bracket_id:
                    # bracket job complete
                    bracket.complete_job(budget)  # IMPORTANT to perform synchronous SH

            # carry out DE selection
            if fitness <= self.de[budget].fitness[parent_id]:
                self.de[budget].population[parent_id] = config
                self.de[budget].fitness[parent_id] = fitness
            # updating incumbents
            if self.de[budget].fitness[parent_id] < self.inc_score:
                self.inc_score = self.de[budget].fitness[parent_id]
                self.inc_config = self.de[budget].population[parent_id]
            # book-keeping
            self._update_trackers(traj=self.inc_score, runtime=cost, budget=budget,
                                  history=(config.tolist(), float(fitness), float(budget)))
        # remove processed future
        self.futures = np.delete(self.futures, [i for i, _ in done_list]).tolist()

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
                self.stop_scheduling = True  # variable used by worker availability check
                for bracket in self.active_brackets:
                    if bracket.bracket_id < self.iteration_counter and not bracket.is_bracket_done():
                        return False
                return True
        else:
            if time.time() - self.start >= total_cost:
                return True
            if len(self.runtime) > 0 and self.runtime[-1] - self.start >= total_cost:
                return True
        return False

    def run(self, fevals=None, brackets=None, total_cost=None, verbose=False):
        """ Main interface to run optimization by DEHB

        This function waits on workers and if a worker is free, asks for a configuration and a
        budget to evaluate on and submits it to the worker. In each loop, it checks if a job
        is complete, fetches the results, carries the necessary processing of it asynchronously
        to the worker computations.

        The duration of the DEHB run can be controlled by specifying one of 3 parameters. If more
        than one are specified, DEHB selects only one in the priority order (high to low):
        1) Number of function evaluations (fevals)
        2) Number of Successive Halving brackets run under Hyperband (brackets)
        3) Total computational cost aggregated by all function evaluations (total_cost)
        """
        self.start = time.time()
        while True:
            if self._is_run_budget_exhausted(fevals, brackets, total_cost):
                break
            if self.is_worker_available():
                job_info = self._get_next_job()
                if verbose:
                    budget = job_info['budget']
                    print("{}, {}, {}".format(
                        job_info['bracket_id'], budget, self.inc_score
                    ))
                self.submit_job(job_info)
                for bracket in self.active_brackets:
                    print('    ', bracket.bracket_id, bracket.sh_bracket, bracket._sh_bracket)
            self._fetch_results_from_workers()
            self.clean_inactive_brackets()

        while len(self.futures) > 0:
            self._fetch_results_from_workers()
            time.sleep(self.min_budget)
            if verbose:
                print("DEHB optimisation over! Waiting to collect results from workers running...")
        if verbose:
            print("End of optimisation!")
        self.runtime = np.array(self.runtime) - self.start
        return np.array(self.traj), np.array(self.runtime), np.array(self.history)

# TODO: don't stop until all self.iteration_counter < self.brackets are complete
# TODO: fevals will be outweighed by lower fidelity evaluations especially for large n_workers
# TODO: solve bug in n=10 for optdigits (occurs for high workers) --> no unevaluated configs to select promotion from
