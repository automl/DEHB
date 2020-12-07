import numpy as np
from distributed import client, Client

from .de import DE, AsyncDE
from .dehb import DEHB, DEHBBase


class SHBracketManager(object):
    def __init__(self, n_configs, budgets):
        assert len(n_configs) == len(budgets)
        self.n_configs = n_configs
        self.budgets = budgets
        self.sh_bracket = {}
        for i, budget in enumerate(budgets):
            self.sh_bracket[budget] = n_configs[i]
        self.n_rungs = len(budgets)
        self.current_rung = 0

    def get_current_budget(self):
        return self.budgets[self.current_rung]

    def get_lower_budget_promotions(self, budget):
        """ Returns the lower budget and the number of configs to be promoted from there
        """
        assert budget in self.budgets
        rung = np.where(budget == self.budgets)[0][0]
        prev_rung = np.clip(rung - 1, a_min=0, a_max=self.n_rungs-1)
        lower_budget = self.budgets[prev_rung]
        num_promote_configs = self.n_configs[rung]
        return lower_budget, num_promote_configs

    def get_next_job_budget(self):
        if self.sh_bracket[self.get_current_budget()] > 0:
            return self.get_current_budget()
        else:
            self.current_rung = (self.current_rung + 1) % self.n_rungs
            if self.sh_bracket[self.get_current_budget()] > 0:
                return self.get_current_budget()
            return None

    def register_job(self, budget=None):
        if budget is None:
            budget = self.get_next_job_budget()
        assert budget in self.budgets
        self.sh_bracket[budget] -= 1
        assert self.sh_bracket[budget] >= 0

    def is_bracket_done(self):
        return ~self.is_pending()

    def is_pending(self):
        """ Returns True if any of the rungs/budgets have still a configuration to submit
        """
        return np.any([self.sh_bracket[budget] > 0 for budget in self.budgets])


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
        self.client = Client(
            n_workers=self.n_workers, processes=True, threads_per_worker=1, scheduler_port=0
        )
        self.futures = []

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
        if isinstance(self.client, Client):
            self.client.close()

    def _f_objective(self, job_info):
        """ Wrapper to call DE's objective function.
        """
        config, budget, parent_idx = job_info['config'], job_info['budget'], job_info['parent_idx']
        fitness, cost = self.de[budget].f_objective(config, budget)
        run_info = {
            'fitness': fitness,
            'cost': cost
        }
        return run_info

    def reset(self):
        super().reset()
        if isinstance(self.client, Client):
            self.client.close()
        self.client = Client(
            n_workers=self.n_workers, processes=True, threads_per_worker=1, scheduler_port=0
        )
        self.futures = []
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

    def is_worker_available(self):
        """ Checks if at least one worker is available to run a job
        """
        return True
        # if np.random.uniform() > 0.5:
        #     return True
        # return False

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
            # just choose the best performing individual from the lowest budget
            config = self.de[low_budget].population[pop_idx[0]]
        return config

    def _get_next_parent_for_subpop(self, budget):
        """ Maintains a looping counter over a subpopulation, to iteratively select a parent
        """
        parent_idx = self.de[budget].parent_counter
        self.de[budget].parent_counter += 1
        self.de[budget].parent_counter = self.de[budget].parent_counter % self._max_pop_size[budget]
        return parent_idx

    def _acquire_config(self, bracket, budget):
        """ Generates/chooses a configuration based on the budget and iteration number
        """
        # select a parent/target
        parent_idx = self._get_next_parent_for_subpop(budget)
        target = self.de[budget].population[parent_idx]
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
                return config, parent_idx

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
        return config, parent_idx

    def _get_next_job(self):
        """ Loads a configuration and budget to be evaluated next by a free worker
        """
        # if no bracket started/active OR all currently active brackets are waiting for results
        # then begin preparations to start next bracket
        if len(self.active_brackets) == 0 or \
                np.all([~bracket.is_pending() for bracket in self.active_brackets]):
            self.iteration_counter += 1  # iteration counter gives the bracket count or bracket ID
            n_configs, budgets = self.get_next_iteration(self.iteration_counter)
            self.active_brackets.append(SHBracketManager(
                n_configs=n_configs, budgets=budgets
            ))

        # at least one bracket has pending jobs or all active_brackets have is_pending() as False
        # in the latter case, the latest added bracket will have pending jobs to run
        for bracket in self.active_brackets:
            # in the order of brackets added, find the first such bracket with pending jobs
            if bracket.is_pending():
                break
        budget = bracket.get_next_job_budget()
        config, parent_idx = self._acquire_config(bracket, budget)
        # notifies the Bracket Manager that a single config is to run for the budget chosen
        bracket.register_job(budget)  # IMPORTANT for Bracket Manager to perform SH
        job_info = {
            "config": config,
            "budget": budget,
            "parent_idx": parent_idx
        }
        return job_info

    def submit_job(self, job_info):
        """ Asks a free worker to run the objective function on config and budget
        """
        # submit to to Dask client
        self.futures.append(
            self.client.submit(self._f_objective, job_info)
        )

    def _fetch_results_from_workers(self):
        """ Iterate over futures and collect results from finished workers
        """
        done_list = [future for future in self.futures if future.done()]
        for future in done_list:
            run_info = future.result()
            self.futures.remove(future)
            fitness, cost = run_info["fitness"], run_info["cost"]
            budget, parent_idx = run_info["budget"], run_info["parent_idx"]
            config = run_info["config"]

            # carry out DE selection
            if fitness <= self.de[budget].fitness[parent_idx]:
                self.de[budget].population[parent_idx] = config
                self.de[budget].fitness[parent_idx] = fitness
            # updating incumbents
            if self.de[budget].fitness[parent_idx] < self.inc_score:
                self.inc_score = self.de[budget].fitness[parent_idx]
                self.inc_config = self.de[budget].population[parent_idx]
            # book-keeping
            self._update_trackers(traj=self.inc_score, runtime=cost, budget=budget,
                                  history=(config.tolist(), float(fitness), float(budget)))

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
                return True
        else:
            if np.sum(self.runtime) >= total_cost:
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
        while True:
            if self._is_run_budget_exhausted(fevals, brackets, total_cost):
                break
            if self.is_worker_available():
                job_info = self._get_next_job()
                if verbose:
                    budget = job_info['budget']
                    print("{}, {}, {}".format(self.iteration_counter, budget, self.inc_score))
                self.submit_job(job_info)
            self._fetch_results_from_workers()
            self.clean_inactive_brackets()
        if verbose:
            print("End of optimisation!")
