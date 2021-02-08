import numpy as np


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

    def __repr__(self):
        cell_width = 9
        cell = "{{:^{}}}".format(cell_width)
        budget_cell = "{{:^{}.2f}}".format(cell_width)
        header = "|{}|{}|{}|{}|".format(
            cell.format("budget"),
            cell.format("pending"),
            cell.format("waiting"),
            cell.format("done")
        )
        _hline = "-" * len(header)
        table = [header, _hline]
        for i, budget in enumerate(self.budgets):
            pending = self.sh_bracket[budget]
            done = self._sh_bracket[budget]
            waiting = np.abs(self.n_configs[i] - pending - done)
            entry = "|{}|{}|{}|{}|".format(
                budget_cell.format(budget),
                cell.format(pending),
                cell.format(waiting),
                cell.format(done)
            )
            table.append(entry)
        table.append(_hline)
        return "\n".join(table)
