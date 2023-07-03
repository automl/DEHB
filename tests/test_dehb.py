import pytest
import ConfigSpace
import numpy as np
import time
from dehb.optimizers.dehb import DEHB

def create_toy_searchspace():
    cs = ConfigSpace.ConfigurationSpace()
    cs.add_hyperparameter(ConfigSpace.UniformFloatHyperparameter("x0", lower=3, upper=10, log=False))
    return cs

def create_toy_optimizer(configspace, min_budget, max_budget, eta, objective_function):
    dim = len(configspace.get_hyperparameters())
    return DEHB(f=objective_function, cs=configspace, dimensions=dim, min_budget=min_budget,
         max_budget=max_budget, eta=eta, n_workers=1)


def objective_function(x, budget, **kwargs):
    y = np.random.uniform()
    cost = 5
    result = {
        "fitness": y,
        "cost": cost
    }
    return result

class TestBudgetExhaustion():
    def test_runtime_exhaustion(self):
        cs = create_toy_searchspace()
        dehb = create_toy_optimizer(configspace=cs, min_budget=3, max_budget=27, eta=3,
                                        objective_function=objective_function)

        dehb.start = time.time() - 10

        assert dehb._is_run_budget_exhausted(total_cost=1), "Run budget should be exhausted"

    def test_fevals_exhaustion(self):
        cs = create_toy_searchspace()
        dehb = create_toy_optimizer(configspace=cs, min_budget=3, max_budget=27, eta=3,
                                    objective_function=objective_function)

        dehb.traj.append("Just needed for the test")

        assert dehb._is_run_budget_exhausted(fevals=1), "Run budget should be exhausted"

    def test_brackets_exhaustion(self):
        cs = create_toy_searchspace()
        dehb = create_toy_optimizer(configspace=cs, min_budget=3, max_budget=27, eta=3,
                                        objective_function=objective_function)

        dehb.iteration_counter = 5

        assert dehb._is_run_budget_exhausted(brackets=1), "Run budget should be exhausted"

