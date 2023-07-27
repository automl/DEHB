import time
import typing

import ConfigSpace
import numpy as np
import pytest
from src.dehb.optimizers.dehb import DEHB


def create_toy_searchspace():
    """Creates a toy searchspace with a single hyperparameter.

    Can be used in order to instantiate a DEHB instance for simple unittests not
    requiring a proper configuration space for optimization.


    Returns:
        ConfigurationSpace: Toy searchspace
    """
    cs = ConfigSpace.ConfigurationSpace()
    cs.add_hyperparameter(
        ConfigSpace.UniformFloatHyperparameter("x0", lower=3, upper=10, log=False))
    return cs

def create_toy_optimizer(configspace: ConfigSpace.ConfigurationSpace, min_budget: float,
                         max_budget: float, eta: int,
                         objective_function: typing.Callable):
    """Creates a DEHB instance.

    Args:
        configspace (ConfigurationSpace): Searchspace to use
        min_budget (float): Minimum budget for DEHB
        max_budget (float): Maximum budget for DEHB
        eta (int): Eta parameter of DEHB
        objective_function (Callable): Function to optimize

    Returns:
        _type_: _description_
    """
    dim = len(configspace.get_hyperparameters())
    return DEHB(f=objective_function, cs=configspace, dimensions=dim,
                min_budget=min_budget,
                max_budget=max_budget, eta=eta, n_workers=1)


def objective_function(x: ConfigSpace.Configuration, budget: float, **kwargs):
    """Toy objective function.

    Args:
        x (ConfigSpace.Configuration): Configuration to evaluate
        budget (float): Budget to evaluate x on

    Returns:
        dict: Result dictionary
    """
    y = np.random.uniform()
    cost = 5
    result = {
        "fitness": y,
        "cost": cost
    }
    return result

class TestBudgetExhaustion():
    """Class that bundles all Budget exhaustion tests.

    These tests include budget exhaustion tests for runtime, number of function
    evaluations and number of brackets to run.
    """
    def test_runtime_exhaustion(self):
        """Test for runtime budget exhaustion."""
        cs = create_toy_searchspace()
        dehb = create_toy_optimizer(configspace=cs, min_budget=3, max_budget=27, eta=3,
                                        objective_function=objective_function)

        dehb.start = time.time() - 10

        assert dehb._is_run_budget_exhausted(total_cost=1), "Run budget should be exhausted"

    def test_fevals_exhaustion(self):
        """Test for function evaluations budget exhaustion."""
        cs = create_toy_searchspace()
        dehb = create_toy_optimizer(configspace=cs, min_budget=3, max_budget=27, eta=3,
                                    objective_function=objective_function)

        dehb.traj.append("Just needed for the test")

        assert dehb._is_run_budget_exhausted(fevals=1), "Run budget should be exhausted"

    def test_brackets_exhaustion(self):
        """Test for bracket budget exhaustion."""
        cs = create_toy_searchspace()
        dehb = create_toy_optimizer(configspace=cs, min_budget=3, max_budget=27, eta=3,
                                        objective_function=objective_function)

        dehb.iteration_counter = 5

        assert dehb._is_run_budget_exhausted(brackets=1), "Run budget should be exhausted"

class TestInitialization:
    """Class that bundles all tests regarding the initialization of DEHB."""
    def test_higher_min_budget(self):
        """Test that verifies, that DEHB breaks if min_budget > max_budget."""
        cs = create_toy_searchspace()
        with pytest.raises(AssertionError):
            create_toy_optimizer(configspace=cs, min_budget=28, max_budget=27, eta=3,
                                        objective_function=objective_function)
