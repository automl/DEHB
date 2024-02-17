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

def create_toy_optimizer(configspace: ConfigSpace.ConfigurationSpace, min_fidelity: float,
                         max_fidelity: float, eta: int,
                         objective_function: typing.Callable):
    """Creates a DEHB instance.

    Args:
        configspace (ConfigurationSpace): Searchspace to use
        min_fidelity (float): Minimum fidelity for DEHB
        max_fidelity (float): Maximum fidelity for DEHB
        eta (int): Eta parameter of DEHB
        objective_function (Callable): Function to optimize

    Returns:
        _type_: _description_
    """
    dim = len(configspace.get_hyperparameters())
    return DEHB(f=objective_function, cs=configspace, dimensions=dim,
                min_fidelity=min_fidelity,
                max_fidelity=max_fidelity, eta=eta, n_workers=1)


def objective_function(x: ConfigSpace.Configuration, fidelity: float, **kwargs):
    """Toy objective function.

    Args:
        x (ConfigSpace.Configuration): Configuration to evaluate
        fidelity (float): fidelity to evaluate x on

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
        dehb = create_toy_optimizer(configspace=cs, min_fidelity=3, max_fidelity=27, eta=3,
                                        objective_function=objective_function)

        dehb.start = time.time() - 10

        assert dehb._is_run_budget_exhausted(total_cost=1), "Run budget should be exhausted"

    def test_fevals_exhaustion(self):
        """Test for function evaluations budget exhaustion."""
        cs = create_toy_searchspace()
        dehb = create_toy_optimizer(configspace=cs, min_fidelity=3, max_fidelity=27, eta=3,
                                    objective_function=objective_function)

        dehb.traj.append("Just needed for the test")

        assert dehb._is_run_budget_exhausted(fevals=1), "Run budget should be exhausted"

    def test_brackets_exhaustion(self):
        """Test for bracket budget exhaustion."""
        cs = create_toy_searchspace()
        dehb = create_toy_optimizer(configspace=cs, min_fidelity=3, max_fidelity=27, eta=3,
                                        objective_function=objective_function)

        dehb.iteration_counter = 5

        assert dehb._is_run_budget_exhausted(brackets=1), "Run budget should be exhausted"

class TestInitialization:
    """Class that bundles all tests regarding the initialization of DEHB."""
    def test_higher_min_fidelity(self):
        """Test that verifies, that DEHB breaks if min_fidelity > max_fidelity."""
        cs = create_toy_searchspace()
        with pytest.raises(AssertionError):
            create_toy_optimizer(configspace=cs, min_fidelity=28, max_fidelity=27, eta=3,
                                        objective_function=objective_function)

    def test_equal_min_max_fidelity(self):
        """Test that verifies, that DEHB breaks if min_fidelity == max_fidelity."""
        cs = create_toy_searchspace()
        with pytest.raises(AssertionError):
            create_toy_optimizer(configspace=cs, min_fidelity=27, max_fidelity=27, eta=3,
                                        objective_function=objective_function)

class TestConfigID:
    """Class that bundles all tests regarding config ID functionality."""
    def test_initialization(self):
        """Verifies, that the initial population is properly tracked by the config repository."""
        cs = create_toy_searchspace()
        dehb = create_toy_optimizer(configspace=cs, min_fidelity=3, max_fidelity=27, eta=3,
                                    objective_function=objective_function)
        # calculate how many configurations have been sampled for the initial populations
        num_configs = 0
        for de_inst in dehb.de.values():
            num_configs += len(de_inst.population)

        # config repository should be exactly this long
        assert len(dehb.config_repository.configs) == num_configs

    def test_single_bracket(self):
        """Verifies, that the population is continously tracked over the run of a single bracket."""
        cs = create_toy_searchspace()
        dehb = create_toy_optimizer(configspace=cs, min_fidelity=3, max_fidelity=27, eta=3,
                                    objective_function=objective_function)
        # calculate how many configurations have been sampled for the initial populations
        num_initial_configs = 0
        for de_inst in dehb.de.values():
            num_initial_configs += len(de_inst.population)

        # run for a single bracket
        dehb.run(brackets=1, verbose=True)

        # for the first bracket, we only mutate on the lowest fidelity and then promote the best
        # configs to the next fidelity. Please note, that this is only the case for the first
        # DEHB bracket!
        # Note: The final + 1 is due to the inner workings of DEHB. If the run budget is exhausted,
        # we keep evolving new configurations without evaluating them, since we are only waiting to
        # to fetch all results started ahead of the budget exhaustion.
        assert len(dehb.config_repository.configs) == num_initial_configs + 9 + 1