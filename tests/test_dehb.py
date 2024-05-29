import builtins
import io
import os
import typing

import ConfigSpace
import numpy as np
import pandas as pd
import pytest
from src.dehb.optimizers.dehb import DEHB


def patch_open(open_func, files):
    def open_patched(path, mode="r", buffering=-1, encoding=None,
                    errors=None, newline=None, closefd=True,
                    opener=None):
        if "w" in mode and not os.path.isfile(path):
            files.append(path)
        return open_func(path, mode=mode, buffering=buffering, 
                         encoding=encoding, errors=errors,
                         newline=newline, closefd=closefd, 
                         opener=opener)
    return open_patched


@pytest.fixture(autouse=True)
def cleanup_files(monkeypatch):
    """This fixture automatically cleans up all files that have been written by the tests after
    execution.
    """
    files = []
    monkeypatch.setattr(builtins, "open", patch_open(builtins.open, files))
    monkeypatch.setattr(io, "open", patch_open(io.open, files))
    yield
    for file in files:
        os.remove(file)

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
                         max_fidelity: float, eta: int, objective_function: typing.Callable,
                         save_freq: typing.Optional[str]=None, output_path: str="logs",
                         resume: bool=False):
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
    dim = len(configspace.get_hyperparameters()) if configspace else 1
    return DEHB(f=objective_function, cs=configspace, dimensions=dim,
                min_fidelity=min_fidelity, output_path=output_path, resume=resume,
                max_fidelity=max_fidelity, eta=eta, save_freq=save_freq, n_workers=1)


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

        dehb.run(total_cost=1)

        assert dehb._is_run_budget_exhausted(), "Run budget should be exhausted"

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
        assert len(dehb.config_repository.configs) == num_initial_configs + 9

class TestAskTell:
    """Class that bundles all tests regarding the ask and tell functionality of DEHB."""
    def test_all_fields_available(self):
        """Verifies, that all fields needed are present in job info returned by ask."""
        cs = create_toy_searchspace()
        dehb = create_toy_optimizer(configspace=cs, min_fidelity=3, max_fidelity=27, eta=3,
                                    objective_function=objective_function)
        conf = dehb.ask()
        assert "config" in conf
        assert "bracket_id" in conf
        assert "config_id" in conf
        assert "fidelity" in conf

    def test_format_configspace(self):
        """Verifies, that the returned config by ask() is of type Configuration
        if a configspace is passed.
        """
        cs = create_toy_searchspace()
        dehb = create_toy_optimizer(configspace=cs, min_fidelity=3, max_fidelity=27, eta=3,
                                    objective_function=objective_function)
        job_info = dehb.ask()
        assert isinstance(job_info["config"], ConfigSpace.Configuration)

    def test_format_no_configspace(self):
        """Verifies, that the returned config by ask() is of type Configuration
        if a configspace is passed.
        """
        dehb = create_toy_optimizer(configspace=None, min_fidelity=3, max_fidelity=27, eta=3,
                                    objective_function=objective_function)
        job_info = dehb.ask()
        assert isinstance(job_info["config"], np.ndarray)

    def test_ask_multiple(self):
        """Verifies, that ask can return multiple configs."""
        cs = create_toy_searchspace()
        dehb = create_toy_optimizer(configspace=cs, min_fidelity=3, max_fidelity=27, eta=3,
                                    objective_function=objective_function)
        job_infos = dehb.ask(2)

        assert len(job_infos) == 2
        assert job_infos[0]["config"] != job_infos[1]["config"]

    def test_ask_twice_different(self):
        """Verifies, that ask can return multiple configs."""
        cs = create_toy_searchspace()
        dehb = create_toy_optimizer(configspace=cs, min_fidelity=3, max_fidelity=27, eta=3,
                                    objective_function=objective_function)
        job_info_a = dehb.ask()
        job_info_b = dehb.ask()
        assert job_info_a != job_info_b

    def test_tell_twice(self):
        """Verifies, that tell should not be allowed to be called more often than ask."""
        cs = create_toy_searchspace()
        dehb = create_toy_optimizer(configspace=cs, min_fidelity=3, max_fidelity=27, eta=3,
                                    objective_function=objective_function)
        # Get single job info
        job_info = dehb.ask()
        res = objective_function(job_info["config"], job_info["fidelity"])

        # Tell twice, first should work
        dehb.tell(job_info, res)
        # Second tell should raise an error
        with pytest.raises(NotImplementedError):
            dehb.tell(job_info, res)

    def test_tell_successful(self):
        """Verifies, that tell successfully saves results."""
        cs = create_toy_searchspace()
        dehb = create_toy_optimizer(configspace=cs, min_fidelity=3, max_fidelity=27, eta=3,
                                    objective_function=objective_function)
        job_info = dehb.ask()
        id = job_info["config_id"]
        fid = job_info["fidelity"]
        conf = job_info["config"]

        # before telling, entry should be empty
        saved_score = dehb.config_repository.configs[id].results[fid].score
        assert saved_score == np.inf

        result = objective_function(conf, fid)
        dehb.tell(job_info, result)

        # after telling, score should be saved
        saved_score = dehb.config_repository.configs[id].results[fid].score
        assert saved_score == result["fitness"]

    def test_tell_error(self):
        """Verifies, that tell throws an error if config ID is non-existent."""
        cs = create_toy_searchspace()
        dehb = create_toy_optimizer(configspace=cs, min_fidelity=3, max_fidelity=27, eta=3,
                                    objective_function=objective_function)
        # get config
        job_info = dehb.ask()
        # adjust config id to non existing id
        job_info["config_id"] = 1337
        # create random result item
        result = {
            "fitness": 42,
            "cost": 123
        }
        # telling with wrong config_id should throw an error
        with pytest.raises(IndexError):
            dehb.tell(job_info, result)

class TestLogging:
    """Class that bundles all tests regarding the logging functionality of DEHB."""
    def test_init_no_save_freq(self):
        """Verifies, that default initializatin is 'end'."""
        cs = create_toy_searchspace()
        dehb = create_toy_optimizer(configspace=cs, min_fidelity=3, max_fidelity=27, eta=3,
                                    save_freq=None, objective_function=objective_function)
        assert dehb.save_freq == "end"
    def test_init_unkown_save_freq(self):
        """Verifies, that default initializatin if save_freq is unkown is 'end'."""
        cs = create_toy_searchspace()
        dehb = create_toy_optimizer(configspace=cs, min_fidelity=3, max_fidelity=27, eta=3,
                                    save_freq="7Hz", objective_function=objective_function)
        assert dehb.save_freq == "end"
    def test_state_before_eval(self):
        """Verifies, that returned state consists of all necessary fields."""
        cs = create_toy_searchspace()
        dehb = create_toy_optimizer(configspace=cs, min_fidelity=3, max_fidelity=27, eta=3,
                                    save_freq="end", objective_function=objective_function,
                                    output_path="state_test")
        state = dehb._get_state()
        hb_params = state["HB_params"]
        assert hb_params["min_fidelity"] == 3
        assert hb_params["max_fidelity"] == 27
        assert hb_params["eta"] == 3
        assert hb_params["min_clip"] is None
        assert hb_params["max_clip"] is None

        # There should not be an incumbent yet
        assert "inc_config" not in state["internals"]

        de_params = state["DE_params"]
        assert de_params["output_path"] == str(dehb.de_params["output_path"])
        de_params.pop("output_path")
        for key in de_params:
            assert de_params[key] == dehb.de_params[key]
    def test_freq_step(self):
        """Verifies, that the save_freq 'step' saves the state at the right times."""
        cs = create_toy_searchspace()
        dehb = create_toy_optimizer(configspace=cs, min_fidelity=3, max_fidelity=27, eta=3,
                                    save_freq="step", objective_function=objective_function,
                                    output_path="step_test")
        # Single ask/tell
        job_info = dehb.ask()
        result = objective_function(job_info["config"], job_info["fidelity"])
        dehb.tell(job_info, result)

        # Now state should be saved --> load history
        history_path = dehb.output_path / "history.parquet.gzip"
        history = pd.read_parquet(history_path)

        assert len(history) == len(dehb.history)

        # Second ask/tell
        job_info = dehb.ask()
        # Result should be worse than first result so that it can not trigger "incumbent" save_freq
        result["fitness"] += 10
        dehb.tell(job_info, result)

        # Now state should be saved --> load history
        history_path = dehb.output_path / "history.parquet.gzip"
        history = pd.read_parquet(history_path)

        assert len(history) == len(dehb.history)

    def test_freq_incumbent(self):
        """Verifies, that the save_freq 'incumbent' saves the state at the right times."""
        cs = create_toy_searchspace()
        dehb = create_toy_optimizer(configspace=cs, min_fidelity=3, max_fidelity=27, eta=3,
                                    save_freq="incumbent", objective_function=objective_function,
                                    output_path="incumbent_test")
        # Single ask/tell
        job_info = dehb.ask()
        result = objective_function(job_info["config"], job_info["fidelity"])
        dehb.tell(job_info, result)

        # Now state should be saved, because first config is always incumbent --> load config_repo
        history_path = dehb.output_path / "history.parquet.gzip"
        history = pd.read_parquet(history_path)

        assert len(history) == len(dehb.history)

        # Second ask/tell
        job_info = dehb.ask()
        # Result should be worse than first result so that it can not trigger "incumbent" save_freq
        result["fitness"] += 10
        dehb.tell(job_info, result)

        # State should not have been updated
        history_path = dehb.output_path / "history.parquet.gzip"
        history = pd.read_parquet(history_path)

        assert len(history) == len(dehb.history) - 1

class TestRestart:
    """Class that bundles all tests regarding the restarting functionality of DEHB."""
    def test_restart_run(self):
        """Verifies, that restarting after calling "run" works as expected."""
        cs = create_toy_searchspace()
        dehb = create_toy_optimizer(configspace=cs, min_fidelity=3, max_fidelity=27, eta=3,
                                    save_freq="step", objective_function=objective_function,
                                    output_path="restart_run_test")
        # Run for a single bracket
        traj, _, _ = dehb.run(brackets=1)
        n_configs = len(dehb.config_repository.configs)
        it_counter = dehb.iteration_counter
        len_traj = len(traj)

        # Load checkpoint saved by previous run
        dehb = create_toy_optimizer(configspace=cs, min_fidelity=3, max_fidelity=27, eta=3,
                                    save_freq="step", objective_function=objective_function,
                                    output_path="restart_run_test", resume=True)

        assert n_configs == len(dehb.config_repository.configs)
        assert it_counter == dehb.iteration_counter
        assert len_traj == len(dehb.traj)

    def test_restart_ask_tell(self):
        """Verifies, that restarting after using ask & tell works as expected."""
        cs = create_toy_searchspace()
        dehb = create_toy_optimizer(configspace=cs, min_fidelity=3, max_fidelity=27, eta=3,
                                    save_freq="step", objective_function=objective_function,
                                    output_path="restart_ask_tell_test")
        # Run for 10 feval
        for _ in range(10):
            job_info = dehb.ask()
            result = objective_function(job_info["config"], job_info["fidelity"])
            dehb.tell(job_info, result)

        n_configs = len(dehb.config_repository.configs)
        it_counter = dehb.iteration_counter
        len_traj = len(dehb.traj)

        # Load checkpoint saved by previous run
        dehb = create_toy_optimizer(configspace=cs, min_fidelity=3, max_fidelity=27, eta=3,
                                    save_freq="step", objective_function=objective_function,
                                    output_path="restart_ask_tell_test", resume=True)

        assert n_configs == len(dehb.config_repository.configs)
        assert it_counter == dehb.iteration_counter
        assert len_traj == len(dehb.traj)

    def test_restart_non_matching_configs(self):
        """Verifies, that restarting throws an error when dehb instances are configured differently."""
        cs = create_toy_searchspace()
        dehb = create_toy_optimizer(configspace=cs, min_fidelity=3, max_fidelity=27, eta=3,
                                    save_freq="step", objective_function=objective_function,
                                    output_path="restart_error_test")
        dehb.save()

        # Try to load checkpoint with different conifguration
        with pytest.raises(AttributeError):
            dehb = create_toy_optimizer(configspace=cs, min_fidelity=8, max_fidelity=123, eta=42,
                                    save_freq="step", objective_function=objective_function,
                                    output_path="restart_error_test", resume=True)

class TestDeprecation:
    """Class that bundles all tests regarding deprecation warnings."""
    def test_budget_deprecation(self):
        """Verifies, that an error is thrown if the user uses the old budget interface."""
        cs = create_toy_searchspace()
        with pytest.raises(TypeError):
            dehb = DEHB(cs, objective_function, len(cs.get_hyperparameters()), min_budget=2,
                        max_budget=5)