import typing

import numpy as np
import pytest
from src.dehb.utils import ConfigRepository


class TestConfigAnnouncing():
    """Class that bundles all tests for announcing configurations to the repository."""
    def test_single_config_fidelity(self):
        """Tests announcing single config with a specified fidelity."""
        repo = ConfigRepository()
        config = np.array([0.5])

        config_id = repo.announce_config(config, 2.)

        assert len(repo.configs) == 1
        assert config_id == 0
        assert repo.configs[config_id].config == config
        # result entry properly given
        assert repo.configs[config_id].results[2.] is not None

    def test_single_config_no_fidelity(self):
        """Tests announcing single config with a specified fidelity."""
        repo = ConfigRepository()
        config = np.array([0.5])

        config_id = repo.announce_config(config)

        assert len(repo.configs) == 1
        assert config_id == 0
        assert repo.configs[config_id].config == config
        # result entry properly given
        assert repo.configs[config_id].results[0.] is not None

    def test_population(self):
        """Tests announcing a whole population."""
        repo = ConfigRepository()
        pop = []
        for i in range(10):
            config = np.array([i / 10])
            pop.append(config)
        pop = np.array(pop)

        config_ids = repo.announce_population(pop)

        assert len(repo.configs) == 10

        for conf_id in config_ids:
            assert repo.configs[conf_id].config == pop[conf_id]

class TestGetConfig():
    """Class that bundles all tests regarding retrieving of configs via config ID."""
    def test_get_successful(self):
        """Test that get retrieves the right configuration."""
        repo = ConfigRepository()
        config = np.array([0.5])

        config_id = repo.announce_config(config)

        retrieved_config = repo.get(config_id)

        assert config == retrieved_config

    def test_get_failure(self):
        """Test to verify that get returns the right error if config ID is unkown."""
        repo = ConfigRepository()
        config = np.array([0.5])

        config_id = repo.announce_config(config)

        with pytest.raises(IndexError):
            repo.get(config_id + 1)

class TestTellResult():
    """This class bundles all tests regarding the `tell_result` method."""
    def test_tell_result_successful(self):
        repo = ConfigRepository()
        config = np.array([0.5])

        fidelity = 2.0
        config_id = repo.announce_config(config, fidelity)
        score = 1
        cost = 2
        info = {
            "test": 123,
        }
        repo.tell_result(config_id, fidelity, score, cost, info)

        config_item = repo.configs[config_id]
        results = config_item.results

        assert len(results) == 1
        assert results[fidelity].score == score
        assert results[fidelity].cost == cost
        assert results[fidelity].info == info