import typing

import numpy as np
from src.dehb.utils import ConfigRepository


class TestConfigAnnouncing():
    """Class that bundles all tests for announcing configurations to the repository."""
    def test_single_config(self):
        """Tests announcing single config."""
        repo = ConfigRepository()
        config = np.array([0.5])

        config_id = repo.announce_config(config, 2)

        assert len(repo.configs) == 1
        assert config_id == 0
        assert repo.configs[config_id].config == config

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