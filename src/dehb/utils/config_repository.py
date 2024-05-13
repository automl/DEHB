from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ConfigItem:
    """Data class to store information regarding a specific configuration.

    The results for this configuration are stored in the `results` dict, using the fidelity it has
    been evaluated on as keys.
    """
    config_id: int
    config: np.ndarray
    results: dict[float, ResultItem]

@dataclass
class ResultItem:
    """Data class storing the result information of a specific configuration + fidelity."""
    score: float
    cost: float
    info: dict[Any, Any]

class ConfigRepository:
    """Bookkeeps all configurations used throughout the course of the optimization.

    Keeps track of the configurations and their results on the different fidelitites.
    A new configuration is announced via `announce_config`. After evaluating the configuration
    on the specified fidelity, use `tell_result` to log the achieved performance, cost etc.

    The configurations are stored in a list of `ConfigItem`.
    """
    def __init__(self) -> None:
        """Initializes the class by calling `self.reset`."""
        self.configs : list[ConfigItem]
        self.reset()

    def reset(self) -> None:
        """Resets the config repository, clearing all collected configurations and results."""
        self.configs = []
        self.initial_configs = []

    def announce_config(self, config: np.ndarray, fidelity=None) -> int:
        """Announces a new configuration with the respective fidelity it should be evaluated on.

        The configuration is then added to the list of so far seen configurations and the ID of the
        configuration is returned.

        Args:
            config (np.ndarray): New configuration
            fidelity (float, optional): Fidelity on which `config` is evaluated or None.
                                        Defaults to None.

        Returns:
            ID of configuration
        """
        config_id = len(self.configs)
        fidelity = float(fidelity or 0)
        result_dict = {
                fidelity: ResultItem(np.inf, -1, {}),
            }
        config_item = ConfigItem(config_id, config.copy(), result_dict)
        self.configs.append(config_item)
        return config_id

    def announce_population(self, population: np.ndarray, fidelity=None) -> np.ndarray:
        """Announce population, retrieving ids for the population.

        Args:
            population (np.ndarray): Population to announce
            fidelity (float, optional): Fidelity on which pop is evaluated or None.
                                        Defaults to None.

        Returns:
            population ids
        """
        population_ids = []
        for indiv in population:
            conf_id = self.announce_config(indiv, float(fidelity or 0))
            population_ids.append(conf_id)
        return np.array(population_ids)

    def announce_fidelity(self, config_id: int, fidelity: float):
        """Announce the evaluation of a new fidelity for a given config.

        This function may only be used if the config already exists in the repository.
        Note: This function is currently unused, but might be used later in order to
        allow for continuation.

        Args:
            config_id (int): ID of Configuration
            fidelity (float): Fidelity on which the config will be evaluated
        """
        try:
            config_item = self.configs[config_id]
        except IndexError as e:
            raise IndexError("Config with the given ID can not be found.") from e

        result_item = {
                fidelity: ResultItem(np.inf, -1, {}),
            }
        config_item.results[fidelity] = result_item

    def tell_result(self, config_id: int, fidelity: float, score: float, cost: float, info: dict):
        """Logs the achieved performance, cost etc. of a specific configuration-fidelity pair.

        Args:
            config_id (int): ID of evaluated configuration
            fidelity (float): Fidelity on which configuration has been evaluated.
            score (float): Achieved score, given by objective function
            cost (float): Cost, given by objective function
            info (dict): Run info, given by objective function
        """
        try:
            config_item = self.configs[config_id]
        except IndexError as e:
            raise IndexError("Config with the given ID can not be found.") from e

        # If configuration has been promoted, there is no fidelity information yet
        if fidelity not in config_item.results:
            config_item.results[fidelity] = ResultItem(score, cost, info)
        else:
            # ResultItem already given for specified fidelity --> update entries
            config_item.results[fidelity].score = score
            config_item.results[fidelity].cost = cost
            config_item.results[fidelity].info = info

    def get(self, config_id: int) -> np.ndarray:
        """Get the configuration with the given ID.

        Args:
            config_id (int): ID of config

        Returns:
            Config in hypercube representation
        """
        try:
            config_item = self.configs[config_id]
        except IndexError as e:
            raise IndexError("Config with the given ID can not be found.") from e
        return config_item.config

    def serialize_configs(self, configs) -> list:
        """Returns the configurations in logging format.

        Args:
            configs (list): Configs to parse into logging format

        Returns:
            Configs in logging format
        """
        serialized_data = []
        for config in configs:
            serialized_config = asdict(config)
            serialized_config["config"] = serialized_config["config"].tolist()
            serialized_data.append(serialized_config)
        return serialized_data
    def save_state(self, save_path: Path):
        """Saves the current state to `save_path`.

        Args:
            save_path (Path): Path where the state should be saved to.
        """
        with save_path.open("w") as f:
            serialized_data = self.serialize_configs(self.configs)
            json.dump(serialized_data, f, indent=2)

    def get_serialized_initial_configs(self):
        """Returns the initial configs in a format, that can be JSON serialized."""
        return self.serialize_configs(self.initial_configs)
