from __future__ import annotations

from dataclasses import dataclass
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

    def announce_config(self, config: np.ndarray, fidelity: float) -> int:
        """Announces a new configuration with the respective fidelity it should be evaluated on.

        The configuration is then added to the list of so far seen configurations and the ID of the
        configuration is returned.

        Args:
            config (np.ndarray): New configuration
            fidelity (float): Fidelity on which `config` is evaluated

        Returns:
            int: ID of configuration
        """
        config_id = len(self.configs)
        result_dict = {
                fidelity: ResultItem(np.inf, -1, {}),
            }
        config_item = ConfigItem(config_id, config, result_dict)
        self.configs.append(config_item)
        return config_id

    def announce_population(self, population: np.ndarray, fidelity=None) -> np.ndarray:
        """Announce population, retrieving ids for the population.

        Args:
            population (np.ndarray): Population to announce
            fidelity (float, optional): Fidelity on which pop is evaluated or None.
                                        Defaults to None.

        Returns:
            np.ndarray: population ids
        """
        population_ids = []
        for indiv in population:
            conf_id = self.announce_config(indiv, float(fidelity or 0))
            population_ids.append(conf_id)
        return np.array(population_ids)

    def announce_fidelity(self, config_id: int, fidelity: float):
        """Announce the evaluation of a new fidelity for a given config.

        This function may only be used if the config already exists in the repository.

        Args:
            config_id (int): ID of Configuration
            fidelity (float): Fidelity on which the config will be evaluated
        """
        if config_id >= len(self.configs) or config_id < 0:
            # TODO: Error message
            return

        config_item = self.configs[config_id]
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
        config_item = self.configs[config_id]

        # If configuration has been promoted, there is no fidelity information yet
        if fidelity not in config_item.results:
            config_item.results[fidelity] = ResultItem(score, cost, info)
        else:
            # ResultItem already given for specified fidelity --> update entries
            config_item.results[fidelity].score = score
            config_item.results[fidelity].cost = cost
            config_item.results[fidelity].info = info