from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConfigItem:
    config_id: int
    config: np.array
    fidelitites: dict

@dataclass
class ResultItem:
    score: float
    cost: float
    info: dict

class ConfigRepository:
    def __init__(self) -> None:
        self.configs = []

    def reset(self) -> None:
        self.configs = []

    def announce_config(self, config: np.array, fidelity: float) -> int:
        config_id = len(self.configs)
        result_item = {
                fidelity: ResultItem(np.inf, -1, {}),
            }
        config_item = ConfigItem(config_id, config, result_item)
        self.configs.append(config_item)
        return config_id

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
        config_item.fidelities[fidelity] = result_item

    def tell_result(self, config_id: int, fidelity: float, score: float, cost: float, info: dict):
        config_item = self.configs[config_id]

        # If configuration has been promoted, there is no fidelity information yet
        if fidelity not in config_item.fidelities:
            config_item.fidelities[fidelity] = ResultItem(score, cost, info)
        else:
            config_item.fidelities[fidelity].score = score
            config_item.fidelities[fidelity].cost = cost
            config_item.fidelities[fidelity].info = info