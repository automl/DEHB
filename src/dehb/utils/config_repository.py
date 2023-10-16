from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConfigItem:
    config_id: int
    config: np.array
    budgets: dict

@dataclass
class BudgetItem:
    score: float
    cost: float
    info: dict

class ConfigRepository:
    def __init__(self) -> None:
        self.configs = []

    def reset(self) -> None:
        self.configs = []

    def announce_config(self, config: np.array, budget: float) -> int:
        config_id = len(self.configs)
        budget_info = {
                budget: BudgetItem(np.inf, -1, {}),
            }
        config_item = ConfigItem(config_id, config, budget_info)
        self.configs.append(config_item)
        return config_id

    def announce_budget(self, config_id: int, budget: float):
        """Announce the evaluation of a new budget for a given config.

        This function may only be used if the config already exists in the repository.

        Args:
            config_id (int): ID of Configuration
            budget (float): Budget the config will be evaluated on
        """
        if config_id >= len(self.configs) or config_id < 0:
            # TODO: Error message
            return

        config_item = self.configs[config_id]
        budget_info = {
                budget: BudgetItem(np.inf, -1, {}),
            }
        config_item.budgets[budget] = budget_info

    def tell_result(self, config_id: int, budget: float, score: float, cost: float, info: dict):
        config_item = self.configs[config_id]

        # If configuration has been promoted, there is no budget information yet
        if budget not in config_item.budgets:
            config_item.budgets[budget] = BudgetItem(score, cost, info)
        else:
            config_item.budgets[budget].score = score
            config_item.budgets[budget].cost = cost
            config_item.budgets[budget].info = info