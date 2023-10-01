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

class ConfigRepository:
    def __init__(self) -> None:
        self.configs = []

    def reset(self) -> None:
        self.configs = []

    def announce_config(self, config: np.array, budget: float) -> int:
        config_id = len(self.configs)
        budget_info = {
                budget: BudgetItem(np.inf, -1),
            }
        config_item = ConfigItem(config_id, config, budget_info)
        self.configs.append(config_item)
        return config_id

    def tell_result(self, config_id: int, budget: float, score: float, cost: float):
        config_item = self.configs[config_id]

        # If configuration has been promoted, there is no budget information yet
        if budget not in config_item.budgets:
            config_item.budgets[budget] = BudgetItem(score, cost)
        else:
            config_item.budgets[budget].score = score
            config_item.budgets[budget].cost = cost