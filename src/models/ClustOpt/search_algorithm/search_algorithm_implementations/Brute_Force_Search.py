"""Exhaustive (brute-force) search over the cartesian product of the search space.

For each clustering algorithm in the search space the routine enumerates a
discretised grid of every parameter dimension and evaluates every
combination. ``max_samples_per_param`` controls how many values are sampled
from continuous (``Real``) ranges; integer and categorical ranges always
enumerate exhaustively.
"""
from __future__ import annotations

import logging
from itertools import product
from typing import Any, Dict, List, Optional

import numpy as np
from skopt.space import Categorical, Integer, Real

from models.ClustOpt.search_algorithm.Base_Search_Algorithm import BaseSearchAlgorithm


logger = logging.getLogger(__name__)


class BruteForceSearch(BaseSearchAlgorithm):
    def __init__(self, search_space, evaluator, max_samples_per_param: int = 3):
        self.max_samples_per_param = int(max_samples_per_param)
        super().__init__(
            search_space,
            evaluator,
            algorithm_name="BruteForceSearch",
            search_algorithm_params={"max_samples_per_param": self.max_samples_per_param},
        )

    def search_core(
        self,
        X_decision: np.ndarray,
        X_full: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        full_space = self.search_space.get_full_search_space()
        algo_names = list(full_space["algorithm"].categories)

        best_score: float = -np.inf
        best_config: Optional[Dict[str, Any]] = None

        for algo_name in algo_names:
            algo_params_space = {
                k: v for k, v in full_space.items() if k.startswith(f"{algo_name}__")
            }

            param_keys = list(algo_params_space.keys())
            sampled_vals = [
                self._discretise(space) for space in algo_params_space.values()
            ]

            for combination in product(*sampled_vals):
                config: Dict[str, Any] = {"algorithm": algo_name}
                config.update(dict(zip(param_keys, combination)))

                score, _ = self.evaluate_config(algo_name, config, X_decision, X_full)
                if score > best_score:
                    best_score = score
                    best_config = config

        logger.info("Best configuration: %s | score=%.4f", best_config, best_score)
        return {"best_config": best_config, "score": best_score}

    # -------------------------------------------------------- discretisation

    def _discretise(self, space: Any) -> List[Any]:
        if isinstance(space, Integer):
            return list(range(space.low, space.high + 1))
        if isinstance(space, Real):
            return np.linspace(space.low, space.high, self.max_samples_per_param).tolist()
        if isinstance(space, Categorical):
            return list(space.categories)
        raise ValueError(f"Unsupported parameter space type: {type(space).__name__}")
