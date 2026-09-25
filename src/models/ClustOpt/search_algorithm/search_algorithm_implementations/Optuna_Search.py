"""Optuna-driven search.

The search uses Optuna's default TPE sampler. Each trial picks a clustering
algorithm and then samples that algorithm's hyper-parameters from the skopt
space description, translated on the fly to ``trial.suggest_*`` calls.

Failed evaluations return a finite very-low sentinel score so Optuna can
order trials and the early-stopping callback (if configured) keeps working
without ``-inf`` warnings.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import optuna
from optuna.study import Study
from optuna.trial import FrozenTrial

from models.ClustOpt.search_algorithm.Base_Search_Algorithm import BaseSearchAlgorithm


logger = logging.getLogger(__name__)


# Optuna prefers finite values for ordering and early-stopping math.
_FAILURE_SCORE: float = -1e12


class EarlyStoppingCallback:
    """Stop the Optuna study after ``patience`` trials without improvement."""

    def __init__(self, patience: int):
        self.patience = int(patience)
        self.best_score = -np.inf
        self.no_improve_count = 0

    def __call__(self, study: Study, trial: FrozenTrial) -> None:
        current_best = study.best_value
        if current_best > self.best_score:
            self.best_score = current_best
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1

        if self.no_improve_count >= self.patience:
            logger.info("Early stopping: no improvement in %d trials.", self.patience)
            study.stop()


class OptunaSearch(BaseSearchAlgorithm):
    def __init__(
        self,
        search_space,
        evaluator,
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None,
        patience: Optional[int] = None,
        seed: Optional[int] = None,
        sampler: Optional[str] = None,
    ):
        self.n_trials = n_trials
        self.timeout = timeout
        self.patience = patience
        # ``seed is None`` keeps the historical unseeded path (Optuna's default
        # TPESampler with a non-deterministic seed). Supplying a seed makes the
        # sampler reproducible, which the Stage-1 matched factorial requires.
        self.seed = None if seed is None else int(seed)
        # ``sampler is None`` keeps every historical path byte-for-byte: TPE,
        # seeded or not, exactly as before. ``sampler="random"`` selects
        # Optuna's uniform RandomSampler, which the E1 random-search control
        # requires: its proposals are independent of the objective values, so
        # the trajectory is a uniform draw from the same search space.
        self.sampler_name = None if sampler is None else str(sampler).lower()
        if self.sampler_name not in (None, "tpe", "random"):
            raise ValueError(
                f"Unsupported Optuna sampler '{sampler}'. "
                "Use 'tpe', 'random', or omit the key for the historical default."
            )
        super().__init__(
            search_space,
            evaluator,
            algorithm_name="OptunaSearch",
            search_algorithm_params={
                "n_trials": n_trials,
                "timeout": timeout,
                "patience": patience,
                "seed": self.seed,
                "sampler": self.sampler_name,
            },
        )

    def search_core(
        self,
        X_decision: np.ndarray,
        X_full: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        full_space = self.search_space.get_full_search_space()
        algo_names = list(full_space["algorithm"].categories)
        if not algo_names:
            raise ValueError("Cannot run Optuna search: the search space is empty.")

        best_score = -np.inf
        best_config: Optional[Dict[str, Any]] = None

        def objective(trial: optuna.Trial) -> float:
            nonlocal best_score, best_config

            algo_name = trial.suggest_categorical("algorithm", algo_names)
            config: Dict[str, Any] = {"algorithm": algo_name}
            prefix = f"{algo_name}__"

            for key, dim in full_space.items():
                if not key.startswith(prefix):
                    continue
                config[key] = self._suggest(trial, key, dim)

            score, _ = self.evaluate_config(algo_name, config, X_decision, X_full)
            trial.set_user_attr("config", config)

            if score > best_score:
                best_score = score
                best_config = config

            return score if np.isfinite(score) else _FAILURE_SCORE

        callbacks = []
        if self.patience is not None:
            callbacks.append(EarlyStoppingCallback(self.patience))

        # Historical path: no sampler argument => Optuna's default, unseeded TPE.
        if self.sampler_name == "random":
            sampler = optuna.samplers.RandomSampler(seed=self.seed)
        else:
            sampler = (
                optuna.samplers.TPESampler(seed=self.seed)
                if self.seed is not None else None
            )
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            callbacks=callbacks,
        )

        logger.info("Best configuration: %s | score=%.4f", best_config, best_score)
        return {"best_config": best_config, "score": best_score}

    # ----------------------------------------------------------- suggestions

    @staticmethod
    def _suggest(trial: optuna.Trial, name: str, dim: Any) -> Any:
        if hasattr(dim, "low") and hasattr(dim, "high"):
            if getattr(dim, "prior", None) == "log-uniform":
                return trial.suggest_float(name, dim.low, dim.high, log=True)
            if getattr(dim, "q", None):
                return trial.suggest_int(name, dim.low, dim.high, step=dim.q)
            if isinstance(dim.low, int) and isinstance(dim.high, int):
                return trial.suggest_int(name, dim.low, dim.high)
            return trial.suggest_float(name, dim.low, dim.high)

        if hasattr(dim, "categories"):
            return trial.suggest_categorical(name, list(dim.categories))

        raise ValueError(f"Unsupported parameter type for '{name}': {type(dim).__name__}")
