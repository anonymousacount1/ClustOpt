"""Registry of search-algorithm implementations.

To register a new search-algorithm implementation, add an entry mapping the
lowercase type-key used in configs to its constructor.
"""
from __future__ import annotations

from typing import Dict, Type

from models.ClustOpt.search_algorithm.Base_Search_Algorithm import BaseSearchAlgorithm
from models.ClustOpt.search_algorithm.search_algorithm_implementations.Brute_Force_Search import (
    BruteForceSearch,
)
from models.ClustOpt.search_algorithm.search_algorithm_implementations.Optuna_Search import (
    OptunaSearch,
)


SEARCH_ALGORITHM_REGISTRY: Dict[str, Type[BaseSearchAlgorithm]] = {
    "brute_force": BruteForceSearch,
    "optuna": OptunaSearch,
}
