"""Deterministic stratified round-robin splitter.

Given the dataset index and a stratum key
(``family | subfamily | difficulty | cluster_count``), each stratum is shuffled
with a fixed seed and its datasets are distributed round-robin across the N
splits. Round-robin (rather than random assignment) guarantees that even tiny
strata are spread as evenly as possible, so all N splits stay balanced on every
stratification dimension.

``split_id`` is 1-based (``1..n_splits``).
"""
from __future__ import annotations

import hashlib
from typing import List

import pandas as pd

STRATUM_COLUMNS: List[str] = ["family", "subfamily", "difficulty", "cluster_count"]


def _stratum_key(row: pd.Series) -> str:
    return "|".join(str(row.get(c)) for c in STRATUM_COLUMNS)


def _stable_offset(stratum: str, seed: int) -> int:
    """A deterministic per-stratum starting offset.

    Spreading each stratum's *start* position around the ring (instead of
    always starting at split 0) prevents the first splits from systematically
    collecting the first dataset of every small stratum.
    """
    h = hashlib.md5(f"{seed}:{stratum}".encode("utf-8")).hexdigest()
    return int(h, 16)


def assign_splits(
    index_df: pd.DataFrame,
    *,
    n_splits: int = 16,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a copy of ``index_df`` with an added integer ``split_id`` (1..N)."""
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}.")
    if index_df.empty:
        out = index_df.copy()
        out["stratum_key"] = []
        out["split_id"] = []
        return out

    df = index_df.copy()
    df["stratum_key"] = df.apply(_stratum_key, axis=1)

    split_ids = pd.Series(index=df.index, dtype="int64")

    for stratum, group in df.groupby("stratum_key", sort=True):
        # Deterministic shuffle within the stratum (sort by dataset_id first so
        # the input order can't affect the result, then permute by seed).
        ordered = group.sort_values("dataset_id")
        rng = _StratumRng(seed=seed, stratum=str(stratum))
        permuted_index = rng.permutation(list(ordered.index))
        offset = _stable_offset(str(stratum), seed) % n_splits
        for i, idx in enumerate(permuted_index):
            split_ids.at[idx] = ((i + offset) % n_splits) + 1  # 1-based

    df["split_id"] = split_ids.astype(int)
    return df


class _StratumRng:
    """A tiny deterministic RNG seeded per (seed, stratum).

    Uses numpy's RandomState seeded from a hash of the stratum so the
    permutation is reproducible and independent of global RNG state.
    """

    def __init__(self, *, seed: int, stratum: str) -> None:
        import numpy as np
        h = hashlib.md5(f"{seed}:{stratum}".encode("utf-8")).hexdigest()
        self._rs = np.random.RandomState(int(h[:8], 16))

    def permutation(self, items: list) -> list:
        import numpy as np
        arr = np.array(items, dtype=object)
        perm = self._rs.permutation(len(arr))
        return [arr[i] for i in perm]
