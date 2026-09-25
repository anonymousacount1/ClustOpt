"""Warmstart repository for the derived ML2DAC MKRs.

ML2DAC's ApplicationPhase selects warmstart configurations by sorting a similar
dataset's evaluated configs by ARI and taking the best ones (in the original
code ARI is stored as a loss and sorted ascending; our master stores raw ARI so
we sort descending). We materialise, for every train record, all 32
configurations ranked by ARI so any top-k (1/5/10/25/32) can be sliced later.
"""
from __future__ import annotations

import pandas as pd

WARMSTART_COLUMNS = [
    "record_id", "config_id", "rank_by_ari", "ari",
    "algorithm", "hyperparameters_json", "predicted_k", "n_clusters_found",
    "split_id", "valid_result",
]


def build_warmstart_repository(evaluations: pd.DataFrame) -> pd.DataFrame:
    """All 32 configs per record, ranked by ARI descending (rank 1 = best ARI).

    Invalid configs (NaN ARI) sort last (rank stable within record). Ranking is
    dense+stable so ties keep a deterministic order by config_id.
    """
    cols = [c for c in WARMSTART_COLUMNS if c in evaluations.columns]
    df = evaluations[cols].copy()
    # Sort: best ARI first; NaN ARI (invalid) last; config_id as deterministic tiebreak.
    df = df.sort_values(
        ["record_id", "ari", "config_id"],
        ascending=[True, False, True],
        na_position="last",
        kind="mergesort",
    )
    df["rank_by_ari"] = df.groupby("record_id").cumcount() + 1
    return df.reset_index(drop=True)
