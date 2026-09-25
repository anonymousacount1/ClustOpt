import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau


def _safe_float_array(x: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    return arr


def _extract_metric_basenames(df: pd.DataFrame) -> list[str]:
    """
    Discovers metric base names from:
      - "silhouette (w=1.0)"  -> "silhouette"
      - "silhouette_norm"     -> "silhouette"
    """
    basenames = set()

    for col in df.columns:
        if "(w=" in col:
            basenames.add(col.split(" (w=")[0])

    for col in df.columns:
        if col.endswith("_norm"):
            basenames.add(col[:-5])

    return sorted(basenames)


def _get_metric_values(df: pd.DataFrame, base: str, prefer_normalized: bool = True) -> tuple[np.ndarray, str]:
    """
    Returns metric values and which column was used.
    Preference:
      1) base_norm
      2) base (w=...)
    """
    norm_col = f"{base}_norm"
    if prefer_normalized and norm_col in df.columns:
        return _safe_float_array(df[norm_col]), norm_col

    raw_cols = [c for c in df.columns if c.startswith(base) and "(w=" in c]
    if raw_cols:
        return _safe_float_array(df[raw_cols[0]]), raw_cols[0]

    raise ValueError(f"Metric '{base}' not found as '{base}_norm' or '{base} (w=...)'.")


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    # Treat NaNs as -inf (worst) for "higher is better"
    s = scores.copy()
    s[~np.isfinite(s)] = -np.inf
    return np.argsort(-s)[:k]


def _bottomk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    # Treat NaNs as +inf (worst) for "lower is better"
    s = scores.copy()
    s[~np.isfinite(s)] = np.inf
    return np.argsort(s)[:k]


def _overlap_ratio(idx_a: np.ndarray, idx_b: np.ndarray) -> float:
    a, b = set(map(int, idx_a)), set(map(int, idx_b))
    return len(a.intersection(b)) / max(1, len(a))


def _ndcg_at_k_from_relevance(relevances: np.ndarray, order_idx: np.ndarray, k: int) -> float:
    """
    relevances: array aligned with df rows (e.g., ARI values)
    order_idx: indices ordered by some metric's predicted ranking
    Computes NDCG@k using:
      gain = (2^rel - 1) / log2(rank+1)
    Assumes relevances are in [0,1] (ARI often is; if not, still works but gains blow up).
    """
    k = max(1, int(k))
    idx = order_idx[:k]

    rel = relevances.copy()
    rel[~np.isfinite(rel)] = 0.0

    rel_k = rel[idx]
    gains = (2.0 ** rel_k - 1.0)
    discounts = np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gains / discounts))

    # ideal
    ideal_idx = np.argsort(-rel)[:k]
    ideal_rel_k = rel[ideal_idx]
    ideal_gains = (2.0 ** ideal_rel_k - 1.0)
    idcg = float(np.sum(ideal_gains / discounts))

    if idcg <= 1e-12:
        return 0.0
    return dcg / idcg


def _ndcg_full_list(scores: np.ndarray, relevances: np.ndarray) -> float:
    """Full-list NDCG (NDCG@n).

    Rank every candidate by metric score (descending) and grade the resulting
    ordering against ``relevances`` (ARI). Measures global ranking quality —
    not just the top or bottom slice, but the entire ordering.
    """
    n = int(scores.shape[0])
    if n < 2:
        return 0.0
    order = np.argsort(-np.nan_to_num(scores, nan=-np.inf))
    return _ndcg_at_k_from_relevance(relevances, order, k=n)


def _pairwise_accuracy(
    scores: np.ndarray,
    relevances: np.ndarray,
    max_pairs: int = 100_000,
    seed: int = 0,
) -> float:
    """Fraction of pairs whose metric ordering agrees with relevance ordering.

    A pair ``(i, j)`` is *correct* iff ``sign(scores[i] - scores[j])`` equals
    ``sign(relevances[i] - relevances[j])``. Pairs are skipped when either
    side ties (zero difference) or any of the four values is non-finite.

    For ``n*(n-1)/2 <= max_pairs`` every distinct unordered pair is used;
    otherwise ``max_pairs`` random index pairs are sampled (with replacement,
    self-pairs filtered) for tractability on large logs.

    Returns a value in ``[0, 1]`` where ``0.5`` is chance, ``1.0`` is perfect
    agreement and ``0.0`` is perfect disagreement. This complements the global
    rank correlations (Spearman / Kendall) with a focused *local pair
    ordering* view that's robust to non-finite rows.
    """
    scores = np.asarray(scores, dtype=float)
    relevances = np.asarray(relevances, dtype=float)
    n = int(scores.shape[0])
    if n < 2:
        return 0.0

    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        i_arr, j_arr = np.triu_indices(n, k=1)
    else:
        rng = np.random.default_rng(seed)
        i_raw = rng.integers(0, n, size=max_pairs)
        j_raw = rng.integers(0, n, size=max_pairs)
        keep = i_raw != j_raw
        i_arr = np.minimum(i_raw[keep], j_raw[keep])
        j_arr = np.maximum(i_raw[keep], j_raw[keep])

    si, sj = scores[i_arr], scores[j_arr]
    ri, rj = relevances[i_arr], relevances[j_arr]

    finite = np.isfinite(si) & np.isfinite(sj) & np.isfinite(ri) & np.isfinite(rj)
    if not finite.any():
        return 0.0
    si, sj, ri, rj = si[finite], sj[finite], ri[finite], rj[finite]

    ds = si - sj
    dr = ri - rj
    nontie = (ds != 0) & (dr != 0)
    if not nontie.any():
        return 0.0

    concordant = (ds[nontie] > 0) == (dr[nontie] > 0)
    return float(np.mean(concordant))


def compute_metric_utilities(
    csv_path: str,
    output_path: str | None = None,
    prefer_normalized: bool = True,
    top_k_ratio: float = 0.10,
    bottom_k_ratio: float = 0.10,
    # ------------------------------------------------------------------
    # Utility weights. The defaults are tuned to reward *full ranking
    # ability* (Spearman / Kendall / pairwise / NDCG-all) while keeping
    # Top-K / Bottom-K terms as focused diagnostics. They sum to 1.00:
    #
    #   global rank correlation      : w_spearman + w_kendall          = 0.30
    #   pairwise local ordering      : w_pairwise                      = 0.20
    #   full-list NDCG               : w_ndcg_all                      = 0.20
    #   focused Top-K / Bottom-K     : w_top_overlap + w_bottom_overlap
    #                                  + w_top_ndcg + w_bottom_ndcg    = 0.30
    #                                                            total = 1.00
    #
    # The final utility is still clipped to [0, 1] as a safety guard, so
    # callers can retune any single weight without breaking downstream
    # invariants.
    # ------------------------------------------------------------------
    w_spearman: float = 0.18,
    w_kendall: float = 0.12,
    w_pairwise: float = 0.20,
    w_ndcg_all: float = 0.20,
    w_top_overlap: float = 0.05,
    w_bottom_overlap: float = 0.05,
    w_top_ndcg: float = 0.10,
    w_bottom_ndcg: float = 0.10,
    pairwise_max_pairs: int = 100_000,
) -> pd.DataFrame:

    df = pd.read_csv(csv_path)

    # Basic filtering
    if "valid" in df.columns:
        df = df[df["valid"] == True].copy()

    if "ARI" not in df.columns:
        raise ValueError("ARI column not found in CSV.")

    ari = _safe_float_array(df["ARI"])
    # require finite ARI rows
    df = df[np.isfinite(ari)].copy()
    ari = _safe_float_array(df["ARI"])

    n = len(df)
    if n < 10:
        raise ValueError(f"Not enough valid rows ({n}) to compute utilities reliably.")

    k_top = max(1, int(top_k_ratio * n))
    k_bottom = max(1, int(bottom_k_ratio * n))

    # Precompute ARI top/bottom sets
    ari_top_idx = _topk_indices(ari, k_top)
    ari_bottom_idx = _bottomk_indices(ari, k_bottom)

    metric_bases = _extract_metric_basenames(df)

    rows = []
    for base in metric_bases:
        try:
            m, used_col = _get_metric_values(df, base, prefer_normalized=prefer_normalized)
        except ValueError:
            continue

        # Correlations (use only finite pairs)
        mask = np.isfinite(m) & np.isfinite(ari)
        if mask.sum() < max(10, int(0.2 * n)):
            continue

        m2 = m[mask]
        ari2 = ari[mask]

        sp, _ = spearmanr(m2, ari2)
        kt, _ = kendalltau(m2, ari2)

        # Keep the raw signed correlations for downstream inspection / debugging.
        spearman_signed = float(np.nan_to_num(sp))
        kendall_signed = float(np.nan_to_num(kt))

        # IMPORTANT: utilities must reward "higher metric -> higher ARI", i.e.,
        # POSITIVE correlation only. Normalised CVI columns are expected to obey
        # "higher = better"; a metric that strongly anti-correlates with ARI is
        # actually misleading the search and must NOT receive high utility.
        # Using max(0, .) (instead of abs) drops the rewarding effect of strongly
        # negative correlations while preserving the size of positive ones.
        spearman_pos = max(0.0, spearman_signed)
        kendall_pos = max(0.0, kendall_signed)

        # Focused diagnostics: Top-K / Bottom-K ranking agreement
        # (still useful for "did the metric at least nail the extremes?",
        # so we keep them as utility inputs at lower weight).
        m_top_idx = _topk_indices(m, k_top)
        m_bottom_idx = _bottomk_indices(m, k_bottom)

        top_overlap = _overlap_ratio(m_top_idx, ari_top_idx)
        bottom_overlap = _overlap_ratio(m_bottom_idx, ari_bottom_idx)

        # NDCG@K:
        # - Top NDCG: rank by metric high->low, relevance = ARI
        # - Bottom NDCG: rank by metric low->high, relevance = (1 - ARI)  (so "bad ARI" is relevant)
        top_ndcg = _ndcg_at_k_from_relevance(ari, np.argsort(-np.nan_to_num(m, nan=-np.inf)), k_top)
        bottom_ndcg = _ndcg_at_k_from_relevance(1.0 - np.clip(ari, 0.0, 1.0), np.argsort(np.nan_to_num(m, nan=np.inf)), k_bottom)

        # Full-ranking-ability terms:
        # - ndcg_all: NDCG over the entire list (global ordering quality).
        # - pairwise_accuracy: fraction of pairs ordered consistently with ARI
        #   (local-pair view; complements global rank correlation).
        ndcg_all = _ndcg_full_list(m, ari)
        pairwise_accuracy = _pairwise_accuracy(m, ari, max_pairs=pairwise_max_pairs)

        # Utility formula — interpretation of each block:
        #   spearman_pos / kendall_pos : global rank-correlation strength
        #                                (positive-only; anti-correlation is
        #                                 not rewarded).
        #   pairwise_accuracy          : local pair-ordering consistency.
        #   ndcg_all                   : full-list ranking quality.
        #   top/bottom_overlap & ndcg  : focused diagnostics for the extremes.
        # Anti-correlated metrics are blocked at the *_pos step; the other
        # terms are bounded in [0, 1] by construction (or 0.5 for a random
        # pairwise accuracy), so the formula stays well-behaved.
        utility = (
            w_spearman * spearman_pos +
            w_kendall * kendall_pos +
            w_pairwise * pairwise_accuracy +
            w_ndcg_all * ndcg_all +
            w_top_overlap * top_overlap +
            w_bottom_overlap * bottom_overlap +
            w_top_ndcg * top_ndcg +
            w_bottom_ndcg * bottom_ndcg
        )

        # Safety guard: every input above is already in [0, 1] (positive-clipped
        # correlations, ratios, NDCGs, pairwise accuracy). The clip + nan_to_num
        # below makes that invariant explicit and ensures no downstream code
        # sees NaN/inf, regardless of how the caller has retuned the weights.
        utility = float(np.nan_to_num(utility, nan=0.0, posinf=1.0, neginf=0.0))
        utility = float(np.clip(utility, 0.0, 1.0))

        rows.append({
            "metric": base,
            "used_column": used_col,
            "n_rows_used_for_corr": int(mask.sum()),
            # Signed values: useful for diagnosing anti-correlated metrics.
            "spearman_signed": spearman_signed,
            "kendall_signed": kendall_signed,
            # Positive-clipped values: the terms actually consumed by utility_score.
            "spearman_pos": spearman_pos,
            "kendall_pos": kendall_pos,
            # Legacy *_abs columns are kept for backward compatibility with any
            # downstream code that already reads them; they are NOT used by
            # utility_score anymore (which now uses *_pos).
            "spearman_abs": abs(spearman_signed),
            "kendall_abs": abs(kendall_signed),
            # New full-ranking-ability columns.
            "pairwise_accuracy": pairwise_accuracy,
            "ndcg_all": ndcg_all,
            # Focused diagnostics (unchanged).
            "topk_overlap": top_overlap,
            "bottomk_overlap": bottom_overlap,
            "topk_ndcg": top_ndcg,
            "bottomk_ndcg": bottom_ndcg,
            "utility_score": utility,
        })

    out = pd.DataFrame(rows).sort_values("utility_score", ascending=False)

    if output_path is None:
        base_dir = os.path.dirname(os.path.abspath(csv_path))
        output_path = os.path.join(base_dir, "metric_utilities.csv")

    out.to_csv(output_path, index=False)
    # ASCII-only print: the previous "✅" character crashed under Windows
    # consoles whose default code page (cp1252 / cp1255 / ...) cannot encode
    # non-BMP glyphs, taking the whole utility step with it.
    print(f"[OK] Saved utilities to: {output_path}")
    return out


if __name__ == "__main__":
    # Example usage:
    compute_metric_utilities(
        csv_path="models\\ClustOpt\\result_analysis\\2026-02-17_21-23-22_ampPartition_Optuna_2clusters\\results.csv",
        prefer_normalized=True,
        top_k_ratio=0.10,
        bottom_k_ratio=0.10,
    )
