"""
losses.py
=========

Loss functions for utility-vector regression with ranking awareness.

Implemented:
  * :class:`WeightedMSELoss`     - elementwise MSE with optional weights.
  * :class:`PairwiseRankingLoss` - margin ranking over sampled metric pairs.
  * :class:`TopKWeightedMSE`     - MSE up-weighting the true top-K metrics.
  * :class:`CombinedUtilityLoss` - 0.60*MSE + 0.25*Pairwise + 0.15*TopK.

The combined loss also supports an ``mse_only`` ablation mode.  All components
operate on a batch of shape ``[B, M]`` (B samples, M metrics).
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LossConfig


class WeightedMSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        se = (pred - target) ** 2
        if weights is not None:
            return (se * weights).sum() / weights.sum().clamp_min(1e-12)
        return se.mean()


class PairwiseRankingLoss(nn.Module):
    """Margin ranking loss over a sampled subset of metric pairs per sample.

    For each sampled pair (i, j) with ``target_i - target_j`` larger than
    ``tie_epsilon`` in magnitude, the model is encouraged to order the
    predictions the same way with a margin.  Pairs that are (near-)ties in the
    target are ignored.
    """

    def __init__(
        self,
        margin: float = 0.05,
        max_pairs_per_sample: int = 256,
        tie_epsilon: float = 1e-4,
    ) -> None:
        super().__init__()
        self.margin = margin
        self.max_pairs = max_pairs_per_sample
        self.tie_epsilon = tie_epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b, m = pred.shape
        if m < 2:
            return pred.sum() * 0.0
        device = pred.device
        n_pairs = min(self.max_pairs, m * (m - 1) // 2)
        # Sample i, j index pairs (with i != j) shared across the batch for speed.
        i_idx = torch.randint(0, m, (n_pairs,), device=device)
        j_idx = torch.randint(0, m, (n_pairs,), device=device)
        valid = i_idx != j_idx
        i_idx, j_idx = i_idx[valid], j_idx[valid]
        if i_idx.numel() == 0:
            return pred.sum() * 0.0

        t_i = target[:, i_idx]
        t_j = target[:, j_idx]
        p_i = pred[:, i_idx]
        p_j = pred[:, j_idx]

        diff = t_i - t_j
        mask = diff.abs() > self.tie_epsilon
        if not mask.any():
            return pred.sum() * 0.0

        # sign target: +1 if t_i > t_j else -1
        y = torch.sign(diff)
        # margin ranking loss: max(0, -y*(p_i - p_j) + margin)
        loss = F.relu(-y * (p_i - p_j) + self.margin)
        loss = loss[mask]
        return loss.mean()


class TopKWeightedMSE(nn.Module):
    """MSE where metrics in each sample's true top-K receive a higher weight."""

    def __init__(self, k: int = 10, extra_weight: float = 2.0) -> None:
        super().__init__()
        self.k = k
        self.extra_weight = extra_weight
        self.mse = WeightedMSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b, m = target.shape
        k = min(self.k, m)
        with torch.no_grad():
            topk_idx = torch.topk(target, k=k, dim=1).indices
            is_top = torch.zeros_like(target)
            is_top.scatter_(1, topk_idx, 1.0)
            weights = 1.0 + self.extra_weight * is_top
        return self.mse(pred, target, weights)


class CombinedUtilityLoss(nn.Module):
    """Weighted sum of MSE + pairwise ranking + top-K weighted MSE.

    ``forward`` returns ``(total_loss, components_dict)`` where the dict holds
    detached scalar floats for each component for logging.
    """

    def __init__(self, cfg: LossConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.mse = WeightedMSELoss()
        self.pairwise = PairwiseRankingLoss(
            margin=cfg.pairwise_margin,
            max_pairs_per_sample=cfg.max_pairs_per_sample,
            tie_epsilon=cfg.tie_epsilon,
        )
        self.topk = TopKWeightedMSE(k=cfg.topk, extra_weight=cfg.topk_extra_weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        mse_val = self.mse(pred, target)
        if self.cfg.mode == "mse_only":
            return mse_val, {"mse": float(mse_val.detach()), "pairwise": 0.0, "topk": 0.0}

        pair_val = self.pairwise(pred, target)
        topk_val = self.topk(pred, target)
        total = (
            self.cfg.mse_weight * mse_val
            + self.cfg.pairwise_rank_weight * pair_val
            + self.cfg.topk_mse_weight * topk_val
        )
        comps: Dict[str, float] = {
            "mse": float(mse_val.detach()),
            "pairwise": float(pair_val.detach()),
            "topk": float(topk_val.detach()),
        }
        return total, comps


def build_loss(cfg: LossConfig) -> CombinedUtilityLoss:
    return CombinedUtilityLoss(cfg)
