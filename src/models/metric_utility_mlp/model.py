"""
model.py
========

MLP models that map 250 dataset meta-features to 60 metric-utility scores in
``[0, 1]``.

  * :class:`SharedTrunk` - the common feature extractor.
  * :class:`SingleHeadMetricUtilityMLP` - trunk -> Linear(256, 60) -> Sigmoid.
  * :class:`MultiHeadMetricUtilityMLP` - trunk -> per-metric-family heads,
    outputs scattered back into the canonical target order, then Sigmoid.
  * :func:`build_model` - factory driven by :class:`ModelConfig`.

The multi-head model is given the head mapping (``head_name -> target indices``)
produced by :mod:`head_groups`, so head outputs are always reassembled in the
exact ``target_columns`` order regardless of how metrics are grouped.
"""

from __future__ import annotations

import io
from collections import OrderedDict
from typing import Dict, List

import torch
import torch.nn as nn

from .config import ModelConfig

_ACTIVATIONS = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "elu": nn.ELU,
}


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unsupported activation '{name}'. Options: {list(_ACTIVATIONS)}")
    return _ACTIVATIONS[name]()


class SharedTrunk(nn.Module):
    """Stack of Linear -> [BatchNorm] -> Activation -> Dropout blocks.

    BatchNorm is applied to all but the final hidden layer (matching the plan).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        dims = [cfg.input_dim] + list(cfg.hidden_dims)
        dropouts = list(cfg.dropout)
        if len(dropouts) != len(cfg.hidden_dims):
            raise ValueError("dropout list length must match hidden_dims length.")

        layers: List[nn.Module] = []
        n_hidden = len(cfg.hidden_dims)
        for i in range(n_hidden):
            in_dim, out_dim = dims[i], dims[i + 1]
            layers.append(nn.Linear(in_dim, out_dim))
            # No BatchNorm on the last hidden layer (per architecture spec).
            if cfg.batch_norm and i < n_hidden - 1:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(_make_activation(cfg.activation))
            if dropouts[i] > 0:
                layers.append(nn.Dropout(dropouts[i]))
        self.net = nn.Sequential(*layers)
        self.output_dim = dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SingleHeadMetricUtilityMLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.trunk = SharedTrunk(cfg)
        self.head = nn.Linear(self.trunk.output_dim, cfg.output_dim)
        self.output_dim = cfg.output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.trunk(x)
        return torch.sigmoid(self.head(z))


class _MetricHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MultiHeadMetricUtilityMLP(nn.Module):
    """Shared trunk + one head per metric family, reassembled to target order."""

    def __init__(self, cfg: ModelConfig, head_mapping: Dict[str, List[int]]) -> None:
        super().__init__()
        if not head_mapping:
            raise ValueError("head_mapping must be a non-empty {head: [indices]} dict.")
        self.trunk = SharedTrunk(cfg)
        self.output_dim = cfg.output_dim

        # Register heads and the (buffer) target indices each head writes to.
        self.heads = nn.ModuleDict()
        self._head_names: List[str] = []
        for head_name, indices in head_mapping.items():
            self.heads[head_name] = _MetricHead(
                self.trunk.output_dim, cfg.head_hidden_dim, len(indices), cfg.head_dropout
            )
            self.register_buffer(
                f"_idx_{head_name}",
                torch.as_tensor(indices, dtype=torch.long),
                persistent=False,
            )
            self._head_names.append(head_name)

        total = sum(len(v) for v in head_mapping.values())
        if total != cfg.output_dim:
            raise ValueError(
                f"Head mapping covers {total} targets but output_dim={cfg.output_dim}."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.trunk(x)
        out = torch.zeros(x.shape[0], self.output_dim, device=x.device, dtype=z.dtype)
        for head_name in self._head_names:
            idx = getattr(self, f"_idx_{head_name}")
            out.index_copy_(1, idx, self.heads[head_name](z))
        return torch.sigmoid(out)


def build_model(cfg: ModelConfig, head_mapping: Dict[str, List[int]] | None = None) -> nn.Module:
    if cfg.type == "single_head_mlp":
        return SingleHeadMetricUtilityMLP(cfg)
    if cfg.type == "multi_head_mlp":
        if head_mapping is None:
            raise ValueError("multi_head_mlp requires a head_mapping.")
        return MultiHeadMetricUtilityMLP(cfg, head_mapping)
    raise ValueError(f"Unknown model.type '{cfg.type}'.")


def architecture_summary(model: nn.Module) -> str:
    """Human-readable architecture string incl. parameter counts."""
    buf = io.StringIO()
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buf.write(f"Model class: {model.__class__.__name__}\n")
    buf.write(f"Total parameters: {n_params:,}\n")
    buf.write(f"Trainable parameters: {n_train:,}\n\n")
    buf.write(repr(model))
    buf.write("\n")
    return buf.getvalue()
