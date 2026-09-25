"""
trainer.py
==========

Single-fold training loop with:
  * AdamW optimiser, ReduceLROnPlateau scheduler on validation combined loss,
  * gradient clipping,
  * early stopping on validation combined loss,
  * best / last checkpointing,
  * per-epoch train & validation logs (loss components + cheap val metrics).

The trainer is model-/loss-agnostic: it receives already-built ``model`` and
``loss_fn`` objects and the dataloaders for one fold.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .losses import CombinedUtilityLoss
from .metrics import compute_core_val_metrics


@dataclass
class TrainResult:
    best_epoch: int
    best_val_loss: float
    stopped_epoch: int
    train_log: List[Dict] = field(default_factory=list)
    val_log: List[Dict] = field(default_factory=list)
    best_state_dict: dict = field(default_factory=dict)


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: CombinedUtilityLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float,
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
    """Run one epoch. ``optimizer=None`` -> evaluation mode. Returns mean loss,
    mean components, and (only when evaluating) stacked preds/targets."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_n = 0
    comp_acc: Dict[str, float] = {"mse": 0.0, "pairwise": 0.0, "topk": 0.0}
    preds_buf: List[np.ndarray] = []
    targs_buf: List[np.ndarray] = []

    with torch.set_grad_enabled(training):
        for x, y, _ in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss, comps = loss_fn(pred, y)
            if training:
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            bs = x.shape[0]
            total_loss += float(loss.detach()) * bs
            total_n += bs
            for k in comp_acc:
                comp_acc[k] += comps[k] * bs
            if not training:
                preds_buf.append(pred.detach().cpu().numpy())
                targs_buf.append(y.detach().cpu().numpy())

    mean_loss = total_loss / max(1, total_n)
    mean_comps = {k: v / max(1, total_n) for k, v in comp_acc.items()}
    preds = np.concatenate(preds_buf) if preds_buf else np.empty((0, 0))
    targs = np.concatenate(targs_buf) if targs_buf else np.empty((0, 0))
    return mean_loss, mean_comps, preds, targs


def train_one_fold(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: CombinedUtilityLoss,
    cfg: TrainingConfig,
    device: torch.device,
    fold_dir: Path,
    log_every: int = 10,
    verbose: bool = True,
) -> TrainResult:
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
        min_lr=cfg.scheduler_min_lr,
    )

    best_val = float("inf")
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    train_log: List[Dict] = []
    val_log: List[Dict] = []
    stopped_epoch = cfg.epochs - 1

    fold_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.epochs):
        tr_loss, tr_comps, _, _ = _run_epoch(
            model, train_loader, loss_fn, device, optimizer, cfg.gradient_clip_norm
        )
        val_loss, val_comps, val_pred, val_true = _run_epoch(
            model, val_loader, loss_fn, device, None, cfg.gradient_clip_norm
        )
        val_metrics = compute_core_val_metrics(val_true, val_pred)
        scheduler.step(val_loss)
        cur_lr = optimizer.param_groups[0]["lr"]

        train_log.append(
            {"epoch": epoch, "train_loss": tr_loss, **{f"train_{k}": v for k, v in tr_comps.items()}, "lr": cur_lr}
        )
        val_log.append(
            {
                "epoch": epoch,
                "val_loss": val_loss,
                **{f"val_{k}": v for k, v in val_comps.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "lr": cur_lr,
            }
        )

        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and (epoch % log_every == 0 or improved or epoch == cfg.epochs - 1):
            print(
                f"    epoch {epoch:3d} | train {tr_loss:.5f} | val {val_loss:.5f} "
                f"| val_mae {val_metrics['mae']:.4f} | ndcg@10 {val_metrics['ndcg@10']:.4f} "
                f"| lr {cur_lr:.2e}{'  *best' if improved else ''}"
            )

        if epochs_no_improve >= cfg.early_stopping_patience:
            stopped_epoch = epoch
            if verbose:
                print(f"    early stopping at epoch {epoch} (best epoch {best_epoch}).")
            break

    # Persist checkpoints.
    torch.save({"model_state": model.state_dict(), "epoch": stopped_epoch}, fold_dir / "last_model.pt")
    torch.save({"model_state": best_state, "epoch": best_epoch}, fold_dir / "best_model.pt")

    # Restore best weights into the live model for downstream evaluation.
    model.load_state_dict(best_state)

    return TrainResult(
        best_epoch=best_epoch,
        best_val_loss=best_val,
        stopped_epoch=stopped_epoch,
        train_log=train_log,
        val_log=val_log,
        best_state_dict=best_state,
    )
