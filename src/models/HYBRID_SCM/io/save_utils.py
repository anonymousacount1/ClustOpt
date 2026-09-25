from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.HYBRID_SCM.src.run_naming import build_run_name
from models.HYBRID_SCM.src.rasterizer import render_dataset


_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_WINDOWS_SAFE_PATH_BUDGET = 210


# =========================================================
# HELPERS
# =========================================================

def _timestamp() -> str:
    import datetime as _dt
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _slug(value: object, max_len: int = 48) -> str:
    s = _SAFE_RE.sub("_", str(value).strip())
    s = re.sub(r"_+", "_", s).strip("._-")
    if not s:
        s = "artifact"
    return s[:max_len]


def _fit_name_to_parent(parent: Path, name: str, budget: int = _WINDOWS_SAFE_PATH_BUDGET) -> str:
    """
    Trim the last path part so that parent/name stays comfortably below the
    conservative Windows path budget. This avoids MAX_PATH issues even when
    nested files and folders are later created under the run directory.
    """
    parent_len = len(str(parent))
    spare = max(24, budget - parent_len - 1)
    return name[:spare]


# =========================================================
# RUN DIRECTORY
# =========================================================

def ensure_run_dir(script_dir: Path, cfg=None, meta=None) -> Path:
    results_root = script_dir / "results_analysis"
    results_root.mkdir(parents=True, exist_ok=True)

    if cfg is not None and meta is not None:
        run_name = build_run_name(cfg, meta)
    else:
        run_name = _timestamp()

    run_name = _fit_name_to_parent(results_root, _slug(run_name, max_len=120))
    run_dir = results_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# =========================================================
# INTERNAL SAVER
# =========================================================

def save_artifacts(
    run_dir: Path,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    metadata: Dict[str, Any],
    replay_config: Dict[str, Any],
    tag: str,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    safe_tag = _slug(tag, max_len=36)
    df = pd.DataFrame(X, columns=feature_names)
    df["cluster_id"] = y.astype(int)

    # =========================
    # main dataset artifacts
    # =========================
    df.to_csv(run_dir / f"{safe_tag}_data.csv", index=False)

    (run_dir / f"{safe_tag}_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    (run_dir / f"{safe_tag}_replay_config.json").write_text(
        json.dumps(replay_config, indent=2),
        encoding="utf-8",
    )

    # =========================
    # pairplots
    # =========================
    pair_dir = run_dir / "pairplots"
    pair_dir.mkdir(parents=True, exist_ok=True)
    _save_pairplots(pair_dir, df, feature_names)

    # =========================
    # rendering outputs
    # =========================
    rendering_spec = metadata.get("effective_rendering_spec", {})
    if rendering_spec and rendering_spec.get("enabled", False):
        render_dir = run_dir / "rendering"
        render_dir.mkdir(parents=True, exist_ok=True)

        rendering = render_dataset(
            X=X,
            y=y,
            feature_names=feature_names,
            rendering_spec=rendering_spec,
        )

        grayscale = rendering.get("grayscale")
        binary_mask = rendering.get("binary_mask")
        cluster_masks = rendering.get("cluster_masks", {})
        summary = rendering.get("summary", {})

        if grayscale is not None:
            plt.imsave(render_dir / "grayscale.png", grayscale, cmap="gray")

        if binary_mask is not None:
            plt.imsave(render_dir / "binary_mask.png", binary_mask, cmap="gray")

        if cluster_masks:
            cm_dir = render_dir / "cluster_masks"
            cm_dir.mkdir(parents=True, exist_ok=True)
            for cid, mask in cluster_masks.items():
                plt.imsave(cm_dir / f"cluster_{cid}.png", mask, cmap="gray")

        (render_dir / "render_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )


# =========================================================
# PAIRPLOTS
# =========================================================

def _save_pairplots(pair_dir: Path, df: pd.DataFrame, feature_names: List[str]) -> None:
    cluster_ids = df["cluster_id"].to_numpy()
    unique_ids = np.unique(cluster_ids)

    id_to_idx = {cid: i for i, cid in enumerate(unique_ids)}
    color_idx = np.array([id_to_idx[c] for c in cluster_ids], dtype=int)

    add_legend = len(unique_ids) <= 12

    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            xi = feature_names[i]
            xj = feature_names[j]

            fig = plt.figure()
            sc = plt.scatter(
                df[xi].values,
                df[xj].values,
                c=color_idx,
                s=10,
                alpha=0.75,
            )

            plt.xlabel(xi)
            plt.ylabel(xj)
            plt.title(f"{xi} vs {xj} (colored by cluster_id)")

            if add_legend:
                handles = []
                labels = []
                cmap = sc.cmap
                norm = sc.norm

                for cid in unique_ids:
                    idx = id_to_idx[cid]
                    handles.append(
                        plt.Line2D(
                            [0],
                            [0],
                            marker="o",
                            linestyle="",
                            markersize=6,
                            markerfacecolor=cmap(norm(idx)),
                            markeredgecolor="none",
                        )
                    )
                    labels.append(f"cluster {cid}")

                plt.legend(handles, labels, loc="best", frameon=True)

            xi_safe = _slug(xi, max_len=16)
            xj_safe = _slug(xj, max_len=16)
            fig.savefig(
                pair_dir / f"{i:02d}_{j:02d}_{xi_safe}_vs_{xj_safe}.png",
                dpi=160,
                bbox_inches="tight",
            )
            plt.close(fig)

# =========================
# Stage 1-4 extensions
# =========================
_old_save_artifacts = save_artifacts


def save_artifacts(  # type: ignore[override]
    run_dir: Path,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    metadata: Dict[str, Any],
    replay_config: Dict[str, Any],
    tag: str,
) -> None:
    _old_save_artifacts(run_dir, X, y, feature_names, metadata, replay_config, tag)

    safe_tag = _slug(tag, max_len=36)
    rendering_spec = metadata.get("effective_rendering_spec", {})
    if rendering_spec and rendering_spec.get("enabled", False):
        render_dir = run_dir / "rendering"
        render_dir.mkdir(parents=True, exist_ok=True)
        rendering = render_dataset(X=X, y=y, feature_names=feature_names, rendering_spec=rendering_spec)
        extra_map = {
            "distance_transform": ("distance_transform.png", "viridis"),
            "skeleton_mask": ("skeleton_mask.png", "gray"),
            "edge_map": ("edge_map.png", "gray"),
            "gradient_orientation_map": ("gradient_orientation_map.png", "twilight"),
            "local_thickness_map": ("local_thickness_map.png", "magma"),
        }
        for key, (fname, cmap) in extra_map.items():
            arr = rendering.get(key)
            if arr is not None:
                plt.imsave(render_dir / fname, arr, cmap=cmap)
        cid = rendering.get("cluster_id_raster")
        if cid is not None:
            np.save(render_dir / "cluster_id_raster.npy", cid)

    structural = metadata.get("structural_ground_truth")
    if structural:
        (run_dir / f"{safe_tag}_structural_ground_truth.json").write_text(json.dumps(structural, indent=2), encoding="utf-8")
