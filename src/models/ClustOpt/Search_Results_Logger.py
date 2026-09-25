"""Logging and reporting layer for ClustOpt search runs.

The logger collects one row per evaluated configuration and exposes:

* :meth:`log_result` — record a single (config, scores, labels) triple,
* :meth:`to_dataframe` / :meth:`to_csv` — export the collected log,
* :meth:`generate_summary_report` — human-readable textual summary,
* a small set of matplotlib plots,
* :meth:`make_full_report` — single call that wires CSV + summary + plots
  for a given dataset into a timestamped directory.

Public API stays compatible with the previous implementation; the changes
are internal hardening (None-safety, robust path handling, structured
logging instead of emoji-printed messages, etc.).
"""
from __future__ import annotations

import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


logger = logging.getLogger(__name__)


_DEFAULT_REPORT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "result_analysis")
)


def _metric_name(m: Any) -> str:
    """Return ``m.name`` if present, otherwise ``str(m)``."""
    return m.name if hasattr(m, "name") else str(m)


def _safe_finite(value: Any) -> Optional[float]:
    """Return a float if the value is finite, otherwise None."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


class SearchResultsLogger:
    """Collects and exports clustering search results."""

    def __init__(
        self,
        metrics: Iterable[Any],
        weights: Dict[str, float],
        search_algorithm_name: str = "",
        search_space: Optional[Dict[str, Any]] = None,
        search_algorithm_params: Optional[Dict[str, Any]] = None,
    ):
        self.metrics: List[str] = [_metric_name(m) for m in metrics]
        self.weights: Dict[str, float] = dict(weights or {})
        self.rows: List[Dict[str, Any]] = []
        self.search_algorithm_name = search_algorithm_name
        self.full_search_space: Optional[Dict[str, Any]] = search_space
        self.search_algorithm_params: Dict[str, Any] = dict(search_algorithm_params or {})

    # ---------------------------------------------------------------- logging

    def log_result(
        self,
        algorithm: str,
        config: Dict[str, Any],
        raw_scores: Dict[str, float],
        normalized_scores: Dict[str, float],
        aggregate_score: float,
        labels: Optional[Sequence[Any]],
        exception: Optional[str] = None,
        extra_fields: Optional[Dict[str, Any]] = None,
    ) -> None:
        labels_arr = None if labels is None else np.asarray(labels)
        validity = self._label_stats(labels_arr)

        row: Dict[str, Any] = {
            "algorithm": algorithm,
            "config_str": str(config),
            "aggregate_score": aggregate_score,
            "valid": validity["valid"],
            "n_clusters_wo_noise": validity["n_clusters_wo_noise"],
            "noise_ratio": validity["noise_ratio"],
            "n_labels_unique": validity["n_labels_unique"],
            "exception": exception or "",
        }
        if extra_fields:
            row.update(extra_fields)

        for metric in self.metrics:
            w = self.weights.get(metric, 1.0)
            row[f"{metric} (w={w})"] = raw_scores.get(metric)
            row[f"{metric}_norm"] = normalized_scores.get(metric)

        self.rows.append(row)

    @staticmethod
    def _label_stats(labels: Optional[np.ndarray]) -> Dict[str, Any]:
        if labels is None or labels.size == 0:
            return {
                "valid": False,
                "n_clusters_wo_noise": 0,
                "noise_ratio": None if labels is None else 0.0,
                "n_labels_unique": 0,
            }

        unique = set(labels.tolist())
        noise_ratio = float(np.mean(labels == -1))
        unique_wo_noise = {u for u in unique if u != -1}
        return {
            "valid": len(unique_wo_noise) > 1,
            "n_clusters_wo_noise": len(unique_wo_noise),
            "noise_ratio": noise_ratio,
            "n_labels_unique": len(unique),
        }

    # ---------------------------------------------------------------- exports

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def to_csv(self, filepath: str) -> str:
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
        return filepath

    def set_search_space(self, search_space: Dict[str, Any]) -> None:
        self.full_search_space = search_space

    # ----------------------------------------------------------- text reports

    def _format_search_space_hierarchical(self) -> str:
        if not self.full_search_space:
            return "  (no search-space metadata recorded)"

        grouped: Dict[str, List[tuple]] = defaultdict(list)
        for key, val in self.full_search_space.items():
            if key == "algorithm":
                grouped["General"].append((key, val))
            elif "__" in key:
                algo, param = key.split("__", 1)
                grouped[algo].append((param, val))
            else:
                grouped["Other"].append((key, val))

        lines: List[str] = []
        for group, items in grouped.items():
            lines.append(f"  {group}:")
            for param, val in items:
                lines.append(f"    - {param}: {val}")
        return "\n".join(lines)

    def generate_summary_report(self, filepath: str) -> str:
        df = self.to_dataframe()
        if df.empty:
            with open(filepath, "w", encoding="utf-8") as fh:
                fh.write("No results to report.\n")
            return filepath

        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write("=== Clustering Search Summary Report ===\n\n")

            fh.write(">> Search Space:\n")
            fh.write(self._format_search_space_hierarchical() + "\n\n")

            fh.write(f">> Search Algorithm Used: {self.search_algorithm_name or 'Unknown'}\n\n")

            if self.search_algorithm_params:
                fh.write(">> Search Algorithm Parameters:\n")
                for param, val in self.search_algorithm_params.items():
                    fh.write(f"  - {param}: {val}\n")
                fh.write("\n")

            fh.write(">> CVI Metrics Evaluated:\n")
            for metric in self.metrics:
                weight = self.weights.get(metric, 1.0)
                fh.write(f"  - {metric} (weight = {weight})\n")
            fh.write("\n")

            fh.write(">> Best Configuration per Clustering Algorithm:\n")
            for algo in df["algorithm"].unique():
                best = (
                    df[df["algorithm"] == algo]
                    .sort_values("aggregate_score", ascending=False)
                    .iloc[0]
                )
                fh.write(f"\nAlgorithm: {algo}\n")
                fh.write(f"  Config: {best['config_str']}\n")
                fh.write(f"  Aggregate Score: {best['aggregate_score']:.4f}\n")
                for metric in self.metrics:
                    weight = self.weights.get(metric, 1.0)
                    fh.write(
                        f"  {metric} (w={weight}): {best.get(f'{metric} (w={weight})')} | "
                        f"Normalized: {best.get(f'{metric}_norm')}\n"
                    )

            best_overall = df.sort_values("aggregate_score", ascending=False).iloc[0]
            fh.write("\n>> Best Overall Configuration:\n")
            fh.write(f"  Algorithm: {best_overall['algorithm']}\n")
            fh.write(f"  Config: {best_overall['config_str']}\n")
            fh.write(f"  Aggregate Score: {best_overall['aggregate_score']:.4f}\n")
            for metric in self.metrics:
                weight = self.weights.get(metric, 1.0)
                fh.write(
                    f"  {metric} (w={weight}): {best_overall.get(f'{metric} (w={weight})')} | "
                    f"Normalized: {best_overall.get(f'{metric}_norm')}\n"
                )

            fh.write(
                "\n(Full details of all configurations are available in the corresponding CSV results file.)\n"
            )
        return filepath

    # ---------------------------------------------------------------- charts

    def generate_models_comparison_bar_plot(
        self, df: pd.DataFrame, metrics: List[str], save_path: str
    ) -> None:
        if df.empty:
            logger.warning("Cannot generate models comparison plot: empty dataframe.")
            return

        best_per_algo = df.sort_values("aggregate_score", ascending=False).groupby("algorithm").head(1)
        algo_names = best_per_algo["algorithm"].tolist()
        metric_data: Dict[str, List[Optional[float]]] = {metric: [] for metric in metrics}
        aggregate_scores: List[Optional[float]] = []

        for _, row in best_per_algo.iterrows():
            for metric in metrics:
                metric_data[metric].append(_safe_finite(row.get(f"{metric}_norm")))
            aggregate_scores.append(_safe_finite(row.get("aggregate_score")))

        x = np.arange(len(algo_names))
        width = 0.12
        fig, ax = plt.subplots(figsize=(14, 6))
        colors = plt.get_cmap("tab10")

        total_groups = len(metrics) + 1
        max_val = 0.0

        for i, metric in enumerate(metrics):
            offsets = x + (i - total_groups / 2) * width + width / 2
            heights = [v if v is not None else 0.0 for v in metric_data[metric]]
            bars = ax.bar(offsets, heights, width, label=f"{metric} (norm)", color=colors(i))
            for bar, raw in zip(bars, metric_data[metric]):
                if raw is None:
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2, raw + 0.01,
                    f"{raw:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
                )
                max_val = max(max_val, raw)

        offsets = x + (len(metrics) - total_groups / 2) * width + width / 2
        agg_heights = [v if v is not None else 0.0 for v in aggregate_scores]
        bars = ax.bar(offsets, agg_heights, width, label="Aggregate Score", color=colors(len(metrics)))
        for bar, raw in zip(bars, aggregate_scores):
            if raw is None:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2, raw + 0.01,
                f"{raw:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
            )
            max_val = max(max_val, raw)

        ax.set_ylabel("Normalized Score", fontsize=12)
        ax.set_title(
            "Normalized Metric Scores and Aggregate Score per Algorithm",
            fontsize=13, fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(algo_names, fontsize=11)
        ax.legend(
            title="Metrics", fontsize=10, title_fontsize=11,
            loc="upper left", bbox_to_anchor=(1.02, 1),
        )
        ax.set_ylim([0, min(max_val + 0.15, 1.25)])
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        plt.tight_layout(rect=[0, 0, 0.96, 1])
        plt.savefig(save_path)
        plt.close()
        logger.info("Comparison bar plot saved to: %s", save_path)

    def generate_metric_comparison_plot(
        self, df: pd.DataFrame, metrics: List[str], save_path: str
    ) -> None:
        if df.empty:
            logger.warning("Cannot generate metric comparison plot: empty dataframe.")
            return

        best_per_algo = df.sort_values("aggregate_score", ascending=False).groupby("algorithm").head(1)
        algos = best_per_algo["algorithm"].tolist()

        extended_metrics = metrics + ["aggregate_score"]
        n_metrics = len(extended_metrics)
        n_algos = len(algos)
        if n_algos == 0:
            logger.warning("Cannot generate metric comparison plot: no algorithms in log.")
            return

        group_spacing = 3.0
        group_width = 2.0
        bar_width = group_width / (n_algos * 1.8)
        x_centers = np.arange(n_metrics) * group_spacing

        fig, ax = plt.subplots(figsize=(22, 6))
        colors = plt.get_cmap("tab10")
        y_label_offset = 0.03
        y_limit = 1.6

        for algo_idx, algo in enumerate(algos):
            row = best_per_algo[best_per_algo["algorithm"] == algo].iloc[0]
            values = [_safe_finite(row.get(f"{metric}_norm", row.get(metric))) for metric in metrics]
            values.append(_safe_finite(row.get("aggregate_score")))
            heights = [v if v is not None else 0.0 for v in values]

            offset = -group_width / 2 + (algo_idx + 0.5) * (group_width / n_algos)
            bar_positions = x_centers + offset
            ax.bar(bar_positions, heights, width=bar_width, color=colors(algo_idx), label=algo)

            for xpos, raw in zip(bar_positions, values):
                if raw is None:
                    continue
                ax.text(
                    xpos, raw + y_label_offset, f"{raw:.2f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold", color="black",
                )

        ax.set_xticks(x_centers)
        ax.set_xticklabels(extended_metrics, fontsize=11, rotation=30, ha="right")
        ax.set_ylabel("Normalized Score", fontsize=12)
        ax.set_title(
            "Normalized CVI Metrics and Aggregate Score per Algorithm",
            fontsize=13, fontweight="bold",
        )
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_ylim([0, y_limit])

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(
            by_label.values(), by_label.keys(),
            title="Algorithm", fontsize=10, title_fontsize=11,
            loc="upper left", bbox_to_anchor=(1.02, 1),
        )

        plt.tight_layout(rect=[0, 0, 0.96, 1])
        plt.savefig(save_path)
        plt.close()
        logger.info("Clean metric comparison plot saved to: %s", save_path)

    def generate_single_metric_plot(
        self, df: pd.DataFrame, metric_name: str, save_path: str
    ) -> None:
        """Bar plot of the raw (non-normalised) metric across best-per-algo rows."""
        raw_metric_col = next(
            (col for col in df.columns if col.startswith(metric_name + " (w=")),
            None,
        )
        if raw_metric_col is None:
            raise ValueError(f"Metric column for '{metric_name}' not found in DataFrame.")

        best_per_algo = df.sort_values("aggregate_score", ascending=False).groupby("algorithm").head(1)
        algos = best_per_algo["algorithm"].tolist()
        values = [
            best_per_algo.loc[best_per_algo["algorithm"] == algo, raw_metric_col].values[0]
            for algo in algos
        ]

        fig, ax = plt.subplots(figsize=(9, 5))
        bar_width = 0.4
        bars = ax.bar(algos, values, width=bar_width, color="steelblue", edgecolor="black")

        ax.set_ylabel(f"{metric_name} (raw)", fontsize=12)
        ax.set_title(f"{metric_name.capitalize()} Comparison Across Algorithms", fontsize=14, fontweight="bold")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        finite_values = [v for v in values if _safe_finite(v) is not None]
        top_padding = 0.01 * max(finite_values) if finite_values else 0.01
        for bar, val in zip(bars, values):
            if _safe_finite(val) is None:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2, val + top_padding,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info("Single metric plot saved to: %s", save_path)

    # ----------------------------------------------------------- orchestration

    def make_full_report(
        self,
        dataset_name: str,
        data: Any,
        model_dict: Dict[str, Any],
        plot_fn: Any,
        output_root: Optional[str] = None,
    ) -> str:
        """Materialise CSV + summary + per-metric plots inside a timestamped folder.

        ``output_root`` lets callers (notably tests / experiments) redirect
        the destination away from the package source tree; if ``None`` the
        previous default (``<package>/result_analysis``) is kept.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_dir_name = f"{timestamp}_{dataset_name}"
        base_dir = output_root or _DEFAULT_REPORT_ROOT
        os.makedirs(base_dir, exist_ok=True)

        full_dir = os.path.join(base_dir, report_dir_name)
        os.makedirs(full_dir, exist_ok=True)

        csv_path = os.path.join(full_dir, "results.csv")
        txt_path = os.path.join(full_dir, "summary.txt")
        self.to_csv(csv_path)
        self.generate_summary_report(txt_path)

        graph_dir = os.path.join(full_dir, "graphs")
        metrics_dir = os.path.join(graph_dir, "metrics")
        data_predict_dir = os.path.join(graph_dir, "data_predictions")
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(data_predict_dir, exist_ok=True)

        df = self.to_dataframe()
        self.generate_models_comparison_bar_plot(
            df, self.metrics, os.path.join(metrics_dir, "models_comparison_plot.png")
        )
        self.generate_metric_comparison_plot(
            df, self.metrics, os.path.join(metrics_dir, "metrics_comparison_plot.png")
        )

        for metric in self.metrics:
            try:
                self.generate_single_metric_plot(
                    df, metric, os.path.join(metrics_dir, f"{metric}_comparison_plot.png")
                )
            except Exception as exc:
                logger.warning("Failed to plot metric '%s': %s", metric, exc)

        for name, model in (model_dict or {}).items():
            save_path = os.path.join(data_predict_dir, f"{name}.png")
            try:
                plot_fn(data, model=model, title=f"Clusters - {name}", save_path=save_path)
                logger.info("Saved plot for %s to: %s", name, save_path)
            except Exception as exc:
                logger.warning("Failed to generate plot for %s: %s", name, exc)

        logger.info("Saved full report to: %s", full_dir)
        return full_dir
