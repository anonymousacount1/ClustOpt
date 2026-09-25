"""Distribution tables and human-readable reports for split assignments."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def distribution_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Cross-tab of ``column`` values (rows) against split_id (columns)."""
    if column not in df.columns:
        return pd.DataFrame()
    table = pd.crosstab(df[column].astype(str), df["split_id"])
    table = table.reindex(sorted(table.columns), axis=1)
    table["total"] = table.sum(axis=1)
    return table.reset_index().rename(columns={column: column})


def full_strata_distribution(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("family", "subfamily", "difficulty", "cluster_count") if c in df.columns]
    if not cols:
        return pd.DataFrame()
    table = (
        df.groupby(cols + ["split_id"]).size().rename("count").reset_index()
        .pivot_table(index=cols, columns="split_id", values="count", fill_value=0)
    )
    table["total"] = table.sum(axis=1)
    return table.reset_index()


def split_summary_frame(df: pd.DataFrame) -> pd.DataFrame:
    sizes = df["split_id"].value_counts().sort_index()
    rows = []
    for split_id, count in sizes.items():
        sub = df[df["split_id"] == split_id]
        rows.append({
            "split_id": int(split_id),
            "n_datasets": int(count),
            "n_records_3views": int(count) * 3,
            "n_families": int(sub["family"].nunique()) if "family" in sub else 0,
            "n_subfamilies": int(sub["subfamily"].nunique()) if "subfamily" in sub else 0,
            "n_difficulties": int(sub["difficulty"].nunique()) if "difficulty" in sub else 0,
        })
    return pd.DataFrame(rows)


def build_distribution_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {
        "split_distribution_by_family": distribution_by(df, "family"),
        "split_distribution_by_subfamily": distribution_by(df, "subfamily"),
        "split_distribution_by_difficulty": distribution_by(df, "difficulty"),
        "split_distribution_by_cluster_count": distribution_by(df, "cluster_count"),
        "split_distribution_full_strata": full_strata_distribution(df),
    }


def render_summary_md(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    n_splits: int,
    seed: int,
    validation_report,
) -> str:
    lines = []
    lines.append("# Experiment Split Summary\n")
    lines.append(f"- Total datasets: **{len(df)}**")
    lines.append(f"- Number of splits: **{n_splits}**")
    lines.append(f"- Seed: **{seed}**")
    lines.append(f"- Records (3 views each): **{len(df) * 3}**")
    if not summary_df.empty:
        sizes = summary_df["n_datasets"]
        lines.append(f"- Split size min/max/mean: "
                     f"{int(sizes.min())} / {int(sizes.max())} / {sizes.mean():.1f}")
        lines.append(f"- Max size imbalance: **{int(sizes.max() - sizes.min())}**")
    lines.append("")

    lines.append("## Validation\n")
    lines.append(f"- Status: **{'OK' if validation_report.ok else 'FAILED'}**")
    if validation_report.errors:
        lines.append("- Errors:")
        for e in validation_report.errors:
            lines.append(f"  - {e}")
    if validation_report.warnings:
        lines.append("- Warnings:")
        for w in validation_report.warnings:
            lines.append(f"  - {w}")
    lines.append("")

    lines.append("## Datasets per split\n")
    lines.append("| split_id | n_datasets | n_records | families | subfamilies |")
    lines.append("| --- | --- | --- | --- | --- |")
    for _, r in summary_df.iterrows():
        lines.append(
            f"| {int(r['split_id'])} | {int(r['n_datasets'])} | "
            f"{int(r['n_records_3views'])} | {int(r['n_families'])} | "
            f"{int(r['n_subfamilies'])} |"
        )
    lines.append("")

    if "difficulty" in df.columns:
        lines.append("## Difficulty distribution (overall)\n")
        for val, cnt in df["difficulty"].astype(str).value_counts().items():
            lines.append(f"- {val}: {int(cnt)}")
        lines.append("")

    return "\n".join(lines) + "\n"
