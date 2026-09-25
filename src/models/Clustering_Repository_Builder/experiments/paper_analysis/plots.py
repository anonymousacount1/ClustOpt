"""Reusable plotting layer for the paper analysis.

Design rules (§20), enforced by construction rather than by convention:

* **No hardcoded method labels.** Every axis label comes from the registry's
  ``short_label``; callers pass frames, never label lists.
* **Every figure ships its data.** :func:`save_figure` writes ``<name>.png``,
  ``<name>.svg`` and ``<name>_data.csv`` together, so a figure can always be
  re-plotted or audited.
* **Deterministic ordering.** Categories are ordered by an explicit key (value,
  or a registry order), never by dict/groupby accident.
* **Never colour-only.** Group distinctions also use hatch patterns or markers.
* **Selection provenance in the caption.** ``PlotSpec.caption`` is written into
  the figure and the sidecar metadata, so an oracle/retrospective figure cannot
  be mistaken for a fixed-method one.
* **Importance levels.** Each figure declares ``paper_main`` /
  ``paper_supplement`` / ``appendix`` / ``exploratory``; the CLI can restrict
  generation so a run does not emit hundreds of unreadable plots.

Matplotlib is imported with the ``Agg`` backend, so this is safe headless.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.figure import Figure                               # noqa: E402

from . import paths

IMPORTANCE_LEVELS: Tuple[str, ...] = (
    "paper_main", "paper_supplement", "appendix", "exploratory",
)
_IMPORTANCE_RANK = {lvl: i for i, lvl in enumerate(IMPORTANCE_LEVELS)}

# Colour-blind-safe qualitative palette (Okabe-Ito), plus hatches so no meaning
# is carried by colour alone.
PALETTE: Tuple[str, ...] = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9",
    "#F0E442", "#000000",
)
HATCHES: Tuple[str, ...] = ("", "///", "...", "xxx", "\\\\\\", "ooo", "***", "+++")
MARKERS: Tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*")

# Stable colour/hatch per utility source so a source keeps its identity across
# every figure in the paper.
SOURCE_STYLE: Dict[str, Tuple[str, str]] = {
    "real": (PALETTE[2], "..."),
    "mlp": (PALETTE[0], ""),
    "knn": (PALETTE[1], "///"),
    "uniform": (PALETTE[4], "xxx"),
    "single_cvi": (PALETTE[5], "\\\\\\"),
    "external_internal": (PALETTE[3], "ooo"),
}

DEFAULT_RC: Dict[str, Any] = {
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": ":",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.autolayout": False,
}


@dataclass
class PlotSpec:
    """Everything needed to emit one figure and register it in the manifest."""

    name: str                      # basename without extension
    title: str
    ylabel: str = ""
    xlabel: str = ""
    caption: str = ""
    importance: str = "paper_supplement"
    source_tables: Tuple[str, ...] = ()
    selection_type: str = ""
    notes: str = ""

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name, "title": self.title, "caption": self.caption,
            "importance": self.importance,
            "source_tables": list(self.source_tables),
            "selection_type": self.selection_type, "notes": self.notes,
        }


@dataclass
class PlotResult:
    spec: PlotSpec
    png: Optional[Path] = None
    svg: Optional[Path] = None
    data_csv: Optional[Path] = None
    skipped_reason: str = ""

    @property
    def created(self) -> bool:
        return self.png is not None


@dataclass
class PlotContext:
    """Output directory + generation policy shared by a whole comparison suite."""

    output_dir: Path
    min_importance: str = "exploratory"
    enabled: bool = True
    also_svg: bool = True
    results: List[PlotResult] = field(default_factory=list)

    def allows(self, importance: str) -> bool:
        if not self.enabled:
            return False
        return (_IMPORTANCE_RANK.get(importance, 99)
                <= _IMPORTANCE_RANK.get(self.min_importance, 99))

    def record(self, result: PlotResult) -> PlotResult:
        self.results.append(result)
        return result

    def created_frame(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for r in self.results:
            rows.append({
                "name": r.spec.name, "title": r.spec.title,
                "importance": r.spec.importance,
                "png": str(r.png) if r.png else "",
                "svg": str(r.svg) if r.svg else "",
                "data_csv": str(r.data_csv) if r.data_csv else "",
                "caption": r.spec.caption,
                "selection_type": r.spec.selection_type,
                "source_tables": ";".join(r.spec.source_tables),
                "created": r.created, "skipped_reason": r.skipped_reason,
            })
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figure plumbing
# --------------------------------------------------------------------------- #
def save_figure(fig: Figure, spec: PlotSpec, ctx: PlotContext,
                data: Optional[pd.DataFrame] = None) -> PlotResult:
    """Write PNG (+SVG) and the plotted data, then register the result."""
    if spec.caption:
        fig.text(0.01, -0.02, _wrap(spec.caption, 150), ha="left", va="top",
                 fontsize=7, style="italic", color="#333333", wrap=True)
    paths.mkdirs(ctx.output_dir)
    png = ctx.output_dir / f"{spec.name}.png"
    fig.savefig(paths.ext(png), format="png", bbox_inches="tight")
    svg = None
    if ctx.also_svg:
        svg = ctx.output_dir / f"{spec.name}.svg"
        try:
            fig.savefig(paths.ext(svg), format="svg", bbox_inches="tight")
        except Exception:                                     # pragma: no cover
            svg = None
    plt.close(fig)
    data_csv = None
    if data is not None and not data.empty:
        data_csv = ctx.output_dir / f"{spec.name}_data.csv"
        paths.write_csv(data_csv, data, index=False)
    return ctx.record(PlotResult(spec, png=png, svg=svg, data_csv=data_csv))


def _skip(spec: PlotSpec, ctx: PlotContext, reason: str) -> PlotResult:
    return ctx.record(PlotResult(spec, skipped_reason=reason))


def _wrap(text: str, width: int) -> str:
    words, lines, cur = str(text).split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def _fig(width: float, height: float) -> Tuple[Figure, Any]:
    with plt.rc_context(DEFAULT_RC):
        fig, ax = plt.subplots(figsize=(width, height))
    return fig, ax


def _style_for(row: pd.Series) -> Tuple[str, str]:
    src = str(row.get("utility_source", "") or "")
    return SOURCE_STYLE.get(src, (PALETTE[0], ""))


def _bar_width(n: int) -> float:
    return max(4.5, min(16.0, 0.42 * n + 2.0))


# --------------------------------------------------------------------------- #
# Bar charts
# --------------------------------------------------------------------------- #
def bar_chart(summary: pd.DataFrame, spec: PlotSpec, ctx: PlotContext, *,
              value_col: str, label_col: str = "method_short_label",
              error_low_col: Optional[str] = None,
              error_high_col: Optional[str] = None,
              sort: str = "value_desc", annotate: bool = True,
              horizontal: bool = False, reference_value: Optional[float] = None,
              reference_label: str = "") -> PlotResult:
    """Deterministically ordered bar chart with optional asymmetric error bars."""
    if not ctx.allows(spec.importance):
        return _skip(spec, ctx, f"importance below {ctx.min_importance}")
    if summary.empty or value_col not in summary.columns:
        return _skip(spec, ctx, f"missing column {value_col}")
    data = summary.dropna(subset=[value_col]).copy()
    if data.empty:
        return _skip(spec, ctx, "no finite values")
    if sort == "value_desc":
        data = data.sort_values(value_col, ascending=False, kind="mergesort")
    elif sort == "value_asc":
        data = data.sort_values(value_col, ascending=True, kind="mergesort")
    elif sort == "label":
        data = data.sort_values(label_col, kind="mergesort")

    labels = data[label_col].astype(str).tolist()
    values = pd.to_numeric(data[value_col], errors="coerce").to_numpy(float)
    yerr = None
    if error_low_col and error_high_col and error_low_col in data.columns:
        lo = values - pd.to_numeric(data[error_low_col], errors="coerce").to_numpy(float)
        hi = pd.to_numeric(data[error_high_col], errors="coerce").to_numpy(float) - values
        yerr = np.vstack([np.abs(lo), np.abs(hi)])

    n = len(labels)
    if horizontal:
        fig, ax = _fig(7.5, max(3.0, 0.32 * n + 1.5))
    else:
        fig, ax = _fig(_bar_width(n), 4.4)

    colors = [_style_for(r)[0] for _, r in data.iterrows()]
    hatches = [_style_for(r)[1] for _, r in data.iterrows()]
    pos = np.arange(n)
    if horizontal:
        bars = ax.barh(pos, values, xerr=yerr, color=colors, edgecolor="#222222",
                       linewidth=0.6, error_kw={"lw": 0.8, "capsize": 2})
        ax.set_yticks(pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(spec.ylabel or value_col)
        ax.set_ylabel(spec.xlabel)
    else:
        bars = ax.bar(pos, values, yerr=yerr, color=colors, edgecolor="#222222",
                      linewidth=0.6, error_kw={"lw": 0.8, "capsize": 2})
        ax.set_xticks(pos)
        ax.set_xticklabels(labels, rotation=55, ha="right")
        ax.set_ylabel(spec.ylabel or value_col)
        ax.set_xlabel(spec.xlabel)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    if annotate and n <= 40:
        for bar, value in zip(bars, values):
            if horizontal:
                ax.annotate(f"{value:.3f}", (bar.get_width(),
                                             bar.get_y() + bar.get_height() / 2),
                            xytext=(3, 0), textcoords="offset points",
                            va="center", fontsize=6.5)
            else:
                ax.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width() / 2,
                                             bar.get_height()),
                            xytext=(0, 2), textcoords="offset points",
                            ha="center", fontsize=6.5)
    if reference_value is not None and np.isfinite(reference_value):
        line = ax.axhline if not horizontal else ax.axvline
        line(reference_value, color="#444444", linestyle="--", linewidth=1.0,
             label=reference_label or f"reference = {reference_value:.3f}")
        ax.legend(loc="best", fontsize=7)
    ax.set_title(spec.title)
    return save_figure(fig, spec, ctx, data)


def grouped_bar_chart(pivot: pd.DataFrame, spec: PlotSpec, ctx: PlotContext, *,
                      value_name: str = "value") -> PlotResult:
    """Grouped bars from a ``rows x columns`` pivot (e.g. policy x source)."""
    if not ctx.allows(spec.importance):
        return _skip(spec, ctx, f"importance below {ctx.min_importance}")
    if pivot.empty:
        return _skip(spec, ctx, "empty pivot")
    groups = list(pivot.index.astype(str))
    series = list(pivot.columns.astype(str))
    n_g, n_s = len(groups), len(series)
    width = 0.8 / max(n_s, 1)
    fig, ax = _fig(max(5.5, 1.15 * n_g + 2.0), 4.4)
    pos = np.arange(n_g)
    for i, col in enumerate(series):
        values = pd.to_numeric(pivot[col], errors="coerce").to_numpy(float)
        style = SOURCE_STYLE.get(col.lower(),
                                 (PALETTE[i % len(PALETTE)], HATCHES[i % len(HATCHES)]))
        bars = ax.bar(pos + i * width - 0.4 + width / 2, values, width * 0.92,
                      label=col, color=style[0], edgecolor="#222222", linewidth=0.5)
        for bar in bars:
            bar.set_hatch(style[1])
    ax.set_xticks(pos)
    ax.set_xticklabels(groups, rotation=20, ha="right")
    ax.set_ylabel(spec.ylabel or value_name)
    ax.set_xlabel(spec.xlabel)
    ax.set_title(spec.title)
    ax.legend(fontsize=7.5, ncol=min(n_s, 4))
    return save_figure(fig, spec, ctx, pivot.reset_index())


# --------------------------------------------------------------------------- #
# Box plots
# --------------------------------------------------------------------------- #
def boxplot(frame: pd.DataFrame, spec: PlotSpec, ctx: PlotContext, *,
            value_col: str, group_col: str = "method_short_label",
            order: Optional[Sequence[str]] = None, log_y: bool = False,
            annotate_median: bool = True,
            reference_value: Optional[float] = None) -> PlotResult:
    """Per-group distribution box plot (log-scaled for runtime)."""
    if not ctx.allows(spec.importance):
        return _skip(spec, ctx, f"importance below {ctx.min_importance}")
    if frame.empty or value_col not in frame.columns:
        return _skip(spec, ctx, f"missing column {value_col}")
    work = frame[[group_col, value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna()
    if log_y:
        work = work[work[value_col] > 0]
    if work.empty:
        return _skip(spec, ctx, "no finite values")
    if order is None:
        medians = work.groupby(group_col)[value_col].median().sort_values(
            ascending=log_y)      # runtime: cheapest first; ARI: best first
        order = list(medians.index.astype(str))
    groups = [work.loc[work[group_col] == g, value_col].to_numpy(float)
              for g in order]
    fig, ax = _fig(_bar_width(len(order)), 4.6)
    # ``labels`` was renamed to ``tick_labels`` in Matplotlib 3.9 and is dropped
    # in 3.11; pick the keyword this installation accepts.
    tick_kw = ("tick_labels"
               if tuple(int(p) for p in matplotlib.__version__.split(".")[:2]) >= (3, 9)
               else "labels")
    bp = ax.boxplot(groups, showfliers=False, patch_artist=True,
                    medianprops={"color": "#111111", "lw": 1.2},
                    whiskerprops={"lw": 0.8}, capprops={"lw": 0.8},
                    **{tick_kw: [str(g) for g in order]})
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(PALETTE[i % len(PALETTE)])
        patch.set_alpha(0.55)
        patch.set_edgecolor("#222222")
        patch.set_hatch(HATCHES[i % len(HATCHES)])
    if log_y:
        ax.set_yscale("log")
    if annotate_median and len(order) <= 40:
        for i, values in enumerate(groups, start=1):
            if values.size:
                med = float(np.median(values))
                ax.annotate(f"{med:.3g}", (i, med), xytext=(0, 4),
                            textcoords="offset points", ha="center", fontsize=6.5)
    if reference_value is not None and np.isfinite(reference_value):
        ax.axhline(reference_value, color="#444444", linestyle="--", linewidth=1.0)
    ax.set_xticklabels([str(g) for g in order], rotation=55, ha="right")
    ax.set_ylabel(spec.ylabel or value_col)
    ax.set_xlabel(spec.xlabel)
    ax.set_title(spec.title)
    return save_figure(fig, spec, ctx, work)


def paired_delta_boxplot(delta_frame: pd.DataFrame, spec: PlotSpec,
                         ctx: PlotContext, *, delta_col: str,
                         group_col: str, order: Optional[Sequence[str]] = None
                         ) -> PlotResult:
    """Paired-delta box plot with a zero reference line."""
    result = boxplot(delta_frame, spec, ctx, value_col=delta_col,
                     group_col=group_col, order=order, log_y=False,
                     reference_value=0.0)
    return result


# --------------------------------------------------------------------------- #
# Heatmaps
# --------------------------------------------------------------------------- #
def heatmap(pivot: pd.DataFrame, spec: PlotSpec, ctx: PlotContext, *,
            cmap: str = "viridis", value_fmt: str = "{:.3f}",
            annotate: Optional[bool] = None, vmin: Optional[float] = None,
            vmax: Optional[float] = None, center: Optional[float] = None
            ) -> PlotResult:
    """Annotated heatmap from a ``rows x columns`` numeric pivot."""
    if not ctx.allows(spec.importance):
        return _skip(spec, ctx, f"importance below {ctx.min_importance}")
    if pivot.empty:
        return _skip(spec, ctx, "empty pivot")
    values = pivot.apply(pd.to_numeric, errors="coerce")
    n_r, n_c = values.shape
    if annotate is None:
        annotate = n_r * n_c <= 400
    if center is not None:
        span = float(np.nanmax(np.abs(values.to_numpy(float) - center)) or 1.0)
        vmin, vmax, cmap = center - span, center + span, "RdBu_r"
    fig, ax = _fig(max(5.0, 0.55 * n_c + 3.0), max(3.2, 0.34 * n_r + 2.0))
    im = ax.imshow(values.to_numpy(float), aspect="auto", cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks(np.arange(n_c))
    ax.set_xticklabels([str(c) for c in values.columns], rotation=55, ha="right")
    ax.set_yticks(np.arange(n_r))
    ax.set_yticklabels([str(r) for r in values.index])
    ax.set_xlabel(spec.xlabel)
    ax.set_ylabel(spec.ylabel)
    ax.set_title(spec.title)
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    if annotate:
        arr = values.to_numpy(float)
        finite = arr[np.isfinite(arr)]
        mid = float(np.nanmedian(finite)) if finite.size else 0.0
        for i in range(n_r):
            for j in range(n_c):
                v = arr[i, j]
                if not np.isfinite(v):
                    continue
                ax.text(j, i, value_fmt.format(v), ha="center", va="center",
                        fontsize=6.0,
                        color="white" if v < mid else "black")
    return save_figure(fig, spec, ctx, values.reset_index())


# --------------------------------------------------------------------------- #
# Scatter / curves
# --------------------------------------------------------------------------- #
def pareto_scatter(summary: pd.DataFrame, spec: PlotSpec, ctx: PlotContext, *,
                   x_col: str, y_col: str, label_col: str = "method_short_label",
                   frontier_col: Optional[str] = None, log_x: bool = True
                   ) -> PlotResult:
    """Accuracy-vs-runtime scatter with the Pareto frontier drawn explicitly.

    No composite "ARI per second" score is computed: the trade-off is shown, not
    collapsed into one number.
    """
    if not ctx.allows(spec.importance):
        return _skip(spec, ctx, f"importance below {ctx.min_importance}")
    needed = {x_col, y_col}
    if summary.empty or not needed <= set(summary.columns):
        return _skip(spec, ctx, f"missing columns {sorted(needed - set(summary.columns))}")
    data = summary.copy()
    data["_x"] = pd.to_numeric(data[x_col], errors="coerce")
    data["_y"] = pd.to_numeric(data[y_col], errors="coerce")
    data = data.dropna(subset=["_x", "_y"])
    if log_x:
        data = data[data["_x"] > 0]
    if data.empty:
        return _skip(spec, ctx, "no finite points")

    fig, ax = _fig(6.8, 4.8)
    for i, (_, r) in enumerate(data.iterrows()):
        color, _hatch = _style_for(r)
        ax.scatter(r["_x"], r["_y"], s=58, color=color,
                   marker=MARKERS[i % len(MARKERS)], edgecolor="#222222",
                   linewidth=0.6, zorder=3)
        ax.annotate(str(r.get(label_col, "")), (r["_x"], r["_y"]),
                    xytext=(4, 3), textcoords="offset points", fontsize=6.5)
    frontier = (data[data[frontier_col].astype(bool)]
                if frontier_col and frontier_col in data.columns
                else _pareto_front(data, "_x", "_y"))
    if len(frontier) > 1:
        f = frontier.sort_values("_x")
        ax.plot(f["_x"], f["_y"], color="#444444", linestyle="-", linewidth=1.1,
                alpha=0.8, zorder=2, label="Pareto frontier")
        ax.legend(fontsize=7.5)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(spec.xlabel or x_col)
    ax.set_ylabel(spec.ylabel or y_col)
    ax.set_title(spec.title)
    return save_figure(fig, spec, ctx, data.drop(columns=["_x", "_y"]))


def _pareto_front(data: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """Non-dominated points minimising ``x`` and maximising ``y``."""
    ordered = data.sort_values([x_col, y_col], ascending=[True, False])
    keep, best_y = [], -np.inf
    for idx, row in ordered.iterrows():
        if row[y_col] > best_y + 1e-15:
            keep.append(idx)
            best_y = row[y_col]
    return data.loc[keep]


def pareto_frontier_table(summary: pd.DataFrame, *, x_col: str, y_col: str
                          ) -> pd.DataFrame:
    """Frontier membership as a table (``comparison_04_pareto_frontier.csv``)."""
    if summary.empty or not {x_col, y_col} <= set(summary.columns):
        return pd.DataFrame()
    data = summary.copy()
    data["_x"] = pd.to_numeric(data[x_col], errors="coerce")
    data["_y"] = pd.to_numeric(data[y_col], errors="coerce")
    data = data.dropna(subset=["_x", "_y"])
    if data.empty:
        return pd.DataFrame()
    front = set(_pareto_front(data, "_x", "_y").index)
    data["is_pareto_optimal"] = data.index.isin(front)
    data["dominated_by"] = ""
    for idx, row in data.iterrows():
        if row["is_pareto_optimal"]:
            continue
        dom = data[(data["_x"] <= row["_x"]) & (data["_y"] >= row["_y"])
                   & (data.index != idx)]
        label_col = ("method_short_label" if "method_short_label" in dom.columns
                     else dom.columns[0])
        data.at[idx, "dominated_by"] = ";".join(
            str(v) for v in dom[label_col].head(4))
    return data.drop(columns=["_x", "_y"]).assign(
        pareto_x_metric=x_col, pareto_y_metric=y_col,
        pareto_note="x minimised, y maximised; no composite ARI/second score is "
                    "computed")


def line_curve(frame: pd.DataFrame, spec: PlotSpec, ctx: PlotContext, *,
               x_col: str, y_col: str, series_col: Optional[str] = None,
               ci_low_col: Optional[str] = None, ci_high_col: Optional[str] = None
               ) -> PlotResult:
    """Line/curve plot with optional shaded confidence band per series."""
    if not ctx.allows(spec.importance):
        return _skip(spec, ctx, f"importance below {ctx.min_importance}")
    if frame.empty or x_col not in frame.columns or y_col not in frame.columns:
        return _skip(spec, ctx, f"missing columns {x_col}/{y_col}")
    data = frame.copy()
    data["_x"] = pd.to_numeric(data[x_col], errors="coerce")
    data["_y"] = pd.to_numeric(data[y_col], errors="coerce")
    data = data.dropna(subset=["_x", "_y"])
    if data.empty:
        return _skip(spec, ctx, "no finite points")

    fig, ax = _fig(6.6, 4.3)
    series = ([(None, data)] if not series_col or series_col not in data.columns
              else list(data.groupby(series_col, sort=True)))
    for i, (name, grp) in enumerate(series):
        grp = grp.sort_values("_x")
        ax.plot(grp["_x"], grp["_y"], marker=MARKERS[i % len(MARKERS)],
                markersize=4, linewidth=1.3, color=PALETTE[i % len(PALETTE)],
                label=str(name) if name is not None else None)
        if ci_low_col and ci_high_col and {ci_low_col, ci_high_col} <= set(grp.columns):
            lo = pd.to_numeric(grp[ci_low_col], errors="coerce")
            hi = pd.to_numeric(grp[ci_high_col], errors="coerce")
            ok = lo.notna() & hi.notna()
            if bool(ok.any()):
                ax.fill_between(grp.loc[ok, "_x"], lo[ok], hi[ok], alpha=0.18,
                                color=PALETTE[i % len(PALETTE)], linewidth=0)
    ax.set_xlabel(spec.xlabel or x_col)
    ax.set_ylabel(spec.ylabel or y_col)
    ax.set_title(spec.title)
    if series_col and len(series) > 1:
        ax.legend(fontsize=7.5, ncol=min(len(series), 3))
    return save_figure(fig, spec, ctx, data.drop(columns=["_x", "_y"]))


def stacked_bar(pivot: pd.DataFrame, spec: PlotSpec, ctx: PlotContext, *,
                order: Optional[Sequence[str]] = None, normalise: bool = True
                ) -> PlotResult:
    """Stacked composition bars (metric categories, winner composition, ...)."""
    if not ctx.allows(spec.importance):
        return _skip(spec, ctx, f"importance below {ctx.min_importance}")
    if pivot.empty:
        return _skip(spec, ctx, "empty pivot")
    data = pivot.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if normalise:
        totals = data.sum(axis=1).replace(0, np.nan)
        data = data.div(totals, axis=0).fillna(0.0)
    if order is not None:
        data = data.reindex([o for o in order if o in data.index])
    fig, ax = _fig(_bar_width(len(data.index)), 4.6)
    bottom = np.zeros(len(data.index))
    for i, col in enumerate(data.columns):
        values = data[col].to_numpy(float)
        bars = ax.bar(np.arange(len(data.index)), values, bottom=bottom,
                      label=str(col), color=PALETTE[i % len(PALETTE)],
                      edgecolor="#222222", linewidth=0.4, width=0.78)
        for bar in bars:
            bar.set_hatch(HATCHES[i % len(HATCHES)])
        bottom += values
    ax.set_xticks(np.arange(len(data.index)))
    ax.set_xticklabels([str(i) for i in data.index], rotation=55, ha="right")
    ax.set_ylabel(spec.ylabel or ("share" if normalise else "count"))
    ax.set_xlabel(spec.xlabel)
    ax.set_title(spec.title)
    ax.legend(fontsize=7, ncol=min(len(data.columns), 4))
    return save_figure(fig, spec, ctx, data.reset_index())


def k_confusion_matrix(best_view: pd.DataFrame, spec: PlotSpec, ctx: PlotContext,
                       *, method_name: str, max_k: int = 10) -> PlotResult:
    """True-K vs selected-K confusion matrix for one method."""
    sub = best_view[best_view["method_name"] == method_name]
    if sub.empty:
        return _skip(spec, ctx, f"no rows for {method_name}")
    true_k = pd.to_numeric(sub["true_k"], errors="coerce")
    sel_k = pd.to_numeric(sub["selected_k"], errors="coerce")
    ok = true_k.notna() & sel_k.notna()
    if not bool(ok.any()):
        return _skip(spec, ctx, "no finite K pairs")
    table = pd.crosstab(true_k[ok].clip(upper=max_k).astype(int),
                        sel_k[ok].clip(upper=max_k).astype(int))
    table.index.name = "true_k"
    table.columns.name = "selected_k"
    return heatmap(table, spec, ctx, cmap="Blues", value_fmt="{:.0f}",
                   annotate=True)
