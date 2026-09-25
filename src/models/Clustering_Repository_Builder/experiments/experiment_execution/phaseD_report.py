"""Phase-D sweep helper: report a completed subfamily and emit the next one.

Reads the ordered subfamily list (phaseD_subfamily_order.json) and progress file
(phaseD_run_progress.json), prints a compact best-view-ARI report for the given
completed subfamily index, marks it done, and prints the NEXT subfamily's
index + directory so the driver can launch it.

Usage::
    python -m models.Clustering_Repository_Builder.experiments.experiment_execution.phaseD_report --done <idx>
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[4]
CR = REPO / "results_analysis" / "clustering_repository"
ORDER = CR / "phaseD_subfamily_order.json"
PROGRESS = CR / "phaseD_run_progress.json"
SUFFIX = "phaseD_no_earlystop_50trials_dynamic_ablation"


def _report(sub_dir: str, name: str) -> None:
    from .output_writer import _ext, path_exists  # long-path-safe helpers

    sd = Path(sub_dir) / "experiment_execution_summaries" / f"split_01_{SUFFIX}"
    csv = sd / "execution_summary.csv"
    if not path_exists(csv):
        print(f"  [{name}] no execution_summary.csv"); return
    with open(_ext(csv), "r", encoding="utf-8") as fh:  # \\?\ prefix on Windows
        df = pd.read_csv(fh)
    n = df["dataset_id"].nunique()
    st = df["status"].value_counts().to_dict()
    es_ok = bool((df["early_stopping_enabled"].dropna() == False).all()) if "early_stopping_enabled" in df else None
    tr_ok = bool((pd.to_numeric(df.get("optuna_trials_requested"), errors="coerce").dropna() == 50).all())
    ok = df[df["status"] == "success"].copy()
    ok["ari"] = pd.to_numeric(ok["ari"], errors="coerce")
    print(f"  [{name}] datasets={n} status={st} early_stop_off={es_ok} trials50={tr_ok}")
    if not ok.empty:
        best = ok.loc[ok.groupby(["method_name", "dataset_id"])["ari"].idxmax()]
        bv = best.groupby("method_name")["ari"].mean().sort_values(ascending=False)
        print("    top5: " + " | ".join(f"{m}={v:.4f}" for m, v in bv.head(5).items()))
        print("    worst2: " + " | ".join(f"{m}={v:.4f}" for m, v in bv.tail(2).items()))
        dyn = best[best["method_name"].str.contains("dynamic")]
        if not dyn.empty and "selected_metric_count" in dyn:
            kd = {m: round(g["selected_metric_count"].mean(), 1) for m, g in dyn.groupby("method_name")}
            print(f"    dynamic mean K: {kd}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--done", type=int, required=True)
    args = ap.parse_args()

    order = json.loads(ORDER.read_text())
    prog = json.loads(PROGRESS.read_text())
    idx = args.done
    row = next(r for r in order if r["idx"] == idx)
    print(f"=== Subfamily {idx + 1}/86 DONE: {row['family']}/{row['subfamily']} ===")
    _report(row["dir"], row["subfamily"])

    prog["completed_indices"] = sorted(set(prog["completed_indices"] + [idx]))
    PROGRESS.write_text(json.dumps(prog, indent=1))
    done = len(prog["completed_indices"])
    print(f"progress: {done}/86 subfamilies complete")

    nxt = next((r for r in order if r["idx"] not in prog["completed_indices"]), None)
    if nxt is None:
        print("NEXT: ALL_DONE")
    else:
        print(f"NEXT_IDX={nxt['idx']}")
        print(f"NEXT_FAMILY={nxt['family']}")
        print(f"NEXT_SUBFAMILY={nxt['subfamily']}")
        print(f"NEXT_DIR={nxt['dir']}")


if __name__ == "__main__":
    main()
