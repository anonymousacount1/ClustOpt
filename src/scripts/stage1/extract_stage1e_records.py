"""Stage-1E record extraction (Split 1, online_fixed50).

Reads the PERSISTED per-run artifacts of the completed online sweep and writes a
single flat record table. Never re-runs anything, never touches an experiment
directory (read-only), and never derives numbers from console output.

Per run (dataset x view x arm) it records:
  * the selected outcome            status.json / best_result.json
  * the trial population            clustopt_results.csv
  * search/selection decomposition  best-visited ARI, regret
  * reranker evidence               objective-descending rank of the best-ARI
                                    trial, and whether it falls in Top-1/3/5/10

Parallel by subfamily (8 processes, 1 compute thread each -- the same resource
protocol the sweep used). Long-path safe.

Run:
    .venv_clustopt/Scripts/python.exe scripts/stage1/extract_stage1e_records.py
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from models.Clustering_Repository_Builder.utility_generation.worker_setup import (  # noqa: E402
    configure_process,
)

configure_process()

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from models.Clustering_Repository_Builder.experiments.experiment_execution.config import (  # noqa: E402
    ExperimentExecutionConfig, METHOD_NAMES_STAGE1_2X3, VIEW_IDS,
)
from models.Clustering_Repository_Builder.experiments.experiment_execution.method_runner import (  # noqa: E402
    run_output_dir,
)
from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E402
    _ext, ensure_dir, path_exists, read_json,
)
from models.Clustering_Repository_Builder.experiments.experiment_execution.run_offline_2x3_all_subfamilies import (  # noqa: E402
    ARM_ORDER, _split1_dataset_dirs, discover_subfamilies,
)

ARM_OF = {m: c for c, m in ARM_ORDER}
INVENTORY_OF = {"HU": "head14", "HP": "head14", "HO": "head14",
                "FU": "full60", "FP": "full60", "FO": "full60"}
UTILITY_OF = {"HU": "uniform", "HP": "predicted", "HO": "oracle",
              "FU": "uniform", "FP": "predicted", "FO": "oracle"}
DEPLOYABLE = {"HU": True, "HP": True, "HO": False,
              "FU": True, "FP": True, "FO": False}

OUT = Path(REPO) / "results_analysis/clustering_repository/stage1_2x3_aggregation/online_fixed50"
RECORDS = OUT / "online_2x3_record_results.csv"

_TRUE = ("true", "1", "yes")


def _trial_stats(csv_path: Path) -> dict:
    """Search-population statistics for one run, read from clustopt_results.csv.

    Ranks are computed over the VALID trials only, ordered by the objective the
    policy actually maximised (``aggregate_score``, descending). Ties in the
    objective are broken by the stored trial order -- the same first-wins
    convention the pipeline uses -- so the rank is what a reranker would face.
    """
    out = {
        "n_trials_rows": np.nan, "n_valid_rows": np.nan,
        "best_visited_ari": np.nan, "worst_visited_ari": np.nan,
        "mean_visited_ari": np.nan,
        "rank_of_best_ari_trial": np.nan, "n_tied_at_max_J": np.nan,
        "best_in_top1": np.nan, "best_in_top3": np.nan,
        "best_in_top5": np.nan, "best_in_top10": np.nan,
        "ari_at_rank1": np.nan, "max_ari_in_top3": np.nan,
        "max_ari_in_top5": np.nan, "max_ari_in_top10": np.nan,
        "objective_ari_spearman": np.nan,
    }
    if not path_exists(csv_path):
        return out
    try:
        t = pd.read_csv(_ext(csv_path))
    except Exception:
        return out
    out["n_trials_rows"] = int(len(t))
    if "valid" in t.columns:
        valid_mask = t["valid"].astype(str).str.strip().str.lower().isin(_TRUE)
    else:
        valid_mask = pd.Series(True, index=t.index)
    t = t[valid_mask].copy()
    t["ARI"] = pd.to_numeric(t.get("ARI"), errors="coerce")
    t["aggregate_score"] = pd.to_numeric(t.get("aggregate_score"), errors="coerce")
    t = t.dropna(subset=["ARI", "aggregate_score"])
    out["n_valid_rows"] = int(len(t))
    if t.empty:
        return out

    ari = t["ARI"].to_numpy(float)
    J = t["aggregate_score"].to_numpy(float)
    out["best_visited_ari"] = float(ari.max())
    out["worst_visited_ari"] = float(ari.min())
    out["mean_visited_ari"] = float(ari.mean())
    out["n_tied_at_max_J"] = int((np.abs(J - J.max()) <= 1e-12).sum())

    # stable descending sort on J -- preserves stored trial order within ties
    order = np.argsort(-J, kind="stable")
    ari_ranked = ari[order]
    best_pos = int(np.argmax(ari_ranked))          # first occurrence of max ARI
    out["rank_of_best_ari_trial"] = best_pos + 1
    for k in (1, 3, 5, 10):
        out[f"best_in_top{k}"] = float(best_pos < k)
        out[f"max_ari_in_top{k}"] = float(ari_ranked[:k].max())
    out["ari_at_rank1"] = float(ari_ranked[0])

    if len(t) >= 3 and np.ptp(J) > 0 and np.ptp(ari) > 0:
        try:
            from scipy import stats as sst
            out["objective_ari_spearman"] = float(sst.spearmanr(J, ari).statistic)
        except Exception:
            pass
    return out


def extract_subfamily(payload: tuple) -> list:
    """One subfamily -> list of per-run record dicts. Runs in a worker process."""
    configure_process()
    fam, subname, sub_str, split_csv_str = payload
    sub_path = Path(sub_str)
    cfg = ExperimentExecutionConfig(
        repo_root=Path(REPO), subfamily_dir=sub_path,
        split_assignments=Path(split_csv_str), split_id=1,
        split_dir_suffix="online_fixed50", method_set="stage1_2x3",
        methods=tuple(METHOD_NAMES_STAGE1_2X3), max_workers=8)
    split_dir = cfg.split_dir_name()

    rows = []
    for d in _split1_dataset_dirs(cfg):
        d = Path(d)
        for method in METHOD_NAMES_STAGE1_2X3:
            code = ARM_OF[method]
            for view in VIEW_IDS:
                o = run_output_dir(d, split_dir, method, view)
                st = read_json(o / "status.json") or {}
                br = read_json(o / "best_result.json") or {}
                sm = read_json(o / "selected_metrics.json") or {}
                row = {
                    "family": fam, "subfamily": subname,
                    "dataset_id": d.name, "view_id": view,
                    "method_name": method, "arm": code,
                    "metric_inventory": INVENTORY_OF[code],
                    "utility_strategy": UTILITY_OF[code],
                    "deployable": DEPLOYABLE[code],
                    "status": st.get("status"),
                    "ari": st.get("ari"),
                    "selected_k": st.get("selected_k"),
                    "true_k": st.get("true_k"),
                    "runtime_sec": st.get("runtime_sec"),
                    "requested_trials": br.get("requested_trials"),
                    "delivered_trials": br.get("delivered_trials",
                                               br.get("n_trials_completed")),
                    "valid_trials": br.get("valid_trials"),
                    "failed_trials": br.get("failed_trials"),
                    "search_seed": br.get("search_seed"),
                    "objective_J": br.get("best_objective_score"),
                    "algorithm": br.get("selected_algorithm"),
                    "k_correct": br.get("k_correct"),
                    "cvi_type": br.get("cvi_type"),
                    "used_fallback": br.get("used_fallback"),
                    "selected_metrics": "|".join(sm.get("selected_metrics", []))
                    if isinstance(sm.get("selected_metrics"), list) else None,
                    "n_selected_metrics": (len(sm["selected_metrics"])
                                           if isinstance(sm.get("selected_metrics"), list)
                                           else np.nan),
                    "predictor": (sm.get("resolver_debug") or {}).get("model_run_dir"),
                    "resolver_top_k": (sm.get("resolver_debug") or {}).get("top_k"),
                }
                row.update(_trial_stats(o / "clustopt_results.csv"))
                rows.append(row)
    return rows


def main() -> int:
    print("=" * 78)
    print("STAGE-1E RECORD EXTRACTION (online_fixed50, Split 1)")
    print("=" * 78)
    ensure_dir(OUT)
    analyzed = Path(REPO) / "results_analysis/clustering_repository/analyzed_data"
    split_csv = analyzed / "experiment_splits/dataset_split_assignments.csv"
    subs = discover_subfamilies(analyzed)
    print(f"  subfamilies: {len(subs)}   workers: 8   threads/worker: 1")

    payloads = [(s.parent.parent.name, s.name, str(s), str(split_csv)) for s in subs]
    t0 = time.time()
    all_rows, done = [], 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(extract_subfamily, p): p[1] for p in payloads}
        for fut in as_completed(futs):
            all_rows.extend(fut.result())
            done += 1
            if done % 10 == 0 or done == len(payloads):
                print(f"    {done}/{len(payloads)} subfamilies "
                      f"({len(all_rows):,} records, {time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["family", "subfamily", "dataset_id", "view_id", "arm"])
    df.to_csv(_ext(RECORDS), index=False)
    print(f"\n  records: {len(df):,}  ->  {RECORDS.name}")
    print(f"  datasets: {df['dataset_id'].nunique()}  "
          f"families: {df['family'].nunique()}  "
          f"subfamilies: {df['subfamily'].nunique()}")
    print(f"  elapsed: {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
