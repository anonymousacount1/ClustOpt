"""Run ML2DAC Original In-Domain on split 1 for one subfamily (8 workers).

Evaluation only. Loads the in-domain Original MKR (trained on splits 2..16) and
runs the in-domain application phase on every split-1 dataset of the subfamily,
writing per-dataset and per-subfamily outputs in the SAME format as the original
ML2DAC adapter (reusing result_converter + write_run_summaries).
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_DEFAULT = r"."
CONFIG_ROOT_DEFAULT = r"models\ClustOpt\configs\utilities_maps"
RECORD_TYPES = ("1d_x", "1d_y", "2d")

_G: Dict[str, object] = {}


def _ensure_paths(repo_root: str) -> None:
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _init_worker(repo_root, mkr_dir, master_mf, config_root, params):
    _ensure_paths(repo_root)
    from experiments.external_baselines.ML2DAC.indomain.mkr import InDomainMKR
    _G["mkr"] = InDomainMKR(mkr_dir, master_mf)
    _G.update(repo_root=repo_root, config_root=config_root, params=params)


def _work(ds: dict) -> dict:
    from experiments.external_baselines.ML2DAC.indomain.dataset import process_dataset_indomain
    p = _G["params"]
    try:
        return process_dataset_indomain(
            ds["dataset_folder"], split_id=1, record_types=list(RECORD_TYPES),
            repo_root=_G["repo_root"], mkr=_G["mkr"], config_root=_G["config_root"],
            n_warmstarts=p["n_warmstarts"], n_optimizer=p["n_optimizer"],
            k_range=tuple(p["k_range"]), seed=p["seed"], overwrite=p["overwrite"],
            variant=p["variant"],
            constrain_search_space=bool(p.get("constrain_search_space", False)))
    except Exception as exc:  # noqa: BLE001
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        return {"dataset_id": ds.get("dataset_id"), "family": ds.get("family"),
                "subfamily": ds.get("subfamily"), "fatal_error": f"{type(exc).__name__}: {exc}",
                "traceback": tb, "record_statuses": {}, "best_ari": None, "total_runtime_sec": 0.0}


def _fmt_elapsed(seconds: float) -> str:
    seconds = int(max(seconds, 0)); h, rem = divmod(seconds, 3600); m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ML2DAC Original In-Domain — split 1, one subfamily")
    ap.add_argument("--family", required=True)
    ap.add_argument("--subfamily", required=True)
    ap.add_argument("--repo-root", default=REPO_DEFAULT)
    ap.add_argument("--master-root", required=True)
    ap.add_argument("--mkr-variant-dir", required=True)
    ap.add_argument("--config-root", default=CONFIG_ROOT_DEFAULT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--variant", default="ML2DAC_Original_InDomain")
    ap.add_argument("--constrain-search-space", action="store_true",
                    help="Constrain the candidate clustering configs to the ClustOpt "
                         "phaseE2 search map (auto-enabled when --variant ends with "
                         "'_same_search_space'). Meta-ranking is unchanged.")
    ap.add_argument("--n-warmstarts", type=int, default=25)
    ap.add_argument("--n-optimizer", type=int, default=25)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit-datasets", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    _ensure_paths(args.repo_root)
    import json
    import numpy as np
    import pandas as pd
    from experiments.external_baselines.ML2DAC.runners.run_ml2dac_subfamily_split import (
        run_summary_dir, write_run_summaries,
    )

    master = Path(args.master_root)
    rec = pd.read_parquet(master / "records.parquet",
                          columns=["dataset_id", "dataset_folder", "family", "subfamily", "split_id"]).drop_duplicates("dataset_id")
    sel = rec[(rec.split_id == 1) & (rec.family == args.family) & (rec.subfamily == args.subfamily)]
    datasets: List[dict] = sel.to_dict("records")
    if args.limit_datasets:
        datasets = datasets[: args.limit_datasets]
    assert all(d["split_id"] == 1 for d in datasets), "non-split-1 dataset selected"

    print(f"[indomain] {args.family}/{args.subfamily}: {len(datasets)} split-1 datasets "
          f"(loaded_mkr_train_splits=2..16, running_test_split=1)", flush=True)
    if args.dry_run:
        for d in datasets[:10]:
            print("   ", d["dataset_id"], d["dataset_folder"])
        return 0

    constrain = bool(args.constrain_search_space) or args.variant.endswith("_same_search_space")
    params = {"n_warmstarts": args.n_warmstarts, "n_optimizer": args.n_optimizer,
              "k_range": [args.k_min, args.k_max], "seed": args.seed, "overwrite": args.overwrite,
              "variant": args.variant, "constrain_search_space": constrain}
    init_args = (args.repo_root, args.mkr_variant_dir, str(master / "metafeatures.parquet"),
                 args.config_root, params)

    t0 = time.perf_counter()
    results: List[dict] = []
    if args.workers <= 1:
        _init_worker(*init_args)
        for d in datasets:
            results.append(_work(d))
            print(f"   [{len(results)}/{len(datasets)}] {results[-1].get('dataset_id')} "
                  f"best_ari={results[-1].get('best_ari')}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                                 initargs=init_args) as ex:
            futs = {ex.submit(_work, d): d for d in datasets}
            done = 0
            for fut in as_completed(futs):
                results.append(fut.result()); done += 1
                if done % 10 == 0 or done == len(datasets):
                    print(f"   [{done}/{len(datasets)}] running…", flush=True)

    wallclock = time.perf_counter() - t0
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = run_summary_dir(Path(args.repo_root), 1, args.subfamily, ts)
    art0 = next((r.get("artifact_status") for r in results if r.get("artifact_status")), {})
    run_config = {
        "method": "ML2DAC", "variant": args.variant,
        "mode": "in_domain",
        "mkr_source": "Unified_MKR/derived/ml2dac_split1/" + Path(args.mkr_variant_dir).name,
        "subfamily_name": args.subfamily, "family": args.family, "split_id": 1,
        "heldout_test_split": 1, "train_splits": list(range(2, 17)),
        "record_types": list(RECORD_TYPES), "workers": args.workers,
        "constrain_search_space": constrain,
        "search_space_source": "clustopt_phaseE2_map" if constrain else "ml2dac_native",
        "n_warmstarts": args.n_warmstarts, "n_optimizer_loops": args.n_optimizer,
        "total_iterations": args.n_warmstarts + args.n_optimizer, "k_range": [args.k_min, args.k_max],
        "overwrite": args.overwrite, "n_datasets": len(datasets),
        "wallclock_sec": wallclock, "wallclock_str": _fmt_elapsed(wallclock),
        "artifact_status": art0, "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_run_summaries(out_dir, results=results, run_config=run_config)

    # ---- @@@ progress line ----
    df = pd.DataFrame([{
        "fatal": bool(r.get("fatal_error")),
        "any_success": any(s == "success" for s in (r.get("record_statuses") or {}).values()),
        "best_ari": r.get("best_ari"), "selected_cvi": r.get("selected_cvi"),
        "selected_cvi_source": r.get("selected_cvi_source"),
        "algorithm": r.get("selected_algorithm"),
        "selected_k": r.get("selected_k"), "true_k": r.get("true_k"),
    } for r in results])
    completed = int(df["any_success"].sum())
    failed = int((~df["any_success"]).sum())
    ok = df[df["any_success"]]
    bv = pd.to_numeric(ok["best_ari"], errors="coerce").dropna()
    kk = ok.dropna(subset=["selected_k", "true_k"])
    kacc = float((kk["selected_k"] == kk["true_k"]).mean()) if len(kk) else None
    done = {
        "subfamily": args.subfamily, "attempted": len(datasets), "completed": completed,
        "failed": failed,
        "mean_best_view_ari": float(bv.mean()) if len(bv) else None,
        "median_best_view_ari": float(bv.median()) if len(bv) else None,
        "k_accuracy": kacc,
        "runtime_min": round(wallclock / 60, 1),
        "selected_cvi": {str(k): int(v) for k, v in ok["selected_cvi"].value_counts().items()},
        "selected_cvi_source": {str(k): int(v) for k, v in ok["selected_cvi_source"].value_counts().items()},
        "algorithms": {str(k): int(v) for k, v in ok["algorithm"].value_counts().items()},
        "output": str(out_dir),
    }
    print("@@@ SUBFAMILY DONE: " + json.dumps(done), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
