"""Run AutoClust In-Domain on split 1 for one subfamily (8 workers).

Evaluation only. Loads the in-domain AutoClust models (trained on splits 2..16)
and runs the in-domain application phase on every split-1 dataset of the
subfamily, writing outputs in the same format as the original AutoClust adapter.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

REPO_DEFAULT = r"."
CONFIG_ROOT_DEFAULT = r"models\ClustOpt\configs\utilities_maps"
RECORD_TYPES = ("1d_x", "1d_y", "2d")

_G: Dict[str, object] = {}


def _ensure_paths(repo_root: str) -> None:
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _init_worker(repo_root, model_dir, master_mf, config_root, params):
    _ensure_paths(repo_root)
    from experiments.external_baselines.AutoClust.indomain.models import InDomainAutoClust
    _G["model"] = InDomainAutoClust(model_dir, master_mf)
    _G.update(repo_root=repo_root, config_root=config_root, params=params)


def _work(ds: dict) -> dict:
    from experiments.external_baselines.AutoClust.indomain.dataset import process_dataset_indomain
    p = _G["params"]
    try:
        return process_dataset_indomain(
            ds["dataset_folder"], split_id=1, record_types=list(RECORD_TYPES),
            repo_root=_G["repo_root"], model=_G["model"], config_root=_G["config_root"],
            n_evaluations=p["n_evaluations"], k_range=tuple(p["k_range"]),
            seed=p["seed"], overwrite=p["overwrite"], variant=p["variant"],
            constrain_search_space=bool(p.get("constrain_search_space", False)),
            resume=bool(p.get("resume", True)))
    except Exception as exc:  # noqa: BLE001
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        return {"dataset_id": ds.get("dataset_id"), "family": ds.get("family"),
                "subfamily": ds.get("subfamily"), "fatal_error": f"{type(exc).__name__}: {exc}",
                "traceback": tb, "record_statuses": {}, "best_ari": None, "total_runtime_sec": 0.0}


def _fmt_elapsed(s: float) -> str:
    s = int(max(s, 0)); h, r = divmod(s, 3600); m, sec = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="AutoClust In-Domain — split 1, one subfamily")
    ap.add_argument("--family", required=True)
    ap.add_argument("--subfamily", required=True)
    ap.add_argument("--repo-root", default=REPO_DEFAULT)
    ap.add_argument("--master-root", required=True)
    ap.add_argument("--model-variant-dir", required=True)
    ap.add_argument("--config-root", default=CONFIG_ROOT_DEFAULT)
    ap.add_argument("--variant", default="AutoClust_Original_InDomain")
    ap.add_argument("--constrain-search-space", action="store_true",
                    help="Constrain the candidate clustering configs to the ClustOpt "
                         "phaseE2 search map (auto-enabled when --variant ends with "
                         "'_same_search_space'). AutoClust meta-learning is unchanged.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-evaluations", type=int, default=50)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--overwrite", action="store_true")
    # Stage-3A2: explicit strict resume. Default ON, matching the historical
    # default, but the completion test is now status==success AND full config
    # compatibility (see indomain.dataset.unit_is_complete).
    ap.add_argument("--resume", dest="resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="recompute every record even if a compatible success "
                         "exists (does not delete other variants)")
    ap.add_argument("--limit-datasets", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    _ensure_paths(args.repo_root)
    import json
    import pandas as pd
    from experiments.external_baselines.AutoClust.runners.run_autoclust_subfamily_split import (
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

    print(f"[ac-indomain] {args.family}/{args.subfamily}: {len(datasets)} split-1 datasets "
          f"(loaded_model_train_splits=2..16, running_test_split=1)", flush=True)
    if args.dry_run:
        for d in datasets[:10]:
            print("   ", d["dataset_id"], d["dataset_folder"])
        return 0

    constrain = bool(args.constrain_search_space) or args.variant.endswith("_same_search_space")
    params = {"n_evaluations": args.n_evaluations, "k_range": [args.k_min, args.k_max],
              "seed": args.seed, "overwrite": args.overwrite, "variant": args.variant,
              "constrain_search_space": constrain,
              "search_space_source": ("clustopt_phaseE2_map" if constrain
                                      else "autoclust_native"),
              "resume": bool(args.resume)}
    init_args = (args.repo_root, args.model_variant_dir, str(master / "metafeatures.parquet"),
                 args.config_root, params)

    t0 = time.perf_counter()
    results: List[dict] = []
    if args.workers <= 1:
        _init_worker(*init_args)
        for d in datasets:
            results.append(_work(d)); print(f"   [{len(results)}/{len(datasets)}] {results[-1].get('dataset_id')} "
                                            f"best_ari={results[-1].get('best_ari')}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                                 initargs=init_args) as ex:
            futs = {ex.submit(_work, d): d for d in datasets}
            done = 0
            for fut in as_completed(futs):
                results.append(fut.result()); done += 1
                if done % 5 == 0 or done == len(datasets):
                    print(f"   [{done}/{len(datasets)}] running…", flush=True)

    wallclock = time.perf_counter() - t0
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = run_summary_dir(Path(args.repo_root), 1, args.subfamily, ts)
    art0 = next((r.get("artifact_status") for r in results if r.get("artifact_status")), {})
    run_config = {
        "method": "AutoClust", "variant": args.variant, "mode": "in_domain",
        "mkr_source": "Unified_MKR/derived/autoclust_split1/" + Path(args.model_variant_dir).name,
        "subfamily_name": args.subfamily, "family": args.family, "split_id": 1,
        "heldout_test_split": 1, "train_splits": list(range(2, 17)),
        "record_types": list(RECORD_TYPES), "workers": args.workers,
        "constrain_search_space": constrain,
        "search_space_source": "clustopt_phaseE2_map" if constrain else "autoclust_native",
        "n_evaluations": args.n_evaluations, "total_evaluations": args.n_evaluations,
        "k_range": [args.k_min, args.k_max], "overwrite": args.overwrite,
        "n_datasets": len(datasets), "wallclock_sec": wallclock,
        "wallclock_str": _fmt_elapsed(wallclock), "artifact_status": art0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_run_summaries(out_dir, results=results, run_config=run_config)

    # ---- @@@ progress line ----
    df = pd.DataFrame([{
        # "skipped" == previously-succeeded (resume): count as completed, not failed.
        "any_success": (not r.get("fatal_error")) and any(
            s in ("success", "skipped") for s in (r.get("record_statuses") or {}).values()),
        "best_ari": r.get("best_ari"), "algorithm": r.get("selected_algorithm"),
        "selected_k": r.get("selected_k"), "true_k": r.get("true_k"),
        "runtime": r.get("total_runtime_sec"),
    } for r in results])
    ok = df[df["any_success"]]
    bv = pd.to_numeric(ok["best_ari"], errors="coerce").dropna()
    kk = ok.dropna(subset=["selected_k", "true_k"])
    kacc = float((kk["selected_k"] == kk["true_k"]).mean()) if len(kk) else None
    rt = pd.to_numeric(df["runtime"], errors="coerce").dropna()
    done = {
        "subfamily": args.subfamily, "attempted": len(datasets),
        "completed": int(df["any_success"].sum()), "failed": int((~df["any_success"]).sum()),
        "mean_best_view_ari": float(bv.mean()) if len(bv) else None,
        "median_best_view_ari": float(bv.median()) if len(bv) else None,
        "k_accuracy": kacc, "runtime_min": round(wallclock / 60, 1),
        "mean_runtime_per_dataset_sec": float(rt.mean()) if len(rt) else None,
        "median_runtime_per_dataset_sec": float(rt.median()) if len(rt) else None,
        # selector dist == final algorithm dist (HPO is over the selected algorithm)
        "algorithm_selector_dist": {str(k): int(v) for k, v in ok["algorithm"].value_counts().items()},
        "final_algorithm_dist": {str(k): int(v) for k, v in ok["algorithm"].value_counts().items()},
        "predicted_k_dist": {str(int(k)): int(v) for k, v in ok["selected_k"].dropna().value_counts().items()},
        "output": str(out_dir),
    }
    print("@@@ SUBFAMILY DONE: " + json.dumps(done), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
