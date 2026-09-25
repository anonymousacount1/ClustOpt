"""Smoke test for the AutoClust In-Domain phase (one dataset, 3 views)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_DEFAULT = r"."
CONFIG_ROOT_DEFAULT = r"models\ClustOpt\configs\utilities_maps"
K_BASED = {"kmeans", "minibatch_kmeans", "gmm", "agglomerative", "birch"}
PER_RECORD_FILES = ["result.json", "summary_metrics.json", "raw_result.json",
                    "autoclust_artifact_report.json", "labels.npy", "run_log.txt"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="arcs_circles_rings")
    ap.add_argument("--subfamily", default=None)
    ap.add_argument("--repo-root", default=REPO_DEFAULT)
    ap.add_argument("--master-root", required=True)
    ap.add_argument("--model-variant-dir", required=True)
    ap.add_argument("--variant", default="AutoClust_Original_InDomain")
    ap.add_argument("--expect-candidates", type=int, default=7)
    ap.add_argument("--n-evaluations", type=int, default=50)
    ap.add_argument("--config-root", default=CONFIG_ROOT_DEFAULT)
    args = ap.parse_args(argv)

    if args.repo_root not in sys.path:
        sys.path.insert(0, args.repo_root)
    import pandas as pd
    from experiments.external_baselines.AutoClust.indomain.models import InDomainAutoClust
    from experiments.external_baselines.AutoClust.indomain.dataset import process_dataset_indomain, variant_dir_for

    master = Path(args.master_root)
    rec = pd.read_parquet(master / "records.parquet",
                          columns=["dataset_id", "dataset_folder", "family", "subfamily", "split_id"]).drop_duplicates("dataset_id")
    sel = rec[(rec.split_id == 1) & (rec.family == args.family)]
    if args.subfamily:
        sel = sel[sel.subfamily == args.subfamily]
    sel = sel.sort_values("dataset_id")
    if not len(sel):
        print("SMOKE FAIL: no split-1 dataset"); return 1
    ds = sel.iloc[0].to_dict()
    print(f"[smoke] dataset={ds['dataset_id']} subfamily={ds['subfamily']}")

    model = InDomainAutoClust(args.model_variant_dir, str(master / "metafeatures.parquet"))
    print(f"[smoke] models loaded: selector={model.selector_model_type}, train_splits={model.train_split_ids}, "
          f"candidate_cvis={len(model.candidate_cvis)}, predictor_arch={model.predictor_arch}")

    result = process_dataset_indomain(
        ds["dataset_folder"], split_id=1, record_types=["1d_x", "1d_y", "2d"],
        repo_root=args.repo_root, model=model, config_root=args.config_root,
        n_evaluations=args.n_evaluations, k_range=(2, 5), overwrite=True, variant=args.variant)

    vdir = variant_dir_for(Path(ds["dataset_folder"]), 1, args.variant)
    checks = {"models_loaded": model.selector is not None and model.predictor is not None,
              "candidate_cvi_count": len(model.candidate_cvis),
              "candidate_count_matches_expected": len(model.candidate_cvis) == args.expect_candidates,
              "no_extended_cvis": Path(args.model_variant_dir).name != "extended_cvis" or args.variant.endswith("Extended_InDomain"),
              "dataset_summary_written": (vdir / "dataset_summary.json").exists(),
              "readme_written": (vdir / "README_AUTOCLUST_RUN.md").exists(),
              "best_view_ari_computed": result.get("best_ari") is not None}
    per = {}
    for rt in ("1d_x", "1d_y", "2d"):
        rd = vdir / rt
        files_ok = {f: (rd / f).exists() for f in PER_RECORD_FILES}
        r = json.loads((rd / "result.json").read_text(encoding="utf-8")) if (rd / "result.json").exists() else {}
        raw = json.loads((rd / "raw_result.json").read_text(encoding="utf-8")) if (rd / "raw_result.json").exists() else {}
        hist = raw.get("config_history", [])
        kbad = [h for h in hist if h.get("algorithm") in K_BASED
                for kk in [h.get("hyperparameters", {}).get("n_clusters")
                           or h.get("hyperparameters", {}).get("n_components")]
                if kk is not None and not (2 <= int(kk) <= 5)]
        per[rt] = {
            "all_files_present": all(files_ok.values()), "files": files_ok,
            "status": r.get("status"), "selected_algorithm": r.get("selected_algorithm"),
            "optimizer_evaluations": r.get("optimizer_evaluations"),
            "evals_eq_50": (r.get("optimizer_evaluations") == args.n_evaluations),
            "mlp_objective_used": r.get("mlp_objective_used"),
            "metrics_keys": sorted((r.get("metrics") or {}).keys()),
            "n_configs_evaluated": len(hist),
            "k_based_within_2_5": len(kbad) == 0,
        }
    checks["per_record"] = per
    checks["all_records_files_present"] = all(v["all_files_present"] for v in per.values())
    checks["all_k_within_2_5"] = all(v["k_based_within_2_5"] for v in per.values())
    checks["all_evals_eq_50"] = all(v["evals_eq_50"] for v in per.values() if v["status"] == "success")
    required_metrics = {"ari", "nmi", "ami", "v_measure", "homogeneity", "completeness",
                        "fowlkes_mallows", "purity", "selected_k", "true_k", "exact_k_match",
                        "n_predicted_clusters_excluding_noise", "n_noise_points"}
    checks["metrics_schema_complete"] = all(
        required_metrics.issubset(set(v["metrics_keys"])) for v in per.values() if v["status"] == "success")

    passed = all(checks[k] for k in ["models_loaded", "candidate_count_matches_expected", "no_extended_cvis",
                                     "dataset_summary_written", "readme_written", "best_view_ari_computed",
                                     "all_records_files_present", "all_k_within_2_5", "all_evals_eq_50",
                                     "metrics_schema_complete"])
    print(json.dumps(checks, indent=2, default=str))
    print(f"\nSMOKE TEST: {'PASS' if passed else 'FAIL'}")
    print(f"best_record_type={result.get('best_record_type')} best_ari={result.get('best_ari')} "
          f"selected_algorithm={result.get('selected_algorithm')}")
    print(f"output: {vdir}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
