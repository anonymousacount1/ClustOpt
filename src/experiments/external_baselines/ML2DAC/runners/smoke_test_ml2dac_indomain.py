"""Smoke test for the ML2DAC Original In-Domain phase (one dataset, 3 records).

Verifies the in-domain run produces the SAME on-disk file set as the original
ML2DAC adapter (1d_x/1d_y/2d with result.json/summary_metrics.json/raw_result.json/
ml2dac_artifact_report.json/labels.npy/run_log.txt + dataset_summary.json + README).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_DEFAULT = r"."
CONFIG_ROOT_DEFAULT = r"models\ClustOpt\configs\utilities_maps"
ORIGINAL_CVI_ABBREVS = {"CH", "DBI", "SIL", "DBCV", "DI", "CJI", "COP"}
K_BASED = {"kmeans", "minibatch_kmeans", "gmm", "agglomerative", "birch"}
PER_RECORD_FILES = ["result.json", "summary_metrics.json", "raw_result.json",
                    "ml2dac_artifact_report.json", "labels.npy", "run_log.txt"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="arcs_circles_rings")
    ap.add_argument("--subfamily", default=None)
    ap.add_argument("--repo-root", default=REPO_DEFAULT)
    ap.add_argument("--master-root", required=True)
    ap.add_argument("--mkr-variant-dir", required=True)
    ap.add_argument("--variant", default="ML2DAC_Original_InDomain")
    ap.add_argument("--expect-candidates", type=int, default=7)
    ap.add_argument("--config-root", default=CONFIG_ROOT_DEFAULT)
    args = ap.parse_args(argv)

    if args.repo_root not in sys.path:
        sys.path.insert(0, args.repo_root)
    import pandas as pd
    from experiments.external_baselines.ML2DAC.indomain.mkr import InDomainMKR
    from experiments.external_baselines.ML2DAC.indomain.dataset import process_dataset_indomain, variant_dir_for

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

    mkr = InDomainMKR(args.mkr_variant_dir, str(master / "metafeatures.parquet"))
    print(f"[smoke] MKR loaded: {mkr.classifier_type}, train_splits={mkr.train_split_ids}, "
          f"candidate_cvis={mkr.candidate_cvis}")

    result = process_dataset_indomain(
        ds["dataset_folder"], split_id=1, record_types=["1d_x", "1d_y", "2d"],
        repo_root=args.repo_root, mkr=mkr, config_root=args.config_root, overwrite=True,
        variant=args.variant)

    vdir = variant_dir_for(Path(ds["dataset_folder"]), 1, args.variant)
    checks = {"mkr_loaded": mkr.classifier is not None,
              "candidate_cvi_count": len(mkr.candidate_cvis),
              "candidate_count_matches_expected": len(mkr.candidate_cvis) == args.expect_candidates,
              "dataset_summary_written": (vdir / "dataset_summary.json").exists(),
              "readme_written": (vdir / "README_ML2DAC_RUN.md").exists(),
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
            "status": r.get("status"), "selected_cvi": r.get("selected_cvi"),
            "selected_cvi_source": (r.get("raw_ml2dac_result") or raw).get("selected_cvi_source")
                                   or (r.get("details") or {}).get("selected_cvi_source"),
            "selected_cvi_present": bool(r.get("selected_cvi")),
            "warmstart_count": r.get("warmstart_count"),
            "metrics_keys": sorted((r.get("metrics") or {}).keys()),
            "n_configs_evaluated": raw.get("n_configs_evaluated"),
            "total_le_50": (raw.get("n_configs_evaluated") or 0) <= 50,
            "k_based_within_2_5": len(kbad) == 0,
        }
    checks["per_record"] = per
    checks["all_records_files_present"] = all(v["all_files_present"] for v in per.values())
    checks["all_selected_cvi_present"] = all(v["selected_cvi_present"] for v in per.values() if v["status"] == "success")
    checks["selected_cvi_sources"] = sorted({v["selected_cvi_source"] for v in per.values() if v["status"] == "success"})
    checks["all_k_within_2_5"] = all(v["k_based_within_2_5"] for v in per.values())
    checks["all_total_le_50"] = all(v["total_le_50"] for v in per.values())
    # metrics schema must include the full original set
    required_metrics = {"ari", "nmi", "ami", "v_measure", "homogeneity", "completeness",
                        "fowlkes_mallows", "purity", "selected_k", "true_k", "exact_k_match",
                        "n_predicted_clusters_excluding_noise", "n_noise_points"}
    checks["metrics_schema_complete"] = all(
        required_metrics.issubset(set(v["metrics_keys"])) for v in per.values() if v["status"] == "success")

    passed = all(checks[k] for k in ["mkr_loaded", "candidate_count_matches_expected",
                                     "dataset_summary_written", "readme_written",
                                     "best_view_ari_computed", "all_records_files_present",
                                     "all_selected_cvi_present", "all_k_within_2_5",
                                     "all_total_le_50", "metrics_schema_complete"])
    print(json.dumps(checks, indent=2, default=str))
    print(f"\nSMOKE TEST: {'PASS' if passed else 'FAIL'}")
    print(f"best_record_type={result.get('best_record_type')} best_ari={result.get('best_ari')}")
    print(f"output: {vdir}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
