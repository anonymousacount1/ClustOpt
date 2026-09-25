"""Controlled benchmark: public entry point.

    python experiments/controlled_benchmark/run.py generate --family standard_clustering_blobs
    python experiments/controlled_benchmark/run.py features --subfamily-root <family>/subfamilies/<subfamily>
    python experiments/controlled_benchmark/run.py utilities -- <options>      # utility targets (candidate enumeration + ARI)
    python experiments/controlled_benchmark/run.py training-table -- <options> # unified training table
    python experiments/controlled_benchmark/run.py splits -- <options>         # the 16 dataset-level splits
    python experiments/controlled_benchmark/run.py train-neural -- <options>   # neural utility-profile predictor
    python experiments/controlled_benchmark/run.py train-knn -- <options>      # k-NN utility-profile predictor
    python experiments/controlled_benchmark/run.py arm --config configs/clustopt/neural_profile_top5/xy_2d.json \
                                                       --dataset-dir <dataset folder> --view xy_2d
    python experiments/controlled_benchmark/run.py batch -- <options>          # arms over one subfamily and split

Options after ``--`` are passed unchanged to the implementation (use ``-- --help``).
Generated data and run outputs are written under ``src/results_analysis/`` (not tracked).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))
from _dispatch import ROOT, SRC, run_module  # noqa: E402

MODULES = {
    "utilities": "models.Clustering_Repository_Builder.utility_generation.run_subfamily_utility_generation",
    "training-table": "models.Clustering_Repository_Builder.training_dataset_builder.build_training_dataset",
    "splits": "models.Clustering_Repository_Builder.experiments.data_splitting.run_create_splits",
    "train-neural": "models.metric_utility_mlp.run_train_split_holdout",
    "train-knn": "models.metric_utility_knn.run_phase_e1",
    "batch": "models.Clustering_Repository_Builder.experiments.experiment_execution.run_subfamily_experiments",
}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    passthrough = []
    if "--" in argv:
        i = argv.index("--")
        argv, passthrough = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="generate one family of the synthetic repository")
    g.add_argument("--family", required=True, help="name of a file in data/family_configs/ (without .json)")
    f = sub.add_parser("features", help="label-free meta-features for a generated subfamily")
    f.add_argument("--subfamily-root", required=True)
    f.add_argument("--max-datasets", type=int, default=None)
    a = sub.add_parser("arm", help="run one configuration on one dataset view")
    a.add_argument("--config", required=True)
    a.add_argument("--dataset-dir", required=True)
    a.add_argument("--view", default="xy_2d", choices=["x_only", "y_only", "xy_2d"])
    a.add_argument("--n-trials", type=int, default=None)
    for k in MODULES:
        sub.add_parser(k, help="pass-through (options after --)")
    args = ap.parse_args(argv)

    if args.cmd == "generate":
        cfg = ROOT / "data" / "family_configs" / ("%s.json" % args.family)
        return run_module("models.HYBRID_SCM.run_family_repository_generation", ["--family-config", str(cfg)] + passthrough)
    if args.cmd == "features":
        extra = ["--max-datasets", str(args.max_datasets)] if args.max_datasets else []
        return run_module("models.Clustering_Repository_Builder.features_extraction.run_extract_subfamily_features",
                          ["--subfamily-root", str(Path(args.subfamily_root).resolve())] + extra + passthrough)
    if args.cmd == "arm":
        from clustopt import search
        out = search.run(args.config, args.dataset_dir, args.view, n_trials=args.n_trials)
        print("selected indices:", list(out.selected_metrics))
        print("selected configuration:", out.selected_algorithm, out.selected_params)
        print("ARI vs ground truth: %.4f" % search.adjusted_rand(args.dataset_dir, out.y_pred, args.view))
        return 0
    if args.cmd == "batch" and "--repo-root" not in passthrough:
        passthrough = ["--repo-root", str(SRC)] + passthrough
    return run_module(MODULES[args.cmd], passthrough)


if __name__ == "__main__":
    raise SystemExit(main())
