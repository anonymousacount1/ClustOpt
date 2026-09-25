"""Tables C+D -- the deployable observability layer, built once per slate.

The 10 canonical regimes x 48,039 slates = 480,390 mask rows. Each row carries:

  * the 60-dim availability mask (which CVIs a deployed search would have),
  * the ranked selected metric indices (padded to 10 with -1),
  * BOTH weight vectors -- RAW and SOFTMAX_T05 -- over those same metrics.

Tables C and D are deliberately one physical table. A RAW policy and a SOFTMAX
policy sharing (source, K) select an identical metric set, because ``select_top_k``
depends only on the utility vector and K; only the weighting differs. Storing them
separately would duplicate 480,390 mask rows to carry a second weight column, and
duplicating the candidate matrix per policy would cost 30.7 M rows. Neither is
done: 20 policy contexts are addressed as (regime row, weight column).

Utilities are the Stage-2A-1 single-exclusion OOF vectors -- the FINAL-TRAINING
source. The nested-CV manifest is a source MAPPING only and materialises nothing.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E402,E501
    _ext, ensure_dir, write_csv_atomic, write_json_atomic,
)
from models.ClustOpt_Candidate_Reranker.data import candidate_schema as CS  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.data import slate_schema as SS  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.protocol import cv_protocol as CV  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.protocol import masking_regimes as MR  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.protocol import policy_contexts as PC  # noqa: E402,E501

OUT_REL = ("results_analysis/clustopt_candidate_reranker/"
           "stage2b0_data_protocol_audit")
GZ = {"index": False, "compression": "gzip"}


def build_mask_manifest(repo: Path) -> Dict[str, Any]:
    """480,390 mask rows + coverage diagnostics, from single-exclusion OOF."""
    metric_names = CS.load_metric_names(repo)
    ctx = SS.load_slate_context(repo)
    ident = ctx["ident"]
    util = {s: SS.utility_matrix(ctx["util"], s, metric_names)
            for s in MR.SOURCES}

    n = len(ident)
    regimes = MR.all_regimes()
    ds = ident["dataset_id"].to_numpy()
    sp = ident["split_id"].to_numpy()
    vw = ident["view_id"].to_numpy()
    hold = ident["oof_heldout_split"].to_numpy()

    frames: List[pd.DataFrame] = []
    freq = {r["regime_id"]: np.zeros(60, dtype=np.int64) for r in regimes}
    kvals: Dict[str, List[int]] = {r["regime_id"]: [] for r in regimes}
    mask_store: Dict[str, np.ndarray] = {}

    for r in regimes:
        src, km, rid = r["utility_source"], r["k_mode"], r["regime_id"]
        U = util[src]
        M = np.zeros((n, 60), dtype=np.int8)
        idx = np.full((n, MR.MAX_SELECTED), -1, dtype=np.int16)
        wr = np.zeros((n, MR.MAX_SELECTED), dtype=np.float32)
        wsm = np.zeros((n, MR.MAX_SELECTED), dtype=np.float32)
        ak = np.zeros(n, dtype=np.int16)
        krel = np.full(n, -1, dtype=np.int16)

        t0 = time.time()
        for i in range(n):
            res = MR.resolve_regime(U[i], metric_names, km)
            M[i] = res["mask"]
            idx[i] = MR.pad_indices(res["selected_indices"])
            wr[i] = MR.pad_weights(res["w_raw"])
            wsm[i] = MR.pad_weights(res["w_softmax"])
            ak[i] = res["actual_k"]
            if res["k_rel_raw"] is not None:
                krel[i] = int(res["k_rel_raw"])
        freq[rid] = M.sum(axis=0)
        kvals[rid] = ak.tolist()
        mask_store[rid] = M

        f = pd.DataFrame({
            "dataset_id": ds, "split_id": sp, "view_id": vw,
            "oof_heldout_split": hold, "regime_id": rid,
            "utility_source": src, "k_mode": km,
            "k_rule": r["k_rule"], "actual_k": ak, "k_rel_raw": krel,
        })
        for j in range(MR.MAX_SELECTED):
            f[f"sel_metric_{j:02d}"] = idx[:, j]
        for j in range(MR.MAX_SELECTED):
            f[f"w_raw_{j:02d}"] = wr[:, j]
        for j in range(MR.MAX_SELECTED):
            f[f"w_soft_{j:02d}"] = wsm[:, j]
        for j, m in enumerate(metric_names):
            f[f"mask__{m}"] = M[:, j]
        frames.append(f)
        print(f"   {rid:<16} mean K {ak.mean():6.3f}  "
              f"metrics used {int((M.sum(axis=0) > 0).sum()):2d}/60  "
              f"({time.time()-t0:.0f}s)", flush=True)

    manifest = pd.concat(frames, ignore_index=True)
    return {"manifest": manifest, "freq": freq, "kvals": kvals,
            "masks": mask_store, "metric_names": metric_names,
            "regimes": regimes, "ident": ident}


def mask_diagnostics(built: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Coverage, selection frequency, MLP/KNN overlap, Dynamic-K distribution."""
    metric_names = built["metric_names"]
    masks, freq = built["masks"], built["freq"]
    n = len(built["ident"])

    cov = []
    for r in built["regimes"]:
        rid = r["regime_id"]
        M = masks[rid]
        k = np.asarray(built["kvals"][rid])
        cov.append({
            "regime_id": rid, "utility_source": r["utility_source"],
            "k_mode": r["k_mode"], "n_slates": n,
            "mean_actual_k": float(k.mean()), "median_actual_k": float(np.median(k)),
            "min_actual_k": int(k.min()), "max_actual_k": int(k.max()),
            "n_distinct_metrics_ever_selected": int((freq[rid] > 0).sum()),
            "frac_metrics_ever_selected": float((freq[rid] > 0).mean()),
            "mean_metric_coverage": float(M.mean()),
            "most_selected_metric": metric_names[int(np.argmax(freq[rid]))],
            "most_selected_share": float(freq[rid].max() / n),
        })
    cov_df = pd.DataFrame(cov)

    rows = []
    for rid, f in freq.items():
        for j, m in enumerate(metric_names):
            rows.append({"regime_id": rid, "metric": m,
                         "n_selected": int(f[j]), "share": float(f[j] / n)})
    fsel = pd.DataFrame(rows)

    ov = []
    for km in MR.K_MODES:
        A = masks[MR.regime_id("mlp", km)]
        B = masks[MR.regime_id("knn", km)]
        inter = (A & B).sum(axis=1)
        union = ((A | B) > 0).sum(axis=1)
        ov.append({
            "k_mode": km, "mean_intersection": float(inter.mean()),
            "mean_union": float(union.mean()),
            "mean_jaccard": float(np.mean(inter / np.maximum(union, 1))),
            "frac_slates_identical_mask": float(np.mean((A == B).all(axis=1))),
            "frac_slates_disjoint_mask": float(np.mean(inter == 0)),
        })
    # Overlap across K levels within a source is nested by construction
    # (Top-K sets grow with K); recorded to make that explicit rather than assumed.
    for src in MR.SOURCES:
        for a, b in (("top1", "top3"), ("top3", "top5"), ("top5", "top10")):
            A, B = masks[MR.regime_id(src, a)], masks[MR.regime_id(src, b)]
            ov.append({
                "k_mode": f"{src}:{a}_in_{b}",
                "mean_intersection": float((A & B).sum(axis=1).mean()),
                "mean_union": float(((A | B) > 0).sum(axis=1).mean()),
                "mean_jaccard": float(np.mean((A & B).sum(axis=1)
                                              / np.maximum(((A | B) > 0).sum(axis=1), 1))),
                "frac_slates_identical_mask": float(np.mean((A == B).all(axis=1))),
                "frac_slates_disjoint_mask": float(
                    np.mean((A & B).sum(axis=1) == 0)),
            })
    ov_df = pd.DataFrame(ov)

    dk = []
    for src in MR.SOURCES:
        k = np.asarray(built["kvals"][MR.regime_id(src, "dynamic")])
        vc = pd.Series(k).value_counts().sort_index()
        for kv, cnt in vc.items():
            dk.append({"utility_source": src, "dynamic_k": int(kv),
                       "n_slates": int(cnt), "share": float(cnt / len(k))})
    dk_df = pd.DataFrame(dk)
    return {"coverage": cov_df, "frequency": fsel, "overlap": ov_df,
            "dynamic_k": dk_df}


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    args = p.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    out = ensure_dir(repo / OUT_REL)

    print("=" * 78)
    print("STAGE 2B-0  --  DEPLOYABLE MASK / POLICY-CONTEXT LAYER (Tables C+D)")
    print("=" * 78, flush=True)

    regimes = MR.all_regimes()
    ctxs = PC.all_policy_contexts()
    write_json_atomic(out / "masking_regime_registry.json", {
        "n_regimes": len(regimes), "regimes": regimes,
        "registry_hash": MR.regime_registry_hash(regimes),
        "sources": list(MR.SOURCES), "k_modes": list(MR.K_MODES),
        "max_selected": MR.MAX_SELECTED,
        "mask_semantics": "1 = CVI computed by a deployed search; 0 = never "
                          "evaluated. Masked values are NaN, never 0.",
    })
    write_json_atomic(out / "policy_context_registry.json", {
        "n_policy_contexts": len(ctxs), "contexts": ctxs,
        "policy_context_hash": PC.policy_context_hash(ctxs),
        "registry_semantics_hash": PC.registry_semantics_hash(),
        "factorisation": "2 sources x 5 K regimes x 2 weighting modes = 20",
        "no_duplication": "RAW and SOFTMAX share the mask row; only the weight "
                          "column differs.",
    })
    print(f"  {len(regimes)} regimes | {len(ctxs)} policy contexts "
          f"over {len({c['regime_id'] for c in ctxs})} regimes", flush=True)

    print("\n-- resolving masks from single-exclusion OOF utilities --", flush=True)
    built = build_mask_manifest(repo)
    man = built["manifest"]
    man.to_csv(_ext(out / "final_training_mask_manifest.csv.gz"), **GZ)
    print(f"\n  mask manifest {man.shape}", flush=True)

    diag = mask_diagnostics(built)
    write_csv_atomic(out / "mask_coverage_summary.csv", diag["coverage"])
    diag["frequency"].to_csv(_ext(out / "metric_selection_frequency.csv.gz"), **GZ)
    write_csv_atomic(out / "mlp_knn_mask_overlap.csv", diag["overlap"])
    write_csv_atomic(out / "dynamic_k_distribution.csv", diag["dynamic_k"])

    nested = pd.DataFrame(CV.nested_cv_plan())
    nested.to_csv(_ext(out / "nested_cv_mask_source_manifest.csv.gz"), **GZ)
    write_json_atomic(out / "cv_reuse_plan.json", {
        "atomic_unit": "dataset (all views, candidates, regimes, policy "
                       "contexts of a dataset stay in one fold)",
        "final_training_mode": CV.FINAL_TRAINING_MODE,
        "nested_cv_mode": CV.NESTED_CV_MODE,
        "single_exclusion_root": CV.SINGLE_EXCLUSION_REL,
        "pair_exclusion_root": CV.PAIR_EXCLUSION_REL,
        "utility_derived_blocks": list(CV.UTILITY_DERIVED_BLOCKS),
        "model_free_blocks": list(CV.MODEL_FREE_BLOCKS),
        "n_train_bindings": int((nested["role"] == "TRAIN").sum()),
        "n_test_bindings": int((nested["role"] == "TEST").sum()),
        "n_distinct_pair_sources": int(
            nested.loc[nested["role"] == "TRAIN", "pair_id"].nunique()),
        "new_utility_training_required": False,
        "note": "Raw candidate blocks (meta-features, configuration, partition "
                "descriptors, offline CVIs) are model-free and need no "
                "cross-fitting; only utility-derived blocks do.",
    })
    print(f"  nested manifest {nested.shape}, "
          f"{nested.loc[nested['role']=='TRAIN','pair_id'].nunique()} distinct "
          f"pair sources", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
