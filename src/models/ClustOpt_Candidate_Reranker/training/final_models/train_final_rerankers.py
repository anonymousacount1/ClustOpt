"""Train and persist the ten FINAL dedicated contextual rerankers.

Population: ALL Splits 2-16 (16,013 datasets, 48,039 slates), eligible candidates
only. Utility context and observability masks come from the Stage-2B0
final-training package, which is built on Stage-2A-1 SINGLE-exclusion OOF -- for a
dataset in split t the source was trained without t. Pair-exclusion belongs to
nested model selection and is deliberately not used here.

Split 1 is never opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E402,E501
    _ext, ensure_dir, path_exists, read_json, write_csv_atomic, write_json_atomic,
)
from models.ClustOpt_Candidate_Reranker.data import candidate_eligibility as CE  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.data import slate_schema as SS  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training import feature_builder as FB  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training import model_registry as MR  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training import train_fold as TF  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training.context_ablation.run_context_ablation import (  # noqa: E402,E501
    WorkerLock,
)
from models.ClustOpt_Candidate_Reranker.training.run_masking_feasibility import (  # noqa: E402,E501
    CandidateStore, _memory_probes,
)
from models.ClustOpt_Candidate_Reranker.training.unified_masking import (  # noqa: E402,E501
    unified_feature_builder as UFB,
)
from . import policy_mapping as PM

B0_REL = ("results_analysis/clustopt_candidate_reranker/"
          "stage2b0_data_protocol_audit")
OUT_REL = ("results_analysis/clustopt_candidate_reranker/"
           "stage2b5a_final_models_and_split1_preparation")
DEV_SPLITS: Tuple[int, ...] = tuple(range(2, 17))
GZ = {"index": False, "compression": "gzip"}
LOCK_NAME = "stage2b5a_worker.pid"
EXPECTED_SLATES = 48039
EXPECTED_DATASETS = 16013


def load_final_masks(repo: Path, store: CandidateStore
                     ) -> Dict[str, Dict[str, np.ndarray]]:
    """Per-regime slate-aligned mask + actual K from the frozen 2B-0 package.

    The manifest is the authoritative final-training package: single-exclusion
    OOF, one row per (slate, regime). It is read rather than recomputed so the
    final models are bound to exactly the artifact Stage 2B-0 froze.
    """
    man = pd.read_csv(_ext(repo / B0_REL / "final_training_mask_manifest.csv.gz"))
    if (man["split_id"] == 1).any():
        raise SystemExit("final mask manifest contains Split 1")
    if not (man["oof_heldout_split"] == man["split_id"]).all():
        raise SystemExit("mask manifest is not single-exclusion")
    mcols = [f"mask__{m}" for m in store.metric_names]
    key = store.slate_key.to_numpy()
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for rid, g in man.groupby("regime_id", sort=False):
        k = (g["dataset_id"].astype(str) + "|" + g["view_id"]).to_numpy()
        pos = pd.Index(key).get_indexer(k)
        if (pos < 0).any():
            raise AssertionError(f"{rid}: mask rows do not align to slates")
        order = np.argsort(pos)
        if len(g) != len(key):
            raise AssertionError(f"{rid}: {len(g)} mask rows for {len(key)} slates")
        out[rid] = {"mask": g[mcols].to_numpy(np.int8)[order],
                    "actual_k": g["actual_k"].to_numpy(np.int16)[order]}
    if len(out) != PM.N_RERANKERS:
        raise AssertionError(f"expected 10 regimes, got {len(out)}")
    return out


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(_ext(p), "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--rerankers", nargs="*", default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    repo = Path(args.repo_root).resolve()
    out = ensure_dir(repo / OUT_REL)
    models_root = ensure_dir(out / "final_models")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    mem, sysmem = _memory_probes()

    print("=" * 78)
    print("STAGE 2B-5A  --  FINAL RERANKER TRAINING (Splits 2-16)")
    print("=" * 78, flush=True)
    print(f"  context C2 LOCAL_UTIL ({UFB.EXPECTED_DIM} dims) | "
          f"model {MR.config_hash()} | target T2_SLATE_REGRET | "
          f"eligibility {CE.PREDICATE_ID}", flush=True)

    with WorkerLock(out / LOCK_NAME):
        store = CandidateStore(repo)
        if (store.split_id == 1).any():
            raise SystemExit("candidate store contains Split 1")
        if len(store.slate_start) != EXPECTED_SLATES:
            raise SystemExit(f"expected {EXPECTED_SLATES} slates")

        ctx = SS.load_slate_context(repo)
        util_all = np.hstack([
            SS.utility_matrix(ctx["util"], "mlp", store.metric_names),
            SS.utility_matrix(ctx["util"], "knn", store.metric_names)]
        ).astype(np.float32)
        ukey = (ctx["ident"]["dataset_id"].astype(str) + "|"
                + ctx["ident"]["view_id"]).to_numpy()
        upos = pd.Index(ukey).get_indexer(store.slate_key.to_numpy())
        if (upos < 0).any():
            raise SystemExit("utility context does not align to slates")
        util_all = util_all[upos]
        if not np.isfinite(util_all).all():
            raise SystemExit("non-finite value in the final utility context")

        masks = load_final_masks(repo, store)
        mapping = PM.build_mapping()
        write_json_atomic(out / "policy_to_reranker_mapping.json", {
            "n_policies": PM.N_POLICIES, "n_rerankers": PM.N_RERANKERS,
            "mapping": mapping, "mapping_hash": PM.mapping_hash(mapping),
            "registry_semantics_hash": PM.registry_semantics_hash(),
            "rationale": "RAW and SOFTMAX sharing (source, K) share a reranker "
                         "because their candidate-CVI mask is identical and the "
                         "C2 interface exposes no weighting mode, weights or J. "
                         "They remain SEPARATE online runs.",
        })

        rows = store.rows_for_slates(np.arange(len(store.slate_start)))
        rows = rows[store.eligible[rows]]
        slate_pos = pd.Series(np.arange(len(store.slate_start)),
                              index=np.arange(len(store.slate_start))
                              ).reindex(store.slate_of_row[rows]
                                        ).to_numpy().astype(np.int64)
        y = TF.target_regret(store.ari[rows], store.oracle_row[rows])
        write_csv_atomic(out / "final_training_population.csv", pd.DataFrame([{
            "n_datasets": int(pd.unique(store.dataset_id).size),
            "n_slates": int(len(store.slate_start)),
            "n_candidate_rows": int(len(store.ari)),
            "n_eligible_training_rows": int(len(rows)),
            "splits": ",".join(str(s) for s in DEV_SPLITS),
            "split_1_used": False,
            "utility_semantics": "stage2a1 single-exclusion OOF (per-split held out)",
            "mask_source": "stage2b0 final_training_mask_manifest.csv.gz",
        }]))
        print(f"  training population: {len(rows)} eligible rows over "
              f"{len(store.slate_start)} slates", flush=True)

        targets = [r for r in PM.RERANKER_IDS
                   if not args.rerankers or r in args.rerankers]
        if args.preflight:
            targets = targets[:1]
        summ: List[Dict[str, Any]] = []
        for rid in targets:
            src, km = rid.split("_", 1)
            regime = f"{src.lower()}__{km.lower()}"
            mdir = ensure_dir(models_root / rid)
            if not args.overwrite and path_exists(mdir / "model_manifest.json") \
                    and not args.preflight:
                m = read_json(mdir / "model_manifest.json")
                if m and m.get("status") == "complete":
                    summ.append(m)
                    print(f"   [{rid:<12}] already complete, skipped", flush=True)
                    continue
            import gc
            gc.collect()
            before = mem()
            t0 = time.perf_counter()
            X, n_common = UFB.build_unified_X(
                view_codes=store.view_code[rows], cat_rows=store.cat_row[rows],
                cat_matrix=store.cat_matrix, partition=store.partition[rows],
                cvi=store.cvi[rows], cvi_valid=store.cvi_valid[rows],
                observability=masks[regime]["mask"],
                actual_k=masks[regime]["actual_k"],
                source_code=np.full(len(store.slate_start),
                                    FB.SOURCES.index(src.lower()), np.int64),
                kmode_code=np.full(len(store.slate_start),
                                   FB.K_MODES.index(km.lower()), np.int64),
                slate_pos=slate_pos, util=util_all)
            build_sec = time.perf_counter() - t0
            UFB.assert_no_hidden_leakage(X, masks[regime]["mask"], slate_pos,
                                         n_common)
            after = mem()
            names = UFB.feature_names(store.metric_names, store.cat_names,
                                      [f"mlp__{m}" for m in store.metric_names]
                                      + [f"knn__{m}" for m in store.metric_names])
            t1 = time.perf_counter()
            model = MR.build_model()
            model.fit(X, y)
            fit_sec = time.perf_counter() - t1
            peak = max(after, mem())
            if args.preflight:
                payload = {"reranker_id": rid, "n_rows": int(len(X)),
                           "n_features": int(X.shape[1]), "dtype": str(X.dtype),
                           "input_matrix_mb": round(X.nbytes / 2 ** 20, 1),
                           "rss_before_mb": round(before, 1),
                           "rss_after_build_mb": round(after, 1),
                           "rss_peak_mb": round(peak, 1),
                           "build_sec": round(build_sec, 1),
                           "fit_sec": round(fit_sec, 1)}
                total, avail = sysmem()
                write_json_atomic(out / "resource_preflight.json", {
                    "probe": payload, "system_total_mb": round(total, 1),
                    "system_available_mb": round(avail, 1),
                    "safe": bool(avail - peak > 300),
                    "recommended_concurrency": 1, "float32_used": True,
                    "features_reduced": "none", "model_changed": "none",
                    "estimated_total_hours": round(fit_sec * 10 / 3600, 2)})
                print(f"   PREFLIGHT {rid}: {X.shape} X="
                      f"{X.nbytes/2**20:.0f}MB peak={peak:.0f}MB "
                      f"fit={fit_sec:.0f}s | avail {avail:.0f}MB", flush=True)
                del X, model
                gc.collect()
                return 0

            # ---- persist + reload equivalence --------------------------------
            import joblib
            mp = mdir / "model.joblib"
            joblib.dump(model, _ext(mp), compress=3)
            probe = X[:2000]
            pred_before = model.predict(probe)
            reloaded = joblib.load(_ext(mp))
            pred_after = reloaded.predict(probe)
            if not np.array_equal(pred_before, pred_after):
                raise AssertionError(f"{rid}: predictions changed after reload")
            del X, model, reloaded
            gc.collect()

            man = {
                "status": "complete", "reranker_id": rid,
                "observability_regime": regime,
                "utility_source": src.lower(), "k_mode": km.lower(),
                "training_population": "all Splits 2-16 fixed32 slates",
                "split_ids": list(DEV_SPLITS), "split_1_used": False,
                "n_training_rows": int(len(rows)),
                "n_training_slates": int(len(store.slate_start)),
                "n_datasets": EXPECTED_DATASETS,
                "feature_schema": UFB.CONDITION,
                "feature_schema_hash": UFB.CFB.schema_hash(names)
                if hasattr(UFB, "CFB") else UFB.CONDITION,
                "n_features": UFB.EXPECTED_DIM,
                "target": "T2_SLATE_REGRET",
                "target_definition": "regret = max ARI over eligible candidates "
                                     "in the slate minus candidate ARI",
                "utility_semantics": "stage2a1 single-exclusion OOF; for a "
                                     "dataset in split t the source was trained "
                                     "without t",
                "mask_source": "stage2b0 final_training_mask_manifest.csv.gz",
                "model_config": MR.HGBR_CONFIG,
                "model_config_hash": MR.config_hash(),
                "eligibility": CE.PREDICATE_ID,
                "eligibility_hash": CE.definition_hash(),
                "random_seed": MR.GLOBAL_SEED,
                "sklearn_version": MR.API_NOTES["sklearn_version"],
                "python": platform.python_version(),
                "fit_timestamp_unix": time.time(),
                "fit_duration_sec": fit_sec,
                "build_duration_sec": build_sec,
                "peak_rss_mb": round(peak, 1),
                "model_file": "model.joblib",
                "model_file_bytes": int(Path(_ext(mp)).stat().st_size),
                "model_file_sha256_16": file_sha256(mp),
                "reload_equivalence_checked": True,
                "reload_equivalence_rows": int(len(probe)),
                "source_commit": commit,
            }
            write_json_atomic(mdir / "model_manifest.json", man)
            summ.append(man)
            print(f"   [{rid:<12}] rows={len(rows)} fit={fit_sec:.0f}s "
                  f"peak={peak:.0f}MB size="
                  f"{man['model_file_bytes']/2**20:.1f}MB "
                  f"sha={man['model_file_sha256_16']}", flush=True)

        if summ and not args.preflight:
            write_csv_atomic(out / "final_model_training_summary.csv",
                             pd.DataFrame([{k: v for k, v in m.items()
                                            if not isinstance(v, (dict, list))}
                                           for m in summ]))
        print(f"\n  {len(summ)} final rerankers ready", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
