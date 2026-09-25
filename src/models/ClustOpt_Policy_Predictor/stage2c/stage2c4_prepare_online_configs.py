"""Stage 2C-4A: materialise the 20 twin configs and inject U_{-s} per dataset.

Two artefacts are produced, and nothing else:

1. **Twenty ClustOpt experiment configs** under
   ``models/ClustOpt/configs/experiments/stage2c4/``. Each is one (regime,
   weighting) twin. The clustering search space is copied BYTE-FOR-BYTE from the
   corresponding frozen production config, the Optuna contract is the frozen
   Stage-2B5 one (50 trials, 150 s), and the ONLY field that differs between a
   RAW and a SOFTMAX twin is ``cvi.params.weighting``.

2. **The injected out-of-fold utility vectors**. For every sampled dataset in
   split ``s`` the Stage-2A1 single-exclusion prediction U_{-s} is written beside
   the dataset's own utility folder as ``oof_utility_stage2c4_<source>.json``.
   The vector is a PREDICTION from models that never saw split ``s``; it is fed
   through the ``oracle_dynamic`` resolver purely because that resolver is the
   one that reads a utility vector from a file (``real_utility_path``). The true
   oracle vector is never read, and each injected file records the numeric
   distance to it so the distinction is auditable rather than asserted.

Deliberate choice, recorded here because it is the one place this stage differs
from the historical Split-1 online runs: the twins use ``seed='auto'``
(``base_seed=42``), the frozen Stage-1 per-(dataset, view) derivation, which is
INDEPENDENT of the arm. RAW and SOFTMAX therefore start from identical sampler
randomness, so the measured ``delta_mode`` is attributable to the objective
rather than to the draw. The historical Split-1 runs were unseeded; that does not
affect Stage-2C4B, which is a lookup-only replay over those existing results.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E402,E501
    _ext, ensure_dir, read_json, write_csv_atomic, write_json_atomic,
)
from models.Clustering_Repository_Builder.experiments.experiment_execution.seeding import (  # noqa: E402,E501
    DEFAULT_BASE_SEED, derive_seed,
)
from models.ClustOpt_Policy_Predictor.stage2c import stage2c4_common as C  # noqa: E402,E501

CONFIG_ROOT_REL = "models/ClustOpt/configs/experiments"
#: Where each regime's frozen production config lives; the search space and the
#: generic CVI params are copied from it so they cannot drift.
SOURCE_CONFIG = {
    "mlp": "phaseD/regressor_top5_raw",
    "knn": "phaseE2_knn/knn_top5_raw",
}
ORACLE_REL = "utility/%s/utility_vector.json"


def _search_space(repo: Path, view: str) -> Dict[str, Any]:
    """The clustering search space, asserted identical across every source."""
    spaces = []
    for sub in SOURCE_CONFIG.values():
        d = json.loads((repo / CONFIG_ROOT_REL / sub / ("%s.json" % view))
                       .read_text("utf-8"))
        spaces.append(d)
    key = json.dumps(spaces[0]["search_space"], sort_keys=True)
    for d in spaces[1:]:
        if json.dumps(d["search_space"], sort_keys=True) != key:
            raise AssertionError("production search spaces disagree for %s" % view)
    return spaces[0]["search_space"], spaces[0].get("params_generic")


def build_config(repo: Path, regime_id: str, mode: str, view: str
                 ) -> Dict[str, Any]:
    src = C.source_of(regime_id)
    space, _ = _search_space(repo, view)
    generic = json.loads(
        (repo / CONFIG_ROOT_REL / SOURCE_CONFIG[src] / ("%s.json" % view))
        .read_text("utf-8"))["cvi"].get("params_generic")
    return {
        "search_space": copy.deepcopy(space),
        "search_algorithm": {"type": "optuna", "params": {
            "n_trials": C.N_TRIALS, "timeout": C.TIMEOUT_SEC,
            "seed": "auto", "base_seed": C.BASE_SEED}},
        "cvi": {
            "type": "oracle_dynamic",
            "params": {
                "top_k": C.top_k_of(regime_id),
                "weighting": C.WEIGHTING_IMPL[mode],
                "softmax_temperature": C.SOFTMAX_TEMPERATURE,
                "metric_name_prefix": "utility__",
                "utility_source": C.injected_utility_rel("<view_id>", src),
                "fallback_on_error": False,
                "fallback_metrics": {"silhouette": 1.0},
            },
            "params_generic": copy.deepcopy(generic),
        },
        "phaseD": {
            "phase": "ClustOpt_Stage2C4A_Online_Weighting_Mode",
            "method_name": C.method_name(regime_id, mode),
            "regime_id": regime_id,
            "weighting_arm": mode,
            "utility_source": src,
            "metric_ranking_source": src,
            "dynamic_k_source": ("predicted" if C.k_mode_of(regime_id) == "dynamic"
                                 else "none"),
            "weighting_mode": ("raw_normalized" if mode == "RAW"
                               else "softmax_t05"),
            "softmax_temperature": (None if mode == "RAW"
                                    else C.SOFTMAX_TEMPERATURE),
            "top_k_config": C.top_k_of(regime_id),
            "regressor_path": None,
            "real_utility_path": C.injected_utility_rel("<view_id>", src),
            "real_utility_semantics": ("Stage-2A1 single-exclusion OOF "
                                       "PREDICTION U_{-s}; not the oracle "
                                       "utility vector"),
            "optuna_trials_requested": C.N_TRIALS,
            "optuna_timeout_sec": C.TIMEOUT_SEC,
            "early_stopping_enabled": False,
        },
    }


def twin_diff(a: Dict[str, Any], b: Dict[str, Any], path: str = "") -> List[str]:
    """Every leaf key at which two twin configs differ."""
    out: List[str] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            out.extend(twin_diff(a.get(k), b.get(k), "%s.%s" % (path, k)))
    elif a != b:
        out.append(path.lstrip("."))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    p.add_argument("--n", type=int, default=12)
    args = p.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    out = ensure_dir(repo / C.OUT_REL)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
    t0 = time.time()
    print("=" * 78)
    print("STAGE 2C-4A  PREPARE TWIN CONFIGS + INJECT U_{-s}")
    print("=" * 78, flush=True)

    align = C.assert_registry_alignment()
    sample = C.load_sample(repo, n=args.n)
    rows = C.scientific_rows(sample)
    print("  sample hash %s verified | %d rows, %d datasets, n<=%d"
          % (C.FROZEN_SAMPLE_HASH, len(rows), rows["dataset_id"].nunique(),
             args.n), flush=True)

    # ---- 1. twenty twin configs -------------------------------------------
    croot = ensure_dir(repo / CONFIG_ROOT_REL / C.CONFIG_SUBDIR)
    cfg_rows: List[Dict[str, Any]] = []
    for rid in C.RERANKER_IDS:
        pair: Dict[str, Dict[str, Any]] = {}
        for mode in C.MODES:
            name = C.method_name(rid, mode)
            d = ensure_dir(croot / name)
            for v in C.VIEWS:
                cfg = build_config(repo, rid, mode, v)
                write_json_atomic(d / ("%s.json" % v), cfg)
                if v == "x_only":
                    pair[mode] = cfg
            cfg_rows.append({
                "regime_id": rid, "weighting_arm": mode, "method_name": name,
                "config_dir": str(d), "top_k": str(C.top_k_of(rid)),
                "weighting_impl": C.WEIGHTING_IMPL[mode],
                "n_trials": C.N_TRIALS, "timeout_sec": C.TIMEOUT_SEC,
                "seed_policy": "auto/base_seed=%d" % C.BASE_SEED})
        diff = twin_diff(pair["RAW"], pair["SOFTMAX"])
        expected = {"cvi.params.weighting", "phaseD.method_name",
                    "phaseD.weighting_arm", "phaseD.weighting_mode",
                    "phaseD.softmax_temperature"}
        if set(diff) != expected:
            raise AssertionError("%s twins differ at %s (expected %s)"
                                 % (rid, sorted(diff), sorted(expected)))
    CFG = pd.DataFrame(cfg_rows)
    write_csv_atomic(out / "stage2c4_twin_config_inventory.csv", CFG)
    print("  %d twin configs written; only cvi.params.weighting differs "
          "within a pair" % len(CFG), flush=True)

    # ---- 2. inject U_{-s} --------------------------------------------------
    names = C.metric_names(repo)
    inj: List[Dict[str, Any]] = []
    for s, g in rows.groupby("split_id"):
        s = int(s)
        idx = C.load_oof_utilities(repo, s, names)
        for _, r in (g[["dataset_id", "view_id", "family_id", "subfamily_id"]]
                     .drop_duplicates().iterrows()):
            dd = C.dataset_dir(repo, r["family_id"], r["subfamily_id"],
                               r["dataset_id"])
            if not Path(_ext(dd)).is_dir():
                raise SystemExit("dataset folder missing: %s" % dd)
            oracle = read_json(dd / (ORACLE_REL % r["view_id"])) or {}
            ovec = oracle.get("utilities", oracle)
            for src in ("mlp", "knn"):
                vec = idx.get(src, r["dataset_id"], r["view_id"])
                if len(vec) != 60 or not np.isfinite(vec).all():
                    raise SystemExit("bad U_{-s} for %s/%s/%s"
                                     % (r["dataset_id"], r["view_id"], src))
                payload = {
                    "utilities": {"utility__%s" % m: float(v)
                                  for m, v in zip(names, vec)},
                    "provenance": {
                        "stage": "2C-4A", "kind": "OOF_PREDICTED_UTILITY",
                        "is_oracle": False,
                        "held_out_split": s,
                        "training_splits": [x for x in C.OUTER_SPLITS if x != s],
                        "source": ("%s/holdout_split_%02d/%s/oof_predictions."
                                   "csv.gz" % (C.OOF_UTILITY_REL, s, src)),
                        "artifact_id": idx.artifact_id,
                        "utility_predictor": src,
                        "metric_order_hash": None,
                        "vector_hash": C.vector_hash(vec),
                        "split_1_used": False,
                    },
                }
                f = dd / C.injected_utility_rel(r["view_id"], src)
                ensure_dir(f.parent)
                write_json_atomic(f, payload)
                ov = np.asarray([float(ovec.get("utility__%s" % m, np.nan))
                                 for m in names], dtype=float)
                same = bool(np.isfinite(ov).all()
                            and np.allclose(ov, vec, atol=1e-12))
                inj.append({
                    "split_id": s, "dataset_id": r["dataset_id"],
                    "view_id": r["view_id"], "family_id": r["family_id"],
                    "subfamily_id": r["subfamily_id"], "utility_source": src,
                    "injected_rel_path": C.injected_utility_rel(r["view_id"], src),
                    "injected_abs_path": str(f),
                    "vector_hash": C.vector_hash(vec),
                    "vector_min": float(vec.min()), "vector_max": float(vec.max()),
                    "oracle_available": bool(np.isfinite(ov).all()),
                    "max_abs_diff_vs_oracle": (float(np.max(np.abs(ov - vec)))
                                               if np.isfinite(ov).all()
                                               else float("nan")),
                    "spearman_vs_oracle": (
                        float(pd.Series(ov).corr(pd.Series(vec), method="spearman"))
                        if np.isfinite(ov).all() else float("nan")),
                    "identical_to_oracle": same,
                    "heldout_split_excluded": True,
                    "artifact_id": idx.artifact_id,
                })
        print("   split %02d: %d dataset/view injections"
              % (s, 2 * len(g[["dataset_id", "view_id"]].drop_duplicates())),
              flush=True)
    INJ = pd.DataFrame(inj)
    if bool(INJ["identical_to_oracle"].any()):
        raise SystemExit("an injected vector is byte-identical to the oracle "
                         "vector; refusing to proceed")
    write_csv_atomic(out / "oof_utility_injection_manifest.csv", INJ)

    # ---- 3. pair inventory -------------------------------------------------
    inv: List[Dict[str, Any]] = []
    for _, r in rows.iterrows():
        dd = C.dataset_dir(repo, r["family_id"], r["subfamily_id"],
                           r["dataset_id"])
        seed = derive_seed(C.BASE_SEED, r["dataset_id"], r["view_id"])
        for mode in C.MODES:
            m = C.method_name(r["regime_id"], mode)
            inv.append({
                "split_id": int(r["split_id"]), "rank": int(r["rank"]),
                "dataset_id": r["dataset_id"], "view_id": r["view_id"],
                "family_id": r["family_id"], "subfamily_id": r["subfamily_id"],
                "regime_id": r["regime_id"], "utility_source": r["source"],
                "k_mode": r["k_mode"], "weighting_arm": mode,
                "method_name": m, "dataset_dir": str(dd),
                "run_dir": str(C.online_run_dir(dd, int(r["split_id"]), m,
                                                r["view_id"])),
                "search_seed": int(seed),
                "in_fallback_n8": bool(r["in_fallback_n8"]),
                "row_id": r["row_id"]})
    INV = pd.DataFrame(inv)
    write_csv_atomic(out / "online_pair_inventory.csv", INV)

    seeds = INV.groupby(["dataset_id", "view_id"])["search_seed"].nunique()
    if int(seeds.max()) != 1:
        raise AssertionError("twins do not share a search seed")

    man = {
        "stage": C.STAGE, "sample_hash": C.FROZEN_SAMPLE_HASH,
        "n_per_split": int(args.n),
        "sample_regenerated": False,
        "registry_alignment": align,
        "n_twin_configs": int(len(CFG)),
        "config_root": str(croot),
        "search_contract": {"sampler": "Optuna TPE", "n_trials": C.N_TRIALS,
                            "timeout_sec": C.TIMEOUT_SEC,
                            "seed": "auto", "base_seed": C.BASE_SEED,
                            "seed_derivation": "seeding.derive_seed(base, "
                                               "dataset_id, view_id) -- arm "
                                               "independent, so twins match",
                            "source": "frozen Stage-2B5 online semantics "
                                      "(50 trials / 150 s) as recorded in "
                                      "prepare_split1_evaluation.timeout_"
                                      "semantics and the phaseD configs"},
        "search_space_source": SOURCE_CONFIG,
        "twin_difference": ["cvi.params.weighting"],
        "utility_injection": {
            "resolver": "oracle_dynamic (the file-reading resolver)",
            "semantics": "Stage-2A1 single-exclusion OOF PREDICTION U_{-s}",
            "is_oracle": False,
            "n_files": int(len(INJ)),
            "n_identical_to_oracle": int(INJ["identical_to_oracle"].sum()),
            "median_max_abs_diff_vs_oracle": float(
                INJ["max_abs_diff_vs_oracle"].median()),
            "median_spearman_vs_oracle": float(INJ["spearman_vs_oracle"].median()),
        },
        "split_dir_suffix": C.SPLIT_DIR_SUFFIX,
        "expected_rows": int(len(rows)),
        "expected_searches": int(2 * len(rows)),
        "split_1_used": False,
        "source_commit": commit, "stage2c4p_commit": C.STAGE2C4P_COMMIT,
        "runtime_sec": time.time() - t0,
    }
    write_json_atomic(out / "online_pair_preparation_manifest.json", man)
    print("  injected %d U_{-s} files (median |diff| vs oracle %.4f, median "
          "spearman %.3f)"
          % (len(INJ), INJ["max_abs_diff_vs_oracle"].median(),
             INJ["spearman_vs_oracle"].median()), flush=True)
    print("  inventory: %d searches over %d scientific rows"
          % (len(INV), len(rows)), flush=True)
    print("  done in %.1fs" % (time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
