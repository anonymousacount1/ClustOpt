"""Stage 2B-1 orchestrator: 11 information conditions x 15 outer folds = 165 fits.

Loading strategy: the candidate blocks (1,537,248 x 60 CVI float32 + masks +
targets, ~520 MB resident) are read ONCE and indexed by slate. Re-reading them
per fold would cost 15 passes over 250 MB of gzip for no benefit.

Mask sourcing per outer fold s reuses the Stage-2A-3B0 outer-fold tables verbatim:
``train_pair_utility_features`` (pair-exclusion, trained on S \\ {s,t}) for the
14 training splits, ``test_single_oof_utility_features`` (single-exclusion for s)
for the held-out split. **No utility model is trained here.**

Every (condition, fold) unit is resumable and validated by hash, never by file
existence alone.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from models.Clustering_Repository_Builder.experiments.experiment_execution.output_writer import (  # noqa: E402,E501
    _ext, ensure_dir, path_exists, read_json, write_csv_atomic, write_json_atomic,
)
from models.ClustOpt_Candidate_Reranker.data import candidate_eligibility as CE  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.data import candidate_schema as CS  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.protocol import masking_regimes as MR_REG  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training import evaluate_fold as EV  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training import feature_builder as FB  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training import model_registry as MR  # noqa: E402,E501
from models.ClustOpt_Candidate_Reranker.training import train_fold as TF  # noqa: E402,E501

B0_REL = ("results_analysis/clustopt_candidate_reranker/"
          "stage2b0_data_protocol_audit")
NESTED_REL = ("results_analysis/clustopt_policy_predictor/"
              "stage2a3b0_nested_crossfit")
OUT_REL = ("results_analysis/clustopt_candidate_reranker/"
           "stage2b1_masking_feasibility")
OUTER_SPLITS: Tuple[int, ...] = tuple(range(2, 17))
GZ = {"index": False, "compression": "gzip"}
FULL_PRED_CONDITIONS = {"RICH_60", "MLP_TOP5"}


# --------------------------------------------------------------------- store
class CandidateStore:
    """All candidate blocks in memory, plus a slate index."""

    def __init__(self, repo: Path) -> None:
        b0 = repo / B0_REL
        t0 = time.time()
        ident = pd.read_csv(_ext(b0 / "candidate_identifiers.csv.gz"))
        self.dataset_id = ident["dataset_id"].to_numpy()
        self.split_id = ident["split_id"].to_numpy(np.int16)
        self.view_id = ident["view_id"].to_numpy()
        self.candidate_index = ident["candidate_index"].to_numpy(np.int16)
        self.metric_names = CS.load_metric_names(repo)

        self.cvi = pd.read_csv(_ext(b0 / "candidate_cvi_features.csv.gz")
                               ).to_numpy(np.float32)
        self.cvi_valid = pd.read_csv(
            _ext(b0 / "candidate_cvi_validity_mask.csv.gz")).to_numpy(np.int8)
        part = pd.read_csv(_ext(b0 / "candidate_partition_descriptors.csv.gz"))
        self.partition = part[list(FB.PARTITION_COLS)].to_numpy(np.float32)
        self.eligible = CE.is_eligible(part["part__valid"].to_numpy())
        tgt = pd.read_csv(_ext(b0 / "candidate_targets.csv.gz"),
                          usecols=["candidate_ari", "slate_regret",
                                   "slate_oracle_ari"])
        self.ari = tgt["candidate_ari"].to_numpy(np.float64)
        self.regret = tgt["slate_regret"].to_numpy(np.float32)
        self.oracle_row = tgt["slate_oracle_ari"].to_numpy(np.float64)

        cat = pd.read_csv(_ext(b0 / "candidate_configuration.csv.gz"))
        self.cat_matrix, self.cat_names = FB.build_catalogue_matrix(cat)
        self.cat_offset = FB.catalogue_index(cat)
        self.view_code = np.array(
            [FB.VIEW_IDS.index(v) for v in self.view_id], dtype=np.int64)
        self.cat_row = np.array(
            [self.cat_offset[v] for v in self.view_id], dtype=np.int64
        ) + self.candidate_index.astype(np.int64)

        self._build_slate_index()
        print(f"  store loaded: {len(self.ari)} candidate rows, "
              f"{len(self.slate_start)} slates ({time.time()-t0:.0f}s)", flush=True)

    def _build_slate_index(self) -> None:
        key = pd.Series(self.dataset_id).astype(str) + "|" + pd.Series(self.view_id)
        codes, uniques = pd.factorize(key, sort=False)
        starts = np.flatnonzero(np.r_[True, np.diff(codes) != 0])
        sizes = np.diff(np.r_[starts, len(codes)])
        if not (sizes == 32).all():
            raise AssertionError("non-contiguous or non-32 slate detected")
        self.slate_start = starts
        self.slate_size = sizes
        self.slate_dataset = self.dataset_id[starts]
        self.slate_view = self.view_id[starts]
        self.slate_split = self.split_id[starts]
        self.slate_key = pd.Index(uniques)
        self.slate_of_row = codes.astype(np.int32)

    def rows_for_slates(self, slate_idx: np.ndarray) -> np.ndarray:
        return np.concatenate([np.arange(self.slate_start[i],
                                         self.slate_start[i] + self.slate_size[i])
                               for i in slate_idx])


# ------------------------------------------------------------- mask sourcing
def load_fold_utilities(repo: Path, s: int, metric_names: Sequence[str]
                        ) -> Dict[str, Any]:
    """Pair-exclusion utilities for training slates, single-exclusion for test."""
    d = repo / NESTED_REL / "outer_folds" / f"outer_{s:02d}"
    tr_i = pd.read_csv(_ext(d / "train_identifiers.csv.gz"))
    tr_u = pd.read_csv(_ext(d / "train_pair_utility_features.csv.gz"))
    te_i = pd.read_csv(_ext(d / "test_identifiers.csv.gz"))
    te_u = pd.read_csv(_ext(d / "test_single_oof_utility_features.csv.gz"))
    # Train rows are labelled by ``row_split`` (their own split) and carry the
    # pair-exclusion provenance; test rows keep ``split_id`` and single-exclusion.
    if (te_i["split_id"] != s).any():
        raise AssertionError(f"outer fold {s}: test identifiers carry other splits")
    if (tr_i["row_split"] == s).any():
        raise AssertionError(f"outer fold {s}: held-out split present in training")
    if set(tr_i["source_kind"].unique()) != {"pair_exclusion"}:
        raise AssertionError(f"outer fold {s}: training utilities are not "
                             f"pair-exclusion")
    if set(te_i["source_kind"].unique()) != {"single_exclusion"}:
        raise AssertionError(f"outer fold {s}: test utilities are not "
                             f"single-exclusion")
    if (tr_i["outer_split"] != s).any():
        raise AssertionError(f"outer fold {s}: training rows bound to another fold")

    def block(u: pd.DataFrame, src: str) -> np.ndarray:
        pre = "oof_mlp_utility__" if src == "mlp" else "oof_knn_utility__"
        cols = [f"{pre}{m}" for m in metric_names]
        miss = [c for c in cols if c not in u.columns]
        if miss:
            raise AssertionError(f"{len(miss)} {src} utility columns missing")
        return u[cols].to_numpy(np.float64)

    return {
        "train_key": (tr_i["dataset_id"].astype(str) + "|"
                      + tr_i["view_id"]).to_numpy(),
        "test_key": (te_i["dataset_id"].astype(str) + "|"
                     + te_i["view_id"]).to_numpy(),
        "train_util": {s2: block(tr_u, s2) for s2 in ("mlp", "knn")},
        "test_util": {s2: block(te_u, s2) for s2 in ("mlp", "knn")},
        "train_source": "stage2a3b0 pair-exclusion (S \\ {s,t})",
        "test_source": "stage2a1 single-exclusion (S \\ {s})",
    }


def resolve_masks(util: np.ndarray, metric_names: Sequence[str], k_mode: str
                  ) -> Dict[str, np.ndarray]:
    """Per-slate observability mask, actual K, ranked indices and both weightings."""
    n = len(util)
    M = np.zeros((n, 60), dtype=np.int8)
    idx = np.full((n, MR_REG.MAX_SELECTED), -1, dtype=np.int64)
    wr = np.zeros((n, MR_REG.MAX_SELECTED), dtype=np.float64)
    ws = np.zeros((n, MR_REG.MAX_SELECTED), dtype=np.float64)
    ak = np.zeros(n, dtype=np.int16)
    for i in range(n):
        r = MR_REG.resolve_regime(util[i], metric_names, k_mode)
        M[i] = r["mask"]
        idx[i] = MR_REG.pad_indices(r["selected_indices"])
        wr[i] = MR_REG.pad_weights(r["w_raw"])
        ws[i] = MR_REG.pad_weights(r["w_softmax"])
        ak[i] = r["actual_k"]
    return {"mask": M, "sel_idx": idx, "w_raw": wr, "w_soft": ws, "actual_k": ak}


# ------------------------------------------------------------------ resume
def unit_dir(out: Path, cond: str, s: int) -> Path:
    return out / "outer_fold_predictions" / cond / f"fold_{s:02d}"


def unit_is_complete(out: Path, cond: str, s: int, uhash: str) -> bool:
    m = read_json(unit_dir(out, cond, s) / "unit_manifest.json")
    if not m:
        return False
    return (m.get("status") == "complete" and m.get("unit_hash") == uhash
            and bool(m.get("leakage_checks_pass")))


# -------------------------------------------------------------------- main
def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--conditions", nargs="*", default=None)
    p.add_argument("--splits", type=int, nargs="*", default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    repo = Path(args.repo_root).resolve()
    out = ensure_dir(repo / OUT_REL)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()

    print("=" * 78)
    print("STAGE 2B-1  --  CANDIDATE RERANKER MASKING FEASIBILITY")
    print("=" * 78, flush=True)
    print(f"  model {MR.MODEL_FAMILY} ({MR.config_hash()}) | "
          f"target T2_SLATE_REGRET | eligibility {CE.PREDICATE_ID}", flush=True)

    store = CandidateStore(repo)
    conditions = FB.all_conditions()
    if args.conditions:
        conditions = [c for c in conditions if c["condition_id"] in args.conditions]
    splits = args.splits or list(OUTER_SPLITS)
    if 1 in splits:
        raise SystemExit("Split 1 is forbidden")

    reg_rows = []
    for c in conditions:
        reg_rows.append({**c, "model_family": MR.MODEL_FAMILY,
                         "model_config_hash": MR.config_hash(),
                         "target": "T2_SLATE_REGRET"})
    write_csv_atomic(out / "configuration_registry.csv", pd.DataFrame(reg_rows))

    # ---- production-J equivalence check (deterministic sample) -------------
    ver = _verify_j(repo, store)
    write_json_atomic(out / "production_j_equivalence.json", ver)
    print(f"  production J equivalence: {ver['equivalent']} "
          f"(max dev {ver['max_abs_deviation']:.3e} over {ver['n_checked']} "
          f"slates)", flush=True)
    if not ver["equivalent"]:
        raise SystemExit("production J reconstruction is not exact -- stopping")

    if args.preflight:
        return _preflight(repo, store, out)

    t_all = time.time()
    n_done = n_skip = 0
    for s in splits:
        fu = load_fold_utilities(repo, s, store.metric_names)
        tr_slates = np.flatnonzero(store.slate_split != s)
        te_slates = np.flatnonzero(store.slate_split == s)
        key = store.slate_key.to_numpy()
        tr_pos = pd.Index(key[tr_slates]).get_indexer(fu["train_key"])
        te_pos = pd.Index(key[te_slates]).get_indexer(fu["test_key"])
        if (tr_pos < 0).any() or (te_pos < 0).any():
            raise AssertionError(f"fold {s}: utility identifiers do not align")
        tr_order = np.argsort(tr_pos)
        te_order = np.argsort(te_pos)

        # Training uses ELIGIBLE candidates only -- see train_fold.target_regret.
        tr_rows_all = store.rows_for_slates(tr_slates)
        tr_rows = tr_rows_all[store.eligible[tr_rows_all]]
        te_rows = store.rows_for_slates(te_slates)
        te_starts = np.cumsum(np.r_[0, store.slate_size[te_slates]])[:-1]
        te_sizes = store.slate_size[te_slates]

        y_tr = TF.target_regret(store.ari[tr_rows], store.oracle_row[tr_rows])
        # position of each retained training row within the fold's slate list
        tr_slate_rank = pd.Series(np.arange(len(tr_slates)),
                                  index=tr_slates).reindex(
            store.slate_of_row[tr_rows]).to_numpy()
        tr_slate_pos = tr_slate_rank.astype(np.int64)
        ari_te = store.ari[te_rows]
        orc_te = store.oracle_row[te_rows]
        elig_te = store.eligible[te_rows]
        ds_te = store.dataset_id[te_rows]
        vw_te = store.view_id[te_rows]

        for cond in conditions:
            cid = cond["condition_id"]
            t0 = time.time()
            if cond["kind"] == "RICH":
                masks_tr = masks_te = None
                mask_source = "none (RICH exposes all 60)"
            else:
                km, src = cond["k_mode"], cond["utility_source"]
                mtr = resolve_masks(fu["train_util"][src][tr_order],
                                    store.metric_names, km)
                mte = resolve_masks(fu["test_util"][src][te_order],
                                    store.metric_names, km)
                masks_tr, masks_te = mtr, mte
                mask_source = f"{fu['train_source']} / {fu['test_source']}"

            X_tr, names, n_common = _build(store, cond, tr_rows, masks_tr,
                                           tr_slate_pos)
            uhash = MR.unit_hash(cid, s, FB.feature_schema_hash(names),
                                 CE.definition_hash(), mask_source)
            if not args.overwrite and unit_is_complete(out, cid, s, uhash):
                n_skip += 1
                del X_tr
                continue

            X_te, _, _ = _build(store, cond, te_rows, masks_te)
            if cond["kind"] != "RICH":
                obs_rows = np.repeat(masks_te["mask"], 32, axis=0)
                FB.assert_no_hidden_leakage(X_te, store.cvi[te_rows], obs_rows,
                                            n_common)

            res = TF.run_unit(condition=cond, X_tr=X_tr, y_tr=y_tr, X_te=X_te,
                              eligible_te=elig_te, slate_starts=te_starts,
                              slate_sizes=te_sizes, ari_te=ari_te,
                              oracle_te=orc_te)
            del X_tr, X_te

            sel_ari = res["selected_ari"]
            slate_orc = orc_te[te_starts]
            slate_ds = ds_te[te_starts]
            slate_vw = vw_te[te_starts]
            m = EV.slate_metrics(sel_ari, slate_orc, slate_vw, slate_ds)

            base: Dict[str, Any] = {}
            if cond["kind"] != "RICH":
                base = _native_baselines(store, te_rows, te_starts, te_sizes,
                                         masks_te, elig_te, ari_te, slate_orc,
                                         slate_ds, slate_vw)
            stored = _write_unit(out, cid, s, uhash, cond, res, m, base, names,
                                 mask_source, commit, store, te_rows, te_starts,
                                 slate_ds, slate_vw, sel_ari, slate_orc, elig_te)
            n_done += 1
            print(f"   [{cid:<12} fold {s:02d}] BV={m['mean_bestview_selected_ari']:.4f} "
                  f"reg={m['mean_oracle_regret']:.4f} "
                  f"exact={m['frac_exact_oracle']:.3f}"
                  + (f" recRAW={stored['recovery_raw']:+.3f}"
                     f" recSOFT={stored['recovery_softmax']:+.3f}"
                     if base else "")
                  + f"  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n  {n_done} units executed, {n_skip} skipped, "
          f"{time.time()-t_all:.0f}s", flush=True)
    return 0


def _build(store: CandidateStore, cond: Dict[str, Any], rows: np.ndarray,
           masks: Optional[Dict[str, np.ndarray]],
           slate_pos: Optional[np.ndarray] = None):
    """``slate_pos`` maps each row to its position in the mask array.

    A plain ``repeat(..., 32)`` only works when every candidate of every slate is
    present. Training keeps eligible candidates only, so the mapping is explicit.
    """
    obs = ak = None
    if cond["kind"] != "RICH":
        pos = (slate_pos if slate_pos is not None
               else np.repeat(np.arange(len(masks["mask"])), 32))
        obs = masks["mask"][pos]
        ak = masks["actual_k"][pos]
    return FB.build_X(cond, view_codes=store.view_code[rows],
                      cat_rows=store.cat_row[rows], cat_matrix=store.cat_matrix,
                      cat_names=store.cat_names, partition=store.partition[rows],
                      cvi=store.cvi[rows], cvi_valid=store.cvi_valid[rows],
                      metric_names=store.metric_names,
                      observability=obs, actual_k=ak)


def _native_baselines(store, te_rows, te_starts, te_sizes, masks, elig, ari,
                      slate_orc, slate_ds, slate_vw) -> Dict[str, Any]:
    """argmax J under RAW and SOFTMAX with production semantics."""
    cvi = store.cvi[te_rows].astype(np.float64)
    idx = np.repeat(masks["sel_idx"], 32, axis=0)
    out: Dict[str, Any] = {}
    for lab, wkey in (("raw", "w_raw"), ("softmax", "w_soft")):
        w = np.repeat(masks[wkey], 32, axis=0)
        j = EV.native_j_scores(cvi, idx, w)
        pick = EV.select_per_slate(j, elig, te_starts, te_sizes, minimise=False)
        b_ari = ari[pick]
        bm = EV.slate_metrics(b_ari, slate_orc, slate_vw, slate_ds)
        out[f"b_{lab}_view"] = bm["mean_selected_ari"]
        out[f"b_{lab}_bestview"] = bm["mean_bestview_selected_ari"]
        out[f"b_{lab}_regret"] = bm["mean_oracle_regret"]
        out[f"b_{lab}_selected_ari"] = b_ari
        out[f"b_{lab}_pick"] = pick
    return out


def _write_unit(out, cid, s, uhash, cond, res, m, base, names, mask_source,
                commit, store, te_rows, te_starts, slate_ds, slate_vw,
                sel_ari, slate_orc, elig_te) -> None:
    d = ensure_dir(unit_dir(out, cid, s))
    sl = pd.DataFrame({
        "dataset_id": slate_ds, "split_id": s, "view_id": slate_vw,
        "selected_candidate_index": store.candidate_index[te_rows][res["chosen"]],
        "selected_ari": sel_ari, "candidate_oracle_ari": slate_orc,
        "oracle_regret": slate_orc - sel_ari,
    })
    if base:
        sl["b_raw_ari"] = base["b_raw_selected_ari"]
        sl["b_softmax_ari"] = base["b_softmax_selected_ari"]
        sl["b_raw_candidate_index"] = store.candidate_index[te_rows][base["b_raw_pick"]]
        sl["b_softmax_candidate_index"] = store.candidate_index[te_rows][
            base["b_softmax_pick"]]
    sl.to_csv(_ext(d / "slate_selections.csv.gz"), **GZ)

    if cid in FULL_PRED_CONDITIONS:
        pd.DataFrame({
            "dataset_id": store.dataset_id[te_rows], "split_id": s,
            "view_id": store.view_id[te_rows],
            "candidate_index": store.candidate_index[te_rows],
            "eligible": elig_te.astype(np.int8),
            "predicted_regret": res["pred"],
            "candidate_ari": store.ari[te_rows],
            "candidate_oracle_ari": store.oracle_row[te_rows],
        }).to_csv(_ext(d / "candidate_predictions.csv.gz"), **GZ)

    metrics = {**m, **{k: v for k, v in base.items()
                       if not isinstance(v, np.ndarray)}}
    if base:
        metrics["recovery_raw"] = EV.recovery(
            m["mean_bestview_selected_ari"], base["b_raw_bestview"],
            m["mean_bestview_candidate_oracle"])
        metrics["recovery_softmax"] = EV.recovery(
            m["mean_bestview_selected_ari"], base["b_softmax_bestview"],
            m["mean_bestview_candidate_oracle"])
        metrics["beats_raw"] = bool(m["mean_bestview_selected_ari"]
                                    > base["b_raw_bestview"])
        metrics["beats_softmax"] = bool(m["mean_bestview_selected_ari"]
                                        > base["b_softmax_bestview"])
    write_json_atomic(d / "unit_manifest.json", {
        "status": "complete", "unit_hash": uhash, "condition_id": cid,
        "outer_split": int(s), "kind": cond["kind"],
        "utility_source": cond["utility_source"], "k_mode": cond["k_mode"],
        "feature_schema_hash": FB.feature_schema_hash(names),
        "n_features": len(names), "eligibility_hash": CE.definition_hash(),
        "mask_source": mask_source, "model_config_hash": MR.config_hash(),
        "target": "T2_SLATE_REGRET", "fit_info": res["fit_info"],
        "metrics": metrics, "leakage_checks_pass": True,
        "split_1_used": False, "source_commit": commit,
        "full_predictions_stored": cid in FULL_PRED_CONDITIONS,
    })
    return metrics


def _verify_j(repo: Path, store: CandidateStore) -> Dict[str, Any]:
    """Vectorised J vs production score_candidates on a deterministic sample."""
    rng_slates = list(range(0, len(store.slate_start), 4001))[:12]
    fu = load_fold_utilities(repo, 2, store.metric_names)
    worst, n = 0.0, 0
    for i in rng_slates:
        a = store.slate_start[i]
        block = store.cvi[a:a + 32].astype(np.float64)
        r = MR_REG.resolve_regime(fu["test_util"]["mlp"][0], store.metric_names,
                                  "top5")
        v = EV.verify_production_equivalence(
            block, [], store.metric_names, r["selected_names"], r["w_raw"])
        worst = max(worst, v["max_abs_deviation"])
        n += 1
    return {"n_checked": n, "max_abs_deviation": worst,
            "equivalent": bool(worst <= 1e-12),
            "reference": "offline_candidate_execution.score_candidates"}


def _memory_probes():
    """Process RSS and system memory via the Windows API.

    ``psutil`` is not installed in ``.venv_clustopt`` and the environment lock
    must not change for a resource probe, so this reads psapi/kernel32 through
    ``ctypes`` instead.
    """
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]

    class MEMSTAT(ctypes.Structure):
        _fields_ = [("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    # Explicit arg/restypes matter here: GetCurrentProcess returns a pseudo-handle
    # that ctypes truncates to int without them, which makes the call fail
    # silently and report 0.
    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi")
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,
                                           ctypes.POINTER(PMC), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

    def rss_mb() -> float:
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(),
                                          ctypes.byref(c), c.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return c.WorkingSetSize / 2 ** 20

    def system_mb():
        m = MEMSTAT()
        m.dwLength = ctypes.sizeof(MEMSTAT)
        if not k32.GlobalMemoryStatusEx(ctypes.byref(m)):
            raise OSError("GlobalMemoryStatusEx failed")
        return m.ullTotalPhys / 2 ** 20, m.ullAvailPhys / 2 ** 20

    return rss_mb, system_mb


def _preflight(repo: Path, store: CandidateStore, out: Path) -> int:
    """Resource-only probe on ONE outer-training population. No test metrics."""
    import gc
    mem, sysmem = _memory_probes()

    s = 2
    fu = load_fold_utilities(repo, s, store.metric_names)
    tr_slates = np.flatnonzero(store.slate_split != s)
    key = store.slate_key.to_numpy()
    tr_pos = pd.Index(key[tr_slates]).get_indexer(fu["train_key"])
    tr_order = np.argsort(tr_pos)
    tr_rows_all = store.rows_for_slates(tr_slates)
    tr_rows = tr_rows_all[store.eligible[tr_rows_all]]
    y = TF.target_regret(store.ari[tr_rows], store.oracle_row[tr_rows])
    tr_slate_rank = pd.Series(np.arange(len(tr_slates)), index=tr_slates
                              ).reindex(store.slate_of_row[tr_rows]
                                        ).to_numpy().astype(np.int64)

    rows = []
    for cond in (FB.all_conditions()[0],
                 [c for c in FB.all_conditions()
                  if c["condition_id"] == "MLP_TOP5"][0]):
        gc.collect()
        base_mem = mem()
        t0 = time.time()
        masks = None
        if cond["kind"] != "RICH":
            masks = resolve_masks(fu["train_util"]["mlp"][tr_order],
                                  store.metric_names, "top5")
        X, names, _ = _build(store, cond, tr_rows, masks, tr_slate_rank)
        build_sec = time.time() - t0
        x_mb = X.nbytes / 2 ** 20
        after_build = mem()
        t1 = time.time()
        model = MR.build_model()
        model.fit(X, y)
        fit_sec = time.time() - t1
        peak = mem()
        t2 = time.time()
        model.predict(X[:100000])
        pred_sec = (time.time() - t2) * (len(X) / 100000)
        rows.append({
            "condition_id": cond["condition_id"], "n_rows": int(len(X)),
            "n_features": int(X.shape[1]), "dtype": str(X.dtype),
            "input_matrix_mb": round(x_mb, 1),
            "rss_before_mb": round(base_mem, 1),
            "rss_after_build_mb": round(after_build, 1),
            "rss_peak_fit_mb": round(peak, 1),
            "fit_delta_mb": round(peak - after_build, 1),
            "build_sec": round(build_sec, 1), "fit_sec": round(fit_sec, 1),
            "predict_full_sec_est": round(pred_sec, 1),
        })
        print(f"   {cond['condition_id']:<10} {X.shape} X={x_mb:.0f}MB "
              f"peak={peak:.0f}MB fit={fit_sec:.0f}s", flush=True)
        del X, model
        gc.collect()

    total, avail = sysmem()
    worst = max(r["rss_peak_fit_mb"] for r in rows)
    payload = {
        "model": MR.MODEL_FAMILY, "config": MR.HGBR_CONFIG,
        "api_notes": MR.API_NOTES, "probes": rows,
        "system_total_mb": round(total, 1), "system_available_mb": round(avail, 1),
        "worst_peak_rss_mb": worst,
        "safe_concurrency": 1 if not np.isfinite(avail) else
                            max(1, int(avail // max(worst, 1))),
        "recommended_concurrency": 1,
        "recommendation_reason": "sequential fits keep peak RSS at one model; "
                                 "wall time is not the binding constraint",
        "estimated_total_fit_hours": round(
            sum(r["fit_sec"] for r in rows) / 2 * 165 / 3600, 2),
        "float32_used": True, "feature_blocks_dropped": "none",
    }
    write_json_atomic(out / "resource_preflight.json", payload)
    print(f"\n  worst peak RSS {worst:.0f} MB | available {avail:.0f} MB | "
          f"est. total {payload['estimated_total_fit_hours']} h", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
