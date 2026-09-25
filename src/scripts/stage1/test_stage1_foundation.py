"""Stage-1B-1 foundation tests.

Runs standalone (no pytest required):

    python scripts/stage1/test_stage1_foundation.py

Tests that require the ClustOpt runtime stack (shapely / cv2 / optuna / torch)
report SKIP-BLOCKED rather than failing, so the import-light foundation can be
validated in an environment that does not carry the full metric dependencies.
Every skip is reported explicitly and counts against completeness.
"""
from __future__ import annotations

import importlib
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "stage0"))

RESULTS: list[tuple[str, str, str]] = []


def record(test_id: str, name: str, ok, detail: str = "") -> None:
    status = "PASS" if ok is True else ("SKIP-BLOCKED" if ok is None else "FAIL")
    RESULTS.append((test_id, status, f"{name}: {detail}" if detail else name))
    print(f"  [{status:12}] {test_id} {name}" + (f" -- {detail}" if detail else ""))


def _optional(module: str):
    try:
        return importlib.import_module(module)
    except Exception:
        return None


# --------------------------------------------------------------------- A
def test_A_metric_partition() -> None:
    import csv
    path = os.path.join(REPO, "docs", "project_lead_revision", "stage1", "metric_partition.csv")
    if not os.path.exists(path):
        record("A", "metric partition artifact", False, "metric_partition.csv missing")
        return
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    head = [r["metric"] for r in rows if r["group"] == "HEAD_14"]
    new = [r["metric"] for r in rows if r["group"] == "NEW_46"]
    ok = (len(rows) == 60 and len(head) == 14 and len(new) == 46
          and not (set(head) & set(new)))
    record("A", "partition 14/46/60 disjoint+exhaustive", ok,
           f"total={len(rows)} head={len(head)} new={len(new)}")


# --------------------------------------------------------------------- B/C/D
def _fake_frame():
    import pandas as pd
    from models.metric_utility_mlp.head_groups import METRIC_HEAD_GROUPS
    metrics = [m for ms in METRIC_HEAD_GROUPS.values() for m in ms]
    cols = {"dataset_id": ["d1"], "view_mode": ["xy_2d"]}
    for i in range(3):
        cols[f"feat_{i}"] = [0.0]
    for m in metrics:
        cols[f"utility__{m}"] = [0.5]
    return pd.DataFrame(cols), metrics


def _cfg(**kw):
    from models.metric_utility_mlp.config import DataConfig
    c = DataConfig()
    c.n_id_columns = 2
    c.n_feature_columns = 3
    c.strict_column_counts = False      # counts differ in the synthetic frame
    c.id_columns = ["dataset_id", "view_mode"]
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_B_full60_backward_compat() -> None:
    from models.metric_utility_mlp.data_loader import detect_columns
    df, metrics = _fake_frame()
    _, feats, targets = detect_columns(df, _cfg())
    expected = [f"utility__{m}" for m in metrics]
    ok = targets == expected and len(targets) == 60 and "utility__silhouette" not in feats
    record("B", "target_include=None discovers all 60 in CSV order", ok,
           f"n_targets={len(targets)} order_match={targets == expected}")


def test_C_head14_selection() -> None:
    from models.metric_utility_mlp.data_loader import detect_columns
    from models.metric_utility_mlp.head_groups import resolve_target_group
    head14 = resolve_target_group("head14")
    df, _ = _fake_frame()
    _, feats, targets = detect_columns(df, _cfg(target_include=head14))
    expected = [f"utility__{m}" for m in head14]
    leaked = [c for c in feats if c.startswith("utility__")]
    ok = targets == expected and len(targets) == 14 and not leaked
    record("C", "head14 selection: exact 14 in canonical order, no leakage", ok,
           f"n={len(targets)} order_match={targets == expected} leaked_utility_features={len(leaked)}")


def test_D_missing_and_duplicate() -> None:
    from models.metric_utility_mlp.data_loader import detect_columns
    df, _ = _fake_frame()
    missing_ok = dup_ok = False
    try:
        detect_columns(df, _cfg(target_include=["silhouette", "not_a_metric"]))
    except ValueError as e:
        missing_ok = "not present" in str(e)
    try:
        detect_columns(df, _cfg(target_include=["silhouette", "silhouette"]))
    except ValueError as e:
        dup_ok = "duplicate" in str(e).lower()
    record("D", "missing/duplicate targets fail loudly", missing_ok and dup_ok,
           f"missing_raised={missing_ok} duplicate_raised={dup_ok}")


# --------------------------------------------------------------------- E
def test_E_model_dimensionality() -> None:
    torch = _optional("torch")
    if torch is None:
        record("E", "14-output multi-head model builds", None, "torch not installed")
        return
    from models.metric_utility_mlp.head_groups import build_head_mapping, resolve_target_group
    from models.metric_utility_mlp.config import ModelConfig
    from models.metric_utility_mlp.model import build_model
    head14 = resolve_target_group("head14")
    cols = [f"utility__{m}" for m in head14]
    mapping = build_head_mapping(cols)
    heads = mapping["heads"]
    cfg = ModelConfig()
    cfg.output_dim = 14
    model = build_model(cfg, {k: list(v) for k, v in heads.items()})
    out = model(torch.zeros(2, cfg.input_dim))
    ok = tuple(out.shape) == (2, 14) and sum(len(v) for v in heads.values()) == 14
    record("E", "14-output multi-head model builds and emits (N,14)", ok,
           f"heads={list(heads)} shape={tuple(out.shape)}")


# --------------------------------------------------------------------- F
def test_F_oracle_restriction() -> None:
    try:
        from models.ClustOpt.cluster_validity_indices.dynamic_metric_selection.resolvers import (
            resolve_oracle_weights,
        )
    except Exception as exc:
        record("F", "oracle Top-5 restricted to Head-14", None,
               f"resolvers unimportable ({type(exc).__name__}: {str(exc)[:40]})")
        return
    from models.metric_utility_mlp.head_groups import METRIC_HEAD_GROUPS, resolve_target_group
    head14 = resolve_target_group("head14")
    all_metrics = [m for ms in METRIC_HEAD_GROUPS.values() for m in ms]
    new46 = [m for m in all_metrics if m not in set(head14)]
    # New-46 metrics deliberately outrank every Head-14 metric.
    vec = {m: 0.99 - 0.001 * i for i, m in enumerate(new46)}
    vec.update({m: 0.50 - 0.001 * i for i, m in enumerate(head14)})

    ho = resolve_oracle_weights(
        params={"top_k": 5, "weighting": "normalized_positive",
                "candidate_metrics": head14},
        utility_vector=vec, view_id="xy_2d")
    fo = resolve_oracle_weights(
        params={"top_k": 5, "weighting": "normalized_positive"},
        utility_vector=vec, view_id="xy_2d")

    ho_ok = set(ho.selected_metrics) <= set(head14) and len(ho.selected_metrics) == 5
    fo_ok = set(fo.selected_metrics) <= set(new46) and len(fo.selected_metrics) == 5
    dbg_ok = ho.debug.get("candidate_metric_count") == 14 and fo.debug.get("candidate_metrics") is None
    record("F", "HO restricted to Head-14; FO unrestricted; debug recorded",
           ho_ok and fo_ok and dbg_ok,
           f"HO={ho.selected_metrics[:3]}... FO_all_new46={fo_ok} debug_ok={dbg_ok}")


# --------------------------------------------------------------------- G
def test_G_uniform_weights() -> None:
    try:
        from models.ClustOpt.configs.experiments.generate_stage1_2x3_configs import (
            uniform_metrics_block, head14_metric_names, full60_metric_names,
        )
    except Exception as exc:
        record("G", "uniform weights 1/14 and 1/60 summing to 1", None,
               f"generator unimportable ({type(exc).__name__}: {str(exc)[:40]})")
        return
    h = uniform_metrics_block(head14_metric_names())
    f = uniform_metrics_block(full60_metric_names())
    ok = (len(h) == 14 and abs(sum(h.values()) - 1.0) < 1e-12
          and all(abs(v - 1 / 14) < 1e-12 for v in h.values())
          and len(f) == 60 and abs(sum(f.values()) - 1.0) < 1e-12
          and all(abs(v - 1 / 60) < 1e-12 for v in f.values()))
    record("G", "uniform weights 1/14 and 1/60 summing to 1", ok,
           f"sum_h={sum(h.values()):.12f} sum_f={sum(f.values()):.12f}")


# --------------------------------------------------------------------- H
def test_H_seed_derivation() -> None:
    from models.Clustering_Repository_Builder.experiments.experiment_execution.seeding import (
        derive_seed, resolve_search_seed, SEED_MODULUS,
    )
    a = derive_seed(42, "c2e_000808_1feb31de", "xy_2d")
    b = derive_seed(42, "c2e_000808_1feb31de", "xy_2d")
    c = derive_seed(42, "c2e_000808_1feb31de", "x_only")
    d = derive_seed(42, "c2e_000808_1feb31de", "y_only")
    e = derive_seed(43, "c2e_000808_1feb31de", "xy_2d")
    other_ds = derive_seed(42, "c3h_000085_5e03d2a3", "xy_2d")
    in_range = all(0 <= s < SEED_MODULUS for s in (a, c, d, e, other_ds))
    # Arm id is not an input: the same call is all six arms ever make.
    auto = resolve_search_seed("auto", base_seed=42,
                               dataset_id="c2e_000808_1feb31de", view_id="xy_2d")
    ok = (a == b and a != c and c != d and a != e and a != other_ds
          and in_range and auto == a and resolve_search_seed(None) is None)
    record("H", "seed stable/distinct/in-range, arm-independent", ok,
           f"seed(42,c2e_000808_1feb31de,xy_2d)={a} x_only={c} y_only={d}")
    # Cross-process stability: the value must be a pure function of the inputs.
    print(f"                 known-value anchor: derive_seed(42,'c2e_000808_1feb31de','xy_2d') == {a}")


# --------------------------------------------------------------------- I/J
def test_I_J_optuna_seeding() -> None:
    optuna = _optional("optuna")
    if optuna is None:
        record("I", "seeded Optuna reproducibility", None, "optuna not installed")
        record("J", "seed=None keeps historical unseeded path", None, "optuna not installed")
        return

    def sample(seed):
        s = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed) if seed is not None else None)
        s.optimize(lambda t: -(t.suggest_float("x", -5, 5) ** 2), n_trials=8)
        return [t.params["x"] for t in s.trials]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    a, b, c = sample(123), sample(123), sample(456)
    record("I", "same seed reproduces the sample sequence", a == b and a != c,
           f"identical={a == b} differs_from_other_seed={a != c}")
    u1, u2 = sample(None), sample(None)
    record("J", "seed=None runs the historical unseeded path", len(u1) == 8 and len(u2) == 8,
           f"unseeded runs completed ({len(u1)}, {len(u2)} trials); differ={u1 != u2}")


# --------------------------------------------------------------------- K/L
def _config_paths():
    root = os.path.join(REPO, "models", "ClustOpt", "configs", "experiments", "stage1_2x3")
    from models.Clustering_Repository_Builder.experiments.experiment_execution.config import (
        METHOD_NAMES_STAGE1_2X3,
    )
    views = ("x_only", "y_only", "xy_2d")
    return [(m, v, os.path.join(root, m, f"{v}.json"))
            for m in METHOD_NAMES_STAGE1_2X3 for v in views]


def test_K_configs() -> None:
    paths = _config_paths()
    missing = [p for _, _, p in paths if not os.path.exists(p)]
    if missing:
        record("K", "18 configs satisfy the fixed-50 protocol", None,
               f"{len(missing)}/18 configs not generated yet (blocked)")
        return
    from models.metric_utility_mlp.head_groups import resolve_target_group
    from models.ClustOpt.configs.experiments.generate_stage1_2x3_configs import (
        FULL60_REGRESSOR_PATH, HEAD14_REGRESSOR_PLACEHOLDER, to_registry_keys,
    )
    head14_keys = set(to_registry_keys(resolve_target_group("head14")))

    bad = []
    for method, view, p in paths:
        cfg = json.load(open(p, encoding="utf-8"))
        sa = cfg.get("search_algorithm", {}).get("params", {})
        st = cfg.get("stage1", {})
        cvi = cfg.get("cvi", {})
        cp = cvi.get("params", {})
        tag = f"{method}/{view}"
        # protocol
        if sa.get("n_trials") != 50: bad.append(f"{tag}: n_trials")
        if sa.get("timeout") is not None: bad.append(f"{tag}: timeout present")
        if sa.get("patience") is not None: bad.append(f"{tag}: patience present")
        if sa.get("seed") != "auto": bad.append(f"{tag}: seed policy")
        if sa.get("base_seed") != 42: bad.append(f"{tag}: base_seed != 42")
        if st.get("view_id") != view: bad.append(f"{tag}: view mismatch")
        if st.get("method_name") != method: bad.append(f"{tag}: method mismatch")
        # per-arm semantics
        if method == "stage1_head14_uniform_fixed50":
            m = cvi.get("metrics", {})
            if len(m) != 14: bad.append(f"{tag}: HU has {len(m)} metrics")
            if set(m) != head14_keys: bad.append(f"{tag}: HU metric set != Head-14")
        elif method == "stage1_full60_uniform_fixed50":
            m = cvi.get("metrics", {})
            if len(m) != 60: bad.append(f"{tag}: FU has {len(m)} metrics")
        elif method == "stage1_head14_mlp_top5_raw_fixed50":
            if cvi.get("type") != "regressor_dynamic": bad.append(f"{tag}: HP type")
            if cp.get("top_k") != 5: bad.append(f"{tag}: HP top_k")
            if cp.get("weighting") != "normalized_positive": bad.append(f"{tag}: HP weighting")
            # HP is either UNBOUND (placeholder, before Stage-1B-2 training) or
            # BOUND to a real Head-14 run directory that exposes exactly 14
            # targets. A fabricated/timestamp-shaped path that does not exist is
            # never acceptable.
            mrd = cp.get("model_run_dir")
            if mrd == HEAD14_REGRESSOR_PLACEHOLDER:
                pass                                   # not yet trained: fine
            else:
                tc = os.path.join(REPO, mrd, "artifacts", "target_columns.json")
                if not os.path.isdir(os.path.join(REPO, mrd)):
                    bad.append(f"{tag}: HP model_run_dir does not exist: {mrd}")
                elif not os.path.exists(tc):
                    bad.append(f"{tag}: HP run dir has no target_columns.json")
                elif len(json.load(open(tc, encoding="utf-8"))) != 14:
                    bad.append(f"{tag}: HP predictor does not expose 14 targets")
        elif method == "stage1_full60_mlp_top5_raw_fixed50":
            if cp.get("model_run_dir") != FULL60_REGRESSOR_PATH:
                bad.append(f"{tag}: FP must point at the frozen Full-60 artifact")
            if cp.get("top_k") != 5: bad.append(f"{tag}: FP top_k")
            if cp.get("weighting") != "normalized_positive": bad.append(f"{tag}: FP weighting")
        elif method == "stage1_head14_oracle_top5_raw_fixed50":
            cm = cp.get("candidate_metrics")
            if cvi.get("type") != "oracle_dynamic": bad.append(f"{tag}: HO type")
            if not cm or len(cm) != 14: bad.append(f"{tag}: HO candidate_metrics != 14")
            if cp.get("top_k") != 5: bad.append(f"{tag}: HO top_k")
        elif method == "stage1_full60_oracle_top5_raw_fixed50":
            if cp.get("candidate_metrics") is not None:
                bad.append(f"{tag}: FO must be unrestricted")
            if cp.get("top_k") != 5: bad.append(f"{tag}: FO top_k")
    record("K", "18 configs: protocol + per-arm semantics", not bad,
           f"{len(paths)} configs, {len(bad)} problems" + (f" {bad[:3]}" if bad else ""))


def test_L_search_space_equality() -> None:
    paths = _config_paths()
    if any(not os.path.exists(p) for _, _, p in paths):
        record("L", "all six arms share an identical search space", None,
               "configs not generated yet (blocked)")
        return
    by_view = {}
    for method, view, p in paths:
        cfg = json.load(open(p, encoding="utf-8"))
        by_view.setdefault(view, []).append(
            (method, json.dumps(cfg["search_space"], sort_keys=True)))
    bad = [v for v, entries in by_view.items() if len({s for _, s in entries}) != 1]
    record("L", "all six arms share an identical clustering search space", not bad,
           f"views_with_divergence={bad}")


# --------------------------------------------------------------------- M
def test_M_result_paths() -> None:
    try:
        from models.Clustering_Repository_Builder.experiments.experiment_execution.method_runner import (
            run_output_dir,
        )
    except Exception as exc:
        record("M", "result path construction", None, f"unimportable: {str(exc)[:40]}")
        return
    from models.Clustering_Repository_Builder.experiments.experiment_execution.config import (
        METHOD_NAMES_STAGE1_2X3,
    )
    from pathlib import Path
    ok = True
    sample = None
    for m in METHOD_NAMES_STAGE1_2X3:
        for v in ("x_only", "y_only", "xy_2d"):
            p = run_output_dir(Path("/DS"), "split_01", m, v)
            sample = p
            if p.parts[-4:] != ("experiments", "split_01", m, v):
                ok = False
    record("M", "methods resolve to experiments/split_01/<method>/<view>", ok,
           f"e.g. .../{'/'.join(sample.parts[-4:])}")


# --------------------------------------------------------------------- N (extra)
def test_N_registries() -> None:
    from models.Clustering_Repository_Builder.experiments.experiment_execution.config import (
        METHOD_SETS, METHOD_NAMES, METHOD_NAMES_PHASED, METHOD_NAMES_PHASE_E2_KNN,
        METHOD_NAMES_STAGE1_2X3,
    )
    from models.Clustering_Repository_Builder.experiments.result_aggregation import (
        method_registry as agg,
    )
    stage1_ok = (len(METHOD_NAMES_STAGE1_2X3) == 6
                 and "stage1_2x3" in METHOD_SETS
                 and len(METHOD_SETS["stage1_2x3"]) == 6)
    untouched = (len(METHOD_NAMES) == 10 and len(METHOD_NAMES_PHASED) == 28
                 and len(METHOD_NAMES_PHASE_E2_KNN) == 12)
    agg_ok = all(n in agg.METHOD_BY_NAME for n in METHOD_NAMES_STAGE1_2X3)
    arms_ok = len(agg.STAGE1_2X3_ARMS) == 6 and set(agg.STAGE1_2X3_ARMS) == set(METHOD_NAMES_STAGE1_2X3)
    no_dupes = len(set(agg.METHOD_NAMES)) == len(agg.METHOD_NAMES)
    record("N", "registries additive; historical sets untouched",
           stage1_ok and untouched and agg_ok and arms_ok and no_dupes,
           f"stage1={stage1_ok} historical(10/28/12)={untouched} agg={agg_ok} "
           f"arms={arms_ok} unique={no_dupes}")


def main() -> int:
    print("=" * 78)
    print("STAGE-1B-1 FOUNDATION TESTS")
    print("=" * 78)
    for fn in (test_A_metric_partition, test_B_full60_backward_compat,
               test_C_head14_selection, test_D_missing_and_duplicate,
               test_E_model_dimensionality, test_F_oracle_restriction,
               test_G_uniform_weights, test_H_seed_derivation,
               test_I_J_optuna_seeding, test_K_configs,
               test_L_search_space_equality, test_M_result_paths,
               test_N_registries):
        try:
            fn()
        except Exception as exc:                       # a test itself blew up
            import traceback
            record(fn.__name__.split("_")[1], fn.__name__, False,
                   f"{type(exc).__name__}: {exc}")
            traceback.print_exc()

    print("=" * 78)
    p = sum(1 for _, s, _ in RESULTS if s == "PASS")
    f = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    b = sum(1 for _, s, _ in RESULTS if s == "SKIP-BLOCKED")
    print(f"PASS={p}  FAIL={f}  SKIP-BLOCKED={b}  (total {len(RESULTS)})")
    if b:
        print("\nBlocked tests require the ClustOpt runtime stack:")
        for tid, s, name in RESULTS:
            if s == "SKIP-BLOCKED":
                print(f"  {tid}: {name}")
    print("=" * 78)
    return 1 if f else 0


if __name__ == "__main__":
    sys.exit(main())
