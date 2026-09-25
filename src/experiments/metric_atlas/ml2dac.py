"""Stage 3B step 6: which CVI ML2DAC actually picks, and whether it picks well.

ML2DAC is the one baseline that names a single index per dataset, so it is the
only place where "which metric did the method choose" is a real question rather
than a fabricated one.

Two sources, both already on disk:

* ``selected_cvi`` in every Split-1 ``result.json`` -- what the method chose;
* ``cvi_star`` -- the offline label the method is *trying* to predict, i.e. the
  candidate whose Spearman correlation with ARI is highest on that record.

For Split 1 the stored ``cvi_star.parquet`` covers only the training splits, so
the Split-1 labels are recomputed with the project's own frozen
``ml2dac_cvi_selection.compute_cvi_star`` over evaluations that already exist in
the master MKR. That involves no clustering, no search and no model fit: it is a
deterministic re-read of stored metric and ARI columns.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import common as C

ARMS = (
    ("A0", "ML2DAC_Original_InDomain_same_search_space", "original_cvis"),
    ("A1", "ML2DAC_OriginalPlusEstablished_InDomain_same_search_space",
     "original_plus_established"),
    ("A2", "ML2DAC_OriginalPlusNew46_InDomain_same_search_space",
     "original_plus_new46"),
    ("A3", "ML2DAC_Extended_InDomain_same_search_space", "extended_cvis"),
)
VIEW_DIRS = {"1d_x": "x_only", "1d_y": "y_only", "2d": "xy_2d"}


def canonical_name_map() -> Dict[str, str]:
    """ML2DAC short CVI name -> canonical metric_id.

    The seven ORIGINAL CVIs are stored in ``result.json`` under ML2DAC's own
    abbreviations (SIL, CH, DBI, DBCV, DI, CJI, COP) while the established and
    New46 additions are stored under canonical ids. Comparing a selection with
    CVI* therefore requires this map; it is imported from the adapter that
    created the abbreviations, never re-typed here.
    """
    from experiments.external_baselines.ML2DAC.indomain.dataset import (  # noqa: E402
        CVI_ABBREV,
    )
    return {abbrev: canon for canon, abbrev in CVI_ABBREV.items()}


def _read_json_long(p: Path) -> Optional[dict]:
    import json
    q = Path(C.ext(p))
    if not q.exists():
        return None
    try:
        return json.loads(q.read_text(encoding="utf-8"))
    except Exception:
        return None


def collect_selections(repo: Path, log) -> pd.DataFrame:
    """One row per (arm, dataset, view): the CVI ML2DAC selected and the ARI."""
    rec = pd.read_parquet(repo / C.MKR_REL / "master" / "records.parquet",
                          columns=["dataset_id", "dataset_folder", "family",
                                   "subfamily", "split_id"]).drop_duplicates("dataset_id")
    s1 = rec[rec.split_id == 1]
    rows: List[Dict[str, Any]] = []
    for arm, method, _ in ARMS:
        n = 0
        for r in s1.itertuples():
            base = Path(r.dataset_folder) / "experiments" / "split_01" / method
            for vd, view in VIEW_DIRS.items():
                j = _read_json_long(base / vd / "result.json")
                if not j or str(j.get("status", "")).lower() != "success":
                    continue
                rows.append({
                    "arm": arm, "method_id": method,
                    "dataset_id": r.dataset_id, "family": r.family,
                    "subfamily": r.subfamily, "view_type": view,
                    "record_id": "%s__%s" % (r.dataset_id, view),
                    "selected_cvi": j.get("selected_cvi"),
                    "selected_cvi_source": (j.get("details") or {}).get("selected_cvi_source"),
                    "ari": (j.get("metrics") or {}).get("ari"),
                    "k_correct": (j.get("metrics") or {}).get("k_correct"),
                })
                n += 1
        log("ML2DAC %s: %d successful view records read" % (arm, n))
    return pd.DataFrame(rows)


def split1_cvi_star(repo: Path, log) -> pd.DataFrame:
    """Split-1 CVI* per arm, via the project's own frozen selection function."""
    from unified_mkr import ml2dac_cvi_selection as CS  # noqa: E402
    M = repo / C.MKR_REL / "master"
    rec = pd.read_parquet(M / "records.parquet",
                          columns=["record_id", "dataset_id", "split_id", "family",
                                   "subfamily", "view_type"])
    s1_ids = set(rec[rec.split_id == 1]["record_id"])
    reg = pd.read_parquet(M / "metric_registry.parquet")
    directions = dict(zip(reg["metric_id"], reg["direction"]))

    ev = pd.read_parquet(M / "evaluations.parquet")
    ev = ev[ev["record_id"].isin(s1_ids)]
    log("split-1 evaluations for CVI*: %d rows over %d records"
        % (len(ev), ev["record_id"].nunique()))

    out = []
    for arm, method, vdir in ARMS:
        cand = pd.read_parquet(repo / C.MKR_REL / "derived" / "ml2dac_split1" / vdir
                               / "cvi_candidates.parquet")
        ids = [m for m in cand["metric_id"] if m in ev.columns]
        star, _ = CS.compute_cvi_star(ev, ids, directions, rec, log=None)
        star["arm"] = arm
        star["n_candidates"] = len(ids)
        out.append(star)
        log("CVI* %s: %d candidates, %d records, %d resolved, %d distinct labels"
            % (arm, len(ids), len(star), int(star["resolved"].sum()),
               star.loc[star["resolved"], "best_cvi"].nunique()))
    return pd.concat(out, ignore_index=True)


def analyse(sel: pd.DataFrame, star: pd.DataFrame, A: pd.DataFrame, log
            ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grp = A.set_index("metric_id")["group"].to_dict()
    sub = A.set_index("metric_id")["subgroup"].to_dict()
    sel = sel.copy()
    alias = canonical_name_map()
    sel["selected_cvi_raw"] = sel["selected_cvi"]
    sel["selected_cvi"] = sel["selected_cvi"].map(lambda x: alias.get(x, x))
    unresolved = sorted(set(sel.loc[~sel["selected_cvi"].isin(grp), "selected_cvi"]))
    if unresolved:
        raise AssertionError("unmapped selected_cvi names: %s" % unresolved[:10])
    sel["selected_group"] = sel["selected_cvi"].map(grp).fillna("Unknown")
    sel["selected_subgroup"] = sel["selected_cvi"].map(sub).fillna("")

    # ---- selection frequency per (arm, metric) ----
    rows = []
    for arm, g in sel.groupby("arm"):
        n = len(g)
        vc = g["selected_cvi"].value_counts()
        for m, c in vc.items():
            sl = g[g.selected_cvi == m]
            rows.append({
                "arm": arm, "metric_id": m, "group": grp.get(m, "Unknown"),
                "subgroup": sub.get(m, ""), "n_selected": int(c),
                "selection_freq": float(c) / n,
                "mean_ari_when_selected": float(pd.to_numeric(sl["ari"],
                                                              errors="coerce").mean()),
                "n_view_records": n,
            })
    S = pd.DataFrame(rows).sort_values(["arm", "selection_freq"], ascending=[True, False])

    # ---- group-level selection share + CVI* accuracy per arm ----
    j = sel.merge(star[["record_id", "arm", "best_cvi", "resolved", "n_candidates"]],
                  on=["record_id", "arm"], how="left")
    j["correct"] = (j["selected_cvi"] == j["best_cvi"]) & j["resolved"].fillna(False)
    arows = []
    for arm, g in j.groupby("arm"):
        res = g[g["resolved"].fillna(False)]
        gsh = g.groupby("selected_group").size() / len(g)
        ssh = star[star.arm == arm]
        ssh = ssh[ssh.resolved]
        star_grp = ssh["best_cvi"].map(grp).value_counts(normalize=True)
        arows.append({
            "arm": arm,
            "n_candidates": int(g["n_candidates"].dropna().iloc[0]) if g["n_candidates"].notna().any() else np.nan,
            "n_view_records": int(len(g)),
            "n_resolved_cvi_star": int(len(res)),
            "selection_accuracy_vs_cvi_star": float(res["correct"].mean()) if len(res) else np.nan,
            "n_distinct_selected": int(g["selected_cvi"].nunique()),
            "n_distinct_cvi_star": int(ssh["best_cvi"].nunique()),
            "selected_share_Original": float(gsh.get("Original", 0.0)),
            "selected_share_Established": float(gsh.get("Established", 0.0)),
            "selected_share_New46": float(gsh.get("New46", 0.0)),
            "cvi_star_share_Original": float(star_grp.get("Original", 0.0)),
            "cvi_star_share_Established": float(star_grp.get("Established", 0.0)),
            "cvi_star_share_New46": float(star_grp.get("New46", 0.0)),
            "mean_ari": float(pd.to_numeric(g["ari"], errors="coerce").mean()),
            "majority_class_share_cvi_star": float(
                ssh["best_cvi"].value_counts(normalize=True).iloc[0]) if len(ssh) else np.nan,
        })
    Arm = pd.DataFrame(arows).sort_values("arm")

    # ---- per-CVI recall (of the records whose CVI* is m, how often chosen) ----
    rrows = []
    for arm, g in j.groupby("arm"):
        res = g[g["resolved"].fillna(False)]
        for m, gm in res.groupby("best_cvi"):
            rrows.append({
                "arm": arm, "metric_id": m, "group": grp.get(m, "Unknown"),
                "subgroup": sub.get(m, ""),
                "n_records_with_this_cvi_star": int(len(gm)),
                "recall": float((gm["selected_cvi"] == m).mean()),
                "cvi_star_share": float(len(gm) / len(res)),
            })
    R = pd.DataFrame(rrows)

    # ---- family-level selection ----
    frows = []
    for (arm, fam), g in sel.groupby(["arm", "family"]):
        share = g.groupby("selected_group").size() / len(g)
        frows.append({
            "arm": arm, "family": fam, "n_view_records": int(len(g)),
            "share_Original": float(share.get("Original", 0.0)),
            "share_Established": float(share.get("Established", 0.0)),
            "share_New46": float(share.get("New46", 0.0)),
            "top_selected_cvi": str(g["selected_cvi"].value_counts().index[0]),
            "mean_ari": float(pd.to_numeric(g["ari"], errors="coerce").mean()),
        })
    F = pd.DataFrame(frows)

    for _, r in Arm.iterrows():
        log("ML2DAC %s: %s candidates | CVI* accuracy %.4f | New46 selected share "
            "%.3f | CVI* New46 share %.3f"
            % (r["arm"], r["n_candidates"], r["selection_accuracy_vs_cvi_star"],
               r["selected_share_New46"], r["cvi_star_share_New46"]))
    return S, Arm, R, F
