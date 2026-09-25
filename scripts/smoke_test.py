"""End-to-end smoke test of the release (about 5-15 minutes on a laptop CPU).

A  dependencies importable            F  load the released candidate reranker
B  import CLUSTOPT (60 indices)        G  reconstruct one external dataset + checksum
C  generate a synthetic dataset        H  regenerate a main result table from released results
D  run a CLUSTOPT search on it         I  regenerate a main figure from released results
E  load the released utility predictor J  CLUSTOPT as reported (C4) and C1 through clustopt.pipeline
                                          reproduce 12 published partitions exactly, without
                                          touching the (undistributed) policy-selector models

Writes ``smoke_test_report.json`` in the release root. Uses only this folder
(plus the network for step G only if ``--with-network`` is given; the default
reconstructs a scikit-learn generator dataset, which needs no download).
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
WORK = ROOT / ".smoke"
REPORT = {}


def step(key, title):
    def deco(fn):
        def run():
            t0 = time.time()
            try:
                detail = fn()
                REPORT[key] = {"title": title, "status": "PASS", "detail": detail}
            except Exception as exc:  # noqa: BLE001
                REPORT[key] = {"title": title, "status": "FAIL",
                               "detail": "%s: %s" % (type(exc).__name__, exc),
                               "trace": traceback.format_exc()[-2000:]}
            REPORT[key]["seconds"] = round(time.time() - t0, 1)
            print("[%s] %-4s %s  (%.1fs)" % (key, REPORT[key]["status"], title, REPORT[key]["seconds"]))
        return run
    return deco


@step("A", "dependencies declared and importable")
def a():
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    mods = {"numpy": "numpy", "scipy": "scipy", "scikit-learn": "sklearn", "pandas": "pandas",
            "optuna": "optuna", "hdbscan": "hdbscan", "scikit-optimize": "skopt", "torch": "torch",
            "shapely": "shapely", "networkx": "networkx", "joblib": "joblib",
            "matplotlib": "matplotlib", "cdbw": "cdbw", "pyarrow": "pyarrow", "openpyxl": "openpyxl"}
    missing_decl = [k for k in mods if k not in req and k != "torch"]
    missing_imp = [k for k, m in mods.items() if importlib.util.find_spec(m) is None]
    if missing_imp:
        raise RuntimeError("not importable: %s" % missing_imp)
    return {"not_declared_in_requirements": missing_decl}


@step("B", "import CLUSTOPT and list its validity indices")
def b():
    import clustopt
    from clustopt import validity_indices
    n = len(validity_indices.list_indices())
    if n != 60:
        raise RuntimeError("expected 60 indices, found %d" % n)
    return {"n_indices": n, "version": clustopt.__version__}


STATE = {}


@step("C", "generate one synthetic dataset from a released family configuration")
def c():
    import csv
    import clustopt  # noqa: F401
    from models.HYBRID_SCM.family_generation.planning import expand_jobs, load_family_plan
    from models.HYBRID_SCM import run_family_repository_generation as G
    fam = "standard_clustering_blobs"
    plan = load_family_plan(ROOT / ("data/family_configs/%s.json" % fam))
    job = expand_jobs(plan)[0]
    if WORK.exists():
        shutil.rmtree(WORK)
    run_dir = WORK / fam
    row = G._generate_one(plan, job, run_dir)
    if row.status != "ok":
        raise RuntimeError(row.error_message)
    cat = ROOT / "data/synthetic_generator/dataset_catalogue.csv"
    folder = Path(row.output_dir).name
    ref = next(r for r in csv.DictReader(open(cat, encoding="utf-8")) if r["dataset_id"] == folder)
    same = (int(ref["dataset_seed"]) == int(job.dataset_seed)
            and int(ref["n_points"]) == int(row.n_points_generated))
    if not same:
        raise RuntimeError("regenerated dataset does not match the released catalogue row")
    STATE["dataset_dir"] = Path(row.output_dir)
    STATE["subfamily_root"] = Path(row.output_dir).parent
    return {"dataset_id": row.dataset_id, "n_points": row.n_points_generated, "seed": job.dataset_seed,
            "matches_released_catalogue": same}


@step("D", "run a CLUSTOPT search (neural-profile variant, top-5, no reranking) on the generated dataset")
def d():
    from clustopt import meta_features, search
    subprocess.run([sys.executable, str(ROOT / "scripts/prepare_workspace.py")], check=True, capture_output=True)
    meta_features.extract(STATE["subfamily_root"], max_datasets=1)
    out = search.run(ROOT / "configs/clustopt/neural_profile_top5/xy_2d.json", STATE["dataset_dir"], "xy_2d", n_trials=10)
    if out.used_fallback:
        raise RuntimeError("the utility predictor was not used (fallback objective)")
    ari = search.adjusted_rand(STATE["dataset_dir"], out.y_pred, "xy_2d")
    return {"selected_indices": list(out.selected_metrics), "selected_algorithm": out.selected_algorithm,
            "n_trials": out.n_trials_completed, "ari_vs_ground_truth": ari}


@step("E", "load the released utility predictor and predict a utility profile")
def e():
    import pandas as pd
    from clustopt import utility_prediction
    p = utility_prediction.load_neural("neural_full60")
    feats = pd.read_csv(STATE["dataset_dir"] / "features" / "features_records.csv")
    row = feats[feats["view_mode"] == "xy_2d"].iloc[0].to_dict() if "view_mode" in feats else feats.iloc[0].to_dict()
    u = p.predict(row)
    top = sorted(u.items(), key=lambda kv: -kv[1])[:5]
    return {"n_utilities": len(u), "top5": [t[0] for t in top]}


@step("F", "load the released candidate rerankers")
def f():
    from clustopt import reranking
    names = reranking.available()
    m = reranking.load("KNN_TOP10")
    return {"available": names, "loaded": type(m).__name__, "n_features_in": getattr(m, "n_features_in_", None)}


@step("G", "reconstruct an external dataset and verify its checksum")
def g():
    import clustopt  # noqa: F401
    from Independent_Domain_Benchmark.datasets import preprocessing as P
    from Independent_Domain_Benchmark.datasets import sklearn_shapes as SS
    man = json.loads((ROOT / "data/external_benchmark/dataset_manifest.json").read_text(encoding="utf-8"))
    e = next(x for x in man["datasets"] if x["dataset_id"] == "sk_blobs_easy")
    X, _y = SS.generate(e["generator"], e["difficulty"])
    prep = P.prepare(X, e["dataset_id"], e["source"])
    if prep.feature_checksum != e["representation_checksum"]:
        raise RuntimeError("checksum %s != %s" % (prep.feature_checksum, e["representation_checksum"]))
    return {"dataset_id": e["dataset_id"], "checksum": prep.feature_checksum}


def _numeric_equal(a, b):
    import numpy as np
    import pandas as pd
    A, B = pd.read_csv(a), pd.read_csv(b)
    if list(A.columns) != list(B.columns) or len(A) != len(B):
        return False
    for c in A.columns:
        if pd.api.types.is_numeric_dtype(A[c]):
            if not np.allclose(A[c].to_numpy(float), B[c].to_numpy(float), rtol=0, atol=1e-12, equal_nan=True):
                return False
        elif not (A[c].astype(str) == B[c].astype(str)).all():
            return False
    return True


@step("H", "regenerate main result tables from released aggregate results")
def h():
    subprocess.run([sys.executable, "criterion_baselines.py"], cwd=ROOT / "analysis/statistics", check=True,
                   capture_output=True)
    new = ROOT / "analysis/output/statistics/controlled_criterion_baselines.csv"
    ref = ROOT / "results/controlled/criterion_baselines/canonical_frame/controlled_criterion_baselines.csv"
    same_crit = _numeric_equal(new, ref)
    r = subprocess.run([sys.executable, str(ROOT / "scripts/reproduce_paper_artifacts.py")], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("paper build failed: " + r.stderr[-1500:])
    ev = ROOT / "analysis/output/paper/evidence"
    diffs = [p.name for p in sorted((ROOT / "results/paper_evidence").glob("*.csv"))
             if (ev / p.name).exists() and not _numeric_equal(ev / p.name, p)]
    tabs = sorted(p.name for p in (ROOT / "analysis/output/paper/tables").glob("*.tex"))
    if not same_crit or diffs or not tabs:
        raise RuntimeError("criterion table identical=%s; evidence tables differing=%s; n_tables=%d"
                           % (same_crit, diffs, len(tabs)))
    return {"criterion_baselines_identical": same_crit, "evidence_tables_identical": True, "n_latex_tables": len(tabs)}


@step("I", "regenerate main figures from released aggregate results")
def i():
    figs = sorted(p.name for p in (ROOT / "analysis/output/paper/figures").glob("*.pdf"))
    if not figs:
        raise RuntimeError("no figure produced")
    return {"n_figures": len(figs), "examples": figs[:6]}


@step("J", "CLUSTOPT as reported (C4) and C1 reproduce published external partitions")
def j():
    import gzip
    import clustopt  # noqa: F401
    from clustopt import pipeline
    from Independent_Domain_Benchmark.datasets import preprocessing as P
    from Independent_Domain_Benchmark.datasets import sklearn_shapes as SS
    from models.Independent_Domain_Benchmark.methods import clustopt_adapter as CA

    # any access to the policy-selector models would fail this step
    def _forbidden(self, arm):
        raise RuntimeError("policy-selector model PP_%s was accessed by a C1/C4 run" % arm)
    CA.ClustOptArtifacts._resolve_pp = _forbidden

    man = {x["dataset_id"]: x for x in json.loads(
        (ROOT / "data/external_benchmark/dataset_manifest.json").read_text(encoding="utf-8"))["datasets"]}
    ids = ("sk_blobs_easy", "sk_moons_hard")
    published = {}
    with gzip.open(ROOT / "results/external/per_dataset/final_partitions_by_unit.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["dataset_id"] in ids and r["method_id"] in pipeline.ARMS.values():
                published[(r["dataset_id"], r["view_id"], r["method_id"])] = r["final_labels_hash"]
    rows, bad = [], []
    for ds in ids:
        e = man[ds]
        X, _y = SS.generate(e["generator"], e["difficulty"])
        prep = P.prepare(X, e["dataset_id"], e["source"])
        if prep.feature_checksum != e["representation_checksum"]:
            raise RuntimeError("checksum mismatch for %s" % ds)
        for view in ("x_only", "y_only", "xy_2d"):
            for reranked in (True, False):
                res = pipeline.run(prep.X_2d, view, reranked=reranked)
                ref = published[(ds, view, pipeline.ARMS[reranked])]
                ok = res.details["final_labels_hash"] == ref
                rows.append("%s/%s/%s %s" % (ds, view, res.method, "match" if ok else "DIFFERENT"))
                if not ok:
                    bad.append(rows[-1])
    if bad:
        raise RuntimeError("partitions differing from the published ones: %s" % bad)
    return {"reproduced": "%d/%d" % (len(rows), len(rows)), "units": rows,
            "policy_selector_models_accessed": False}


def main() -> int:
    for fn in (a, b, c, d, e, f, g, h, i, j):
        fn()
    (ROOT / "smoke_test_report.json").write_text(json.dumps(REPORT, indent=2, default=str), encoding="utf-8")
    bad = [k for k, v in REPORT.items() if v["status"] != "PASS"]
    print("FAILED:", bad if bad else "none")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
