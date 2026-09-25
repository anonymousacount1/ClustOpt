"""Verify an AutoClust environment: ML2DAC building-block imports + artifacts.

Run (inside the ml2dac env, from the repo root)::

    ~/miniconda3/envs/ml2dac/bin/python \
      experiments/external_baselines/AutoClust/setup/check_autoclust_install.py --repo-root .

Exit 0 if the ML2DAC building blocks import AND the AutoClust artifacts are present;
exit 2 otherwise (with diagnostics).
"""
from __future__ import annotations

import argparse
import importlib
import platform
import sys
import traceback
from pathlib import Path


def _repo_root_default() -> Path:
    return Path(__file__).resolve().parents[4]


def main() -> int:
    ap = argparse.ArgumentParser(description="Check the AutoClust install + artifacts.")
    ap.add_argument("--repo-root", type=str, default=str(_repo_root_default()))
    args = ap.parse_args()
    repo_root = Path(args.repo_root).resolve()

    print("=" * 64)
    print("AutoClust install check (uses the ML2DAC env + submodule)")
    print("=" * 64)
    print(f"python    : {sys.version.splitlines()[0]}")
    print(f"executable: {sys.executable}")
    print(f"platform  : {platform.platform()}")
    print(f"repo_root : {repo_root}")

    src = repo_root / "external" / "ml2dac" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    print(f"ml2dac src: {src} (exists={src.is_dir()})")

    deps_ok = True
    for mod in ("numpy", "pandas", "sklearn", "smac", "ConfigSpace"):
        try:
            m = importlib.import_module(mod)
            print(f"  - {mod}: {getattr(m, '__version__', '?')}")
        except Exception as exc:  # noqa: BLE001
            print(f"  - {mod}: MISSING ({type(exc).__name__})")
            deps_ok = False

    import_ok = True
    err = None
    try:
        importlib.import_module("MetaLearning.MetaFeatureExtractor")
        importlib.import_module("ClusterValidityIndices.CVIHandler")
        importlib.import_module("ClusteringCS.ClusteringCS")
        importlib.import_module("Optimizer.OptimizerSMAC")
    except Exception as exc:  # noqa: BLE001
        import_ok = False
        err = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    print(f"import ML2DAC building blocks: {'OK' if import_ok else 'FAILED'}")
    if not import_ok:
        print(err)

    rw = src / "Experiments" / "RelatedWork" / "related_work"
    required = {"AutoClust.py": src / "Experiments" / "RelatedWork" / "AutoClust.py",
                "meanshift_kdtree.pkl": rw / "meanshift_kdtree.pkl",
                "meanshift_metafeatures.csv": rw / "meanshift_metafeatures.csv",
                "rw_mlp.pkl (trained MLP)": rw / "rw_mlp.pkl",
                "related_work_offline_opt.csv": rw / "related_work_offline_opt.csv"}
    art_ok = True
    print("AutoClust artifacts:")
    for name, p in required.items():
        ok = p.exists()
        art_ok = art_ok and ok
        print(f"  - {name}: {'OK' if ok else 'MISSING'}")

    ready = bool(deps_ok and import_ok and art_ok)
    print("-" * 64)
    print(f"VERDICT: {'AutoClust READY (env + artifacts ok)' if ready else 'NOT READY -- see above'}")
    if platform.system() != "Linux":
        print("NOTE: run real experiments under WSL/Linux (hard timeout uses fork/setsid/killpg).")
    print("=" * 64)
    return 0 if ready else 2


if __name__ == "__main__":
    sys.exit(main())
