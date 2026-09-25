"""Merge each split-1 dataset's Phase-D ablation folder into the canonical
split_01 folder (safe, long-path-aware, idempotent).

Per dataset:
  1) delete the listed old ClustOpt phase-A/AB folders + old dataset summary
     from experiments/split_01/
  2) rename (move) every child of experiments/split_01_<SUFFIX>/ into split_01/
     -- os.rename is a pure directory rename within the same parent, so it never
     copies deep contents (immune to MAX_PATH on nested files) and is atomic.
  3) remove the now-empty split_01_<SUFFIX>/ -- ONLY via os.rmdir (empty). If it
     is not empty (a move failed), it is left in place and the dataset is flagged;
     never rm -rf, so unmoved data is never destroyed.

All filesystem ops use the Windows extended-length ``\\?\`` prefix so >260-char
paths work. MODE=dry reports actions without changing anything.

Usage:
    python phaseD_merge_split01.py <dataset_list_file> <dry|run>
"""
from __future__ import annotations

import os
import shutil
import sys

SUFFIX = "split_01_phaseD_no_earlystop_50trials_dynamic_ablation"
DELETE_ITEMS = [
    "all_metrics_uniform_phase_A", "calinski_harabasz_single_phase_A",
    "davies_bouldin_single_phase_A", "dbcv_single_phase_A",
    "regressor_top1_softmax_t05", "regressor_top1_softmax_t05_phase_AB",
    "regressor_top3_softmax_t05", "regressor_top3_softmax_t05_phase_AB",
    "regressor_top5_softmax_t05", "regressor_top5_softmax_t05_phase_AB",
    "regressor_top10_softmax_t05", "regressor_top10_softmax_t05_phase_AB",
    "silhouette_single_phase_A", "uniform_classic_cvi_phase_A",
    "dataset_experiment_summary.json",
]


def _ext(path: str) -> str:
    s = os.path.abspath(path)
    if os.name == "nt" and not s.startswith("\\\\?\\"):
        s = "\\\\?\\" + s
    return s


def _exists(path: str) -> bool:
    return os.path.exists(_ext(path))


def _rm(path: str) -> None:
    ep = _ext(path)
    if os.path.isdir(ep) and not os.path.islink(ep):
        shutil.rmtree(ep)
    else:
        os.remove(ep)


def main() -> int:
    list_file, mode = sys.argv[1], sys.argv[2]
    assert mode in ("dry", "run")
    run = mode == "run"

    with open(list_file, "r", encoding="utf-8") as fh:
        dataset_dirs = [ln.strip() for ln in fh if ln.strip()]

    n_ds = n_skip = n_del = n_mov = n_over = n_rmdir = 0
    failures = []

    for ds in dataset_dirs:
        exp = os.path.join(ds, "experiments")
        src = os.path.join(exp, SUFFIX)
        dst = os.path.join(exp, "split_01")
        if not _exists(src):
            n_skip += 1
            continue
        if not _exists(dst):
            print(f"WARN no split_01, creating: {ds}")
            if run:
                os.makedirs(_ext(dst), exist_ok=True)
        n_ds += 1

        # 1) delete listed items
        for it in DELETE_ITEMS:
            t = os.path.join(dst, it)
            if _exists(t):
                n_del += 1
                if run:
                    _rm(t)

        # 2) move (rename) phaseD children into split_01
        try:
            children = os.listdir(_ext(src))
        except FileNotFoundError:
            children = []
        for name in children:
            s_child = os.path.join(src, name)
            target = os.path.join(dst, name)
            if _exists(target):
                n_over += 1
                if run:
                    _rm(target)
            n_mov += 1
            if run:
                os.rename(_ext(s_child), _ext(target))

        # 3) remove now-empty src (safe: only if truly empty)
        if run:
            remaining = os.listdir(_ext(src)) if _exists(src) else []
            if not remaining:
                os.rmdir(_ext(src))
                n_rmdir += 1
            else:
                failures.append((ds, remaining))
                print(f"ERROR src not empty, left in place: {ds} -> {remaining}")
        else:
            n_rmdir += 1

    print("----")
    print(f"MODE={mode} datasets_merged={n_ds} skipped(no_phaseD)={n_skip} "
          f"deleted_items={n_del} moved_items={n_mov} overwrites={n_over} "
          f"phaseD_dirs_removed={n_rmdir} failures={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
