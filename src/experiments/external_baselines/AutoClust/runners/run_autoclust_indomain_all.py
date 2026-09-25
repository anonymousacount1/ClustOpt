"""Orchestrate AutoClust In-Domain (split 1) over many families.

Discovers families/subfamilies from the analyzed_data tree and runs each
subfamily as an isolated subprocess (8 workers). Resumable: by default each
subfamily resumes (already-complete datasets are skipped); pass --overwrite to
recompute. Emits "@@@" progress markers. Expected to take a long time.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List

REPO_DEFAULT = r"."
NON_FAMILY_DIRS = {"experiment_splits", "unified_training_dataset"}


def _discover_families(analyzed_root: Path, skip: set) -> List[str]:
    return [d.name for d in sorted(analyzed_root.iterdir())
            if d.is_dir() and d.name not in NON_FAMILY_DIRS and d.name not in skip
            and (d / "subfamilies").is_dir()]


def _subfamilies(analyzed_root: Path, family: str) -> List[str]:
    return sorted(p.name for p in (analyzed_root / family / "subfamilies").iterdir() if p.is_dir())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run AutoClust In-Domain over families")
    ap.add_argument("--repo-root", default=REPO_DEFAULT)
    ap.add_argument("--master-root", required=True)
    ap.add_argument("--model-variant-dir", required=True)
    ap.add_argument("--variant", default="AutoClust_Original_InDomain")
    ap.add_argument("--analyzed-data-root", default=None)
    ap.add_argument("--skip-families", nargs="*", default=[])
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-evaluations", type=int, default=50)
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute completed datasets (default: resume/skip them)")
    # Stage-3A2: explicit strict resume, forwarded to the subfamily runner.
    ap.add_argument("--resume", dest="resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="recompute every record even if a compatible success exists")
    args = ap.parse_args(argv)

    repo = Path(args.repo_root)
    analyzed = Path(args.analyzed_data_root) if args.analyzed_data_root else \
        repo / "results_analysis" / "clustering_repository" / "analyzed_data"
    runner = repo / "experiments" / "external_baselines" / "AutoClust" / "runners" / "run_autoclust_indomain_subfamily.py"
    logdir = repo / "experiments" / "external_baselines" / "AutoClust" / "_run_logs" / "indomain_all"
    logdir.mkdir(parents=True, exist_ok=True)

    families = args.families or _discover_families(analyzed, set(args.skip_families))
    fam_total = len(families)
    print(f"@@@ ALL START: {fam_total} families ({args.variant}, split 1, {args.workers} workers, "
          f"{'overwrite' if args.overwrite else 'resume'})", flush=True)
    print(f"@@@ families: {families}", flush=True)

    ok = 0
    for fi, fam in enumerate(families, 1):
        subs = _subfamilies(analyzed, fam)
        print(f"@@@ FAMILY [{fi}/{fam_total}] START: {fam} ({len(subs)} subfamilies)", flush=True)
        (logdir / fam).mkdir(parents=True, exist_ok=True)
        for si, sub in enumerate(subs, 1):
            print(f"@@@ [{fi}/{fam_total}][{si}/{len(subs)}] SUBFAMILY START: {fam}/{sub}", flush=True)
            t0 = time.time()
            cmd = [sys.executable, str(runner), "--family", fam, "--subfamily", sub,
                   "--master-root", args.master_root, "--model-variant-dir", args.model_variant_dir,
                   "--variant", args.variant, "--workers", str(args.workers),
                   "--n-evaluations", str(args.n_evaluations)]
            if args.overwrite:
                cmd.append("--overwrite")
            cmd.append("--resume" if args.resume else "--no-resume")
            proc = subprocess.run(cmd, capture_output=True, text=True)
            (logdir / fam / f"{sub}.log").write_text(
                proc.stdout + "\n=== STDERR ===\n" + proc.stderr, encoding="utf-8")
            mins = round((time.time() - t0) / 60, 1)
            done_line = next((ln for ln in proc.stdout.splitlines()
                              if ln.startswith("@@@ SUBFAMILY DONE")), None)
            if proc.returncode == 0 and done_line:
                print(f"@@@ [{fi}/{fam_total}][{si}/{len(subs)}] SUBFAMILY OK: {fam}/{sub} (rc=0, {mins}m)", flush=True)
                print(done_line, flush=True); ok += 1
            else:
                print(f"@@@ [{fi}/{fam_total}][{si}/{len(subs)}] SUBFAMILY FAILED: {fam}/{sub} "
                      f"(rc={proc.returncode}, {mins}m)", flush=True)
                print(f"@@@   error tail: " + "\n".join((proc.stderr or proc.stdout).splitlines()[-6:]), flush=True)
        print(f"@@@ FAMILY [{fi}/{fam_total}] COMPLETE: {fam}", flush=True)

    print(f"@@@ ALL COMPLETE: {ok} subfamilies OK across {fam_total} families", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
