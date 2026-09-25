"""Orchestrate ML2DAC Original In-Domain (split 1) over many families.

Discovers families/subfamilies from the analyzed_data tree and runs each
subfamily as an isolated subprocess (8 workers, --overwrite). Robust to Windows
line-ending / shell-quoting issues (no bash). Emits "@@@" progress markers.
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
    fams = []
    for d in sorted(analyzed_root.iterdir()):
        if not d.is_dir() or d.name in NON_FAMILY_DIRS or d.name in skip:
            continue
        if (d / "subfamilies").is_dir():
            fams.append(d.name)
    return fams


def _subfamilies(analyzed_root: Path, family: str) -> List[str]:
    sub_root = analyzed_root / family / "subfamilies"
    return sorted(p.name for p in sub_root.iterdir() if p.is_dir())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run ML2DAC In-Domain over remaining families")
    ap.add_argument("--repo-root", default=REPO_DEFAULT)
    ap.add_argument("--master-root", required=True)
    ap.add_argument("--mkr-variant-dir", required=True)
    ap.add_argument("--variant", default="ML2DAC_Original_InDomain")
    ap.add_argument("--analyzed-data-root", default=None)
    ap.add_argument("--skip-families", nargs="*", default=["arcs_circles_rings"])
    ap.add_argument("--families", nargs="*", default=None,
                    help="explicit family list (default: all discovered minus skip)")
    ap.add_argument("--workers", type=int, default=8)
    # Overwrite is ON by default (unchanged behaviour). --no-overwrite enables
    # resume: the subfamily runner then skips datasets whose result.json exists.
    ap.add_argument("--overwrite", dest="overwrite", action="store_true", default=True)
    ap.add_argument("--no-overwrite", dest="overwrite", action="store_false",
                    help="resume: skip already-completed datasets instead of redoing them.")
    args = ap.parse_args(argv)

    repo = Path(args.repo_root)
    analyzed = Path(args.analyzed_data_root) if args.analyzed_data_root else \
        repo / "results_analysis" / "clustering_repository" / "analyzed_data"
    runner = repo / "experiments" / "external_baselines" / "ML2DAC" / "runners" / "run_ml2dac_indomain_subfamily.py"
    logdir = repo / "experiments" / "external_baselines" / "ML2DAC" / "_run_logs" / "indomain_all"
    logdir.mkdir(parents=True, exist_ok=True)

    families = args.families or _discover_families(analyzed, set(args.skip_families))
    fam_total = len(families)
    print(f"@@@ ALL START: {fam_total} families (ML2DAC_Original_InDomain, split 1, {args.workers} workers)", flush=True)
    print(f"@@@ families: {families}", flush=True)

    overall_ok = 0
    for fi, fam in enumerate(families, 1):
        subs = _subfamilies(analyzed, fam)
        print(f"@@@ FAMILY [{fi}/{fam_total}] START: {fam} ({len(subs)} subfamilies)", flush=True)
        (logdir / fam).mkdir(parents=True, exist_ok=True)
        for si, sub in enumerate(subs, 1):
            print(f"@@@ [{fi}/{fam_total}][{si}/{len(subs)}] SUBFAMILY START: {fam}/{sub}", flush=True)
            t0 = time.time()
            cmd = [sys.executable, str(runner), "--family", fam, "--subfamily", sub,
                   "--master-root", args.master_root, "--mkr-variant-dir", args.mkr_variant_dir,
                   "--variant", args.variant, "--workers", str(args.workers)]
            if args.overwrite:
                cmd.append("--overwrite")
            proc = subprocess.run(cmd, capture_output=True, text=True)
            (logdir / fam / f"{sub}.log").write_text(
                proc.stdout + "\n=== STDERR ===\n" + proc.stderr, encoding="utf-8")
            mins = round((time.time() - t0) / 60, 1)
            done_line = next((ln for ln in proc.stdout.splitlines()
                              if ln.startswith("@@@ SUBFAMILY DONE")), None)
            if proc.returncode == 0 and done_line:
                print(f"@@@ [{fi}/{fam_total}][{si}/{len(subs)}] SUBFAMILY OK: {fam}/{sub} (rc=0, {mins}m)", flush=True)
                print(done_line, flush=True)
                overall_ok += 1
            else:
                print(f"@@@ [{fi}/{fam_total}][{si}/{len(subs)}] SUBFAMILY FAILED: {fam}/{sub} "
                      f"(rc={proc.returncode}, {mins}m)", flush=True)
                tail = "\n".join((proc.stderr or proc.stdout).splitlines()[-6:])
                print(f"@@@   error tail: {tail}", flush=True)
        print(f"@@@ FAMILY [{fi}/{fam_total}] COMPLETE: {fam}", flush=True)

    print(f"@@@ ALL COMPLETE: {overall_ok} subfamilies OK across {fam_total} families", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
