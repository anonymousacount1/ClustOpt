"""Stage 3A-1: freeze the two new metric-inventory specifications.

Writes ``variant_specs/original_plus_established.json`` and
``variant_specs/original_plus_new46.json``. Each spec is a machine-readable,
hash-pinned scientific object: the exact requested canonical metric ids, the
expected live count after the builders' all-NaN rule, the provenance of every
source consulted, the alias map, the split policy and the seed.

The specs are frozen BEFORE any derived repository is built, and no Split-1
outcome is read at any point -- this module opens the master metric registry, the
canonical taxonomy sources, and (read-only, for the backward-compatibility gate)
the *candidate lists* of the existing A0/A3 variants. It never reads an ARI.

Run from the Unified_MKR directory::

    python runners/freeze_variant_specs.py
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_THIS = Path(__file__).resolve()
_PKG_PARENT = _THIS.parents[1]
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from unified_mkr import io_utils  # noqa: E402
from unified_mkr import metric_partitions as MP  # noqa: E402

SPEC_DIR = "variant_specs"
SPEC_SCHEMA_VERSION = "stage3a1.variant_spec.v1"
TRAIN_SPLITS = list(range(2, 17))
HELDOUT_SPLIT = 1
RANDOM_SEED = 1234


def _log(msg: str) -> None:
    print("[variant-specs] %s" % msg, flush=True)


def _now() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def spec_hash(spec: Dict[str, Any]) -> str:
    """Hash of everything that defines the variant scientifically.

    Deliberately excludes ``created_at``, ``source_commit`` and the hash field
    itself, so re-freezing an unchanged design reproduces the same hash.
    """
    payload = {k: spec[k] for k in (
        "schema_version", "variant_name", "semantic_variant", "baseline_scope",
        "requested_metric_ids", "requested_metric_count",
        "expected_live_metric_count", "expected_dead_metric_ids",
        "dead_metric_exclusion_rule", "alias_resolution_map",
        "partition_membership", "split_policy", "random_seed",
        "source_provenance")}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def build_spec(*, variant_name: str, semantic: str, requested: List[str],
               parts: Dict[str, Any], membership: Dict[str, Any],
               commit: str, registry_hash: str) -> Dict[str, Any]:
    dead = [m for m in MP.EXPECTED_DEAD if m in requested]
    spec: Dict[str, Any] = {
        "schema_version": SPEC_SCHEMA_VERSION,
        "variant_name": variant_name,
        "semantic_variant": semantic,
        "baseline_scope": ["autoclust", "ml2dac"],
        "requested_metric_ids": sorted(requested),
        "requested_metric_count": len(requested),
        "expected_live_metric_count": len(set(requested) - set(dead)),
        "expected_live_metric_ids": sorted(set(requested) - set(dead)),
        "expected_dead_metric_ids": sorted(dead),
        "dead_metric_exclusion_rule": MP.DEAD_EXCLUSION_RULE,
        "alias_resolution_map": parts["alias_map"],
        "partition_membership": membership,
        "split_policy": {"train_splits": TRAIN_SPLITS,
                         "heldout_test_split": HELDOUT_SPLIT,
                         "excluded_splits": []},
        "random_seed": RANDOM_SEED,
        "source_provenance": {
            "original_set": ("master metric_registry.parquet flag "
                             "used_by_ml2dac, cross-checked against "
                             "unified_mkr.metric_registry.ML2DAC_CVIS"),
            "new46_set": parts["new46_provenance"],
            "established_set": ("residual: extended_requested - original - "
                                "new46, where extended_requested is the "
                                "builders' own pool (used_by_ml2dac OR "
                                "used_by_clustopt)"),
            "master_metric_registry_hash": registry_hash,
            "partition_set_hashes": parts["hashes"],
            "name_inference_used": False,
            "outcome_information_used": False,
        },
        "scientific_wording": (
            "46 newly designed metrics, of which 44 are evaluable on this "
            "training repository and 2 are deterministically excluded as "
            "all-NaN" if semantic == "A2" else
            "the 7 original baseline CVIs plus the 10 established additions"),
        "created_at": _now(),
        "source_commit": commit,
    }
    spec["manifest_hash"] = spec_hash(spec)
    return spec


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Freeze Stage-3A1 variant specs")
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    repo = Path(args.repo_root).resolve() if args.repo_root else io_utils.find_repo_root()
    io_utils.ensure_repo_on_path(repo)
    io_utils.ensure_ml2dac_on_path(repo)
    mkr = repo / "experiments" / "external_baselines" / "Unified_MKR"
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True).stdout.strip()

    print("=" * 78)
    print("STAGE 3A-1  FREEZE METRIC-INVENTORY VARIANT SPECS")
    print("=" * 78, flush=True)

    reg_path = mkr / "master" / "metric_registry.parquet"
    registry = io_utils.read_parquet(reg_path)
    if registry is None:
        raise SystemExit("cannot read %s" % reg_path)
    registry_hash = hashlib.sha256(
        Path(io_utils.ext(reg_path)).read_bytes()).hexdigest()[:16]

    parts = MP.resolve_partitions(repo, registry)
    _log("partitions resolved: original=%d established=%d new46=%d "
         "extended_requested=%d (%d/%d gates)"
         % (len(parts["original"]), len(parts["established"]),
            len(parts["new46"]), len(parts["extended_requested"]),
            sum(g["passed"] for g in parts["gates"]), len(parts["gates"])))

    # ---- §10 backward-compatibility against the existing A0/A3 --------------
    compat: List[Dict[str, Any]] = []
    for baseline, root in (("autoclust", mkr / "derived" / "autoclust_split1"),
                           ("ml2dac", mkr / "derived" / "ml2dac_split1")):
        compat.extend(MP.verify_against_existing(parts, root, baseline))
    bad = [c for c in compat
           if not (c["requested_match"] and c["live_match"] and c["dead_match"]
                   and c["all_directions_present"])]
    for c in compat:
        _log("compat %-9s %-14s requested=%s live=%s dead=%s dirs=%s "
             "(existing req=%d live=%d)"
             % (c["baseline"], c["variant"], c["requested_match"],
                c["live_match"], c["dead_match"], c["all_directions_present"],
                c["n_requested_existing"], c["n_live_existing"]))
    if bad:
        raise SystemExit("BACKWARD-COMPATIBILITY FAILED: %s" % bad)

    membership = {
        "original": parts["original"],
        "established_additions": parts["established"],
        "new46": parts["new46"],
    }
    specs = {
        MP.VARIANT_ORIGINAL_PLUS_ESTABLISHED: build_spec(
            variant_name=MP.VARIANT_ORIGINAL_PLUS_ESTABLISHED, semantic="A1",
            requested=parts["a1_requested"], parts=parts,
            membership={"original": membership["original"],
                        "established_additions": membership["established_additions"]},
            commit=commit, registry_hash=registry_hash),
        MP.VARIANT_ORIGINAL_PLUS_NEW46: build_spec(
            variant_name=MP.VARIANT_ORIGINAL_PLUS_NEW46, semantic="A2",
            requested=parts["a2_requested"], parts=parts,
            membership={"original": membership["original"],
                        "new46": membership["new46"]},
            commit=commit, registry_hash=registry_hash),
    }

    out = io_utils.ensure_dir(mkr / SPEC_DIR)
    for name, spec in specs.items():
        p = out / ("%s.json" % name)
        if Path(io_utils.ext(p)).exists() and not args.overwrite:
            prev = io_utils.read_json(p) or {}
            if prev.get("manifest_hash") != spec["manifest_hash"]:
                raise SystemExit(
                    "%s exists with a DIFFERENT manifest hash (%s != %s); pass "
                    "--overwrite only if the design genuinely changed"
                    % (p, prev.get("manifest_hash"), spec["manifest_hash"]))
            _log("%s unchanged (hash %s)" % (name, spec["manifest_hash"]))
            continue
        io_utils.write_json_atomic(p, spec)
        _log("wrote %s  requested=%d live=%d dead=%d  hash=%s"
             % (p.name, spec["requested_metric_count"],
                spec["expected_live_metric_count"],
                len(spec["expected_dead_metric_ids"]), spec["manifest_hash"]))

    report = {
        "stage": "3A-1", "created_at": _now(), "source_commit": commit,
        "master_metric_registry_hash": registry_hash,
        "partition_counts": {k: len(parts[k]) for k in
                             ("original", "established", "new46",
                              "extended_requested", "a1_requested",
                              "a1_expected_live", "a2_requested",
                              "a2_expected_live")},
        "partition_set_hashes": parts["hashes"],
        "partition_gates": parts["gates"],
        "alias_resolution_map": parts["alias_map"],
        "expected_dead_metric_ids": parts["expected_dead"],
        "new46_provenance": parts["new46_provenance"],
        "backward_compatibility": compat,
        "backward_compatibility_passed": not bad,
        "specs": {n: {"path": str(out / ("%s.json" % n)),
                      "manifest_hash": s["manifest_hash"],
                      "requested": s["requested_metric_count"],
                      "expected_live": s["expected_live_metric_count"]}
                  for n, s in specs.items()},
        "outcome_information_used": False,
    }
    io_utils.write_json_atomic(out / "variant_spec_freeze_report.json", report)
    _log("freeze report written; ALL GATES PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
