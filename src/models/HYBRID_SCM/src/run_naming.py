from __future__ import annotations

from datetime import datetime
import hashlib
import re


_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(value: object, max_len: int = 24) -> str:
    s = _SAFE_RE.sub("_", str(value).strip())
    s = re.sub(r"_+", "_", s).strip("._-")
    if not s:
        s = "run"
    return s[:max_len]


def _short_hash(text: str, n: int = 8) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def build_run_name(cfg, meta):
    """
    Build a Windows-safe but still informative run directory name.

    The previous implementation could easily exceed MAX_PATH on Windows once the
    base project path, dataset tag, and nested artifact names were appended.
    This version keeps the path short and deterministic.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset = _slug(getattr(cfg, "name", "dataset"), max_len=28)

    components = list((meta.get("params_used") or {}).get("components") or [])
    family_counts: dict[str, int] = {}
    for c in components:
        fam = _slug((c or {}).get("type", "comp"), max_len=16)
        family_counts[fam] = family_counts.get(fam, 0) + 1

    fam_items = sorted(family_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    fam_str = "-".join(name for name, _ in fam_items[:2]) if fam_items else "none"
    fam_str = _slug(fam_str, max_len=24)

    mode = _slug(getattr(cfg, "replay_mode", "fresh"), max_len=10)
    seed = getattr(cfg, "seed", "na")

    perturb = meta.get("dataset_perturbations_used") or {}
    flags = []
    if perturb:
        flags.append("pert")
    rendering = meta.get("effective_rendering_spec") or {}
    if rendering.get("enabled", False):
        flags.append("img")
    flag_str = "-".join(flags) if flags else "std"

    fingerprint_payload = {
        "dataset": getattr(cfg, "name", "dataset"),
        "mode": getattr(cfg, "replay_mode", "fresh"),
        "seed": seed,
        "families": [((c or {}).get("type", "comp")) for c in components],
        "flags": sorted(list((perturb or {}).keys())),
    }
    fp = _short_hash(str(fingerprint_payload), n=8)

    return f"{timestamp}__{dataset}__c{len(components)}__{fam_str}__{mode}__s{seed}__{flag_str}__{fp}"
