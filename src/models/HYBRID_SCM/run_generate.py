from __future__ import annotations

import argparse
import time
from pathlib import Path

from models.HYBRID_SCM.io.config_io import (
    load_dataset_config,
    load_bundle_config,
    load_replay_config,
)
from models.HYBRID_SCM.io.save_utils import ensure_run_dir, save_artifacts
from models.HYBRID_SCM.src.generator import generate_dataset
from models.HYBRID_SCM.replay import make_replay_config


def _looks_like_bundle(path: Path) -> bool:
    try:
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        return isinstance(data, dict) and "bundle_name" in data and "datasets" in data
    except Exception:
        return False


def _run_dataset_config(cfg_path: Path) -> None:
    if _looks_like_bundle(cfg_path):
        _run_bundle(cfg_path)
        return

    cfg = load_dataset_config(cfg_path)
    X, y, meta, feat_names = generate_dataset(cfg)

    run_dir = ensure_run_dir(
        script_dir=Path(__file__).resolve().parent,
        cfg=cfg,
        meta=meta,
    )
    replay_cfg = make_replay_config(cfg, meta)

    save_artifacts(
        run_dir=run_dir,
        X=X,
        y=y,
        feature_names=feat_names,
        metadata=meta,
        replay_config=replay_cfg,
        tag=cfg.name,
    )
    print(f"Done. Outputs saved to: {run_dir}")


def _run_bundle(bundle_path: Path) -> None:
    bundle = load_bundle_config(bundle_path)

    for i, ds in enumerate(bundle.datasets, start=1):

        X, y, meta, feat_names = generate_dataset(ds)

        replay_cfg = make_replay_config(ds, meta)

        run_dir = ensure_run_dir(
            script_dir=Path(__file__).resolve().parent,
            cfg=ds,
            meta=meta,
        )

        save_artifacts(
            run_dir=run_dir,
            X=X,
            y=y,
            feature_names=feat_names,
            metadata=meta,
            replay_config=replay_cfg,
            tag=ds.name,
        )

        print(f"[{i}/{len(bundle.datasets)}] Done. Outputs saved to: {run_dir}")
        time.sleep(1.0)


def _run_replay(replay_path: Path) -> None:
    cfg = load_replay_config(replay_path)
    X, y, meta, feat_names = generate_dataset(cfg)

    run_dir = ensure_run_dir(
        script_dir=Path(__file__).resolve().parent,
        cfg=cfg,
        meta=meta,
    )
    replay_cfg = make_replay_config(cfg, meta)

    save_artifacts(
        run_dir=run_dir,
        X=X,
        y=y,
        feature_names=feat_names,
        metadata=meta,
        replay_config=replay_cfg,
        tag=f"{cfg.name}_REPLAY",
    )
    print(f"Replay done. Outputs saved to: {run_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--bundle", type=str, default=None)
    p.add_argument("--replay", type=str, default=None)
    args = p.parse_args()

    if sum(x is not None for x in [args.config, args.bundle, args.replay]) != 1:
        raise SystemExit("Provide exactly one of: --config, --bundle, --replay")

    if args.config:
        _run_dataset_config(Path(args.config))
    elif args.bundle:
        _run_bundle(Path(args.bundle))
    else:
        _run_replay(Path(args.replay))


if __name__ == "__main__":
    main()
