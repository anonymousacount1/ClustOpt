"""Artifact manifest: the single source of truth about what was generated.

The previous packaging layer decided which file belonged in which package section
by looking up its **basename** in a static dict. That breaks silently the moment a
new table appears, a name is reused across suites, or a suite is skipped.

Here, every writer registers what it produced -- section, type, description,
source tables, paper importance, selection type, whether external methods are
covered -- and the packager consumes *that*. A file the analysis did not register
is not packaged, and a registered file that failed to materialise is recorded as
``exists = False`` rather than vanishing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from . import paths

ARTIFACT_TYPES: tuple = ("table", "plot", "report", "workbook", "metadata", "frame")
PAPER_IMPORTANCE: tuple = ("paper_main", "paper_supplement", "appendix",
                           "exploratory")

MANIFEST_COLUMNS: List[str] = [
    "comparison_id", "section", "artifact_type", "relative_path", "description",
    "source_tables", "paper_importance", "paper_candidate", "appendix_only",
    "selection_type", "supports_external_methods", "created", "exists",
    "size_bytes", "notes",
]


@dataclass
class Artifact:
    comparison_id: str
    section: str
    artifact_type: str
    relative_path: str
    description: str
    source_tables: str = ""
    paper_importance: str = "paper_supplement"
    paper_candidate: bool = False
    appendix_only: bool = False
    selection_type: str = ""
    supports_external_methods: bool = True
    created: bool = True
    exists: bool = False
    size_bytes: int = 0
    notes: str = ""

    def to_record(self) -> Dict[str, Any]:
        return asdict(self)


class ArtifactManifest:
    """Accumulates artifacts and resolves them against the output root."""

    def __init__(self, output_root: Path) -> None:
        self.output_root = Path(output_root)
        self.artifacts: List[Artifact] = []

    # ------------------------------------------------------------------ add
    def add(self, path: Optional[Path], *, comparison_id: str, section: str,
            artifact_type: str, description: str,
            source_tables: Sequence[str] = (), paper_importance: str = "paper_supplement",
            paper_candidate: Optional[bool] = None, appendix_only: bool = False,
            selection_type: str = "", supports_external_methods: bool = True,
            created: bool = True, notes: str = "") -> Optional[Path]:
        """Register one artifact. ``path=None`` records a *not created* row.

        Recording the miss is deliberate: a suite that skipped a table because a
        capability was unavailable should leave a visible trace in the manifest.

        ``paper_candidate`` defaults to ``paper_importance == 'paper_main'`` so the
        two cannot drift apart; pass it explicitly only to override.
        """
        if artifact_type not in ARTIFACT_TYPES:                # pragma: no cover
            raise ValueError(f"unknown artifact_type {artifact_type!r}")
        if paper_importance not in PAPER_IMPORTANCE:           # pragma: no cover
            raise ValueError(f"unknown paper_importance {paper_importance!r}")
        if paper_candidate is None:
            paper_candidate = paper_importance == "paper_main"
        rel = ""
        exists = False
        size = 0
        if path is not None:
            p = Path(path)
            try:
                rel = str(p.relative_to(self.output_root))
            except ValueError:
                rel = str(p)
            exists = paths.exists(p)
            size = paths.size_bytes(p) if exists else 0
        self.artifacts.append(Artifact(
            comparison_id=comparison_id, section=section,
            artifact_type=artifact_type, relative_path=rel,
            description=description, source_tables=";".join(source_tables),
            paper_importance=paper_importance, paper_candidate=paper_candidate,
            appendix_only=appendix_only, selection_type=selection_type,
            supports_external_methods=supports_external_methods,
            created=bool(created and path is not None), exists=exists,
            size_bytes=size, notes=notes))
        return path

    def add_table(self, path: Optional[Path], **kwargs) -> Optional[Path]:
        return self.add(path, artifact_type="table", **kwargs)

    def add_report(self, path: Optional[Path], **kwargs) -> Optional[Path]:
        return self.add(path, artifact_type="report", **kwargs)

    def add_workbook(self, path: Optional[Path], **kwargs) -> Optional[Path]:
        return self.add(path, artifact_type="workbook", **kwargs)

    def add_frame(self, path: Optional[Path], **kwargs) -> Optional[Path]:
        return self.add(path, artifact_type="frame", **kwargs)

    def add_metadata(self, path: Optional[Path], **kwargs) -> Optional[Path]:
        return self.add(path, artifact_type="metadata", **kwargs)

    def add_plot_results(self, results: Iterable[Any], *, comparison_id: str,
                         section: str, default_importance: str = "paper_supplement"
                         ) -> None:
        """Register a :class:`plots.PlotContext`'s results (figures + data CSVs)."""
        for r in results:
            spec = r.spec
            self.add(r.png, comparison_id=comparison_id, section=section,
                     artifact_type="plot", description=spec.title,
                     source_tables=spec.source_tables,
                     paper_importance=spec.importance or default_importance,
                     paper_candidate=spec.importance == "paper_main",
                     selection_type=spec.selection_type,
                     created=r.created,
                     notes=(spec.caption or "") + (
                         f" [skipped: {r.skipped_reason}]" if r.skipped_reason else ""))
            if r.svg is not None:
                self.add(r.svg, comparison_id=comparison_id, section=section,
                         artifact_type="plot",
                         description=f"{spec.title} (vector version for the paper)",
                         source_tables=spec.source_tables,
                         paper_importance=spec.importance or default_importance,
                         paper_candidate=spec.importance == "paper_main",
                         selection_type=spec.selection_type)
            if r.data_csv is not None:
                self.add(r.data_csv, comparison_id=comparison_id, section=section,
                         artifact_type="table",
                         description=f"Plotted data behind {spec.name}.png",
                         source_tables=spec.source_tables,
                         paper_importance="appendix", appendix_only=True,
                         selection_type=spec.selection_type)

    # --------------------------------------------------------------- output
    def to_frame(self) -> pd.DataFrame:
        if not self.artifacts:
            return pd.DataFrame(columns=MANIFEST_COLUMNS)
        return pd.DataFrame([a.to_record() for a in self.artifacts],
                            columns=MANIFEST_COLUMNS)

    def refresh_existence(self) -> None:
        """Re-check every registered path on disk (call before writing)."""
        for a in self.artifacts:
            if not a.relative_path:
                continue
            p = self.output_root / a.relative_path
            a.exists = paths.exists(p)
            a.size_bytes = paths.size_bytes(p) if a.exists else 0

    def merge_existing(self) -> pd.DataFrame:
        """Union this run's rows with the manifest already on disk.

        Stages run independently -- the preliminary study, then each comparison
        suite -- and each builds its own manifest. Without this, writing a suite's
        manifest would silently erase every artifact the earlier stages
        registered, and the packager (which is manifest-driven) would then ship a
        package missing the protocol documents and the frozen manifests.

        Rows are upserted on ``relative_path``: a re-run of the same stage
        replaces its own rows, and other stages' rows survive untouched.
        """
        frame = self.to_frame()
        existing_path = self.output_root / "analysis_manifest.csv"
        if not paths.exists(existing_path):
            return frame
        try:
            existing = paths.read_csv(existing_path)
        except Exception:                                      # pragma: no cover
            return frame
        if existing.empty:
            return frame
        for col in MANIFEST_COLUMNS:
            if col not in existing.columns:
                existing[col] = None
        existing = existing[MANIFEST_COLUMNS]
        # A CSV round-trip turns empty strings into NaN; normalise so the
        # upsert key and the path arithmetic below both behave.
        for f in (existing, frame):
            f["relative_path"] = f["relative_path"].fillna("").astype(str)
            f["created"] = f["created"].astype(str).str.lower().isin(
                ("true", "1", "yes"))
        # Upsert on TWO keys. The path identifies a file; the identity
        # (comparison, section, description) identifies the *artifact*, which is
        # what matters when one transitions between created and skipped. Without
        # the identity key, an artifact that was skipped on one run and produced
        # on the next would leave its stale "not created" row behind and the
        # manifest would claim the analysis both did and did not produce it.
        def _identity(f: pd.DataFrame) -> pd.Series:
            return (f["comparison_id"].astype(str) + "|"
                    + f["section"].astype(str) + "|"
                    + f["description"].astype(str))

        new_paths = set(frame.loc[frame["relative_path"] != "", "relative_path"])
        new_identities = set(_identity(frame))
        superseded = (existing["relative_path"].isin(new_paths)
                      | _identity(existing).isin(new_identities))
        merged = pd.concat([existing[~superseded], frame], ignore_index=True)

        def _exists(rel: str) -> bool:
            return bool(rel) and paths.exists(self.output_root / rel)

        merged["exists"] = merged["relative_path"].map(_exists)
        merged["size_bytes"] = [
            paths.size_bytes(self.output_root / rel) if ok else 0
            for rel, ok in zip(merged["relative_path"], merged["exists"])]
        # Never promise the packager a file that is no longer there.
        stale = merged["created"] & ~merged["exists"]
        if bool(stale.any()):
            merged = merged[~stale]
        return merged.reset_index(drop=True)

    def write(self, *, dry_run: bool = False, merge: bool = True
              ) -> Dict[str, Optional[Path]]:
        self.refresh_existence()
        frame = self.merge_existing() if merge else self.to_frame()
        if dry_run:
            return {"csv": None, "json": None}
        csv_path = self.output_root / "analysis_manifest.csv"
        json_path = self.output_root / "analysis_manifest.json"
        paths.write_csv(csv_path, frame, index=False)
        paths.write_json(json_path, {
            "output_root": str(self.output_root),
            "n_artifacts": len(frame),
            "n_existing": int(frame["exists"].sum()) if len(frame) else 0,
            "n_missing": int((~frame["exists"]).sum()) if len(frame) else 0,
            "sections": sorted(frame["section"].unique().tolist()) if len(frame) else [],
            "comparisons": sorted(frame["comparison_id"].unique().tolist())
            if len(frame) else [],
            "artifacts": paths.as_records(frame),
        })
        return {"csv": csv_path, "json": json_path}

    # ---------------------------------------------------------- validation
    def validate(self) -> List[str]:
        problems: List[str] = []
        self.refresh_existence()
        frame = self.to_frame()
        if frame.empty:
            problems.append("artifact manifest is empty")
            return problems
        created_missing = frame[frame["created"] & ~frame["exists"]]
        if len(created_missing):
            problems.append(
                f"{len(created_missing)} artifact(s) registered as created but "
                f"absent on disk: "
                f"{created_missing['relative_path'].head(5).tolist()}")
        dupes = frame[frame["relative_path"].duplicated() & (frame["relative_path"] != "")]
        if len(dupes):
            problems.append(f"{len(dupes)} duplicate relative_path entries: "
                            f"{dupes['relative_path'].head(5).tolist()}")
        empty = frame[frame["exists"] & (frame["size_bytes"] == 0)]
        if len(empty):
            problems.append(f"{len(empty)} registered artifact(s) are zero bytes: "
                            f"{empty['relative_path'].head(5).tolist()}")
        bad_desc = frame[frame["description"].astype(str).str.len() == 0]
        if len(bad_desc):
            problems.append(f"{len(bad_desc)} artifact(s) have no description")
        return problems

    def summary(self) -> Dict[str, Any]:
        frame = self.to_frame()
        if frame.empty:
            return {"n_artifacts": 0}
        return {
            "n_artifacts": int(len(frame)),
            "n_existing": int(frame["exists"].sum()),
            "n_missing_but_registered": int((frame["created"] & ~frame["exists"]).sum()),
            "n_skipped": int((~frame["created"]).sum()),
            "by_type": frame["artifact_type"].value_counts().to_dict(),
            "by_section": frame["section"].value_counts().to_dict(),
            "by_importance": frame["paper_importance"].value_counts().to_dict(),
            "n_paper_candidates": int(frame["paper_candidate"].sum()),
            "total_size_bytes": int(frame["size_bytes"].sum()),
        }
