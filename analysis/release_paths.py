"""Resolve the original artifact locations used by the analysis scripts to the
public release layout.

The analysis scripts were written against the development tree. Rather than
editing their logic, every input path is expressed as a ``LegacyPath`` and
resolved through ``legacy_path_map.json`` (original location -> public file).
Implementation files resolve to ``src/``. Nothing is recomputed here.
"""
from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
_MAP = json.loads((Path(__file__).with_name("legacy_path_map.json")).read_text(encoding="utf-8"))


def resolve(legacy: str) -> Path:
    key = legacy.replace("\\", "/").strip("/")
    if key == "":
        return ROOT / "src"          # the original repository root = the implementation workspace
    if key in _MAP:
        return ROOT / _MAP[key]
    if key.split("/", 1)[0] in ("models", "experiments", "scripts"):
        return ROOT / "src" / key
    raise FileNotFoundError("not part of the public release: %s" % key)


class LegacyPath(os.PathLike):
    """A path in the original layout; resolves lazily to the public file."""

    def __init__(self, legacy: str = ""):
        self._legacy = str(legacy).replace("\\", "/").strip("/")

    def __truediv__(self, other):
        o = str(other).replace("\\", "/").strip("/")
        return LegacyPath("%s/%s" % (self._legacy, o) if self._legacy else o)

    def real(self) -> Path:
        return resolve(self._legacy)

    def __fspath__(self):
        return str(self.real())

    __str__ = __fspath__

    def __repr__(self):
        return "LegacyPath(%r)" % self._legacy

    def read_text(self, *a, **k):
        return self.real().read_text(*a, **k)

    def read_bytes(self):
        return self.real().read_bytes()

    def open(self, *a, **k):
        return self.real().open(*a, **k)

    def exists(self):
        try:
            return self.real().exists()
        except FileNotFoundError:
            return False

    def is_file(self):
        try:
            return self.real().is_file()
        except FileNotFoundError:
            return False

    def relative_to(self, other):
        base = other._legacy if isinstance(other, LegacyPath) else ""
        return PurePosixPath(self._legacy).relative_to(base) if base else PurePosixPath(self._legacy)

    @property
    def name(self):
        return PurePosixPath(self._legacy).name

    @property
    def suffix(self):
        return PurePosixPath(self._legacy).suffix

    @property
    def stem(self):
        return PurePosixPath(self._legacy).stem

    @property
    def parent(self):
        p = str(PurePosixPath(self._legacy).parent)
        return LegacyPath("" if p == "." else p)

    @property
    def parents(self):
        out, cur = [], self
        while cur._legacy:
            cur = cur.parent
            out.append(cur)
        return out


LEGACY_ROOT = LegacyPath("")


def output_dir(kind: str) -> Path:
    p = ROOT / "analysis" / "output" / kind
    p.mkdir(parents=True, exist_ok=True)
    return p
