"""init, drift and resync: the rootdir as the authoritative copy of a confdir (design-v7 §0, §4.0).

Artifact keys: a conf file's relpath, or ``data/<product>`` for a hook product. The source side
is always ``confdir/<key>``; the rootdir side is ``conf/<key>`` or ``initdata/<product>``.
"""

from __future__ import annotations

import importlib.metadata
import os
import shutil
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..config import PipelineConfig, confresolve
from ..config.resolve import IDENTITY_KEYS
from ..config.substitute import denest_toml_dicts
from ..errors import ProvisionError
from .artifacts import conf_files, decl_digest, fingerprint, remove, status
from .hooks import _tagfile, consolidate, read_tag
from .provenance import ArtifactRecord, Provenance, read_provenance, write_provenance


@dataclass(frozen=True)
class ArtifactDrift:
    key: str
    source: str  # recorded vs confdir current
    local: str  # recorded vs rootdir current


@dataclass(frozen=True)
class DriftReport:
    artifacts: list[ArtifactDrift]
    resolved_diff: dict[str, tuple[object, object]]  # (rootdir, confdir)
    structural: list[str]
    milk_version: tuple[str, str] | None  # (recorded, installed) if they differ


@dataclass
class ResyncReport:
    applied: list[str]
    refused: dict[str, str]
    resolved_diff: dict  # (before, after)
    structural: list[str]


def _is_hook(key: str) -> bool:
    return key.startswith("data/")


def _local(rootdir: Path, key: str) -> Path:
    if _is_hook(key):
        return rootdir / "initdata" / key.removeprefix("data/")
    return rootdir / "conf" / key


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _pairs(confdir: Path, rootdir: Path, key: str) -> list[tuple[Path, Path]]:
    """(source, rootdir) paths of one artifact, and of its tag for a hook."""
    src, dst = confdir / key, _local(rootdir, key)
    return (
        [(src, dst), (_tagfile(src), _tagfile(dst))] if _is_hook(key) else [(src, dst)]
    )


def _put(confdir: Path, rootdir: Path, key: str) -> ArtifactRecord:
    """Copy one artifact (and a hook's tag) into the rootdir; return its record."""
    for a, b in _pairs(confdir, rootdir, key):
        remove(b)
        b.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(a, b) if a.is_dir() else shutil.copy2(a, b)
    if _is_hook(key):
        return ArtifactRecord.model_validate(read_tag(confdir / key))
    fp = fingerprint(confdir / key)
    return ArtifactRecord(content_hash=fp.content_hash, size=fp.size)


def _diff(a: PipelineConfig, b: PipelineConfig) -> dict[str, tuple[object, object]]:
    fa, fb = (
        denest_toml_dicts(c.model_dump(mode="json", exclude_none=True)) for c in (a, b)
    )
    return {
        k: (fa.get(k), fb.get(k)) for k in sorted(fa | fb) if fa.get(k) != fb.get(k)
    }


def _structural(a: PipelineConfig, b: PipelineConfig) -> list[str]:
    lines = []
    for cu in sorted(a.compute_units | b.compute_units):
        old, new = a.compute_units.get(cu), b.compute_units.get(cu)
        if old is None:
            lines.append(f"compute unit {cu}: added")
        elif new is None:
            lines.append(f"compute unit {cu}: removed")
        elif old.exec != new.exec:
            lines.append(f"compute unit {cu}: exec {old.exec} -> {new.exec}")
    return lines


def init(
    confdir: Path,
    parent: Path | None = None,
    overrides: Mapping[str, object] | None = None,
) -> Path:
    """Consolidate confdir and copy it into a new <parent>/<name>-rootdir."""
    confdir = Path(confdir)
    config = confresolve(confdir, overrides)
    rootdir = Path(parent or confdir.parent) / f"{config.name}-rootdir"
    if os.path.lexists(rootdir):
        raise ProvisionError(f"{rootdir} already exists; use resync.", path=rootdir)
    consolidate(confdir)
    keys = [rel.as_posix() for rel in conf_files(confdir)]
    keys += [f"data/{decl.product}" for decl in config.hooks]
    records = {key: _put(confdir, rootdir, key) for key in keys}
    for sub in ("initdata", "work", "logs"):
        (rootdir / sub).mkdir(parents=True, exist_ok=True)
    prov = Provenance(
        milk_version=importlib.metadata.version("milk"),
        source_confdir=str(confdir.resolve()),
        initialized=_now(),
        overrides=dict(overrides or {}),
        artifacts=records,
    )
    write_provenance(rootdir, prov)
    confresolve(rootdir)
    return rootdir


def drift(rootdir: Path, confdir: Path | None = None) -> DriftReport:
    """What differs between the rootdir, its records and the confdir. Reads only."""
    rootdir = Path(rootdir)
    prov = read_provenance(rootdir)
    confdir = Path(confdir or prov.source_confdir)
    decls = {
        f"data/{decl.product}": decl_digest(decl.model_dump(mode="json"))
        for decl in confresolve(confdir).hooks
    }
    sources = {rel.as_posix() for rel in conf_files(confdir)} | decls.keys()
    locals_ = {rel.as_posix() for rel in conf_files(rootdir / "conf")}
    rows = []
    for key in sorted(prov.artifacts.keys() | sources | locals_):
        record = prov.artifacts.get(key)
        recorded = record and record.fingerprint
        source = fingerprint(confdir / key) if key in sources else None
        local = _local(rootdir, key)
        local = fingerprint(local) if os.path.lexists(local) else None
        rows.append(
            ArtifactDrift(
                key,
                status(recorded, source, record and record.decl_digest, decls.get(key)),
                status(recorded, local),
            )
        )
    here = confresolve(rootdir)
    there = confresolve(confdir, prov.overrides)
    installed = importlib.metadata.version("milk")
    return DriftReport(
        rows,
        _diff(here, there),
        _structural(here, there),
        None if prov.milk_version == installed else (prov.milk_version, installed),
    )


def resync(
    rootdir: Path,
    *,
    run_active: Callable[[], bool],
    artifacts: Collection[str] | None = None,
    overrides: Mapping[str, object] | None = None,
    overwrite_local: bool = False,
    confdir: Path | None = None,
) -> ResyncReport:
    """Bring the drifted confdir artifacts (default: all) into the rootdir; never deploys."""
    rootdir = Path(rootdir)
    prov = read_provenance(rootdir)
    confdir = Path(confdir or prov.source_confdir)
    if run_active():
        raise ProvisionError("Cannot resync: run process active.", path=rootdir)
    merged = {**prov.overrides, **(overrides or {})}
    before = confresolve(rootdir)
    target = confresolve(confdir, merged)
    if any(getattr(before, k) != getattr(target, k) for k in IDENTITY_KEYS):
        raise ProvisionError(
            "Identity keys differ from the confdir; init a new rootdir instead.",
            path=rootdir,
        )
    consolidate(confdir)

    applied, refused = [], {}
    for row in drift(rootdir, confdir).artifacts:
        if row.source == "unchanged" or (
            artifacts is not None and row.key not in artifacts
        ):
            continue
        if row.local != "unchanged" and not overwrite_local:
            refused[row.key] = f"local edit ({row.local}); use overwrite_local"
            continue
        if row.source == "removed":
            for _, dst in _pairs(confdir, rootdir, row.key):
                remove(dst)
            prov.artifacts.pop(row.key, None)
        else:
            prov.artifacts[row.key] = _put(confdir, rootdir, row.key)
        applied.append(row.key)

    prov.overrides = merged
    prov.last_resync = _now()
    write_provenance(rootdir, prov)
    after = confresolve(rootdir)
    return ResyncReport(
        applied, refused, _diff(before, after), _structural(before, after)
    )
