"""init, drift and resync: the rootdir as the authoritative copy of a confdir.

Artifact keys: a conf file's relpath, or ``data/<product>`` for a hook product. The source side
is always ``confdir/<key>``; the rootdir side is ``conf/<key>`` or ``initdata/<product>``.
The exception is DMMASK, generated in the rootdir and recorded with the digest of [dmhint].
"""

from __future__ import annotations

import importlib.metadata
import os
import shutil
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..config.model import PipelineConfig
from ..config.resolve import confresolve
from ..config.resolve import IDENTITY_KEYS
from ..config.substitute import denest_toml_dicts
from ..errors import ProvisionError
from .artifacts import conf_files, decl_digest, fingerprint, remove, status
from .hooks import _tagfile, consolidate, read_tag
from .provenance import ArtifactRecord, Provenance, read_provenance, write_provenance

DMMASK = "dmmask_hinted.fits"  # the [dmhint] mask as uint8, in rootdir/conf


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
    moka_version: tuple[str, str] | None  # (recorded, installed) if they differ


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


def _dmhint_digest(config: PipelineConfig) -> str | None:
    return config.dmhint and decl_digest(config.dmhint.model_dump(mode="json"))


def _dmmask_status(record: ArtifactRecord | None, digest: str | None) -> str:
    """Source status of DMMASK: its record against the [dmhint] it would be made from now."""
    if digest is None:
        return "removed" if record else "unchanged"
    if record is None:
        return "added"
    return "unchanged" if record.decl_digest == digest else "stale"


def _put_dmmask(rootdir: Path) -> ArtifactRecord:
    """Write DMMASK from the rootdir's [dmhint]: actuators strictly within the beam, all
    without one. Return its record."""
    import numpy as np
    from astropy.io import fits

    config = confresolve(rootdir)
    hint = config.dmhint
    if hint.beam_rad is None:
        mask = np.ones((hint.ysize, hint.xsize), bool)
    else:
        from pycacao.calib.mkmodes import make_disk

        beam = hint.beam_xcent, hint.beam_ycent, hint.beam_rad
        mask = make_disk(beam, (hint.xsize, hint.ysize))
    path = _local(rootdir, DMMASK)
    remove(path)
    fits.writeto(path, mask.astype("uint8"))
    fp = fingerprint(path)
    return ArtifactRecord(
        content_hash=fp.content_hash,
        size=fp.size,
        decl_digest=_dmhint_digest(config),
    )


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
        moka_version=importlib.metadata.version("moka"),
        source_confdir=str(confdir.resolve()),
        initialized=_now(),
        overrides=dict(overrides or {}),
        artifacts=records,
    )
    write_provenance(rootdir, prov)
    if config.dmhint is not None:
        prov.artifacts[DMMASK] = _put_dmmask(rootdir)
        write_provenance(rootdir, prov)
    return rootdir


def drift(rootdir: Path) -> DriftReport:
    """What differs between the rootdir, its records and the confdir. Reads only."""
    rootdir = Path(rootdir)
    return _drift(rootdir, read_provenance(rootdir))


def _drift(rootdir: Path, prov: Provenance) -> DriftReport:
    confdir = Path(prov.source_confdir)
    here = confresolve(rootdir)
    there = confresolve(confdir, prov.overrides)
    decls = {
        f"data/{decl.product}": decl_digest(decl.model_dump(mode="json"))
        for decl in confresolve(confdir).hooks
    }
    sources = {rel.as_posix() for rel in conf_files(confdir)} | decls.keys()
    if there.dmhint is not None:
        sources.add(DMMASK)
    locals_ = {rel.as_posix() for rel in conf_files(rootdir / "conf")}
    rows = []
    for key in sorted(prov.artifacts.keys() | sources | locals_):
        record = prov.artifacts.get(key)
        recorded = record and record.fingerprint
        local = _local(rootdir, key)
        local = fingerprint(local) if os.path.lexists(local) else None
        if key == DMMASK:
            source = _dmmask_status(record, _dmhint_digest(there))
        else:
            current = fingerprint(confdir / key) if key in sources else None
            source = status(
                recorded, current, record and record.decl_digest, decls.get(key)
            )
        rows.append(ArtifactDrift(key, source, status(recorded, local)))
    installed = importlib.metadata.version("moka")
    return DriftReport(
        rows,
        _diff(here, there),
        _structural(here, there),
        None if prov.moka_version == installed else (prov.moka_version, installed),
    )


def resync(
    rootdir: Path,
    *,
    run_active: Callable[[], bool],
    artifacts: Collection[str] | None = None,
    overrides: Mapping[str, object] | None = None,
    overwrite_local: bool = False,
) -> ResyncReport:
    """Bring the drifted confdir artifacts (default: all) into the rootdir; never deploys."""
    rootdir = Path(rootdir)
    prov = read_provenance(rootdir)
    confdir = Path(prov.source_confdir)
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

    # The merged overrides count for DMMASK, made from the resolved [dmhint].
    prov.overrides = merged
    applied, refused, put_dmmask = [], {}, False
    for row in _drift(rootdir, prov).artifacts:
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
        elif row.key == DMMASK:
            # Made from the resynced conf, once every conf file is in.
            put_dmmask = True
            continue
        else:
            prov.artifacts[row.key] = _put(confdir, rootdir, row.key)
        applied.append(row.key)

    prov.last_resync = _now()
    write_provenance(rootdir, prov)
    after = confresolve(rootdir)
    if put_dmmask and _dmhint_digest(after) != _dmhint_digest(target):
        refused[DMMASK] = "rootdir [dmhint] differs from the confdir; resync conf first"
    elif put_dmmask:
        prov.artifacts[DMMASK] = _put_dmmask(rootdir)
        write_provenance(rootdir, prov)
        applied.append(DMMASK)
    return ResyncReport(
        applied, refused, _diff(before, after), _structural(before, after)
    )
