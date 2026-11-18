"""Conf artifacts: fingerprints, the conf file set, and drift status (design-v7 §0, §4.9)."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from ..errors import ProvisionError


@dataclass(frozen=True)
class Fingerprint:
    content_hash: str  # "sha256:<hex>"
    size: int


def fingerprint(path: Path) -> Fingerprint:
    """Bytes-only fingerprint of a file or directory tree. Modes and mtime are ignored."""
    path = Path(path)
    if path.is_symlink():
        raise ProvisionError(f"Refusing symlink {path}.", path=path)
    if path.is_file():
        with open(path, "rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        return Fingerprint(f"sha256:{digest}", path.stat().st_size)
    lines, size = [], 0
    for sub in path.rglob("*"):
        rel = sub.relative_to(path).as_posix()
        if sub.is_dir() and not sub.is_symlink():
            if not any(sub.iterdir()):
                lines.append(f"{rel}/\0\n")
            continue
        sub_fp = fingerprint(sub)
        lines.append(f"{rel}\0{sub_fp.content_hash}\n")
        size += sub_fp.size
    digest = hashlib.sha256("".join(sorted(lines)).encode()).hexdigest()
    return Fingerprint(f"sha256:{digest}", size)


def conf_files(confdir: Path) -> list[Path]:
    """Sorted relpaths of the conf files: **/*.toml outside data/, plus every file under scripts/."""
    confdir = Path(confdir)
    tomls = [
        p for p in confdir.rglob("*.toml") if p.relative_to(confdir).parts[0] != "data"
    ]
    scripts = (confdir / "scripts").rglob("*")
    return sorted({p.relative_to(confdir) for p in [*tomls, *scripts] if p.is_file()})


def decl_digest(decl: dict) -> str:
    """Digest of a hook declaration, independent of key order."""
    text = json.dumps(decl, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def status(
    recorded: Fingerprint | None,
    current: Fingerprint | None,
    recorded_decl: str | None = None,
    current_decl: str | None = None,
) -> str:
    """Drift status of one artifact: changed bytes win over a changed declaration."""
    if recorded is None and current is None:
        return "unchanged"
    if current is None:
        return "removed"
    if recorded is None:
        return "added"
    if recorded != current:
        return "modified"
    if recorded_decl != current_decl:
        return "stale"
    return "unchanged"


def remove(path: Path) -> None:
    """Delete a file, symlink or directory tree, if present."""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)
