"""clone: copy a confdir's conf files, nothing else (design-v7 §0, D-8)."""

from __future__ import annotations

import shutil
from pathlib import Path

from ..errors import ProvisionError
from .artifacts import conf_files


def clone(src: Path, dest_parent: Path) -> Path:
    """Copy conf_files(src) to dest_parent / src.name. Runs no hooks and no git."""
    src = Path(src)
    dest = Path(dest_parent) / src.name
    if dest.exists() or dest.is_symlink():
        raise ProvisionError(f"Clone target {dest} already exists.", path=dest)
    for rel in conf_files(src):
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel, dest / rel)
    (dest / ".gitignore").write_text("data/\n")
    return dest
