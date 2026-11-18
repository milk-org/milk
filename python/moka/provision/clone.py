"""clone: copy a confdir's conf files, nothing else (design-v7 §0, D-8)."""

from __future__ import annotations

import shutil
from pathlib import Path

from ..errors import ProvisionError
from .artifacts import conf_files


def clone(
    src: Path, dest_parent: Path, update: bool = False, override: bool = False
) -> Path:
    """Copy conf_files(src) to dest_parent / src.name. Runs no hooks and no git.

    update: copy over an existing target. override: also first delete everything
    in the target but its .git, so it ends up as a fresh clone.
    """
    src = Path(src)
    dest = Path(dest_parent) / src.name
    if dest.is_symlink() or (dest.exists() and not (update or override)):
        raise ProvisionError(f"Clone target {dest} already exists.", path=dest)
    if override and dest.exists():
        for entry in dest.iterdir():
            if entry.name == ".git":
                continue
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()
    for rel in conf_files(src):
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel, dest / rel)
    (dest / ".gitignore").write_text("data/\n")
    return dest
