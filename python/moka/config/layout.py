"""What a folder looks like, without resolving it: confdir or rootdir (design-v7 §0, D2)."""

from __future__ import annotations

from pathlib import Path


def is_confdir(path: Path) -> bool:
    path = Path(path)
    return (
        (path / "conf.toml").is_file()
        and not (path / "provenance.toml").exists()
        and path.name.endswith("-conf")
    )


def is_rootdir(path: Path) -> bool:
    path = Path(path)
    return (
        (path / "provenance.toml").is_file()
        and (path / "conf" / "conf.toml").is_file()
        and path.name.endswith("-rootdir")
    )


def list_pipeline_dirs(folder: Path) -> list[str]:
    """Sorted names of the confdirs and rootdirs in folder, whatever their names."""
    return sorted(
        p.name for p in Path(folder).iterdir() if is_confdir(p) or is_rootdir(p)
    )
