"""Operate-layer fixtures (R4): a self-cleaning SHM dir, a fixed boot id, and a simulated FPS world.

Nothing here imports milk.operate at module level, so the plugin loads before R4a exists.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def shm_dir_cleanafter() -> Iterator[Path]:
    """The session $MILK_SHM_DIR; whatever the test adds to it is removed afterwards.

    Never repoint MILK_SHM_DIR per test: libfps reads it once per process.
    """
    from milk import env

    path = env.shm_dir()
    before = set(path.rglob("*"))
    yield path
    for p in sorted(set(path.rglob("*")) - before, reverse=True):  # children first
        if p.is_dir() and not p.is_symlink():
            p.rmdir()
        else:
            p.unlink()


@pytest.fixture
def boot(monkeypatch):
    """Pin the boot id to "boot-A". Tests call boot("boot-B") to simulate a reboot."""

    def set_boot(value: str) -> None:
        monkeypatch.setattr("milk.operate.rundir.current_boot_id", lambda: value)

    set_boot("boot-A")
    return set_boot


@pytest.fixture
def paper(monkeypatch) -> dict[str, tuple[bool, bool]]:
    """Simulated FPS world for py tests: {fps_name: (conf_running, run_running)}.

    state.fps_status reads it, so a name that is absent here is an FPS that does not exist.
    """
    world: dict[str, tuple[bool, bool]] = {}
    monkeypatch.setattr("milk.operate.state.fps_status", lambda name: world.get(name))
    return world
