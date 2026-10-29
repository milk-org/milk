"""Operate-layer fixtures (R4): a private SHM dir, a fixed boot id, and a simulated FPS world.

Nothing here imports milk.operate at module level, so the plugin loads before R4a exists.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def shm(tmp_path, monkeypatch) -> Path:
    """A private $MILK_SHM_DIR for one test (rundirs land in shm/pipe_rundirs/)."""
    path = tmp_path / "shm"
    path.mkdir()
    monkeypatch.setenv("MILK_SHM_DIR", str(path))
    return path


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
