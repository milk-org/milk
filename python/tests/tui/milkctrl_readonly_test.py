"""Passive viewing never writes the processinfo registry; CONTROL 'C' settles dead slots on request.

Findings: gap-2, links-4, robust-6.
"""

from __future__ import annotations

import mmap
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.timeout(60)

LISTSIZE = 50000  # PROCESSINFOLISTSIZE
ACTIVE_OFF = 4 * LISTSIZE  # PIDarray int32[], then active int32[]


def _registry(shm: str) -> Path:
    return Path(shm) / "processinfo.list.shm"


def _slot_state(shm: str, pid: int) -> int:
    with open(_registry(shm), "rb") as f:
        m = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        try:
            for i in range(LISTSIZE):
                if int.from_bytes(m[4 * i : 4 * i + 4], "little", signed=True) == pid:
                    off = ACTIVE_OFF + 4 * i
                    return int.from_bytes(m[off : off + 4], "little", signed=True)
        finally:
            m.close()
    raise AssertionError(f"PID {pid} not in registry")


def _killed_proc(entities, name: str) -> int:
    """A registered process killed with SIGKILL: its slot stays active=1 with a proc file."""
    pid = entities.make_proc(name)
    entities.kill(pid)
    return pid


def test_viewer_never_creates_registry(milkctrl, tui_env):
    _, shm = tui_env
    m = milkctrl()
    m.wait_for(r"F2:DASH")
    time.sleep(1.5)  # at least one full scan
    assert not _registry(shm).exists()
    assert m.quit() == 0


def test_viewer_leaves_registry_bytes_unchanged(milkctrl, entities, tui_env):
    _, shm = tui_env
    pid = _killed_proc(entities, "rodead")
    before = _registry(shm).read_bytes()
    m = milkctrl()
    m.keys("F4")
    m.wait_for(r"rodead.*CRSH")
    time.sleep(1.5)
    assert _registry(shm).read_bytes() == before
    assert _slot_state(shm, pid) == 1
    assert m.quit() == 0


def test_viewer_works_with_readonly_files(milkctrl, entities, tui_env):
    _, shm = tui_env
    entities.make_proc("roproc")
    for f in Path(shm).glob("*.shm"):
        f.chmod(0o444)
    m = milkctrl()
    m.keys("F4")
    m.wait_for(r"roproc")
    assert m.quit() == 0


def test_cleanup_settles_dead_slot(milkctrl, entities, tui_env):
    _, shm = tui_env
    pid = _killed_proc(entities, "reapme")
    m = milkctrl()
    m.keys("F4")
    m.wait_for(r"reapme.*CRSH")
    m.keys("c")
    m.wait_for(r"CONTROL")
    m.keys("C")
    m.wait_for(r"registry slot\(s\) settled")
    assert _slot_state(shm, pid) == 3  # CRASHED: file kept, loopstat was not STOP
    assert m.quit() == 0
