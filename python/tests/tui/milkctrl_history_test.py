"""Per-item history survives scans: a stalled loop turns STALE (robust-13, links-3)."""

from __future__ import annotations

import os
import signal
import time

import pytest

pytestmark = pytest.mark.timeout(60)

STALE_BG = "48;2;55;45;20"  # dark theme bg_stale


def _row_ansi(m, name):
    for line in m.screen_ansi().splitlines():
        if name in line:
            return line
    return ""


def _wait_row_bg(m, name, bg, present=True, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if (bg in _row_ansi(m, name)) == present:
            return
        time.sleep(0.1)
    raise AssertionError(
        f"{name}: bg {bg} present={present} not reached:\n{_row_ansi(m, name)}"
    )


def test_stalled_loop_turns_stale(milkctrl, entities):
    entities.make_proc("aaa_cursor")  # sorted first: holds the selection highlight
    pid = entities.make_proc("stallme")
    m = milkctrl()
    m.keys("F4")
    m.wait_for(r"stallme")
    os.kill(pid, signal.SIGUSR1)  # loopcnt freezes, loopstat stays ACTIVE
    _wait_row_bg(m, "stallme", STALE_BG)  # after OV_STALE_SCANS scans
    m.wait_for(r"stallme.*STAL")
    os.kill(pid, signal.SIGUSR1)  # resumes
    _wait_row_bg(m, "stallme", STALE_BG, present=False)
    assert m.quit() == 0
