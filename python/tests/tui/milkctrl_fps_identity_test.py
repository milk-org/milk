"""FPS files are mapped read-only by identity; a re-created FPS is remapped and edits reach it.

Findings: robust-5, control-6, links-6, gap-2.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.timeout(60)

_GET = "import sys; from pyMilk.interfacing.fps import FPS; print(FPS(sys.argv[1]).get_param(sys.argv[2]))"


def _get_param(env, fps, key) -> str:
    r = subprocess.run(
        [sys.executable, "-c", _GET, fps, key],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""


def _clock(m) -> str:
    found = re.findall(r"\d\d:\d\d:\d\d", m.status())
    return found[-1] if found else ""


def test_scan_keeps_running_with_fps(milkctrl, entities):
    entities.make_fps("liveps", "gain:FLOAT64:0.5")
    m = milkctrl()
    first = _clock(m)
    deadline = time.monotonic() + 6
    while _clock(m) == first:
        assert time.monotonic() < deadline, "status clock stopped: scan thread stuck"
        time.sleep(0.2)
    assert m.quit() == 0


def test_recreated_fps_is_remapped(milkctrl, entities):
    entities.make_fps("refps", "gain:FLOAT64:0.5")
    m = milkctrl()
    m.keys("F5")
    m.wait_for(r"gain\s+FLT\s+0\.5")
    entities.make_fps("refps", "gain:FLOAT64:2.5")  # remove + create: new inode
    m.wait_for(r"gain\s+FLT\s+2\.5")
    assert m.quit() == 0


def _edit_gain(m, value: str) -> None:
    m.keys("Enter")  # into the parameter panel; gain is the first row
    m.keys("Enter")
    m.wait_for(r"new value")
    m.text(value)
    m.keys("Enter")


def test_param_edit_after_recreate_reaches_live_fps(milkctrl, entities, tui_env):
    env, _ = tui_env
    entities.make_fps("edfps", "gain:FLOAT64:0.5")
    m = milkctrl()
    m.keys("c")
    m.wait_for(r"CONTROL")
    m.keys("F5")
    m.wait_for(r"gain\s+FLT\s+0\.5")
    _edit_gain(m, "1.25")
    m.wait_for(r"gain\s+FLT\s+1\.25")
    assert float(_get_param(env, "edfps", "gain")) == pytest.approx(1.25)

    entities.make_fps("edfps", "gain:FLOAT64:0.5")  # re-created under the same name
    m.wait_for(r"gain\s+FLT\s+0\.5")
    m.keys("Escape", "Left")  # back to the FPS list before re-entering
    _edit_gain(m, "2.5")
    m.wait_for(r"gain\s+FLT\s+2\.5")
    assert float(_get_param(env, "edfps", "gain")) == pytest.approx(2.5)
    m.keys("Escape")
    assert m.quit() == 0
