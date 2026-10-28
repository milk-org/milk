"""Consolidated tests for milk-CTRL TUI (views, keys, colours, read-only safety, FPS edits, and loops)."""

from __future__ import annotations

import mmap
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

OVERVIEW = Path(__file__).resolve().parents[3] / "src" / "cli" / "overview"
EMOJI = re.compile(r"[\U0001F000-\U0001FFFF☀-✒✘-➿️ℹ⏸]")
ESCAPED_EMOJI = re.compile(r"\\xf0\\x9f|\\xe2\\x9a|\\xe2\\x84\\xb9|\\xe2\\x99")
_GET = "import sys; from pyMilk.interfacing.fps import FPS; print(FPS(sys.argv[1]).get_param(sys.argv[2]))"

pytestmark = pytest.mark.timeout(60)


def _get_param(env, fps, key) -> str:
    r = subprocess.run(
        [sys.executable, "-c", _GET, fps, key],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""


def _edit_gain(m, value: str) -> None:
    m.keys("Enter")
    m.keys("Enter")
    m.wait_for(r"new value")
    m.text(value)
    m.keys("Enter")


def _ring(entities):
    for s in ["aa1", "aa2", "lpA", "lpB", "zz9"]:
        entities.make_stream(s)
    entities.make_fps("fpsA", "streamin:STREAMNAME:lpB", "streamout:STREAMNAME:lpA")
    entities.make_fps("fpsB", "streamin:STREAMNAME:lpA", "streamout:STREAMNAME:lpB")


def test_no_emoji_in_overview_sources():
    bad = []
    for ext in ("*.c", "*.h", "*.inc"):
        for p in OVERVIEW.rglob(ext):
            if "tests" in p.parts:
                continue
            for n, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
                if EMOJI.search(line) or ESCAPED_EMOJI.search(line):
                    bad.append(f"{p.name}:{n}: {line.strip()}")
    assert not bad, "\n".join(bad)


def test_smoke_navigation_and_statusbar(milkctrl, entities):
    entities.make_proc("jkproc")
    m = milkctrl(y=40)
    m.wait_for(r"F2:DASH")
    m.keys("F4")
    m.wait_for(r"jkproc")
    m.keys("j")
    m.wait_for(r"Unmapped key: 'j'")
    m.keys("k")
    m.wait_for(r"'k' requires CONTROL mode")
    m.keys("c")
    m.wait_for(r"Control mode ON")
    assert m.quit() == 0


def test_colour_modes_and_night_theme(milkctrl, entities):
    entities.make_stream("nightsel")
    m = milkctrl("-T", "night")
    m.wait_for(r"nightsel")
    time.sleep(0.3)
    s = " ".join(re.findall(r"\x1b\[([0-9;]*)m", m.screen_ansi()))
    for fixed in ("20;180;20", "255;190;0", "240;175;20", "30;120;60", "50;90;160"):
        assert fixed not in s, fixed
    assert m.quit() == 0


def test_viewer_readonly_and_cleanup(milkctrl, entities, tui_env):
    _, shm = tui_env
    reg = Path(shm) / "processinfo.list.shm"
    pid = entities.make_proc("reapme")
    entities.kill(pid)
    before = reg.read_bytes()
    m = milkctrl()
    m.keys("F4")
    m.wait_for(r"reapme.*CRSH")
    time.sleep(1.0)
    assert reg.read_bytes() == before
    m.keys("c")
    m.wait_for(r"CONTROL")
    m.keys("C")
    m.wait_for(r"CONFIRM")
    m.keys("y")
    m.wait_for(r"registry slot\(s\) settled")
    assert m.quit() == 0


def test_fps_remap_and_param_edit(milkctrl, entities, tui_env):
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
    m.keys("Escape", "Left")
    _edit_gain(m, "1.5x")
    m.wait_for(r"not written: not a number")
    assert float(_get_param(env, "edfps", "gain")) == pytest.approx(1.25)
    m.keys("Escape")
    assert m.quit() == 0


def test_f7_loops_sort_and_rename(milkctrl, entities):
    _ring(entities)
    m = milkctrl(y=40)
    m.keys("F7")
    m.wait_for(r"L01 L01")
    row = next(line for line in m.lines() if "L01 L01" in line)
    assert "IDLE" in row and "RUN" not in row, row
    m.keys("r")
    m.keys("C-u")
    m.text("myloop")
    m.keys("Enter")
    m.wait_for(r"renamed to 'myloop'")
    m.wait_for(r"L01 myloop", timeout=1.0)
    assert m.quit() == 0
