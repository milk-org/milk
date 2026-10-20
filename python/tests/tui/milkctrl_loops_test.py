"""F7 loop member tables follow the entities when the UI re-sorts them (arch-1, robust-12, links-9)."""

from __future__ import annotations

import re
import time

import pytest

pytestmark = pytest.mark.timeout(60)


def _ring(entities):
    for s in ["aa1", "aa2", "lpA", "lpB", "zz9"]:
        entities.make_stream(s)
    entities.make_fps("fpsA", "streamin:STREAMNAME:lpB", "streamout:STREAMNAME:lpA")
    entities.make_fps("fpsB", "streamin:STREAMNAME:lpA", "streamout:STREAMNAME:lpB")


def _check_f7(m):
    m.keys("F7")
    m.wait_for(r"lpA\s+F32")
    screen = m.screen()
    assert "lpB" in screen
    assert "aa1" not in screen and "aa2" not in screen and "zz9" not in screen


def test_f7_members_survive_sort(milkctrl, entities):
    """Both directions: whichever one differs from the scan order exposes stale indices."""
    _ring(entities)
    m = milkctrl(y=40)
    for arrow in ["▼", "▲"]:
        m.keys("F3")
        m.wait_for(r"zz9")
        m.keys("[")
        m.wait_for(rf"NAME{arrow}")
        _check_f7(m)
    assert m.quit() == 0


def test_declared_fps_ring_is_idle(milkctrl, entities):
    """An FPS ring with no run process is IDLE, not RUN by vacuous truth (links-10)."""
    _ring(entities)
    m = milkctrl(y=40)
    m.keys("F7")
    m.wait_for(r"L01 L01")
    row = next(line for line in m.lines() if "L01 L01" in line)
    assert "IDLE" in row and "RUN" not in row, row
    assert m.quit() == 0


def test_loop_rename_shows_at_once_and_persists(milkctrl, entities):
    """Renaming updates the saved-name registry only; the display reads it, before and after rescans."""
    _ring(entities)
    m = milkctrl(y=40)
    m.keys("F7")
    m.wait_for(r"L01 L01")
    m.keys("r")
    m.keys("C-u")
    m.text("myloop")
    m.keys("Enter")
    m.wait_for(r"renamed to 'myloop'")
    m.wait_for(r"L01 myloop", timeout=1.0)
    time.sleep(2.5)  # two rescans rebuild the loops from the registry
    assert re.search(r"L01 myloop", m.screen())
    assert m.quit() == 0


def test_f7_lists_fps_members(milkctrl, entities):
    """The circuit pane lists the loop's FPS with their conf/run state."""
    _ring(entities)
    m = milkctrl(y=40)
    m.keys("F7")
    m.wait_for(r"FPS\s+CONF/RUN\s+STATUS")
    m.wait_for(r"fps[AB]\s+off /off\s+OFF.*on the cycle")
    assert m.quit() == 0
