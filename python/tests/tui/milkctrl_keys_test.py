"""milk-CTRL help overlay and '?' behave as documented (ux-1, ux-18, arch-9)."""

from __future__ import annotations

import re
import time

import pytest

pytestmark = pytest.mark.timeout(60)


def _gone(m, text, timeout=5.0):
    deadline = time.monotonic() + timeout
    while text in m.screen():
        assert time.monotonic() < deadline, f"{text!r} still on screen"
        time.sleep(0.05)


def test_h_toggles_help(milkctrl):
    m = milkctrl(y=40)
    m.wait_for(r"F2:DASH")
    m.keys("h")
    m.wait_for(r"HELP & CONTROLS")
    m.keys("h")
    _gone(m, "HELP & CONTROLS")
    m.wait_for(r"F2:DASH")
    assert m.quit() == 0


def test_question_mark_opens_jump_prompt(milkctrl):
    m = milkctrl(y=40)
    m.wait_for(r"F2:DASH")
    m.keys("?")
    m.wait_for(r"\[STREAMS\] \?")
    assert "HELP & CONTROLS" not in m.screen()
    m.keys("Escape")
    _gone(m, "[STREAMS] ?")
    assert m.quit() == 0


def test_intro_scrolls_to_last_line(milkctrl):
    m = milkctrl(y=40)
    m.wait_for(r"F2:DASH")
    m.keys("h")
    m.wait_for(r"HELP & CONTROLS")
    m.keys("1")
    m.wait_for(r"ABOUT milk-CTRL")
    m.keys("End")
    m.wait_for(r"Quit milk-CTRL cleanly")
    m.keys("Escape")
    _gone(m, "ABOUT milk-CTRL")
    assert m.quit() == 0


def test_status_labels_match_tabs(milkctrl):
    """Breadcrumb uses the tab names; the sub-tab badge names the tab shown (it said DETAIL for LOOPS)."""
    m = milkctrl()
    m.wait_for(r"DASH ›")
    assert "[CONN]" in m.status()
    m.keys("BTab")  # next DASH sub-tab: LOOPS
    m.wait_for(r"\[LOOPS\]")
    m.keys("F3")
    m.wait_for(r"STRM ›")
    assert "[LOOPS]" not in m.status()  # no graph panel in F3
    assert " p c " not in m.status()
    assert re.search(r"self\s+\d+\.\d%", m.header())
    assert m.quit() == 0
