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


def test_f1_opens_help(milkctrl):
    """F1 is a help key (it used to log "Unmapped key")."""
    m = milkctrl(y=40)
    m.wait_for(r"F2:DASH")
    m.keys("F1")
    m.wait_for(r"HELP & CONTROLS")
    assert "Unmapped" not in m.screen()
    m.keys("h")
    _gone(m, "HELP & CONTROLS")
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


def test_toggle_keys_still_act(milkctrl):
    """Keys moved to binding tables keep their effect: compact, pause, CONTROL, G hides the log."""
    m = milkctrl(y=40)
    m.wait_for(r"F2:DASH")
    m.keys("d")
    m.wait_for(r"Compact mode ON")
    m.keys("F")
    m.wait_for(r"Display paused")
    m.keys("F")
    m.wait_for(r"Display resumed")
    m.keys("c")
    m.wait_for(r"Control mode ON")
    m.keys("G")
    _gone(m, "Control mode ON")
    assert m.quit() == 0


def test_j_and_k_are_not_navigation(milkctrl, entities):
    """No vi keys: j is unbound, k is SIGTERM on PROC/FPS and asks for CONTROL mode outside it."""
    entities.make_proc("jkproc")
    m = milkctrl(y=40)
    m.keys("F4")
    m.wait_for(r"jkproc")
    m.keys("j")
    m.wait_for(r"Unmapped key: 'j'")
    m.keys("k")
    m.wait_for(r"'k' requires CONTROL mode")
    assert m.quit() == 0
