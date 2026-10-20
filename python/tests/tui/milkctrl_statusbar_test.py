"""Status-bar buttons are hit-tested where they are drawn (ux-13)."""

from __future__ import annotations

import re

import pytest

pytestmark = pytest.mark.timeout(60)


def test_exit_button_click_exits(milkctrl):
    m = milkctrl()
    last = m.lines()[-1]
    col = last.index("[x] exit") + 1  # 1-based column of the drawn label
    m.click(col + 1, len(m.lines()))
    assert m.exit_code() == 0


def test_narrow_terminal_keeps_essentials(milkctrl):
    """At 80x24 the bars drop low-priority items: mode badge, HELP, exit button and clock stay (C1)."""
    m = milkctrl(x=80, y=24)
    lines = m.lines()
    assert "READ ONLY" in lines[0] or "R/O" in lines[0], lines[0]
    assert "[h: HELP]" in lines[1], lines[1]
    assert "[x] exit" in lines[-1] and re.search(r"\d\d:\d\d:\d\d", lines[-1]), lines[
        -1
    ]
    for line in (lines[0], lines[1], lines[-1]):
        assert len(line.rstrip()) <= 80


def test_narrow_header_badge_click_toggles_control(milkctrl):
    """The mode badge is clicked where the header drew it."""
    m = milkctrl(x=80, y=24)
    head = m.lines()[0]
    label = "READ ONLY" if "READ ONLY" in head else "R/O"
    m.click(head.index(label) + 2, 1)
    m.wait_for(r"CONTROL|CTRL ")
    assert m.quit() == 0


def test_narrow_exit_button_click_exits(milkctrl):
    m = milkctrl(x=80, y=24)
    last = m.lines()[-1]
    m.click(last.index("[x] exit") + 2, 24)
    assert m.exit_code() == 0
