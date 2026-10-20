"""Status-bar buttons are hit-tested where they are drawn (ux-13)."""

from __future__ import annotations

import time

import pytest

pytestmark = pytest.mark.timeout(60)


def test_exit_button_click_exits(milkctrl):
    m = milkctrl()
    last = m.lines()[-1]
    col = last.index("[x] exit") + 1  # 1-based column of the drawn label
    m.click(col + 1, len(m.lines()))
    assert m.exit_code() == 0


def test_no_invisible_exit_area_when_cut_off(milkctrl):
    """At 80 columns the hints overflow and the buttons are not drawn: their old spot must not exit."""
    m = milkctrl(x=80, y=24)
    assert "[x] exit" not in m.lines()[-1]
    m.click(80 - 18 + 2, 24)
    time.sleep(0.5)
    assert m.alive()
    assert m.quit() == 0
