"""Actions hit the highlighted row; a vanished target moves the cursor to its neighbour.

Findings: control-1, control-2, robust-4 (identity), ux-7.
"""

from __future__ import annotations

import os
import re
import time

import pytest

from tests.tui.milkctrl_loops_test import _ring

pytestmark = pytest.mark.timeout(60)


def _first_row(m, names):
    """Name of the first data row of the visible list panel among @names."""
    for line in m.lines():
        for n in names:
            if re.search(rf"[║│]\s*\S*\s*{n}\s", line):
                return n
    return None


def test_delete_hits_highlighted_row_under_loop_isolation(milkctrl, entities, tui_env):
    _, shm = tui_env
    _ring(entities)
    m = milkctrl(y=40)
    m.keys("F7")
    m.wait_for(r"L01")
    m.keys("Enter")  # isolate the loop: STREAMS shows only lpA and lpB
    m.wait_for(r"Loop isolation filter: ON")
    m.keys("F3")
    m.wait_for(r"lpA")
    time.sleep(0.3)
    assert "aa1" not in m.screen() and "zz9" not in m.screen()
    target = _first_row(m, ["lpA", "lpB"])
    assert target is not None
    m.keys("c")
    m.wait_for(r"CONTROL")
    m.keys("DC")
    m.wait_for(r"CONFIRM")
    m.keys("y")
    m.wait_for(rf"\"{target}\".*deleted")
    assert not os.path.exists(os.path.join(shm, f"{target}.im.shm"))
    for keep in ["aa1", "aa2", "zz9"]:
        assert os.path.exists(os.path.join(shm, f"{keep}.im.shm"))
    assert m.quit() == 0


def test_vanished_target_moves_to_neighbour(milkctrl, entities, tui_env):
    _, shm = tui_env
    for s in ["t1", "t2", "t3", "t4", "t5"]:
        entities.make_stream(s)
    m = milkctrl()
    m.keys("F3")
    m.wait_for(r"t5")
    m.keys("Down", "Down")  # third row
    time.sleep(0.3)
    third = None
    rows = [ln for ln in m.lines() if re.search(r"[║│]\s*\S*\s*t\d\s", ln)]
    third = re.search(r"(t\d)", rows[2]).group(1)
    first = re.search(r"(t\d)", rows[0]).group(1)
    os.unlink(os.path.join(shm, f"{third}.im.shm"))
    mm = m.wait_for(rf"\"{third}\" is gone; cursor on \"(t\d)\"")
    assert mm.group(1) != first  # neighbour, not back to row 0
    assert m.quit() == 0
