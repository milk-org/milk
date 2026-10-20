"""CONNECTIONS tree roots: a process shows its own streams only, an FPS has a tree (links-2, robust-18)."""

from __future__ import annotations

import time

import pytest

from tests.tui.milkctrl_loops_test import _ring

pytestmark = pytest.mark.timeout(60)


def test_unconnected_process_root_shows_no_streams(milkctrl, entities):
    for s in ["aa1", "zz9"]:
        entities.make_stream(s)
    entities.make_proc("loner")
    m = milkctrl()
    m.wait_for(r"loner")
    m.keys("Tab")  # DASH focus: STREAMS -> PROCS
    m.wait_for(r"No graph available")
    assert m.quit() == 0


def test_fps_root_has_tree(milkctrl, entities):
    _ring(entities)
    m = milkctrl(y=40)
    m.wait_for(r"fpsA")
    m.keys("Tab", "Tab")  # DASH focus: STREAMS -> PROCS -> FPS
    m.wait_for(
        r"lp[AB] \[L01\]"
    )  # tree rows carry the loop tag; the old code drew no tree
    time.sleep(0.3)
    assert "No graph available" not in m.screen()
    assert m.quit() == 0
