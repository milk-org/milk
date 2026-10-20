"""CONNECTIONS tree roots: a process shows its own streams only, an FPS has a tree (links-2, robust-18)."""

from __future__ import annotations

import re
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


def _chain(entities):
    """Declared chain ca -> fpsA -> cb -> fpsB -> cc, plus an unrelated zz9."""
    for s in ["ca", "cb", "cc", "zz9"]:
        entities.make_stream(s)
    entities.make_fps("fpsA", "streamin:STREAMNAME:ca", "streamout:STREAMNAME:cb")
    entities.make_fps("fpsB", "streamin:STREAMNAME:cb", "streamout:STREAMNAME:cc")


def _preview(m):
    return m.lines()[2]


def _graph_first_row(m):
    """Text of the first CONNECTIONS row: the line below the MODE header, right panel."""
    lines = m.lines()
    for i, line in enumerate(lines):
        col = line.find("MODE ")
        if col >= 0:
            return lines[i + 1][col - 1 :]
    return ""


def _graph_enter_third_row(m):
    m.keys("L", "L")  # lineage mode Trigger -> Input -> Full
    m.wait_for(r"MODE\s+Full")
    m.keys("Tab", "Tab", "Tab")  # DASH focus: STREAMS -> PROCS -> FPS -> GRAPH
    time.sleep(0.3)
    m.keys("Down", "Down")
    time.sleep(0.3)
    assert "ca" in _graph_first_row(
        m
    ), m.screen()  # the cursor does not re-root the tree
    m.keys("Enter")
    deadline = time.monotonic() + 5
    while "cc" not in _preview(m) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert "cc" in _preview(m), _preview(m)


def test_graph_cursor_does_not_reroot(milkctrl, entities):
    _chain(entities)
    m = milkctrl(y=40)
    m.wait_for(r"fpsB")
    _graph_enter_third_row(m)
    assert m.quit() == 0


def test_graph_enter_under_filter(milkctrl, entities):
    _chain(entities)
    m = milkctrl(y=40)
    m.wait_for(r"fpsB")
    m.keys("/")
    m.text("ca|cc")
    m.keys("Enter")
    m.wait_for(r"STREAMS regex filter applied")
    _graph_enter_third_row(m)
    assert m.quit() == 0


def test_jump_search_sticks(milkctrl, entities):
    for s in ["aa1", "aa2", "zz9"]:
        entities.make_stream(s)
    m = milkctrl()
    m.wait_for(r"zz9")
    m.keys("?")
    m.text("zz")
    m.keys("Enter")
    m.wait_for(r"Jumped to stream: zz9")
    time.sleep(3.5)  # several scans
    assert "zz9" in _preview(m), _preview(m)
    assert m.quit() == 0
