"""Harness smoke tests: milk-CTRL starts in an isolated environment, shows entities and quits cleanly."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.timeout(60)


def test_streams_view_lists_stream(milkctrl, entities):
    entities.make_stream("tuismoke")
    m = milkctrl()
    m.keys("F3")
    m.wait_for(r"tuismoke")
    assert m.quit() == 0


def test_process_and_fps_are_listed(milkctrl, entities):
    entities.make_proc("tuiproc")
    entities.make_fps("tuifps", "gain:FLOAT64:0.5")
    m = milkctrl()
    m.keys("F4")
    m.wait_for(r"tuiproc")
    m.keys("F5")
    m.wait_for(r"tuifps")
    assert m.quit() == 0
