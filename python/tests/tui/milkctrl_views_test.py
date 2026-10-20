"""Keys act only on panels that are on screen (ux-4, ux-5): TAB, ENTER/D and ESC per view."""

from __future__ import annotations

import os
import time

import pytest

pytestmark = pytest.mark.timeout(60)


def test_tab_in_streams_view_keeps_stream_focus(milkctrl, entities, tui_env):
    _, shm = tui_env
    entities.make_stream("tabkeep")
    m = milkctrl()
    m.keys("F3")
    m.wait_for(r"tabkeep")
    m.keys("Tab")  # used to move focus to the hidden PROCS panel
    m.keys("c")
    m.wait_for(r"CONTROL")
    m.keys("C-e")
    m.wait_for(r"CONFIRM")
    m.keys("y")
    m.wait_for(r"\"tabkeep\".*deleted")
    assert not os.path.exists(os.path.join(shm, "tabkeep.im.shm"))
    assert m.quit() == 0


@pytest.mark.parametrize("view,key", [("F3", "Enter"), ("F4", "D")])
def test_details_keys_hint_outside_dash(milkctrl, entities, view, key):
    entities.make_stream("hintme")
    m = milkctrl()
    m.keys(view)
    time.sleep(0.3)
    m.keys(key)
    m.wait_for(r"DETAILS is shown in F2:DASH and F6:GRAPH")
    assert m.quit() == 0
