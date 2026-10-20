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


@pytest.mark.parametrize("label", ["F3:STRM", "F7:LOOPS"])
def test_tab_click_switches_view(milkctrl, label):
    """Tabs are hit-tested where the tab bar drew them."""
    m = milkctrl()
    m.wait_for(r"F2:DASH")
    tabs = m.lines()[1]
    m.click(tabs.index(label) + 2, 2)
    m.wait_for(rf"▌ {label} ▐")
    assert m.quit() == 0


def test_help_button_and_context(milkctrl):
    """The [h: HELP] button opens help, whose context names the current view (F7 used to say CONN)."""
    m = milkctrl()
    m.keys("F7")
    m.wait_for(r"▌ F7:LOOPS")
    tabs = m.lines()[1]
    m.click(tabs.index("h: HELP") + 2, 2)
    m.wait_for(r"Context: \[LOOPS /")
    m.keys("h")  # close help: q would only close the overlay
    m.wait_for(r"FEEDBACK LOOPS")
    assert m.quit() == 0
