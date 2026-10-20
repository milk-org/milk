"""Sub-tabs (CONNECTIONS, LOOPS, DETAILS, RESOURCES): one table drives the tab bar, drawing, clicks and keys."""

from __future__ import annotations

import pytest

from tests.tui.milkctrl_fps_identity_test import _get_param

pytestmark = pytest.mark.timeout(60)


def _tab_col(m, label: str) -> int:
    """1-based screen column of sub-tab @label on the panel's top border (row 3 in F6)."""
    return m.lines()[2].index(label) + 2


def test_f6_draws_the_selected_subtab(milkctrl, entities):
    """F6 used to switch its sub-tab state but always draw CONNECTIONS."""
    entities.make_fps("subfps", "gain:FLOAT64:0.5")
    m = milkctrl()
    m.keys("F5")
    m.wait_for(r"subfps")
    m.keys("F6")
    m.wait_for(r"CONNECTIONS")
    m.click(_tab_col(m, "DETAILS"), 3)
    m.wait_for(r"Parameters \(2\)")
    assert m.quit() == 0


def test_resources_keeps_its_target_after_a_tab_click(milkctrl, entities):
    """Clicking the tab focuses the panel; RESOURCES used to lose the entry and show UNKNOWN."""
    entities.make_fps("resfps", "gain:FLOAT64:0.5")
    m = milkctrl()
    m.keys("F5")
    m.wait_for(r"resfps")
    m.keys("F6")
    m.wait_for(r"CONNECTIONS")
    m.click(_tab_col(m, "RESOURCES"), 3)
    m.wait_for(r"No active process for resfps")
    assert "UNKNOWN" not in m.screen()
    assert m.quit() == 0


def test_enter_edits_a_details_parameter(milkctrl, entities, tui_env):
    """ENTER on a DETAILS parameter opens the editor (it used to toggle the sub-tab instead)."""
    env, _ = tui_env
    entities.make_fps("detfps", "gain:FLOAT64:0.5")
    m = milkctrl()
    m.keys("c")
    m.wait_for(r"CONTROL")
    m.keys("F5")
    m.wait_for(r"detfps")
    m.keys("F6", "D")
    m.wait_for(r"Parameters \(2\)")
    m.keys(
        "Down", "Down"
    )  # the first arrow shows the cursor on Name, the second moves it to gain
    m.keys("Enter")
    m.wait_for(r"new value")
    m.text("1.5")
    m.keys("Enter")
    m.wait_for(r"1\.5")
    assert float(_get_param(env, "detfps", "gain")) == pytest.approx(1.5)
    assert m.quit() == 0


def test_f7_click_selects_a_loop(milkctrl, entities):
    """Clicks in F7's loop list select the loop (they used to go to the CONNECTIONS handler)."""
    for s in ["lpA", "lpB", "lpC", "lpD"]:
        entities.make_stream(s)
    entities.make_fps("fpsA", "streamin:STREAMNAME:lpB", "streamout:STREAMNAME:lpA")
    entities.make_fps("fpsB", "streamin:STREAMNAME:lpA", "streamout:STREAMNAME:lpB")
    entities.make_fps("fpsC", "streamin:STREAMNAME:lpD", "streamout:STREAMNAME:lpC")
    entities.make_fps("fpsD", "streamin:STREAMNAME:lpC", "streamout:STREAMNAME:lpD")
    m = milkctrl(y=40)
    m.keys("F7")
    m.wait_for(r"LOOP L01 CIRCUIT")
    hit = m.wait_for(r"^.*L02 L02")
    m.click(6, hit.string.count("\n", 0, hit.start()) + 1)
    m.wait_for(r"LOOP L02 CIRCUIT")
    assert m.quit() == 0
