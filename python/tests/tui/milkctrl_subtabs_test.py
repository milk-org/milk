"""Sub-tabs (CONNECTIONS, LOOPS, DETAILS, RESOURCES): one table drives the tab bar, drawing, clicks and keys."""

from __future__ import annotations

import os
import subprocess
import time

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


def _perf_fds(m) -> int:
    """perf_event fds held by the milk-CTRL running in pane @m."""
    out = subprocess.run(
        ["pgrep", "-P", str(m.pane.pane_pid), "-x", "milk-CTRL"],
        capture_output=True,
        text=True,
    ).stdout.split()
    n = 0
    for fd in os.listdir(f"/proc/{out[0]}/fd"):
        try:
            n += os.readlink(f"/proc/{out[0]}/fd/{fd}") == "anon_inode:[perf_event]"
        except OSError:
            pass
    return n


def _wait(cond, timeout=5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.1)
    return False


def test_perf_counters_released_when_resources_hidden(milkctrl, entities):
    """Hardware counters stay attached to the inspected process only while RESOURCES shows them."""
    entities.make_proc("perfme")
    m = milkctrl()
    m.keys("F4")
    m.wait_for(r"perfme")
    m.keys(
        "F2", "BTab", "BTab", "BTab"
    )  # DASH keeps the process focus; sub-tab RESOURCES
    m.wait_for(r"RESOURCES")
    if not _wait(lambda: _perf_fds(m) > 0):
        pytest.skip("perf_event_open not permitted here")
    m.keys("F3")
    assert _wait(lambda: _perf_fds(m) == 0)
    assert m.quit() == 0


def test_connections_rows_show_health(milkctrl, entities):
    """CONNECTIONS rows carry the stream rate and the reader's state, also at 80 columns."""
    for s in ["hlA", "hlB"]:
        entities.make_stream(s)
    entities.make_fps("hfA", "streamin:STREAMNAME:hlB", "streamout:STREAMNAME:hlA")
    m = milkctrl(x=80, y=24)
    m.keys("F3")
    m.wait_for(r"hlA")
    m.keys("F6")
    m.wait_for(r"Hz STATE\s+EXEC\s+MISS")
    m.wait_for(r"hl[AB].*\s0\s+-")  # an idle stream: 0 Hz, no reader state
    assert m.quit() == 0
