"""milk-CTRL never hands names from shared memory to a shell (robust-1, control-4)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.timeout(60)

EVIL = "x$(touch${IFS}PWNED)"


def test_inspect_passes_names_literally(milkctrl, entities, tmp_path):
    entities.make_hostile_stream("evsrc", EVIL)
    m = milkctrl()
    m.keys("F3")
    m.wait_for(r"x\$\(touch")
    m.keys("i")
    m.wait_for(r"Press ENTER to return")
    m.keys("Enter")
    m.wait_for(r"F3:STRM")
    assert not (tmp_path / "cwd" / "PWNED").exists()
    assert not (tmp_path / "home" / "PWNED").exists()
    assert m.quit() == 0


def test_tmux_flags_still_detected(milkctrl, entities, tmux_server):
    entities.make_fps("tmxfps", "gain:FLOAT64:0.5")
    entities.tmux_session(tmux_server, "tmxfps")
    m = milkctrl()
    m.keys("F5")
    m.wait_for(r"tmxfps\s+(?!---)\S{3}\s")
    assert m.quit() == 0
