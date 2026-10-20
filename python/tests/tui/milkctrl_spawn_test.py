"""milk-CTRL never hands names from shared memory to a shell (robust-1, control-4)."""

from __future__ import annotations

import time

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


def _fps_ctrl(milkctrl, name):
    m = milkctrl()
    m.keys("c")
    m.wait_for(r"CONTROL")
    m.keys("F5")
    m.wait_for(name)
    return m


def test_fps_start_refuses_unsafe_name(milkctrl, entities):
    entities.make_fps("bad.name")
    m = _fps_ctrl(milkctrl, r"bad\.name")
    m.keys("s")  # CONFstart
    m.wait_for(r"refused: name outside \[A-Za-z0-9_-\]")
    assert m.quit() == 0


def test_fps_actions_do_not_reach_prefix_session(milkctrl, entities, tmux_server):
    entities.make_fps("pfx")
    entities.tmux_session(tmux_server, "pfxlong")  # 'tmux -t pfx' would select this one
    m = _fps_ctrl(milkctrl, r"pfx")
    m.keys("s")
    m.wait_for(r"tmux name matches another session")
    assert tmux_server.has_session("pfxlong")
    windows = [
        w.window_name for w in tmux_server.sessions.get(session_name="pfxlong").windows
    ]
    assert windows == ["ctrl", "conf", "run"]
    assert m.quit() == 0


def test_runstart_reports_failed_conf_check(milkctrl, entities):
    entities.make_fps("nochk")  # fresh FPS: conf never ran, CHECKOK not set
    m = _fps_ctrl(milkctrl, r"nochk")
    m.keys("r")
    m.wait_for(r"RUNstart not sent, conf check not OK")
    assert m.quit() == 0


def test_snapshot_files_are_unique_private_and_reported(milkctrl, tmp_path):
    m = milkctrl()
    m.keys("W")
    m.wait_for(r"Snapshot written to \S+")
    m.keys("W")
    time.sleep(0.5)
    snaps = sorted((tmp_path / "tmp").glob("milk-CTRL_snapshot_*.txt"))
    assert len(snaps) == 2
    for s in snaps:
        assert (s.stat().st_mode & 0o777) == 0o600
    assert m.quit() == 0


def test_snapshot_failure_is_reported(milkctrl, tmp_path):
    m = milkctrl(env_extra={"TMPDIR": str(tmp_path / "missing")})
    m.keys("W")
    m.wait_for(r"Snapshot failed: No such file or directory")
    assert m.quit() == 0
