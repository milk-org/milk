"""milk-streamCTRL is milk-CTRL started under that name: F3, the old option, the holder scan, read-only viewing."""

from __future__ import annotations

import time

import pytest

pytestmark = pytest.mark.timeout(60)

EVIL = "x$(touch${IFS}PWNED)"


def test_profile_opens_the_streams_view(legacy_pane, entities, tmp_path):
    """The header names the profile; F3 shows the list and the inspector; -d and the old -w are accepted."""
    other = tmp_path / "othershm"
    other.mkdir()
    entities.make_stream("elsewhere")
    (entities.shm / "elsewhere.im.shm").rename(other / "elsewhere.im.shm")
    p = legacy_pane("milk-streamCTRL", "-w", "-d", str(other))
    p.wait_for(r"milk-streamCTRL")
    p.wait_for(r"STREAMS\s+\(1\)")
    p.wait_for(r"STREAM elsewhere")
    p.wait_for(
        r"old keys; -w does"
    )  # the start-up note (the log panel may cut the end)
    p.keys("q")
    assert p.exit_code() == 0


def test_holder_scan_runs_no_shell(legacy_pane, entities, tmp_path):
    """HOLDERS reads /proc maps: a stream name never reaches a shell (fuser and popen are gone)."""
    entities.make_hostile_stream("hostile", EVIL)
    p = legacy_pane("milk-streamCTRL")
    p.wait_for(r"x\$\(touch")
    p.keys(
        "Right", "Right", "Right", "Right", "Right"
    )  # inspector, then SEMS, TRACE, KEYS, HOLDERS
    p.wait_for(r"\d+ process(es)? maps? it")
    time.sleep(2.5)  # another pass
    assert not list(tmp_path.rglob("PWNED")), "stream name reached a shell"
    p.keys("q")
    assert p.exit_code() == 0


def test_viewing_does_not_write_streams(legacy_pane, entities):
    """Viewing maps streams read-only: the file is unchanged (streamCTRL once stamped lastaccesstime, sctl-d11)."""
    path = entities.make_stream("watched")
    before = path.read_bytes()
    p = legacy_pane("milk-streamCTRL")
    p.wait_for(r"watched")
    p.keys("Right")  # the inspector reads the stream in full
    time.sleep(1.5)
    assert path.read_bytes() == before
    p.keys("q")
    assert p.exit_code() == 0
