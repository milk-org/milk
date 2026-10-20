"""milk-streamCTRL stopgaps: holder scan without a shell, terminal restored (W7 #2, sctl-d3/d4)."""

from __future__ import annotations

import time

import pytest

pytestmark = pytest.mark.timeout(60)

EVIL = "x$(touch${IFS}PWNED)"


def test_holder_scan_runs_no_shell(legacy_pane, entities, tmp_path):
    entities.make_hostile_stream("hostile", EVIL)
    p = legacy_pane("milk-streamCTRL")
    p.wait_for(r"x\$\(touch")
    p.keys("F6")  # open-files view
    time.sleep(0.5)
    p.keys("F6")  # second press runs fuser on every stream
    p.wait_for(r"Last scan on\s+\d\d:\d\d")
    time.sleep(2.0)
    assert not list(tmp_path.rglob("PWNED")), "stream name reached a shell"
    assert p.alive()
    p.keys("x")
    assert p.exit_code() == 0
