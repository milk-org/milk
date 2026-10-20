"""Stream rows: files that are not usable streams, symbolic links (round 4)."""

from __future__ import annotations

import os
import time

import pytest

pytestmark = pytest.mark.timeout(60)


def test_unusable_files_are_listed_with_a_reason(milkctrl, entities):
    """A junk *.im.shm and a dangling link are rows with the reason, not silently dropped."""
    entities.make_stream("goodst")
    (entities.shm / "junkst.im.shm").write_bytes(b"not a stream")
    os.symlink("/nonexistent/x.im.shm", entities.shm / "lostst.im.shm")
    m = milkctrl()
    m.keys("F3")
    m.wait_for(r"junkst ✗ (not a stream file|damaged or truncated)")
    m.wait_for(r"lostst ✗ dangling link")
    m.wait_for(r"goodst")
    assert m.quit() == 0


def test_symlink_shows_its_target(milkctrl, entities):
    """A link to a listed stream shows name→target and is not a second graph node."""
    entities.make_stream("realst")
    os.symlink(entities.shm / "realst.im.shm", entities.shm / "linkst.im.shm")
    m = milkctrl()
    m.keys("F3")
    m.wait_for(r"linkst→realst")
    m.wait_for(r"2 stm")
    assert "0 edg" in m.header()
    assert m.quit() == 0


def test_owner_and_creator_columns_sort(milkctrl, entities):
    """OWNER and the new CPID column are sortable: the > cycle reaches both."""
    entities.make_stream("cpidst")
    m = milkctrl()
    m.keys("F3")
    m.wait_for(r"CPID")
    seen = set()
    for _ in range(12):
        m.keys(">")
        time.sleep(0.2)  # the frame for this key
        hdr = next(x for x in m.lines() if "NAME" in x)
        seen |= {c for c in ("OWNER", "CPID") if c + "▲" in hdr}
    assert seen == {"OWNER", "CPID"}
    assert m.quit() == 0
