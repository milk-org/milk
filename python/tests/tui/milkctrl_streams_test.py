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


def test_inspector_sems_lists_writers(milkctrl, entities):
    """F3 inspector, SEMS page: one row per semaphore with its value and the process that posted it (milk's
    semaphore copy does not register as a reader, so READER stays empty here)."""
    writer, _ = entities.make_stream_pair("pairst")
    m = milkctrl("--view", "streams", "^pairst$")
    m.wait_for(r"STREAM pairst")
    m.keys("Right")  # into the inspector (INFO)
    m.wait_for(r"Creator\s+\d+")
    m.keys("Right")  # next page: SEMS
    m.wait_for(r"#\s+VALUE\s+READER\s+WRITER")
    m.wait_for(rf"\s9\s+\d+\s+{writer} shmimpoke")  # the tenth semaphore too
    assert "[SEMS]" in m.lines()[-1]
    assert m.quit() == 0


def test_inspector_keys_lists_keywords(milkctrl, entities):
    """F3 inspector, KEYS page: the stream's keywords with type, value and comment."""
    entities.make_stream("kwst", 4, 4)
    entities.add_keywords(
        "kwst", ("GAIN", "L", "12", "loop gain"), ("MODE", "S", "closed", "loop mode")
    )
    m = milkctrl("--view", "streams", "^kwst$")
    m.wait_for(r"STREAM kwst")
    m.keys("Right")
    m.wait_for(r"Keywords\s+2 set")
    for _ in range(3):  # INFO -> SEMS -> TRACE -> KEYS
        m.keys("Right")
    m.wait_for(r"GAIN\s+L\s+12\s+loop gain")
    m.wait_for(r"MODE\s+S\s+closed\s+loop mode")
    assert m.quit() == 0


def test_inspector_trace_names_the_writer(milkctrl, entities):
    """F3 inspector, TRACE page: the process that wrote pairst_out, its trigger and its timing."""
    _, reader = entities.make_stream_pair("pairst")
    m = milkctrl("--view", "streams", "^pairst_out$")
    m.wait_for(r"STREAM pairst_out")
    m.keys("Right", "Right", "Right")  # into INFO, then SEMS, TRACE
    m.wait_for(r"1 hop, end to end")
    m.wait_for(rf"0\. {reader} shmimcopy")
    m.wait_for(r"trigger\s+\S+.*(waiting|received|timed out)")
    m.wait_for(r"started\s+\S+ ago\s+compute")
    assert m.quit() == 0


def test_inspector_holders_lists_mapping_processes(milkctrl, entities):
    """F3 inspector, HOLDERS page: the poke process maps pairst; milk-CTRL itself is not listed."""
    writer, reader = entities.make_stream_pair("pairst")
    m = milkctrl("--view", "streams", "^pairst$")
    m.wait_for(r"STREAM pairst")
    m.keys(
        "Right", "Right", "Right", "Right", "Right"
    )  # into INFO, then SEMS, TRACE, KEYS, HOLDERS
    m.wait_for(
        r"2 processes map it"
    )  # the poke and the copy; milk-CTRL, which maps it too, is left out
    m.wait_for(rf"\s{writer}\s+milk-fpsexec-")
    m.wait_for(rf"\s{reader}\s+milk-fpsexec-")
    assert m.quit() == 0
