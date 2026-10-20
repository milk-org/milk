"""y/N confirmation for destructive verbs and FPS batch marks keyed by name (robust-7, control-2/3, ux-5)."""

from __future__ import annotations

import os
import time

import pytest

pytestmark = pytest.mark.timeout(60)


def _control(m, view):
    m.keys("c")
    m.wait_for(r"CONTROL")
    m.keys(view)


def test_delete_cancelled_keeps_stream(milkctrl, entities, tui_env):
    _, shm = tui_env
    entities.make_stream("keepme")
    m = milkctrl()
    _control(m, "F3")
    m.wait_for(r"keepme")
    m.keys("DC")
    m.wait_for(r"Delete stream")
    m.keys("n")
    m.wait_for(r"Cancelled")
    time.sleep(0.3)
    assert "CONFIRM" not in m.screen()
    assert os.path.exists(os.path.join(shm, "keepme.im.shm"))
    assert m.quit() == 0


def test_confirm_rechecks_target(milkctrl, entities, tui_env):
    """The modal re-resolves its target on 'y': a stream removed meanwhile is not acted on."""
    _, shm = tui_env
    entities.make_stream("vanish")
    m = milkctrl()
    _control(m, "F3")
    m.wait_for(r"vanish")
    m.keys("DC")
    m.wait_for(r"vanish  inode")
    os.unlink(os.path.join(shm, "vanish.im.shm"))
    m.wait_for(r"vanish  \(no longer present\)")
    m.keys("y")
    m.wait_for(r"\"vanish\" is gone: nothing done")
    assert m.quit() == 0


def test_sigkill_process_after_yes(milkctrl, entities):
    pid = entities.make_proc("killme")
    m = milkctrl()
    _control(m, "F4")
    m.wait_for(r"killme")
    m.keys("K")
    m.wait_for(r"SIGKILL process")
    os.kill(pid, 0)  # still alive while the modal is open
    m.keys("y")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if os.waitpid(pid, os.WNOHANG) != (0, 0):
            break
        time.sleep(0.1)
    else:
        raise AssertionError("process survived the confirmed SIGKILL")
    assert m.quit() == 0


def test_fps_batch_marks(milkctrl, entities):
    for n in ["mk1", "mk2"]:
        entities.make_fps(n)
    m = milkctrl()
    m.keys("c")
    m.wait_for(r"CONTROL")
    m.keys("Tab", "Tab")  # DASH focus: STREAMS -> PROCS -> FPS
    m.wait_for(r"mk2")
    m.keys("a")
    m.wait_for(r"FPS ◆2")
    m.keys("/")
    m.text("mk1")
    m.keys("Enter")
    m.wait_for(r"FPS ◆2 \(1 hidden\)")
    m.keys("k")
    m.wait_for(r"Batch SIGTERM")
    m.wait_for(r"2 marked FPS")
    m.keys("Escape")  # cancels the modal, marks kept
    m.wait_for(r"Cancelled")
    m.keys("Escape")  # clears the filter
    m.wait_for(r"FPS ◆2 ")
    m.keys("Escape")  # last step of the back-chain: marks
    m.wait_for(r"FPS batch marks cleared")
    assert "◆2" not in m.screen()
    assert m.quit() == 0
