"""milk-CTRL terminal safety: tty restore on every exit path, captured library output, raw shm strings.

Findings: robust-8, robust-9, robust-10.
"""

from __future__ import annotations

import os
import signal
import time

import pytest

pytestmark = pytest.mark.timeout(60)

RESTORE = rb"\x1b\[\?1006l.*\x1b\[\?1049l"


def test_quit_restores_tty(milkctrl_pty):
    with milkctrl_pty() as run:
        run.read_until(rb"F2:DASH")
        run.write(b"q")
        assert run.wait() == 0
        run.read_until(RESTORE)
        assert run.tty_restored()


@pytest.mark.parametrize(
    "sig", [signal.SIGSEGV, signal.SIGABRT, signal.SIGQUIT, signal.SIGFPE]
)
def test_fatal_signal_restores_tty(milkctrl_pty, sig):
    with milkctrl_pty() as run:
        run.read_until(rb"F2:DASH")
        os.kill(run.proc.pid, sig)
        assert run.wait() == -sig
        run.read_until(RESTORE)
        assert run.tty_restored()


@pytest.mark.parametrize("sig", [signal.SIGHUP, signal.SIGTERM])
def test_hup_term_exit_cleanly(milkctrl_pty, sig):
    with milkctrl_pty() as run:
        run.read_until(rb"F2:DASH")
        os.kill(run.proc.pid, sig)
        assert run.wait() == 0
        assert run.tty_restored()


def test_library_output_goes_to_cmdlog(milkctrl, tui_env):
    """ImageStreamIO's error on a truncated .im.shm is shown in the command log, not over the UI."""
    _, shm = tui_env
    with open(os.path.join(shm, "bad.im.shm"), "wb") as f:
        f.write(b"\0" * 64)
    m = milkctrl()
    m.wait_for(r"^ \d\d:\d\d:\d\d \S+ +Error in the file \(too small\)")
    assert "milk-CTRL" in m.lines()[0]
    assert m.quit() == 0


def test_control_chars_in_stream_name(milkctrl, entities, tmux_server):
    """An OSC title sequence hidden in a stream file name is drawn as '?' and never reaches the tty."""
    entities.make_hostile_stream("ccsrc", "evil\x1b]0;PWNED\x07nm")
    m = milkctrl()
    m.keys("F3")
    m.wait_for(r"evil\?\]0;PWNED\?nm")
    title = m.pane.cmd("display", "-p", "#{pane_title}").stdout
    assert "PWNED" not in "".join(title)
    assert m.quit() == 0


@pytest.mark.parametrize(
    "seq,tab",
    [
        (b"\x1b[13~", "F3:STRM"),  # rxvt / PuTTY F3
        (b"\x1b[[D", "F4:PROC"),  # Linux console F4
        (b"\x1bOR", "F3:STRM"),  # xterm SS3 F3
    ],
)
def test_fkey_variants_switch_views(milkctrl, seq, tab):
    m = milkctrl()
    m.wait_for(r"▌ F2:DASH ▐")
    m.send_hex(seq)
    m.wait_for(rf"▌ {tab} ▐")
    assert m.quit() == 0


def test_mouse_burst_no_stray_keys(milkctrl):
    """A burst of hover reports must not leak parameter bytes as keys (';', digits, 'M')."""
    m = milkctrl()
    m.wait_for(r"▌ F2:DASH ▐")
    m.keys("m")  # hover tracking on
    m.wait_for(r"HOVER: ON")
    burst = b"".join(f"\x1b[<35;{20 + i};{10 + i % 5}M".encode() for i in range(60))
    m.send_hex(burst)
    m.send_hex(b"\x1b[<35;")  # a report cut in the middle
    time.sleep(0.4)
    screen = m.screen()
    assert m.alive()
    assert "▌ F2:DASH ▐" in screen and "HOVER: ON" in screen
    assert "[STREAMS]" not in screen  # no prompt opened by a stray key
    assert m.quit() == 0
