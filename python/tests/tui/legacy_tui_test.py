"""Consolidated tests for legacy TUIs (milk-fpsCTRL, milk-procCTRL, milk-streamCTRL) and companion tools."""

from __future__ import annotations

import fcntl
import os
import pty
import select
import struct
import subprocess
import termios
import time

import pytest

from tests.tui.conftest import _install_dir

pytestmark = pytest.mark.timeout(60)


@pytest.mark.parametrize("tool", ["milk-streamCTRL", "milk-procCTRL", "milk-fpsCTRL"])
def test_usage_error_exits_2(tui_env, tool):
    env, _ = tui_env
    r = subprocess.run(
        [str(_install_dir() / "bin" / tool), "--no-such-option"],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=20,
    )
    assert r.returncode == 2


@pytest.mark.parametrize("tool", ["milk-streamCTRL", "milk-fpsCTRL"])
def test_stdin_blocking_after_quit(tui_env, tool):
    env, _ = tui_env
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 160, 0, 0))
    p = subprocess.Popen(
        [str(_install_dir() / "bin" / tool)],
        env=env,
        stdin=slave,
        stdout=slave,
        stderr=slave,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5
        seen = 0
        while time.monotonic() < deadline and seen < 200:
            if select.select([master], [], [], 0.2)[0]:
                seen += len(os.read(master, 65536))
        time.sleep(0.5)
        os.write(master, b"x")
        deadline = time.monotonic() + 10
        while p.poll() is None and time.monotonic() < deadline:
            if select.select([master], [], [], 0.2)[0]:
                os.read(master, 65536)
        assert p.poll() is not None, f"{tool} did not quit on 'x'"
        assert not fcntl.fcntl(slave, fcntl.F_GETFL) & os.O_NONBLOCK
    finally:
        if p.poll() is None:
            p.kill()
        os.close(master)
        os.close(slave)


def test_procctrl_exits_1_without_scanner(tui_env):
    env, _ = tui_env
    r = subprocess.run(
        [str(_install_dir() / "bin" / "milk-procCTRL")],
        env=env,
        input=b"n\n",
        capture_output=True,
        timeout=20,
    )
    assert b"daemon is required" in r.stdout + r.stderr
    assert r.returncode == 1


def test_streamctrl_profile_and_readonly_viewing(legacy_pane, entities):
    path = entities.make_stream("watched")
    before = path.read_bytes()
    p = legacy_pane("milk-streamCTRL")
    p.wait_for(r"milk-streamCTRL")
    p.wait_for(r"watched")
    p.keys("Right")
    time.sleep(1.0)
    assert path.read_bytes() == before
    p.keys("q")
    assert p.exit_code() == 0


def test_procinfo_info_labels(entities, tui_env):
    env, _ = tui_env
    entities.make_proc("lblproc")
    out = subprocess.run(
        ["milk-procinfo-info", "lblproc"],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    ).stdout
    assert "RUNNING" in out
    assert "CTRL val          : RUN" in out
    assert "IMMEDIATE (0)" in out
