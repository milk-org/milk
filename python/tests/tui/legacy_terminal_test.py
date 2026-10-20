"""Legacy TUIs: usage errors exit 2; stdin is blocking again after quitting (W7 #5, pc-5, fpsctrl-d8)."""

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

TOOLS = ["milk-streamCTRL", "milk-procCTRL", "milk-fpsCTRL"]


@pytest.mark.parametrize("tool", TOOLS)
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


@pytest.mark.parametrize(
    "tool", ["milk-streamCTRL", "milk-fpsCTRL"]
)  # procCTRL needs its scanner
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
    """Declining to start the scanner daemon is a failure to start, not success (pc-10)."""
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
