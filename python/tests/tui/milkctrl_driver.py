"""tmux-pane driver for milk-CTRL and the legacy TUIs (milk-streamCTRL, milk-fpsCTRL, milk-procCTRL)."""

from __future__ import annotations

import fcntl
import os
import pty
import re
import resource
import select
import shlex
import signal
import struct
import subprocess
import termios
import time
import uuid

import libtmux


class TuiPane:
    """One TUI running in its own tmux session; the command's exit code is echoed as __rc=N."""

    def __init__(
        self,
        server: libtmux.Server,
        argv: list[str],
        env: dict[str, str],
        x: int = 200,
        y: int = 50,
    ):
        self.server = server
        self.name = f"tui-{uuid.uuid4().hex[:8]}"
        envs = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(env.items()))
        cmd = f"env {envs} {shlex.join(argv)}; echo __rc=$?; sleep 600"
        self.session = server.new_session(
            session_name=self.name, x=x, y=y, window_command=cmd
        )
        self.pane = self.session.active_window.active_pane

    # ------------------------------------------------------------ input

    def keys(self, *names: str) -> None:
        """Send tmux key names (e.g. 'F3', 'Escape', 'Tab', 'C-e') or single characters."""
        for k in names:
            self.pane.cmd("send-keys", k)

    def text(self, s: str) -> None:
        """Type literal text."""
        self.pane.cmd("send-keys", "-l", s)

    def send_hex(self, data: bytes) -> None:
        """Send raw bytes (escape sequences, SGR mouse reports)."""
        self.pane.cmd("send-keys", "-H", *[f"{b:02x}" for b in data])

    def click(self, col: int, row: int, button: int = 0) -> None:
        """SGR 1006 press+release at 1-based (col, row)."""
        self.send_hex(f"\x1b[<{button};{col};{row}M".encode())
        self.send_hex(f"\x1b[<{button};{col};{row}m".encode())

    def paste(self, s: str) -> None:
        """Paste through tmux so bracketed-paste markers are sent when the app asked for them."""
        self.server.cmd("set-buffer", "-b", self.name, s)
        self.pane.cmd("paste-buffer", "-p", "-d", "-b", self.name)

    # ------------------------------------------------------------ output

    def screen(self) -> str:
        return "\n".join(self.pane.cmd("capture-pane", "-p").stdout)

    def screen_ansi(self) -> str:
        return "\n".join(self.pane.cmd("capture-pane", "-p", "-e").stdout)

    def lines(self) -> list[str]:
        return self.pane.cmd("capture-pane", "-p").stdout

    def wait_for(
        self, pattern: str, timeout: float = 8.0, ansi: bool = False
    ) -> re.Match:
        """Poll the screen every 50 ms until the regex matches; fail with the last screen."""
        rx = re.compile(pattern, re.M)
        deadline = time.monotonic() + timeout
        last = ""
        while time.monotonic() < deadline:
            last = self.screen_ansi() if ansi else self.screen()
            m = rx.search(last)
            if m:
                return m
            time.sleep(0.05)
        raise AssertionError(f"timeout waiting for {pattern!r}; screen was:\n{last}")

    def exit_code(self, timeout: float = 5.0) -> int:
        m = self.wait_for(r"__rc=(\d+)", timeout)
        return int(m.group(1))

    def alive(self) -> bool:
        return "__rc=" not in self.screen()

    def kill(self) -> None:
        try:
            self.session.kill()
        except Exception:
            pass


class MilkCtrl(TuiPane):
    """milk-CTRL specific helpers."""

    def quit(self) -> int:
        self.keys("q")
        return self.exit_code()

    def header(self) -> str:
        lines = self.lines()
        return lines[0] if lines else ""

    def status(self) -> str:
        lines = [ln for ln in self.lines() if ln.strip()]
        return lines[-1] if lines else ""


class PtyRun:
    """A TUI on a pty we own: keeps the slave fd so termios and O_NONBLOCK can be checked after exit."""

    def __init__(
        self, argv: list[str], env: dict[str, str], rows: int = 50, cols: int = 160
    ):
        self.master, self.slave = pty.openpty()
        fcntl.ioctl(
            self.slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0)
        )
        self.orig_attr = termios.tcgetattr(self.slave)
        self.orig_flags = fcntl.fcntl(self.slave, fcntl.F_GETFL)
        self.out = b""

        def pre():
            fcntl.ioctl(0, termios.TIOCSCTTY, 0)
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

        self.proc = subprocess.Popen(
            argv,
            env=env,
            stdin=self.slave,
            stdout=self.slave,
            stderr=self.slave,
            start_new_session=True,
            preexec_fn=pre,
        )

    def _pump(self, timeout: float) -> None:
        r, _, _ = select.select([self.master], [], [], timeout)
        if r:
            try:
                self.out += os.read(self.master, 65536)
            except OSError:
                pass

    def read_until(self, pattern: bytes, timeout: float = 8.0) -> None:
        deadline = time.monotonic() + timeout
        while re.search(pattern, self.out) is None:
            if time.monotonic() > deadline:
                raise AssertionError(
                    f"timeout waiting for {pattern!r}; got {self.out[-400:]!r}"
                )
            self._pump(0.05)

    def write(self, data: bytes) -> None:
        os.write(self.master, data)

    def wait(self, timeout: float = 8.0) -> int:
        deadline = time.monotonic() + timeout
        while self.proc.poll() is None:
            if time.monotonic() > deadline:
                raise AssertionError("process did not exit")
            self._pump(0.05)
        self._pump(0.05)
        return self.proc.returncode

    def tty_restored(self) -> bool:
        """Canonical mode and echo are back, and stdin is blocking again."""
        attr = termios.tcgetattr(self.slave)
        flags = fcntl.fcntl(self.slave, fcntl.F_GETFL)
        lflag_ok = (attr[3] & (termios.ICANON | termios.ECHO)) == (
            self.orig_attr[3] & (termios.ICANON | termios.ECHO)
        )
        return lflag_ok and not (flags & os.O_NONBLOCK)

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.proc.wait(timeout=5)
        for fd in (self.master, self.slave):
            try:
                os.close(fd)
            except OSError:
                pass
