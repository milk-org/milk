"""tmux-pane driver for milk-CTRL and the legacy TUIs (milk-streamCTRL, milk-fpsCTRL, milk-procCTRL)."""

from __future__ import annotations

import re
import shlex
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
