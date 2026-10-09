"""
milk-cli hardening & crash-prevention tests.

Pytest port of tests/cli/test_cli_hardening.sh. Every milk-cli child is
spawned through `milk_proc`, which guarantees the whole process group is
killed and all fds are closed, whatever happens in the test body.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import pty
import select
import signal
import subprocess
import termios
import threading
import time
import typing as typ
from pathlib import Path

import pytest

from milk.cliwrap import HAVE_CLI, HAVE_TREESITTER, MILK_CLI_EXEC

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.timeout(20),
]

REPO_ROOT = Path(__file__).resolve().parents[3]
PROMPT = b"milk-cli > "


needs_treesitter = pytest.mark.skipif(
    not HAVE_TREESITTER, reason="milk-cli built without tree-sitter"
)


class _Milk:
    """Handle yielded by `milk_proc`."""

    def __init__(self, proc: subprocess.Popen, master: int | None) -> None:
        self.proc = proc
        self.master = master
        self.log = b""  # everything read from the PTY so far

    def write(self, data: bytes) -> None:
        assert self.master is not None
        os.write(self.master, data)

    def _read_chunk(self, wait: float) -> bytes | None:
        """None on timeout, b"" on EOF/hangup."""
        assert self.master is not None
        ready, _, _ = select.select([self.master], [], [], wait)
        if not ready:
            return None
        try:
            return os.read(self.master, 4096)
        except OSError:  # EIO once the slave side is closed
            return b""

    def read_until(self, pattern: bytes | None, timeout: float = 3.0) -> bytes:
        """Read until `pattern` shows up; with None, until EOF."""
        buf = b""
        end = time.monotonic() + timeout
        while (pattern is None or pattern not in buf) and time.monotonic() < end:
            chunk = self._read_chunk(0.05)
            if chunk == b"":
                break
            buf += chunk or b""
        self.log += buf
        return buf

    def drain(self, quiet: float = 0.2, timeout: float = 3.0) -> bytes:
        """Read until the PTY has been silent for `quiet` seconds."""
        buf = b""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            chunk = self._read_chunk(quiet)
            if not chunk:
                break
            buf += chunk
        self.log += buf
        return buf


@contextlib.contextmanager
def milk_proc(
    args: list[str],
    *,
    pty_mode: bool = False,
    env: dict[str, str] | None = None,
    stdin: int = subprocess.DEVNULL,
) -> typ.Generator[_Milk]:
    """Spawn milk-cli in its own session; always reap it and its children."""
    assert MILK_CLI_EXEC is not None
    master = slave = None
    proc = None
    try:
        if pty_mode:
            master, slave = pty.openpty()
            io: dict[str, typ.Any] = dict(stdin=slave, stdout=slave, stderr=slave)
        else:
            io = dict(
                stdin=stdin,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
            )
        proc = subprocess.Popen(
            [MILK_CLI_EXEC, *args],
            env={**os.environ, "TERM": "xterm-256color", **(env or {})},
            start_new_session=True,  # pgid == pid, so killpg reaches children
            # Without a controlling tty, Ctrl+C never becomes a SIGINT.
            preexec_fn=(
                (lambda: fcntl.ioctl(0, termios.TIOCSCTTY, 0)) if pty_mode else None
            ),
            **io,
        )
        if slave is not None:
            os.close(slave)
            slave = None
        yield _Milk(proc, master)
    finally:
        if proc is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream is not None:
                    stream.close()
        for fd in (master, slave):
            if fd is not None:
                os.close(fd)


def assert_no_crash(returncode: int) -> None:
    # Negative: killed by signal. >=128: shell-style signal exit.
    assert 0 <= returncode < 128, f"milk-cli crashed (exit {returncode})"


def run_milk(
    args: list[str], *, stdin_text: str | None = None, timeout: float = 10
) -> tuple[int, str]:
    """Run milk-cli to completion; fail on crash or hang."""
    with milk_proc(
        args, stdin=subprocess.DEVNULL if stdin_text is None else subprocess.PIPE
    ) as m:
        try:
            out, _ = m.proc.communicate(stdin_text, timeout=timeout)
        except subprocess.TimeoutExpired:
            pytest.fail(f"milk-cli hung: {args!r}")
    assert_no_crash(m.proc.returncode)
    return m.proc.returncode, out


# --- Section 1: startup options ---------------------------------------------
@pytest.mark.parametrize("name", ["", ".", "a.b.c"])
def test_startup_name_option(name):
    rc, out = run_milk(["-n", name, "-c", "echo ok"])
    assert rc == 0
    assert "ok" in out


def test_unknown_option_errors_cleanly():
    rc, _ = run_milk(["-x"])
    assert rc != 0


# --- Section 2: recursion limits --------------------------------------------
def test_function_recursion_bounded(tmp_path):
    script = tmp_path / "rec_func.milk"
    script.write_text("function rec {\n    rec\n}\nrec\n")
    _, out = run_milk(["-s", str(script)])
    assert "recursion" in out.lower()


def test_source_recursion_bounded(tmp_path):
    script = tmp_path / "rec_source.milk"
    script.write_text(f"source {script}\n")
    _, out = run_milk(["-s", str(script)])
    assert "recursion" in out.lower()


# --- Sections 3-4: arithmetic edge cases & expansion buffers ----------------
NOCRASH_CMDS = [
    "calc -9223372036854775808 % -1",
    "calc -9223372036854775808 / -1",
    "calc 100 / 0",
    "calc 100 % 0",
    'cat <<< "herestring test"',
]


@pytest.mark.parametrize("cmd", NOCRASH_CMDS)
def test_command_does_not_crash(cmd):
    run_milk(["-c", cmd])


# (command, substring expected in output or None)
OK_CMDS = [
    ("echo $(( 1 << 64 ))", None),
    ("echo $(( 1 >> 100 ))", None),
    ("echo {1..100000}", None),
    ("(echo sub1) && (echo sub2)", "sub2"),
    ("showmatch; showmatch off; showmatch on", "Showmatch ON"),
]


@pytest.mark.parametrize("cmd,expected", OK_CMDS)
def test_command_succeeds(cmd, expected):
    rc, out = run_milk(["-c", cmd])
    assert rc == 0
    if expected is not None:
        assert expected.lower() in out.lower()


# --- Section 5: FIFO input --------------------------------------------------
def test_fifo_large_input(tmp_path):
    fifo = tmp_path / "test_fifo"
    os.mkfifo(fifo)

    def feed() -> None:
        with contextlib.suppress(OSError):  # reader may be gone
            time.sleep(0.2)
            with open(fifo, "w") as f:
                f.write("echo fifo_ok;" + "a" * 4096 + "\n")
            time.sleep(0.2)
            with open(fifo, "w") as f:
                f.write("exit\n")

    feeder = threading.Thread(target=feed, daemon=True)
    with milk_proc(["-f", "-F", str(fifo)]) as m:
        feeder.start()
        try:
            m.proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pytest.fail("milk-cli hung on FIFO input")
        finally:
            # A feeder blocked in open() needs a reader to get released.
            with contextlib.suppress(FileNotFoundError):  # milk may unlink it
                os.close(os.open(fifo, os.O_RDONLY | os.O_NONBLOCK))
            feeder.join(timeout=2)
    assert_no_crash(m.proc.returncode)


# --- Section 6: interactive fault isolation ---------------------------------
def test_pty_sigsegv_intercepted():
    with milk_proc([], pty_mode=True) as m:
        m.read_until(PROMPT, 5)
        m.write(b"sleep 0.3\n")
        time.sleep(0.02)
        os.kill(m.proc.pid, signal.SIGSEGV)
        time.sleep(0.02)
        m.write(b"echo SURVIVED_SEGV\nexit\n")
        m.read_until(None, 10)
        m.proc.wait(timeout=5)
    assert b"CRASH INTERCEPTED" in m.log
    assert b"SURVIVED_SEGV" in m.log
    assert m.proc.returncode >= 0


def test_pty_ctrl_c_cancels_command():
    with milk_proc([], pty_mode=True) as m:
        m.read_until(PROMPT, 5)
        m.write(b"sleep 30\n")
        time.sleep(0.1)
        m.write(b"\x03")
        # Prompt must come back well before the sleep would have ended.
        assert PROMPT in m.read_until(PROMPT, 3), "Ctrl+C did not cancel sleep"
        m.write(b"echo SURVIVED_SIGINT\nexit\n")
        m.read_until(None, 10)
        m.proc.wait(timeout=5)
    assert b"SURVIVED_SIGINT" in m.log


# --- Section 7: block folding & indentation ---------------------------------
def test_cliindent_toggle(tmp_path):
    script = tmp_path / "indent_toggle.milk"
    script.write_text(
        "cliindent\ncliindent off\ncliindent on\ncliindent 2\ncliindent\n"
    )
    _, out = run_milk(["-s", str(script)])
    assert "Auto-indentation is ON (2 spaces)" in out


@needs_treesitter
def test_clifold_block_outline():
    script = REPO_ROOT / "scripts" / "makecircleofdisks.milk"
    _, out = run_milk([], stdin_text=f"clifold {script}\nexit\n")
    assert "Total blocks: 4" in out


@needs_treesitter
def test_cliformat_reindents(tmp_path):
    script = tmp_path / "unformatted_loop.milk"
    script.write_text("for i in 1 2; do\necho $i\ndone\n")
    _, out = run_milk([], stdin_text=f"cliformat {script}\nexit\n")
    assert "    echo $i" in out


@needs_treesitter
def test_pty_continuation_auto_indent():
    with milk_proc([], pty_mode=True) as m:
        m.read_until(PROMPT, 5)
        for line in (b"for i in 1 2; do\n", b"echo test\n", b"done\n"):
            m.write(line)
            m.drain()
        m.write(b"exit\n")
        m.read_until(None, 5)
    assert b">     " in m.log
    assert b"test\r\ntest\r\n" in m.log


# --- Section 8: delimiter & keyword matching (showmatch) --------------------
REVERSE_VIDEO = b"\x1b[7m"


@needs_treesitter
def test_pty_showmatch_highlights():
    with milk_proc([], pty_mode=True, env={"COLORTERM": "truecolor"}) as m:
        m.read_until(PROMPT, 5)

        m.write(b"(1)")
        assert REVERSE_VIDEO in m.drain(), "delimiter match not highlighted"
        m.write(b"\n")
        m.drain()

        m.write(b"if true; then echo 1; fi")
        assert REVERSE_VIDEO in m.drain(), "block keyword match not highlighted"
        m.write(b"\n")
        m.drain()

        m.write(b"showmatch off\n")
        m.read_until(b"Showmatch OFF")
        m.drain()
        m.write(b"(1)")
        assert REVERSE_VIDEO not in m.drain(), "highlight still on after off"

        m.write(b"\nexit\n")


# --- Section 9: real-time syntax diagnostics (syndiag) ----------------------
def test_syndiag_toggle():
    rc, _ = run_milk([], stdin_text="syndiag\nsyndiag off\nsyndiag on\nexit\n")
    assert rc == 0


@needs_treesitter
def test_pty_syndiag_unclosed_quote():
    underline = b"\x1b[4;"
    with milk_proc([], pty_mode=True) as m:
        m.read_until(PROMPT, 5)

        m.write(b'echo "uncl sfr')
        out = m.drain()
        assert underline in out, "no underline on unclosed quote"
        assert b"203m" in out or b"31m" in out, "underline is not red"

        m.write(b'"')
        out = m.drain()
        assert underline not in out, "underline persists after closing quote"
        assert b"38;5;150m" in out or b"32m" in out, "string not recolored"

        m.write(b"\n")
        m.drain()
        m.write(b"syndiag off\n")
        m.read_until(b"Syntax diagnostics OFF")
        m.drain()
        m.write(b'echo "uncl sfr')
        assert underline not in m.drain(), "underline still on after off"

        m.write(b"\nexit\n")
