"""
milk-cli hardening & crash-prevention tests.

Pytest port of tests/cli/test_cli_hardening.sh. Every milk-cli child is
spawned through `milk_proc`, which guarantees the whole process group is
killed and all fds are closed, whatever happens in the test body.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import pytest

from milk.cliwrap import HAVE_CLI, HAVE_TREESITTER
from tests.cli.conftest import assert_no_crash, milk_proc, run_milk

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.timeout(20),
]

REPO_ROOT = Path(__file__).resolve().parents[3]
PROMPT = b"milk-cli > "


needs_treesitter = pytest.mark.skipif(
    not HAVE_TREESITTER, reason="milk-cli built without tree-sitter"
)


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
    rc, out = run_milk(["-s", str(script)])
    assert "maximum function nesting level exceeded" in out
    assert rc == 1


def test_source_recursion_bounded(tmp_path):
    script = tmp_path / "rec_source.milk"
    script.write_text(f"@source {script}\n")
    _, out = run_milk(["-s", str(script)])
    assert "recursion" in out.lower()


# --- Sections 3-4: arithmetic edge cases & expansion buffers ----------------
NOCRASH_CMDS = [
    "@calc '-9223372036854775808 % -1'",
    "@calc '-9223372036854775808 / -1'",
    "@calc '100 / 0'",
    "@calc '100 % 0'",
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
    ("@config showmatch; @config showmatch off; @config showmatch on", "Showmatch ON"),
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
        "@config indent\n@config indent off\n@config indent on\n@config indent 2\n@config indent\n"
    )
    _, out = run_milk(["-s", str(script)])
    assert "Auto-indentation is ON (2 spaces)" in out


@needs_treesitter
def test_clifold_block_outline():
    script = REPO_ROOT / "scripts" / "makecircleofdisks.milk"
    _, out = run_milk(["--outline", str(script)])
    assert "Total blocks: 4" in out


@needs_treesitter
def test_cliformat_reindents(tmp_path):
    script = tmp_path / "unformatted_loop.milk"
    script.write_text("for i in 1 2; do\necho $i\ndone\n")
    _, out = run_milk(["--format", str(script)])
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

        m.write(b"@config showmatch off\n")
        m.read_until(b"Showmatch OFF")
        m.drain()
        m.write(b"(1)")
        assert REVERSE_VIDEO not in m.drain(), "highlight still on after off"

        m.write(b"\nexit\n")


# --- Section 9: real-time syntax diagnostics (syndiag) ----------------------
def test_syndiag_toggle():
    rc, _ = run_milk(
        [],
        stdin_text="@config syndiag\n@config syndiag off\n@config syndiag on\nexit\n",
    )
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
        m.write(b"@config syndiag off\n")
        m.read_until(b"Syntax diagnostics OFF")
        m.drain()
        m.write(b'echo "uncl sfr')
        assert underline not in m.drain(), "underline still on after off"

        m.write(b"\nexit\n")


# --- Section 11: Standalone math expression and continuation escape ---------
def test_math_expression_no_multiline():
    rc, out = run_milk([], stdin_text="@calc 1-3\nexit\n")
    assert rc == 0
    assert "-2" in out
    assert "\n> " not in out


@needs_treesitter
def test_pty_continuation_prompt_ctrl_c():
    with milk_proc([], pty_mode=True) as m:
        m.read_until(PROMPT, 5)
        m.write(b"for i in 1 2; do\n")
        m.read_until(b"> ", 5)
        m.write(b"\x03")  # Ctrl-C
        m.read_until(PROMPT, 5)
        m.write(b"echo recovered\n")
        m.read_until(b"recovered", 5)
        m.write(b"exit\n")
