"""
milk-cli hardening & crash-prevention tests.

Pytest port of tests/cli/test_cli_hardening.sh. Every milk-cli child is
spawned through `milk_proc`, which guarantees the whole process group is
killed and all fds are closed, whatever happens in the test body.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

from milk.cliwrap import HAVE_CLI, HAVE_TREESITTER
from tests.cli.conftest import assert_no_crash, milk_bash, milk_proc, run_milk

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
def test_fifo_runs_split_and_long_lines_whole(tmp_path):
    """A line written in two pieces, and a line longer than 999 bytes, each run whole."""
    long = "x" * 4096
    res = milk_bash(
        "milk-cli -q -F ff < /dev/null > out.txt 2>&1 &\n"
        "P=$!\n"
        "for _ in $(seq 100); do [ -p ff ] && break; sleep 0.05; done\n"
        "{ printf 'echo hel'; sleep 0.3; printf 'lo\\n'; } > ff\n"
        f"echo 'echo LONG {long} END; echo AFTER' > ff\n"
        "echo exit > ff\n"
        'wait "$P"; echo "rc=$?"; cat out.txt\n',
        tmp_path,
        timeout=20,
    )
    lines = res.stdout.splitlines()
    for want in ("rc=0", "hello", f"LONG {long} END", "AFTER"):
        assert want in lines, res.stdout[-2000:] + res.stderr
    assert "not found" not in res.stdout, res.stdout[-2000:]


def test_fifo_named_path_runs_commands(tmp_path):
    """`-F PATH` makes its FIFO at PATH, for this user only, and runs the lines written to it."""
    fifo = tmp_path / "named_fifo"
    with milk_proc(["-f", "-F", str(fifo)], stdin=subprocess.PIPE) as m:
        end = time.time() + 5
        while time.time() < end and not fifo.exists():
            time.sleep(0.05)
        assert fifo.is_fifo(), "no FIFO at the path given to -F"
        assert (fifo.stat().st_mode & 0o777) == 0o600
        with open(fifo, "w") as f:
            f.write("echo fifo_named_ok\nexit\n")
        try:
            out, _ = m.proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            pytest.fail("milk-cli hung on FIFO input")
    assert_no_crash(m.proc.returncode)
    assert "fifo_named_ok" in out, out


def test_fifo_refuses_a_path_that_is_not_its_fifo(tmp_path):
    """A file at the `-F` path is neither read as commands nor removed."""
    path = tmp_path / "notes.txt"
    path.write_text("echo must_not_run\n")
    rc, out = run_milk(["-f", "-F", str(path)])
    assert rc != 0 and "is not a FIFO" in out, out
    assert "must_not_run" not in out.replace("echo must_not_run", ""), out
    assert path.read_text() == "echo must_not_run\n"


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


def test_fifo_default_name_follows_n_and_open_answers(tmp_path):
    """-f names its FIFO after -n (also when milk-cli runs by full path); open answers under -q."""
    rc, out = run_milk(
        ["-q", "-n", "myctl", "-f"], stdin_text="milk.fifo status\nexit\n"
    )
    assert rc == 0 and "/.myctl.fifo." in out, out
    given = tmp_path / "given.fifo"
    os.mkfifo(given, 0o600)
    rc, out = run_milk(
        ["-q"], stdin_text=f"milk.fifo open {given}\nmilk.fifo close\nexit\n"
    )
    assert rc == 0 and f"opened: {given}" in out, out
    assert given.is_fifo()


def test_fifo_close_keeps_a_fifo_it_was_given(tmp_path):
    r = milk_bash(
        'mkfifo -m 600 g.fifo; milk.fifo open "$PWD/g.fifo"; milk.fifo close; '
        "[ -p g.fifo ] && echo kept",
        tmp_path,
    )
    assert "kept" in r.stdout, r.stdout + r.stderr


def test_fifo_outlives_stdin(tmp_path):
    """A backgrounded `milk-cli -F` (stdin /dev/null) reads its FIFO until `exit`, then removes it."""
    res = milk_bash(
        'F="$PWD/ctl.fifo"\n'
        'milk-cli -F "$F" > out.txt 2>&1 &\n'
        "P=$!\n"
        'for _ in $(seq 100); do [ -p "$F" ] && break; sleep 0.05; done\n'
        'echo "echo fifo_after_eof" > "$F"\n'
        'echo "exit 3" > "$F"\n'
        'wait "$P"; echo "rc=$?"\n'
        'grep -x fifo_after_eof out.txt; [ -e "$F" ] || echo removed\n',
        tmp_path,
        timeout=20,
    )
    assert res.stdout.split() == ["rc=3", "fifo_after_eof", "removed"], (
        res.stdout + res.stderr
    )


def test_pty_prompt_ctrl_c_drops_partial_line():
    """Ctrl-C at an idle interactive prompt drops the partial line instead of executing it (`front-0`)."""
    with milk_proc([], pty_mode=True) as m:
        m.read_until(PROMPT, 5)
        m.write(b"echo SHOULD_NOT_RUN")
        time.sleep(0.15)
        m.write(b"\x03")
        m.read_until(PROMPT, 5)
        m.drain()
        m.write(b"echo AFTER_CTRLC\n")
        out = m.read_until(b"AFTER_CTRLC", 5)
        assert b"SHOULD_NOT_RUN" not in out.split(b"^C")[-1], out
        m.write(b"exit\n")


@pytest.mark.timeout(60)
def test_pty_multiline_continuation_over_16k():
    """Interactive multi-line continuation buffer grows past 16 KiB (`front-6`)."""
    with milk_proc(["-q"], pty_mode=True) as m:
        m.read_until(PROMPT, 5)
        m.write(b"@config synhl off\n")
        m.read_until(PROMPT, 5)
        m.write(b"{\n")
        m.read_until(b"> ", 5)
        chunk = "a" * 1000
        m.write(f"s='{chunk}'\n".encode())
        for _ in range(17):
            m.read_until(b"> ", 5)
            m.write(f"s+='{chunk}'\n".encode())
        m.read_until(b"> ", 5)
        m.write(b'echo "BIGLEN=${#s}"\n}\n')
        out = m.read_until(b"BIGLEN=18000", 10)
        assert b"BIGLEN=18000" in out, out
        m.write(b"exit\n")


@pytest.mark.timeout(40)
def test_pty_tab_completion_escapes_shell_metachars(tmp_path):
    """TAB completion escapes shell metacharacters in stream names in milk-cli and bash (`front-8`)."""
    with milk_proc(["-q"], pty_mode=True) as m:
        m.read_until(PROMPT, 5)
        m.write(
            f'cd "{tmp_path}" && mem.mk2Dim "@S:zq\\$(touch INJECTED)" 2 2 >/dev/null && '
            'mem.mk2Dim "@S:b;id" 2 2 >/dev/null\n'.encode()
        )
        m.read_until(PROMPT, 5)
        m.write(b"@get s.zq")
        time.sleep(0.2)
        m.write(b"\t")
        m.read_until(b"INJECTED", 5)
        m.write(b"xsize\n")
        m.read_until(PROMPT, 5)
        m.write(b"@get s.b")
        time.sleep(0.2)
        m.write(b"\t")
        m.read_until(b"id", 5)
        m.write(b"xsize\n")
        m.read_until(PROMPT, 5)
        m.write(b"exit\n")
    assert not (
        tmp_path / "INJECTED"
    ).exists(), "TAB completion in milk-cli executed $(...)"

    import pty
    import select

    env, shm_dir = __import__(
        "tests.cli.conftest", fromlist=["_isolated_shm_env"]
    )._isolated_shm_env()
    try:
        pid, fd = pty.fork()
        if pid == 0:
            os.environ.update(env)
            os.chdir(str(tmp_path))
            os.execvp("bash", ["bash", "--norc", "-i"])
        try:

            def drain_for(sec):
                end = time.time() + sec
                buf = b""
                while time.time() < end:
                    if select.select([fd], [], [], 0.1)[0]:
                        try:
                            d = os.read(fd, 65536)
                            if not d:
                                break
                            buf += d
                        except OSError:
                            break
                return buf

            drain_for(1.0)
            os.write(fd, b'source "$MILK_INSTALLDIR/bin/milk.bash"\r')
            drain_for(3.0)
            os.write(fd, b'mem.mk2Dim "@S:zq\\$(touch INJECTED)" 2 2 >/dev/null\r')
            drain_for(1.5)
            os.write(fd, b"@get s.zq")
            drain_for(0.3)
            os.write(fd, b"\t")
            drain_for(0.6)
            os.write(fd, b"xsize\r")
            drain_for(1.5)
            os.write(fd, b"exit\r")
            drain_for(1.0)
        finally:
            os.close(fd)
            os.waitpid(pid, 0)
    finally:
        import shutil

        shutil.rmtree(shm_dir, ignore_errors=True)
    assert not (
        tmp_path / "INJECTED"
    ).exists(), "TAB completion in milk.bash executed $(...)"
