"""
The milk-cli program itself, and milk-script where they share code: `exit` in subshells,
traps, terminal controls, piped input, the session log, calc-mode quoting, `milk-cli -h`
and finding its own install.
"""

from __future__ import annotations

import pytest
import os
import re
import shutil
import subprocess
import time

from milk.cliwrap import (
    HAVE_CLI,
    MILK_CLI_EXEC,
)
from tests.cli.conftest import (
    _install_dir,
    _isolated_shm_env,
    _run,
    milk_bash,
    milk_cli_script,
    milk_cli_stdin,
    milk_script,
    pty_session,
    value,
)

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.timeout(120),
]


def _bare_env(tmp_path):
    """No install variables, a PATH without any milk bin: only the executable's own path is known."""
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        "MILK_SHM_DIR": str(tmp_path),
        "MILK_PROC_DIR": str(tmp_path),
        "MILK_QUIET": "1",
        "TMUX_TMPDIR": os.environ.get("TMUX_TMPDIR", "/tmp"),
    }
    return env


def test_subshell_and_bare_exit(tmp_path):
    res = milk_cli_stdin("y=$(exit 3)\n( exit 0 )\necho two", tmp_path)
    assert "two" in res.stdout
    res2 = milk_script("false\nexit", tmp_path)
    assert res2.returncode == 1


def test_time_and_exit_trap(tmp_path):
    res = milk_cli_script(
        "trap '@calc -v z 1+1; echo \"deferdone=$z\"' EXIT\n@time echo hi\necho after",
        tmp_path,
        timeout=15,
    )
    assert res.returncode != 124
    assert "after" in res.stdout
    assert value(res.stdout, "deferdone") == "2"


def test_pty_interactive_controls(tmp_path):
    out = pty_session(
        [
            ("line", "x=5", 0.4),
            ("line", "for i in 1 2 3; do sleep 1; done", 0.4),
            ("raw", "\x03", 0.8),
            ("line", 'echo "x=[$x]"', 0.6),
            ("line", 'read v; echo "got=[$v]"', 0.4),
            ("line", "hello", 0.6),
        ],
        tmp_path,
    )
    assert "x=[5]" in out
    assert "got=[hello]" in out


def test_calc_mode_quoting_follows_the_shell_and_calculator_lexers(tmp_path):
    res = milk_cli_stdin(
        '@calc\ns = "a # b"   # a comment\n"1 + 2"\n\'2 * 3\' # note\n"4" + "5"\n@calc\n'
        'echo "s=$(@calc s)"',
        tmp_path,
    )
    assert [ln for ln in res.stdout.splitlines() if ln] == [
        "s a # b",
        "3",
        "6",
        "45",
        "s=a # b",
    ], (
        res.stdout + res.stderr
    )


def test_milk_cli_help_lists_the_options_it_takes():
    """`milk-cli -h` no longer lists -E, -m and -Z, which milk-cli rejects, and lists -q, --session,
    --server and --idle-timeout."""
    env, shm_dir = _isolated_shm_env()
    try:
        helptext = _run([MILK_CLI_EXEC, "-h"], env, 20).stdout
        rejected = _run([MILK_CLI_EXEC, "-E", "-c", "true"], env, 20)
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)
    opts = re.findall(
        r"^\s+(-\S+)", helptext.split("OPTIONS")[1].split("====")[0], re.M
    )
    for gone in ("-E,", "-m", "-Z,"):
        assert gone not in opts, opts
    for new in ("-q,", "--session", "--server", "--idle-timeout"):
        assert new in opts, opts
    assert rejected.returncode != 0, rejected.stdout + rejected.stderr


def test_session_log_records_each_line(tmp_path):
    """`@config sessionlog FILE` logs each line read after it, with the time and the ms since logging
    started; nothing wrote those entries since the code that did lost its caller."""
    env, shm_dir = _isolated_shm_env()
    log = tmp_path / "s.log"
    try:
        proc = subprocess.Popen(
            [MILK_CLI_EXEC, "-q"],
            env=env,
            cwd=str(tmp_path),
            text=True,
            start_new_session=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for line in (
            f"@config sessionlog {log}",
            "echo one",
            "echo two",
            "@config sessionlog off",
            "echo three",
        ):
            proc.stdin.write(line + "\n")
            proc.stdin.flush()
            time.sleep(0.4)  # one line per read, as typed
        proc.stdin.close()
        proc.wait(timeout=20)
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)
    text = log.read_text()
    entries = re.findall(
        r"^\[\d{4}-\d\d-\d\dT[\d:]+\] \[ *[\d.]+ ms\] (.*)$", text, re.M
    )
    assert text.startswith("# Session started"), text
    assert entries == ["echo one", "echo two", "@config sessionlog off"], text


def test_fast_producer_piped_into_milk_cli_runs_every_line(tmp_path):
    """400 000 lines arriving faster than they run are all executed (drain capped, not unbounded)."""
    env, shm_dir = _isolated_shm_env()
    try:
        n = 400_000
        res = _run(
            [MILK_CLI_EXEC, "-q"],
            env,
            timeout=90,
            stdin="echo x\n" * n,
            cwd=str(tmp_path),
        )
        assert res.returncode == 0, res.stderr[-500:]
        assert res.stdout.count("x\n") == n, (len(res.stdout), res.stderr[-300:])
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)


def test_piped_lines_logged_and_calc_batch_not_held(tmp_path):
    res = milk_bash(
        'rm -f "$HOME/.milk_history_log"\n'
        "printf 'echo one\\necho two\\n@calc\\nz = 2 + 3\\n@calc\\necho \"z=$z\"\\n' | milk-cli -q >/dev/null 2>&1\n"
        'echo "entries=$(grep -c $\'\\tP\\t\' "$HOME/.milk_history_log")"\n'
        "( printf 'echo one\\n@calc\\ny = (3\\n'; sleep 1; printf '@calc\\necho end\\n' ) | milk-cli -q > held.txt 2>&1 &\n"
        'sleep 0.5; echo "early=$(grep -m1 . held.txt)"; wait',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "entries") == "6", out
    # before: the open parenthesis of the calc line held the whole batch, `one` included, until EOF
    assert value(res.stdout, "early") == "one", out
    assert "end" in (tmp_path / "held.txt").read_text(), out


def test_binaries_locate_their_own_install(tmp_path):
    """Run by absolute path with MILK_INSTALLDIR unset and no milk bin on PATH, milk-cli and milk-script
    export their own install, load its modules and resolve units from its bin."""
    inst = _install_dir().resolve()
    f = tmp_path / "s.milk"
    f.write_text(
        'echo "dir=$MILK_INSTALLDIR"; echo "unit=$(which milk-fpsexec-milk-calc)"\n'
        'mem.mk2Dim "@S:selfloc" 2 2 >/dev/null; echo "mk=$?"\n'
    )
    res = _run(
        [str(inst / "bin" / "milk-cli"), "-q", "-s", str(f)],
        _bare_env(tmp_path),
        20,
        cwd=str(tmp_path),
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "dir") == str(inst), out
    assert value(res.stdout, "unit") == str(
        inst / "bin" / "milk-fpsexec-milk-calc"
    ), out
    assert value(res.stdout, "mk") == "0", out
    res = _run(
        [str(inst / "bin" / "milk-script"), "-q", str(f)],
        _bare_env(tmp_path),
        20,
        cwd=str(tmp_path),
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "dir") == str(inst), out
    assert value(res.stdout, "unit") == str(
        inst / "bin" / "milk-fpsexec-milk-calc"
    ), out
    # a value set by the caller is kept
    env = _bare_env(tmp_path)
    env["MILK_INSTALLDIR"] = "/nonexistent"
    res = _run(
        [str(inst / "bin" / "milk-script"), "-q", str(f)], env, 20, cwd=str(tmp_path)
    )
    assert value(res.stdout, "dir") == "/nonexistent", res.stdout + res.stderr
