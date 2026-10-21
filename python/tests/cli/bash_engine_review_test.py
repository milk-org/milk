"""
Regression repros for the bash-engine review (feat/bash-engine).

Finding IDs (T*, C*, P*, S*, Sec*) refer to the review / work plan.
"""

from __future__ import annotations

import os
import pty
import re
import select
import signal
import subprocess
import time
from pathlib import Path

import pytest

from milk.cliwrap import HAVE_CLI, MILK_CLI_EXEC
from tests.cli.bash_integration_test import _install_dir, _isolated_shm_env, _run_bash

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.timeout(40),
]

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def bug(finding: str, reason: str, strict: bool = True):
    """Xfail marker tagged with a review finding ID (strict unless the repro is timing-dependent)."""
    return pytest.mark.xfail(strict=strict, reason=f"[{finding}] {reason}")


# ---------------------------------------------------------------- helpers


def _session_pids(sid: int) -> list[int]:
    pids = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            stat = Path(f"/proc/{d}/stat").read_text()
        except OSError:
            continue
        fields = stat.rsplit(")", 1)[1].split()
        if int(fields[3]) == sid:
            pids.append(int(d))
    return pids


def _kill_session(sid: int) -> None:
    """Kill every process in session @sid (milk-cli puts bash in its own pgrp)."""
    for pid in _session_pids(sid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _kill_by_cmdline(token: str) -> None:
    """Kill processes whose command line contains @token (auto-spawned milk servers live in their own session)."""
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == os.getpid():
            continue
        try:
            cmd = (
                Path(f"/proc/{d}/cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="replace")
            )
        except OSError:
            continue
        if token in cmd:
            try:
                os.kill(int(d), signal.SIGKILL)
            except ProcessLookupError:
                pass


def _run(
    argv: list[str],
    env: dict[str, str],
    timeout: float,
    stdin: str | None = None,
    cwd: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """subprocess.run that also reaps the whole session on timeout."""
    proc = subprocess.Popen(
        argv,
        env=env,
        cwd=cwd,
        text=True,
        start_new_session=True,
        stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        out, err = proc.communicate(stdin, timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_session(proc.pid)
        try:
            out, err = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:  # pipes held by a daemon outside the session
            proc.kill()
            out, err = "", ""
        return subprocess.CompletedProcess(
            argv, 124, _ANSI.sub("", out or ""), _ANSI.sub("", err or "")
        )
    _kill_session(proc.pid)
    return subprocess.CompletedProcess(
        argv, proc.returncode, _ANSI.sub("", out), _ANSI.sub("", err)
    )


def _env() -> tuple[dict[str, str], str]:
    """Isolated SHM dir plus a private HOME so ~/.milk_history and ~/.milkrc are never touched."""
    env, shm = _isolated_shm_env({"MILK_QUIET": "1"})
    home = Path(shm) / "home"
    home.mkdir(exist_ok=True)
    env["HOME"] = str(home)
    return env, shm


def milk_cli_script(
    script: str, tmp_path: Path, timeout: float = 20
) -> subprocess.CompletedProcess[str]:
    """Run @script as a file via `milk-cli -s`."""
    env, _ = _env()
    f = tmp_path / "s.milk"
    f.write_text(script + "\n")
    return _run([MILK_CLI_EXEC, "-s", str(f)], env, timeout, cwd=str(tmp_path))


def milk_cli_stdin(
    script: str, tmp_path: Path, timeout: float = 20
) -> subprocess.CompletedProcess[str]:
    """Pipe @script into milk-cli as REPL input."""
    env, _ = _env()
    return _run([MILK_CLI_EXEC], env, timeout, stdin=script + "\n", cwd=str(tmp_path))


def milk_script(
    script: str,
    tmp_path: Path,
    args: list[str] | None = None,
    opts: list[str] | None = None,
    timeout: float = 20,
) -> subprocess.CompletedProcess[str]:
    """Run @script with the standalone milk-script interpreter."""
    env, _ = _env()
    f = tmp_path / "ms.milk"
    f.write_text(script + "\n")
    exe = str(_install_dir() / "bin" / "milk-script")
    return _run(
        [exe, "-q", *(opts or []), str(f), *(args or [])],
        env,
        timeout,
        cwd=str(tmp_path),
    )


def translate(src: str) -> str:
    """Translate @src with `milk-call translate --legacy -` (no server needed)."""
    env, _ = _env()
    res = _run(
        [str(_install_dir() / "bin" / "milk-call"), "translate", "--legacy", "-"],
        env,
        10,
        stdin=src,
    )
    assert res.returncode == 0, res.stderr
    return res.stdout


def value(out: str, key: str) -> str | None:
    """Return the value of the last `key=value` line in @out."""
    m = re.findall(rf"^{re.escape(key)}=(.*)$", out, re.M)
    return m[-1] if m else None


def pty_session(
    steps: list[tuple[str, str, float]], tmp_path: Path, warmup: float = 2.0
) -> str:
    """Drive interactive milk-cli on a pty; steps are (kind, payload, wait) with kind in line|raw."""
    env, _ = _env()
    env["TERM"] = "dumb"
    pid, fd = pty.fork()
    if pid == 0:
        os.chdir(tmp_path)
        os.execve(MILK_CLI_EXEC, [MILK_CLI_EXEC], env)
    out = b""

    def drain(t: float) -> None:
        nonlocal out
        end = time.time() + t
        while time.time() < end:
            r, _, _ = select.select([fd], [], [], 0.05)
            if r:
                try:
                    d = os.read(fd, 65536)
                except OSError:
                    return
                if not d:
                    return
                out += d

    try:
        drain(warmup)
        for kind, payload, wait in steps:
            os.write(
                fd, (payload + "\r").encode() if kind == "line" else payload.encode()
            )
            drain(wait)
    finally:
        sid = os.getsid(pid) if Path(f"/proc/{pid}").exists() else pid
        _kill_session(sid)
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass
        os.close(fd)
    return _ANSI.sub("", out.decode(errors="replace")).replace("\r", "")


# ------------------------------------------------- T*: translator (bash/milk_bash_translate.c)


def test_T1_shift_in_arith_is_not_heredoc():
    out = translate("echo $((1<<2))\necho @{1+2}\n")
    assert "milkcalc" in out.splitlines()[1]


def test_T2_multiline_string_untouched():
    src = 'echo "x\n@ 1+2 in string\nmk2Dim a 1 1\n"\n'
    assert translate(src) == src


def test_T2_continuation_not_new_statement():
    out = translate("echo a \\\n  mk2Dim b 1 1\n")
    assert "mem.mk2Dim" not in out


def test_T3_at_v_quoted_expression(tmp_path):
    res = milk_cli_script(
        '@calc -v r "1+2"\necho "r=$r"\n@calc -v s \'2*3\'\necho "s=$s"', tmp_path
    )
    assert value(res.stdout, "r") == "3"
    assert value(res.stdout, "s") == "6"


def test_T3_at_v_error_on_unknown_image(tmp_path):
    res = milk_cli_script(
        '@calc -v r "imean(nosuchimage)"\necho "rc=$? r=$r"', tmp_path
    )
    assert "r=imean(nosuchimage)" not in res.stdout
    assert "rc=0" not in res.stdout


def test_T4_at_expr_not_split_on_operators(tmp_path):
    res = milk_cli_script(
        '@calc "x = (2 > 1) && (2 < 1)"\n@calc -v r "x"\necho "r=$r"', tmp_path
    )
    assert value(res.stdout, "r") == "0"


def test_T5_top_level_and_sourced_agree(tmp_path):
    (tmp_path / "sub.milk").write_text('listvar >/dev/null 2>&1; echo "sub=$?"\n')
    res = milk_cli_script(
        'listvar >/dev/null 2>&1; echo "top=$?"\n@source sub.milk', tmp_path
    )
    assert value(res.stdout, "top") is not None
    assert value(res.stdout, "top") == value(res.stdout, "sub")


def test_WP8_qualified_names_only(tmp_path):
    body = 'mem.listvar >/dev/null 2>&1; echo "{0}q=$?"\nlistvar >/dev/null 2>&1; echo "{0}b=$?"\n'
    (tmp_path / "sub.milk").write_text(body.format("s"))
    res = milk_cli_script(body.format("t") + "@source sub.milk", tmp_path)
    for k, v in (("tq", "0"), ("tb", "127"), ("sq", "0"), ("sb", "127")):
        assert value(res.stdout, k) == v, res.stdout


def test_T6_inline_result_not_globbed(tmp_path):
    (tmp_path / "a.zz").touch()
    (tmp_path / "b.zz").touch()
    res = milk_cli_script(
        "mem.mk2Dim t6 4 4\n@set s.t6.kw.OBS '*.zz'\necho @{s.t6.kw.OBS}", tmp_path
    )
    assert "*.zz" in res.stdout
    assert "a.zz" not in res.stdout


def test_T9_multiword_value_not_truncated(tmp_path):
    res = milk_cli_script(
        'mem.mk2Dim t9 4 4\n@set s.t9.kw.OBS="Sub aru"\necho "setrc=$?"\n@get -v v s.t9.kw.OBS\necho "v=$v"',
        tmp_path,
    )
    assert value(res.stdout, "setrc") == "0" and value(res.stdout, "v") == "Sub aru", (
        res.stdout + res.stderr
    )


def test_T10_hash_in_param_length(tmp_path):
    res = milk_cli_script('arr=(a b c)\nn=$(echo ${#arr[@]})\necho "n=$n"', tmp_path)
    assert value(res.stdout, "n") == "3"
    assert "unexpected end" not in res.stdout + res.stderr


def test_T11_dollar_single_quote_escape():
    out = translate("echo $'it\\'s' @{s.a.b}\n")
    assert "milkget" in out


def test_T12_second_heredoc_body_untouched():
    src = "cat <<A <<B\na\nA\nhelp\nB\n"
    assert translate(src) == src


def test_T13_lineno_matches_source(tmp_path):
    res = milk_cli_script('echo "l=$LINENO"', tmp_path)
    assert value(res.stdout, "l") == "1"


@pytest.mark.parametrize(
    "src, must_have, must_not_have",
    [
        ("xs=@{s.${cam}.xsize}\n", "{ milkget -v xs s.${cam}.xsize ||", "calc"),
        ("x=@{s.c.xsize}\n", "{ milkget -v x s.c.xsize ||", "calc"),
        ('x="@{s.c.xsize}"\n', "{ milkget -v x s.c.xsize ||", "REPLY"),
        ("a && y=@{1+2} || b\n", 'a && { milkcalc -v y "1+2" ||', None),
        ("w=@{s.c.xsize} cmd\n", "${| milkget -v REPLY s.c.xsize", None),
        ("export z=@{s.c.xsize}\n", "${| milkget -v REPLY s.c.xsize", None),
        (
            'echo "${@s.c.xsize}" @s.c.xsize\n',
            'echo "${@s.c.xsize}" @s.c.xsize',
            "milkget",
        ),
        ('echo "\\@{lit}"\n', 'echo "@{lit}"', "milkcalc"),
        ("echo foo@{a,b}\n", "foo@{a,b}", "milkcalc"),
        ('echo "$@f.x.y" user@s.ex.org\n', '"$@f.x.y" user@s.ex.org', "milkget"),
        ("val=@{a * 10 + b[2, 3]}\n", '{ milkcalc -v val "a * 10 + b[2, 3]" ||', None),
        ("echo @{a && b}; ls\n", 'milkcalc -v REPLY "a && b"', None),
        (
            "x=$(mem.listim)\ny=`mem.listim`\n",
            "x=$(mem.listim)\ny=`mem.listim`",
            None,
        ),
        ("x=$(listim)\n", "x=$(listim)", "mem.listim"),
    ],
)
def test_WP7_translation_cases(src, must_have, must_not_have):
    out = translate(src)
    assert must_have in out, out
    if must_not_have is not None:
        assert must_not_have not in out, out


def test_WP7_lineno_with_errexit(tmp_path):
    env, _ = _env()
    f = tmp_path / "l.milk"
    f.write_text('true\necho "l=$LINENO"\n')
    res = _run([MILK_CLI_EXEC, "-e", "-s", str(f)], env, 20, cwd=str(tmp_path))
    assert value(res.stdout, "l") == "2"


# --------------------------------------- C*: coprocess, REPL, milk-script


def test_C1_subshell_and_bare_exit(tmp_path):
    res = milk_cli_stdin("y=$(exit 3)\n( exit 0 )\necho two", tmp_path)
    assert "two" in res.stdout
    res2 = milk_script("false\nexit", tmp_path)
    assert res2.returncode == 1


def test_C2_time_and_exit_trap(tmp_path):
    res = milk_cli_script(
        "trap '@calc -v z 1+1; echo \"deferdone=$z\"' EXIT\n@time echo hi\necho after",
        tmp_path,
        timeout=15,
    )
    assert res.returncode != 124
    assert "after" in res.stdout
    assert value(res.stdout, "deferdone") == "2"


def test_C3_C5_pty_interactive_controls(tmp_path):
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


def test_P_and_S_consolidated():
    """Consolidated regression checks for P* (calc/shm_prop) and S* (server/builtins)."""
    res = _run_bash("""
        # P1/P2/P3: scalar & image expressions, division by zero, negative indices
        mem.mk2Dim "@S:pim" 8 8
        @calc "pim = 4.0"
        @calc "pim[-1, -1] = 10.0"
        [[ "$(@calc "pim[7, 7]")" == "10" ]]

        # S1/S2: server status and builtin return codes
        @server status >/dev/null
        rc=0
        @get s.no_such_stream.xsize 2>/dev/null || rc=$?
        [[ "$rc" -ne 0 ]]
        echo "PS_OK"
    """)
    assert res.returncode == 0, res.stderr
    assert "PS_OK" in res.stdout
