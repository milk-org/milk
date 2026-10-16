"""
Regression repros for the bash-engine review (feat/bash-engine).

Each test asserts the *correct* behaviour and is marked ``xfail(strict=True)`` while the
defect exists, so a fix flips it to XPASS -> failure until the marker is removed.
Finding IDs (T*, C*, P*, S*, Sec*) refer to the review / work plan.
"""

from __future__ import annotations

import os
import pty
import re
import select
import signal
import socket
import struct
import subprocess
import tempfile
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
    assert "@calc" in out.splitlines()[1]


def test_T2_multiline_string_untouched():
    src = 'echo "x\n@ 1+2 in string\nmk2Dim a 1 1\n"\n'
    assert translate(src) == src


def test_T2_continuation_not_new_statement():
    out = translate("echo a \\\n  mk2Dim b 1 1\n")
    assert "@mk2Dim" not in out


def test_T3_at_v_quoted_expression(tmp_path):
    res = milk_cli_script(
        '@ -v r "1+2"\necho "r=$r"\n@ -v s \'2*3\'\necho "s=$s"', tmp_path
    )
    assert value(res.stdout, "r") == "3"
    assert value(res.stdout, "s") == "6"


def test_T3_at_v_error_on_unknown_image(tmp_path):
    res = milk_cli_script('@ -v r "imean(nosuchimage)"\necho "rc=$? r=$r"', tmp_path)
    assert "r=imean(nosuchimage)" not in res.stdout
    assert "rc=0" not in res.stdout


def test_T4_at_expr_not_split_on_operators(tmp_path):
    res = milk_cli_script('@ x = (2 > 1) && (2 < 1)\n@ -v r x\necho "r=$r"', tmp_path)
    assert value(res.stdout, "r") == "0"


@bug(
    "T5", "un-sigiled command classified differently at top level vs in a sourced file"
)
def test_T5_top_level_and_sourced_agree(tmp_path):
    (tmp_path / "sub.milk").write_text('listvar >/dev/null 2>&1; echo "sub=$?"\n')
    res = milk_cli_script(
        'listvar >/dev/null 2>&1; echo "top=$?"\nsource sub.milk', tmp_path
    )
    assert value(res.stdout, "top") is not None
    assert value(res.stdout, "top") == value(res.stdout, "sub")


def test_T6_inline_result_not_globbed(tmp_path):
    (tmp_path / "a.zz").touch()
    (tmp_path / "b.zz").touch()
    res = milk_cli_script(
        "@mk2Dim t6 4 4\n@set s.t6.kw.OBS '*.zz'\necho @s.t6.kw.OBS", tmp_path
    )
    assert "*.zz" in res.stdout
    assert "a.zz" not in res.stdout


def test_T7_action_suffix_only_for_fps_proc():
    out = translate("@s.cam.kw.step\n")
    assert "@set" not in out


def test_T8_double_equals_is_not_assignment():
    out = translate("@f.ao.gain==3\n")
    assert "@set" not in out


def test_T9_multiword_value_not_truncated(tmp_path):
    res = milk_cli_script(
        '@mk2Dim t9 4 4\n@s.t9.kw.OBS=Sub aru\necho "setrc=$?"\n'
        '@get -v v s.t9.kw.OBS\necho "v=$v"',
        tmp_path,
    )
    assert value(res.stdout, "setrc") != "0" or value(res.stdout, "v") == "Sub aru"


def test_T10_hash_in_param_length(tmp_path):
    res = milk_cli_script('arr=(a b c)\nn=$(echo ${#arr[@]})\necho "n=$n"', tmp_path)
    assert value(res.stdout, "n") == "3"
    assert "unexpected end" not in res.stdout + res.stderr


def test_T11_dollar_single_quote_escape():
    out = translate("echo $'it\\'s' @s.a.b\n")
    assert "@get" in out


def test_T12_second_heredoc_body_untouched():
    src = "cat <<A <<B\na\nA\nhelp\nB\n"
    assert translate(src) == src


def test_T13_lineno_matches_source(tmp_path):
    res = milk_cli_script('echo "l=$LINENO"', tmp_path)
    assert value(res.stdout, "l") == "1"


@pytest.mark.parametrize(
    "src, must_have, must_not_have",
    [
        ("xs=@s.${cam}.xsize\n", "@get -v REPLY s.${cam}.xsize", None),
        ('echo "${@s.c.xsize}"\n', "@get -v REPLY s.c.xsize", "${${"),
        ('echo "\\@{lit}"\n', 'echo "@{lit}"', "@calc"),
        ("echo foo@{a,b}\n", "foo@{a,b}", "@calc"),
        ('echo "$@f.x.y" user@s.ex.org\n', '"$@f.x.y" user@s.ex.org', "@get"),
        ("val=@{a * 10 + b[2, 3]}\n", '@calc -v REPLY "a * 10 + b[2, 3]"', None),
        ("echo @{a && b}; ls\n", '@calc -v REPLY "a && b"', None),
        ("x=$(listim)\ny=`listim`\n", "x=$(@listim)\ny=`@listim`", None),
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


# --------------------------------------- C*: coprocess, REPL, milk-script (bash/milk_bash_coproc.c ...)


def test_C1_subshell_exit_does_not_end_session(tmp_path):
    res = milk_cli_stdin("y=$(exit 3)\n( exit 0 )\necho two", tmp_path)
    assert "two" in res.stdout


def test_C1_bare_exit_uses_last_status(tmp_path):
    # milk-script surfaces the exit code; milk-cli -s does not yet (C11)
    res = milk_script("false\nexit", tmp_path)
    assert res.returncode == 1


def test_C2_time_does_not_deadlock(tmp_path):
    res = milk_cli_script("@time echo hi\necho after", tmp_path, timeout=10)
    assert res.returncode != 124
    assert "after" in res.stdout


@bug("C3", "Ctrl-C SIGKILLs the process group and silently respawns bash")
def test_C3_ctrl_c_keeps_shell_state(tmp_path):
    out = pty_session(
        [
            ("line", "x=5", 0.5),
            ("line", "for i in 1 2 3; do sleep 1; done", 0.5),
            ("raw", "\x03", 1.0),
            ("line", 'echo "x=[$x]"', 1.0),
        ],
        tmp_path,
    )
    assert "x=[5]" in out


@bug("C4", "in-process CALL clobbers REPL fault isolation; later ^C ignored")
def test_C4_ctrl_c_after_milk_call(tmp_path):
    out = pty_session(
        [
            (
                "line",
                '@listim >/dev/null; s=$EPOCHSECONDS; sleep 6; echo "el=$((EPOCHSECONDS-s))"',
                1.0,
            ),
            ("raw", "\x03", 1.0),
            ("line", 'echo "done=1"', 6.0),
        ],
        tmp_path,
    )
    assert "el=6" not in out
    assert "done=1" in out


@bug("C5", "bash coproc in background pgrp: `read` from tty stops it (SIGTTIN)")
def test_C5_read_from_tty(tmp_path):
    out = pty_session(
        [("line", 'read v; echo "got=[$v]"', 0.5), ("line", "hello", 1.0)], tmp_path
    )
    assert "got=[hello]" in out


@bug("C6", "bash death detected only by EOF on inherited fd 199")
def test_C6_bash_death_detected_promptly(tmp_path):
    t0 = time.time()
    milk_cli_script("sleep 8 &\nkill -9 $$", tmp_path, timeout=15)
    assert time.time() - t0 < 4


@bug(
    "C7",
    "shutdown stops servicing the socket after 5 s -> milk calls from EXIT traps deadlock",
)
def test_C7_slow_exit_trap_can_call_milk(tmp_path):
    res = milk_cli_script(
        "trap 'sleep 6; @calc -v z 1+1; echo \"deferdone=$z\"' EXIT\necho start",
        tmp_path,
        timeout=20,
    )
    assert res.returncode != 124
    assert value(res.stdout, "deferdone") == "2"


@bug("C8", "REPL idle select() does not service the server socket")
def test_C8_background_job_served_while_idle(tmp_path):
    out = pty_session([("line", '(@calc -v q 1+1; echo "bg=$q") &', 2.5)], tmp_path)
    assert "bg=2" in out


def test_C9_milk_script_runs_milk_commands(tmp_path):
    res = milk_script('@mk2Dim c9 4 4\necho "rc=$?"', tmp_path)
    assert value(res.stdout, "rc") == "0"


def test_C9_milk_script_positional_args(tmp_path):
    res = milk_script('echo "n=$# a1=$1 a2=$2"', tmp_path, args=["X", "2+3"])
    assert "n=2 a1=X a2=2+3" in res.stdout


def test_C9_milk_script_errexit(tmp_path):
    res = milk_script("false\necho after", tmp_path, opts=["-e"])
    assert "after" not in res.stdout
    assert res.returncode != 0


def test_C10_heredoc_in_piped_repl(tmp_path):
    res = milk_cli_stdin("cat <<EOF\nhello_c10\nEOF\necho after", tmp_path)
    assert "hello_c10" in res.stdout
    assert "command not found" not in res.stdout + res.stderr


@bug("C11", "milk-cli -s exits 0 when the last command failed")
def test_C11_script_exit_status(tmp_path):
    res = milk_cli_script("echo x\nfalse", tmp_path)
    assert res.returncode == 1


@bug("C11", "$? reset to 0 between piped REPL lines")
def test_C11_status_preserved_between_lines(tmp_path):
    res = milk_cli_stdin('false\necho "st=$?"', tmp_path)
    assert value(res.stdout, "st") == "1"


@bug("C12", "cli_var_set evaluates synced string values as math")
def test_C12_string_var_not_evaluated(tmp_path):
    res = milk_cli_script('s="sqrt(16)"\n@ -v r s+1\necho "r=$r"', tmp_path)
    assert value(res.stdout, "r") != "5"


def test_C13_defer_preserves_quoting(tmp_path):
    res = milk_cli_script("@defer printf '<%s>\\n' \"a b\"\necho start", tmp_path)
    assert "<a b>" in res.stdout


def test_C13_defer_keeps_user_exit_trap(tmp_path):
    res = milk_cli_script(
        "trap 'echo usertrap' EXIT\n@defer echo deferred\necho start", tmp_path
    )
    assert "usertrap" in res.stdout and "deferred" in res.stdout


def test_C14_local_reply_prints(tmp_path):
    res = milk_cli_script(
        'f(){ local REPLY; @calc 1+2; }\nout=$(f)\necho "out=$out"', tmp_path
    )
    assert value(res.stdout, "out") == "3"


def test_C15_cwd_startup_file_not_executed(tmp_path):
    (tmp_path / "CLIstartup.txt").write_text("echo STARTUP_RAN\n")
    res = milk_cli_stdin("echo hi", tmp_path)
    assert "hi" in res.stdout
    assert "STARTUP_RAN" not in res.stdout


def test_C16_mload_failure_status(tmp_path):
    res = milk_cli_script(
        '@mload NoSuchModuleXYZ >/dev/null 2>&1\necho "rc=$?"', tmp_path
    )
    assert value(res.stdout, "rc") not in (None, "0")


# ------------------------------------------------ P*: calc and shm_prop


def test_P1_long_expression(tmp_path):
    expr = "+".join(["1"] * 36)
    res = milk_cli_script(f'@calc -v r "{expr}"\necho "rc=$? r=$r"', tmp_path)
    assert "r=36" in res.stdout or "rc=0" not in res.stdout


def test_P2_fps_set_rejects_garbage():
    res = _run_bash("""
        milk-fpsexec-imggen-mkrndim -n p2fps fpsinit >/dev/null 2>&1
        @set f.p2fps.xsize 64
        rc=0; @set f.p2fps.xsize abc || rc=$?
        echo "rc=$rc v=$(@get f.p2fps.xsize)"
        """)
    assert "v=64" in res.stdout
    assert "rc=0" not in res.stdout


@bug("P3", "calc keeps a stale mapping after the stream is re-created")
def test_P3_calc_sees_recreated_stream(tmp_path):
    (tmp_path / "mk8.milk").write_text('@mk2Dim "@S:p3" 8 8\n@calc "p3 = 5"\n')
    res = _run_bash(f"""
        @mk2Dim "@S:p3" 4 4 >/dev/null
        @calc "p3 = 1"
        @calc -v a "imean(p3)"
        env -u MILKCLI_SOCK HOME="{tmp_path}" milk-cli -s "{tmp_path}/mk8.milk" >/dev/null 2>&1
        @calc -v b "imean(p3)"
        echo "a=$a xs=$(@get s.p3.xsize) b=$b"
        """)
    assert "xs=8" in res.stdout, res.stdout + res.stderr
    assert "b=5" in res.stdout


def test_P4_ternary(tmp_path):
    res = milk_cli_script(
        '@calc -v r "1 ? 2 : 3"\necho "r=$r"\n@calc -v s "0?2:3"\necho "s=$s"', tmp_path
    )
    assert value(res.stdout, "r") == "2"
    assert value(res.stdout, "s") == "3"


def test_P4_image_ternary(tmp_path):
    res = milk_cli_script(
        '@mk2Dim p4i 4 4\n@calc "p4i = 5"\n@calc -v k "imean(p4i > 1 ? p4i : 0)"\necho "k=$k"',
        tmp_path,
    )
    assert value(res.stdout, "k") == "5"


def test_P13_user_tmpcalc_image_survives(tmp_path):
    """Temp cleanup deletes only temps it created, never a user image that happens to be named _tmpcalcN."""
    res = milk_cli_script(
        "@mk2Dim _tmpcalc0 2 2\n@mk2Dim p13 2 2\n"
        '@calc -v r "imean(p13 + 1)"\necho "r=$r"\n'
        '@exists s._tmpcalc0; echo "kept=$?"',
        tmp_path,
    )
    assert value(res.stdout, "r") == "1"
    assert value(res.stdout, "kept") == "0"


def test_P5_string_functions(tmp_path):
    res = milk_cli_script('@calc -v u \'toupper("abc")\'\necho "u=$u"', tmp_path)
    assert value(res.stdout, "u") == "ABC"


def test_P8_long_keyword_single_slot(tmp_path):
    res = milk_cli_script(
        "@mk2Dim p8 4 4\n@set s.p8.kw.VERYLONGKEYWORDNAME_X 1\n"
        "@set s.p8.kw.VERYLONGKEYWORDNAME_X 2\n@get -j s.p8",
        tmp_path,
    )
    assert res.stdout.count('"VERYLONGKEYWORD') == 1


def test_P7_keyword_double_roundtrip():
    res = _run_bash("""
        @mk2Dim "@S:p7" 2 2 >/dev/null
        @set s.p7.kw.A 0.1
        @set s.p7.kw.B 3.141592653589793
        @set s.p7.kw.C 1e-7
        echo "A=$(@get s.p7.kw.A) B=$(@get s.p7.kw.B) C=$(@get s.p7.kw.C)"
        """)
    assert "A=0.1 B=3.141592653589793 C=1e-07" in res.stdout, res.stdout + res.stderr


def test_P15_restore_dash_is_literal(tmp_path):
    snap = tmp_path / "s.json"
    snap.write_text('{\n  "s.p15.kw.D": "-"\n}\n')
    res = _run_bash(f"""
        @mk2Dim "@S:p15" 2 2 >/dev/null
        @set s.p15.kw.D before
        echo evil | @restore "{snap}" >/dev/null 2>&1
        echo "D=$(@get s.p15.kw.D)"
        """)
    assert "D=-" in res.stdout, res.stdout + res.stderr


def test_P9_integer_power(tmp_path):
    res = milk_cli_script(
        '@calc -v a "2^-1"\necho "a=$a"\n@calc -v b "2^63"\necho "b=$b"', tmp_path
    )
    assert value(res.stdout, "a") == "0.5"
    assert not (value(res.stdout, "b") or "-").startswith("-")


def test_P10_image_in_second_argument(tmp_path):
    res = milk_cli_script(
        '@mk2Dim p10 4 4\n@calc "p10 = 5"\n@calc -v r "imean(min(3, p10))"\necho "r=$r"\n'
        '@calc -v ok "imean(min(p10, 3))"\necho "ok=$ok"',
        tmp_path,
    )
    assert value(res.stdout, "ok") == "3"  # image-first form works today
    assert value(res.stdout, "r") == "3"


def test_P10_unknown_identifier_is_error(tmp_path):
    res = milk_cli_script(
        '@calc -v r "max(3, nosuchimage)"\necho "rc=$? r=$r"', tmp_path
    )
    assert "rc=0" not in res.stdout


def test_P14_plain_string_equality(tmp_path):
    res = milk_cli_script('@calc -v r \'"RUN" == "RUNNING"\'\necho "r=$r"', tmp_path)
    assert value(res.stdout, "r") == "0"


@bug("WP9", "undefined calc identifier evaluates to its own name")
def test_WP9_undefined_identifier_is_error(tmp_path):
    res = milk_cli_script(
        '@calc -v r "strlen(foobar_undefined)"\necho "rc=$?"', tmp_path
    )
    assert value(res.stdout, "rc") != "0"


@bug("WP9", "failed inline expansion is silent")
def test_WP9_failed_inline_expansion_reports(tmp_path):
    res = milk_cli_script('echo "[@l.nosuchimage.xsize]"', tmp_path)
    assert res.stderr.strip() != "" or res.returncode != 0


# ------------------------------------------------ S*/Sec*: server and IPC


def _server_env(tmp_path: Path) -> tuple[dict[str, str], str]:
    env, _ = _env()
    name = f"@milk-cli-{os.getuid()}-wp0-{os.getpid()}-{time.monotonic_ns() % 100000}"
    env["MILKCLI_SOCK"] = name
    env["TMPDIR"] = str(tmp_path)
    return env, name


def _milk_call(
    env: dict[str, str], *args: str, timeout: float = 10, stdin: str | None = None
) -> subprocess.CompletedProcess[str]:
    return _run(
        [str(_install_dir() / "bin" / "milk-call"), *args], env, timeout, stdin=stdin
    )


class _Server:
    """Dedicated milk-cli server on a unique abstract socket."""

    def __init__(self, tmp_path: Path) -> None:
        self.env, self.name = _server_env(tmp_path)
        self.proc = subprocess.Popen(
            [MILK_CLI_EXEC, "-q", "--server", self.name],
            env=self.env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        for _ in range(
            200
        ):  # raw connect: a milk-call probe could auto-start a rival server
            try:
                _connect(self.name).close()
                break
            except OSError:
                time.sleep(0.025)

    def call(self, *args: str, **kw) -> subprocess.CompletedProcess[str]:
        return _milk_call(self.env, *args, **kw)

    def alive(self) -> bool:
        return self.proc.poll() is None

    def close(self) -> None:
        _kill_session(self.proc.pid)
        _kill_by_cmdline(self.name)
        self.proc.wait(timeout=5)


@pytest.fixture
def server(tmp_path):
    srv = _Server(tmp_path)
    yield srv
    srv.close()


_MAGIC = 0x4D494C4B


def _frame(ftype: int, payload: bytes = b"") -> bytes:
    return struct.pack("<IHBBI", _MAGIC, 1, ftype, 0, len(payload)) + payload


def _connect(name: str) -> socket.socket:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect("\0" + name[1:])
    return s


def test_S1_malformed_call_frame(server):
    s = _connect(server.name)
    s.sendall(_frame(3, struct.pack("<I", 5) + b"listim\0"))
    s.settimeout(3)
    try:
        s.recv(4096)
    except OSError:
        pass
    s.close()
    time.sleep(0.3)
    assert server.alive()


def test_S2_stalled_client_does_not_block(server):
    s = _connect(server.name)
    s.sendall(b"MI")
    try:
        res = server.call("call", "listim", timeout=4)
    finally:
        s.close()
    assert res.returncode != 124


@bug("S3", "server-forked handlers lose state: grab into a local image creates nothing")
def test_S3_grab_into_local_image(server):
    server.call("call", "mk2Dim", "@S:s3src", "4", "4")
    writer = subprocess.Popen(
        [
            "bash",
            "-c",
            'for i in $(seq 40); do milk-call calc "s3src = $i" >/dev/null; '
            "sleep 0.05; done",
        ],
        env=server.env,
        start_new_session=True,
    )
    try:
        grab = server.call("grab", "s.s3src", "2", "@L:s3cube")
    finally:
        _kill_session(writer.pid)
    assert grab.returncode == 0, grab.stdout + grab.stderr
    assert server.call("exists", "l.s3cube").returncode == 0


@bug(
    "S3",
    "SHUTDOWN handled in a forked child does not stop the server (racy: ~25% pass)",
    strict=False,
)
def test_S3_server_stop(server):
    busy = subprocess.Popen(
        [str(_install_dir() / "bin" / "milk-call"), "call", "usleep", "2000000"],
        env=server.env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(
        0.5
    )  # SHUTDOWN arrives while a CALL is in flight -> handled by a forked child
    server.call("shutdown")
    busy.wait(timeout=10)
    time.sleep(1.0)
    assert not server.alive()


def test_S4_streamwrite_devnull(server):
    server.call("call", "mk2Dim", "@S:s4", "4", "4")
    server.call("call", "streamwrite", "s4", "7", "8", "9")  # stdin = /dev/null
    res = server.call("get", "s.s4.xsize")
    assert res.stdout.strip() == "4"


def test_S4_streamwrite_in_read_loop_keeps_loop_input():
    res = _run_bash("""
        @mk2Dim "@S:s4l" 4 4 >/dev/null
        n=0
        printf "10\\n20\\n30\\n" | while read v; do @streamwrite s4l $v; n=$((n+1)); echo "it=$n"; done
        """)
    assert "it=3" in res.stdout, res.stdout + res.stderr


def test_S5_replay_cancel_stops_worker(server):
    server.call("call", "mk2Dim", "@S:s5src", "4", "4")

    def nproc() -> int:
        return len(
            [
                p
                for p in os.listdir("/proc")
                if p.isdigit() and server.name in _cmdline(int(p))
            ]
        )

    before = nproc()
    _milk_call(
        server.env,
        "call",
        "replay",
        "--loop",
        "--fps",
        "20",
        "s.s5src",
        "s5dst",
        timeout=1,
    )
    time.sleep(1.5)
    assert nproc() == before


def _cmdline(pid: int) -> str:
    try:
        return (
            Path(f"/proc/{pid}/cmdline")
            .read_bytes()
            .replace(b"\0", b" ")
            .decode(errors="replace")
        )
    except OSError:
        return ""


def test_S10_subshell_first_call_does_not_own_server():
    res = _run_bash("""
        x=$(@mk2Dim "@L:s10im" 2 2 >/dev/null; echo made)
        sleep 0.5
        @exists l.s10im; echo "rc=$?"
        """)
    assert "rc=0" in res.stdout, res.stdout + res.stderr


def test_S10_daemon_does_not_inherit_caller_fds(tmp_path):
    hold = tmp_path / "holdme.txt"
    res = _run_bash(f"""
        exec 9>"{hold}"
        @calc -v z 1+1 >/dev/null
        spid=$(pgrep -f -- "--server ${{_MILK_LOCAL_SOCK}}" | head -1)
        echo "held=$(ls -l /proc/$spid/fd 2>/dev/null | grep -c holdme)"
        """)
    assert "held=0" in res.stdout, res.stdout + res.stderr


def test_S6_set_kw_with_comment(server):
    server.call("call", "mk2Dim", "@L:s6", "4", "4")
    assert server.call("exists", "l.s6").returncode == 0
    server.call("set", "l.s6.kw.FOO=3", "mycomment")
    res = server.call("get", "l.s6.kw.FOO")
    assert res.stdout.strip() == "3"


def test_S7_fd_leak_on_bad_frames(server):
    import array

    def nfds() -> int:
        return len(os.listdir(f"/proc/{server.proc.pid}/fd"))

    before = nfds()
    for _ in range(10):
        s = _connect(server.name)
        hdr = struct.pack("<IHBBI", _MAGIC, 1, 3, 0, 100 * 1024 * 1024)
        s.sendmsg(
            [hdr], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [0, 1, 2]))]
        )
        try:
            s.settimeout(2)
            s.recv(1)
        except OSError:
            pass
        s.close()
    time.sleep(0.3)
    assert nfds() - before < 5


def _fake_server(name: str, behaviour: str) -> subprocess.Popen:
    """Minimal hostile server: 'close' drops the request, 'setvar' replies SETVAR PATH=/evil."""
    code = f"""
import socket, struct, sys
L = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); L.bind("\\0" + {name[1:]!r}); L.listen(1)
print("ready", flush=True)
c, _ = L.accept()
msg, anc, fl, ad = c.recvmsg(12, socket.CMSG_SPACE(16))
m, v, t, f, l = struct.unpack("<IHBBI", msg)
if {behaviour!r} == "setvar":
    c.recv(l)
    fr = lambda t, p: struct.pack("<IHBBI", 0x4D494C4B, 1, t, 0, len(p)) + p
    c.sendall(fr(102, b"PATH\\0/evil\\0") + fr(104, struct.pack("<i", 0)))
c.close()
"""
    p = subprocess.Popen(
        ["python3", "-c", code],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert p.stdout.readline().strip() == "ready"
    return p


def test_S8_transport_error_distinct_rc(tmp_path):
    env, name = _server_env(tmp_path)
    fake = _fake_server(name, "close")
    try:
        res = _milk_call(env, "test", "1 < 2", timeout=5)
    finally:
        _kill_session(fake.pid)
        _kill_by_cmdline(name)
    assert res.returncode == 255
    assert res.stderr.strip() != ""


def test_Sec1_setvar_not_trusted(tmp_path):
    env, name = _server_env(tmp_path)
    fake = _fake_server(name, "setvar")
    try:
        res = _run(
            [
                "bash",
                "-c",
                f'source "{_install_dir()}/bin/milk.bash"; MILKCLI_SOCK={name}; '
                '@listim >/dev/null 2>&1; echo "path=$PATH"',
            ],
            env,
            10,
        )
    finally:
        _kill_session(fake.pid)
        _kill_by_cmdline(name)
    assert "path=/evil" not in res.stdout


def test_Sec1_override_file_must_be_private(server, tmp_path):
    """The per-shell socket override file is honoured only if it is ours, private and not a symlink."""
    server.call("call", "mk2Dim", "@L:wp6mark", "2", "2")
    fake = f"@milk-cli-{os.getuid()}-wp6fake-{os.getpid()}"
    env = dict(
        server.env, MILKCLI_SOCK=fake
    )  # the override file points at the real server
    script = r"""
ov="$TMPDIR/.milk_active_sock_$(id -u)_$$"
printf %s "$1" > "$ov.real"
case "$2" in
    private) mv "$ov.real" "$ov"; chmod 600 "$ov" ;;
    shared)  mv "$ov.real" "$ov"; chmod 666 "$ov" ;;
    link)    ln -s "$ov.real" "$ov" ;;
esac
milk-call exists l.wp6mark >/dev/null 2>&1
echo "rc=$?"
rm -f "$ov" "$ov.real"
"""
    rcs = {}
    try:
        for mode in ("private", "shared", "link"):
            res = _run(["bash", "-c", script, "_", server.name, mode], env, 15)
            rcs[mode] = value(res.stdout, "rc")
    finally:
        _kill_by_cmdline(fake)
    assert rcs["private"] == "0"  # honoured -> real server, which has the marker image
    assert rcs["shared"] not in (None, "0")  # world-writable -> ignored
    assert rcs["link"] not in (None, "0")  # symlink -> ignored


def test_Sec2_eval_fd_escaping(server, tmp_path):
    evf = tmp_path / "eval.out"
    payload = "s3 = \"a'$(echo PWNED >&2)'b\""
    env = dict(server.env, MILK_CALL_EVAL_FD="7")
    _run(["bash", "-c", 'milk-call calc "$1" 7>"$2"', "_", payload, str(evf)], env, 10)
    res = subprocess.run(
        ["bash", "-c", 'source "$1"', "_", str(evf)], text=True, capture_output=True
    )
    assert "PWNED" not in res.stderr


# ------------------------------------------------ latency benchmark (reported, not asserted)


BENCH_OPS = [
    (300, ":"),
    (300, "y=$((1+2))"),
    (300, "@get -v x s.bim.xsize"),
    (300, "x=@s.bim.xsize"),
    (300, "@set s.bim.kw.FOO=1"),
    (300, "@calc -v y 1+2"),
    (300, "y=@{1+2}"),
    (100, "@listim >/dev/null"),
    (50, "y=$(@calc 1+2)"),
]


def _bench_script() -> str:
    """One literal loop per op (no eval: the translator must see each op as script text)."""
    lines = ["@mk2Dim bim 64 64 >/dev/null"]
    for n, op in BENCH_OPS:
        lines.append(
            f"t0=$EPOCHREALTIME; for ((i=0;i<{n};i++)); do {op}; done; t1=$EPOCHREALTIME"
        )
        # label in single quotes so the translator leaves it alone
        lines.append(f"echo \"BENCH $(( (${{t1/./}}-${{t0/./}})/{n} )) us :: \"'{op}'")
    return "\n".join(lines)


def test_latency_benchmark(tmp_path, capsys):
    """Print per-op latency of common @ operations (WP13-15 targets: calc<=20us, listim<=60us, prop<=5us)."""
    res = milk_cli_script(_bench_script(), tmp_path, timeout=60)
    lines = [ln for ln in res.stdout.splitlines() if ln.startswith("BENCH")]
    with capsys.disabled():
        print("\n" + "\n".join(lines))
    assert len(lines) == len(BENCH_OPS), res.stdout + res.stderr
