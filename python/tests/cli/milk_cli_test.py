"""The milk-cli program itself, and milk-script where they share code.

Purpose:     milk-cli is both the interactive prompt and a script runner: its exits, traps and
             Ctrl-C behave as in bash, no piped line is lost or held back, and it finds its own
             install without any setup.
Does:        `exit` in a subshell does not end milk-cli, and a bare `exit` keeps the last
             status; a file named STOPCLI does not end it; an EXIT trap runs; Ctrl-C at a pty prompt stops a loop and keeps the shell's
             variables; calc-mode lines follow the shell's quoting and comments and the
             calculator's strings; `milk-cli -h` lists the options milk-cli takes;
             `@config sessionlog` logs each line read; 400 000 piped lines all run; piped lines
             reach the history log, and an unclosed parenthesis in calc mode does not hold the
             batch; milk-cli and milk-script run by path export their own install.
Covers:      milk-cli (src/CLImain.c, src/cli/CLIcore/CLIcore.c: options, input loop, piped
             batches), its help (libmilkscript/core/CLIcore_help_topics.c), cli_sessionlog()
             (CLIcore/CLIcore_UI_builtins.c), the history log (CLIcore/CLIcore_UI_history.c),
             CLI_export_own_installdir() (libmilkscript/core/CLIcore_datainit.c), milk-script
             (libmilkscript/milkscript_main.c).
Uses:        conftest.py: milk_bash, milk_cli_script, milk_cli_stdin, milk_script, pty_session,
             value, _run, _isolated_shm_env, _install_dir.
See also:    docs/guide/CLIcore.md, "Command-Line Options", "Persistent History",
             "Session Logging"; docs/guide/calculator.md, "Calculator (`@calc`, `@test`, `@pipe`)".
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
    HAVE_TREESITTER,
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


def test_a_file_named_stopcli_does_not_end_milk_cli(tmp_path):
    """A file `STOPCLI` in the directory milk-cli runs in is a file like any other."""
    (tmp_path / "STOPCLI").touch()
    res = milk_cli_script('echo "ran=1"', tmp_path)
    assert value(res.stdout, "ran") == "1", res.stdout + res.stderr
    assert res.returncode == 0, res.stdout + res.stderr


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


def test_milk_cli_rejects_removed_idle_option(tmp_path):
    """`milk-cli --idle` (stop the shell while X is in use via the never-installed runidle) is removed."""
    res = milk_bash(
        'milk-cli -q --idle -c "echo IDLE_RAN" 2>&1; echo "rc=$?"\n'
        '@help cmdopts 2>&1 | grep -c -- "--idle " || true',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert "runidle" not in out, out
    # getopt reads `--idle` as an abbreviation of --idle-timeout, which then needs a number
    assert "--idle-timeout needs a number of seconds" in out, out
    assert "IDLE_RAN" not in out and "rc=0" not in out, out
    assert out.rstrip().endswith("0"), out


def test_quiet_cli_prints_only_what_the_commands_print(tmp_path):
    """`milk-cli -q` with -c, -s or piped input adds no blank line of its own."""
    res = milk_bash(
        "printf 'echo hi\\n' > s.milk\n"
        'a=$(milk-cli -q -c "echo hi" | wc -c); b=$(milk-cli -q -s s.milk | wc -c)\n'
        'c=$(echo "echo hi" | milk-cli -q | wc -c); echo "n=$a $b $c"',
        tmp_path,
    )
    assert value(res.stdout, "n") == "3 3 3", res.stdout + res.stderr


def test_cmd_string_runs_whole(tmp_path):
    """`milk-cli -c` runs the whole string, however long, and exits with its status."""
    res = milk_bash(
        "s=$(printf %05000d 0)\n"
        'milk-cli -q -c "x=$s; echo len=\\${#x}; exit 3"; echo "rc=$?"',
        tmp_path,
    )
    assert value(res.stdout, "len") == "5000", res.stdout + res.stderr
    assert value(res.stdout, "rc") == "3", res.stdout + res.stderr


def test_script_arguments_reach_the_script(tmp_path):
    """`milk-cli -s file args` and `milk-script file args` set $0 to the file and $1.. to args."""
    f = tmp_path / "a.milk"
    f.write_text('echo "zero=$(basename "$0")"; echo "args=$1|$2|$#"\n')
    env, shm_dir = _isolated_shm_env()
    try:
        res = _run(
            [MILK_CLI_EXEC, "-q", "-s", str(f), "x y", "-e"], env, 20, cwd=str(tmp_path)
        )
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)
    out = res.stdout + res.stderr
    assert value(res.stdout, "args") == "x y|-e|2", out
    assert value(res.stdout, "zero") == "a.milk", out
    res = milk_script(f.read_text(), tmp_path, args=["p", "q"])
    assert value(res.stdout, "args") == "p|q|2", res.stdout + res.stderr


def test_history_stats_typed_and_piped_save(tmp_path):
    """@history --stats lists the module commands run, not the commands behind keywords; piped
    milk-cli has no prompt history: @history save refuses (2) and writes nothing."""
    res = milk_bash(
        "printf 'mem.listim >/dev/null\\n@config synhl off >/dev/null\\n@history >/dev/null\\n"
        '@help @history >/dev/null\\n@history --stats\\n@history save hs.milk; echo "save=$?"\\n\''
        " | milk-cli -q 2>err.txt",
        tmp_path,
    )
    rows = re.findall(r"^\s+(\S+)\s+(\d+)$", res.stdout, re.M)
    assert rows == [("mem.listim", "1")], res.stdout
    assert "save=2" in res.stdout, res.stdout
    assert "@history save: at the milk-cli prompt" in (tmp_path / "err.txt").read_text()
    assert not (tmp_path / "hs.milk").exists()


def test_milk_script_stdout_is_the_script_output(tmp_path):
    """A script's stdout is its data: no MILK_SHM_DIR line, '.' dots or LOADED lines."""
    f = tmp_path / "s.milk"
    f.write_text('#!/usr/bin/env milk-script\necho "data=$1"\n')
    f.chmod(0o755)
    res = milk_bash(f'env -u MILK_QUIET "{f}" 7', tmp_path)
    assert res.stdout == "data=7\n", res.stdout + res.stderr
    assert res.returncode == 0, res.stdout + res.stderr


def test_top_level_break_and_continue_do_not_abort_coprocess(tmp_path):
    """Top-level `break`/`continue` in milk-cli/milk-script stay bounded and keep vars (`front-1`)."""
    res = milk_bash(
        "milk-cli -q -c 'declare -A m=([k]=v); continue; echo \"after_cont=${m[k]}\"'\n"
        "milk-cli -q -c 'declare -A m=([k]=w); break; echo \"after_brk=${m[k]}\"'\n"
        'printf "x=kept\\ncontinue\\necho piped=\\$x\\n" | milk-cli -q',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert "after_cont=v" in out, out
    assert "after_brk=w" in out, out
    assert "piped=kept" in out, out


@pytest.mark.skipif(not HAVE_TREESITTER, reason="milk-cli built without tree-sitter")
def test_format_preserves_heredoc_and_multiline_quotes(tmp_path):
    """`milk-cli --format` keeps heredoc bodies, terminators and quoted multiline strings (`front-3`)."""
    src = tmp_path / "fmt.milk"
    script = (
        "if true; then\n"
        "cat <<'EOF'\n"
        "  keep indent\n"
        "if\n"
        "EOF\n"
        'msg="line1   \n'
        '   line2"\n'
        "fi\n"
    )
    src.write_text(script)
    res = milk_bash(f'milk-cli --format "{src}"', tmp_path)
    assert res.returncode == 0, res.stdout + res.stderr
    assert (
        "\ncat <<'EOF'\n" in res.stdout or "\n    cat <<'EOF'\n" in res.stdout
    ), res.stdout
    assert "\n  keep indent\nif\nEOF\n" in res.stdout, res.stdout
    assert (
        'msg="line1   \n   line2"\n' in res.stdout
        or '    msg="line1   \n   line2"\n' in res.stdout
    ), res.stdout


def test_exit_invalid_and_edge_arguments(tmp_path):
    """`exit` with invalid/empty/octal/multi args matches bash exit semantics (`front-9`)."""
    res = milk_bash(
        "for c in 'exit \"\"' 'exit abc' 'exit 3x' 'exit 0x10' 'exit 1e2'; do\n"
        '  milk-cli -q -c "$c" >/dev/null 2>&1; echo "bad=$?"\n'
        "done\n"
        'milk-cli -q -c "exit 010"; echo "oct=$?"\n'
        'milk-cli -q -c "exit \\" 5 \\""; echo "spc=$?"\n'
        'milk-cli -q -c "exit -- 7"; echo "dd=$?"\n'
        'milk-cli -q -c "exit 1 2" >/dev/null 2>tma.err; echo "tma=$?"\n'
        'grep -q "too many arguments" tma.err && echo "tma_msg=ok"\n'
        'out_cont=$(milk-cli -q -c "exit abc; echo still=\\$?" 2>bad.err)\n'
        'echo "$out_cont"; echo "nmsg=$(grep -c "numeric argument required" bad.err)"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert out.count("bad=2") == 5, out
    assert "oct=10" in out, out
    assert "spc=5" in out, out
    assert "dd=7" in out, out
    assert "tma=1" in out and "tma_msg=ok" in out, out
    assert "still=2" in out and "nmsg=1" in out, out


def test_startup_file_long_path_fifo_and_direct(tmp_path):
    """`-s` accepts startup script paths exceeding 256 bytes in direct and `-f` FIFO modes (`front-11`)."""
    deep = tmp_path / ("d_" + "a" * 90) / ("d_" + "b" * 90) / ("d_" + "c" * 90)
    deep.mkdir(parents=True)
    scr = deep / "long_startup.milk"
    assert len(str(scr)) > 260
    scr.write_text('echo "LONG_""S_OK"\nexit 0\n')
    res = milk_bash(
        f'milk-cli -q -s "{scr}"\n' f'timeout 10 milk-cli -q -f -n longp -s "{scr}"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert res.returncode == 0, out
    assert out.count("LONG_S_OK") == 2, out
