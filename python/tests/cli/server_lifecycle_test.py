"""Session servers start, report and stop as units do, and a session outlives its commands' errors.

Purpose:     milk-cli servers start through milk_detach, as units do
             (docs/arch/unit_lifecycle.md, U18): a failed start says why, the output of a crash
             stays, @session end ends even a server that ignores its shutdown command, and a
             service manager hears from it. A session must also outlive a crash or a Ctrl-C in a
             command it runs.
Does:        a server that cannot listen makes @session new fail with its reason (ERRNO=) and the
             end of its output, and `@session new --tmux` fails with a hint; a crash message
             stays in the output file; a server that ignores its shutdown command is ended by
             signals and its files go; a private server ends with its shell, a named one survives
             it, and neither keeps the caller's descriptors; $NOTIFY_SOCKET gets READY=1, then
             STOPPING=1 on SIGTERM; Ctrl-C stops a long @calc while the session answers other
             requests; Ctrl-C or a lost shell stops a keyword the session runs itself (@pipe) and
             the session goes on; a command sent with another MILK_SHM_DIR leaves the
             session's log where it is; a crash in a C extension leaves the session running; a shell
             keeps one connection between requests and starts the server again after it ended;
             captured command output is complete.
Covers:      the server start (libmilkscript/ipc/milk_ipc_common.c, on
             src/engine/libmilkcommon/process/milk_detach.c), its readiness and accept loop
             (libmilkscript/server/milk_server.c), the stop (libmilkscript/client/milk_sessions.c),
             the kept connection (libmilkscript/client/milk_client_conn.c), crash interception
             (libmilkscript/core/CLIcore_signals.c), milk_sdnotify.c (src/engine/libmilkcommon).
Uses:        conftest.py: milk_bash, value; the names fixture (servers of the test's own, ended
             after it) and the FUNCS shell functions (srvpid, srvout).
See also:    docs/arch/unit_lifecycle.md, "Where It Lives"; docs/guide/sessions.md,
             "Managing Sessions (`@session`)", "Using milk from Plain Bash"; unit_lifecycle_test.py.
"""

from __future__ import annotations

import contextlib
import os
import socket
import time
import uuid

import pytest

from milk.cliwrap import HAVE_CLI
from tests.cli.conftest import milk_bash, value

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.timeout(120),
]

# srvpid [NAME]: the PID @session status shows; srvout NAME: its output file
FUNCS = r"""
srvpid() { @session status "$@" | sed 's/\x1b\[[0-9;]*m//g' | sed -n 's/.*(PID \([0-9]*\),.*/\1/p'; }
srvout() { @session status "$1" | sed 's/\x1b\[[0-9;]*m//g' | sed -n 's/.*Output *: //p'; }
"""


@pytest.fixture
def names():
    """names(): a named server of this test's own (abstract socket names are global); stopped after."""
    made = []

    def make() -> str:
        made.append(f"t{uuid.uuid4().hex[:8]}")
        return made[-1]

    yield make
    if made:
        milk_bash(
            "".join(f"@session end {n} >/dev/null 2>&1\n" for n in made), timeout=60
        )


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def gone(pid: int, timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while alive(pid) and time.monotonic() < end:
        time.sleep(0.05)
    return not alive(pid)


def test_failed_start_reports_why(tmp_path, names):
    """A server that cannot listen (its socket name is taken) makes @session new fail with the
    reason it sent (ERRNO=) and the end of its output; --tmux fails with a hint."""
    name = names()
    holder = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    holder.bind(f"\0milk-cli-{os.getuid()}-{name}")
    try:
        res = milk_bash(
            f'@session new {name} -d 2>start.log; echo "rc=$?"\n'
            f'@session new {name} --tmux 2>tmux.log; echo "tmux=$?"',
            tmp_path,
        )
    finally:
        holder.close()
    err = (tmp_path / "start.log").read_text()
    out = res.stdout + res.stderr + err
    assert value(res.stdout, "rc") == "1", out
    assert (
        f"did not start: cannot listen on @milk-cli-{os.getuid()}-{name}: Address already in use"
        in err
    ), out
    assert (
        "milk-cli --server: bind: Address already in use" in err
    ), out  # the end of its output
    assert value(res.stdout, "tmux") == "2", out
    assert (
        "--tmux was removed: sessions start detached"
        in (tmp_path / "tmux.log").read_text()
    ), out


def test_crash_output_is_kept_and_a_stuck_server_is_signalled(tmp_path, names):
    """A server's crash message stays in its output file; a server that ignores its shutdown
    command (stopped here) is ended by signals, and its files go."""
    crash, stuck = names(), names()
    res = milk_bash(
        FUNCS + "ulimit -c 0\n"
        f"@session new {crash} -d >/dev/null; p=$(srvpid {crash}); o=$(srvout {crash})\n"
        'kill -SEGV "$p"; for _ in $(seq 100); do kill -0 "$p" 2>/dev/null || break; sleep 0.05; done\n'
        'echo "crash=$(grep -c "sig_handler received Segmentation fault" "$o")"\n'
        f"@session new {stuck} -d >/dev/null; p=$(srvpid {stuck}); o=$(srvout {stuck})\n"
        f'kill -STOP "$p"; @session end {stuck} 2>stop.log; echo "stop=$?"\n'
        'echo "alive=$(kill -0 "$p" 2>/dev/null && echo 1 || echo 0)"\n'
        'echo "files=$(ls "$o" 2>/dev/null | wc -l)"',
        tmp_path,
        timeout=60,
    )
    out = res.stdout + res.stderr + (tmp_path / "stop.log").read_text()
    assert value(res.stdout, "crash") == "1", out
    assert value(res.stdout, "stop") == "0", out
    assert value(res.stdout, "alive") == "0", out
    assert "ignored its shutdown command: stopped by SIGKILL" in out, out
    assert value(res.stdout, "files") == "0", out


def test_owners_and_descriptors(tmp_path, names):
    """A private server ends with the shell that owns it, a named one survives it; neither keeps
    the caller's descriptors."""
    name = names()
    res = milk_bash(
        FUNCS + 'exec 7>"$PWD/held.txt"\n'
        'mem.mk2Dim "@S:im" 4 4 >/dev/null\n'
        'echo "private=$(srvpid)"\n'
        f"@session new {name} -d >/dev/null\n"
        f'echo "named=$(srvpid {name})"\n'
        f"for p in $(srvpid) $(srvpid {name}); do ls -l /proc/$p/fd; done | grep -c held.txt "
        '| sed "s/^/held=/"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    private, named = int(value(res.stdout, "private")), int(value(res.stdout, "named"))
    assert value(res.stdout, "held") == "0", out
    assert gone(private), f"the private server {private} outlived its shell\n{out}"
    assert alive(named), f"the named server {named} ended with the shell\n{out}"
    milk_bash(f"@session end {name}", tmp_path)  # its shutdown command
    assert gone(named), f"the named server {named} ignored @session end\n{out}"


def test_server_tells_notify_socket(tmp_path, names):
    """A server that a service manager runs (systemd's Type=notify) says READY=1 once it listens, and on
    SIGTERM (the manager's stop) STOPPING=1 and exit status 0; the servers milk starts itself do not
    inherit the socket."""
    name = names()
    sock_path = tmp_path / "notify.sock"
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    sock.bind(str(sock_path))
    try:
        res = milk_bash(
            FUNCS
            + 'mem.mk2Dim "@S:im" 4 4 >/dev/null\n'  # a private server, started by milk
            f'milk-cli -q --server "@milk-cli-$(id -u)-{name}" </dev/null >srv.log 2>&1 &\n'
            f'for _ in $(seq 100); do [ -n "$(srvpid {name})" ] && break; sleep 0.05; done\n'
            'kill -TERM $!; wait $!; echo "rc=$?"',
            tmp_path,
            extra_env={"NOTIFY_SOCKET": str(sock_path)},
        )
        sock.setblocking(False)
        msgs = []
        with contextlib.suppress(BlockingIOError):
            while True:
                msgs.append(sock.recv(4096).decode())
    finally:
        sock.close()
    out = res.stdout + res.stderr + f"\nmessages: {msgs}"
    assert value(res.stdout, "rc") == "0", out
    assert msgs == [
        f"READY=1\nSTATUS=listening on @milk-cli-{os.getuid()}-{name}\n",
        "STOPPING=1\n",
    ], out


def test_ctrl_c_stops_a_running_calc_while_the_session_answers(tmp_path):
    res = milk_bash(
        'mem.mk2Dim im 8 8 >/dev/null; @calc "im = 3"; t0=$EPOCHREALTIME\n'
        '( @calc "for i = 1:100000 { for j = 1:100000 { x = i + j } }" 2>loop.err\n'
        '  echo "loop_rc=$?" ) & pid=$!\n'
        "sleep 1; t1=$EPOCHREALTIME\n"
        '@test "imean(im) > 2"; echo "test_rc=$?"\n'
        'echo "test_ms=$(( (${EPOCHREALTIME/./} - ${t1/./}) / 1000 ))"\n'
        "kill -INT $pid; wait $pid\n"
        'echo "loop_ms=$(( (${EPOCHREALTIME/./} - ${t0/./}) / 1000 ))"\n'
        '@calc -v m "imean(im * 2)"; echo "after=$m"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "test_rc") == "0", out
    assert int(value(res.stdout, "test_ms")) < 500, out  # served while the loop runs
    assert value(res.stdout, "loop_rc") == "130", out
    assert int(value(res.stdout, "loop_ms")) < 5000, out
    assert "interrupted" in (tmp_path / "loop.err").read_text(), out
    assert value(res.stdout, "after") == "6", out


def test_ctrl_c_or_a_lost_shell_in_a_session_keyword_keeps_the_session(tmp_path):
    """A keyword the session runs itself (@pipe here) has no command frame of its own: Ctrl-C and
    the hang-up of its shell stop it; the session and its local image remain."""
    res = milk_bash(
        'mem.mk2Dim "@L:loc" 4 4 >/dev/null; @calc "loc = 5"\n'
        'loop="x = 0; for i = 1:100000 { for j = 1:100000 { x = i + j } }"\n'
        '( @pipe "$loop" 2>loop.err; echo "loop_rc=$?" ) & pid=$!\n'
        "sleep 1; kill -INT $pid; wait $pid\n"
        '@calc -v m "imean(loc)"; echo "after_int=$m"\n'
        '( @pipe "$loop" 2>/dev/null ) & pid=$!\n'
        "sleep 1; kill -KILL $pid; wait $pid 2>/dev/null\n"
        '@calc -v m "imean(loc)"; echo "after_kill=$m"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "loop_rc") == "130", out
    assert "interrupted" in (tmp_path / "loop.err").read_text(), out
    assert value(res.stdout, "after_int") == "5", out
    assert value(res.stdout, "after_kill") == "5", out


def test_a_call_from_another_shm_dir_leaves_the_session_log(tmp_path):
    """A command sent with another MILK_SHM_DIR runs, and the session's log, which is how other
    shells find the session, stays one file in the directory the session started in."""
    res = milk_bash(
        '@calc "x = 1"; A=$MILK_SHM_DIR; B=$(mktemp -d "$PWD/shmB.XXXX")\n'
        "before=$(ls -i $A | grep milk-srv | sort | tr '\\n' ' ')\n"
        'MILK_SHM_DIR=$B mem.mk2Dim "@S:other" 4 4 >/dev/null; echo "rc=$?"\n'
        "after=$(ls -i $A | grep milk-srv | sort | tr '\\n' ' ')\n"
        '[ "$before" = "$after" ] && echo "home=same"; echo "stray=$(ls $B | grep -c milk-srv)"\n'
        '@log -n 1 | grep -q "mem.mk2Dim" && echo "logged=1"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "rc") == "0" and value(res.stdout, "home") == "same", out
    assert value(res.stdout, "stray") == "0", out
    assert value(res.stdout, "logged") == "1", out


def test_a_crash_in_calc_keeps_the_session(tmp_path):
    (tmp_path / "crash.c").write_text(r"""
#include "milk_calc_ext.h"
#define CR_PARAMS(X) \
    X(".k", k, FPTYPE_FLOAT64, 1.0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, "unused")
MILK_CALC_PARAMS(crash_now, CR_PARAMS);
static int crash_img(IMAGE *out, const IMAGE *in, const struct crash_now_par *P)
{
    (void) out;
    (void) in;
    (void) P;
    volatile int *p = 0;
    *p = 1;
    return 0;
}
MILK_CALC_EXT_EXPORT(MILK_CALC_REG_IMG("crash_now", crash_img, crash_now));
""")
    res = milk_bash(
        'mem.mk2Dim "@L:loc" 4 4 >/dev/null; @calc "loc = 5"\n'
        'mem.mk2Dim im 8 8 >/dev/null; @calc "im = 3"\n'
        '@calc --lib crash.c "o = crash_now(im)" 2>crash.err; echo "crash_rc=$?"\n'
        '@calc -v m "imean(loc)"; echo "loc=$m"',  # a local image: this session's server lives
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "crash_rc") == "1", out
    assert "CRASH INTERCEPTED" in (tmp_path / "crash.err").read_text(), out
    assert value(res.stdout, "loc") == "5", out


def test_a_shell_keeps_one_connection_between_requests(tmp_path):
    """The shell's connection stays the same socket; a subshell has its own; a script's own fds,
    numbered or `{var}`, work beside it (bash protects a close-on-exec fd from `exec N>`, so the
    connection sits high); after the server ends, the next request starts it again."""
    res = milk_bash(
        'mem.mk2Dim "@S:k6" 4 4 >/dev/null; @calc "k6 = 1"\n'
        "sock() { for f in /proc/$1/fd/*; do [[ $(readlink $f) == socket:* ]] && ((${f##*/} >= 64)) && echo ${f##*/}:$(readlink $f); done | head -1; }\n"
        'a=$(sock $$); @calc "k6 = k6 + 1"; mem.listim >/dev/null; @test "imean(k6) > 0"; b=$(sock $$)\n'
        'echo "same=$([ -n "$a" ] && [ "$a" = "$b" ] && echo 1)"\n'
        '( @calc "k6 = k6 + 1"; c=$(sock $BASHPID); echo "sub=$([ -n "$c" ] && [ "$c" != "$a" ] && echo 1)" )\n'
        'exec 3>three.txt; @calc "k6 = k6 + 1"; echo x >&3; exec 3>&-\n'
        'exec {v}>var.txt; @calc "k6 = k6 + 1"; echo y >&$v; exec {v}>&-\n'
        "echo \"fds=$(cat three.txt var.txt | tr -d '\\n') val=$(@calc 'imean(k6)')\"\n"
        "milkshutdown >/dev/null 2>&1\n"
        '@calc "k6 = k6 + 1"; echo "restart=$(@calc \'imean(k6)\')"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "same") == "1", out
    assert value(res.stdout, "sub") == "1", out
    assert value(res.stdout, "fds") == "xy val=5", out
    assert value(res.stdout, "restart") == "6", out


def test_captured_command_output_is_complete(tmp_path):
    res = milk_bash(
        'for i in $(seq 1 300); do mem.mk2Dim "cap6_$i" 2 2 >/dev/null; done\n'
        'echo "pipe=$(mem.listim | grep -c cap6_)"\n'
        'mem.listim > out.txt; echo "file=$(grep -c cap6_ out.txt)"\n'
        'echo "err=$(mem.mk2Dim 2>&1 >/dev/null | grep -ci usage)"\n'
        '@log -n 3 | grep -q cap6_ && echo "logged=1"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert [value(res.stdout, k) for k in ("pipe", "file", "logged")] == [
        "300",
        "300",
        "1",
    ], out
    assert value(res.stdout, "err") not in (None, "0"), out
