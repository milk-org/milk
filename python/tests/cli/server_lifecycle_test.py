"""
Phase 5 of the unit-lifecycle plan (U18): milk-cli servers start through milk_detach, as units do. A
server reports READY=1 or why it failed, keeps its output in a file next to its log, stops on its
shutdown command or else by signals, and tells $NOTIFY_SOCKET when a service manager runs it.
`@session new --tmux` is gone. The plan's §7 item numbers are in each test's docstring.
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
    """Item 25: a server that cannot listen (its socket name is taken) makes @session new fail with the
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
    """Item 25: a server's crash message stays in its output file; a server that ignores its shutdown
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
    """Item 26: a private server ends with the shell that owns it, a named one survives it; neither keeps
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
