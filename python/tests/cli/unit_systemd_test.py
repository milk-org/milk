"""
Phase 4 of the unit-lifecycle plan (U17): units as systemd services. Item 21 needs no systemd: a
foreground run tells $NOTIFY_SOCKET what it tells a milk starter. Items 22-24 run test units in the
user manager, and are skipped where `systemctl --user` does not answer: each script writes a
template like share/milk/systemd/milk-unit@.service in $XDG_RUNTIME_DIR/systemd/user, and the
fixture stops its units and removes it.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import uuid
from pathlib import Path

import pytest

from milk.cliwrap import HAVE_CLI
from tests.cli.conftest import milk_bash, value
from tests.cli.unit_lifecycle_test import (
    FUNCS,
    ltunit,
    no_worker_left,
)  # noqa: F401 (fixtures)

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.timeout(120),
]


def _user_manager() -> bool:
    if shutil.which("systemctl") is None or not os.environ.get("XDG_RUNTIME_DIR"):
        return False
    res = subprocess.run(
        ["systemctl", "--user", "show", "-p", "Version"],
        capture_output=True,
        timeout=10,
    )
    return res.returncode == 0


needs_user_manager = pytest.mark.skipif(
    not _user_manager(), reason="no systemd user manager answers here"
)

# mkunit LINE...: the template $UP@.service for this script's units, as milk-unit@.service, with LINE
# added to [Service]; sd: systemctl --user; prop NAME KEY: a property of $UP@NAME.service; active,
# inactive, rated (its status line has the loop rate), restarted NAME; until_ok CMD...: retry CMD
# for 10 s
SD_FUNCS = r"""
UD="$XDG_RUNTIME_DIR/systemd/user"
mkunit() {
    mkdir -p "$UD"
    { echo "[Service]"; echo "Type=notify"
      echo "ExecStart=$MILK_INSTALLDIR/bin/milk-fps-run %i"
      echo "EnvironmentFile=$MILK_SHM_DIR/%i.fps.d/env"
      echo "StandardOutput=append:$MILK_SHM_DIR/%i.fps.d/run.out"
      echo "StandardError=inherit"; echo "TimeoutStopSec=5"
      printf '%s\n' "$@"; } > "$UD/$UP@.service"
    systemctl --user daemon-reload
}
sd() { systemctl --user "$@"; }
prop() { systemctl --user show -p "$2" --value "$UP@$1.service"; }
active() { systemctl --user is-active -q "$UP@$1.service"; }
inactive() { ! active "$1"; }
rated() { prop "$1" StatusText | grep -q " Hz"; }
restarted() { [ "$(prop "$1" NRestarts)" -ge 1 ]; }
until_ok() { for _ in $(seq 100); do "$@" && return 0; sleep 0.1; done; return 1; }
"""


@pytest.fixture
def unit_prefix():
    """UP for SD_FUNCS: a template name of this test's own; its units stop and its file goes after."""
    prefix = f"milk-ut-{uuid.uuid4().hex[:8]}"
    yield prefix
    run = ["systemctl", "--user"]
    subprocess.run([*run, "stop", f"{prefix}@*"], capture_output=True, timeout=30)
    subprocess.run(
        [*run, "reset-failed", f"{prefix}@*"], capture_output=True, timeout=30
    )
    path = (
        Path(os.environ.get("XDG_RUNTIME_DIR", "/nonexistent"))
        / "systemd/user"
        / f"{prefix}@.service"
    )
    if path.exists():
        path.unlink()
        subprocess.run([*run, "daemon-reload"], capture_output=True, timeout=30)


def out_of(res: subprocess.CompletedProcess[str], tmp_path: Path) -> str:
    logs = "".join(
        f"\n--- {p.name}\n{p.read_text()}" for p in sorted(tmp_path.glob("*.log"))
    )
    return res.stdout + res.stderr + logs


def test_notify_socket_messages(tmp_path):
    """Item 21: a foreground run tells $NOTIFY_SOCKET what it tells a starter (STATUS=, READY=1), pings
    the watchdog while its loop idles, and sends STOPPING=1 as it exits; without systemd it records
    no service."""
    sock_path = tmp_path / "notify.sock"
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    sock.bind(str(sock_path))
    try:
        res = milk_bash(
            FUNCS + 'mem.mk2Dim "@S:cam" 8 8 >/dev/null; @calc "cam = 2"\n'
            "calc_unit n1 o1\n"
            'NOTIFY_SOCKET="$SOCK" WATCHDOG_USEC=2000000 milk-fpsexec-milk-calc -n n1 run '
            ">run.log 2>&1 &\n"
            'pid=$!; sleep 4.5; milk-fps-runstop n1 >/dev/null; wait $pid; echo "exit=$?"\n'
            'echo "starter=$(info n1 starter)"\n'
            'echo "marker=$([ -e "$U/n1.fps.d/systemd" ] && echo 1 || echo 0)"',
            tmp_path,
            extra_env={"SOCK": str(sock_path)},
            timeout=60,
        )
        sock.setblocking(False)
        msgs = []
        while True:
            try:
                msgs.append(sock.recv(4096).decode())
            except BlockingIOError:
                break
    finally:
        sock.close()
    out = out_of(res, tmp_path) + f"\nmessages: {msgs}"
    assert value(res.stdout, "exit") == "0", out
    assert msgs and msgs[0] == "STATUS=connected\n", out
    assert any(m.startswith("READY=1\n") for m in msgs), out
    assert sum("WATCHDOG=1" in m for m in msgs) >= 2, out
    assert any(
        m.startswith("STATUS=running, ") and m.endswith(" Hz\n") for m in msgs
    ), out
    assert msgs[-1] == "STOPPING=1\nSTATUS=exited with status 0\n", out
    assert value(res.stdout, "starter") == "foreground", out
    assert value(res.stdout, "marker") == "0", out


@needs_user_manager
def test_service_starts_reports_and_fails_on_a_signal(tmp_path, unit_prefix):
    """Item 22: `systemctl --user start` returns once the loop runs; the status line has the loop rate;
    the snapshot reaches the unit through EnvironmentFile= unchanged; the output goes to run.out;
    .procinfo.cset is skipped; after kill -9, systemd's result names the signal."""
    res = milk_bash(
        FUNCS + SD_FUNCS + 'mem.mk2Dim "@S:cam" 8 8 >/dev/null; @calc "cam = 2"\n'
        "export ODD='a \"quoted\" $HOME `x` \\back'\n"
        "calc_unit s1 o1\n"
        "milk-fps-set s1.procinfo.cset milktestcset >/dev/null\n"
        "mkunit\n"
        'sd start "$UP@s1.service"; echo "start=$?"\n'
        'echo "state=$(state s1)"\n'
        "until_ok rated s1\n"
        'echo "status=$(prop s1 StatusText)"\n'
        'pid=$(prop s1 MainPID); echo "same=$([ "$pid" = "$(info s1 pid)" ] && echo 1 || echo 0)"\n'
        "echo \"odd=$(tr '\\0' '\\n' < /proc/$pid/environ | grep '^ODD=' | cut -d= -f2-)\"\n"
        'echo "starter=$(info s1 starter)"\n'
        'echo "marker=$(cat "$U/s1.fps.d/systemd")"\n'
        'grep -c "runstart by systemd:$UP@s1.service" "$U/s1.fps.d/run.out" | sed "s/^/marked=/"\n'
        'grep -c "cset milktestcset ignored" "$U/s1.fps.d/run.out" | sed "s/^/cset=/"\n'
        'kill -9 "$pid"; until_ok inactive s1\n'
        'echo "result=$(prop s1 Result) code=$(prop s1 ExecMainCode) sig=$(prop s1 ExecMainStatus)"\n'
        'echo "after=$(state s1)"',
        tmp_path,
        extra_env={"UP": unit_prefix},
        timeout=90,
    )
    out = out_of(res, tmp_path)
    unit = f"{unit_prefix}@s1.service"
    assert value(res.stdout, "start") == "0", out
    assert value(res.stdout, "state") == "running", out
    assert value(res.stdout, "status").startswith("running, "), out
    assert value(res.stdout, "same") == "1", out
    assert value(res.stdout, "odd") == 'a "quoted" $HOME `x` \\back', out
    assert value(res.stdout, "starter") == f"systemd:{unit}", out
    assert value(res.stdout, "marker") == f"run={unit} user", out
    assert value(res.stdout, "marked") == "1", out
    assert value(res.stdout, "cset") == "1", out
    assert (
        value(res.stdout, "result") == "signal code=2 sig=9"
    ), out  # CLD_KILLED by SIGKILL
    assert value(res.stdout, "after") == "failed", out


@needs_user_manager
def test_milk_tools_go_through_systemctl_until_released(tmp_path, unit_prefix):
    """Item 23: once systemd ran a unit, milk-fps-runstop and milk-fps-runstart go through systemctl
    (no restart after the stop); milk-fps-run refuses a unit milk runs; after --release, milk starts
    and stops it itself."""
    res = milk_bash(
        FUNCS + SD_FUNCS + 'mem.mk2Dim "@S:cam" 8 8 >/dev/null; @calc "cam = 2"\n'
        "calc_unit s2 o2\n"
        "mkunit Restart=on-failure RestartSec=0.2\n"
        'sd start "$UP@s2.service"\n'
        'milk-fps-runstop s2 >stop.log 2>&1; echo "stop=$?"\n'
        'sleep 1; echo "active=$(sd is-active "$UP@s2.service") restarts=$(prop s2 NRestarts)"\n'
        'milk-fps-runstart s2 >start.log 2>&1; echo "start=$?"\n'
        'echo "again=$(sd is-active "$UP@s2.service") state=$(state s2)"\n'
        'milk-fps-info s2 | grep -c "run by $UP@s2.service (user)" | sed "s/^/info=/"\n'
        'milk-fps-run --release s2 >release.log 2>&1; echo "release=$?"\n'
        'sd stop "$UP@s2.service"\n'
        'milk-fps-runstart s2 >/dev/null 2>&1; echo "milk=$? starter=$(info s2 starter)"\n'
        'sd start "$UP@s2.service" 2>/dev/null; echo "refused=$?"\n'
        'milk-fps-run s2 >run.log 2>&1; echo "direct=$?"\n'
        'milk-fps-runstop s2 >/dev/null; echo "end=$(state s2)"',
        tmp_path,
        extra_env={"UP": unit_prefix},
        timeout=90,
    )
    out = out_of(res, tmp_path)
    unit = f"{unit_prefix}@s2.service"
    assert value(res.stdout, "stop") == "0", out
    assert f"s2 run: stopped by {unit}" in (tmp_path / "stop.log").read_text(), out
    assert value(res.stdout, "active") == "inactive restarts=0", out
    assert value(res.stdout, "start") == "0", out
    assert f"({unit})" in (tmp_path / "start.log").read_text(), out
    assert value(res.stdout, "again") == "active state=running", out
    assert value(res.stdout, "info") == "1", out
    assert value(res.stdout, "release") == "0", out
    assert value(res.stdout, "milk").startswith("0 starter=milk-fps-runstart pid "), out
    assert value(res.stdout, "refused") != "0", out
    assert value(res.stdout, "direct") == "1", out
    assert "already running" in (tmp_path / "run.log").read_text(), out
    assert value(res.stdout, "end") == "exited", out


@needs_user_manager
def test_watchdog_restarts_a_stuck_loop_not_an_idle_one(tmp_path, unit_prefix, ltunit):
    """Item 24: with WatchdogSec=, an idle loop (its waits time out) keeps running; a loop stuck in its
    computation is killed and restarted."""
    res = milk_bash(
        FUNCS + SD_FUNCS + 'mem.mk2Dim "@S:lt_in" 4 4 >/dev/null\n'
        '"$LT" -n w1 fpsinit >/dev/null\n'
        "mkunit Restart=on-failure RestartSec=0.5 WatchdogSec=4 LimitCORE=0\n"
        'sd start "$UP@w1.service"; echo "start=$?"\n'
        'sleep 9; echo "idle=$(sd is-active "$UP@w1.service") restarts=$(prop w1 NRestarts)"\n'
        "milk-fps-set w1.hold 30 >/dev/null; mem.imsetsempost lt_in -1 >/dev/null\n"
        "until_ok restarted w1\n"
        'echo "stuck=$(prop w1 NRestarts)"\n'
        "until_ok active w1\n"
        'echo "back=$(sd is-active "$UP@w1.service")"',
        tmp_path,
        extra_env={"UP": unit_prefix, "LT": ltunit},
        timeout=90,
    )
    out = out_of(res, tmp_path)
    assert value(res.stdout, "start") == "0", out
    assert value(res.stdout, "idle") == "active restarts=0", out
    assert value(res.stdout, "stuck") == "1", out
    assert value(res.stdout, "back") == "active", out
