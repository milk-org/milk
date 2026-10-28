"""
Phase 0 of the unit-lifecycle plan (processinfo runtime fixes, U14), on a real milk.calc unit:
SIGTERM ends an idle loop, a fault kills the unit as CRASHED instead of being swallowed, and a stop
request that a frame wakes up ends the loop without computing that frame.
"""

from __future__ import annotations

import pytest

from milk.cliwrap import HAVE_CLI
from tests.cli.conftest import milk_bash, value

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.timeout(120),
]

# Start unit $1 on input `cam` writing $2, in the background; report whether $pid is gone in $3 s.
HELPERS = r"""
until_file() { for _ in $(seq 100); do [ -e "$MILK_SHM_DIR/$1.im.shm" ] && return 0; sleep 0.05; done; return 1; }
start_unit() { milk-fpsexec-milk-calc -n "$1" -t cam -e "$2 = cam * 2" fpsinit >/dev/null
               milk-fpsexec-milk-calc -n "$1" run >run.log 2>&1 & pid=$!; until_file "$2"; sleep 0.3; }
gone_within() { for _ in $(seq $(( $1 * 20 ))); do kill -0 $pid 2>/dev/null || { echo "alive=0"; return; }; sleep 0.05; done
                echo "alive=1"; kill -KILL $pid 2>/dev/null; }
ulimit -c 0
mem.mk2Dim "@S:cam" 8 8 >/dev/null; @calc "cam = 2"
"""


def test_sigterm_ends_an_idle_loop(tmp_path):
    """SIGTERM ends a unit waiting on its trigger stream, through the loop's own exit (status 0)."""
    res = milk_bash(
        HELPERS + "start_unit t8 out8\n"
        'kill -TERM $pid; gone_within 4; wait $pid; echo "code=$?"',
        tmp_path,
    )
    log = (tmp_path / "run.log").read_text()
    assert value(res.stdout, "alive") == "0", res.stdout + res.stderr + log
    assert value(res.stdout, "code") == "0", res.stdout + res.stderr + log


def test_fault_kills_the_unit_as_crashed(tmp_path):
    """SIGSEGV ends the unit with that signal (not a flag the idle loop never reads) and leaves
    loopstat CRASHED (6) in its processinfo."""
    res = milk_bash(
        HELPERS + "start_unit t9 out9\n"
        'kill -SEGV $pid; gone_within 3; wait $pid; echo "code=$?"\n'
        'echo "loopstat=$(@get p.t9.loopstat)"',
        tmp_path,
    )
    log = (tmp_path / "run.log").read_text()
    assert value(res.stdout, "alive") == "0", res.stdout + res.stderr + log
    assert value(res.stdout, "code") == "139", res.stdout + res.stderr + log
    assert value(res.stdout, "loopstat") == "6", res.stdout + res.stderr + log


def test_stop_request_woken_by_a_frame_does_not_compute(tmp_path):
    """After `ctrl=stop`, the frame that wakes the loop is not computed: out10 gets no new post."""
    res = milk_bash(
        HELPERS + "start_unit t10 out10\n"
        "c0=$(@get s.out10.cnt0)\n"
        '@set p.t10.ctrl=stop; @calc "cam = 7"\n'
        "gone_within 4; wait $pid\n"
        'echo "posts=$(( $(@get s.out10.cnt0) - c0 ))"',
        tmp_path,
    )
    log = (tmp_path / "run.log").read_text()
    assert value(res.stdout, "alive") == "0", res.stdout + res.stderr + log
    assert value(res.stdout, "posts") == "0", res.stdout + res.stderr + log
