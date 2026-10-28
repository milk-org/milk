"""
Phase 1 of the unit-lifecycle plan (U3-U11): `runstart` starts a detached worker and returns once its
loop is active, `runstop` asks it to exit then signals it, and each unit records its processes and
output in <shm>/<name>.fps.d/. The plan's §7 item numbers are in each test's docstring. Phase 2's
shell front-end (`@set f.X.run|conf`, the state properties, `@waitfor f.X running|stopped`) is at the
end.

Units are milk.calc units, or `ltunit`, a C unit built here with milk-cc that sleeps and prints on
demand. Workers live in their own session: a fixture kills those a test leaves behind.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from milk.cliwrap import HAVE_CLI
from tests.cli.conftest import _install_dir, _isolated_shm_env, _run, milk_bash, value

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.timeout(120),
]

LTUNIT_C = r"""
#include <time.h>

#include "milk_runnable.h"

static char    lt_in[FUNCTION_PARAMETER_STRMAXLEN] = "lt_in";
static double  lt_delay                            = 0.0;
static double  lt_hold                             = 0.0;
static int64_t lt_lines                            = 0;

#define LT_PARAMS(X)                                                                          \
    X(".in", lt_in, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_TRIGGER_STREAM, "Input stream")     \
    X(".delay", &lt_delay, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT, "Sleep before the loop") \
    X(".hold", &lt_hold, FPTYPE_FLOAT64, 1, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,          \
      "Sleep in each iteration")                                                             \
    X(".lines", &lt_lines, FPTYPE_INT64, 1, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,          \
      "Lines printed by each iteration")

MILK_PARAMS_DEFINE(ltunit, LT_PARAMS);

static void lt_sleep(double s)
{
    struct timespec t = { (time_t) s, (long) ((s - (double) (time_t) s) * 1e9) };
    nanosleep(&t, NULL);
}

static errno_t compute_ltunit(const MILK_RUNNABLE *self)
{
    long n = 0;
    if (lt_delay > 0)
    {
        lt_sleep(lt_delay);
    }
    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    {
        for (int64_t i = 0; i < lt_lines; i++)
        {
            printf("tick %ld line %ld\n", n, (long) i);
        }
        n++;
        if (lt_hold > 0)
        {
            lt_sleep(lt_hold);
        }
    }
    MILK_PROCINFO_END(self)
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_ltunit = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = {
        .fps_name    = "ltunit",
        .cmdkey      = "ltunit",
        .description = "unit lifecycle test unit",
    },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(ltunit),
    .compute = compute_ltunit,
};

EXPORT_FPSEXEC(milk, "ltunit", runnable_ltunit);
"""

# U: the unit directories; info NAME KEY [ROLE]: last KEY of NAME's <role>.info; state NAME: the run
# state milk-fps-list shows; gone PID: wait up to 10 s for PID to exit. (pgrep patterns below are
# written "l2[:]run" so that they do not match the bash that runs the script.)
FUNCS = r"""
ulimit -c 0
U="$MILK_SHM_DIR"
info() { grep "^$2=" "$U/$1.fps.d/${3:-run}.info" 2>/dev/null | tail -1 | cut -d= -f2-; }
state() { milk-fps-list "^$1\$" | sed 's/\x1b\[[0-9;]*m//g' |
          awk -v n="$1" '$1 == n { i = 5; if ($i ~ /^[0-9]+$/) i++; print $i }'; }
gone() { for _ in $(seq 200); do kill -0 "$1" 2>/dev/null || return 0; sleep 0.05; done; return 1; }
alive() { kill -0 "$1" 2>/dev/null && echo 1 || echo 0; }
calc_unit() { milk-fpsexec-milk-calc -n "$1" -t cam -e "$2 = cam * 2" fpsinit >/dev/null; }
"""
HELPERS = FUNCS + r"""
mem.mk2Dim "@S:cam" 8 8 >/dev/null; @calc "cam = 2"
mem.mk2Dim "@S:lt_in" 4 4 >/dev/null
"""


@pytest.fixture(scope="module")
def ltunit(tmp_path_factory) -> str:
    """Path of milk-fpsexec-ltunit, built once by milk-cc."""
    d = tmp_path_factory.mktemp("ltunit")
    (d / "ltunit.c").write_text(LTUNIT_C)
    env, shm_dir = _isolated_shm_env()
    try:
        res = _run(
            [str(_install_dir() / "bin" / "milk-cc"), "ltunit.c"], env, 120, cwd=str(d)
        )
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)
    exe = d / "milk-fpsexec-ltunit"
    if res.returncode != 0 or not exe.exists():
        pytest.skip(f"milk-cc could not build the test unit: {res.stderr[-400:]}")
    return str(exe)


@pytest.fixture(autouse=True)
def no_worker_left():
    """Kill the workers a test left: they run in their own session, under a milk_bash shm dir."""
    yield
    marker = (
        f"MILK_SHM_DIR={os.environ.get('MILK_SHM_DIR', '/tmp')}/milk_bash_".encode()
    )
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            env = Path(f"/proc/{d}/environ").read_bytes().split(b"\0")
        except OSError:
            continue
        if b"MILK_UNIT_WORKER=1" in env and any(e.startswith(marker) for e in env):
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(d), signal.SIGKILL)


def lt_bash(
    script: str, tmp_path: Path, ltunit: str, **kw
) -> subprocess.CompletedProcess[str]:
    return milk_bash(HELPERS + script, tmp_path, extra_env={"LT": ltunit}, **kw)


def out_of(res: subprocess.CompletedProcess[str], tmp_path: Path) -> str:
    logs = "".join(
        f"\n--- {p.name}\n{p.read_text()}" for p in sorted(tmp_path.glob("*.log"))
    )
    return res.stdout + res.stderr + logs


def test_runstart_returns_once_the_loop_runs(tmp_path):
    """Item 1: runstart returns with the loop active and the process alive; runstop returns once the
    process is gone; run.info has the exit and the stop lines."""
    res = lt_bash(
        "calc_unit l1 o1\n"
        'milk-fpsexec-milk-calc -n l1 runstart >start.log 2>&1; echo "start=$?"\n'
        'pid=$(info l1 pid); echo "alive=$(alive $pid)"; echo "state=$(state l1)"\n'
        '@calc "cam = 5"; sleep 0.3; echo "out=$(@get s.o1.mean)"\n'
        'milk-fpsexec-milk-calc -n l1 runstop >stop.log 2>&1; echo "stop=$?"\n'
        'echo "after=$(alive $pid)"; echo "exit=$(info l1 exit)"; echo "how=$(info l1 stop)"\n'
        'echo "final=$(state l1)"',
        tmp_path,
        "",
    )
    out = out_of(res, tmp_path)
    assert value(res.stdout, "start") == "0", out
    assert value(res.stdout, "alive") == "1", out
    assert value(res.stdout, "state") == "running", out
    assert value(res.stdout, "out") == "10", out
    assert value(res.stdout, "stop") == "0", out
    assert value(res.stdout, "after") == "0", out
    assert value(res.stdout, "exit") == "0", out
    assert value(res.stdout, "how") == "request", out
    assert value(res.stdout, "final") == "exited", out
    assert "ready after" in (tmp_path / "start.log").read_text(), out


def test_parameter_check_failure_is_reported(tmp_path, ltunit):
    """Item 2: a parameter check failure exits 1 with the messages, leaves no worker, and run.out
    says why."""
    res = lt_bash(
        "$LT -n l2 fpsinit .in=nosuch >/dev/null\n"
        '$LT -n l2 runstart >start.log 2>&1; echo "start=$?"\n'
        'sleep 0.2; echo "workers=$(pgrep -fc "l2[:]runstart")"; echo "state=$(state l2)"\n'
        'grep -c "cannot load stream nosuch" "$U/l2.fps.d/run.out" | sed "s/^/explained=/"',
        tmp_path,
        ltunit,
    )
    out = out_of(res, tmp_path)
    start = (tmp_path / "start.log").read_text()
    assert value(res.stdout, "start") == "1", out
    assert (
        "parameter check failed" in start and "cannot load stream nosuch" in start
    ), out
    assert value(res.stdout, "workers") == "0", out
    assert value(res.stdout, "state") == "failed", out
    assert value(res.stdout, "explained") != "0", out


def test_setup_failure_is_never_ready(tmp_path):
    """Item 3: a compute function that fails in its own setup (missing trigger stream) makes runstart
    fail; it never reports ready."""
    res = lt_bash(
        'milk-fpsexec-milk-calc -n l3 -t nosuch -e "o3 = nosuch * 2" fpsinit >/dev/null\n'
        'milk-fpsexec-milk-calc -n l3 runstart >start.log 2>&1; echo "start=$?"',
        tmp_path,
        "",
    )
    out = out_of(res, tmp_path)
    start = (tmp_path / "start.log").read_text()
    assert value(res.stdout, "start") == "1", out
    assert "exited before its loop started" in start and "ready after" not in start, out


def test_slow_start_times_out_then_runs(tmp_path, ltunit):
    """Item 4: a loop not active within --timeout gives 124 and state starting, then running."""
    res = lt_bash(
        "$LT -n l4 fpsinit .delay=1.5 >/dev/null\n"
        '$LT -n l4 --timeout 0.3 runstart >start.log 2>&1; echo "start=$?"\n'
        'echo "now=$(state l4)"; sleep 2; echo "later=$(state l4)"\n'
        "milk-fps-runstop l4 >/dev/null",
        tmp_path,
        ltunit,
    )
    out = out_of(res, tmp_path)
    assert value(res.stdout, "start") == "124", out
    assert value(res.stdout, "now") == "starting", out
    assert value(res.stdout, "later") == "running", out


def test_missing_executable_is_reported(tmp_path, ltunit):
    """Item 5: a start whose execfullpath is gone reports ENOENT."""
    res = lt_bash(
        'cp "$LT" ./milk-fpsexec-ltunit\n'
        "./milk-fpsexec-ltunit -n l5 fpsinit >/dev/null\n"
        "rm ./milk-fpsexec-ltunit\n"
        'milk-fps-runstart l5 >start.log 2>&1; echo "start=$?"',
        tmp_path,
        ltunit,
    )
    out = out_of(res, tmp_path)
    assert value(res.stdout, "start") == "1", out
    assert (
        "execve: No such file or directory" in (tmp_path / "start.log").read_text()
    ), out


def test_idle_loop_stops_on_request(tmp_path):
    """Item 6: an idle semaphore loop stops through the request in under 1 s, without a signal, and
    the wake-up does not compute."""
    res = lt_bash(
        "calc_unit l6 o6\n"
        "milk-fpsexec-milk-calc -n l6 runstart >/dev/null\n"
        "c0=$(@get s.o6.cnt0)\n"
        'milk-fps-runstop l6 >stop.log 2>&1; echo "stop=$?"\n'
        'echo "posts=$(( $(@get s.o6.cnt0) - c0 ))"; echo "how=$(info l6 stop)"\n'
        'grep -c "stop: request" "$U/l6.fps.d/run.out" | sed "s/^/marker=/"',
        tmp_path,
        "",
    )
    out = out_of(res, tmp_path)
    stop = (tmp_path / "stop.log").read_text()
    assert value(res.stdout, "stop") == "0", out
    assert value(res.stdout, "how") == "request", out
    assert value(res.stdout, "posts") == "0", out
    assert value(res.stdout, "marker") == "1", out
    seconds = float(stop.split("request, ")[1].split(" s)")[0])
    assert seconds < 1.0, out


def test_unit_without_processinfo_is_stopped_by_sigint(tmp_path, ltunit):
    """Item 7: a unit without processinfo has no request to answer: stop goes straight to SIGINT."""
    res = lt_bash(
        "$LT -n l7 --no-procinfo fpsinit .hold=30 >/dev/null\n"
        '$LT -n l7 runstart >/dev/null; echo "start=$?"\n'
        "mem.imsetsempost lt_in -1 >/dev/null; sleep 0.3\n"
        'milk-fps-runstop l7 >stop.log 2>&1; echo "stop=$?"; echo "how=$(info l7 stop)"\n'
        'echo "state=$(state l7)"',
        tmp_path,
        ltunit,
    )
    out = out_of(res, tmp_path)
    assert value(res.stdout, "start") == "0", out
    assert value(res.stdout, "stop") == "0", out
    assert value(res.stdout, "how") == "SIGINT", out
    assert value(res.stdout, "state") == "exited", out


def _bash_in(
    env: dict[str, str], script: str, cwd: Path
) -> subprocess.CompletedProcess[str]:
    """Like milk_bash, in the shm directory of @env, which outlives the call."""
    full = f'source "{_install_dir() / "bin" / "milk.bash"}"\n{FUNCS}{script}\n'
    return _run(["/bin/bash", "-c", full], env, 60, cwd=str(cwd))


def test_unit_outlives_its_starter_and_leaves_no_zombie(tmp_path):
    """Item 11: the worker survives its starter's whole session; 20 start/stop cycles from this
    process leave it no child."""
    env, shm_dir = _isolated_shm_env()
    try:
        res = _bash_in(
            env,
            'mem.mk2Dim "@S:cam" 8 8 >/dev/null\ncalc_unit l11 o11\n'
            'milk-fps-runstart l11 >/dev/null; echo "start=$?"',
            tmp_path,
        )
        assert value(res.stdout, "start") == "0", res.stdout + res.stderr
        res = _bash_in(env, 'pid=$(info l11 pid); echo "alive=$(alive $pid)"', tmp_path)
        assert value(res.stdout, "alive") == "1", res.stdout + res.stderr

        bindir = _install_dir() / "bin"
        for _ in range(20):
            for tool in ("milk-fps-runstop", "milk-fps-runstart"):
                r = subprocess.run(
                    [str(bindir / tool), "l11"], env=env, capture_output=True, text=True
                )
                assert r.returncode == 0, r.stdout + r.stderr
        with contextlib.suppress(ChildProcessError):  # no child at all
            assert os.waitpid(-1, os.WNOHANG)[0] == 0  # or none that exited
        r = subprocess.run(
            [str(bindir / "milk-fps-runstop"), "l11"],
            env=env,
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, r.stdout + r.stderr
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)


def test_concurrent_starts_make_one_worker(tmp_path):
    """Item 12: three processes, and two threads of one process, start one unit: one worker."""
    lib = _install_dir() / "lib" / "libmilkcomputeunit.so"
    threads = (
        "import ctypes, sys, threading\n"
        "lib = ctypes.CDLL(sys.argv[1])\n"
        "lib.fps_unit_start.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_double, ctypes.c_int]\n"
        "rc = []\n"
        "go = lambda: rc.append(lib.fps_unit_start(sys.argv[2].encode(), 0, 0.0, 1))\n"
        "ts = [threading.Thread(target=go) for _ in range(2)]\n"
        "[t.start() for t in ts]\n"
        "[t.join() for t in ts]\n"
        "print('rcs=' + ','.join(map(str, sorted(rc))))\n"
    )
    res = milk_bash(
        HELPERS + "calc_unit l12 o12\n"
        "for i in 1 2 3; do milk-fps-runstart l12 >start$i.log 2>&1 & done; wait\n"
        'echo "workers=$(pgrep -fc "l12[:]runstart --fg")"\n'
        "calc_unit l12b o12b\n"
        f'"{sys.executable}" -c "$PYCODE" "{lib}" l12b\n'
        'echo "threads=$(pgrep -fc "l12b[:]runstart --fg")"\n'
        "milk-fps-runstop l12 >/dev/null; milk-fps-runstop l12b >/dev/null",
        tmp_path,
        extra_env={"PYCODE": threads},
    )
    out = out_of(res, tmp_path)
    assert value(res.stdout, "workers") == "1", out
    assert value(res.stdout, "threads") == "1", out
    assert value(res.stdout, "rcs") == "0,0", out


def test_environment_snapshot_reaches_the_worker(tmp_path):
    """Item 13: a variable set when fpsinit ran reaches the worker though its starter lacks it, and
    .procinfo.NBthread sets the OpenMP team the worker actually runs."""
    res = lt_bash(
        "export LTVAR=captured; calc_unit l13 o13; unset LTVAR\n"
        "milk-fps-set l13.procinfo.NBthread 2 >/dev/null\n"
        "milk-fps-runstart l13 >/dev/null\n"
        'pid=$(info l13 pid); @calc "cam = 3"; sleep 0.5\n'
        "echo \"var=$(tr '\\0' '\\n' < /proc/$pid/environ | grep ^LTVAR= | cut -d= -f2)\"\n"
        'echo "threads=$(ls /proc/$pid/task | wc -l)"\n'
        "milk-fps-runstop l13 >/dev/null",
        tmp_path,
        "",
    )
    out = out_of(res, tmp_path)
    assert value(res.stdout, "var") == "captured", out
    assert value(res.stdout, "threads") == "2", out


def test_output_files_rotate_survive_a_crash_and_go_with_the_fps(tmp_path, ltunit):
    """Item 14: output past MILK_UNIT_OUT_MAX rotates; the last lines before a crash are kept;
    the next start rotates; milk-fps-rm removes the unit directory and the processinfo files.
    """
    res = lt_bash(
        "MILK_UNIT_OUT_MAX=2K $LT -n l14 fpsinit .lines=40 >/dev/null\n"
        "$LT -n l14 runstart >/dev/null\n"
        # the size is checked at most once a second: post until it rotates
        "for i in $(seq 8); do mem.imsetsempost lt_in -1 >/dev/null; sleep 0.6; "
        '[ -s "$U/l14.fps.d/run.out.1" ] && break; done\n'
        'echo "rotated=$([ -s "$U/l14.fps.d/run.out.1" ] && echo 1 || echo 0)"\n'
        "pid=$(info l14 pid); mem.imsetsempost lt_in -1 >/dev/null; sleep 0.3\n"
        "kill -SEGV $pid; gone $pid\n"
        'echo "last=$(tail -2 "$U/l14.fps.d/run.out" | head -1)"\n'
        'echo "end=$(tail -1 "$U/l14.fps.d/run.out")"\n'
        "$LT -n l14 runstart >/dev/null\n"
        'echo "kept=$(tail -1 "$U/l14.fps.d/run.out.1")"\n'
        'milk-fps-rm -f l14 >/dev/null; echo "rm=$?"\n'
        'echo "dir=$([ -e "$U/l14.fps.d" ] && echo 1 || echo 0)"\n'
        'echo "procs=$(ls "$MILK_PROC_DIR" | grep -c "^proc\\.l14\\.")"',
        tmp_path,
        ltunit,
    )
    out = out_of(res, tmp_path)
    assert value(res.stdout, "rotated") == "1", out
    assert re.fullmatch(r"tick \d+ line 39", value(res.stdout, "last")), out
    assert value(res.stdout, "end") == "=== killed by SIGSEGV", out
    assert value(res.stdout, "kept") == "=== killed by SIGSEGV", out
    assert value(res.stdout, "rm") == "0", out
    assert value(res.stdout, "dir") == "0", out
    assert value(res.stdout, "procs") == "0", out


def test_worker_inherits_no_descriptor(tmp_path):
    """Item 15: the worker holds none of its starter's descriptors and no ready pipe once ready."""
    res = lt_bash(
        "calc_unit l15 o15\n"
        'exec 7>"$PWD/held.txt"\n'
        "milk-fps-runstart l15 >/dev/null\n"
        "pid=$(info l15 pid)\n"
        'echo "held=$(ls -l /proc/$pid/fd | grep -c held.txt)"\n'
        'echo "pipes=$(ls -l /proc/$pid/fd | grep -c pipe:)"\n'
        'echo "stdin=$(readlink /proc/$pid/fd/0)"\n'
        'echo "stdout=$(basename "$(readlink /proc/$pid/fd/1)")"\n'
        "milk-fps-runstop l15 >/dev/null",
        tmp_path,
        "",
    )
    out = out_of(res, tmp_path)
    assert value(res.stdout, "held") == "0", out
    assert value(res.stdout, "pipes") == "0", out
    assert value(res.stdout, "stdin") == "/dev/null", out
    assert value(res.stdout, "stdout") == "run.out", out


def test_foreground_modes_and_no_tmux(tmp_path):
    """Item 16: run, --fg and MILK_UNIT_FG=1 stay in the foreground; a worker runs with --fg and
    starts no other; -tmux fails with a hint."""
    res = lt_bash(
        "calc_unit l16 o16\n"
        'timeout 1 milk-fpsexec-milk-calc -n l16 run >/dev/null 2>&1; echo "run=$?"\n'
        'timeout 1 milk-fpsexec-milk-calc -n l16 --fg runstart >/dev/null 2>&1; echo "fg=$?"\n'
        "MILK_UNIT_FG=1 timeout 1 milk-fpsexec-milk-calc -n l16 runstart >/dev/null 2>&1; "
        'echo "env=$?"\n'
        "milk-fpsexec-milk-calc -n l16 runstart >/dev/null\n"
        "pid=$(info l16 pid)\n"
        "echo \"argv=$(tr '\\0' ' ' < /proc/$pid/cmdline | cut -d' ' -f2-)\"\n"
        'echo "children=$(pgrep -c -P $pid)"\n'
        "milk-fps-runstop l16 >/dev/null\n"
        'milk-fpsexec-milk-calc -n l16 -tmux runstart 2>tmux.log; echo "tmux=$?"',
        tmp_path,
        "",
    )
    out = out_of(res, tmp_path)
    assert value(res.stdout, "run") == "124", out
    assert value(res.stdout, "fg") == "124", out
    assert value(res.stdout, "env") == "124", out
    assert value(res.stdout, "argv").strip() == "l16:runstart --fg", out
    assert value(res.stdout, "children") == "0", out
    assert value(res.stdout, "tmux") == "2", out
    assert "use --fg" in (tmp_path / "tmux.log").read_text(), out


def test_killed_worker_is_failed_and_its_bits_cleared(tmp_path):
    """Item 17: after kill -9 the state is failed with no exit record, and the stale run bits are
    cleared."""
    res = lt_bash(
        "calc_unit l17 o17\n"
        "milk-fps-runstart l17 >/dev/null\n"
        'echo "before=$(@get f.l17.status)"\n'
        "pid=$(info l17 pid); kill -9 $pid; gone $pid\n"
        'echo "state=$(state l17)"; echo "status=$(@get f.l17.status)"; echo "exit=[$(info l17 exit)]"\n'
        'echo "runpid=$(@get f.l17.runpid)"',
        tmp_path,
        "",
    )
    out = out_of(res, tmp_path)
    assert value(res.stdout, "before") == "running", out
    assert value(res.stdout, "state") == "failed", out
    assert value(res.stdout, "status") == "failed", out
    assert value(res.stdout, "exit") == "[]", out
    assert value(res.stdout, "runpid") == "0", out


def test_crash_leftover_processinfo_is_moved(tmp_path):
    """Item 27: after a crash, the next start moves the dead worker's processinfo file to
    run.procinfo.1; the proc directory keeps none of it."""
    res = lt_bash(
        "calc_unit l27 o27\n"
        "milk-fps-runstart l27 >/dev/null\n"
        "old=$(info l27 pid); kill -SEGV $old; gone $old\n"
        'echo "left=$(ls "$MILK_PROC_DIR" | grep -c "^proc\\.l27\\.0*$old\\.")"\n'
        "milk-fps-runstart l27 >/dev/null\n"
        'echo "after=$(ls "$MILK_PROC_DIR" | grep -c "^proc\\.l27\\.0*$old\\.")"\n'
        'echo "moved=$([ -s "$U/l27.fps.d/run.procinfo.1" ] && echo 1 || echo 0)"\n'
        "milk-fps-runstop l27 >/dev/null",
        tmp_path,
        "",
    )
    out = out_of(res, tmp_path)
    assert value(res.stdout, "left") == "1", out
    assert value(res.stdout, "after") == "0", out
    assert value(res.stdout, "moved") == "1", out


def test_stopper_with_another_proc_dir(tmp_path):
    """Item 28: a stopper whose MILK_PROC_DIR differs from the worker's still stops it through
    CTRLval (procdir from run.info)."""
    res = lt_bash(
        'mkdir -p "$U/pa" "$U/pb"; calc_unit l28 o28\n'
        'MILK_PROC_DIR="$U/pa" milk-fps-runstart l28 >/dev/null\n'
        'echo "procdir=$(basename "$(info l28 procdir)")"\n'
        'MILK_PROC_DIR="$U/pb" milk-fps-runstop l28 >/dev/null; echo "how=$(info l28 stop)"',
        tmp_path,
        "",
    )
    out = out_of(res, tmp_path)
    assert value(res.stdout, "procdir") == "pa", out
    assert value(res.stdout, "how") == "request", out


def test_fpsinit_adds_processinfo_by_default(tmp_path, ltunit):
    """Item 29: fpsinit adds the processinfo entries; --no-procinfo makes a one-shot unit."""
    res = lt_bash(
        "$LT -n l29 fpsinit >/dev/null\n"
        'echo "pinfo=$(@get f.l29.procinfo.loopcntMax)"\n'
        "$LT -n l29b --no-procinfo fpsinit .lines=1 >/dev/null\n"
        '@get f.l29b.procinfo.loopcntMax >/dev/null 2>&1; echo "nopinfo=$?"\n'
        "mem.imsetsempost lt_in -1 >/dev/null\n"
        'timeout 10 $LT -n l29b run >run.out 2>&1; echo "run=$?"\n'
        'echo "ticks=$(grep -c "^tick" run.out)"',
        tmp_path,
        ltunit,
    )
    out = out_of(res, tmp_path)
    assert value(res.stdout, "pinfo") == "-1", out
    assert value(res.stdout, "nopinfo") != "0", out
    assert value(res.stdout, "run") == "0", out
    assert value(res.stdout, "ticks") == "1", out


def test_unit_dir_off_tmpfs_warns(tmp_path):
    """Item 31: a unit whose directory is not on tmpfs starts, with a warning in run.out."""
    base = Path(tempfile.mkdtemp(prefix="milk-ul-", dir="/var/tmp"))
    try:
        if (
            subprocess.run(
                ["stat", "-f", "-c", "%T", str(base)], capture_output=True, text=True
            ).stdout.strip()
            == "tmpfs"
        ):
            pytest.skip("/var/tmp is tmpfs here")
        res = lt_bash(
            f'export MILK_UNIT_DIR="{base}"; calc_unit l31 o31\n'
            'milk-fps-runstart l31 >/dev/null; echo "start=$?"\n'
            f'grep -c "is not on tmpfs" "{base}/l31.fps.d/run.out" | sed "s/^/warned=/"\n'
            "milk-fps-runstop l31 >/dev/null",
            tmp_path,
            "",
        )
        out = out_of(res, tmp_path)
        assert value(res.stdout, "start") == "0", out
        assert value(res.stdout, "warned") == "1", out
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_group_writable_fps_is_refused(tmp_path):
    """Item 33: an FPS file that group or others can write is refused."""
    res = lt_bash(
        'calc_unit l33 o33; chmod g+w "$U/l33.fps.shm"\n'
        'milk-fps-runstart l33 >start.log 2>&1; echo "start=$?"',
        tmp_path,
        "",
    )
    out = out_of(res, tmp_path)
    assert value(res.stdout, "start") == "1", out
    assert "writable by group or others" in (tmp_path / "start.log").read_text(), out


# ---- Phase 2: the shell front-end (U10, U11) ----


def test_set_run_and_conf_report_unit_states(tmp_path):
    """@set f.X.run|conf=start|stop return once done; status, run and conf are the unit states (conf
    is not mistaken for the conf.* parameters); @running and @ls follow the run process.
    """
    res = lt_bash(
        "calc_unit s1 o1\n"
        'echo "s0=$(@get f.s1.status)"; echo "c0=$(@get f.s1.conf)"\n'
        '@set f.s1.run=start >start.log 2>&1; echo "start=$?"\n'
        'echo "s1=$(@get f.s1.status)"; echo "r1=$(@get f.s1.run)"\n'
        'echo "up=$(@running f.s1 && echo y)"\n'
        '@ls -l "f.*" | grep -c "s1 *running" | sed "s/^/ls=/"\n'
        '@set f.s1.conf=start >conf.log 2>&1; echo "conf=$?"; echo "c1=$(@get f.s1.conf)"\n'
        'echo "ts=$(@get f.s1.conf.timestring)"\n'
        '@set f.s1.conf=stop f.s1.run=stop >stop.log 2>&1; echo "stop=$?"\n'
        'echo "s2=$(@get f.s1.status)"; echo "c2=$(@get f.s1.conf)"\n'
        'echo "up2=$(@running f.s1 || echo n)"',
        tmp_path,
        "",
    )
    out = out_of(res, tmp_path)
    assert (
        value(res.stdout, "s0") == "stopped" and value(res.stdout, "c0") == "stopped"
    ), out
    assert value(res.stdout, "start") == "0", out
    assert "ready after" in (tmp_path / "start.log").read_text(), out
    assert (
        value(res.stdout, "s1") == "running" and value(res.stdout, "r1") == "running"
    ), out
    assert value(res.stdout, "up") == "y" and value(res.stdout, "ls") == "1", out
    assert (
        value(res.stdout, "conf") == "0" and value(res.stdout, "c1") == "running"
    ), out
    assert value(res.stdout, "ts") != "start", out
    assert value(res.stdout, "stop") == "0", out
    assert (
        value(res.stdout, "s2") == "exited" and value(res.stdout, "c2") == "exited"
    ), out
    assert value(res.stdout, "up2") == "n", out


def test_set_failures_run_once_with_a_hint(tmp_path, ltunit):
    """A failed @set f.X.run=start runs once, never again through the server (D29); :tmux and bad
    values fail with a hint; a start timeout exported in the shell reaches the tool."""
    res = lt_bash(
        "$LT -n s2 fpsinit .in=nosuch >/dev/null\n"
        '@set f.s2.run=start >fail.log 2>&1; echo "fail=$?"\n'
        'grep -c "runstart by" "$U/s2.fps.d/run.out" | sed "s/^/starts=/"\n'
        '@set f.s2.run=start:tmux >tmux.log 2>&1; echo "tmux=$?"\n'
        '@set f.s2.run=go >bad.log 2>&1; echo "bad=$?"\n'
        "$LT -n s3 fpsinit .delay=1.5 >/dev/null\n"
        "export MILK_UNIT_START_TIMEOUT=0.3\n"
        '@set f.s3.run=start >slow.log 2>&1; echo "slow=$?"\n'
        "unset MILK_UNIT_START_TIMEOUT; milk-fps-runstop s3 >/dev/null",
        tmp_path,
        ltunit,
    )
    out = out_of(res, tmp_path)
    assert value(res.stdout, "fail") == "1", out
    assert "parameter check failed" in (tmp_path / "fail.log").read_text(), out
    assert value(res.stdout, "starts") == "1", out
    assert value(res.stdout, "tmux") == "1", out
    assert "tmux was removed" in (tmp_path / "tmux.log").read_text(), out
    assert value(res.stdout, "bad") == "1", out
    assert "invalid value 'go'" in (tmp_path / "bad.log").read_text(), out
    assert value(res.stdout, "slow") == "1", out
    assert "not ready after" in (tmp_path / "slow.log").read_text(), out


def test_waitfor_unit_states(tmp_path):
    """@waitfor f.X running|stopped follows the run process (stopped: whatever the last exit),
    @waitfor f.X waits for the FPS, and RUN/IDLE values fail at once with a hint."""
    res = lt_bash(
        "calc_unit w1 o1\n"
        "( sleep 0.5; milk-fps-runstart w1 >/dev/null ) &\n"
        '@waitfor f.w1 running -t 5; echo "running=$? s1=$(@get f.w1.status)"\n'
        "wait\n"
        "( sleep 0.5; milk-fps-runstop w1 >/dev/null ) &\n"
        '@waitfor f.w1 stopped -t 5; echo "stopped=$? s2=$(@get f.w1.status)"\n'
        "wait\n"
        "( sleep 0.5; calc_unit w2 o2 ) &\n"
        '@waitfor f.w2 -t 5; echo "exists=$?"\n'
        "wait\n"
        'SECONDS=0; @waitfor f.w1.status=RUN -t 5 2>old.log; echo "old=$? t=$SECONDS"\n'
        '@waitfor f.w1 sleeping -t 1 2>bad.log; echo "bad=$?"',
        tmp_path,
        "",
    )
    out = out_of(res, tmp_path)
    assert value(res.stdout, "running") == "0 s1=running", out
    assert value(res.stdout, "stopped") == "0 s2=exited", out
    assert value(res.stdout, "exists") == "0", out
    assert value(res.stdout, "old") == "1 t=0", out
    assert "'RUN' is not a unit state" in (tmp_path / "old.log").read_text(), out
    assert value(res.stdout, "bad") == "1", out


def test_run_out_property(tmp_path):
    """f.X.run.out is the end of the unit's run.out; empty before its first run."""
    res = lt_bash(
        "calc_unit r1 o1\n"
        'echo "never=[$(@get f.r1.run.out)]"\n'
        "milk-fps-runstart r1 >/dev/null; milk-fps-runstop r1 >/dev/null\n"
        '@get f.r1.run.out | tail -n 1 | grep -c "stop: request" | sed "s/^/last=/"\n'
        '@get f.r1.run.out | grep -c "runstart by" | sed "s/^/first=/"',
        tmp_path,
        "",
    )
    out = out_of(res, tmp_path)
    assert value(res.stdout, "never") == "[]", out
    assert value(res.stdout, "last") == "1" and value(res.stdout, "first") == "1", out
