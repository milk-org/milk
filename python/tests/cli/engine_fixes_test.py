"""
Regression tests for the 2026-10-23 engine review fixes (compute units, calculator, client, front-end).

Each test names the finding it guards: a unit must not request RT priority 0 (H3), an @set during a
tick is not overwritten by the loop's write-back (H4), a re-created trigger stream is remapped (H5),
semaphore slots are released after a wait (H8), `im /= image` reports a zero divisor (H9), a rejected
scalar divisor posts no frame (M11), an output written by several statements is posted once per
tick (M4), a fast producer piped into milk-cli still gets its lines run (M10), and a unit launched
under taskset keeps its mask (H1, L1). Found later (2026-10-26): an ON/OFF parameter that defaults to
ON is ON in its unit, a failed @set prints its error once, git's `@{-1}` is left as written, and `p.`
reads find a unit's processinfo where it writes it. Found by the docs audit (2026-10-26): an image as
the condition of @test, @assert, @waitfor or `if` is an error, an image that cannot be assigned is an
error (not "Cannot change name" and status 0), a block's error is printed once, `milk-cli -h` lists the
options milk-cli takes, and `@config sessionlog` logs each line.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from milk.cliwrap import HAVE_CLI, MILK_CLI_EXEC
from tests.cli.conftest import (
    _install_dir,
    _isolated_shm_env,
    _run,
    milk_bash,
    milk_cli_script,
    value,
)

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.timeout(120),
]

# Shell helpers: wait for a stream file, stop a unit and reap it.
HELPERS = r"""
until_file() { for _ in $(seq 100); do [ -e "$MILK_SHM_DIR/$1.im.shm" ] && return 0; sleep 0.05; done; return 1; }
stop() { kill -INT "$1" 2>/dev/null; for _ in $(seq 60); do kill -0 "$1" 2>/dev/null || break; sleep 0.05; done
         kill -0 "$1" 2>/dev/null && echo "still running"; wait "$1" 2>/dev/null; }
mem.mk2Dim "@S:cam" 8 8 >/dev/null; @calc "cam = 2"
"""


def test_H3_unit_start_requests_no_rt_priority(tmp_path):
    """A unit whose FPS keeps the default RTprio (-1) neither warns nor writes 'RT prio failed'."""
    res = milk_bash(
        HELPERS
        + 'milk-fpsexec-milk-calc -n h3 -t cam -e "out3 = cam * 2" fpsinit >/dev/null\n'
        "milk-fpsexec-milk-calc -n h3 run >run.log 2>&1 & pid=$!\n"
        'until_file out3; sleep 0.3; @calc "cam = 3"; sleep 0.3\n'
        'echo "msg=$(@get p.h3.msg)"; echo "rtprio=$(@get f.h3.procinfo.RTprio)"; stop $pid',
        tmp_path,
    )
    log = (tmp_path / "run.log").read_text()
    assert "RT priority" not in log, log
    assert "RT prio" not in (value(res.stdout, "msg") or ""), res.stdout + res.stderr
    assert value(res.stdout, "rtprio") == "-1", res.stdout + res.stderr


def test_M4_output_written_twice_is_posted_once_per_tick(tmp_path):
    """`out = cam*2; out = out + 1` advances out's cnt0 by one per input frame, with the final value."""
    res = milk_bash(
        HELPERS
        + 'milk-fpsexec-milk-calc -n m4 -t cam -e "out4 = cam * 2; out4 = out4 + 1" fpsinit >/dev/null\n'
        "milk-fpsexec-milk-calc -n m4 run >run.log 2>&1 & pid=$!\n"
        "until_file out4; sleep 0.3\n"
        "c0=$(@get s.out4.cnt0)\n"
        'for i in 1 2 3 4 5; do @calc "cam = $i"; sleep 0.1; done; sleep 0.3\n'
        'echo "posts=$(( $(@get s.out4.cnt0) - c0 ))"; echo "val=$(@get s.out4.mean)"; stop $pid',
        tmp_path,
    )
    assert value(res.stdout, "posts") == "5", (
        res.stdout + res.stderr + (tmp_path / "run.log").read_text()
    )
    assert value(res.stdout, "val") == "11", res.stdout + res.stderr


def test_H9_M11_zero_divisors_error_and_post_nothing(tmp_path):
    """`a /= b` with a zero pixel is an error that leaves `a` as it was; `a = b / x` with x == 0
    neither changes `a` nor posts a frame."""
    res = milk_bash(
        'mem.mk2Dim "@S:a" 4 4 >/dev/null; mem.mk2Dim "@S:b" 4 4 >/dev/null\n'
        '@calc "a = 8; b = where(ix(b) == 0, 0, 2)"\n'
        "c0=$(@get s.a.cnt0)\n"
        '@calc "a /= b" 2>err.log; echo "rc=$?"; echo "amean=$(@get s.a.mean)"\n'
        'x=0; @calc "a = b / x" 2>>err.log; echo "rc2=$?"\n'
        'echo "dcnt=$(( $(@get s.a.cnt0) - c0 ))"',
        tmp_path,
    )
    err = (tmp_path / "err.log").read_text()
    assert value(res.stdout, "rc") not in (None, "0"), res.stdout + err
    assert "zero" in err.lower(), err
    assert value(res.stdout, "amean") == "8", res.stdout + err
    assert value(res.stdout, "rc2") not in (None, "0"), res.stdout + err
    assert value(res.stdout, "dcnt") == "0", res.stdout + err


def test_H8_semaphore_slots_released_after_waits(tmp_path):
    """Six concurrent `@waitfor` on one stream, then none of the stream's reader slots is still held."""
    res = milk_bash(
        'mem.mk2Dim "@S:x" 4 4 >/dev/null\n'
        'for i in 1 2 3 4 5 6; do ( @waitfor s.x --timeout 0.6 >/dev/null 2>&1; echo "w=$?" ) & done\n'
        "sleep 0.2; mem.imseminfo x > during.txt 2>&1; wait\n"
        "mem.imseminfo x > after.txt 2>&1\n"
        '@waitfor s.x --timeout 0.2 >/dev/null 2>&1; echo "last=$?"\n'
        "mem.imseminfo x > final.txt 2>&1",
        tmp_path,
    )
    assert res.stdout.count("w=124") == 6, res.stdout + res.stderr

    def held(path: Path) -> list[int]:
        pids = []
        for ln in path.read_text().splitlines():
            m = re.match(r"\s*(\d+)\s+(\d+)\s+(-?\d+)\s+(-?\d+)\s*$", ln)
            if m and int(m.group(4)) > 0:
                pids.append(int(m.group(4)))
        return pids

    assert held(tmp_path / "during.txt"), (tmp_path / "during.txt").read_text()
    assert held(tmp_path / "after.txt") == [], (tmp_path / "after.txt").read_text()
    assert held(tmp_path / "final.txt") == [], (tmp_path / "final.txt").read_text()


def test_H4_set_during_slow_ticks_is_not_overwritten(tmp_path):
    """A parameter set from the shell while the unit computes keeps the new value (no write-back race)."""
    res = milk_bash(
        HELPERS
        + 'mem.mk2Dim "@S:big" 1024 1024 >/dev/null; mem.mk2Dim "@S:bigo" 1024 1024 >/dev/null\n'
        '@calc "big = 1"\n'
        'milk-fpsexec-milk-calc -n h4 -t bigo -p "gain:float=1" -e "bigo = sin(big) + exp(big) * gain" fpsinit >/dev/null\n'
        "OMP_NUM_THREADS=1 milk-fpsexec-milk-calc -n h4 run >run.log 2>&1 & pid=$!\n"
        "sleep 0.5; mem.imsetsempost bigo -1 >/dev/null; sleep 0.3\n"
        "bad=0\n"
        'for i in $(seq 1 20); do @set f.h4.gain=$i; sleep 0.03; g=$(@get f.h4.gain); [ "$g" = "$i" ] || bad=$((bad+1)); done\n'
        'echo "bad=$bad"; echo "cnt=$(@get s.bigo.cnt0)"; stop $pid',
        tmp_path,
    )
    log = (tmp_path / "run.log").read_text()
    assert int(value(res.stdout, "cnt") or 0) > 50, res.stdout + res.stderr + log
    assert value(res.stdout, "bad") == "0", res.stdout + res.stderr + log


def test_H5_recreated_trigger_stream_is_remapped(tmp_path):
    """The trigger stream deleted and created again (new inode) is picked up after a timeout, which
    `p.<unit>.timeouts` counts."""
    res = milk_bash(
        HELPERS + 'mem.mk2Dim "@S:trg" 4 4 >/dev/null; @calc "trg = 1"\n'
        'milk-fpsexec-milk-calc -n h5 -t trg -e "out5 = trg * 2" fpsinit >/dev/null\n'
        "milk-fpsexec-milk-calc -n h5 run >run.log 2>&1 & pid=$!\n"
        'until_file out5; sleep 0.3; @calc "trg = 1"; sleep 0.3; c0=$(@get s.out5.cnt0)\n'
        'mem.rmshmim trg >/dev/null 2>&1; sleep 0.2; mem.mk2Dim "@S:trg" 4 4 >/dev/null\n'
        "sleep 3.0\n"
        'for i in 1 2 3; do @calc "trg = 3"; sleep 0.3; done\n'
        'echo "adv=$(( $(@get s.out5.cnt0) - c0 ))"; echo "to=$(@get p.h5.timeouts)"; echo "val=$(@get s.out5.mean)"\n'
        "stop $pid",
        tmp_path,
    )
    log = (tmp_path / "run.log").read_text()
    assert int(value(res.stdout, "adv") or 0) >= 1, res.stdout + res.stderr + log
    assert int(value(res.stdout, "to") or 0) >= 1, res.stdout + res.stderr + log
    assert value(res.stdout, "val") == "6", res.stdout + res.stderr + log


def test_onoff_default_on_reaches_the_unit(tmp_path):
    """streamave's .comp.ave defaults to ON, so the unit posts the mean of every 2 frames. The FPS used to
    start with the flag OFF, which the unit reads, so it never wrote its output."""
    res = milk_bash(
        HELPERS
        + "milk-fpsexec-mem-streamave -n sa1 -loops fpsinit cam avg 2 >/dev/null\n"
        'echo "ave=$(@get f.sa1.comp.ave) rms=$(@get f.sa1.comp.rms)"\n'
        "milk-fpsexec-mem-streamave -n sa1 -loops run >run.log 2>&1 & pid=$!\n"
        "until_file avg; sleep 0.3; c0=$(@get s.avg.cnt0)\n"
        'for _ in 1 2 3 4 5 6; do @calc "cam = 5"; sleep 0.1; done; sleep 0.3\n'
        'echo "posts=$(( $(@get s.avg.cnt0) - c0 ))"; echo "val=$(@get s.avg.mean)"; stop $pid',
        tmp_path,
    )
    out = res.stdout + res.stderr + (tmp_path / "run.log").read_text()
    assert value(res.stdout, "ave") == "ON rms=OFF", out
    assert int(value(res.stdout, "posts") or 0) >= 2, out
    assert value(res.stdout, "val") == "5", out


def test_failed_set_prints_its_error_once(tmp_path):
    """An @set that fails in the shell is not sent again to the server, which printed the same error a
    second time. An @set on a server-local image still goes to the server, with no error.
    """
    specs = [
        "f.se.procinfo.loopcntMax=abc",
        "f.se.nosuchparam=1",
        "f.nosuch.x=1",
        "f.se.procinfo.triggermode=often",
        "s.nosuch.kw.A=1",
        "p.nosuch.ctrl=pause",
        "s.cam.nosuchprop=1",
    ]
    res = milk_bash(
        HELPERS
        + 'milk-fpsexec-milk-calc -n se -e "o = cam" fpsinit >/dev/null\n'
        + "".join(
            f'@set "{s}" >set{i}.log 2>&1; echo "rc{i}=$?"\n'
            for i, s in enumerate(specs)
        )
        + 'mem.mk2Dim "@L:limg" 4 4 >/dev/null; @set s.limg.kw.A=1 >lim.log 2>&1; echo "lrc=$?"\n'
        'echo "lim=$(@get l.limg.kw.A)"',
        tmp_path,
    )
    for i, spec in enumerate(specs):
        lines = (tmp_path / f"set{i}.log").read_text().splitlines()
        assert value(res.stdout, f"rc{i}") not in (None, "0"), (
            spec + res.stdout + res.stderr
        )
        assert len(lines) == 1, f"{spec}: {lines}"
    assert value(res.stdout, "lrc") == "0", res.stdout + res.stderr
    assert (tmp_path / "lim.log").read_text() == ""
    assert value(res.stdout, "lim") == "1", res.stdout + res.stderr


def test_integer_braces_are_left_to_git(tmp_path):
    """`@{-1}` and `@{3}` (git's forms) are passed to bash as written, in words and assignments. An `@{`
    right after a letter is not expanded either, unless the expansion is quoted."""
    res = milk_cli_script(
        "echo a=@{-1} @{3}\n"
        'v=@{-2}; echo "b=$v"\n'
        'w="@{-1}"; echo "c=$w"\n'
        "echo d=img@{1+1}\n"
        'echo e=img"@{1+1}"',
        tmp_path,
    )
    assert value(res.stdout, "a") == "@{-1} @{3}", res.stdout + res.stderr
    assert value(res.stdout, "b") == "@{-2}", res.stdout + res.stderr
    assert value(res.stdout, "c") == "@{-1}", res.stdout + res.stderr
    assert value(res.stdout, "d") == "img@{1+1}", res.stdout + res.stderr
    assert value(res.stdout, "e") == "img2", res.stdout + res.stderr


@pytest.mark.skipif(
    not os.access("/milk/shm", os.W_OK), reason="needs a writable /milk/shm"
)
def test_proc_reads_look_where_units_write(tmp_path):
    """With MILK_PROC_DIR apart from MILK_SHM_DIR, `@get p.` and `@ls` find the unit's processinfo in
    MILK_PROC_DIR. They read MILK_SHM_DIR instead when MILK_PROC_DIR was under /milk/shm or /dev/shm.
    """
    proc_dir = tempfile.mkdtemp(prefix="milk_proc_", dir="/milk/shm")
    try:
        res = milk_bash(
            HELPERS
            + 'milk-fpsexec-milk-calc -n pd1 -loopd 0.02 -e "outp = cam" fpsinit >/dev/null\n'
            "milk-fpsexec-milk-calc -n pd1 run >run.log 2>&1 & pid=$!\n"
            "until_file outp; sleep 0.3\n"
            'echo "status=$(@get p.pd1.status)"; echo "ls=$(@ls -l p.pd1 | grep -c pd1)"; stop $pid',
            tmp_path,
            extra_env={"MILK_PROC_DIR": proc_dir},
        )
    finally:
        shutil.rmtree(proc_dir, ignore_errors=True)
    out = res.stdout + res.stderr + (tmp_path / "run.log").read_text()
    assert value(res.stdout, "status") == "ACTIVE", out
    assert value(res.stdout, "ls") == "1", out


def test_image_conditions_are_errors(tmp_path):
    """An image is neither true nor false: @test, @assert, @waitfor and `if` fail (status 2) with a hint,
    and @waitfor at once rather than at its timeout. imin()/imax() of the image are the conditions.
    """
    res = milk_cli_script(
        'mem.mk2Dim "@S:im" 4 4 >/dev/null; @calc "im = 0"\n'
        '@test "im > 1" 2>t.err; echo "test=$?"\n'
        '@assert "im > 1" 2>/dev/null; echo "assert=$?"\n'
        's=$SECONDS; @waitfor "im > 1" -t 5 2>/dev/null; echo "waitfor=$? waited=$((SECONDS - s))"\n'
        '@calc "if im > 1: x = 1" 2>/dev/null; echo "if=$?"\n'
        '@test "imax(im > 1)"; echo "any=$?"\n'
        '@test "imin(im < 1)"; echo "all=$?"',
        tmp_path,
        timeout=30,
    )
    out = res.stdout + res.stderr
    assert (
        "imin(...) for all pixels, imax(...) for any"
        in (tmp_path / "t.err").read_text()
    ), out
    assert value(res.stdout, "test") == "2", out
    assert value(res.stdout, "assert") not in (None, "0"), out
    assert value(res.stdout, "waitfor") in ("2 waited=0", "2 waited=1"), out
    assert value(res.stdout, "if") == "2", out
    assert value(res.stdout, "any") == "1", out
    assert value(res.stdout, "all") == "0", out


def test_failed_image_assignments_are_errors(tmp_path):
    """`u, v = a, b` assigns an existing image as `v = b` does: a stream of another size is an error that
    leaves it as it was, a number fills it. An image assigned to a variable is an error. Both used to
    print "Cannot change name" and return 0. A block's error is printed once."""
    res = milk_cli_script(
        'mem.mk2Dim "@S:im" 4 4 >/dev/null; @calc "im = 2"\n'
        'mem.mk2Dim "@S:v" 2 2 >/dev/null; @calc "v = 7"\n'
        '@calc "u, v = im + 1, im * 2" 2>e1.txt; echo "mismatch=$?"; echo "v=$(@get s.v.xsize) $(@get s.v.mean)"\n'
        'mem.mk2Dim "@S:w" 4 4 >/dev/null\n'
        '@calc "u, w = im + 1, im + 5"; echo "same=$? $(@get s.w.mean)"\n'
        '@calc "u, w = 3, 4"; echo "numbers=$? $(@get s.w.mean)"\n'
        '@calc "x = 5"; @calc "x = im * 3" 2>e2.txt; echo "var=$?"; @calc -v y x; echo "x=$y"\n'
        '@calc "if 1: y = nosuchname + 1" 2>e3.txt; echo "block=$?"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    for f in ("e1.txt", "e2.txt", "e3.txt"):
        out += f"\n--- {f}\n" + (tmp_path / f).read_text()
    assert value(res.stdout, "mismatch") == "2", out
    assert "Dimension mismatch" in (tmp_path / "e1.txt").read_text(), out
    assert value(res.stdout, "v") == "2 7", out
    assert value(res.stdout, "same") == "0 7", out
    assert value(res.stdout, "numbers") == "0 4", out
    assert value(res.stdout, "var") == "2", out
    assert (
        "'x' is a variable (calculator or bash)" in (tmp_path / "e2.txt").read_text()
    ), out
    assert value(res.stdout, "x") == "5", out
    assert "Cannot change name" not in out, out
    assert value(res.stdout, "block") == "2", out
    assert (tmp_path / "e3.txt").read_text().count("undefined identifier") == 1, out


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


def test_M10_fast_producer_piped_into_milk_cli_runs_every_line(tmp_path):
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


@pytest.mark.skipif(
    not os.path.exists("/sys/devices/cpu_core/cpus")
    or len(os.sched_getaffinity(0)) < 24,
    reason="needs a hybrid P+E CPU with at least 24 CPUs allowed",
)
def test_H1_L1_unit_under_taskset_keeps_its_cpu_mask(tmp_path):
    """A unit started with a P+E mask by taskset is still on that mask after large OpenMP statements."""
    env, shm_dir = _isolated_shm_env()
    bindir = _install_dir() / "bin"
    proc = None
    try:
        r = _run(
            [str(bindir / "milk-script"), "-q", "-"],
            env,
            timeout=30,
            stdin='mem.mk2Dim "@S:aff" 2048 2048\n',
            cwd=str(tmp_path),
        )
        assert r.returncode == 0, r.stderr
        proc = subprocess.Popen(
            [
                "taskset",
                "-c",
                "8-23",
                str(bindir / "milk-fpsexec-milk-calc"),
                "-n",
                "aff",
                "-loopd",
                "0.001",
                "-e",
                "aff = aff * 2 + 1",
                "exec",
            ],
            env=env,
            cwd=str(tmp_path),
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2.0)
        assert proc.poll() is None, "unit exited early"
        status = Path(f"/proc/{proc.pid}/status").read_text()
        allowed = re.search(r"^Cpus_allowed_list:\s*(\S+)", status, re.M).group(1)
        assert allowed == "8-23", allowed
    finally:
        if proc is not None:
            proc.kill()
            proc.wait()
        shutil.rmtree(shm_dir, ignore_errors=True)
