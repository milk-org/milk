"""The milk.calc unit program language: guards, skip, post, frame variables, history rings, ema.

Purpose:     a unit runs its statements on every frame of its trigger stream; these constructs
             decide what runs, what is posted and what state carries from one frame to the next,
             and a unit must keep running through a bad edit or a racing writer.
Does:        zeros() shapes; one program with `if <scalar>: <statement>`, `skip`, `post(name)`,
             the trigger frame's `_cnt0` `_ts` `_missed`, and `hist(s, N)` / `prev(s[, k])`
             rings; errors for `if` without ':' and for a prev() deeper than its ring; a guarded
             statement keeps a self-triggered unit above 100 kHz, and a false guard writes
             nothing; ema() and accumulators that start at zero on frame 0; an edit that does not
             compile leaves the unit running; an output written twice is posted once per tick; a
             parameter set during a slow tick keeps its value; torn reads under a fast producer
             are run again and counted (`_torn`), in a pass, a reduction and a whole-image step
             alike; an input stream that another process creates again is mapped again. A unit
             started with `run` on a stream whose
             semaphore was posted before it started processes that frame first (tick 0), so the
             frames posted by the test start at tick 1.
Covers:      milk-fpsexec-milk-calc (src/calc/units/milk_unit_calc.c), the program runner
             (src/calc/run/mc_prog.c), the rings (plan/mc_history.c), the parser and resolver
             (front/mc_parse.c, front/mc_resolve.c), ema (run/mc_prestage.c), torn-read retries
             (run/mc_exec.c, run/mc_reduce.c).
Uses:        conftest.py: milk_bash, value, CAM, UNIT_HELPERS; the HELPERS and STOP shell
             functions.
See also:    docs/guide/calculator.md, "Compute Units Without C"; units_sessions_test.py
             (milk.calc options and parameters).
"""

from __future__ import annotations

import pytest

from milk.cliwrap import HAVE_CLI
from tests.cli.conftest import (
    milk_bash,
    value,
    CAM,
    UNIT_HELPERS,
)

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.timeout(120),
]

HELPERS = r"""
stop() { kill -INT "$1" 2>/dev/null; for _ in $(seq 60); do kill -0 "$1" 2>/dev/null || break; sleep 0.05; done
         wait "$1" 2>/dev/null; }
mem.mk2Dim "@S:cam" 4 4 >/dev/null; @calc "cam = 1"
"""


def test_zeros_creates_shaped_images(tmp_path):
    res = milk_bash(
        '@calc "z1 = zeros(4, 3); z2 = f64(zeros(5)); z3 = zeros(2, 2, 2) + 1"\n'
        'echo "s1=$(@get l.z1.xsize)x$(@get l.z1.ysize):$(@get l.z1.dtype)"\n'
        'echo "s2=$(@get l.z2.naxis):$(@get l.z2.dtype)"\n'
        '@calc -v t "itot(z3)"; echo "t3=$t"\n'
        '@calc "bad = zeros(0, 4)" 2>/dev/null; echo "badrc=$?"',
        tmp_path,
    )
    assert value(res.stdout, "s1") == "4x3:FLOAT", res.stdout + res.stderr
    assert value(res.stdout, "s2") == "1:DOUBLE", res.stdout + res.stderr
    assert value(res.stdout, "t3") == "8", res.stdout + res.stderr
    assert value(res.stdout, "badrc") != "0"


def test_guards_skip_post_metadata_and_history(tmp_path):
    """One program with every construct; frames 2 3 5 8 13 then a saturated frame and 20."""
    (tmp_path / "prog.calc").write_text(
        "hist(cam, 4)\n"
        "if imax(cam) > 1000: skip\n"
        "diff = cam - prev(cam)\n"
        "diff2 = cam - prev(cam, 2)\n"
        "mean4 = imean(cam_hist, 2)\n"
        "cnt = _cnt0\n"
        "miss = _missed\n"
        "ts = _ts\n"
        "if _n % 2 == 0: post(tag)\n"
        "if cnt > 3: k = k + 1\n"
    )
    res = milk_bash(
        HELPERS
        + 'mem.mk2Dim "@S:tag" 2 2 >/dev/null; @calc "tag = 0"; t0=$(@get s.tag.cnt0)\n'
        'milk-fpsexec-milk-calc -n s4 -t cam -p "cnt:int=0 miss:int=0 ts:float=0 k:int=0" -f prog.calc fpsinit >/dev/null\n'
        "milk-fpsexec-milk-calc -n s4 run >run.log 2>&1 & pid=$!\n"
        "sleep 0.6\n"
        'for v in 2 3 5 8 13; do @calc "cam = $v"; sleep 0.15; done; sleep 0.3\n'
        '@calc -v d,d2,m4 "diff[0,0]; diff2[0,0]; mean4[0,0]"; echo "a=$d,$d2,$m4"\n'
        'echo "sh=$(@get s.diff.naxis):$(@get s.cam_hist.xsize)x$(@get s.cam_hist.ysize)x$(@get s.cam_hist.zsize)"\n'
        'echo "meta=$(@get f.s4.cnt),$(@get f.s4.miss),$(@get f.s4.k)"; ts=$(@get f.s4.ts); echo "ts=$ts"\n'
        'echo "posts=$(( $(@get s.tag.cnt0) - t0 ))"\n'
        '@calc "cam = 5000"; sleep 0.3; @calc -v d "diff[0,0]"; echo "b=$d,$(@get f.s4.cnt)"\n'
        '@calc "cam = 20"; sleep 0.3; @calc -v d,d2 "diff[0,0]; diff2[0,0]"; echo "c=$d,$d2"\n'
        'echo "error=[$(@get f.s4.unit.error)]"; stop $pid',
        tmp_path,
    )
    log = (tmp_path / "run.log").read_text()
    out = res.stdout + res.stderr + log
    assert value(res.stdout, "error") == "[]", out
    # tick 5 (cam 13): prev 8, prev2 5; the ring seen inside the tick holds frames 1..4 (8, 2, 3, 5)
    assert value(res.stdout, "a") == "5,8,4.5", out
    assert (
        value(res.stdout, "sh") == "2:4x4x4"
    ), out  # prev() results are 2-D; the ring 3-D
    # _cnt0 of the last tick is cam's counter (1 creation + 1 assignment + 5 frames), no frame missed,
    # k counted the ticks with cnt > 3
    assert value(res.stdout, "meta") == "7,0,4", out
    assert float(value(res.stdout, "ts")) > 1.6e9, out
    assert value(res.stdout, "posts") == "3", out  # ticks 0, 2, 4
    # the saturated frame is skipped: diff keeps its value, cnt is not updated, but the ring records it
    assert value(res.stdout, "b") == "5,7", out
    assert value(res.stdout, "c") == "-4980,7", out


def test_skip_ends_a_block(tmp_path):
    """`skip` right before `}` skips the frame, as before a newline or `;`."""
    (tmp_path / "sk.calc").write_text("if imax(cam) > 1000 { skip }\nout = cam\n")
    res = milk_bash(
        HELPERS + 'mem.mk2Dim "@S:out" 4 4 >/dev/null; @calc "out = 1; cam = 5000"\n'
        'milk-fpsexec-milk-calc -n sk -f sk.calc exec >sk.log 2>&1; echo "rc=$?"\n'
        'echo "kept=$(@get s.out.mean)"; @calc "cam = 7"\n'
        'milk-fpsexec-milk-calc -n sk -f sk.calc exec >>sk.log 2>&1; echo "ran=$(@get s.out.mean)"',
        tmp_path,
    )
    out = res.stdout + res.stderr + (tmp_path / "sk.log").read_text()
    assert value(res.stdout, "rc") == "0", out
    assert value(res.stdout, "kept") == "1" and value(res.stdout, "ran") == "7", out


def test_if_without_colon_and_prev_depth_are_errors(tmp_path):
    res = milk_bash(
        HELPERS
        + 'milk-fpsexec-milk-calc -n e1 -t cam -e "if cam > 1 out = cam" fpsinit >/dev/null\n'
        "milk-fpsexec-milk-calc -n e1 run >run1.log 2>&1 & p1=$!\n"
        'milk-fpsexec-milk-calc -n e2 -t cam -e "hist(cam, 3); d = cam - prev(cam, 3)" fpsinit >/dev/null\n'
        "milk-fpsexec-milk-calc -n e2 run >run2.log 2>&1 & p2=$!\n"
        'sleep 0.5; @calc "cam = 2"; sleep 0.4\n'
        'echo "e1=[$(@get f.e1.unit.error)]"; echo "e2=[$(@get f.e2.unit.error)]"; stop $p1; stop $p2',
        tmp_path,
    )
    assert "if without ':'" in (value(res.stdout, "e1") or ""), res.stdout + res.stderr
    assert "ring holds 3 frames" in (value(res.stdout, "e2") or ""), (
        res.stdout + res.stderr
    )


def test_guard_is_kept_and_cheap(tmp_path):
    """A guarded 16^2 statement in a self-triggered unit runs at the unguarded rate (well over 100 kHz)."""
    res = milk_bash(
        'mem.mk2Dim "@S:src" 16 16 >/dev/null; mem.mk2Dim "@S:out" 16 16 >/dev/null\n'
        'milk-fpsexec-milk-calc -n g1 -t out -p "g:float=1" -e "if g > 0: out = src + 1" fpsinit >/dev/null\n'
        "milk-fpsexec-milk-calc -n g1 run >run.log 2>&1 & pid=$!\n"
        "sleep 0.5; mem.imsetsempost out -1 >/dev/null; sleep 0.5\n"
        'a=$(@get s.out.cnt0); sleep 1; b=$(@get s.out.cnt0); echo "rate=$(( b - a ))"\n'
        '@set f.g1.g=0; sleep 0.2; c=$(@get s.out.cnt0); sleep 0.3; d=$(@get s.out.cnt0); echo "off=$(( d - c ))"\n'
        "kill -INT $pid; wait $pid 2>/dev/null",
        tmp_path,
    )
    assert int(value(res.stdout, "rate") or 0) > 100000, res.stdout + res.stderr
    # with the guard false nothing is written, so the self-trigger stops (a few late posts at most)
    assert int(value(res.stdout, "off") or 999) < 5, res.stdout + res.stderr


def test_ema_and_frame0_auto_zero_accumulators(tmp_path):
    """ema(expr, alpha) maintains persistent exponential moving average state across ticks (reset on
    _n == 0), and self-referential accumulators (acc = acc + cam, k = k + 1) auto-zero at frame 0.
    """
    (tmp_path / "prog.calc").write_text(
        "acc = acc + cam\n"
        "k = k + 1\n"
        "sm = ema(cam, 0.5)\n"
        "sm_s = ema(imean(cam), 0.5)\n"
        "k_out = k\n"
        "s_out = sm_s\n"
    )
    res = milk_bash(
        HELPERS
        + 'milk-fpsexec-milk-calc -n em1 -t cam -p "k_out:int=0 s_out:float=0" -f prog.calc fpsinit >/dev/null\n'
        "milk-fpsexec-milk-calc -n em1 run >run.log 2>&1 & pid=$!\n"
        "sleep 0.6\n"
        'for v in 3 7 15; do @calc "cam = $v"; sleep 0.15; done; sleep 0.3\n'
        '@calc -v a,m "acc[0,0]; sm[0,0]"; echo "res=$a,$m,$(@get f.em1.k_out),$(@get f.em1.s_out)"\n'
        'echo "error=[$(@get f.em1.unit.error)]"; stop $pid',
        tmp_path,
    )
    out = res.stdout + res.stderr + (tmp_path / "run.log").read_text()
    assert value(res.stdout, "error") == "[]", out
    # tick 0 (cam=1): acc=1, k=1, sm=1
    # tick 1 (cam=3): acc=4, k=2, sm=2
    # tick 2 (cam=7): acc=11, k=3, sm=4.5
    # tick 3 (cam=15): acc=26, k=4, sm=9.75
    assert value(res.stdout, "res") == "26,9.75,4,9.75", out


STOP = (
    'stop() { kill -INT "$1" 2>/dev/null; for _ in $(seq 60); do kill -0 "$1" 2>/dev/null || break;'
    ' sleep 0.05; done; wait "$1" 2>/dev/null; }\n'
)


def test_unit_keeps_running_when_an_edit_does_not_compile(tmp_path):
    res = milk_bash(
        CAM
        + 'milk-fpsexec-milk-calc -n le1 -t cam -e "out_le = cam * 2" fpsinit >/dev/null\n'
        "milk-fpsexec-milk-calc -n le1 run >run.log 2>&1 & pid=$!\n"
        "sleep 0.6\n"
        '@set f.le1.unit.expr="out_le = cam *"; sleep 0.2\n'
        '@calc "cam = 4"; sleep 0.3; @calc -v b "imean(out_le)"; echo "bad_edit=$b"\n'
        'echo "error=[$(@get f.le1.unit.error)]"\n'
        '@set f.le1.unit.expr="out_le = cam * 3"; sleep 0.2\n'
        '@calc "cam = 5"; sleep 0.3; @calc -v c "imean(out_le)"; echo "good_edit=$c"\n'
        "kill -INT $pid; wait $pid",
        tmp_path,
    )
    log = (tmp_path / "run.log").read_text()
    out = res.stdout + res.stderr + log
    assert value(res.stdout, "bad_edit") == "8", out
    assert value(res.stdout, "error") == "[]", out
    assert value(res.stdout, "good_edit") == "15", out
    assert "not applied, still running the previous statements" in log, out


def test_output_written_twice_is_posted_once_per_tick(tmp_path):
    """`out = cam*2; out = out + 1` advances out's cnt0 by one per input frame, with the final value."""
    res = milk_bash(
        UNIT_HELPERS
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


def test_set_during_slow_ticks_is_not_overwritten(tmp_path):
    """A parameter set from the shell while the unit computes keeps the new value (no write-back race)."""
    res = milk_bash(
        UNIT_HELPERS
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


def test_recreated_input_stream_is_read_again(tmp_path):
    """A stream that is not the trigger, deleted and created again by another process while the
    unit runs, is mapped again within half a second: the unit must not go on reading the old file.
    """
    res = milk_bash(
        UNIT_HELPERS + 'mem.mk2Dim "@S:dark" 8 8 >/dev/null; @calc "dark = 1"\n'
        'milk-fpsexec-milk-calc -n rc1 -t cam -e "out = cam - dark" fpsinit >/dev/null\n'
        "milk-fpsexec-milk-calc -n rc1 run >run.log 2>&1 & pid=$!\n"
        'sleep 0.6; @calc "cam = 5"; sleep 0.3; echo "before=$(@get s.out.mean)"\n'
        'mem.rmshmim dark >/dev/null; mem.mk2Dim "@S:dark" 8 8 >/dev/null; @calc "dark = 3"\n'
        'for v in 6 7 8 9; do @calc "cam = $v"; sleep 0.25; done\n'
        'echo "after=$(@get s.out.mean)"; echo "error=[$(@get f.rc1.unit.error)]"; stop $pid',
        tmp_path,
    )
    out = res.stdout + res.stderr + (tmp_path / "run.log").read_text()
    assert value(res.stdout, "before") == "4", out
    assert value(res.stdout, "after") == "6", out  # 9 - 3; 8 with the old dark
    assert value(res.stdout, "error") == "[]", out


def test_torn_counter_with_fast_producer(tmp_path):
    """A consumer reading a stream another unit rewrites at full speed re-runs torn passes and counts them."""
    res = milk_bash(
        STOP
        + 'mem.mk2Dim "@S:src" 128 128 >/dev/null; mem.mk2Dim "@S:out" 128 128 >/dev/null; @calc "src = 1"\n'
        'milk-fpsexec-milk-calc -n prod -t src -e "src = src + 1" fpsinit >/dev/null\n'
        'milk-fpsexec-milk-calc -n cons -t out -p "torn:int=0" -e "out = src * 2 + 1; torn = _torn" fpsinit >/dev/null\n'
        "milk-fpsexec-milk-calc -n cons run >cons.log 2>&1 & pc=$!\n"
        "sleep 0.4; mem.imsetsempost out -1 >/dev/null; sleep 0.4\n"
        'echo "alone=$(@get f.cons.torn)"\n'
        "milk-fpsexec-milk-calc -n prod run >prod.log 2>&1 & pp=$!\n"
        "sleep 0.4; mem.imsetsempost src -1 >/dev/null; sleep 0.8\n"
        'echo "with=$(@get f.cons.torn)"; echo "error=[$(@get f.cons.unit.error)]"\n'
        "stop $pp; stop $pc",
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "alone") == "0", out
    assert int(value(res.stdout, "with") or 0) > 0, out
    assert value(res.stdout, "error") == "[]", out


@pytest.mark.parametrize(
    "stmt",
    ["m = imean(src); out = m", "out = gauss(src, 1)", "out = fftshift(src)"],
    ids=["reduction", "filter", "linalg"],
)
def test_torn_counter_outside_elementwise_passes(tmp_path, stmt):
    """A reduction, a filter and a library step reading a stream rewritten at full speed count torn reads too."""
    res = milk_bash(
        STOP
        + 'mem.mk2Dim "@S:src" 128 128 >/dev/null; mem.mk2Dim "@S:out" 128 128 >/dev/null; @calc "src = 1"\n'
        'milk-fpsexec-milk-calc -n prod -t src -e "src = src + 1" fpsinit >/dev/null\n'
        f'milk-fpsexec-milk-calc -n cons -t out -p "torn:int=0 m:float=0" -e "{stmt}; torn = _torn" fpsinit >/dev/null\n'
        "milk-fpsexec-milk-calc -n cons run >cons.log 2>&1 & pc=$!\n"
        "sleep 0.4; mem.imsetsempost out -1 >/dev/null; sleep 0.4\n"
        'echo "alone=$(@get f.cons.torn)"\n'
        "milk-fpsexec-milk-calc -n prod run >prod.log 2>&1 & pp=$!\n"
        "sleep 0.4; mem.imsetsempost src -1 >/dev/null; sleep 0.8\n"
        'echo "with=$(@get f.cons.torn)"; echo "error=[$(@get f.cons.unit.error)]"\n'
        "stop $pp; stop $pc",
        tmp_path,
    )
    out = res.stdout + res.stderr + (tmp_path / "cons.log").read_text()
    assert value(res.stdout, "alone") == "0", out
    assert int(value(res.stdout, "with") or 0) > 0, out
    assert value(res.stdout, "error") == "[]", out
