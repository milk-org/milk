"""
The unit program language (plan 'the-new-milk-cli-calc' step 4): `if <scalar>: <statement>`, `skip`,
`post(name)`, the trigger frame's `_cnt0` `_ts` `_missed`, `hist(s, N)` / `prev(s[, k])` ring buffers,
and `zeros()`.

A unit started with `run` on a stream whose semaphore was posted before it started processes that
frame first (tick 0), so the frames posted by the test start at tick 1.
"""

from __future__ import annotations

import pytest

from milk.cliwrap import HAVE_CLI
from tests.cli.conftest import milk_bash, value

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
