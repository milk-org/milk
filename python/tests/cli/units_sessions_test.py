"""
Tests for Phase D (plan §9): compute units from the terminal and sessions, one section per work package.

E8a: the units without C code, milk-fpsexec-milk-calc (milk.calc) and milk-fpsexec-milk-script (milk.script),
on today's fpsexec API. Their loops run as background processes here, stopped with SIGINT, not in tmux.
"""

from __future__ import annotations

import pytest

from milk.cliwrap import HAVE_CLI
from tests.cli.bash_engine_review_test import value
from tests.cli.syntax_redesign_test import milk_bash

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.timeout(60),
]

# Shell helpers: wait for a stream file, post a frame and let the unit run, stop a unit and reap it.
HELPERS = r"""
until_file() { for _ in $(seq 100); do [ -e "$MILK_SHM_DIR/$1.im.shm" ] && return 0; sleep 0.05; done; return 1; }
frame() { @calc "cam = $1"; sleep 0.25; }
stop() { kill -INT "$1" 2>/dev/null; for _ in $(seq 60); do kill -0 "$1" 2>/dev/null || break; sleep 0.05; done
         kill -0 "$1" 2>/dev/null && echo "still running"; wait "$1" 2>/dev/null; }
mem.mk2Dim "@S:cam" 4 4 >/dev/null; @calc "cam = 2"
"""


# ------------------------------------------------------------- E8a: milk.calc


def test_E8_calc_once_makes_a_stream_and_declared_parameters(tmp_path):
    """No trigger: one run. A new image target becomes a stream; declared parameters get their defaults
    (bool true is ON), an assigned one is written back, and `ex` is not taken for `expr`.
    """
    res = milk_bash(
        HELPERS
        + 'milk-fpsexec-milk-calc -n u1 exec "out = cam * gain + 1; tot = itot(out)" "" '
        '"gain:float=3 n:int=2 on:bool=1 off:bool=false ex:int=5 tot:float=0" >/dev/null 2>&1\n'
        'echo "rc=$?"\n'
        'echo "out=$(@get s.out.mean) shared=$(@get s.out.shared)"\n'
        'echo "p=$(@get f.u1.gain),$(@get f.u1.n),$(@get f.u1.on),$(@get f.u1.off),$(@get f.u1.ex)"\n'
        'echo "tot=$(@get f.u1.tot) error=[$(@get f.u1.error)]"',
        tmp_path,
    )
    got = [value(res.stdout, k) for k in ("rc", "out", "p", "tot")]
    assert got == ["0", "7 shared=1", "3,2,ON,OFF,5", "112 error=[]"], (
        res.stdout + res.stderr
    )


def test_E8_calc_bad_declarations_and_long_values(tmp_path):
    res = milk_bash(
        HELPERS
        + 'milk-fpsexec-milk-calc -n b1 fpsinit "out = cam" "" "g:float=x" 2>&1; echo "badval=$?"\n'
        'milk-fpsexec-milk-calc -n b2 fpsinit "out = cam" "" "expr:int=1" 2>&1; echo "reserved=$?"\n'
        'milk-fpsexec-milk-calc -n b3 fpsinit "out = cam" "" "1x:int" 2>&1; echo "badname=$?"\n'
        f'milk-fpsexec-milk-calc -n b4 fpsinit "out = cam + {"1 + " * 20}1" 2>&1; echo "long=$?"',
        tmp_path,
    )
    assert [
        value(res.stdout, k) for k in ("badval", "reserved", "badname", "long")
    ] == ["1"] * 4, (res.stdout + res.stderr)
    assert "bad parameter 'g:float=x'" in res.stdout
    assert "'expr': reserved" in res.stdout
    assert "write the statements in a file (.file)" in res.stdout


def test_E8_calc_loop_follows_frames_and_live_changes(tmp_path):
    """fpsinit, then runstart in another process (it finds the declared parameters in the FPS): each frame
    of .trigger runs the statements; @set of a parameter and of .expr apply from the next frame; an error
    stops the loop with the reason in .error."""
    res = milk_bash(
        HELPERS
        + 'milk-fpsexec-milk-calc -n u2 fpsinit "out2 = cam * gain" cam "gain:float=2" >/dev/null\n'
        "milk-fpsexec-milk-calc -n u2 runstart >run.log 2>&1 & pid=$!\n"
        "until_file out2; sleep 0.3\n"
        'frame 5; echo "a=$(@get s.out2.mean)"\n'
        '@set f.u2.gain=3; frame 5; echo "b=$(@get s.out2.mean)"\n'
        '@set f.u2.expr="out2 = cam + gain"; frame 5; echo "c=$(@get s.out2.mean)"\n'
        '@set f.u2.expr="out2 = cam + nosuch(cam)"; frame 6\n'
        "for _ in $(seq 40); do kill -0 $pid 2>/dev/null || break; sleep 0.05; done\n"
        'echo "exited=$(kill -0 $pid 2>/dev/null && echo no || echo yes)"; stop $pid\n'
        'echo "d=$(@get s.out2.mean)"; echo "error=$(@get f.u2.error)"',
        tmp_path,
    )
    got = [value(res.stdout, k) for k in ("a", "b", "c", "exited", "d", "error")]
    assert got == [
        "10",
        "15",
        "8",
        "yes",
        "8",
        "unknown function 'nosuch' in 'out2 = cam + nosuch(cam)'",
    ], (
        res.stdout + res.stderr + (tmp_path / "run.log").read_text()
    )


def test_E8_calc_program_file_delay_loop_and_stop(tmp_path):
    """.file with comments and several statements, a new image that reads a declared parameter (fused,
    the variable read on each run), the -loopd delay loop, and SIGINT ending the loop through its cleanup.
    """
    (tmp_path / "prog.calc").write_text(
        "# count the ticks\nk = k + 1\nimg = cam * k   # a new image: a stream\nm = imean(img); avg = m\n"
    )
    res = milk_bash(
        HELPERS
        + 'milk-fpsexec-milk-calc -n u3 fpsinit . "" "k:int=0 avg:float=0" prog.calc >/dev/null\n'
        "milk-fpsexec-milk-calc -n u3 -loopd 0.02 runstart >run.log 2>&1 & pid=$!\n"
        "until_file img; sleep 0.5; stop $pid\n"
        'echo "k=$(@get f.u3.k) avg=$(@get f.u3.avg) img=$(@get s.img.mean)"\n'
        'echo "clean=$(grep -c "ran as PID" run.log)"',
        tmp_path,
    )
    m = value(res.stdout, "k")
    assert m is not None, res.stdout + res.stderr
    k, avg, img = int(m.split()[0]), m.split()[1], m.split()[2]
    assert k >= 5 and avg == f"avg={2 * k}" and img == f"img={2 * k}", (
        res.stdout + res.stderr
    )
    assert value(res.stdout, "clean") == "1", (
        res.stdout + (tmp_path / "run.log").read_text()
    )


def test_E8_calc_init_and_initfile_and_frame_vars(tmp_path):
    """One-time .init / .initfile runs before frame 0, is preserved across live .expr changes,
    re-runs when .init changes live, and provides read-only _n, _t, _dt frame variables.
    """
    (tmp_path / "init.calc").write_text("acc = cam * 0; k0 = 10\n")
    res = milk_bash(
        HELPERS
        + 'milk-fpsexec-milk-calc -n u4 fpsinit "acc = acc + cam; fn = _n; ft = _t; fdt = _dt" '
        'cam "k0:int=0 fn:int=-1 ft:float=-1 fdt:float=-1" . . init.calc >/dev/null\n'
        "milk-fpsexec-milk-calc -n u4 runstart >run.log 2>&1 & pid=$!\n"
        "until_file acc; sleep 0.3\n"
        'echo "f0=$(@get s.acc.mean),$(@get f.u4.k0),$(@get f.u4.fn)"\n'
        'frame 3; echo "f1=$(@get s.acc.mean),$(@get f.u4.fn)"\n'
        '@set f.u4.expr="acc = acc + cam * 2; fn = _n"; frame 5\n'
        'echo "f2=$(@get s.acc.mean),$(@get f.u4.fn)"\n'
        '@set f.u4.initfile="" f.u4.init="acc = cam * 0 + 100"; frame 2\n'
        'echo "f3=$(@get s.acc.mean),$(@get f.u4.fn)"\n'
        "stop $pid\n"
        'milk-fpsexec-milk-calc -n ubad exec "_n = 5" >/dev/null 2>&1; echo "ro=$(@get f.ubad.error)"',
        tmp_path,
    )
    got = [value(res.stdout, k) for k in ("f0", "f1", "f2", "f3", "ro")]
    assert got == [
        "2,10,0",
        "5,1",
        "15,2",
        "104,0",
        "cannot assign to read-only variable '_n'",
    ], (
        res.stdout + res.stderr + (tmp_path / "run.log").read_text()
    )


def test_E8_calc_isnan_isfinite_and_stale_stream_remap(tmp_path):
    """IEEE-754 bit-level isnan() and isfinite() work under -Ofast on scalars and images, and
    milk.calc automatically remaps a shared stream recreated with a new inode in /dev/shm.
    """
    res = milk_bash(
        HELPERS
        + '@calc -v s_nan,s_fin1,s_fin2 "isnan(sqrt(-1)); isfinite(exp(1000)); isfinite(42.0)"\n'
        'echo "sc=$s_nan,$s_fin1,$s_fin2"\n'
        '@test "sqrt(-1)"; echo "tnan=$?"\n'
        '@calc "bad = cam * 1.0; bad[0,0] = sqrt(-1); clean = where(isnan(bad), 99, bad)"\n'
        'echo "cl=$(@get l.clean.tot)"\n'
        'mem.mk2Dim "@S:cal" 4 4 >/dev/null; @calc "cal = 3"\n'
        'milk-fpsexec-milk-calc -n u5 fpsinit "out5 = cam + cal" cam >/dev/null\n'
        "milk-fpsexec-milk-calc -n u5 runstart >run.log 2>&1 & pid=$!\n"
        "until_file out5; sleep 0.3\n"
        'frame 2; echo "before=$(@get s.out5.mean)"\n'
        'rm -f "$MILK_SHM_DIR/cal.im.shm"\n'
        'mem.mk2Dim "@S:cal" 4 4 >/dev/null; @calc "cal = 10"\n'
        "sleep 0.1\n"
        'frame 2; echo "after=$(@get s.out5.mean)"; stop $pid',
        tmp_path,
    )
    got = [value(res.stdout, k) for k in ("sc", "tnan", "cl", "before", "after")]
    assert got == ["1,0,1", "1", "129", "5", "12"], (
        res.stdout + res.stderr + (tmp_path / "run.log").read_text()
    )


def test_E8_calc_matmul_and_transpose(tmp_path):
    """transpose(), matmul(transpose(A), B), matmul(A, transpose(B)), and pre-compiled kept
    dst = matmul(...) / dst = transpose(...) in milk.calc."""
    res = milk_bash(
        HELPERS + 'mem.mk2Dim "@S:mA" 3 2 >/dev/null\n'
        'mem.mk2Dim "@S:vB" 2 1 >/dev/null\n'
        '@calc "mA[0,0]=1; mA[1,0]=2; mA[2,0]=3; mA[0,1]=4; mA[1,1]=5; mA[2,1]=6"\n'
        '@calc "vB[0,0]=10; vB[1,0]=20"\n'
        '@calc "mAt = transpose(mA); p1 = matmul(mA, vB); p2 = matmul(transpose(mAt), vB)"\n'
        '@calc "p3 = matmul(mA, transpose(mA))"\n'
        'echo "shapes=$(@get l.mAt.xsize)x$(@get l.mAt.ysize),$(@get l.p3.xsize)x$(@get l.p3.ysize)"\n'
        'echo "tots=$(@get l.p1.tot),$(@get l.p2.tot),$(@get l.p3.tot)"\n'
        'milk-fpsexec-milk-calc -n u6 exec "outm = matmul(mA, vB); outt = transpose(mA)" >/dev/null\n'
        'echo "unit=$(@get s.outm.tot),$(@get s.outt.xsize)x$(@get s.outt.ysize)"',
        tmp_path,
    )
    got = [value(res.stdout, k) for k in ("shapes", "tots", "unit")]
    assert got == ["2x3,3x3", "360,360,261", "360,2x3"], res.stdout + res.stderr


# ------------------------------------------------------------- E8a: milk.script


def test_E8_script_runs_per_frame_follows_cmd_and_stops_on_failure(tmp_path):
    res = milk_bash(
        HELPERS
        + "milk-fpsexec-milk-script -n s1 fpsinit '@get s.cam.mean >> log' cam >/dev/null\n"
        "milk-fpsexec-milk-script -n s1 runstart >run.log 2>&1 & pid=$!\n"
        "for _ in $(seq 50); do @calc 'cam = 1'; sleep 0.1; [ -s log ] && break; done; sleep 0.3; : > log\n"
        'frame 3; frame 4; echo "log=$(tr "\\n" , < log)"\n'
        "@set f.s1.cmd='echo tick >> log2'; frame 5; frame 6; echo \"log2=$(tr '\\n' , < log2)\"\n"
        "@set f.s1.cmd='false'; frame 7\n"
        "for _ in $(seq 40); do kill -0 $pid 2>/dev/null || break; sleep 0.05; done\n"
        'echo "exited=$(kill -0 $pid 2>/dev/null && echo no || echo yes)"; stop $pid\n'
        'echo "error=$(@get f.s1.error)"',
        tmp_path,
    )
    got = [value(res.stdout, k) for k in ("log", "log2", "exited", "error")]
    assert got == ["3,4,", "tick,tick,", "yes", "status 1: false"], (
        res.stdout + res.stderr + (tmp_path / "run.log").read_text()
    )


def test_E8_script_stop_interrupts_a_long_tick(tmp_path):
    """SIGINT during a tick ends it, the loop and the shell (none left behind) without an error."""
    res = milk_bash(
        HELPERS
        + "milk-fpsexec-milk-script -n s2 -loopd 0.05 exec 'sleep 30' >run.log 2>&1 & pid=$!\n"
        "for _ in $(seq 100); do sh=$(pgrep -P $pid -x bash) && break; sleep 0.05; done\n"
        "sleep 0.3; t0=$EPOCHREALTIME; stop $pid\n"
        'echo "ms=$(( (${EPOCHREALTIME/./} - ${t0/./}) / 1000 ))"\n'
        'echo "shell=$(kill -0 $sh 2>/dev/null && echo alive || echo gone) error=[$(@get f.s2.error)]"',
        tmp_path,
    )
    assert value(res.stdout, "shell") == "gone error=[]", res.stdout + res.stderr
    assert int(value(res.stdout, "ms") or 9999) < 2000, res.stdout
