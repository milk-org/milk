"""
Tests for Phase D (plan §9): compute units from the terminal and sessions, one section per work package.

E8a: the units without C code, milk-fpsexec-milk-calc (milk.calc) and milk-fpsexec-milk-script (milk.script),
on today's fpsexec API. Their loops run as background processes here, stopped with SIGINT, not in tmux.

E2: sessions (D33-D35, revised in §9.4): the @session keyword, names, verbs, and the hints that
replace @server. Session names are global (abstract sockets): each test uses names of its own.

E3: owners and lifetimes (D37): the owner and orphaned sessions, the idle timeout, @on watchers that
end with their shell, and override files named after the shell's start.

E5: milk-cli --session NAME (D36 revised: milk-cli keeps its own session, its bash joins NAME), and
@session save/restore.

E4: shared sessions run one command at a time (D38), and history is per session (D39).

E6: @get --watch (D41) and the parameter sweep as a loop (D42); D40, a multi-stream @grab, was
dropped (user decision 2026-10-26).

D32: milk-demopipeline is a milk script that starts each unit once the one before it has written a
frame (@waitfor), with no fixed sleep.

E1: what D30 left after the unit-lifecycle work: the parameter check as f.<unit>.ok/errors/msg,
trigger modes by name, p.<unit>.dt/dtmax, the same loop rates in @get, --json and @ls, and
@waitfor p.<unit> seeing a clean exit.

Docs audit (2026-10-26): @session save keeps the calculator's string variables, and @session new of
a session that runs from another MILK_SHM_DIR names that directory.
"""

from __future__ import annotations

import contextlib
import json
import os
import pty
import re
import select
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import pytest

from milk.cliwrap import HAVE_CLI
from tests.cli.conftest import (
    MILK_CLI_EXEC,
    _install_dir,
    _isolated_shm_env,
    _kill_session,
    _run,
    milk_bash,
    milk_proc,
    value,
)

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
    (bool true is ON), an assigned one is written back, and `expr` is free as a user parameter name
    because the unit definition lives under `.unit.*`.
    """
    res = milk_bash(
        HELPERS
        + 'milk-fpsexec-milk-calc -n u1 -e "out = cam * gain + 1; tot = itot(out)" '
        '-p "gain:float=3 n:int=2 on:bool=1 off:bool=false expr:int=5 tot:float=0" exec >/dev/null 2>&1\n'
        'echo "rc=$?"\n'
        'echo "out=$(@get s.out.mean) shared=$(@get s.out.shared)"\n'
        'echo "p=$(@get f.u1.gain),$(@get f.u1.n),$(@get f.u1.on),$(@get f.u1.off),$(@get f.u1.expr)"\n'
        'echo "tot=$(@get f.u1.tot) error=[$(@get f.u1.unit.error)]"',
        tmp_path,
    )
    got = [value(res.stdout, k) for k in ("rc", "out", "p", "tot")]
    assert got == ["0", "7 shared=1", "3,2,ON,OFF,5", "112 error=[]"], (
        res.stdout + res.stderr
    )


def test_E8_calc_bad_declarations_and_long_values(tmp_path):
    res = milk_bash(
        HELPERS
        + 'milk-fpsexec-milk-calc -n b1 -e "out = cam" -p "g:float=x" fpsinit 2>&1; echo "badval=$?"\n'
        'milk-fpsexec-milk-calc -n b2 -e "out = cam" -p "unit:int=1" fpsinit 2>&1; echo "reserved=$?"\n'
        'milk-fpsexec-milk-calc -n b3 -e "out = cam" -p "1x:int" fpsinit 2>&1; echo "badname=$?"\n'
        f'milk-fpsexec-milk-calc -n b4 -e "out = cam + {"1 + " * 20}1" fpsinit 2>&1; echo "long=$?"',
        tmp_path,
    )
    assert [
        value(res.stdout, k) for k in ("badval", "reserved", "badname", "long")
    ] == ["2"] * 4, (res.stdout + res.stderr)
    assert "bad parameter 'g:float=x'" in res.stdout
    assert "'unit': reserved" in res.stdout
    assert "write the statements in a file (-f)" in res.stdout


def test_E8_calc_loop_follows_frames_and_live_changes(tmp_path):
    """fpsinit, then run in another process (it finds the declared parameters in the FPS): each frame
    of .unit.trigger runs the statements; @set of a parameter and of .unit.expr apply from the next frame;
    an error stops the loop with the reason in .unit.error."""
    res = milk_bash(
        HELPERS
        + 'milk-fpsexec-milk-calc -n u2 -t cam -p "gain:float=2" -e "out2 = cam * gain" fpsinit >/dev/null\n'
        "milk-fpsexec-milk-calc -n u2 run >run.log 2>&1 & pid=$!\n"
        "until_file out2; sleep 0.3\n"
        'frame 5; echo "a=$(@get s.out2.mean)"\n'
        '@set f.u2.gain=3; frame 5; echo "b=$(@get s.out2.mean)"\n'
        '@set f.u2.unit.expr="out2 = cam + gain"; frame 5; echo "c=$(@get s.out2.mean)"\n'
        '@set f.u2.unit.expr="out2 = cam + nosuch(cam)"; frame 6\n'
        "for _ in $(seq 40); do kill -0 $pid 2>/dev/null || break; sleep 0.05; done\n"
        'echo "exited=$(kill -0 $pid 2>/dev/null && echo no || echo yes)"; stop $pid\n'
        'echo "d=$(@get s.out2.mean)"; echo "error=$(@get f.u2.unit.error)"',
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
    """.unit.file with comments and several statements, a new image that reads a declared parameter (fused,
    the variable read on each run), the -loopd delay loop, and SIGINT ending the loop through its cleanup.
    """
    (tmp_path / "prog.calc").write_text(
        "# count the ticks\nk = k + 1\nimg = cam * k   # a new image: a stream\nm = imean(img); avg = m\n"
    )
    res = milk_bash(
        HELPERS
        + 'milk-fpsexec-milk-calc -n u3 -p "k:int=0 avg:float=0" -f prog.calc fpsinit >/dev/null\n'
        "milk-fpsexec-milk-calc -n u3 -loopd 0.02 run >run.log 2>&1 & pid=$!\n"
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
    """One-time .unit.init / .unit.initfile runs before frame 0, is preserved across live .unit.expr
    changes, re-runs when .unit.init changes live, and provides read-only _n, _t, _dt frame variables.
    """
    (tmp_path / "init.calc").write_text("acc = cam * 0; k0 = 10\n")
    res = milk_bash(
        HELPERS
        + 'milk-fpsexec-milk-calc -n u4 -e "acc = acc + cam; fn = _n; ft = _t; fdt = _dt" '
        '-t cam -p "k0:int=0 fn:int=-1 ft:float=-1 fdt:float=-1" -I init.calc fpsinit >/dev/null\n'
        "milk-fpsexec-milk-calc -n u4 run >run.log 2>&1 & pid=$!\n"
        "until_file acc; sleep 0.3\n"
        'echo "f0=$(@get s.acc.mean),$(@get f.u4.k0),$(@get f.u4.fn)"\n'
        'frame 3; echo "f1=$(@get s.acc.mean),$(@get f.u4.fn)"\n'
        '@set f.u4.unit.expr="acc = acc + cam * 2; fn = _n"; frame 5\n'
        'echo "f2=$(@get s.acc.mean),$(@get f.u4.fn)"\n'
        '@set f.u4.unit.initfile="" f.u4.unit.init="acc = cam * 0 + 100"; frame 2\n'
        'echo "f3=$(@get s.acc.mean),$(@get f.u4.fn)"\n'
        "stop $pid\n"
        'milk-fpsexec-milk-calc -n ubad -e "_n = 5" exec >/dev/null 2>&1; echo "ro=$(@get f.ubad.unit.error)"',
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
        'milk-fpsexec-milk-calc -n u5 -t cam -e "out5 = cam + cal" fpsinit >/dev/null\n'
        "milk-fpsexec-milk-calc -n u5 run >run.log 2>&1 & pid=$!\n"
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
        'milk-fpsexec-milk-calc -n u6 -e "outm = matmul(mA, vB); outt = transpose(mA)" exec >/dev/null\n'
        'echo "unit=$(@get s.outm.tot),$(@get s.outt.xsize)x$(@get s.outt.ysize)"',
        tmp_path,
    )
    got = [value(res.stdout, k) for k in ("shapes", "tots", "unit")]
    assert got == ["2x3,3x3", "360,360,261", "360,2x3"], res.stdout + res.stderr


def test_E8_calc_values_set_declared_parameters(tmp_path):
    """The values after the command word set the parameters -p declares (in order, or .name=value,
    negative numbers included); -e on an existing FPS updates it, -p cannot change; a bad, unknown or
    extra value is a usage error that changes nothing; -h lists only the declared parameters (not .unit.*).
    """
    res = milk_bash(
        HELPERS + 'C="milk-fpsexec-milk-calc -n v1"\n'
        'milk-fpsexec-milk-calc -hm | grep -c "none declared" | xargs -I{} echo "empty_h={}"\n'
        '$C -t cam -p "gain:float=1 off:float=0 n:int=0" -e "out = cam * gain + off" fpsinit '
        '3.5 .n=-4 >/dev/null; echo "init=$?"\n'
        'echo "p0=$(@get f.v1.gain),$(@get f.v1.off),$(@get f.v1.n)"\n'
        '$C set . -1.5 >/dev/null; echo "p1=$(@get f.v1.gain),$(@get f.v1.off)"\n'
        '$C set 1 2 3 4 >/dev/null 2>err1; echo "many=$?"\n'
        '$C set .gain=abc >/dev/null 2>err2; echo "bad=$?"\n'
        '$C set .nosuch=1 >/dev/null 2>err3; echo "unknown=$?"\n'
        'echo "p2=$(@get f.v1.gain),$(@get f.v1.off),$(@get f.v1.n)"\n'
        '$C -e "out = cam - off" set >/dev/null; echo "expr=$(@get f.v1.unit.expr)"\n'
        '$C -p "k:int=1" set >/dev/null 2>err4; echo "redeclare=$?"\n'
        '$C -hm | sed -n "/^Parameters:/,/^See Also:/p" > params_sec\n'
        'grep -E "^ +[0-9]+ (gain|off|n) " params_sec | tr -s " " | tr "\\n" "|"; echo\n'
        'echo "unit_in_params=$(grep -cE "unit\\.expr| trigger " params_sec)"\n'
        "cat err1 err2 err3 err4",
        tmp_path,
    )
    got = [
        value(res.stdout, k)
        for k in ("empty_h", "init", "p0", "p1", "many", "bad", "unknown", "p2")
    ]
    assert got == ["1", "0", "3.5,0,-4", "3.5,-1.5", "2", "2", "2", "3.5,-1.5,-4"], (
        res.stdout + res.stderr
    )
    assert value(res.stdout, "expr") == "out = cam - off", res.stdout
    assert value(res.stdout, "redeclare") == "2", res.stdout
    assert value(res.stdout, "unit_in_params") == "0", res.stdout
    assert (
        " 0 gain FLOAT64 3.5" in res.stdout and " 2 n INT64 -4" in res.stdout
    ), res.stdout
    assert (
        "too many values, from '4' on: gain off n take positional values" in res.stdout
    )
    assert "invalid value 'abc' for parameter 'gain': expected a number" in res.stdout
    assert "no parameter 'nosuch'" in res.stdout
    assert "run fpsinit to declare other parameters" in res.stdout


def test_fpsexec_values_by_name_negative_and_errors(tmp_path):
    """Any fpsexec: positional values fill the indexed parameters not set by name, .name=value sets any
    parameter, negative numbers are values, -loopd's value is not a positional value, and errors
    (status 2) change nothing."""
    res = milk_bash(
        "X=milk-fpsexec-clitest\n"
        '$X -n vx fpsinit -12 .p_float64=-2.5 .p_onoff=on >/dev/null; echo "init=$?"\n'
        'echo "a=$(@get f.vx.p_int32),$(@get f.vx.p_float64),$(@get f.vx.p_onoff)"\n'
        '$X -n vx set .p_int32=5 camX >/dev/null; echo "b=$(@get f.vx.p_int32),$(@get f.vx.p_streamname)"\n'
        '$X -n vx set 1 cam 3 >/dev/null 2>&1; echo "many=$?"\n'
        '$X -n vx set .p_uint32=-1 >/dev/null 2>&1; echo "neguint=$?"\n'
        '$X -n vx set -z >/dev/null 2>&1; echo "opt=$?"\n'
        '$X -n vy fpsinit -loopd 0.5 >/dev/null 2>&1; echo "loopd=$? $(@get f.vy.p_int32)"\n',
        tmp_path,
    )
    got = [value(res.stdout, k) for k in ("init", "a", "b", "many", "neguint", "opt")]
    assert got == ["0", "-12,-2.5,ON", "5,camX", "2", "2", "2"], res.stdout + res.stderr
    assert value(res.stdout, "loopd") == "0 123", res.stdout + res.stderr


# ------------------------------------------------------------- E8a: milk.script


def test_E8_script_runs_per_frame_follows_cmd_and_stops_on_failure(tmp_path):
    res = milk_bash(
        HELPERS
        + "milk-fpsexec-milk-script -n s1 -t cam -c '@get s.cam.mean >> log' fpsinit >/dev/null\n"
        "milk-fpsexec-milk-script -n s1 run >run.log 2>&1 & pid=$!\n"
        "for _ in $(seq 50); do @calc 'cam = 1'; sleep 0.1; [ -s log ] && break; done; sleep 0.3; : > log\n"
        'frame 3; frame 4; echo "log=$(tr "\\n" , < log)"\n'
        "@set f.s1.unit.cmd='echo tick >> log2'; frame 5; frame 6; echo \"log2=$(tr '\\n' , < log2)\"\n"
        "@set f.s1.unit.cmd='false'; frame 7\n"
        "for _ in $(seq 40); do kill -0 $pid 2>/dev/null || break; sleep 0.05; done\n"
        'echo "exited=$(kill -0 $pid 2>/dev/null && echo no || echo yes)"; stop $pid\n'
        'echo "error=$(@get f.s1.unit.error)"',
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
        + "milk-fpsexec-milk-script -n s2 -loopd 0.05 -c 'sleep 30' exec >run.log 2>&1 & pid=$!\n"
        "for _ in $(seq 100); do sh=$(pgrep -P $pid -x bash) && break; sleep 0.05; done\n"
        "sleep 0.3; t0=$EPOCHREALTIME; stop $pid\n"
        'echo "ms=$(( (${EPOCHREALTIME/./} - ${t0/./}) / 1000 ))"\n'
        'echo "shell=$(kill -0 $sh 2>/dev/null && echo alive || echo gone) error=[$(@get f.s2.unit.error)]"',
        tmp_path,
    )
    assert value(res.stdout, "shell") == "gone error=[]", res.stdout + res.stderr
    assert int(value(res.stdout, "ms") or 9999) < 2000, res.stdout


def test_E8_calc_reshape_expi_vec_and_casts(tmp_path):
    """vec(), reshape() (including -1 and matmul), expi(), clamp(), and explicit datatype casts."""
    res = milk_bash(
        HELPERS
        + '@calc "v = vec(1, 2, 3, 4, 5, 6); r2 = reshape(v, 3, 2); r1 = reshape(v, -1, 3)"\n'
        'echo "vshape=$(@get l.v.naxis):$(@get l.v.xsize):$(@get l.v.tot)"\n'
        'echo "rshape=$(@get l.r2.xsize)x$(@get l.r2.ysize),$(@get l.r1.xsize)x$(@get l.r1.ysize)"\n'
        '@calc -v p21 "r2[2,1]"; echo "p21=$p21"\n'
        '@calc "phi = vec(0, 1.5707963267948966); z = expi(phi); mag = abs(z); zr = real(z); zi = imag(z)"\n'
        '@calc -v m0,r0,i1 "round(mag[0]); round(zr[0]); round(zi[1])"\n'
        'echo "expi=$(@get l.z.dtype):$m0,$r0,$i1"\n'
        '@calc -v c_u8,c_i8,c_u16,c_cl "u8(300); i8(-200); u16(70000); clamp(5, 0, 3)"\n'
        'echo "scast=$c_u8,$c_i8,$c_u16,$c_cl"\n'
        '@calc "im_u16 = u16(cam * 10 + 3); im_f64 = f64(cam + 0.5); im_i8 = i8(cam - 10)"\n'
        'echo "imcast=$(@get l.im_u16.dtype):$(@get l.im_u16.mean),'
        "$(@get l.im_f64.dtype):$(@get l.im_f64.mean),"
        '$(@get l.im_i8.dtype):$(@get l.im_i8.mean)"\n'
        'mem.mk2Dim "@S:mR" 4 2 >/dev/null; mem.mk2Dim "@S:vR" 2 1 >/dev/null\n'
        '@calc "mR = 2; vR = 3"\n'
        'milk-fpsexec-milk-calc -n u7 -e "out_r = reshape(matmul(mR, vR), 2, 2); out_v = reshape(mR, 8)" exec >/dev/null\n'
        'echo "u7=$(@get s.out_r.xsize)x$(@get s.out_r.ysize):$(@get s.out_r.tot),$(@get s.out_v.xsize):$(@get s.out_v.naxis)"',
        tmp_path,
    )
    got = [
        value(res.stdout, k)
        for k in ("vshape", "rshape", "p21", "expi", "scast", "imcast", "u7")
    ]
    assert got == [
        "1:6:21",
        "3x2,2x3",
        "6",
        "COMPLEX_FLOAT:1,1,1",
        "44,56,4464,3",
        "UINT16:23,DOUBLE:2.5,INT8:-8",
        "2x2:48,8:1",
    ], (
        res.stdout + res.stderr
    )


def test_E8_calc_fused_fastmath_and_zero_alloc_reductions(tmp_path):
    """Fused fast-math opcodes (^2, ^0.5, ^3, ^-1, min, max, clamp, round) and zero-alloc reductions."""
    res = milk_bash(
        HELPERS + '@calc "a = reshape(vec(1, 4, 9, 16), 2, 2)"\n'
        '@calc "fm = (a^0.5)^2 + a^3 - 144*(a^-1) + min(a, 5) + max(a, 10) + clamp(a, 2, 8) + round(a * 0.5)"\n'
        'echo "fm=$(@get l.fm.tot)"\n'
        '@calc "ax0 = itot(a, 0); ax1 = imean(a, 1); msk = a > 2"\n'
        '@calc -v mtot,mmean3,mmin,mmax "itot(a, msk); round(imean(a, msk) * 3); imin(a, msk); imax(a, msk)"\n'
        '@calc -v a00,a01,a10,a11 "ax0[0]; ax0[1]; ax1[0]; ax1[1]"\n'
        'echo "red=$a00,$a01,$a10,$a11|$mtot,$mmean3,$mmin,$mmax"\n'
        '@calc "au16 = u16(a); axu = itot(au16, 0)"\n'
        '@calc -v u0,u1,utot "axu[0]; axu[1]; itot(au16, msk)"\n'
        'echo "u16red=$u0,$u1,$utot"',
        tmp_path,
    )
    got = [value(res.stdout, k) for k in ("fm", "red", "u16red")]
    assert got == [
        "4814",
        "5,25,5,10|29,29,4,16",
        "5,25,29",
    ], (
        res.stdout + res.stderr
    )


def test_E8_calc_stream_params(tmp_path):
    """milk.calc supports stream parameters (in:stream=..., out:stream=...) with live rebinding."""
    res = milk_bash(
        HELPERS + 'mem.mk2Dim "@S:camB" 4 4 >/dev/null; @calc "camB = 10"\n'
        'milk-fpsexec-milk-calc -n u8 -t cam -p "in:stream=cam out:stream=outA gain:float=3" -e "out = in * gain" fpsinit >/dev/null\n'
        "milk-fpsexec-milk-calc -n u8 run >run.log 2>&1 & pid=$!\n"
        "until_file outA; sleep 0.3\n"
        'frame 2; echo "a=$(@get s.outA.mean)"\n'
        "@set f.u8.in=camB f.u8.out=outB\n"
        "sleep 0.1; frame 2\n"
        'until_file outB; echo "b=$(@get s.outB.mean)"; stop $pid',
        tmp_path,
    )
    got = [value(res.stdout, k) for k in ("a", "b")]
    assert got == ["6", "30"], (
        res.stdout + res.stderr + (tmp_path / "run.log").read_text()
    )


def test_E8_script_params_file_and_args_types(tmp_path):
    """milk.script supports .params, .file, and _n; @args supports stream and fits types."""
    (tmp_path / "tick.milk").write_text('echo "$_n:$gain:$mode:$in" >> tick.log\n')
    res = milk_bash(
        HELPERS
        + "fn() { @args 's:stream:in stream' 'f:fits=cal.fits:fits file' -- \"$@\" || return; echo \"arg=$s,$f\"; }\n"
        "fn my_cam; fn my_cam flat.fits; fn 'bad stream' 2>/dev/null; echo \"badrc=$?\"\n"
        "milk-fpsexec-milk-script -n s3 -t cam -p 'gain:float=2.5 mode:str=fast in:stream=cam' "
        "-f tick.milk fpsinit >/dev/null\n"
        "milk-fpsexec-milk-script -n s3 run >run.log 2>&1 & pid=$!\n"
        "for _ in $(seq 50); do @calc 'cam = 1'; sleep 0.1; [ -s tick.log ] && break; done; sleep 0.2; : > tick.log\n"
        "frame 2\n"
        "@set f.s3.gain=7.5 f.s3.mode=accurate\n"
        "sleep 0.1; frame 3; stop $pid\n"
        'echo "ticks=$(tr "\\n" "|" < tick.log)"',
        tmp_path,
    )
    assert re.findall(r"^arg=(.*)$", res.stdout, re.M) == [
        "my_cam,cal.fits",
        "my_cam,flat.fits",
    ], (
        res.stdout + res.stderr
    )
    assert value(res.stdout, "badrc") == "2", res.stdout + res.stderr
    ticks = [t for t in (value(res.stdout, "ticks") or "").split("|") if t]
    assert len(ticks) >= 2, res.stdout + res.stderr + (tmp_path / "run.log").read_text()
    assert ticks[0].endswith(":2.5:fast:cam"), ticks
    assert ticks[-1].endswith(":7.5:accurate:cam"), ticks
    n0 = int(ticks[0].split(":")[0])
    n1 = int(ticks[-1].split(":")[0])
    assert n1 > n0 >= 0, ticks


# ------------------------------------------------------------- E2: sessions

# other CMD: run CMD in another shell, on its own session; sstate [NAME]: the State word of
# @session status; srvpid [NAME]: its server's PID
SESSION_FUNCS = r"""
other() { env -u MILKCLI_SOCK -u _MILK_SERVER_NAME bash --norc -c "source \"\$MILK_INSTALLDIR/bin/milk.bash\"; $1"; }
sstate() { @session status "$@" | sed -n 's/ *State *: \([A-Z]*\).*/\1/p'; }
srvpid() { @session status "$@" | sed -n 's/.*(PID \([0-9]*\),.*/\1/p'; }
"""


@pytest.fixture
def sessions():
    """sessions(): a session name of this test's own; the sessions are ended after the test."""
    made: list[str] = []

    def make() -> str:
        made.append(f"e2{uuid.uuid4().hex[:8]}")
        return made[-1]

    yield make
    if made:
        milk_bash(
            "".join(f"@session end {n} >/dev/null 2>&1\n" for n in made), timeout=60
        )


def test_E2_new_use_leave_and_what_a_session_keeps(tmp_path, sessions):
    """D35: new fails if the session runs, use if it does not, leave returns to the shell's own session.
    D34: a name is [A-Za-z0-9_-] and not a number (a shell's own session). The images and calculator
    variables are the server's: another shell that uses the session sees them."""
    n = sessions()
    res = milk_bash(
        SESSION_FUNCS + f"""
@session new 123 2>err.log; echo "num=$?"
@session new a.b 2>>err.log; echo "bad=$?"
@session use {n} 2>>err.log; echo "missing=$?"
@session new {n} >/dev/null; echo "new=$? name=$_MILK_SERVER_NAME"
@session new {n} 2>>err.log; echo "again=$?"
@calc "v = 42"; mem.mk2Dim im 4 4 >/dev/null; @calc "im = 3"
@session leave >/dev/null; echo "left=$? name=[$_MILK_SERVER_NAME]"
[ "$MILKCLI_SOCK" = "$_MILK_LOCAL_SOCK" ] && echo "own=1"
@calc v >/dev/null 2>&1; echo "own_v=$?"
echo "other=$(other '@session use {n} >/dev/null; echo $(@calc v),$(@calc "imean(im)")')"
@session use {n} >/dev/null; echo "back=$(@calc v) name=$_MILK_SERVER_NAME"
""",
        tmp_path,
    )
    err = (tmp_path / "err.log").read_text()
    out = res.stdout + res.stderr + err
    assert value(res.stdout, "num") == "2", out
    assert value(res.stdout, "bad") == "2", out
    assert value(res.stdout, "missing") == "1", out
    assert value(res.stdout, "new") == f"0 name={n}", out
    assert value(res.stdout, "again") == "1", out
    assert value(res.stdout, "left") == "0 name=[]", out
    assert value(res.stdout, "own") == "1", out
    assert value(res.stdout, "own_v") != "0", out
    assert value(res.stdout, "other") == "42,3", out
    assert value(res.stdout, "back") == f"42 name={n}", out
    assert "'123' is a number, the name of a shell's own session" in err, out
    assert "invalid name 'a.b'" in err, out
    assert f"no session {n} runs (@session new {n})" in err, out
    assert f"session {n} exists (@session use {n})" in err, out


def test_E2_an_ended_session_is_not_restarted(tmp_path, sessions):
    """D34 (user decision 2026-10-26): only @session new starts a named session. A shell whose session
    another shell ended gets a hint, and nothing restarts it; leave gets the shell going again.
    """
    n = sessions()
    res = milk_bash(
        SESSION_FUNCS + f"""
@session new {n} >/dev/null
other '@session end {n}' >/dev/null 2>&1
mem.mk2Dim im 4 4 >/dev/null 2>err.log; echo "rc=$?"
echo "state=$(sstate {n})"
@session leave >/dev/null; mem.mk2Dim im 4 4 >/dev/null; echo "own=$?"
""",
        tmp_path,
    )
    err = (tmp_path / "err.log").read_text()
    out = res.stdout + res.stderr + err
    assert value(res.stdout, "rc") not in (None, "0"), out
    assert (
        f"milk: session {n} has ended: @session new {n}, or @session leave" in err
    ), out
    assert "unable to connect" not in err, out
    assert value(res.stdout, "state") == "ENDED", out
    assert value(res.stdout, "own") == "0", out


def test_E2_who_and_ls(tmp_path, sessions):
    """D35: who lists the shells that use a session; ls shows each session with its state and clients,
    the current one marked, and --json the same fields."""
    n = sessions()
    res = milk_bash(
        SESSION_FUNCS + f"""
@session new {n} >/dev/null
env -u MILKCLI_SOCK -u _MILK_SERVER_NAME bash --norc -c \\
    'source "$MILK_INSTALLDIR/bin/milk.bash" --session {n}; echo up > up.txt; sleep 30' & child=$!
for _ in $(seq 200); do [ -s up.txt ] && break; sleep 0.05; done
echo "me=$$ child=$child"
@session who {n} > who.txt
echo "ls=$(@session ls --json)"
@session ls > ls.txt
kill $child
""",
        tmp_path,
    )
    who = (tmp_path / "who.txt").read_text()
    out = res.stdout + res.stderr + who
    me, child = re.findall(r"^me=(\d+) child=(\d+)$", res.stdout, re.M)[0]
    pids = sorted(line.split()[0] for line in who.splitlines()[1:])
    assert pids == sorted([me, child]), out
    entry = [e for e in json.loads(value(res.stdout, "ls")) if e["name"] == n]
    assert len(entry) == 1, out
    e = entry[0]
    assert (e["active"], e["state"], e["kind"], e["clients"], e["units"]) == (
        True,
        "RUNNING",
        "named",
        2,
        0,
    ), out
    row = [
        line.split()
        for line in (tmp_path / "ls.txt").read_text().splitlines()
        if f" {n} " in line
    ]
    assert row and row[0][:3] == ["*", n, "RUNNING"], out


def test_E2_units_are_listed_with_their_session_and_end_units_stops_them(
    tmp_path, sessions
):
    """D35 and D37 (§9.4): a unit records the session of the shell that started it; status and ls count
    it; end leaves it running, end --units stops it, also once the session has ended."""
    n = sessions()
    res = milk_bash(
        SESSION_FUNCS + f"""
mem.mk2Dim "@S:cam" 4 4 >/dev/null
@session new {n} >/dev/null
milk-fpsexec-milk-calc -n su1 -t cam -e "so1 = cam * 2" fpsinit >/dev/null
milk-fpsexec-milk-calc -n su1 runstart >start.log 2>&1; echo "start=$?"
echo "rec=$(grep '^session=' "$MILK_SHM_DIR/su1.fps.d/run.info" | tail -1)"
echo "status=$(@session status | sed -n 's/ *Units *: //p')"
echo "ls=$(@session ls --json)"
@session end {n} >/dev/null; echo "kept=$(@get f.su1.status) name=[$_MILK_SERVER_NAME]"
echo "ended=$(@session end {n} --units 2>&1 | grep '^Session')"
echo "stopped=$(@get f.su1.status)"
@session new {n} >/dev/null
milk-fpsexec-milk-calc -n su1 runstart >/dev/null 2>&1; echo "restart=$?"
@session end {n} --units >end.log 2>&1; echo "end=$?"
echo "stopped2=$(@get f.su1.status) name=[$_MILK_SERVER_NAME]"
milk-fps-runstop su1 >/dev/null 2>&1
""",
        tmp_path,
        timeout=60,
    )
    out = res.stdout + res.stderr
    for log in ("start.log", "end.log"):
        if (tmp_path / log).exists():
            out += f"\n--- {log}\n" + (tmp_path / log).read_text()
    assert value(res.stdout, "start") == "0", out
    assert value(res.stdout, "rec") == f"session={n}", out
    assert value(res.stdout, "status") == "1", out
    entry = [e for e in json.loads(value(res.stdout, "ls")) if e["name"] == n]
    assert entry and entry[0]["units"] == 1, out
    assert value(res.stdout, "kept") == "running name=[]", out
    assert (
        value(res.stdout, "ended") == f"Session {n} had ended; units stopped: 1."
    ), out
    assert value(res.stdout, "stopped") in ("exited", "stopped"), out
    assert value(res.stdout, "restart") == "0", out
    assert value(res.stdout, "end") == "0", out
    assert f"Session {n} ended." in (tmp_path / "end.log").read_text(), out
    assert value(res.stdout, "stopped2") in ("exited name=[]", "stopped name=[]"), out


def test_E2_prune(tmp_path):
    """D35: prune removes what ended sessions and exited shells left: the logs of dead servers, the
    servers of shells that are gone, and the records of exited shells."""
    n = f"e2{uuid.uuid4().hex[:8]}"
    res = milk_bash(
        SESSION_FUNCS + f"""
sleep 0 & d=$!; wait $d
printf '@milk-cli-%s-x' "$UID" > "$TMPDIR/.milk_active_sock_$UID""_$d"
@session new {n} -d >/dev/null; p=$(srvpid {n}); kill -KILL $p
for _ in $(seq 100); do kill -0 $p 2>/dev/null || break; sleep 0.05; done
milk-cli -q --server "@milk-cli-$UID-$d" </dev/null >orphan.log 2>&1 &
for _ in $(seq 200); do [ "$(sstate $d)" = ORPHANED ] && break; sleep 0.05; done
echo "before=$(sstate $d)"
echo "pruned=$(@session prune)"
echo "after=$(sstate $d)"
[ -e "$TMPDIR/.milk_active_sock_$UID""_$d" ] && echo "record=1" || echo "record=0"
echo "files=$(ls "$MILK_SHM_DIR" | grep -c '^milk-srv\\.')"
""",
        tmp_path,
        extra_env={"TMPDIR": str(tmp_path)},
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "before") == "ORPHANED", out  # its shell has gone (E3)
    assert value(res.stdout, "pruned") == (
        "Pruned ended-session logs: 1, orphaned servers: 1, exited-shell records: 1."
    ), out
    assert value(res.stdout, "after") == "IDLE", out
    assert value(res.stdout, "record") == "0", out
    assert value(res.stdout, "files") == "0", out


def test_E2_milk_bash_session_option(tmp_path, sessions):
    """D34: `source milk.bash --session NAME` uses session NAME, started if it does not run; another
    shell resumes it. -s is not milk's any more: a script's own -s is left alone."""
    n = sessions()
    (tmp_path / "join.sh").write_text(
        'source "$MILK_INSTALLDIR/bin/milk.bash" "$@"\n'
        "echo \"$_MILK_SERVER_NAME $(@session status | sed -n 's/.*(PID \\([0-9]*\\),.*/\\1/p')\"\n"
    )
    res = milk_bash(
        f"""
join() {{ env -u MILKCLI_SOCK -u _MILK_SERVER_NAME bash --norc join.sh "$@"; }}
echo "first=$(join --session {n})"
echo "second=$(join --session={n})"
echo "short=[$(join -s {n})]"
""",
        tmp_path,
    )
    out = res.stdout + res.stderr
    first = value(res.stdout, "first") or ""
    assert re.fullmatch(rf"{n} \d+", first), out
    assert value(res.stdout, "second") == first, out
    assert value(res.stdout, "short") == "[ ]", out


def test_E2_old_spellings_give_hints(tmp_path):
    """D33 (user decision 2026-10-26): @server, its verbs, @log --server, milk.bash --server and
    milk-call server fail with the @session form; there is no --modernize rewrite."""
    res = milk_bash(
        """
@server ls 2>s.log; echo "server=$?"
@session start x 2>v.log; echo "verb=$?"
@log --server x 2>l.log; echo "log=$?"
bash --norc -c 'source "$MILK_INSTALLDIR/bin/milk.bash" --server x' 2>b.log; echo "bash=$?"
milk-call server ls 2>c.log; echo "call=$?"
""",
        tmp_path,
    )
    logs = {f: (tmp_path / f"{f}.log").read_text() for f in "svlbc"}
    out = res.stdout + res.stderr + repr(logs)
    assert value(res.stdout, "server") == "127", out
    assert (
        "milk: @server was removed; use: @session (start -> new or use, list -> ls, local -> leave, "
        "stop -> end, attach -> log -f)" in logs["s"]
    ), out
    assert value(res.stdout, "verb") == "2", out
    assert "@session: start is @session new <name> (or use <name>)" in logs["v"], out
    assert value(res.stdout, "log") == "2", out
    assert "--server was removed; use: --session <name>" in logs["l"], out
    assert value(res.stdout, "bash") == "2", out
    assert (
        "milk.bash: --server was removed; use: source milk.bash --session NAME"
        in logs["b"]
    ), out
    assert value(res.stdout, "call") == "2", out
    assert "milk-call: server was removed; use: session (@session)" in logs["c"], out


def test_E2_end_reports_a_session_that_does_not_end(tmp_path, sessions):
    """A server that ignores its shutdown command and cannot be signalled from here (the identity to
    signal is in its log, in another MILK_SHM_DIR) is not reported as ended: status 1.
    """
    n = sessions()
    res = milk_bash(
        SESSION_FUNCS + f"""
@session new {n} -d >/dev/null; p=$(srvpid {n}); kill -STOP $p
mkdir other; MILK_SHM_DIR="$PWD/other" @session end {n} >end.out 2>end.log; echo "end=$?"
kill -CONT $p; @session end {n} >/dev/null 2>&1; echo "after=$?"
""",
        tmp_path,
        timeout=60,
    )
    log = (tmp_path / "end.log").read_text()
    out = res.stdout + res.stderr + log
    assert value(res.stdout, "end") == "1", out
    assert "still runs: its log is in another MILK_SHM_DIR" in log, out
    assert "ended" not in (tmp_path / "end.out").read_text(), out
    assert value(res.stdout, "after") == "0", out


def _read_until(stream, text: str, timeout: float) -> str:
    """Read the text pipe @stream until @text shows up, at most @timeout s."""
    buf = ""
    fd = stream.fileno()
    end = time.monotonic() + timeout
    while text not in buf and time.monotonic() < end:
        if select.select([fd], [], [], 0.1)[0]:
            chunk = os.read(fd, 4096).decode(errors="replace")
            if not chunk:
                break
            buf += chunk
    return buf


def test_E2_end_ends_a_milk_cli_session(tmp_path):
    """D35: @session end works on a milk-cli session, whose shutdown command milk-cli used to ignore:
    milk-cli exits. Inside milk-cli, its own session ends with milk-cli (exit)."""
    env, shm_dir = _isolated_shm_env()
    try:
        with milk_proc([], env=env, stdin=subprocess.PIPE) as m:
            m.proc.stdin.write("@session end\necho ready=1\n")
            m.proc.stdin.flush()
            out = _read_until(m.proc.stdout, "ready=1", 20)
            res = milk_bash(
                f'@session end {m.proc.pid}; echo "end=$?"',
                tmp_path,
                extra_env={
                    "MILK_SHM_DIR": env["MILK_SHM_DIR"]
                },  # where milk-cli's log is
            )
            rc = m.proc.wait(timeout=20)
            out += m.proc.stdout.read()
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)
    out += res.stdout + res.stderr
    assert "milk-cli's session ends with milk-cli (exit)" in out, out
    assert value(res.stdout, "end") == "0", out
    assert rc == 0, out
    assert "milk-cli: this session was ended (@session end)" in out, out


def test_E2_end_lets_a_busy_milk_cli_finish_its_command(tmp_path):
    """A milk-cli session ended while it runs a command line says so, finishes the line, and exits at its
    prompt: input that arrived meanwhile does not run. (Piped lines that arrive together run as one
    batch.)"""
    env, shm_dir = _isolated_shm_env()
    try:
        with milk_proc([], env=env, stdin=subprocess.PIPE) as m:
            m.proc.stdin.write("echo ready=1; sleep 5\n")
            m.proc.stdin.flush()
            out = _read_until(m.proc.stdout, "ready=1", 20)
            m.proc.stdin.write("echo after=1\n")  # arrives while the line runs
            m.proc.stdin.flush()
            res = milk_bash(
                f'@session end {m.proc.pid}; echo "end=$?"',
                tmp_path,
                extra_env={
                    "MILK_SHM_DIR": env["MILK_SHM_DIR"]
                },  # where milk-cli's log is
            )
            rc = m.proc.wait(timeout=20)
            out += m.proc.stdout.read()
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)
    out += res.stdout + res.stderr
    assert value(res.stdout, "end") == "0", out
    assert (
        "still runs: milk-cli's session ends when its command returns" in res.stderr
    ), out
    assert "Session" not in res.stdout, out  # not reported as ended
    assert rc == 0, out
    assert "this session was ended (@session end)" in out, out
    assert "after=1" not in out, out


# ------------------------------------------------------------- E3: owners and lifetimes


def test_E3_owner_and_orphaned_sessions(tmp_path, sessions):
    """D37: the shell that runs `@session new` owns the session without being tied to it: once that
    shell has exited, ls and status list the session as orphaned."""
    a, b = sessions(), sessions()
    res = milk_bash(
        SESSION_FUNCS + f"""
other '@session new {a} -d >/dev/null; echo "creator=$$"'
@session new {b} -d >/dev/null; echo "me=$$"
echo "ls=$(@session ls --json)"
@session status {a} > status.txt
""",
        tmp_path,
    )
    status = (tmp_path / "status.txt").read_text()
    out = res.stdout + res.stderr + status
    creator, me = int(value(res.stdout, "creator")), int(value(res.stdout, "me"))
    got = {
        e["name"]: (e["state"], e["owner"]) for e in json.loads(value(res.stdout, "ls"))
    }
    assert got[a] == ("ORPHANED", creator), out
    assert got[b] == ("RUNNING", me), out
    assert "State     : ORPHANED" in status, out
    assert f"Owner     : {creator} (exited: the session is orphaned)" in status, out


def test_E3_idle_sessions_end_once_unused(tmp_path, sessions):
    """D37 (user decision 2026-10-26): `--idle T` ends a session once it has had no command for T and
    no shell uses it; a shell that uses it holds it, also while it waits on its own; bad durations
    are refused."""
    a, b = sessions(), sessions()
    res = milk_bash(
        SESSION_FUNCS + f"""
srvout() {{ @session status "$1" | sed -n 's/ *Output *: //p'; }}
@session new {a} -d --idle 1 >/dev/null; oa=$(srvout {a})
@session new {b} --idle 1 >/dev/null
echo "idle=$(@session status {b} | sed -n 's/ *Idle end *: //p')"
@session new x --idle 0 2>/dev/null; echo "zero=$?"
@session new x --idle 1w 2>/dev/null; echo "unit=$?"
@waitfor "1 > 2" -t 3 >/dev/null 2>&1
echo "a=$(sstate {a}) b=$(sstate {b})"
echo "said=$(grep -c 'no command for 1 s, and no shell uses it (--idle)' "$oa")"
@session leave >/dev/null
for _ in $(seq 100); do [ "$(sstate {b})" = ENDED ] && break; sleep 0.1; done
echo "left=$(sstate {b})"
""",
        tmp_path,
        timeout=60,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "idle") == "after 1s without a command", out
    assert value(res.stdout, "zero") == "2", out
    assert value(res.stdout, "unit") == "2", out
    assert value(res.stdout, "a") == "ENDED b=RUNNING", out
    assert value(res.stdout, "said") == "1", out
    assert value(res.stdout, "left") == "ENDED", out


def test_E3_watchers_end_with_their_shell(tmp_path):
    """D37 (user decision 2026-10-26): @on watchers end with the shell that started them, also when it
    is killed, and no longer keep its own session alive."""
    (tmp_path / "w.sh").write_text(
        'source "$MILK_INSTALLDIR/bin/milk.bash"\n'
        'mem.mk2Dim "@S:wcam" 2 2 >/dev/null\n'
        "@on --every 0.3 true\n"
        "@on s.wcam true\n"
        "srv=$(@session status | sed -n 's/.*(PID \\([0-9]*\\),.*/\\1/p')\n"
        'echo "$$ $srv ${_MILK_ON_PIDS[*]}" >> ids.txt\n'
        '[ "${1:-}" = kill ] && kill -KILL $$\n'
        "exit 0\n"
    )
    res = milk_bash(
        """
gone() { for _ in $(seq 100); do kill -0 "$1" 2>/dev/null || return 0; sleep 0.05; done; return 1; }
for how in exit kill; do
    env -u MILKCLI_SOCK -u _MILK_SERVER_NAME bash --norc w.sh $how </dev/null >/dev/null 2>&1
done
while read -r shell srv w1 w2; do
    echo "left=$(gone $srv; echo -n $?)$(gone $w1; echo -n $?)$(gone $w2; echo -n $?)"
done < ids.txt
""",
        tmp_path,
    )
    ids = (tmp_path / "ids.txt").read_text()
    out = res.stdout + res.stderr + ids
    assert [len(line.split()) for line in ids.splitlines()] == [
        4,
        4,
    ], out  # shell, server, 2 watchers
    assert re.findall(r"^left=(\d+)$", res.stdout, re.M) == ["000", "000"], out


def test_E3_override_files_are_named_after_the_shell_start(tmp_path):
    """D37: a shell records the session it uses in a file named after its PID and start time; a file
    that an earlier process with the same PID left is not read, and prune removes it."""
    n = f"e3{uuid.uuid4().hex[:8]}"
    res = milk_bash(
        SESSION_FUNCS + f"""
read -r l < /proc/$$/stat; read -r -a f <<< "${{l##*) }}"
@session new {n} >/dev/null
[ -f "$TMPDIR/.milk_active_sock_${{UID}}_$$_${{f[19]}}" ] && echo "named=1"
printf '@milk-cli-%s-{n}' "$UID" > "$TMPDIR/.milk_active_sock_${{UID}}_$$_1"
@session leave >/dev/null
source "$MILK_INSTALLDIR/bin/milk.bash"; echo "after=[$_MILK_SERVER_NAME]"
echo "who=$(@session who {n} | tail -n +2 | wc -l)"
echo "pruned=$(@session prune)"
[ -e "$TMPDIR/.milk_active_sock_${{UID}}_$$_1" ] && echo "stale=1" || echo "stale=0"
@session end {n} >/dev/null
""",
        tmp_path,
        extra_env={"TMPDIR": str(tmp_path)},
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "named") == "1", out
    assert value(res.stdout, "after") == "[]", out
    assert value(res.stdout, "who") == "0", out
    assert value(res.stdout, "pruned") == (
        "Pruned ended-session logs: 0, orphaned servers: 0, exited-shell records: 1."
    ), out
    assert value(res.stdout, "stale") == "0", out


# ------------------------------------------------------------- E5: milk-cli --session, save/restore


def test_E5_milk_cli_session_runs_its_commands_in_the_session(tmp_path, sessions):
    """D36 revised (user decision 2026-10-26): `milk-cli --session NAME` starts NAME if it does not
    run and its bash joins it; calc mode evaluates there; @history and the REPL settings of @config
    stay milk-cli's; @session leave returns to milk-cli's own session. NAME outlives milk-cli.
    """
    n = sessions()
    env, shm_dir = _isolated_shm_env()
    try:
        script = (
            '@calc "v = 7"\n'
            "@session status | sed -n 's/ *Name *: /name=/p'\n"
            '@config synhl on >/dev/null; echo "synhl=$?"\n'
            '@history >/dev/null; echo "history=$?"\n'
            "@calc\nv * 2\n@calc\n"
            "@session leave >/dev/null\n"
            '@calc v >/dev/null 2>&1; echo "own_v=$?"\n'
            f'@session use {n} >/dev/null; echo "back=$(@calc v)"\n'
            'echo "pid=$PPID"\n'
        )
        res = _run(
            [MILK_CLI_EXEC, "--session", n], env, 30, stdin=script, cwd=str(tmp_path)
        )
        bad = _run(
            [MILK_CLI_EXEC, "--session", "a.b"], env, 30, stdin="", cwd=str(tmp_path)
        )
        both = _run(
            [MILK_CLI_EXEC, "--session", n, "--server", f"@milk-cli-{os.getuid()}-x"],
            env,
            30,
            stdin="",
            cwd=str(tmp_path),
        )
        after = milk_bash(
            f'@session use {n} >/dev/null; echo "v=$(@calc v)"\necho "ls=$(@session ls --json)"',
            tmp_path,
            extra_env={"MILK_SHM_DIR": env["MILK_SHM_DIR"]},
        )
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)
    out = (
        res.stdout + res.stderr + bad.stderr + both.stderr + after.stdout + after.stderr
    )
    assert value(res.stdout, "name") == f"{n} (this shell's)", out
    assert value(res.stdout, "synhl") == "0", out
    assert value(res.stdout, "history") == "0", out
    assert re.search(
        r"^14$", res.stdout, re.M
    ), out  # evaluated in the session, not in milk-cli
    assert value(res.stdout, "own_v") not in (None, "0"), out
    assert value(res.stdout, "back") == "7", out
    assert (
        bad.returncode == 2 and "milk-cli --session: invalid name 'a.b'" in bad.stderr
    ), out
    assert both.returncode != 0 and "not both" in both.stderr, out
    assert value(after.stdout, "v") == "7", out
    entry = [e for e in json.loads(value(after.stdout, "ls")) if e["name"] == n]
    assert (
        entry and entry[0]["state"] == "ORPHANED"
    ), out  # milk-cli, its creator, has exited


def test_E5_save_and_restore(tmp_path, sessions):
    """D36 revised (user decisions 2026-10-26): @session save writes a milk script and FITS files;
    @session restore runs it in the shell's session, and lists the units without starting them.
    """
    a, b = sessions(), sessions()
    res = milk_bash(
        f"""
@session new {a} >/dev/null
@calc "gain = 0.25"; @calc "n = 12"
mem.mk2Dim "@L:lim" 4 4 >/dev/null; @calc "lim = 3"
mem.mk2Dim "@S:e5in" 2 2 >/dev/null
mem.mk2Dim --default .xsize=16 .ysize=8
@config digits 4 >/dev/null
milk-fpsexec-milk-calc -n e5u -t e5in -e "e5out = e5in * 2" fpsinit >/dev/null
milk-fpsexec-milk-calc -n e5u runstart >/dev/null 2>&1
mkdir sub; @session save sub/s.milk > save.out; echo "save=$?"
@session save 2>/dev/null; echo "usage=$?"
@session new {b} >/dev/null
here=$PWD; cd /tmp
@session restore "$here/sub/s.milk" > "$here/restore.out"; echo "restore=$?"
cd - >/dev/null; [ "$PWD" = "$here" ] && echo "oldpwd=1"
echo "got=$(@calc gain),$(@calc n),$(@calc "imean(lim)")"
echo "digits=$(@config digits)"
echo "call=$(mem.mk2Dim --no-run x)"
@session restore nosuch.milk 2>err1.txt; echo "missing=$?"
@session restore save.out 2>err2.txt; echo "other=$?"
milk-fps-runstop e5u >/dev/null 2>&1
""",
        tmp_path,
        timeout=90,
    )
    saved = (tmp_path / "sub" / "s.milk").read_text()
    out = res.stdout + res.stderr + saved
    for f in ("save.out", "restore.out", "err1.txt", "err2.txt"):
        if (tmp_path / f).exists():
            out += f"\n--- {f}\n" + (tmp_path / f).read_text()
    assert value(res.stdout, "save") == "0", out
    assert value(res.stdout, "usage") == "2", out
    assert saved.startswith(f"# milk session save: {a}, "), out
    assert (
        '@calc "gain = 0.25"' in saved
        and "mem.mk2Dim --default .xsize='16' .ysize='8'" in saved
    ), out
    assert "# stream e5in: shared memory, not saved" in saved, out
    assert "# unit e5u: running" in saved, out
    assert (tmp_path / "sub" / "s.milk.d" / "lim.fits").exists(), out
    assert value(res.stdout, "restore") == "0", out
    assert value(res.stdout, "oldpwd") == "1", out
    assert value(res.stdout, "got") == "0.25,12,3", out
    assert value(res.stdout, "digits") == "digits 4", out
    assert ".xsize=16 .ysize=8" in (value(res.stdout, "call") or ""), out
    assert (
        "Units of the saved session, not started: e5u: running"
        in (tmp_path / "restore.out").read_text()
    ), out
    assert (
        value(res.stdout, "missing") == "1" and value(res.stdout, "other") == "1"
    ), out
    assert "cannot read nosuch.milk" in (tmp_path / "err1.txt").read_text(), out
    assert (
        "is not a file that @session save wrote" in (tmp_path / "err2.txt").read_text()
    ), out


# ------------------------------------------------------------- E4: shared sessions, history


def test_E4_another_shells_command_waits_for_a_long_one(tmp_path, sessions):
    """D38 (user decision 2026-10-26): what another process group sends during a long command waits
    for it, then runs on the state it left, instead of in a fork whose changes are lost; --nowait
    fails at once with "busy"; Ctrl-C gives up a waiting command without running it."""
    n = sessions()
    res = milk_bash(
        rf"""
mem.mk2Dim "@S:e4s" 2 2 >/dev/null; mem.imsetsemflush e4s 0 >/dev/null
@session new {n} -d >/dev/null
other() {{ setsid env -u MILKCLI_SOCK -u _MILK_SERVER_NAME bash --norc -c "source \"\$MILK_INSTALLDIR/bin/milk.bash\"; @session use {n} >/dev/null; $1"; }}
other 'mem.imsetsemwait e4s 0 >/dev/null; echo "a=$(date +%s.%N)"' > a.out 2>&1 &
sleep 0.5
other '@calc "x = 5" >/dev/null; echo "b=$(date +%s.%N)"; echo "x=$(@calc x)"' > b.out 2>&1 &
other 'echo $$ > d.pid; @calc "y = 2" >/dev/null 2>&1; echo "d=$?"' > d.out 2>&1 &
for _ in $(seq 200); do [ -s d.pid ] && break; sleep 0.05; done
sleep 0.5  # its request waits now (its client handles Ctrl-C; bash ignored it in the background)
kill -INT "$(cat d.pid)"
other 'milkcall --nowait mem.listim >/dev/null 2>c.err; echo "c=$?"' > c.out 2>&1
sleep 0.3
echo "posted=$(date +%s.%N)"; mem.imsetsempost e4s 0 >/dev/null
wait
other '@calc y >/dev/null 2>&1; echo "y=$?"' > y.out 2>&1
cat a.out b.out c.out d.out y.out
""",
        tmp_path,
        timeout=60,
    )
    err = (tmp_path / "c.err").read_text() if (tmp_path / "c.err").exists() else ""
    out = res.stdout + res.stderr + err
    assert float(value(res.stdout, "b")) >= float(
        value(res.stdout, "posted")
    ), out  # B waited
    assert value(res.stdout, "x") == "5", out  # and its change is kept
    assert value(res.stdout, "c") == "1", out
    assert "milk: busy: mem.imsetsemwait e4s 0 for" in err, out
    assert value(res.stdout, "d") == "130", out
    assert value(res.stdout, "y") not in (None, "0"), out  # given up: never ran


def _pty_cli(env: dict[str, str], cwd: Path) -> tuple[int, int]:
    pid, fd = pty.fork()
    if pid == 0:
        os.chdir(cwd)
        os.execve(MILK_CLI_EXEC, [MILK_CLI_EXEC], env)
    return pid, fd


def _pty_drain(fd: int, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if select.select([fd], [], [], 0.05)[0]:
            try:
                if not os.read(fd, 65536):
                    return
            except OSError:
                return


def test_E4_history_lines_are_appended_not_rewritten(tmp_path):
    """D39: two milk-cli prompts that run at the same time both keep their lines in ~/.milk_history:
    each line is appended as it runs, and exit no longer rewrites the file from one prompt's list.
    """
    env, shm_dir = _isolated_shm_env()
    env["TERM"] = "dumb"
    procs = []
    try:
        procs = [_pty_cli(env, tmp_path) for _ in range(2)]
        for _, fd in procs:
            _pty_drain(fd, 2.0)
        for (_, fd), line in zip(procs, ("echo line-a", "echo line-b")):
            os.write(fd, (line + "\r").encode())
            _pty_drain(fd, 1.0)
        for _, fd in procs:
            os.write(fd, b"exit\r")
            _pty_drain(fd, 2.0)
        hist = (Path(env["HOME"]) / ".milk_history").read_text()
    finally:
        for pid, fd in procs:
            _kill_session(pid)
            with contextlib.suppress(ChildProcessError):
                os.waitpid(pid, 0)
            os.close(fd)
        shutil.rmtree(shm_dir, ignore_errors=True)
    assert "echo line-a" in hist and "echo line-b" in hist, hist


def test_E4_history_is_per_session_from_any_front_end(tmp_path, sessions):
    """D39 (user decision 2026-10-26): the history log records the session each line ran in; @history
    shows the current session's lines from bash and milk-script too (--global: all), search stays the
    milk-cli prompt's; milk-script's own session is labelled as milk-script's."""
    n = sessions()
    env, shm_dir = _isolated_shm_env()
    try:
        _run(
            [MILK_CLI_EXEC, "--session", n],
            env,
            30,
            stdin="echo in-session\n",
            cwd=str(tmp_path),
        )
        _run([MILK_CLI_EXEC], env, 30, stdin="echo outside\n", cwd=str(tmp_path))
        res = milk_bash(
            f"""
@session use {n} >/dev/null
echo "local=$(@history | grep -c in-session),$(@history | grep -c outside)"
echo "global=$(@history --global | grep -c 'echo in-session\\|echo outside')"
@history search x 2>err.txt; echo "search=$?"
""",
            tmp_path,
            extra_env={"HOME": env["HOME"], "MILK_SHM_DIR": env["MILK_SHM_DIR"]},
        )
        (tmp_path / "t.milk").write_text(
            '@session status | sed -n "s/ *Kind *: /kind=/p"\n'
            'echo "g=$(@history --global | grep -c in-session)"\n'
        )
        scr = _run(
            [str(_install_dir() / "bin" / "milk-script"), str(tmp_path / "t.milk")],
            env,
            30,
            cwd=str(tmp_path),
        )
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)
    out = res.stdout + res.stderr + scr.stdout + scr.stderr
    assert value(res.stdout, "local") == "1,0", out
    assert value(res.stdout, "global") == "2", out
    assert value(res.stdout, "search") == "2", out
    assert (
        "@history search: at the milk-cli prompt" in (tmp_path / "err.txt").read_text()
    ), out
    assert value(scr.stdout, "kind") == "milk-script's (in its process)", out
    assert value(scr.stdout, "g") == "1", out


# ------------------------------------------------------------- E6: @get --watch, a sweep


def test_E6_get_watch(tmp_path):
    """D41: @get --watch [T] <spec>... prints the values every T s until Ctrl-C ends it (status 0):
    one tab-separated line per refresh off a terminal. A bad period or a binding is a usage error.
    """
    res = milk_bash(
        'mem.mk2Dim "@S:e6c" 2 2 >/dev/null\n'
        '( for v in 1 2 3; do sleep 0.3; @calc "e6c = $v" >/dev/null; done ) &\n'
        "@get --watch 0.1 s.e6c.cnt0 s.e6c.mean > w.out & wp=$!\n"
        'sleep 1.5; kill -INT $wp; wait $wp; echo "rc=$?"\n'
        '@get --watch 0 s.e6c.cnt0 2>/dev/null; echo "bad=$?"\n'
        '@get --watch -v x s.e6c.cnt0 2>/dev/null; echo "bind=$?"\n',
        tmp_path,
    )
    lines = (tmp_path / "w.out").read_text().splitlines()
    out = res.stdout + res.stderr + "\n".join(lines)
    assert value(res.stdout, "rc") == "0", out
    assert len(lines) >= 5 and all(len(line.split("\t")) == 2 for line in lines), out
    means = [line.split("\t")[1] for line in lines]
    assert (
        means[0] == "0" and "1" in means and means[-1] == "3"
    ), out  # it follows the stream
    assert value(res.stdout, "bad") == "2" and value(res.stdout, "bind") == "2", out


def test_E6_parameter_sweep_loop(tmp_path):
    """D42: a sweep is a loop, as in scripting.md: @set --restore puts the parameter back when the
    function returns, and @calc -v binds what each step measures (on a milk.calc unit).
    """
    res = milk_bash(
        HELPERS
        + 'milk-fpsexec-milk-calc -n e6u -t cam -e "e6out = cam * gain" -p "gain:float=1" fpsinit '
        ">/dev/null\n"
        'milk-fpsexec-milk-calc -n e6u runstart >/dev/null 2>&1; echo "start=$?"\n'
        "sweep() {\n"
        "    @set --restore f.e6u.gain=1\n"
        "    local g m\n"
        "    for g in 0.5 2 3; do\n"
        "        @set f.e6u.gain=$g; frame 2\n"
        '        @calc -v m "imean(e6out)"; echo "step=$g $m"\n'
        "    done\n"
        "}\n"
        "sweep\n"
        'echo "after=$(@get f.e6u.gain)"\n'
        "milk-fps-runstop e6u >/dev/null 2>&1\n",
        tmp_path,
        timeout=60,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "start") == "0", out
    assert re.findall(r"^step=(.*)$", res.stdout, re.M) == ["0.5 1", "2 4", "3 6"], out
    assert value(res.stdout, "after") == "1", out


# ------------------------------------------------------------- D32: units brought up by a milk script

DEMO_UNITS = (
    "mkrnd gaussfilt imtrunc gaussfilt2 resizeim rotateim cropim streamave streammon"
)
DEMO_STREAMS = "random_in blurred_out truncated_out smooth_out resized_out rotated_out cropped_out averaged_out"


def test_D32_demopipeline_starts_each_unit_on_a_frame(tmp_path):
    """milk-demopipeline returns once its nine units run and each stream has a frame (streamave's average
    included), in well under the 16 s of fixed sleeps it used to take; cleanup stops the units and
    removes their files."""
    env, shm_dir = _isolated_shm_env()
    exe = str(_install_dir() / "bin" / "milk-demopipeline")
    try:
        h1 = _run([exe, "-h1"], env, timeout=30, cwd=str(tmp_path))
        assert (h1.returncode, h1.stdout.strip()) == (
            0,
            "start a multi-stage FPS pipeline for demonstration",
        ), (
            h1.stdout + h1.stderr
        )

        t0 = time.monotonic()
        res = _run([exe, "-f", "50"], env, timeout=90, cwd=str(tmp_path))
        took = time.monotonic() - t0
        out = res.stdout + res.stderr
        assert res.returncode == 0, out
        assert took < 10, f"{took:.1f} s\n{out}"

        chk = _run(
            [str(_install_dir() / "bin" / "milk-script"), "-q", "-"],
            env,
            timeout=30,
            stdin=f'for s in {DEMO_STREAMS}; do echo "$s=$(@get s.$s.cnt0)"; done\n'
            f'for u in {DEMO_UNITS}; do echo "$u=$(@get f.$u.status)"; done\n',
            cwd=str(tmp_path),
        )
        got = chk.stdout + chk.stderr
        assert all(
            int(value(chk.stdout, s) or 0) > 0 for s in DEMO_STREAMS.split()
        ), got
        assert all(value(chk.stdout, u) == "running" for u in DEMO_UNITS.split()), got
    finally:
        clean = _run([exe, "cleanup"], env, timeout=60, cwd=str(tmp_path))
        names = set(DEMO_UNITS.split()) | set(DEMO_STREAMS.split())
        # <unit>.fps.shm, <unit>.fps.d, <stream>.im.shm, proc.<unit>.<pid>.shm
        left = sorted(
            f
            for f in os.listdir(shm_dir)
            if f.split(".")[0] in names
            or (f.startswith("proc.") and f.split(".")[1] in names)
        )
        shutil.rmtree(shm_dir, ignore_errors=True)
    assert clean.returncode == 0, clean.stdout + clean.stderr
    assert left == [], left


# ------------------------------------------------------------- E1: unit properties (D30)


def test_E1_loop_rates_agree_everywhere(tmp_path):
    """A 100 Hz unit reads about 100 Hz in @get, --json, @ls -l and @ls --json: the last three took the
    rate from processinfo fields no unit fills, and showed 0. dt and dtmax are in µs."""
    res = milk_bash(
        HELPERS
        + 'milk-fpsexec-milk-calc -n e1r -loopd 0.01 -e "e1r_o = cam * 2" fpsinit >/dev/null\n'
        'milk-fpsexec-milk-calc -n e1r runstart >/dev/null 2>&1; echo "start=$?"; sleep 1.2\n'
        'echo "hz=$(@get p.e1r.hz)"; echo "dt=$(@get p.e1r.dt)"; echo "dtmax=$(@get p.e1r.dtmax)"\n'
        "echo \"json=$(@get --json p.e1r | tr -d '\\n')\"\n"
        "echo \"ls=$(@ls -l 'p.e1r')\"\n"
        "echo \"lsjson=$(@ls --json 'p.e1r' | tr -d '\\n')\"\n"
        "milk-fps-runstop e1r >/dev/null 2>&1\n",
        tmp_path,
        timeout=60,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "start") == "0", out
    assert 80 < float(value(res.stdout, "hz")) < 120, out
    dt, dtmax = float(value(res.stdout, "dt")), float(value(res.stdout, "dtmax"))
    assert 8000 < dt < 12500 and dtmax >= dt, out
    js = json.loads(value(res.stdout, "json"))
    assert 80 < js["loopfreq"] < 120 and 80 < js["hz"] < 120, out
    assert 8000 < js["dt"] < 12500 and js["trigmode"] == "delay", out
    m = re.search(r"\s([0-9.]+) Hz", value(res.stdout, "ls") or "")
    assert m is not None and 80 < float(m.group(1)) < 120, out
    assert 80 < json.loads(value(res.stdout, "lsjson"))[0]["freq_hz"] < 120, out


def test_E1_trigger_modes_by_name(tmp_path):
    """procinfo.triggermode and p.<unit>.trigmode read as names; @set takes a name or a number, and an
    unknown name fails and leaves the mode; @waitfor matches a mode by its number too.
    """
    res = milk_bash(
        HELPERS
        + 'milk-fpsexec-milk-calc -n e1t -t cam -e "e1t_o = cam" fpsinit >/dev/null\n'
        'echo "init=$(@get f.e1t.procinfo.triggermode)"\n'
        '@set f.e1t.procinfo.triggermode=delay; echo "byname=$? $(@get f.e1t.procinfo.triggermode)"\n'
        '@set f.e1t.procinfo.triggermode=3; echo "bynum=$? $(@get f.e1t.procinfo.triggermode)"\n'
        "@set f.e1t.procinfo.triggermode=often 2>err.log\n"
        'echo "bad=$? $(@get f.e1t.procinfo.triggermode)"\n'
        '@waitfor f.e1t.procinfo.triggermode=3 -t 1; echo "waitnum=$?"\n'
        'milk-fpsexec-milk-calc -n e1t runstart >/dev/null 2>&1; echo "start=$?"\n'
        'echo "p=$(@get p.e1t.trigmode)"\n'
        "milk-fps-runstop e1t >/dev/null 2>&1\n",
        tmp_path,
        timeout=60,
    )
    out = res.stdout + res.stderr + (tmp_path / "err.log").read_text()
    assert value(res.stdout, "init") == "semaphore", out
    assert value(res.stdout, "byname") == "0 delay", out
    assert value(res.stdout, "bynum") == "0 semaphore", out
    assert value(res.stdout, "bad") == "1 semaphore", out
    assert "trigger mode 'often'" in (tmp_path / "err.log").read_text(), out
    assert value(res.stdout, "waitnum") == "0", out
    assert (
        value(res.stdout, "start") == "0" and value(res.stdout, "p") == "semaphore"
    ), out


def test_E1_parameter_check_ok_errors_msg(tmp_path):
    """f.<unit>.ok, errors and msg give the last parameter check: 0 before any check, 1 after a passing
    confstep, and after a start that failed its check the count and a '<parameter>: <message>' line;
    --json has ok and errors."""
    res = milk_bash(
        HELPERS
        + 'milk-fpsexec-milk-calc -n e1c -t cam -e "e1c_o = cam" fpsinit >/dev/null\n'
        'echo "fresh=$(@get f.e1c.ok) $(@get f.e1c.errors) [$(@get f.e1c.msg)]"\n'
        '@set f.e1c.conf=step >/dev/null 2>&1; echo "step=$(@get f.e1c.ok) $(@get f.e1c.errors)"\n'
        "@set f.e1c.procinfo.loopcntMax=-5\n"
        'milk-fpsexec-milk-calc -n e1c runstart >/dev/null 2>&1; echo "start=$?"\n'
        'echo "failed=$(@get f.e1c.ok) $(@get f.e1c.errors)"; echo "msg=$(@get f.e1c.msg)"\n'
        "echo \"json=$(@get --json f.e1c | tr -d '\\n')\"\n",
        tmp_path,
        timeout=60,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "fresh") == "0 0 []", out
    assert value(res.stdout, "step") == "1 0", out
    assert value(res.stdout, "start") == "1", out
    assert value(res.stdout, "failed") == "0 1", out
    assert (
        value(res.stdout, "msg") == "procinfo.loopcntMax: int64 value -5 below min -1"
    ), out
    js = json.loads(value(res.stdout, "json"))
    assert js["ok"] == 0 and js["errors"] == 1 and "msg" not in js, out


def test_E1_waitfor_process_sees_a_clean_exit(tmp_path):
    """@waitfor p.<unit> returns once the unit has stopped cleanly (its processinfo file gone), also when
    the wait starts after the exit; a wait for another state still times out."""
    res = milk_bash(
        HELPERS
        + 'milk-fpsexec-milk-calc -n e1w -loopd 0.01 -e "e1w_o = cam" fpsinit >/dev/null\n'
        'milk-fpsexec-milk-calc -n e1w runstart >/dev/null 2>&1; echo "start=$?"\n'
        "( sleep 0.5; milk-fps-runstop e1w >/dev/null 2>&1 ) &\n"
        '@waitfor p.e1w -t 5; echo "during=$?"; wait\n'
        't0=$(date +%s%N); @waitfor p.e1w -t 5; echo "after=$? $(( ($(date +%s%N) - t0) / 1000000 ))"\n'
        '@waitfor p.e1w RUNNING -t 0.3; echo "running=$?"\n',
        tmp_path,
        timeout=60,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "start") == "0" and value(res.stdout, "during") == "0", out
    rc, ms = value(res.stdout, "after").split()
    assert rc == "0" and int(ms) < 2000, out
    assert value(res.stdout, "running") == "124", out


# ------------------------------------------------------------- docs audit (2026-10-26)


def test_save_keeps_string_variables(tmp_path, sessions):
    """@session save writes the calculator's string variables, which live apart from its numbers, quoted
    for bash: it dropped them without a word."""
    a, b = sessions(), sessions()
    res = milk_bash(
        f"""
@session new {a} >/dev/null
@calc 'sv = "abc def"'; @calc "ap = \\"it's\\""
@session save s.milk >/dev/null; echo "save=$?"
@session new {b} >/dev/null
@session restore s.milk >/dev/null; echo "restore=$?"
@calc -v x sv; @calc -v y ap; echo "got=[$x][$y]"
""",
        tmp_path,
        timeout=60,
    )
    out = res.stdout + res.stderr + (tmp_path / "s.milk").read_text()
    assert value(res.stdout, "save") == "0", out
    assert value(res.stdout, "restore") == "0", out
    assert value(res.stdout, "got") == "[abc def][it's]", out


def test_new_names_the_dir_of_a_session_listed_elsewhere(tmp_path, sessions):
    """A session belongs to the MILK_SHM_DIR it started with, where @session ls lists it. @session new
    of its name from another directory fails and names that one."""
    name = sessions()
    other = tempfile.mkdtemp(
        prefix="milk_other_", dir=os.environ.get("MILK_SHM_DIR", "/tmp")
    )
    there = (
        f"env -u MILKCLI_SOCK -u _MILK_SERVER_NAME MILK_SHM_DIR={other} MILK_PROC_DIR={other} "
        'bash --norc -c \'source "$MILK_INSTALLDIR/bin/milk.bash"; '
    )
    try:
        res = milk_bash(
            there + f"@session new {name} -d --idle 1m >/dev/null'\n"
            f'@session new {name} 2>new.err; echo "new=$?"\n'
            f'echo "listed=$(@session ls | grep -c {name})"\n'
            + there
            + f"@session end {name} >/dev/null; echo ended=$?'\n",
            tmp_path,
            timeout=60,
        )
    finally:
        shutil.rmtree(other, ignore_errors=True)
    err = (tmp_path / "new.err").read_text()
    assert value(res.stdout, "new") == "1", res.stdout + err
    assert f"it runs with MILK_SHM_DIR={other}, where @session ls lists it" in err, err
    assert value(res.stdout, "listed") == "0", res.stdout
    assert value(res.stdout, "ended") == "0", res.stdout + res.stderr


def test_session_names_hold_63_characters_at_most(tmp_path):
    """A longer name is refused: a session's files, its status and milk-cli's prompt cut it, and past
    about 92 characters `@session new` failed with "Invalid argument"."""
    res = milk_bash(f'@session new {"n" * 64} 2>err.txt; echo "rc=$?"', tmp_path)
    assert value(res.stdout, "rc") == "2", res.stdout + res.stderr
    assert "longer than 63 characters" in (tmp_path / "err.txt").read_text()
