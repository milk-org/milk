"""Tests for the zero-dependency C deployment primitives and engine lifecycle fixes.

Purpose:     light-bash cacao deployment relies on standalone C binaries (milk-fps-set,
             milk-fps-{confstart,confstop,runstart,runstop,confstep}, milk-fps-wait,
             milk-stream-link, milk-stream-wait) without requiring Python, TOML, or milk-cli
             at deployment runtime.
Does:        exercises milk-fps-set (2-arg, multi-arg key=val, -b batch stdin/file, 2-pass
             atomic validation, SIGNAL_UPDATE + processinfo_change_cnt on .procinfo.*, and
             --skip-running / --respect-writerun / default running guards); multi-unit
             parallel milk-fps-{confstart,confstop,runstart,runstop,confstep} and confstep
             validation without a conf daemon; milk-fps-wait (completion, already-exited,
             crashed worker -> 1, timeout -> 124); milk-stream-link (-s direct, --expect-size,
             regular-file overwrite refusal, atomic hot-swap); milk-stream-wait (delayed
             inotify creation, :WxH validation, +N live semaphore frames, timeout -> 124); and
             milk-fpsexec-linalg-MVMextract (.outref reference offset + CPU BLAS fallback).
Covers:      src/engine/libfps/params/fps_paramvalue.c, src/engine/libfps/tools/milk-fps-set.c,
             src/engine/libcomputeunit/core/fps_lifecycle.c,
             src/engine/libcomputeunit/tools/milk-fps-cmd.c,
             src/engine/libcomputeunit/tools/milk-fps-wait.c,
             src/coremods/COREMOD_memory/bin/milk-stream-link.c,
             src/coremods/COREMOD_memory/bin/milk-stream-wait.c,
             plugins/milk-extra-src/linalgebra/MVMextractModes.cpp.
Uses:        conftest.py: milk_bash, value, _install_dir, _isolated_shm_env, _run; a small
             milk-cc unit (DTUNIT_C) to observe FPS header counters and confcheck execution.
See also:    unit_lifecycle_test.py, removed_forms_test.py.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from milk.cliwrap import HAVE_CLI
from tests.cli.conftest import (
    _install_dir,
    _isolated_shm_env,
    _run,
    milk_bash,
    value,
)

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.timeout(90),
]

DTUNIT_C = r"""
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "ImageStreamIO/ImageStreamIO.h"
#include "milk_runnable.h"

static char    dt_in[FUNCTION_PARAMETER_STRMAXLEN]  = "dt_in";
static char    dt_out[FUNCTION_PARAMETER_STRMAXLEN] = "dt_out";
static double  dt_gain                              = 0.1;
static int64_t dt_size                              = 16;

#define DT_PARAMS(X)                                                                           \
    X(".in", dt_in, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_TRIGGER_STREAM, "Input stream")       \
    X(".out", dt_out, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "Output stream")             \
    X(".gain", &dt_gain, FPTYPE_FLOAT64, 0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, "RT gain") \
    X(".size", &dt_size, FPTYPE_INT64, 0, FPFLAG_DEFAULT_INPUT, "Static structural size")

MILK_PARAMS_DEFINE(dtunit, DT_PARAMS);

static errno_t dtunit_confcheck(void)
{
    if (dcfpsptr != NULL)
    {
        int pidx = functionparameter_GetParamIndex(dcfpsptr, ".size");
        if (pidx >= 0 && dcfpsptr->parray[pidx].val.i64[0] < 4)
        {
            dcfpsptr->parray[pidx].val.i64[0] = 4;
        }
    }
    return RETURN_SUCCESS;
}

static errno_t dtunit_compute(const MILK_RUNNABLE *self)
{
    IMAGE    in_im;
    IMAGE    out_im;
    uint32_t dims[2] = { 4, 1 };

    memset(&in_im, 0, sizeof(in_im));
    memset(&out_im, 0, sizeof(out_im));
    if (ImageStreamIO_openIm(&in_im, dt_in) != IMAGESTREAMIO_SUCCESS)
    {
        return RETURN_FAILURE;
    }
    if (ImageStreamIO_createIm_gpu(&out_im, dt_out, 2, dims, _DATATYPE_FLOAT, -1, 1,
                                   IMAGE_NB_SEMAPHORE, 0, MATH_DATA, 0) != IMAGESTREAMIO_SUCCESS)
    {
        ImageStreamIO_closeIm(&in_im);
        return RETURN_FAILURE;
    }

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    {
        out_im.md->write    = 1;
        out_im.array.F[0]   = (float) dt_gain;
        out_im.array.F[1]   = (float) dt_size;
        out_im.array.F[2]   = (float) (dcfpsptr ? dcfpsptr->md->processinfo_change_cnt : 0);
        out_im.array.F[3]   = (float) processinfo->loopcntMax;
        ImageStreamIO_UpdateIm(&out_im);
    }
    MILK_PROCINFO_END(self)

    ImageStreamIO_closeIm(&out_im);
    ImageStreamIO_closeIm(&in_im);
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_dtunit = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = {
        .fps_name    = "dtunit",
        .cmdkey      = "dtunit",
        .description = "deployment primitive test unit",
    },
    .srcfile   = __FILE__,
    MILK_PARAMS_REF(dtunit),
    .confcheck = dtunit_confcheck,
    .compute   = dtunit_compute,
};

EXPORT_FPSEXEC(milk, "dtunit", runnable_dtunit);
"""


@pytest.fixture(scope="module")
def dtunit(tmp_path_factory):
    """Compile the deployment test unit once for this module."""
    d = tmp_path_factory.mktemp("dtunit")
    src = d / "dtunit.c"
    exe = d / "milk-fpsexec-dtunit"
    src.write_text(DTUNIT_C)
    env, shm = _isolated_shm_env()
    try:
        res = _run(
            [str(_install_dir() / "bin" / "milk-cc"), str(src), "-o", str(exe)],
            env,
            timeout=30,
        )
        assert res.returncode == 0, res.stdout + res.stderr
    finally:
        shutil.rmtree(shm, ignore_errors=True)
    return str(exe)


def test_help_flags_across_deployment_binaries():
    """All deployment C binaries implement -h, -h1, -h2, and -hm cleanly."""
    env, shm = _isolated_shm_env()
    try:
        bindir = _install_dir() / "bin"
        for name in (
            "milk-stream-wait",
            "milk-fps-wait",
            "milk-fps-set",
            "milk-stream-link",
            "milk-fps-runstart",
            "milk-fps-confstart",
        ):
            for flag in ("-h1", "-h2", "-h", "-hm"):
                res = _run([str(bindir / name), flag], env, timeout=10)
                assert res.returncode == 0, f"{name} {flag}: {res.stderr}"
                assert res.stdout.strip() != "", f"{name} {flag} produced empty stdout"
    finally:
        shutil.rmtree(shm, ignore_errors=True)


def test_fps_set_batch_two_pass_and_procinfo_counter(tmp_path: Path, dtunit: str):
    """milk-fps-set supports 2-arg (including negative values), multi-arg key=val, and -b
    batch mode, rejects invalid batches without partial writes, and bumps
    processinfo_change_cnt + SIGNAL_UPDATE on .procinfo.* updates."""
    res = milk_bash(
        r"""
        mem.mk2Dim "@S:dt_in" 4 4 >/dev/null
        "$DT" -n u1 fpsinit >/dev/null
        "$DT" -n u2 fpsinit >/dev/null

        # 2-arg form with negative value (-1) without '--'
        milk-fps-set u1.procinfo.loopcntMax -1 >/dev/null

        # Multi-arg key=val across two FPS instances
        milk-fps-set u1.gain=0.25 u1.out=u1_out u2.gain=0.75 u2.size=32

        # Batch stdin supporting '=', whitespace, legacy 'setval', quotes, and comments
        milk-fps-set -b - <<'EOF'
        # comment line
        u1.size = 64 # inline comment
        setval u2.out "u2_out"
        u2.procinfo.loopcntMax -1
EOF

        # Two-pass atomicity: bad 2nd line must NOT commit 1st line (u1.size stays 64)
        milk-fps-set -b - <<'EOF' 2>bad.err
        u1.size = 128
        u1.gain = not_a_float
EOF
        echo "bad_rc=$?"
        echo "u1_size=$(@get f.u1.size)"
        echo "u2_out=$(@get f.u2.out)"

        # Verify processinfo_change_cnt was incremented by .procinfo.loopcntMax write
        milk-fps-runstart u1 >/dev/null
        milk-stream-wait -t 5 u1_out:4x1 >/dev/null
        @calc "dt_in = dt_in + 1"
        milk-stream-wait -t 5 u1_out+1 >/dev/null
        @calc -v pcnt "@S:u1_out[2, 0]"
        echo "pcnt=$pcnt"
        milk-fps-runstop u1 >/dev/null
        """,
        tmp_path,
        extra_env={"DT": dtunit},
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "bad_rc") == "2", out
    assert value(res.stdout, "u1_size") == "64", out
    assert value(res.stdout, "u2_out") == "u2_out", out
    assert float(value(res.stdout, "pcnt")) >= 1.0, out


def test_fps_set_running_unit_guards(tmp_path: Path, dtunit: str):
    """While a unit is running: default rejects non-WRITERUN params, --respect-writerun skips
    only non-WRITERUN params, and --skip-running skips the entire running unit while still
    updating stopped units in the same batch."""
    res = milk_bash(
        r"""
        mem.mk2Dim "@S:dt_in" 4 4 >/dev/null
        "$DT" -n r1 fpsinit .out=r1_out .gain=0.10 .size=16 >/dev/null
        "$DT" -n s1 fpsinit .out=s1_out .gain=0.10 .size=16 >/dev/null
        milk-fps-runstart r1 >/dev/null

        # 1. Default rejects non-WRITERUN (.size) on running r1, leaves .gain untouched
        milk-fps-set r1.gain=0.50 r1.size=32 2>def.err
        echo "def_rc=$? gain1=$(@get f.r1.gain) size1=$(@get f.r1.size)"

        # 2. --respect-writerun applies WRITERUN (.gain=0.40) and skips locked (.size=32)
        milk-fps-set --respect-writerun -q r1.gain=0.40 r1.size=32
        echo "rw_rc=$? gain2=$(@get f.r1.gain) size2=$(@get f.r1.size)"

        # 3. --skip-running preserves live-tuned r1.gain=0.40 while configuring stopped s1
        milk-fps-set --skip-running -q r1.gain=0.05 r1.size=64 s1.gain=0.80 s1.size=64
        echo "sr_rc=$? r1_gain=$(@get f.r1.gain) s1_gain=$(@get f.s1.gain) s1_size=$(@get f.s1.size)"

        milk-fps-runstop r1 >/dev/null
        """,
        tmp_path,
        extra_env={"DT": dtunit},
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "def_rc") == "1 gain1=0.1 size1=16", out
    assert value(res.stdout, "rw_rc") == "0 gain2=0.4 size2=16", out
    assert value(res.stdout, "sr_rc") == "0 r1_gain=0.4 s1_gain=0.8 s1_size=64", out


def test_multi_unit_cmd_and_confstep_validation(tmp_path: Path, dtunit: str):
    """milk-fps-{confstart,confstop,runstart,runstop,confstep} operate across multiple units
    in parallel, and confstep runs confcheck even when no background conf loop is active.
    """
    res = milk_bash(
        r"""
        mem.mk2Dim "@S:dt_in" 4 4 >/dev/null
        "$DT" -n m1 fpsinit .out=m1_out .size=1 >/dev/null
        "$DT" -n m2 fpsinit .out=m2_out .size=2 >/dev/null
        "$DT" -n m3 fpsinit .out=m3_out .size=8 >/dev/null

        # Multi-unit confstep without confstart clamps .size < 4 -> 4 via dtunit_confcheck
        milk-fps-confstep m1 m2 >/dev/null
        echo "step_rc=$? m1_sz=$(@get f.m1.size) m2_sz=$(@get f.m2.size)"

        # Parallel confstart -> runstart -> runstop -> confstop across 3 units
        milk-fps-confstart m1 m2 m3 >/dev/null
        echo "cs_rc=$?"
        milk-fps-runstart m1 m2 m3 >/dev/null
        echo "rs_rc=$?"
        milk-stream-wait -t 5 m1_out:4x1 m2_out:4x1 m3_out:4x1 >/dev/null
        echo "sw_rc=$?"
        milk-fps-runstop m1 m2 m3 >/dev/null
        echo "rx_rc=$?"
        milk-fps-confstop m1 m2 m3 >/dev/null
        echo "cx_rc=$?"

        # Error propagation when one unit in a multi-unit call does not exist
        milk-fps-runstart m1 no_such_unit >/dev/null 2>&1
        echo "bad_rc=$?"
        milk-fps-runstop m1 >/dev/null
        """,
        tmp_path,
        extra_env={"DT": dtunit},
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "step_rc") == "0 m1_sz=4 m2_sz=4", out
    assert value(res.stdout, "cs_rc") == "0", out
    assert value(res.stdout, "rs_rc") == "0", out
    assert value(res.stdout, "sw_rc") == "0", out
    assert value(res.stdout, "rx_rc") == "0", out
    assert value(res.stdout, "cx_rc") == "0", out
    assert value(res.stdout, "bad_rc") == "1", out


def test_fps_wait_completion_failure_and_timeout(tmp_path: Path, dtunit: str):
    """milk-fps-wait blocks until finite detached units exit (0), succeeds immediately on
    already-exited units (0), reports crashed workers (1), and times out with 124."""
    res = milk_bash(
        r"""
        mem.mk2Dim "@S:dt_in" 4 4 >/dev/null
        "$DT" -n wfin fpsinit .out=wfin_out >/dev/null
        milk-fps-set wfin.procinfo.loopcntMax 2 >/dev/null
        milk-fps-runstart wfin >/dev/null

        # Trigger 2 frames from background so wfin finishes while milk-fps-wait blocks
        ( sleep 0.1; @calc "dt_in = dt_in + 1"; sleep 0.05; @calc "dt_in = dt_in + 1" ) &
        milk-fps-wait -t 5 wfin
        echo "fin_rc=$?"

        # Calling milk-fps-wait again after clean exit returns 0 immediately
        milk-fps-wait -t 1 wfin
        echo "again_rc=$?"

        # Continuous unit times out with 124, and SIGKILL causes exit 1
        "$DT" -n wkill fpsinit .out=wkill_out >/dev/null
        milk-fps-runstart wkill >/dev/null
        milk-fps-wait -q -t 0.15 wkill
        echo "tmo_rc=$?"

        rpid=$(grep ^pid= "$MILK_SHM_DIR/wkill.fps.d/run.info" | cut -d= -f2)
        kill -9 "$rpid"
        milk-fps-wait -q -t 5 wkill
        echo "kill_rc=$?"
        """,
        tmp_path,
        extra_env={"DT": dtunit},
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "fin_rc") == "0", out
    assert value(res.stdout, "again_rc") == "0", out
    assert value(res.stdout, "tmo_rc") == "124", out
    assert value(res.stdout, "kill_rc") == "1", out


def test_stream_link_direct_atomic_and_safety(tmp_path: Path):
    """milk-stream-link -s validates dimensions, refuses to clobber regular .im.shm files,
    and atomically hot-swaps symlinks."""
    res = milk_bash(
        r"""
        mem.mk2Dim "@S:src_a" 16 8 >/dev/null
        mem.mk2Dim "@S:src_b" 16 8 >/dev/null
        mem.mk2Dim "@S:real_stream" 4 4 >/dev/null
        @calc "src_a = 3; src_b = 9"

        # Refuse to clobber a real (non-symlink) SHM file
        milk-stream-link -s src_a real_stream 2>clobber.err
        echo "clobber_rc=$?"

        # Dimension mismatch fails before touching symlink
        milk-stream-link -s src_a --expect-size 32x32 alias_im 2>dim.err
        echo "dim_rc=$?"

        # Direct link with matching --expect-size succeeds
        milk-stream-link -q -s src_a --expect-size 16x8 -p aol7_ wfsim
        echo "ok_rc=$?"
        @calc -v v1 "@S:aol7_wfsim[0, 0]"
        echo "v1=$v1"

        # Atomic hot-swap to src_b leaves no .tmp.* files
        milk-stream-link -q -s src_b -p aol7_ wfsim
        echo "swap_rc=$?"
        target=$(readlink "$MILK_SHM_DIR/aol7_wfsim.im.shm")
        ntmp=$(ls "$MILK_SHM_DIR"/.tmp.* 2>/dev/null | wc -l)
        echo "target=$(basename "$target") ntmp=$ntmp"
        """,
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "clobber_rc") == "1", out
    assert value(res.stdout, "dim_rc") == "1", out
    assert value(res.stdout, "ok_rc") == "0", out
    assert float(value(res.stdout, "v1")) == pytest.approx(3.0), out
    assert value(res.stdout, "swap_rc") == "0", out
    assert value(res.stdout, "target") == "src_b.im.shm ntmp=0", out


def test_stream_wait_creation_dims_frames_and_timeout(tmp_path: Path):
    """milk-stream-wait detects delayed stream creation via inotify, validates :WxH, waits
    for +N live semaphore updates, and returns 124 on timeout."""
    res = milk_bash(
        r"""
        # Delayed background stream creation + 3 real frame updates (cnt0 += 3 + sempost)
        (
            sleep 0.15
            mem.mk2Dim "@S:delayed_im" 10 6 >/dev/null
            for k in 1 2 3; do
                sleep 0.04
                @calc "delayed_im = delayed_im + 1"
            done
        ) &

        milk-stream-wait -t 5 s:delayed_im:10x6+3
        echo "wait_rc=$?"

        # Dimension mismatch exits 1 immediately
        milk-stream-wait -t 2 delayed_im:99x99 2>mismatch.err
        echo "mis_rc=$?"

        # Missing stream times out with 124
        milk-stream-wait -q -t 0.15 never_created_stream
        echo "tmo_rc=$?"
        """,
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "wait_rc") == "0", out
    assert value(res.stdout, "mis_rc") == "1", out
    assert value(res.stdout, "tmo_rc") == "124", out


def test_mvmextract_outref_and_blas_fallback(tmp_path: Path):
    """milk-fpsexec-linalg-MVMextract falls back to BLAS when GPUindex=0 without CUDA and
    adds .outref to the expanded output (out = outref + M * in)."""
    res = milk_bash(
        r"""
        # 2x1 spatial image, 2 modes: M = [[2, 0], [0, 3]], in = [4, 5], ref = [10, 20]
        # Expected expanded output: [10 + 2*4, 20 + 3*5] = [18, 35]
        mem.mk2Dim "@S:m_in" 2 1 >/dev/null
        mem.mk3Dim "@S:m_mat" 2 1 2 >/dev/null
        mem.mk2Dim "@S:m_ref" 2 1 >/dev/null
        @calc "m_in[0, 0] = 4; m_in[1, 0] = 5"
        @calc "m_mat[0, 0, 0] = 2; m_mat[1, 0, 1] = 3"
        @calc "m_ref[0, 0] = 10; m_ref[1, 0] = 20"

        milk-fpsexec-linalg-MVMextract -n simmvm fpsinit >/dev/null
        milk-fps-set -b - <<'EOF'
        simmvm.GPUindex = 0
        simmvm.insname = m_in
        simmvm.immodes = m_mat
        simmvm.outcoeff = m_out
        simmvm.outref = m_ref
        simmvm.axmode = 1
        simmvm.option.MODENORM = OFF
        simmvm.procinfo.triggermode = 0
        simmvm.procinfo.loopcntMax = 1
EOF
        milk-fpsexec-linalg-MVMextract -n simmvm run >/dev/null
        echo "run_rc=$?"
        @calc -v y0 "@S:m_out[0, 0]"
        @calc -v y1 "@S:m_out[1, 0]"
        echo "y0=$y0"
        echo "y1=$y1"
        """,
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "run_rc") == "0", out
    assert float(value(res.stdout, "y0")) == pytest.approx(18.0), out
    assert float(value(res.stdout, "y1")) == pytest.approx(35.0), out
