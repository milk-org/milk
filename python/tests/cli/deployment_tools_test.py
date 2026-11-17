"""Tests for the compiled C deployment primitives, fpspipe/cacaoloop commands, and AO controllers.

Purpose:     high-speed FPS pipeline and cacao AO loop deployment relies on compiled C batch
             binaries (milk-fps-init, milk-fps-set, milk-fps-{confstart,confstop,runstart,runstop,
             confstep}, milk-fps-wait, milk-fps-rm, milk-stream-link, milk-stream-wait,
             milk-stream-rm) and native milk-script / libmilkbash builtins (fpspipe.*,
             cacaoloop.*).
Does:        exercises milk-fps-init (parallel cold init, --print-created, warm access() skip,
             --restart, duplicate guard); milk-fps-set (2-arg, multi-arg key=val, -b batch,
             2-pass atomicity, SIGNAL_UPDATE + processinfo_change_cnt, running guards);
             multi-unit milk-fps-{confstart,confstop,runstart,runstop,confstep} (-q quiet) and
             confstep validation; milk-fps-wait; milk-stream-link (single + 2-phase batch
             rollback); milk-stream-wait; multi-target milk-fps-rm / milk-stream-rm;
             fpspipe.loadfits (.fits.gz + Rice .fits.fz from > 80-char paths, FITS keywords,
             in-place .force=1 reload); fpspipe.* uncommitted exit rollback; standalone RNG init
             in milkdata.c; dmcomb init-failure cleanup; and end-to-end KalAO + SCExAO vispyr2
             .milk AO loop controllers.
Covers:      src/engine/libcomputeunit/tools/milk-fps-init.c,
             src/engine/libcomputeunit/tools/milk-fps-cmd.c,
             src/engine/libcomputeunit/tools/milk-fps-wait.c,
             src/engine/libcomputeunit/tools/milk-fps-rm.c,
             src/engine/libcomputeunit/core/fps_lifecycle.c,
             src/engine/libfps/params/fps_paramvalue.c,
             src/engine/libfps/tools/milk-fps-set.c,
             src/engine/libmilkdata/milkdata.c,
             src/coremods/COREMOD_iofits/loadfits.c,
             src/coremods/COREMOD_memory/bin/{milk-stream-link.c,milk-stream-wait.c,milk-stream-rm.c},
             scripts/commands.d/fpspipe.bash,
             plugins/milk-extra-src/linalgebra/MVMextractModes.cpp,
             plugins/cacao-src/AOloopControl/{commands.d/cacaoloop.bash,examples/*/*.milk},
             plugins/cacao-src/AOloopControl_DM/{AOloopControl_DM_comb.c,DMturbulence.c}.
Uses:        conftest.py: milk_bash, value, _install_dir, _isolated_shm_env, _run; numpy and
             astropy.io.fits; a small milk-cc unit (DTUNIT_C).
See also:    unit_lifecycle_test.py, removed_forms_test.py.
"""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

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
    pytest.mark.timeout(120),
]

_CACAO_SRC = Path(__file__).resolve().parents[3] / "plugins" / "cacao-src"
_KALAO_MILK = (
    _CACAO_SRC
    / "AOloopControl"
    / "examples"
    / "KalAO-dmloop-conf"
    / "cacaoloop-kalao.milk"
)
_VISPYR_MILK = (
    _CACAO_SRC
    / "AOloopControl"
    / "examples"
    / "scexao-vispyr-bin2-conf"
    / "cacaoloop-vispyr2.milk"
)


def _have_cacao() -> bool:
    return (
        _install_dir() / "bin" / "cacao-fpsexec-dmcomb"
    ).exists() and _KALAO_MILK.is_file()


needs_cacao = pytest.mark.skipif(not _have_cacao(), reason="cacao plugin not built")

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
    """All deployment C binaries and .milk controllers implement -h, -h1, -h2, and -hm."""
    env, shm = _isolated_shm_env()
    try:
        bindir = _install_dir() / "bin"
        targets = [
            bindir / name
            for name in (
                "milk-fps-init",
                "milk-fps-rm",
                "milk-stream-rm",
                "milk-stream-wait",
                "milk-fps-wait",
                "milk-fps-set",
                "milk-stream-link",
                "milk-fps-runstart",
                "milk-fps-confstart",
            )
        ]
        if _have_cacao():
            targets.extend([_KALAO_MILK, _VISPYR_MILK])
        for exe in targets:
            for flag in ("-h1", "-h2", "-h", "-hm"):
                res = _run([str(exe), flag], env, timeout=10)
                assert res.returncode == 0, f"{exe.name} {flag}: {res.stderr}"
                assert (
                    res.stdout.strip() != ""
                ), f"{exe.name} {flag} produced empty stdout"
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


def test_fps_init_parallel_warm_restart_and_duplicates(tmp_path: Path, dtunit: str):
    """milk-fps-init initializes units concurrently, prints newly created names, skips warm
    existing units via access(), restarts running units with --restart, and rejects duplicates.
    """
    res = milk_bash(
        r"""
        mem.mk2Dim "@S:dt_in" 4 4 >/dev/null

        # Duplicate target name in batch fails pre-flight validation (exit 2) before spawning
        milk-fps-init -q -b - <<EOF 2>dup.err
        dup1 $DT .size=8
        dup1 $DT .size=16
EOF
        echo "dup_rc=$? dup_exists=$(@exists f.dup1 && echo 1 || echo 0)"

        # Parallel cold initialization across 3 units with --print-created
        created1=$(milk-fps-init -q --print-created -b - <<EOF
        p1 $DT .out=p1_out .gain=0.2 .size=16
        p2 $DT .out=p2_out .gain=0.3 .size=24
        p3 $DT .out=p3_out .gain=0.4 .size=32
EOF
        )
        echo "c1_rc=$? c1=$(echo $created1)"

        # Mutate p1.size and verify warm milk-fps-init skips existing units without clobbering
        milk-fps-set -q p1.size=128
        created2=$(milk-fps-init -q --print-created -b - <<EOF
        p1 $DT .out=p1_out .gain=0.9 .size=8
        p2 $DT .out=p2_out .gain=0.3 .size=24
EOF
        )
        echo "c2_rc=$? c2_len=${#created2} p1_sz=$(@get f.p1.size)"

        # Start p1 quietly, then verify --restart stops, removes, and re-initializes it
        rs_out=$(milk-fps-runstart -q p1)
        echo "rs_len=${#rs_out}"
        milk-stream-wait -t 5 p1_out:4x1 >/dev/null
        created3=$(milk-fps-init -q --restart --print-created p1 "$DT" .out=p1_out .size=40)
        echo "c3_rc=$? c3_len=${#created3} p1_sz2=$(@get f.p1.size) p1_st=$(@get f.p1.status)"
        """,
        tmp_path,
        extra_env={"DT": dtunit},
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "dup_rc") == "2 dup_exists=0", out
    assert value(res.stdout, "c1_rc") == "0 c1=p1 p2 p3", out
    assert value(res.stdout, "c2_rc") == "0 c2_len=0 p1_sz=128", out
    assert value(res.stdout, "rs_len") == "0", out
    assert value(res.stdout, "c3_rc") == "0 c3_len=0 p1_sz2=40 p1_st=exited", out


def test_loadfits_compressed_long_path_and_inplace_reload(tmp_path: Path):
    """fpspipe.loadfits loads .fits.gz and Rice-compressed .fits.fz from paths > 80 chars,
    preserves science FITS keywords while filtering BINTABLE compression headers, skips warm
    streams by default, and updates in-place on .force=1."""
    long_dir = tmp_path / ("deep_calib_path_" + "x" * 72)
    long_dir.mkdir(parents=True)
    assert len(str(long_dir)) > 85

    arr_gz = np.arange(24, dtype=np.float32).reshape((4, 6)) + 3.5
    raw_fits = tmp_path / "raw_gz.fits"
    hdu_gz = fits.PrimaryHDU(arr_gz)
    hdu_gz.header["GAIN"] = (2.5, "detector gain")
    hdu_gz.writeto(raw_fits, overwrite=True)
    with raw_fits.open("rb") as src_f, gzip.open(
        long_dir / "flat_field.fits.gz", "wb"
    ) as dst_f:
        shutil.copyfileobj(src_f, dst_f)

    arr_gz_updated = arr_gz + 100.0
    hdu_gz2 = fits.PrimaryHDU(arr_gz_updated)
    hdu_gz2.writeto(long_dir / "flat_updated.fits", overwrite=True)

    arr_fz = (np.arange(96, dtype=np.float32).reshape((3, 4, 8)) * 2.0).astype(
        np.float32
    )
    comp_hdu = fits.CompImageHDU(data=arr_fz, compression_type="RICE_1")
    comp_hdu.header["EXPTIME"] = (0.001, "integration time")
    fits.HDUList([fits.PrimaryHDU(), comp_hdu]).writeto(long_dir / "resp_cube.fits.fz")

    res = milk_bash(
        r"""
        fpspipe.workdir "pfit" "$TMP/pipe" .prefix="pf_"
        fpspipe.loadfits "flat" "$LONG_DIR/flat_field"
        fpspipe.loadfits "resp" "$LONG_DIR/resp_cube"

        kw_gain=$(@get s.pf_flat.kw.GAIN)
        kw_exp=$(@get s.pf_resp.kw.EXPTIME)
        kw_all=$(@get "s.pf_resp.kw.*")

        cnt_before=$(@get s.pf_flat.cnt0)
        fpspipe.loadfits "flat" "$LONG_DIR/flat_updated"
        cnt_warm=$(@get s.pf_flat.cnt0)

        fpspipe.loadfits "flat" "$LONG_DIR/flat_updated" .force=1
        cnt_forced=$(@get s.pf_flat.cnt0)

        @calc -v v_flat "@S:pf_flat[5, 3]"
        @calc -v v_resp "@S:pf_resp[7, 3, 2]"
        echo "cnt=$cnt_before,$cnt_warm,$cnt_forced"
        echo "v_flat=$v_flat"
        echo "v_resp=$v_resp"
        echo "kw_gain=$kw_gain"
        echo "kw_exp=$kw_exp"
        echo "kw_bad=$(echo "$kw_all" | grep -cE 'TTYPE1|ZIMAGE' || true)"

        fpspipe.commit
        fpspipe.teardown
        """,
        tmp_path,
        extra_env={"TMP": str(tmp_path), "LONG_DIR": str(long_dir)},
    )
    out = res.stdout + res.stderr
    assert res.returncode == 0, out
    c_before, c_warm, c_forced = [int(x) for x in value(res.stdout, "cnt").split(",")]
    assert c_warm == c_before, out
    assert c_forced > c_warm, out
    assert float(value(res.stdout, "v_flat")) == pytest.approx(
        float(arr_gz_updated[3, 5])
    ), out
    assert float(value(res.stdout, "v_resp")) == pytest.approx(
        float(arr_fz[2, 3, 7])
    ), out
    assert float(value(res.stdout, "kw_gain")) == pytest.approx(2.5), out
    assert float(value(res.stdout, "kw_exp")) == pytest.approx(0.001), out
    assert value(res.stdout, "kw_bad") == "0", out


def test_stream_link_batch_rollback_and_quiet_rm(tmp_path: Path, dtunit: str):
    """milk-stream-link validates all pairs before committing any symlink, and milk-fps-rm /
    milk-stream-rm remove multiple targets with -q --ignore-missing."""
    res = milk_bash(
        r"""
        mem.mk2Dim "@S:s1" 8 8 >/dev/null
        mem.mk2Dim "@S:s2" 8 8 >/dev/null
        mem.mk2Dim "@S:bad_sz" 4 4 >/dev/null
        mem.mk2Dim "@S:p_real" 8 8 >/dev/null
        @calc "s1 = 11; s2 = 22"

        # 2-phase rollback: 2nd pair fails --expect-size -> 1st pair (p_b1) is NOT created
        milk-stream-link -q --no-conf --expect-size 8x8 -p p_ s1=b1 bad_sz=b2 2>sz.err
        echo "sz_rc=$? b1_ex=$([ -L "$MILK_SHM_DIR/p_b1.im.shm" ] && echo 1 || echo 0)"

        # 2-phase rollback: 2nd pair targets regular .im.shm -> 1st pair (p_c1) is NOT created
        milk-stream-link -q --no-conf -p p_ s1=c1 s2=real 2>reg.err
        echo "reg_rc=$? c1_ex=$([ -L "$MILK_SHM_DIR/p_c1.im.shm" ] && echo 1 || echo 0)"

        # Valid positional batch + stdin -b batch hot-swap
        milk-stream-link -q --no-conf --expect-size 8x8 -p p_ s1=a1 s2=a2
        echo "ok_rc=$?"
        milk-stream-link -q --no-conf -p p_ -b - <<'EOF'
        # swap a1 -> s2 and add a3 -> s1
        s2 = a1
        s1 a3
EOF
        echo "bat_rc=$?"
        @calc -v va1,va3 "@S:p_a1[0, 0]; @S:p_a3[0, 0]"
        echo "vals=$va1,$va3"

        # Multi-target milk-fps-rm and milk-stream-rm with -q --ignore-missing
        "$DT" -n rmu1 fpsinit >/dev/null
        "$DT" -n rmu2 fpsinit >/dev/null
        rmf_out=$(milk-fps-rm -f -q --ignore-missing rmu1 rmu2 no_such_fps 2>&1)
        rms_out=$(milk-stream-rm -q --ignore-missing p_a1 p_a2 p_a3 s1 s2 no_such_stream 2>&1)
        echo "rm_rc=$? rmf_len=${#rmf_out} rms_len=${#rms_out}"
        echo "rem=$([ -e "$MILK_SHM_DIR/rmu1.fps.shm" ] || [ -L "$MILK_SHM_DIR/p_a1.im.shm" ] && echo 1 || echo 0)"
        """,
        tmp_path,
        extra_env={"DT": dtunit},
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "sz_rc") == "1 b1_ex=0", out
    assert value(res.stdout, "reg_rc") == "1 c1_ex=0", out
    assert value(res.stdout, "ok_rc") == "0", out
    assert value(res.stdout, "bat_rc") == "0", out
    va1, va3 = [float(x) for x in value(res.stdout, "vals").split(",")]
    assert va1 == pytest.approx(22.0) and va3 == pytest.approx(11.0), out
    assert value(res.stdout, "rm_rc") == "0 rmf_len=0 rms_len=0", out
    assert value(res.stdout, "rem") == "0", out


def test_fpspipe_uncommitted_exit_rollback(tmp_path: Path, dtunit: str):
    """fpspipe.workdir arms @defer --exit rollback that cleans up only newly created units and
    streams on mid-deploy failure while leaving pre-existing warm assets intact."""
    fits.PrimaryHDU(np.ones((4, 4), dtype=np.float32)).writeto(tmp_path / "img.fits")
    abort_script = tmp_path / "abort_deploy.milk"
    abort_script.write_text(
        "#!/usr/bin/env -S milk-script -q\n"
        "set -euo pipefail\n"
        'fpspipe.workdir "rbpipe" "$TMP/rbroot" .prefix="rb_"\n'
        'fpspipe.loadfits "warm_s" "$TMP/img.fits"\n'
        'fpspipe.loadfits "new_s" "$TMP/img.fits"\n'
        "fpspipe.init <<EOF\n"
        "warm_u $DT .out=warm_out .size=16\n"
        "new_u  $DT .out=new_out  .size=16\n"
        "EOF\n"
        "milk-fps-runstart -q new_u\n"
        "exit 1\n"
    )
    commit_script = tmp_path / "commit_deploy.milk"
    commit_script.write_text(
        "#!/usr/bin/env -S milk-script -q\n"
        "set -euo pipefail\n"
        'fpspipe.workdir "rbpipe" "$TMP/rbroot" .prefix="rb_"\n'
        'fpspipe.loadfits "warm_s" "$TMP/img.fits"\n'
        'fpspipe.loadfits "new_s" "$TMP/img.fits"\n'
        "fpspipe.init <<EOF\n"
        "warm_u $DT .out=warm_out .size=16\n"
        "new_u  $DT .out=new_out  .size=16\n"
        "EOF\n"
        "fpspipe.commit\n"
        "fpspipe.status >/dev/null\n"
    )

    res = milk_bash(
        r"""
        mem.mk2Dim "@S:dt_in" 4 4 >/dev/null
        mem.mk2Dim "@S:rb_warm_s" 4 4 >/dev/null
        "$DT" -n warm_u fpsinit .out=warm_out .size=64 >/dev/null

        milk-script -q "$TMP/abort_deploy.milk" || rc=$?
        echo "abort_rc=$rc"
        echo "warm_u=$(@exists f.warm_u && @get f.warm_u.size || echo gone)"
        echo "warm_s=$(@exists s.rb_warm_s && echo 1 || echo 0)"
        echo "new_u=$(@exists f.new_u && echo 1 || echo 0)"
        echo "new_s=$(@exists s.rb_new_s && echo 1 || echo 0)"

        milk-script -q "$TMP/commit_deploy.milk"
        echo "commit_rc=$? new_u2=$(@exists f.new_u && echo 1 || echo 0) new_s2=$(@exists s.rb_new_s && echo 1 || echo 0)"
        milk-fps-rm -f -q --ignore-missing warm_u new_u
        milk-stream-rm -q --ignore-missing rb_warm_s rb_new_s dt_in
        """,
        tmp_path,
        extra_env={"DT": dtunit, "TMP": str(tmp_path)},
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "abort_rc") == "1", out
    assert value(res.stdout, "warm_u") == "64", out
    assert value(res.stdout, "warm_s") == "1", out
    assert value(res.stdout, "new_u") == "0", out
    assert value(res.stdout, "new_s") == "0", out
    assert value(res.stdout, "commit_rc") == "0 new_u2=1 new_s2=1", out


@needs_cacao
def test_cacao_dmcomb_failure_cleanup_and_wfscamsim_rng(tmp_path: Path):
    """cacao-fpsexec-dmcomb returns clean failure (rc=1, no SIGSEGV) on missing voltname, and
    standalone cacao-fpsexec-cacaoiot-wfscamsim initializes milk_data.rndgen for noise calls.
    """
    res = milk_bash(
        r"""
        # 1. dmcomb with voltmode=ON and missing voltname fails cleanly (rc=1, not 139 SIGSEGV)
        cacao-fpsexec-dmcomb -n dmc99 fpsinit >/dev/null
        milk-fps-set -q -b - <<'EOF'
        dmc99.DMindex = 99
        dmc99.DMxsize = 8
        dmc99.DMysize = 8
        dmc99.option.voltmode = ON
        dmc99.option.voltname = missing_volt_stream
EOF
        cacao-fpsexec-dmcomb -n dmc99 run >/dev/null 2>&1 || dmc_rc=$?
        echo "dmc_rc=${dmc_rc:-0}"

        # 2. standalone wfscamsim with photon + readout noise succeeds without RNG SIGSEGV
        mem.mk2Dim "@S:opd_in" 8 8 >/dev/null
        @calc "opd_in = 0.05"
        cacao-fpsexec-cacaoiot-wfscamsim -n wsim_t fpsinit >/dev/null
        milk-fps-set -q -b - <<'EOF'
        wsim_t.wfssignal = opd_in
        wsim_t.wfscamim = wfs_rng_out
        wsim_t.compdarkadd = OFF
        wsim_t.fluxtotal = 10000
        wsim_t.compphnoise = ON
        wsim_t.camRON = 1.5
        wsim_t.procinfo.triggermode = 0
        wsim_t.procinfo.loopcntMax = 2
EOF
        cacao-fpsexec-cacaoiot-wfscamsim -n wsim_t run >/dev/null 2>&1
        echo "wsim_rc=$?"
        @calc -v wmax "imax(@S:wfs_rng_out)"
        echo "wmax=$wmax"
        """,
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "dmc_rc") == "1", out
    assert value(res.stdout, "wsim_rc") == "0", out
    assert float(value(res.stdout, "wmax")) > 0.0, out


@needs_cacao
def test_cacaoloop_kalao_deploy_calibrate_converge_and_teardown(tmp_path: Path):
    """cacaoloop-kalao.milk deploys in simulation, runs full linear calibration + SVD CM reload,
    converges closed-loop on an injected mode perturbation, preserves state on warm deploy, and
    leaves zero stream/FPS SHM files on teardown."""
    res = milk_bash(
        r"""
        set -e
        trap '"$KALAO" teardown >/dev/null 2>&1 || true' EXIT

        "$KALAO" deploy --sim >/dev/null
        "$KALAO" calibrate >/dev/null

        # Inject static perturbation of +0.20 rad/um on DM mode 0 via aol1_dmO
        @calc "@S:aol1_dmO = 0.20 * @S:aol1_CMmodesDM[:, :, 0]"
        milk-stream-wait -t 5 aol1_linsimWFS+4 aol1_modevalWFS+4 >/dev/null
        @calc -v ol_err "abs(@S:aol1_modevalWFS[0, 0])"

        # Close loop with gain 0.25 and wait 35 iterations for integrator convergence
        "$KALAO" close 0.25 >/dev/null
        milk-stream-wait -t 5 aol1_modevalWFS+35 >/dev/null
        @calc -v cl_err "abs(@S:aol1_modevalWFS[0, 0])"

        # Warm re-deploy while closed must preserve loopON=ON
        "$KALAO" deploy --sim >/dev/null
        loop_on=$(@get f.mfilt-1.loopON)

        "$KALAO" teardown >/dev/null
        trap - EXIT
        nshm=$(ls "$MILK_SHM_DIR"/*.im.shm "$MILK_SHM_DIR"/*.fps.shm 2>/dev/null | wc -l)
        nrec=$([ -d "$MILK_SHM_DIR/cacaoloop-1" ] && echo 1 || echo 0)
        echo "ol=$ol_err"
        echo "cl=$cl_err"
        echo "loop_on=$loop_on nshm=$nshm nrec=$nrec"
        """,
        tmp_path,
        extra_env={"KALAO": str(_KALAO_MILK)},
        timeout=90,
    )
    out = res.stdout + res.stderr
    assert res.returncode == 0, out
    ol_err = float(value(res.stdout, "ol"))
    cl_err = float(value(res.stdout, "cl"))
    assert ol_err > 0.15, out
    assert cl_err < 0.25 * ol_err, out
    assert value(res.stdout, "loop_on") == "ON nshm=0 nrec=0", out


@needs_cacao
def test_cacaoloop_vispyr2_fits_fz_mode_switch_and_teardown(tmp_path: Path):
    """cacaoloop-vispyr2.milk decompresses respM.fits.fz natively in C, switches atomically
    between sim and hw streams, and removes all stream/FPS SHM state on teardown."""
    res = milk_bash(
        r"""
        set -e
        trap '"$VISPYR" teardown >/dev/null 2>&1 || true' EXIT

        "$VISPYR" deploy --sim >/dev/null
        resp_dim="$(@get s.aol7_simLHSrespM.xsize)x$(@get s.aol7_simLHSrespM.ysize)x$(@get s.aol7_simLHSrespM.zsize)"
        sim_wfs=$(basename "$(readlink "$MILK_SHM_DIR/aol7_wfsim.im.shm")")
        sim_dmc=$(basename "$(readlink "$MILK_SHM_DIR/aol7_dmC.im.shm")")

        mem.mk2Dim "@S:ocam2d" 120 120 >/dev/null
        mem.mk2Dim "@S:dmvolt" 50 50 >/dev/null
        "$VISPYR" mode hw >/dev/null
        hw_wfs=$(basename "$(readlink "$MILK_SHM_DIR/aol7_wfsim.im.shm")")
        hw_dmc=$(basename "$(readlink "$MILK_SHM_DIR/aol7_dmC.im.shm")")

        "$VISPYR" mode sim >/dev/null
        back_wfs=$(basename "$(readlink "$MILK_SHM_DIR/aol7_wfsim.im.shm")")

        "$VISPYR" teardown >/dev/null
        milk-stream-rm -q --ignore-missing ocam2d dmvolt
        trap - EXIT
        naol=$(ls "$MILK_SHM_DIR"/aol7_* "$MILK_SHM_DIR"/dm10* "$MILK_SHM_DIR"/*.fps.shm 2>/dev/null | wc -l)
        nhw=$(ls "$MILK_SHM_DIR"/dm00disp*.im.shm 2>/dev/null | wc -l)
        echo "resp=$resp_dim sim=$sim_wfs,$sim_dmc hw=$hw_wfs,$hw_dmc back=$back_wfs naol=$naol nhw=$nhw"
        """,
        tmp_path,
        extra_env={"VISPYR": str(_VISPYR_MILK)},
        timeout=90,
    )
    out = res.stdout + res.stderr
    assert res.returncode == 0, out
    assert "resp=120x120x2500" in res.stdout, out
    assert "sim=aol7_ocam2dsim.im.shm,dm10disp03.im.shm" in res.stdout, out
    assert "hw=ocam2d.im.shm,dm00disp03.im.shm" in res.stdout, out
    assert "back=aol7_ocam2dsim.im.shm naol=0 nhw=13" in res.stdout, out
