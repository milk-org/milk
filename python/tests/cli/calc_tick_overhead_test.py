"""
Tests for the per-tick overhead work of the 2026-10-27 engine review (P6).

Slice bounds that hold variables (`out[k:k+7,:]`, `ring[:,:,_n % 4]`, `prev(cam)`) are compiled into
the statement's scalar tape and re-placed from it each run, instead of re-tokenizing and re-parsing
the slice text; a kept program re-validates the names of its images and variables only when the
image or variable tables changed (a generation counter bumped on create, delete, rename, remap).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from milk.cliwrap import HAVE_CLI
from tests.cli.conftest import _install_dir, _isolated_shm_env, _run, milk_bash, value

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.timeout(180),
]

perf_timing = pytest.mark.skipif(
    os.environ.get("MILK_PERF_TESTS", "1") == "0",
    reason="MILK_PERF_TESTS=0: timing tests off",
)

HELPERS = r"""
until_file() { for _ in $(seq 100); do [ -e "$MILK_SHM_DIR/$1.im.shm" ] && return 0; sleep 0.05; done; return 1; }
stop() { kill -INT "$1" 2>/dev/null; for _ in $(seq 60); do kill -0 "$1" 2>/dev/null || break; sleep 0.05; done
         kill -0 "$1" 2>/dev/null && echo "still running"; wait "$1" 2>/dev/null; }
"""


def test_compiled_bounds_follow_variables_in_a_unit(tmp_path):
    """A parameter bound, a frame-counter ring index, and literal negative/flip bounds in one unit."""
    (tmp_path / "prog.calc").write_text(
        "out[k:k+7,:] = src[k:k+7,:] * 2\nring[:,:,_n % 4] = src\ntail = src[-8:-1,-*] + 0\n"
    )
    res = milk_bash(
        HELPERS
        + 'mem.mk2Dim "@S:src" 32 16 >/dev/null; mem.mk2Dim "@S:out" 32 16 >/dev/null\n'
        'mem.mk3Dim "@S:ring" 32 16 4 >/dev/null; @calc "src = 1; out = 0; ring = 0"\n'
        'milk-fpsexec-milk-calc -n p6 -t src -p "k:int=0" -f prog.calc fpsinit >/dev/null\n'
        "milk-fpsexec-milk-calc -n p6 run >run.log 2>&1 & pid=$!\n"
        "sleep 0.6\n"
        '@calc "src = 3"; sleep 0.2; @set f.p6.k=8; sleep 0.1; @calc "src = 5"; sleep 0.2; @calc "src = 7"; sleep 0.3\n'
        '@calc -v a,b,c "imean(out[0:7,:]); imean(out[8:15,:]); imean(out[16:31,:])"; echo "out=$a,$b,$c"\n'
        '@calc -v r0,r1,r2,r3 "imean(ring[:,:,0]); imean(ring[:,:,1]); imean(ring[:,:,2]); imean(ring[:,:,3])"; echo "ring=$r0,$r1,$r2,$r3"\n'
        '@calc -v t "itot(tail)"; echo "tail=$t"; echo "error=[$(@get f.p6.unit.error)]"; stop $pid',
        tmp_path,
    )
    out = res.stdout + res.stderr + (tmp_path / "run.log").read_text()
    assert value(res.stdout, "error") == "[]", out
    assert (
        value(res.stdout, "out") == "6,14,0"
    ), out  # rows 0-7 last written at k=0 (src 3), 8-15 at k=8 (src 7)
    planes = [float(v) for v in value(res.stdout, "ring").split(",")]
    assert (
        all(p in (0.0, 1.0, 3.0, 5.0, 7.0) for p in planes)
        and sum(p > 0 for p in planes) >= 3
    ), out
    assert (
        value(res.stdout, "tail") == "896"
    ), out  # 8 x 16 pixels of the last frame (7)


def test_kept_statements_see_recreated_and_shadowed_names(tmp_path):
    """The cached kept statement re-validates its names when the tables change: an operand deleted
    and recreated at another value, and a variable shadowed by a new image, are both seen.
    """
    res = milk_bash(
        'mem.mk2Dim a 8 8 >/dev/null; @calc "a = 2; v = 3"\n'
        '@calc "o = a * 2"; @calc "o = a * 2"; @calc -v o1 "imean(o)"; echo "o1=$o1"\n'
        'mem.rm a >/dev/null; mem.mk2Dim a 8 8 >/dev/null; @calc "a = 7"\n'
        '@calc "o = a * 2"; @calc -v o2 "imean(o)"; echo "o2=$o2"\n'
        '@calc "p = a + v"; @calc "p = a + v"; @calc -v p1 "imean(p)"; echo "p1=$p1"\n'
        'mem.mk2Dim v 8 8 >/dev/null; @calc "v = 10"\n'
        '@calc "p = a + v"; @calc -v p2 "imean(p)"; echo "p2=$p2"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "o1") == "4" and value(res.stdout, "o2") == "14", out
    assert value(res.stdout, "p1") == "10" and value(res.stdout, "p2") == "17", out


@perf_timing
def test_dynamic_bound_slice_costs_about_a_literal_one(tmp_path: Path):
    """Self-triggered 128^2 units: `out[k:k+7,:] = src[k:k+7,:]*2` within 3x of the literal-bound
    statement per tick (the re-parse of the slice text used to make it ~8x)."""
    env, shm_dir = _isolated_shm_env({"OMP_NUM_THREADS": "1"})
    bindir = _install_dir() / "bin"
    script = str(bindir / "milk-script")

    def cnt0(stream):
        out = _run(
            [script, "-q", "-"],
            env,
            timeout=10,
            stdin=f"@get -v c s.{stream}.cnt0\necho $c\n",
        )
        return int(out.stdout.strip().splitlines()[-1])

    def us_per_tick(name, args):
        proc = subprocess.Popen(
            [
                str(bindir / "milk-fpsexec-milk-calc"),
                "-n",
                name,
                "-t",
                "out",
                *args,
                "exec",
            ],
            env=env,
            cwd=str(tmp_path),
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            time.sleep(0.5)
            _run(
                [script, "-q", "-"], env, timeout=10, stdin="mem.imsetsempost out -1\n"
            )
            time.sleep(0.8)
            a = cnt0("out")
            time.sleep(1.5)
            b = cnt0("out")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        assert b > a, f"{name}: no ticks"
        return 1.5e6 / (b - a)

    try:
        r = _run(
            [script, "-q", "-"],
            env,
            timeout=30,
            stdin="mem.mk2Dim src 128 128\nmem.mk2Dim out 128 128\n",
        )
        assert r.returncode == 0, r.stderr
        lit = us_per_tick("lit", ["-e", "out[0:7,:] = src[0:7,:]*2"])
        dyn = us_per_tick(
            "dyn", ["-p", "k:int=0", "-e", "out[k:k+7,:] = src[k:k+7,:]*2"]
        )
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)
    if lit > 20.0:
        pytest.skip(
            f"host too slow or loaded for a tick ratio (literal slice {lit:.1f} us)"
        )
    assert (
        dyn < 3.0 * lit
    ), f"dynamic bounds {dyn:.1f} us/tick vs literal {lit:.1f} us/tick"
