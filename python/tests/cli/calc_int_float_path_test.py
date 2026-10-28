"""
Tests for the float32 fused path on 8- and 16-bit integer images (P1 of the 2026-10-27 engine review).

`u8 i8 u16 i16` operands and destinations are exact in float32, so a statement over them runs the
float32 block kernels (twice the SIMD width of the double path a camera frame used to take). The
values must match the per-operator fallback exactly, integer destinations and slices included, a
stencil on an integer image keeps its pass in double, and a u16 statement must cost about what the
same statement on float images costs.
"""

from __future__ import annotations

import os
import shutil
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

SETUP = (
    'mem.mk2Dim "@S:cam" 64 48 >/dev/null; mem.mk2Dim dark 64 48 >/dev/null\n'
    '@calc "cam = 1000 + 3000 * urand(cam, 3); dark = 100 * urand(dark, 5); g = 2.5"\n'
    '@calc "camu = u16(cam); darku = u16(dark); cami = i16(cam - 2000); cam8 = u8(cam / 20); cams8 = i8(cam / 40 - 100)"\n'
)


def _report_line(out: str, stmt: str) -> str:
    lines = out.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().endswith(stmt) and i + 1 < len(lines):
            return lines[i + 1].strip()
    raise AssertionError(f"no report line for {stmt!r} in:\n{out}")


def test_small_int_images_run_float32_and_match_fallback(tmp_path):
    """u16/i16/u8/i8 operands: float32 pass, values equal to the slice-forced fallback."""
    res = milk_bash(
        SETUP
        + '@calc "o1 = (camu - darku) * g"; @calc "r1 = (camu[::1b] - darku[::1b]) * g"; @calc -v e1 "imax(abs(o1 - r1))"; echo "e1=$e1"\n'
        '@calc "o2 = sqrt(abs(cami)) + cam8 * 0.5 - cams8"; @calc "r2 = sqrt(abs(cami[::1b])) + cam8[::1b] * 0.5 - cams8[::1b]"\n'
        '@calc -v e2 "imax(abs(o2 - r2)) / (imax(abs(r2)) + 1)"; echo "e2=$e2"\n'
        '@calc -v m1,m2 "imean(camu - darku); istd(cami)"; @calc -v n1,n2 "imean(camu[::1b] - darku[::1b]); istd(cami[::1b])"; echo "red=$m1,$n1,$m2,$n2"\n'
        '@calc "o4 = camu[0:31,:] - darku[0:31,:]"; @calc "r4 = camu[0:31::1b,:] - darku[0:31::1b,:]"; @calc -v e4 "imax(abs(o4 - r4))"; echo "e4=$e4"\n'
        '@calc --explain "o1 = (camu - darku) * g; o3 = shift(camu, 1, 0) + 1; r3 = camu[0:31,:] * 2"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert (
        float(value(res.stdout, "e1")) == 0.0 and float(value(res.stdout, "e4")) == 0.0
    ), out
    assert float(value(res.stdout, "e2")) < 1e-6, out
    m1, n1, m2, n2 = (float(v) for v in value(res.stdout, "red").split(","))
    assert abs(m1 - n1) < 1e-9 * abs(n1) and abs(m2 - n2) < 1e-9 * abs(n2), out
    # a stencil on an integer image keeps its (merged) pass in double; a plain slice runs float32
    assert "double" in _report_line(res.stdout, "o1 = (camu - darku) * g"), out
    assert "float32" in _report_line(res.stdout, "r3 = camu[0:31,:] * 2"), out


def test_small_int_destinations_run_float32(tmp_path):
    """A u16 destination, whole or sliced, is written through the float32 pass with the C cast."""
    res = milk_bash(
        SETUP
        + '@calc "@S:o16 = u16(cam)"; @calc "o16 = (camu - darku) * g"; echo "rcw=$?"\n'
        '@calc "r1 = (camu[::1b] - darku[::1b]) * g"; @calc -v e3 "imax(abs(o16 - r1))"; echo "e3=$e3"\n'
        '@calc --explain "o16 = (camu - darku) * g"\n'
        '@calc "o16[0:31,:] = camu[0:31,:] + 7"; echo "rcs=$?"; @calc -v e5 "imax(abs(o16[0:31,:] - (camu[0:31::1b,:] + 7)))"; echo "e5=$e5"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "rcw") == "0" and value(res.stdout, "rcs") == "0", out
    assert (
        float(value(res.stdout, "e3")) <= 0.5
    ), out  # g = 2.5: the cast truncates a .5
    assert "float32" in _report_line(res.stdout, "o16 = (camu - darku) * g"), out
    assert float(value(res.stdout, "e5")) == 0.0, out


@perf_timing
def test_u16_statement_costs_about_the_float_statement(tmp_path: Path):
    """2048^2, one thread: `exp(uA*1e-4) * sqrt(uB)` on u16 inputs within 1.3x of the same on floats.

    The double path a u16 frame used to take has half the SIMD width, so it ran this at ~1.8x.
    """
    script = "\n".join(
        [
            "mem.mk2Dim imA 2048 2048",
            "mem.mk2Dim imB 2048 2048",
            "mem.mk2Dim imC 2048 2048",
            '@calc "imA = 1000 + 2000 * urand(imA, 7); imB = 500 * urand(imB, 13)"',
            '@calc "uA = u16(imA); uB = u16(imB)"',
            '@calc "imC = exp(imA * 1e-4) * sqrt(imB)"',
            '@calc "imD = exp(uA * 1e-4) * sqrt(uB)"',
            "for kind in f u; do",
            "  best=1e9",
            "  for rep in 1 2 3; do",
            "    t0=$EPOCHREALTIME",
            '    if [ $kind = f ]; then for ((i=0; i<8; i++)); do @calc "imC = exp(imA * 1e-4) * sqrt(imB)"; done',
            '    else for ((i=0; i<8; i++)); do @calc "imD = exp(uA * 1e-4) * sqrt(uB)"; done; fi',
            "    t1=$EPOCHREALTIME",
            '    @calc -v best "min($best, ($t1 - $t0) * 1e3 / 8)"',
            "  done",
            '  echo "T_${kind}=$best"',
            "done",
            '@calc "rD = exp(uA[::1b] * 1e-4) * sqrt(uB[::1b])"',
            '@calc -v err "imax(abs(imD - rD)) / (imax(abs(rD)) + 1)"; echo "err=$err"',
        ]
    )
    f = tmp_path / "u16_timing.milk"
    f.write_text(script + "\n")
    env, shm_dir = _isolated_shm_env({"OMP_NUM_THREADS": "1", "MILK_CALC_THREADS": "1"})
    try:
        exe = str(_install_dir() / "bin" / "milk-script")
        res = _run([exe, "-q", str(f)], env, timeout=120, cwd=str(tmp_path))
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)
    assert res.returncode == 0, res.stderr
    t_f = float(value(res.stdout, "T_f"))
    t_u = float(value(res.stdout, "T_u"))
    assert float(value(res.stdout, "err")) < 1e-5, res.stdout
    if t_f > 40.0:
        pytest.skip(
            f"host too slow or loaded for a timing ratio (float statement {t_f:.1f} ms)"
        )
    assert t_u < 1.3 * t_f, f"u16 statement {t_u:.2f} ms vs float {t_f:.2f} ms"
