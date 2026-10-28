"""
Tests for the parallel pre-stages of the 2026-10-27 engine review (P4, P5).

Axis reductions give every thread its own outputs (rows for axis 0, row groups across the planes
for axis 2, planes for axis 1 of a cube; a 2-D axis-1 reduction merges per-thread partials), with
op-specialized SIMD accumulation and no per-frame allocation, so a cube collapse over z is no longer
single-threaded. Segmented reductions gather bins in parallel and scatter an expression on per-thread
partials, `rprof` on an expression reuses a per-pixel bin table, and the Gaussian passes run rows
across the team. Results must equal the per-operator fallback with 1 thread and with 4. Found on the way: the eager
`rprof(...)` was dispatched as `pixunmap` (the tokenizer never marked the token).
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import pytest

from milk.cliwrap import HAVE_CLI
from tests.cli.conftest import _install_dir, _isolated_shm_env, _run

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.timeout(240),
]

perf_timing = pytest.mark.skipif(
    os.environ.get("MILK_PERF_TESTS", "1") == "0",
    reason="MILK_PERF_TESTS=0: timing tests off",
)

PARITY = """
mem.mk2Dim a2 1024 1024
mem.mk3Dim cube 256 256 12
mem.mk2Dim map 1024 1024
@calc "a2 = 100 + 10 * urand(a2, 7); cube = urand(cube, 9) - 0.5; map = floor(ir(a2) / 8)"
@calc "a2[3,5] = sqrt(-1); cube[7,9,2] = sqrt(-1)"
for op in itot imean imin imax istd ivar norm asum; do
  for ax in 0 1 2; do
    @calc "f = ${op}(cube, $ax); s = ${op}(cube[::1b], $ax)"; e=err; @calc -v e "imax(abs(f - s)) / (imax(abs(s)) + 1e-30)"
    echo "c_${op}_${ax}=$e"
  done
  for ax in 0 1; do
    @calc "f2 = ${op}(a2, $ax); s2 = ${op}(a2[::1b], $ax)"; e=err; @calc -v e "imax(abs(f2 - s2)) / (imax(abs(s2)) + 1e-30)"
    echo "a_${op}_${ax}=$e"
  done
done
@calc "o2 = a2 - imean(a2, 1); so2 = a2[::1b] - imean(a2[::1b], 1)"; e=err; @calc -v e "imax(abs(o2 - so2))"; echo "fused_ax1=$e"
@calc "r2 = pixunmap(a2 - 50, map); sr2 = pixunmap(a2[::1b] - 50, map[::1b])"; e=err; @calc -v e "imax(abs(r2 - sr2)) / (imax(abs(sr2)) + 1)"; echo "pixunmap=$e"
@calc "p2 = rprof(a2 * 2 - 1, 500, 520); sp2 = rprof(a2[::1b] * 2 - 1, 500, 520)"; e=err; @calc -v e "imax(abs(p2 - sp2)) / (imax(abs(sp2)) + 1)"; echo "rprof=$e"
@calc "pl = rprof(a2, 500, 520); spl = rprof(a2[::1b], 500, 520)"; e=err; @calc -v e "imax(abs(pl - spl)) / (imax(abs(spl)) + 1)"; echo "rprof_leaf=$e"
@calc "g2 = gauss(a2, 3.0); sg2 = gauss(a2[::1b], 3.0)"; e=err; @calc -v e "imax(abs(g2 - sg2)) / (imax(abs(sg2)) + 1)"; echo "gauss=$e"
"""


def _run_milk(
    script_text: str, tmp_path: Path, nthreads: int, timeout: float = 200
) -> dict[str, float]:
    f = tmp_path / f"run_{nthreads}.milk"
    f.write_text(script_text)
    env, shm_dir = _isolated_shm_env({"MILK_CALC_THREADS": str(nthreads)})
    env.pop("OMP_NUM_THREADS", None)
    try:
        exe = str(_install_dir() / "bin" / "milk-script")
        res = _run([exe, "-q", str(f)], env, timeout=timeout, cwd=str(tmp_path))
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)
    assert res.returncode == 0, res.stderr
    vals: dict[str, float] = {}
    for ln in res.stdout.splitlines():
        m = re.match(r"^([a-z_0-9]+)=([-+0-9.eE]+|nan|inf)$", ln)
        if m:
            vals[m.group(1)] = float(m.group(2))
    return vals


@pytest.mark.parametrize("nthreads", [1, 4])
def test_prestages_match_fallback(tmp_path: Path, nthreads: int):
    """Every axis op along every axis of a cube and a 2-D image (NaNs included), segmented reductions
    on leaves and expressions, and the Gaussian, equal the slice-forced fallback."""
    vals = _run_milk(PARITY, tmp_path, nthreads)
    assert len(vals) >= 8 * 5 + 5, vals
    bad = {k: v for k, v in vals.items() if not (v <= 1e-5)}
    assert not bad, bad


@perf_timing
def test_prestages_scale_with_threads(tmp_path: Path):
    """A cube collapse over z (1M outputs, serial before), a cube collapse over x, a 2048^2 Gaussian
    and a pixunmap of an expression run faster on 4 threads than on 1."""
    script = """
mem.mk2Dim a2 2048 2048
mem.mk3Dim cube 1024 1024 8
mem.mk2Dim map 2048 2048
@calc "a2 = 100 + 10 * urand(a2, 7); cube = urand(cube, 9); map = floor(ir(a2) / 4)"
@calc "m2 = imean(cube, 2); m0 = istd(cube, 0); g2 = gauss(a2, 3.0); r2 = pixunmap(a2 - 50, map)"
bench() { best=1e9; for rep in 1 2 3; do t0=$EPOCHREALTIME; for ((i=0; i<3; i++)); do @calc "$2"; done; t1=$EPOCHREALTIME; @calc -v best "min($best, ($t1 - $t0) * 1e3 / 3)"; done; echo "t_$1=$best"; }
bench ax2 "m2 = imean(cube, 2)"
bench ax0 "m0 = istd(cube, 0)"
bench gauss "g2 = gauss(a2, 3.0)"
bench pixunmap "r2 = pixunmap(a2 - 50, map)"
"""
    if len(os.sched_getaffinity(0)) < 4:
        pytest.skip("fewer than 4 CPUs")
    t1 = _run_milk(script, tmp_path, 1)
    t4 = _run_milk(script, tmp_path, 4)
    if t1["t_ax2"] > 60.0:
        pytest.skip(
            f"host too slow or loaded for a scaling ratio ({t1['t_ax2']:.1f} ms)"
        )
    for k in ("t_ax2", "t_ax0", "t_gauss", "t_pixunmap"):
        assert (
            t4[k] < 0.75 * t1[k]
        ), f"{k}: 1 thread {t1[k]:.2f} ms, 4 threads {t4[k]:.2f} ms"
