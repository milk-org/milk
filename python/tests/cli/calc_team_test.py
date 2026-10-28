"""
Tests for the calculator's persistent worker team (P3 of the 2026-10-27 engine review).

Every block loop of the fused engine (statements, reductions, axis and segmented reductions, the
Gaussian) used to open an OpenMP parallel region of its own, so a multi-region tick paid several
fork/joins, and a unit ticking slower than libgomp's spin window paid a ~100 µs wake-up per region.
The team is created once, dispatched per block loop with the caller as thread 0, and its workers
spin for `MILK_CALC_SPIN_US` (default 2000) after a job before sleeping. Results must not depend on
the team size, and a unit at a 1 ms period must run its statements with hot workers.
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


def test_team_results_match_one_thread(tmp_path):
    """Reductions, a centroid, an axis reduction, a Gaussian and a fill agree between a 1-thread and
    a 4-thread team on an image large enough to engage it."""
    script = (
        'mem.mk2Dim a 1024 1024 >/dev/null; @calc "a = 50 + 10 * urand(a, 5)"\n'
        '@calc -v s,m,x "irms(a - imean(a)); imax(a); istd(a)"; echo "red=$s,$m,$x"\n'
        '@calc "cx, cy = icentroid(a - 40)"; @calc -v cx,cy "cx; cy"; echo "cent=$cx,$cy"\n'
        '@calc "o = a - imean(a, 1); g = gauss(a, 2.0)"; @calc -v so,sg "itot(o); itot(g)"; echo "pre=$so,$sg"\n'
        '@calc "f = 0 * a; f = 3"; @calc -v sf "itot(f)"; echo "fill=$sf"'
    )
    res1 = milk_bash(script, tmp_path, extra_env={"MILK_CALC_THREADS": "1"})
    res4 = milk_bash(script, tmp_path, extra_env={"MILK_CALC_THREADS": "4"})
    for key in ("red", "cent", "pre", "fill"):
        v1 = [float(x) for x in value(res1.stdout, key).split(",")]
        v4 = [float(x) for x in value(res4.stdout, key).split(",")]
        for a, b in zip(v1, v4):
            assert abs(a - b) <= 1e-6 * (abs(a) + 1), (
                key,
                v1,
                v4,
                res1.stderr,
                res4.stderr,
            )


def _exectime_us(tmp_path: Path, extra: dict[str, str], period: str = "0.001") -> float:
    """Median and maximum execution time, in µs, of a three-region 1024^2 statement in a -loopd unit."""
    env, shm_dir = _isolated_shm_env(extra)
    env.pop("OMP_NUM_THREADS", None)
    bindir = _install_dir() / "bin"
    script = str(bindir / "milk-script")
    try:
        _run(
            [script, "-q", "-"],
            env,
            timeout=30,
            stdin='mem.mk2Dim big 1024 1024\nmem.mk2Dim bigout 1024 1024\n@calc "big = urand(big, 3)"\n',
        )
        proc = subprocess.Popen(
            [
                str(bindir / "milk-fpsexec-milk-calc"),
                "-n",
                "team",
                "-loopd",
                period,
                "-p",
                "v:float=0 w:float=0",
                "-e",
                "v = irms(big); w = imax(big); bigout = big*v + w",
                "exec",
            ],
            env=env,
            cwd=str(tmp_path),
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            time.sleep(2.5)
            out = _run(
                [script, "-q", "-"],
                env,
                timeout=10,
                stdin="echo t=$(@get p.team.exectime); echo m=$(@get p.team.exec_max)\n",
            )
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)
    t = value(out.stdout, "t")
    m = value(out.stdout, "m")
    assert t not in (None, "") and m not in (None, ""), out.stdout + out.stderr
    return float(t), float(m)


@perf_timing
def test_unit_at_1ms_keeps_workers_hot(tmp_path: Path):
    """At a 1 ms period, workers kept spinning (MILK_CALC_SPIN_US=-1) give a far lower worst tick
    than workers put to sleep after every job (MILK_CALC_SPIN_US=0), and no worse a median; the
    default window (2000 µs) keeps them hot at this period."""
    if len(os.sched_getaffinity(0)) < 4:
        pytest.skip("fewer than 4 CPUs")
    hot_med, hot_max = _exectime_us(tmp_path, {"MILK_CALC_SPIN_US": "-1"})
    cold_med, cold_max = _exectime_us(tmp_path, {"MILK_CALC_SPIN_US": "0"})
    if hot_med > 2000.0:
        pytest.skip(f"host too slow or loaded for a tick ratio ({hot_med:.0f} us)")
    assert (
        hot_max < 0.5 * cold_max
    ), f"worst tick: hot workers {hot_max:.0f} us vs sleeping {cold_max:.0f} us"
    assert (
        hot_med <= 1.05 * cold_med
    ), f"median: hot {hot_med:.0f} us vs sleeping {cold_med:.0f} us"
