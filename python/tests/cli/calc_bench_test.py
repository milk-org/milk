"""Calculator speed: 30 statements against the old calculator's times, and kernel cache reuse.

Purpose:     keep the compiled passes faster than the calculator they replaced, and keep a
             statement run in a loop from compiling a new kernel at each call.
Does:        times 30 statements (element-wise, math, reductions) at 256^2 and 2048^2 with
             mcalc_prog_run() and a C timing helper, prints them next to the old calculator's
             reference times, and requires a 1.8x geometric-mean speedup on the element-wise
             ones at 256^2; prints the times of the spatial kernels (256^2) and of order
             statistics, argmax and FFT (256^2, 2048^2); checks that 50 slice bounds or
             shift/roll offsets, and runs on two CPUs, compile one cached .so.
Covers:      libmilkcalc: mcalc_prog_new() and mcalc_prog_run() (src/calc/run/mc_prog.c),
             mcalc_eval_str() (src/calc/mc_eval.c), the kernel cache (src/calc/jit/mc_cc.c,
             MILK_CALC_CACHE); @calc.
Uses:        conftest.py: milk_bash, _install_dir; ctypes on the install's libmilkscript.so;
             $CC (default gcc) to build the timing helper.
See also:    docs/guide/calculator.md, "Fusion & Cost Model"; python/tests/perf/calc_bash_bench.py.
Constraints: MILK_PERF_TESTS=0 skips the timing tests; the 1.8x check compares with fixed
             reference times (one thread, pinned, best of 7 runs); 300 s per test.
"""

from __future__ import annotations

import ctypes
import math
import os
import pathlib
import subprocess
import tempfile

import pytest

from milk.cliwrap import HAVE_CLI
from tests.cli.conftest import _install_dir, milk_bash

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.timeout(300),
]

perf_timing = pytest.mark.skipif(
    os.environ.get("MILK_PERF_TESTS", "1") == "0",
    reason="MILK_PERF_TESTS=0: timing tests off",
)

# (id, group, statement, base_iters_at_256)
BENCH_CASES: list[tuple[int, str, str, int]] = [
    (1, "elementwise", "C = A + B", 2000),
    (2, "elementwise", "C += B * 0.1", 2000),
    (3, "elementwise", "C = A * 2.5 + B * 0.5", 2000),
    (4, "elementwise", "C = (A - B) * G + O", 2000),
    (5, "elementwise", "C = (A * 2.0 + B * 3.0 - D) / 4.0", 2000),
    (6, "elementwise", "C = pow(A, 3) + 0.5 * pow(A, 2) - 1.2 * A + 0.3", 2000),
    (7, "elementwise", "U, V = A + B, A - B", 2000),
    (8, "elementwise", "C = ((A + B) * (G - O) + (A - B) * (G + O)) * 0.5", 2000),
    (9, "elementwise", "C = clamp(A, 0.3, 0.9)", 2000),
    (10, "elementwise", "C = where(A > B, A - B, 0.0)", 2000),
    (11, "elementwise", "C = where((A > 0.3) && (B < 0.8), A * B, 0.0)", 2000),
    (12, "elementwise", "C = soft_thresh(A - B, 0.1)", 2000),
    (13, "math", "C = sqrt(A)", 2000),
    (14, "math", "C = exp(-A * A)", 1000),
    (15, "math", "C = sin(A) + cos(B)", 1000),
    (16, "math", "C = atan2(B, A)", 1000),
    (17, "math", "C = hypot(A, B)", 2000),
    (18, "reduction", "v = imean(A)", 2000),
    (19, "reduction", "v = istd(A)", 2000),
    (20, "reduction", "v = dot(A, B)", 2000),
    (21, "reduction", "m, s, lo, hi = istats(A)", 2000),
    (22, "reduction", "v = irms(A - B)", 2000),
    (23, "reduction", "v = imean(A, M)", 2000),
    (24, "reduction", "C = (A - imean(A)) / istd(A)", 2000),
    (25, "reduction", "cx, cy, flux = icentroid(A, 0.5)", 2000),
    (26, "reduction", "C = A - imean(A, 0)", 2000),
    (27, "reduction", "C = A - imean(A, 1)", 2000),
    (28, "elementwise", "C[0:H,0:H] = A[0:H,0:H] * 2.0 + B[0:H,0:H]", 2000),
    (
        29,
        "elementwise",
        "C = 4.0*A - shift(A,1,0) - shift(A,-1,0) - shift(A,0,1) - shift(A,0,-1)",
        1000,
    ),
    (30, "elementwise", "local tmp = A - B; C = tmp * tmp + 0.5 * tmp", 2000),
]

# Stencil and library kernels, 256^2 only (medfilt7 alone takes ~83 ms per call there). Report
# only: the old-calculator baseline has no spatial cases.
SPATIAL_CASES: list[tuple[int, str, str, int]] = [
    (31, "spatial", "C = medfilt3(A)", 20),
    (32, "spatial", "C = medfilt5(A)", 5),
    (33, "spatial", "C = medfilt7(A)", 5),
    (34, "spatial", "C = boxmean(A, 2)", 20),
    (35, "spatial", "C = conv2(A, K)", 20),
    (36, "spatial", "C = imin3(A)", 50),
    (37, "spatial", "C = gauss(A, 2.0)", 50),
]

# Order statistics, argmax and FFT, 256^2 and 2048^2. Report only.
LIBRARY_CASES: list[tuple[int, str, str, int]] = [
    (38, "library", "v = imedian(A)", 20),
    (39, "library", "v = perc(A, 0.9)", 20),
    (40, "library", "px, py = iargmax(A)", 200),
    (41, "library", "F = fft(A)", 20),
    (42, "library", "R = ifft(F)", 20),
]

# Reference times of the old calculator (us, single thread, pinned, best of 7 runs)
OLD_REF_TIMES_US: dict[int, dict[int, float]] = {
    256: {
        1: 6.738,
        2: 11.215,
        3: 9.907,
        4: 18.611,
        5: 17.154,
        6: 22.904,
        7: 15.315,
        8: 32.717,
        9: 5.595,
        10: 20.094,
        11: 27.556,
        12: 23.196,
        13: 13.053,
        14: 38.345,
        15: 62.505,
        16: 51.771,
        17: 21.374,
        18: 6.706,
        19: 17.368,
        20: 4.490,
        21: 21.606,
        22: 21.403,
        23: 19.870,
        24: 32.950,
        25: 20.311,
        26: 21.098,
        27: 23.562,
        28: 5.899,
        29: 30.100,
        30: 22.348,
    },
    2048: {
        1: 1612.562,
        2: 1676.705,
        3: 2563.908,
        4: 3704.803,
        5: 3755.828,
        6: 2408.493,
        7: 3756.226,
        8: 4535.075,
        9: 955.424,
        10: 3058.681,
        11: 3603.214,
        12: 3281.507,
        13: 977.455,
        14: 3617.718,
        15: 4774.142,
        16: 5516.296,
        17: 2739.233,
        18: 449.081,
        19: 1163.288,
        20: 956.906,
        21: 1437.034,
        22: 2155.525,
        23: 1619.272,
        24: 3170.087,
        25: 1668.820,
        26: 2687.875,
        27: 3201.780,
        28: 467.738,
        29: 3414.578,
        30: 3156.735,
    },
}

_BENCH_HELPER_C = r"""
#include <time.h>

extern int mcalc_prog_run(void *p, char *err, unsigned long errsz);

double bench_prog_run(void *p, int iters, int rounds)
{
    char err[256];
    for (int w = 0; w < 2; w++)
    {
        if (mcalc_prog_run(p, err, sizeof(err)) != 0)
        {
            return -1.0;
        }
    }
    double best_us = 1e30;
    for (int r = 0; r < rounds; r++)
    {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int i = 0; i < iters; i++)
        {
            if (mcalc_prog_run(p, err, sizeof(err)) != 0)
            {
                return -1.0;
            }
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double dt = ((double) (t1.tv_sec - t0.tv_sec) * 1e6 +
                     (double) (t1.tv_nsec - t0.tv_nsec) * 1e-3) /
                    (double) iters;
        if (dt < best_us)
        {
            best_us = dt;
        }
    }
    return best_us;
}
"""

_BENCH_HANDLES: tuple[ctypes.CDLL, ctypes.CDLL] | None = None


def _get_bench_libs() -> tuple[ctypes.CDLL, ctypes.CDLL]:
    """Load `libmilkscript.so` and compile/load the C timing helper once per process."""
    global _BENCH_HANDLES
    if _BENCH_HANDLES is not None:
        return _BENCH_HANDLES

    os.environ["OMP_NUM_THREADS"] = "1"
    shmdir = tempfile.mkdtemp(prefix="milk_bench_shm_")
    os.environ["MILK_SHM_DIR"] = shmdir

    libdir = _install_dir() / "lib"
    script_so = ctypes.CDLL(str(libdir / "libmilkscript.so"), mode=ctypes.RTLD_GLOBAL)
    script_so.CLI_data_init.argtypes = []
    script_so.CLI_data_init.restype = ctypes.c_int
    script_so.setSHMdir.argtypes = []
    script_so.setSHMdir.restype = ctypes.c_int
    script_so.mcalc_eval_str.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    script_so.mcalc_eval_str.restype = ctypes.c_int
    script_so.mcalc_udef_try_parse.argtypes = [ctypes.c_char_p]
    script_so.mcalc_udef_try_parse.restype = ctypes.c_int
    script_so.mcalc_prog_new.argtypes = [
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    script_so.mcalc_prog_new.restype = ctypes.c_void_p
    script_so.mcalc_prog_free.argtypes = [ctypes.c_void_p]
    script_so.mcalc_prog_free.restype = None

    script_so.CLI_data_init()
    script_so.setSHMdir()

    helper_dir = pathlib.Path(tempfile.mkdtemp(prefix="milk_bench_helper_"))
    c_path = helper_dir / "bench_helper.c"
    so_path = helper_dir / "libbench_helper.so"
    c_path.write_text(_BENCH_HELPER_C, encoding="utf-8")
    subprocess.run(
        [
            os.environ.get("CC", "gcc"),
            "-O3",
            "-shared",
            "-fPIC",
            str(c_path),
            f"-L{libdir}",
            f"-Wl,-rpath,{libdir}",
            "-lmilkcalc",
            "-o",
            str(so_path),
        ],
        check=True,
    )
    helper_so = ctypes.CDLL(str(so_path))
    helper_so.bench_prog_run.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    helper_so.bench_prog_run.restype = ctypes.c_double

    _BENCH_HANDLES = (script_so, helper_so)
    return _BENCH_HANDLES


def _eval_stmt(script_so: ctypes.CDLL, stmt: str) -> None:
    """Evaluate a single calculator setup statement via `mcalc_eval_str`."""
    rc = ctypes.c_int(0)
    out = ctypes.create_string_buffer(256)
    ok = script_so.mcalc_eval_str(stmt.encode("utf-8"), 0, ctypes.byref(rc), out, 256)
    assert ok == 1 and rc.value == 0, f"mcalc_eval_str failed on {stmt!r}"


def _setup_bench_dataset(script_so: ctypes.CDLL, size: int) -> None:
    """Create and populate the 30-case benchmark dataset at @size x @size."""
    _eval_stmt(script_so, f"H = {size // 2 - 1}")
    for idx, name in enumerate(("A", "B", "C", "D", "G", "O", "M", "U", "V"), start=1):
        _eval_stmt(script_so, f"{name} = zeros({size}, {size})")
        if name in ("A", "B", "C", "D"):
            _eval_stmt(script_so, f"{name} = 0.1 + 0.8 * rand({name}, {idx})")
        elif name == "G":
            _eval_stmt(script_so, f"G = 0.5 + 0.5 * rand(G, {idx})")
        elif name == "O":
            _eval_stmt(script_so, f"O = 0.1 * rand(O, {idx})")
    _eval_stmt(script_so, "M = where(A > 0.3, 1.0, 0.0)")
    _eval_stmt(script_so, "K = zeros(5, 5)")
    _eval_stmt(script_so, "K = K + 0.04")
    script_so.mcalc_udef_try_parse(b"def soft_thresh(x, t) = where(abs(x) > t, x, 0.0)")


def _geomean(vals: list[float]) -> float:
    """Return the geometric mean of positive numbers @vals."""
    if not vals:
        return float("nan")
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def run_bench_suite(
    size: int, cases: list[tuple[int, str, str, int]] = BENCH_CASES
) -> dict[int, float]:
    """Run benchmark @cases at @size on `libmilkcalc`; return `{case_id: us}`."""
    script_so, helper_so = _get_bench_libs()
    _setup_bench_dataset(script_so, size)

    scale = (256.0 / float(size)) ** 2
    times: dict[int, float] = {}
    err = ctypes.create_string_buffer(256)
    for cid, _, stmt, base_iters in cases:
        iters = max(5, int(base_iters * scale))
        if "local " not in stmt:
            for seg in stmt.split(";"):
                seg = seg.strip()
                if seg:
                    _eval_stmt(script_so, seg)
        prog = script_so.mcalc_prog_new(stmt.encode("utf-8"), None, 0, err, 256)
        assert prog, f"mcalc_prog_new failed for {stmt!r}: {err.value.decode()}"
        us = helper_so.bench_prog_run(prog, iters, 7)
        script_so.mcalc_prog_free(prog)
        assert us > 0.0, f"bench_prog_run failed for case {cid} ({stmt!r})"
        times[cid] = float(us)
    return times


def format_bench_report(
    size: int,
    old_times: dict[int, float],
    new_times: dict[int, float] | None,
) -> str:
    """Format the per-case table and per-group geometric means for @size."""
    out: list[str] = []
    out.append(
        f"\n=== Calculator Benchmark ({size}x{size} float32, best of 7 runs, us) ==="
    )
    out.append(
        f"{'ID':>2} | {'Group':11} | {'Expression':54} | "
        f"{'old (us)':>10} | {'new (us)':>10} | {'old/new':>7}"
    )
    out.append("-" * 106)
    for cid, grp, stmt, _ in BENCH_CASES:
        t_old = old_times[cid]
        if (
            new_times is not None
            and cid in new_times
            and not math.isnan(new_times[cid])
        ):
            t_new = new_times[cid]
            ratio = f"{t_old / t_new:6.2f}x"
            new_s = f"{t_new:10.3f}"
        else:
            new_s = f"{'n/a':>10}"
            ratio = f"{'n/a':>7}"
        out.append(
            f"{cid:2d} | {grp:11} | {stmt:54.54} | {t_old:10.3f} | {new_s} | {ratio}"
        )

    out.append("-" * 106)
    for grp_label, grp_filter in (
        ("elementwise (1-12, 28-30)", lambda g: g == "elementwise"),
        ("math (13-17)", lambda g: g == "math"),
        ("reductions (18-27)", lambda g: g == "reduction"),
        ("all (1-30)", lambda _: True),
    ):
        ids = [cid for cid, g, _, _ in BENCH_CASES if grp_filter(g)]
        gm_old = _geomean([old_times[i] for i in ids])
        valid_ids = [
            i
            for i in ids
            if new_times is not None and i in new_times and not math.isnan(new_times[i])
        ]
        if new_times is not None and len(valid_ids) == len(ids):
            gm_new = _geomean([new_times[i] for i in ids])
            out.append(
                f"Geomean {grp_label:27}: old = {gm_old:9.3f} us | "
                f"new = {gm_new:9.3f} us | speedup = {gm_old / gm_new:5.2f}x"
            )
        else:
            out.append(
                f"Geomean {grp_label:27}: old = {gm_old:9.3f} us | new = not available"
            )
    return "\n".join(out)


@perf_timing
def test_single_cli_bench_invocation():
    """Verify single-statement benchmark timing on `C = A + B` at 256^2."""
    script_so, helper_so = _get_bench_libs()
    _setup_bench_dataset(script_so, 256)
    err = ctypes.create_string_buffer(256)
    prog = script_so.mcalc_prog_new(b"C = A + B", None, 0, err, 256)
    assert prog
    us = helper_so.bench_prog_run(prog, 50, 3)
    script_so.mcalc_prog_free(prog)
    assert us > 0.0


@perf_timing
def test_calc_spatial_bench():
    """Time the spatial kernels at 256^2 and print them (no reference to compare against)."""
    times = run_bench_suite(256, SPATIAL_CASES)
    print("\n=== Spatial kernels (256x256 float32, best of 7 runs, us) ===")
    for cid, _, stmt, _ in SPATIAL_CASES:
        print(f"{cid:2d} | {stmt:54.54} | {times[cid]:10.3f}")


@perf_timing
@pytest.mark.parametrize("size", [256, 2048])
def test_calc_library_bench(size: int):
    """Time order statistics, argmax and FFT and print them (no reference to compare against)."""
    times = run_bench_suite(size, LIBRARY_CASES)
    print(f"\n=== Library kernels ({size}x{size} float32, best of 7 runs, us) ===")
    for cid, _, stmt, _ in LIBRARY_CASES:
        print(f"{cid:2d} | {stmt:54.54} | {times[cid]:10.3f}")


@perf_timing
@pytest.mark.parametrize("size", [256, 2048])
def test_calc_bench(size: int):
    """Run all 30 benchmark cases at 256^2 and 2048^2 and compare against reference."""
    old_times = OLD_REF_TIMES_US[size]
    new_times = run_bench_suite(size)
    assert len(new_times) == 30

    report = format_bench_report(size, old_times, new_times)
    print(report)

    elem_ids = [cid for cid, grp, _, _ in BENCH_CASES if grp == "elementwise"]
    red_ids = [cid for cid, grp, _, _ in BENCH_CASES if grp == "reduction"]
    for cid in elem_ids + red_ids:
        t_new = new_times[cid]
        assert not math.isnan(t_new) and t_new > 0.0
    if size == 256:
        gm_old = _geomean([old_times[i] for i in elem_ids])
        gm_new = _geomean([new_times[i] for i in elem_ids])
        speedup = gm_old / gm_new
        assert speedup >= 1.8, (
            f"Elementwise geomean speedup at 256^2 is {speedup:.2f}x (< 1.80x): "
            f"old={gm_old:.3f} us, new={gm_new:.3f} us"
        )


def test_runtime_bounds_and_offsets_compile_single_kernel(tmp_path):
    """50 different slice bounds or shift/roll offsets compile exactly one .so in MILK_CALC_CACHE."""
    cache_slice = tmp_path / "cache_slice"
    cache_shift = tmp_path / "cache_shift"
    cache_slice.mkdir()
    cache_shift.mkdir()

    res_slice = milk_bash(
        "mem.mk2Dim a 128 128 >/dev/null; mem.mk2Dim o 128 128 >/dev/null\n"
        "for k in $(seq 0 49); do\n"
        '  @calc "o[k:k+15, 0:31] = a[k:k+15, 0:31] * 2.0 + 1.0"\n'
        "done\n",
        tmp_path,
        extra_env={"MILK_CALC_CACHE": str(cache_slice)},
    )
    assert res_slice.returncode == 0, res_slice.stdout + res_slice.stderr
    so_slice = list(cache_slice.glob("*.so"))
    assert (
        len(so_slice) == 1
    ), f"Expected 1 .so for 50 slice bounds, got {len(so_slice)}: {so_slice}"

    res_shift = milk_bash(
        "mem.mk2Dim a 64 64 >/dev/null; mem.mk2Dim o 64 64 >/dev/null\n"
        "for k in $(seq 0 49); do\n"
        '  @calc "o = shift(a, k, -k) + roll(a, k, 1)"\n'
        "done\n",
        tmp_path,
        extra_env={"MILK_CALC_CACHE": str(cache_shift)},
    )
    assert res_shift.returncode == 0, res_shift.stdout + res_shift.stderr
    so_shift = list(cache_shift.glob("*.so"))
    assert (
        len(so_shift) == 1
    ), f"Expected 1 .so for 50 shift/roll offsets, got {len(so_shift)}: {so_shift}"


def test_kernel_cache_is_shared_across_cores(tmp_path):
    """One statement run pinned to two different CPUs compiles one .so (the key has no APIC ID)."""
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 2:
        pytest.skip("needs two CPUs")
    cache = tmp_path / "cache"
    cache.mkdir()
    saved = os.sched_getaffinity(0)
    try:
        for cpu in cpus[:2]:
            os.sched_setaffinity(0, {cpu})
            res = milk_bash(
                'mem.mk2Dim a 64 64 >/dev/null; @calc "o = a * 3.0 + 1.0"',
                tmp_path,
                extra_env={"MILK_CALC_CACHE": str(cache)},
            )
            assert res.returncode == 0, res.stdout + res.stderr
    finally:
        os.sched_setaffinity(0, saved)
    so = list(cache.glob("*.so"))
    assert len(so) == 1, f"Expected 1 .so on CPUs {cpus[:2]}, got {len(so)}: {so}"
