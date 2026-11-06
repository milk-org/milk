"""30-case calculator engine micro-benchmark at 256^2 and 2048^2 (§5 Phase 3, §6.4)."""

from __future__ import annotations

import math
import os
import pathlib
import subprocess

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


def _calcdiff_bin() -> pathlib.Path:
    """Return path to installed `milk-calcdiff` executable."""
    return _install_dir() / "bin" / "milk-calcdiff"


def _geomean(vals: list[float]) -> float:
    """Return the geometric mean of positive numbers @vals."""
    if not vals:
        return float("nan")
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def run_bench_suite(size: int, engine: str) -> dict[int, float] | None:
    """Run all 30 benchmark cases at @size on @engine; return `{case_id: us}` or None."""
    exe = _calcdiff_bin()
    scale = (256.0 / float(size)) ** 2
    lines: list[str] = []
    for _, _, stmt, base_iters in BENCH_CASES:
        iters = max(5, int(base_iters * scale))
        lines.append(f"{iters}\t{stmt}")

    proc = subprocess.run(
        [str(exe), "--bench", "--size", str(size), "--engine", engine],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    if proc.returncode == 2 and "not available" in proc.stdout:
        return None
    assert proc.returncode == 0, (
        f"milk-calcdiff --bench --size {size} --engine {engine} failed "
        f"(rc={proc.returncode}):\n{proc.stderr}"
    )
    out_lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    assert len(out_lines) == len(
        BENCH_CASES
    ), f"Expected {len(BENCH_CASES)} timing lines, got {len(out_lines)}: {out_lines}"
    return {BENCH_CASES[i][0]: float(out_lines[i]) for i in range(len(BENCH_CASES))}


def run_bench_both(size: int) -> tuple[dict[int, float], dict[int, float]]:
    """Run all 30 benchmark cases at @size on both engines with interleaved rounds."""
    exe = _calcdiff_bin()
    scale = (256.0 / float(size)) ** 2
    lines: list[str] = []
    for _, _, stmt, base_iters in BENCH_CASES:
        iters = max(5, int(base_iters * scale))
        lines.append(f"{iters}\t{stmt}")

    proc = subprocess.run(
        [str(exe), "--bench", "--size", str(size), "--engine", "both"],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, (
        f"milk-calcdiff --bench --size {size} --engine both failed "
        f"(rc={proc.returncode}):\n{proc.stderr}"
    )
    out_lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    assert len(out_lines) == len(
        BENCH_CASES
    ), f"Expected {len(BENCH_CASES)} timing lines, got {len(out_lines)}: {out_lines}"
    old_times: dict[int, float] = {}
    new_times: dict[int, float] = {}
    for i, ln in enumerate(out_lines):
        cid = BENCH_CASES[i][0]
        parts = ln.split("\t")
        assert len(parts) == 2, f"Expected <old>\\t<new>, got {ln!r}"
        old_times[cid] = float(parts[0])
        new_times[cid] = float(parts[1])
    return old_times, new_times


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
    """Verify single-statement `milk-calcdiff --bench --size 256 --iters 50 --engine old`."""
    exe = _calcdiff_bin()
    proc = subprocess.run(
        [
            str(exe),
            "--bench",
            "--size",
            "256",
            "--iters",
            "50",
            "--engine",
            "old",
            "C = A + B",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    val = float(proc.stdout.strip())
    assert val > 0.0


@perf_timing
@pytest.mark.parametrize("size", [256, 2048])
def test_calc_engine_bench(size: int):
    """Run all 30 benchmark cases at 256^2 and 2048^2 and record old and new times."""
    old_times, new_times = run_bench_both(size)
    assert len(old_times) == 30
    for cid, us in old_times.items():
        assert us > 0.0, f"Case {cid} at {size}^2 returned non-positive time {us}"

    report = format_bench_report(size, old_times, new_times)
    print(report)

    assert new_times is not None
    elem_ids = [cid for cid, grp, _, _ in BENCH_CASES if grp == "elementwise"]
    for cid in elem_ids:
        t_old = old_times[cid]
        t_new = new_times[cid]
        assert not math.isnan(
            t_new
        ), f"Elementwise case {cid} returned NaN on new engine"
        assert t_new <= t_old * 1.05, (
            f"Elementwise case {cid} at {size}^2 is >5% slower than old: "
            f"old={t_old:.3f} us, new={t_new:.3f} us"
        )
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
        extra_env={"MILK_CALC_ENGINE": "new", "MILK_CALC_CACHE": str(cache_slice)},
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
        extra_env={"MILK_CALC_ENGINE": "new", "MILK_CALC_CACHE": str(cache_shift)},
    )
    assert res_shift.returncode == 0, res_shift.stdout + res_shift.stderr
    so_shift = list(cache_shift.glob("*.so"))
    assert (
        len(so_shift) == 1
    ), f"Expected 1 .so for 50 shift/roll offsets, got {len(so_shift)}: {so_shift}"
