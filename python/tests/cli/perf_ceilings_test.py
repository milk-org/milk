"""Asserted latency ceilings and fused-vs-fallback numerical parity tests."""

from __future__ import annotations

import math
import os
import random
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from milk.cliwrap import HAVE_CLI

from .conftest import _install_dir, _isolated_shm_env, _run

pytestmark = pytest.mark.skipif(
    not HAVE_CLI, reason="MILK compiled without CLI support"
)

# Timing assertions are host-dependent: MILK_PERF_TESTS=0 skips them (CI), parity tests still run.
perf_timing = pytest.mark.skipif(
    os.environ.get("MILK_PERF_TESTS", "1") == "0",
    reason="MILK_PERF_TESTS=0: timing tests off",
)


def _pin_cpu() -> int:
    """A CPU this process may run on (the third one when there are enough, as the benchmarks use)."""
    cpus = sorted(os.sched_getaffinity(0))
    return cpus[min(2, len(cpus) - 1)]


def _pinned_argv(exe: str, *args: str) -> list[str]:
    pin = ["taskset", "-c", str(_pin_cpu())] if shutil.which("taskset") else []
    return [*pin, exe, *args]


@perf_timing
def test_perf_latency_ceilings(tmp_path: Path):
    """Verify L0 shell/IPC call latencies and fused calculator statement ceilings (~2.5x Step 0)."""
    env, shm_dir = _isolated_shm_env({"OMP_NUM_THREADS": "1"})
    try:
        exe = str(_install_dir() / "bin" / "milk-script")
        stmts = [
            ("im8", "imC = imA*2.5 + imB*0.5", 8, 250, 22.0),
            ("im128", "imC = imA*2.5 + imB*0.5", 128, 200, 30.0),
            ("im512", "imC = imA*2.5 + imB*0.5", 512, 50, 230.0),
            ("sinexp512", "imC = sin(imA) + exp(imB)", 512, 40, 650.0),
            ("irms512", "v = irms(imA - imB)", 512, 50, 280.0),
            (
                "where512",
                "imC = where((ir(imA)<200)&&(imA>0), imA, 0)",
                512,
                40,
                400.0,
            ),
            ("expi512", "zC = expi(imA)", 512, 30, 700.0),
            ("zmul512", "zC = zA * conj(zB)", 512, 30, 700.0),
        ]

        lines = [
            "mem.mk2Dim @S:s_im 16 16",
            # Calibration check: 200 scalar @calc -v calls; skip on coverage/debug or loaded host
            "t0=$EPOCHREALTIME",
            "for ((i=0; i<200; i++)); do @calc -v a 1+3.2; done",
            "t1=$EPOCHREALTIME",
            '@calc "cal_us = ($t1 - $t0) * 1e6 / 200"',
            'echo "PERF_cal_us=@{cal_us}"',
            # L0 per-call benchmarks (best of 3 batches of 200)
            "best_calc=1e9; best_get=1e9; best_at=1e9",
            "for rep in 1 2 3; do",
            "  t0=$EPOCHREALTIME",
            "  for ((i=0; i<200; i++)); do @calc -v a 1+3.2; done",
            "  t1=$EPOCHREALTIME",
            '  @calc -v best_calc "min($best_calc, ($t1 - $t0) * 1e6 / 200)"',
            "  t0=$EPOCHREALTIME",
            "  for ((i=0; i<200; i++)); do @get -v w s.s_im.xsize; done",
            "  t1=$EPOCHREALTIME",
            '  @calc -v best_get "min($best_get, ($t1 - $t0) * 1e6 / 200)"',
            "  t0=$EPOCHREALTIME",
            "  for ((i=0; i<200; i++)); do x=@{s.s_im.xsize}; done",
            "  t1=$EPOCHREALTIME",
            '  @calc -v best_at "min($best_at, ($t1 - $t0) * 1e6 / 200)"',
            "done",
            'echo "PERF_l0_calc=$best_calc"',
            'echo "PERF_l0_get=$best_get"',
            'echo "PERF_l0_at=$best_at"',
        ]

        cur_n = 0
        for key, stmt, n, count, _ceil in stmts:
            if n != cur_n:
                cur_n = n
                lines.append(
                    f"mem.mk2Dim imA {n} {n}; mem.mk2Dim imB {n} {n}; mem.mk2Dim imC {n} {n}"
                )
                lines.append('@calc "imA = imA + 0.3; imB = imB + 0.2"')
                lines.append('@calc "zA = cplx(imA, imB); zB = cplx(imB, imA)"')
            many = "; ".join([stmt] * count)
            lines.extend(
                [
                    f"best_{key}=1e9",
                    "for rep in 1 2 3; do",
                    f'  @calc --repl "{stmt}" >/dev/null',
                    "  t0=$EPOCHREALTIME",
                    f'  @calc --repl "{many}" >/dev/null',
                    "  t1=$EPOCHREALTIME",
                    f'  @calc -v best_{key} "min($best_{key}, ($t1 - $t0) * 1e6 / {count})"',
                    "done",
                    f'echo "PERF_{key}=$best_{key}"',
                ]
            )

        f = tmp_path / "bench.milk"
        f.write_text("\n".join(lines) + "\n")
        res = _run(_pinned_argv(exe, "-q", str(f)), env, timeout=45, cwd=str(tmp_path))
        assert res.returncode == 0, res.stderr

        vals: dict[str, float] = {}
        for ln in res.stdout.splitlines():
            if ln.startswith("PERF_") and "=" in ln:
                k, v = ln[5:].split("=", 1)
                vals[k.strip()] = float(v.strip())

        if vals.get("cal_us", 999.0) > 25.0:
            pytest.skip(
                f"Host loaded or unoptimized build (cal_us={vals.get('cal_us'):.1f} us)"
            )

        assert vals["l0_calc"] <= 15.0, f"@calc -v too slow: {vals['l0_calc']:.1f} us"
        assert vals["l0_get"] <= 15.0, f"@get -v too slow: {vals['l0_get']:.1f} us"
        assert vals["l0_at"] <= 18.0, f"x=@{{...}} too slow: {vals['l0_at']:.1f} us"

        for key, stmt, n, _count, ceiling in stmts:
            got = vals[key]
            assert (
                got <= ceiling
            ), f"{stmt} ({n}^2): {got:.1f} us > ceiling {ceiling:.1f} us"
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)


def test_fused_vs_fallback_parity(tmp_path: Path):
    """Check fused evaluator vs slice-forced per-operator fallback on 60 random expressions."""
    env, shm_dir = _isolated_shm_env({"OMP_NUM_THREADS": "1"})
    try:
        rng = random.Random(20261023)
        atoms = ["A", "B", "C", "0.75", "1.25", "ix(A)/16.0", "ir(A)/16.0"]
        unaries = [
            "sin",
            "cos",
            "exp",
            "atan",
            "sqrt",
            "abs",
            "floor",
            "ceil",
            "sinh",
            "tanh",
        ]
        binops = ["+", "-", "*", "/"]
        redops = ["itot", "imean", "irms", "imin", "imax", "asum", "norm"]

        def gen_expr(depth: int) -> str:
            if depth <= 0 or rng.random() < 0.25:
                return rng.choice(atoms)
            kind = rng.random()
            if kind < 0.35:
                fn = rng.choice(unaries)
                sub = gen_expr(depth - 1)
                if fn == "sqrt":
                    return f"sqrt(abs({sub}) + 0.1)"
                if fn in ("exp", "sinh"):
                    return f"{fn}(clamp({sub}, -2.0, 2.0))"
                return f"{fn}({sub})"
            if kind < 0.80:
                op = rng.choice(binops)
                lhs = gen_expr(depth - 1)
                rhs = gen_expr(depth - 1)
                if op == "/":
                    return f"(({lhs}) / (abs({rhs}) + 0.5))"
                return f"(({lhs}) {op} ({rhs}))"
            if kind < 0.90:
                p = rng.choice(["2", "3", "0.5", "1.5"])
                sub = gen_expr(depth - 1)
                return f"((abs({sub}) + 0.2) ^ {p})"
            c1 = gen_expr(depth - 1)
            c2 = gen_expr(depth - 1)
            t_br = gen_expr(depth - 1)
            f_br = gen_expr(depth - 1)
            if rng.random() < 0.5:
                return f"where(({c1}) > ({c2}), {t_br}, {f_br})"
            return f"((({c1}) <= ({c2})) ? ({t_br}) : ({f_br}))"

        lines = [
            "mem.mk2Dim A 16 16",
            "mem.mk2Dim B 16 16",
            "mem.mk2Dim C 16 16",
            '@calc "A = 0.2 + 0.8 * urand(A, 11); B = 0.3 + 0.7 * urand(B, 22); C = -0.4 + 0.9 * urand(C, 33)"',
            '@calc "Ad = f64(A); Bd = f64(B); Cd = f64(C)"',
        ]

        for idx in range(50):
            expr = gen_expr(3)
            if not any(v in expr for v in ("A", "B", "C")):
                expr = f"(A + ({expr}))"
            # Replace A/B/C on the fallback side with A[::1b]: a bin-1 slice copies the image (plain
            # slices are read as views by the fused engine), so the expression takes the
            # per-operator path
            fb_expr = (
                expr.replace("ix(A)", "ix(__A)")
                .replace("ir(A)", "ir(__A)")
                .replace("A", "A[::1b]")
                .replace("B", "B[::1b]")
                .replace("C", "C[::1b]")
                .replace("__A[::1b]", "A")
            )
            lines.append(
                f'@calc "imF = {expr}; imS = {fb_expr}"; '
                f'@calc -v err_{idx} "imax(abs(imF - imS)) / (imax(abs(imS)) + 1)"; '
                f'echo "PAR_e32_{idx}=$err_{idx}"'
            )

            if idx < 10:
                expr64 = expr.replace("A", "Ad").replace("B", "Bd").replace("C", "Cd")
                fb64 = (
                    expr64.replace("ix(Ad)", "ix(__Ad)")
                    .replace("ir(Ad)", "ir(__Ad)")
                    .replace("Ad", "Ad[::1b]")
                    .replace("Bd", "Bd[::1b]")
                    .replace("Cd", "Cd[::1b]")
                    .replace("__Ad[::1b]", "Ad")
                )
                rop = redops[idx % len(redops)]
                lines.append(
                    f'@calc "imFd = {expr64}; imSd = {fb64}"; '
                    f'@calc -v ed_{idx} "imax(abs(imFd - imSd))"; '
                    f'@calc -v rd_{idx} "abs({rop}({expr64}) - {rop}({fb64}))"; '
                    f'echo "PAR_e64_{idx}=$ed_{idx}"; '
                    f'echo "PAR_r64_{idx}=$rd_{idx}"'
                )

        f = tmp_path / "parity.milk"
        f.write_text("\n".join(lines) + "\n")
        exe = str(_install_dir() / "bin" / "milk-script")
        res = _run([exe, "-q", str(f)], env, timeout=30, cwd=str(tmp_path))
        assert res.returncode == 0, res.stderr

        checked = 0
        for ln in res.stdout.splitlines():
            if not ln.startswith("PAR_") or "=" not in ln:
                continue
            k, v = ln[4:].split("=", 1)
            assert v != "", f"empty value for {k}; stderr:\n{res.stderr}"
            err = float(v)
            checked += 1
            if k.startswith("e32_"):
                assert (
                    err <= 1e-5
                ), f"float32 fused vs fallback relative mismatch ({k}={err})"
            else:
                assert err <= 1e-11, f"float64 fused vs fallback mismatch ({k}={err})"
        assert checked == 70
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)


@perf_timing
def test_milk_calc_threads_scaling(tmp_path: Path):
    """Verify MILK_CALC_THREADS=1 vs MILK_CALC_THREADS=4 parity and scaling on 2048^2."""
    script_lines = [
        "mem.mk2Dim imA 2048 2048",
        "mem.mk2Dim imB 2048 2048",
        "mem.mk2Dim imC 2048 2048",
        '@calc "imA = 0.5 + 0.5 * urand(imA, 7); imB = 0.2 + 0.5 * urand(imB, 13)"',
        '@calc "imC = imA^1.7"',
        "best_ms=1e9",
        "for rep in 1 2 3; do",
        "  t0=$EPOCHREALTIME",
        '  for ((i=0; i<8; i++)); do @calc "imC = imA^1.7"; done',
        "  t1=$EPOCHREALTIME",
        '  @calc -v best_ms "min($best_ms, ($t1 - $t0) * 1e3 / 8)"',
        "done",
        '@calc -v v_rms "irms(imC - imB)"',
        'echo "OMP_ms=$best_ms"',
        'echo "OMP_rms=$v_rms"',
    ]
    f = tmp_path / "omp_scale.milk"
    f.write_text("\n".join(script_lines) + "\n")
    exe = str(_install_dir() / "bin" / "milk-script")

    results: dict[int, tuple[float, float]] = {}
    for nthr in (1, 4):
        env, shm_dir = _isolated_shm_env({"MILK_CALC_THREADS": str(nthr)})
        env.pop("OMP_NUM_THREADS", None)
        try:
            res = _run([exe, "-q", str(f)], env, timeout=30, cwd=str(tmp_path))
            assert res.returncode == 0, res.stderr
            ms = 0.0
            rms = 0.0
            for ln in res.stdout.splitlines():
                if ln.startswith("OMP_ms="):
                    ms = float(ln.split("=", 1)[1])
                elif ln.startswith("OMP_rms="):
                    rms = float(ln.split("=", 1)[1])
            assert ms > 0.0
            results[nthr] = (ms, rms)
        finally:
            shutil.rmtree(shm_dir, ignore_errors=True)

    ms1, rms1 = results[1]
    ms4, rms4 = results[4]
    assert (
        abs(rms1 - rms4) <= 1e-6
    ), f"1-thread vs 4-thread RMS mismatch: {rms1} vs {rms4}"
    if len(os.sched_getaffinity(0)) >= 4 and ms1 < 30.0:
        assert (
            ms4 < ms1 * 0.75
        ), f"Expected 4-thread speedup on 2048^2 imA^1.7: 1t={ms1:.2f}ms, 4t={ms4:.2f}ms"


def test_stmt_cache_invalidation(tmp_path: Path):
    """Verify @calc LRU statement cache hits with live scalar updates and invalidates on resize/retype."""
    env, shm_dir = _isolated_shm_env({"OMP_NUM_THREADS": "1"})
    try:
        lines = [
            "mem.mk2Dim imA 8 8; mem.mk2Dim imB 8 8; mem.mk2Dim imC 8 8",
            '@calc "imA = 2.0; imB = 3.0; k = 4.0"',
            '@calc "imC = imA * k + imB"',
            '@calc "k = 10.0"',
            '@calc "imC = imA * k + imB"',
            '@calc -v v1 "imean(imC)"',
            "mem.mk2Dim imA 16 16; mem.mk2Dim imB 16 16; mem.mk2Dim imC 16 16",
            '@calc "imA = 1.5; imB = 0.5"',
            '@calc "imC = imA * k + imB"',
            '@calc -v v2 "itot(imC)"',
            '@calc "imA = f64(imA); imB = f64(imB); imC = f64(imC); k = 2.0"',
            '@calc "imC = imA * k + imB"',
            '@calc -v v3 "imean(imC)"',
            'echo "CACHE_v1=$v1"',
            'echo "CACHE_v2=$v2"',
            'echo "CACHE_v3=$v3"',
        ]
        f = tmp_path / "cache_inv.milk"
        f.write_text("\n".join(lines) + "\n")
        exe = str(_install_dir() / "bin" / "milk-script")
        res = _run([exe, "-q", str(f)], env, timeout=20, cwd=str(tmp_path))
        assert res.returncode == 0, res.stderr
        out = dict(
            ln[6:].split("=", 1)
            for ln in res.stdout.splitlines()
            if ln.startswith("CACHE_") and "=" in ln
        )
        assert abs(float(out["v1"]) - 23.0) < 1e-6
        assert abs(float(out["v2"]) - 15.5 * 256.0) < 1e-4
        assert abs(float(out["v3"]) - 3.5) < 1e-12
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)


def _milk_script_out(tmp_path: Path, lines: list[str], prefix: str) -> dict[str, str]:
    """Run @lines in milk-script with an isolated SHM dir; return its `<prefix>key=value` lines."""
    env, shm_dir = _isolated_shm_env({"OMP_NUM_THREADS": "1"})
    try:
        f = tmp_path / "run.milk"
        f.write_text("\n".join(lines) + "\n")
        exe = str(_install_dir() / "bin" / "milk-script")
        res = _run([exe, "-q", str(f)], env, timeout=60, cwd=str(tmp_path))
        assert res.returncode == 0, res.stdout + res.stderr
        n = len(prefix)
        return dict(
            ln[n:].split("=", 1)
            for ln in res.stdout.splitlines()
            if ln.startswith(prefix) and "=" in ln
        )
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)


def test_calc_temporaries_reused_without_page_faults(tmp_path: Path):
    """A statement run again gets its temporaries back: zC = zA * conj(zB) on 2100x2100 complex images
    (35 MB, more than the 32 MiB that milk's malloc policy keeps) faults no page in once warm. Before,
    each statement allocated, zeroed and freed a fresh 35 MB temporary (8600 page faults).
    """
    out = _milk_script_out(
        tmp_path,
        [
            'mem.mk2Dim "@L:imA" 2100 2100 >/dev/null; mem.mk2Dim "@L:imB" 2100 2100 >/dev/null',
            '@calc "imA = imA + 0.3; imB = imB + 0.2"',
            '@calc "zA = cplx(imA, imB); zB = cplx(imB, imA)"',
            '@calc "zC = zA * conj(zB)"; @calc "zC = zA * conj(zB)"',
            "read -r -a st < /proc/$PPID/stat; f0=${st[9]}",  # milk-script runs @calc; $$ is its bash
            'for i in $(seq 10); do @calc "zC = zA * conj(zB)"; done',
            "read -r -a st < /proc/$PPID/stat; f1=${st[9]}",
            'echo "POOL_faults=$(( (f1 - f0) / 10 ))"',
            'echo "POOL_re=@{imean(real(zC))}"',
        ],
        "POOL_",
    )
    assert int(out["faults"]) < 100, out
    assert abs(float(out["re"]) - 0.12) < 1e-6, out


def test_calc_reused_temporaries_give_fresh_results(tmp_path: Path):
    """Complex operations whose temporaries come back from the pool, not zeroed, write every element:
    products, negation, expi, scaling, parts, and a result renamed into a new image. The pool keeps at
    most four temporaries (listed as _tmpcalcpool<k>)."""
    out = _milk_script_out(
        tmp_path,
        [
            'mem.mk2Dim "@L:a" 64 64 >/dev/null; mem.mk2Dim "@L:b" 64 64 >/dev/null',
            '@calc "a = 3; b = 2"',
            '@calc "za = cplx(a, b); zb = cplx(b, a)"',
            'for i in 1 2 3; do @calc "zc = za * conj(zb)"; done',
            'echo "R_mul=@{imean(real(zc))} @{imean(imag(zc))}"',
            'for i in 1 2; do @calc "zc = -za"; done',
            'echo "R_neg=@{imean(real(zc))} @{imean(imag(zc))}"',
            'for i in 1 2; do @calc "zc = expi(a)"; done',
            'echo "R_expi=@{imean(real(zc))} @{imean(imag(zc))}"',
            'for i in 1 2; do @calc "zc = za * 2"; done',
            'for i in 1 2; do @calc "r = real(zc)"; @calc "q = imag(zc)"; done',
            'echo "R_parts=@{imean(r)} @{imean(q)}"',
            '@calc "zd = za + zb"',
            'echo "R_new=@{imean(real(zd))} @{imean(imag(zd))}"',
            'for n in 8 16 24 32 40 48; do mem.mk2Dim "@L:s" $n $n >/dev/null; @calc "s = 1"; '
            '@calc "zs = cplx(s, s)"; @calc "zs = zs * conj(zs)"; done',
            'echo "R_pool=$(mem.listim | grep -c _tmpcalcpool)"',
        ],
        "R_",
    )

    def pair(key: str) -> list[float]:
        return [float(v) for v in out[key].split()]

    assert pair("mul") == [12.0, -5.0], out  # (3 + 2i)(2 - 3i)
    assert pair("neg") == [-3.0, -2.0], out
    assert pair("expi") == pytest.approx([math.cos(3.0), math.sin(3.0)], abs=1e-6), out
    assert pair("parts") == [6.0, 4.0], out
    assert pair("new") == [5.0, 5.0], out
    assert 1 <= int(out["pool"]) <= 4, out


def test_kept_slash_eq_and_scalar_reductions_and_exprs(tmp_path: Path):
    """Verify kept `/=`, scalar reductions, and scalar expressions in @calc and milk.calc."""
    env, shm_dir = _isolated_shm_env({"OMP_NUM_THREADS": "1"})
    try:
        prog_file = tmp_path / "prog.calc"
        prog_file.write_text(
            "out = cam - ref\n"
            "out /= 2.0\n"
            "rms = irms(cam - ref)\n"
            "dotv = dot(cam, ref)\n"
            "k = k + 1\n"
            "g += 0.5 * imean(out)\n"
        )
        lines = [
            'mem.mk2Dim "@S:cam" 8 8; mem.mk2Dim "@S:ref" 8 8; mem.mk2Dim "@S:acc" 8 8',
            '@calc "cam = 6.0 + (ix(cam) - 3.5); ref = 2.0 + (ix(ref) - 3.5); acc = 12.0; div = 2.0"',
            '@calc "acc /= div"',
            '@calc "div = 3.0"',
            '@calc "acc /= div"',
            '@calc "acc /= (cam - ref)"',
            '@calc -v acc_m "imean(acc)"',
            'milk-fpsexec-milk-calc -n u_kept -p "k:int=0 rms:float=0 g:float=1.0 dotv:float=0" -f prog.calc exec >/dev/null 2>&1',
            "milk-fpsexec-milk-calc -n u_kept run >/dev/null 2>&1",
            '@calc -v exp_dot "dot(cam, ref)"',
            'echo "KEPT_acc=$acc_m"',
            'echo "KEPT_out=$(@get s.out.mean)"',
            'echo "KEPT_k=$(@get f.u_kept.k)"',
            'echo "KEPT_rms=$(@get f.u_kept.rms)"',
            'echo "KEPT_dot=$(@get f.u_kept.dotv)"',
            'echo "KEPT_exp_dot=$exp_dot"',
            'echo "KEPT_g=$(@get f.u_kept.g)"',
        ]
        f = tmp_path / "kept_step4.milk"
        f.write_text("\n".join(lines) + "\n")
        exe = str(_install_dir() / "bin" / "milk-script")
        res = _run([exe, "-q", str(f)], env, timeout=20, cwd=str(tmp_path))
        assert res.returncode == 0, res.stderr
        out = dict(
            ln[5:].split("=", 1)
            for ln in res.stdout.splitlines()
            if ln.startswith("KEPT_") and "=" in ln
        )
        assert abs(float(out["acc"]) - 0.5) < 1e-6
        assert abs(float(out["out"]) - 2.0) < 1e-6
        assert int(out["k"]) == 2
        assert abs(float(out["rms"])) < 1e-6
        assert abs(float(out["dot"]) - float(out["exp_dot"])) < 1e-4
        assert abs(float(out["g"]) - 3.0) < 1e-6
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)


def test_waitfor_pinned_stream_sem_reuse_and_cleanup(tmp_path: Path):
    """Verify @waitfor pins stream mapping, reuses semaphore slot across calls, and releases on exit."""
    env, shm_dir = _isolated_shm_env({"OMP_NUM_THREADS": "1"})
    try:
        lines = [
            'mem.mk2Dim "@S:wf_s" 8 8',
            # Advance cnt0 to 100 so @waitfor --cnt <= 100 returns immediately after pinning/checking
            'for ((i=0; i<100; i++)); do @calc "wf_s = wf_s + 1"; done',
            "t0=$EPOCHREALTIME",
            "for ((i=1; i<=50; i++)); do @waitfor --cnt $i s.wf_s; done",
            "t1=$EPOCHREALTIME",
            '@calc -v wf_us "($t1 - $t0) * 1e6 / 50"',
            'echo "WF_us=$wf_us"',
        ]
        f = tmp_path / "wf_pin.milk"
        f.write_text("\n".join(lines) + "\n")
        exe = str(_install_dir() / "bin" / "milk-script")
        res = _run([exe, "-q", str(f)], env, timeout=20, cwd=str(tmp_path))
        assert res.returncode == 0, res.stderr
        wf_us = 0.0
        for ln in res.stdout.splitlines():
            if ln.startswith("WF_us="):
                wf_us = float(ln.split("=", 1)[1])
        assert wf_us > 0.0
        # Verify that after the process exits, another process can still claim all semaphore slots
        check_lines = [
            "@waitfor --cnt 1 s.wf_s",
            'echo "WF_ok=1"',
        ]
        f2 = tmp_path / "wf_check.milk"
        f2.write_text("\n".join(check_lines) + "\n")
        res2 = _run([exe, "-q", str(f2)], env, timeout=10, cwd=str(tmp_path))
        assert res2.returncode == 0, res2.stderr
        assert "WF_ok=1" in res2.stdout
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)


def test_stream_stats_cache_and_simd(tmp_path: Path):
    """Verify s.<im>.{min,max,mean,std,var,tot,count} cache reuse and invalidation on frame update."""
    env, shm_dir = _isolated_shm_env({"OMP_NUM_THREADS": "1"})
    try:
        lines = [
            'mem.mk2Dim "@S:st_im" 512 512',
            '@calc "st_im = 2.0 + (ix(st_im) >= 0)"',
            "v_min=@{s.st_im.min}",
            "t0=$EPOCHREALTIME",
            "for ((i=0; i<25; i++)); do",
            "  a=@{s.st_im.min}; b=@{s.st_im.max}; c=@{s.st_im.mean}",
            "  d=@{s.st_im.std}; e=@{s.st_im.var}; g=@{s.st_im.tot}; h=@{s.st_im.count}",
            "done",
            "t1=$EPOCHREALTIME",
            '@calc -v cached_us "($t1 - $t0) * 1e6 / 25"',
            '@calc "st_im = st_im * 3.0"',
            "v_min2=@{s.st_im.min}",
            "v_max2=@{s.st_im.max}",
            "v_mean2=@{s.st_im.mean}",
            'echo "ST_min1=$v_min"',
            'echo "ST_min2=$v_min2"',
            'echo "ST_max2=$v_max2"',
            'echo "ST_mean2=$v_mean2"',
            'echo "ST_cached_us=$cached_us"',
        ]
        f = tmp_path / "st_cache.milk"
        f.write_text("\n".join(lines) + "\n")
        exe = str(_install_dir() / "bin" / "milk-script")
        res = _run([exe, "-q", str(f)], env, timeout=20, cwd=str(tmp_path))
        assert res.returncode == 0, res.stderr
        out = dict(
            ln[3:].split("=", 1)
            for ln in res.stdout.splitlines()
            if ln.startswith("ST_") and "=" in ln
        )
        assert abs(float(out["min1"]) - 2.0) < 1e-6
        assert abs(float(out["min2"]) - 6.0) < 1e-6
        assert abs(float(out["max2"]) - 9.0) < 1e-6
        assert abs(float(out["mean2"]) - 7.5) < 1e-6
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)


def test_grab_lockfree_and_dropped_keyword(tmp_path: Path):
    """Verify @grab copies frames outside the command lock and records .kw.DROPPED / .k.DROPPED."""
    env, shm_dir = _isolated_shm_env({"OMP_NUM_THREADS": "1"})
    try:
        lines = [
            'mem.mk2Dim "@S:g_src" 16 16',
            '(@sleep 0.02; for ((i=1; i<=4; i++)); do @calc "g_src = $i"; @sleep 0.015; done) &',
            "bg=$!",
            '@grab -t 2.0 s.g_src 4 "@S:g_cube"',
            "wait $bg",
            'echo "GRAB_drop_kw=$(@get s.g_cube.kw.DROPPED)"',
            'echo "GRAB_drop_k=$(@get s.g_cube.k.DROPPED)"',
            'echo "GRAB_nz=$(@get s.g_cube.zsize)"',
            'echo "GRAB_mean=$(@get s.g_cube.mean)"',
        ]
        f = tmp_path / "grab_drop.milk"
        f.write_text("\n".join(lines) + "\n")
        exe = str(_install_dir() / "bin" / "milk-script")
        res = _run([exe, "-q", str(f)], env, timeout=20, cwd=str(tmp_path))
        assert res.returncode == 0, res.stderr
        out = dict(
            ln[5:].split("=", 1)
            for ln in res.stdout.splitlines()
            if ln.startswith("GRAB_") and "=" in ln
        )
        assert int(out["drop_kw"]) == 0
        assert int(out["drop_k"]) == 0
        assert int(out["nz"]) == 4
        assert abs(float(out["mean"]) - 2.5) < 1e-6
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)


def test_piped_milk_cli_batch_latency_and_parity(tmp_path: Path):
    """Verify piped milk-cli batches stdin lines (within ~15 us/line) and handles @calc mode toggles."""
    env, shm_dir = _isolated_shm_env({"OMP_NUM_THREADS": "1"})
    try:
        cli_exe = str(_install_dir() / "bin" / "milk-cli")
        # 1. Correctness: mixed bash + @calc mode toggle + multi-line loop + exit
        piped_script = (
            "x=10\n"
            "@calc\n"
            "a = x + 2.5\n"
            "b = a * 2.0\n"
            "exit\n"
            "for i in 1 2 3; do\n"
            "  x=$((x + i))\n"
            "done\n"
            'echo "PIPED_res=$x:@{b}"\n'
            "exit\n"
        )
        res = _run(
            [cli_exe, "-q"], env, timeout=15, stdin=piped_script, cwd=str(tmp_path)
        )
        assert res.returncode == 0, res.stderr
        assert "PIPED_res=16:25" in res.stdout

        # 2. Per-line latency ceiling for 400 piped lines in milk-cli -q
        nlines = 400
        bench_script = (
            "t0=$EPOCHREALTIME\n"
            "@calc\n"
            + ("a=1+3.2\n" * nlines)
            + "exit\n"
            + "t1=$EPOCHREALTIME\n"
            + f'@calc -v per_us "($t1 - $t0) * 1e6 / {nlines}"\n'
            + 'echo "PIPED_us=$per_us"\n'
            + "exit\n"
        )
        res_b = _run(
            _pinned_argv(cli_exe, "-q"),
            env,
            timeout=15,
            stdin=bench_script,
            cwd=str(tmp_path),
        )
        assert res_b.returncode == 0, res_b.stderr
        per_line_us = float(
            next(
                ln.split("=", 1)[1]
                for ln in res_b.stdout.splitlines()
                if ln.startswith("PIPED_us=")
            )
        )
        assert (
            per_line_us < 20.0
        ), f"piped milk-cli calc line too slow: {per_line_us:.1f} us/line"
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)


def test_openblas_lapacke_solvers_and_matmul(tmp_path: Path):
    """Verify BLAS/LAPACKE matmul, gesv, posv, gels, solve, inv, and pinv accuracy and speed."""
    env, shm_dir = _isolated_shm_env(
        {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    )
    try:
        lines = [
            # 1. 64x64 linear systems & SVD pseudo-inverse residual checks
            "mem.mk2Dim A 64 64",
            "mem.mk2Dim B 64 4",
            '@calc "A = eye(64) * 10.0 + urand(A, 11) * 0.1"',
            '@calc "B = urand(B, 22)"',
            '@calc "X_lu = gesv(A, B)"',
            '@calc -v err_lu "imax(abs(matmul(A, X_lu) - B))"',
            '@calc "S = matmul(transpose(A), A); X_chol = posv(S, B)"',
            '@calc -v err_chol "imax(abs(matmul(S, X_chol) - B))"',
            '@calc "X_qr = gels(A, B)"',
            '@calc -v err_qr "imax(abs(matmul(A, X_qr) - B))"',
            '@calc "A_inv = inv(A)"',
            '@calc -v err_inv "imax(abs(matmul(A, A_inv) - eye(64)))"',
            '@calc "A_pinv = pinv(A, 1e-10)"',
            '@calc -v err_pinv "imax(abs(matmul(A, A_pinv) - eye(64)))"',
            # 2. 256x256 matmul timing check
            "mem.mk2Dim M1 256 256",
            "mem.mk2Dim M2 256 256",
            "mem.mk2Dim M3 256 256",
            '@calc "M1 = urand(M1, 1); M2 = urand(M2, 2)"',
            "t0=$EPOCHREALTIME",
            'for ((i=0; i<10; i++)); do @calc "M3 = matmul(M1, M2)"; done',
            "t1=$EPOCHREALTIME",
            '@calc -v mm_us "($t1 - $t0) * 1e6 / 10"',
            'echo "LA_err_lu=$err_lu"',
            'echo "LA_err_chol=$err_chol"',
            'echo "LA_err_qr=$err_qr"',
            'echo "LA_err_inv=$err_inv"',
            'echo "LA_err_pinv=$err_pinv"',
            'echo "LA_mm_us=$mm_us"',
        ]
        f = tmp_path / "la_test.milk"
        f.write_text("\n".join(lines) + "\n")
        exe = str(_install_dir() / "bin" / "milk-script")
        res = _run(_pinned_argv(exe, "-q", str(f)), env, timeout=25, cwd=str(tmp_path))
        assert res.returncode == 0, res.stderr
        out = dict(
            ln[3:].split("=", 1)
            for ln in res.stdout.splitlines()
            if ln.startswith("LA_") and "=" in ln
        )
        assert float(out["err_lu"]) < 1e-4
        assert float(out["err_chol"]) < 1e-2
        assert float(out["err_qr"]) < 1e-4
        assert float(out["err_inv"]) < 1e-4
        assert float(out["err_pinv"]) < 1e-4
        assert float(out["mm_us"]) < 5000.0
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)


def test_non_pow2_coord_stats_and_fused_complex_part(tmp_path: Path):
    """Verify non-power-of-2 coordinate/reduction kernels and fused real/imag/abs/arg on complex images."""
    env, shm_dir = _isolated_shm_env({"OMP_NUM_THREADS": "1"})
    try:
        lines = [
            # 1. Non-power-of-2 (100 x 137) coordinates, reductions, and element-wise kernels
            "mem.mk2Dim npA 100 137",
            "mem.mk2Dim npB 100 137",
            '@calc "npA = sin(itheta(npA, 49.5, 68.0)) + ir(npA, 49.5, 68.0) * 0.01"',
            '@calc "npB = sin(itheta(npA[::1b], 49.5, 68.0)) + ir(npA[::1b], 49.5, 68.0) * 0.01"',
            '@calc -v err_coord "imax(abs(npA - npB))"',
            '@calc -v err_rms "abs(irms(npA^2 + 0.5) - irms(npA[::1b]^2 + 0.5))"',
            # 2. Fused FI_CPART (real, imag, abs, arg) in float32 and float64 vs non-fused complex expr
            '@calc "zA = cplx(npA + 1.5, npB + 0.5); zB = cplx(npA * 0.5, npB * 0.25)"',
            '@calc "cF = abs(zA)^2 + real(zB) * 0.5 - imag(zA) + cos(arg(zA))"',
            '@calc "cS = abs(zA + 0)^2 + real(zB + 0) * 0.5 - imag(zA + 0) + cos(arg(zA + 0))"',
            '@calc -v err_cf32 "imax(abs(cF - cS))"',
            '@calc -v err_red32 "abs(itot(abs(zA)^2) - itot(abs(zA + 0)^2))"',
            '@calc "zAd = cplx(f64(npA) + 1.5, f64(npB) + 0.5); zBd = cplx(f64(npA) * 0.5, f64(npB) * 0.25)"',
            '@calc "cFd = abs(zAd)^2 + real(zBd) * 0.5 - imag(zAd) + cos(arg(zAd))"',
            '@calc "cSd = abs(zAd + 0)^2 + real(zBd + 0) * 0.5 - imag(zAd + 0) + cos(arg(zAd + 0))"',
            '@calc -v err_cf64 "imax(abs(cFd - cSd))"',
            '@calc -v err_red64 "abs(itot(abs(zAd)^2) - itot(abs(zAd + 0)^2))"',
            'echo "NP_coord=$err_coord"',
            'echo "NP_rms=$err_rms"',
            'echo "NP_cf32=$err_cf32"',
            'echo "NP_red32=$err_red32"',
            'echo "NP_cf64=$err_cf64"',
            'echo "NP_red64=$err_red64"',
        ]
        f = tmp_path / "np_cplx.milk"
        f.write_text("\n".join(lines) + "\n")
        exe = str(_install_dir() / "bin" / "milk-script")
        res = _run([exe, "-q", str(f)], env, timeout=20, cwd=str(tmp_path))
        assert res.returncode == 0, res.stderr
        out = dict(
            ln[3:].split("=", 1)
            for ln in res.stdout.splitlines()
            if ln.startswith("NP_") and "=" in ln
        )
        assert float(out["coord"]) <= 1e-6
        assert float(out["rms"]) <= 1e-6
        assert float(out["cf32"]) <= 1e-5
        assert float(out["red32"]) <= 1e-1
        assert float(out["cf64"]) <= 1e-12
        assert float(out["red64"]) <= 1e-9
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)


@perf_timing
def test_unit_tick_ceilings(tmp_path: Path):
    """Self-triggered milk.calc units (see python/tests/perf/unit_tick_bench.py): the loop floor, a
    128^2 kept statement single-threaded, and a 1024^2 kept statement under the team policy.
    """
    env, shm_dir = _isolated_shm_env({"OMP_NUM_THREADS": "1"})
    bindir = _install_dir() / "bin"
    script = str(bindir / "milk-script")

    def cnt0(e, stream):
        out = _run(
            [script, "-q", "-"],
            e,
            timeout=10,
            stdin=f"@get -v c s.{stream}.cnt0\necho $c\n",
        )
        return int(out.stdout.strip().splitlines()[-1])

    def us_per_tick(name, stream, args, e):
        proc = subprocess.Popen(
            [
                str(bindir / "milk-fpsexec-milk-calc"),
                "-n",
                name,
                "-t",
                stream,
                *args,
                "exec",
            ],
            env=e,
            cwd=str(tmp_path),
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            time.sleep(0.5)
            _run(
                [script, "-q", "-"],
                e,
                timeout=10,
                stdin=f"mem.imsetsempost {stream} -1\n",
            )
            time.sleep(0.8)
            a = cnt0(e, stream)
            time.sleep(2.0)
            b = cnt0(e, stream)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        assert b > a, f"{name}: no ticks"
        return 2.0e6 / (b - a)

    try:
        r = _run(
            [script, "-q", "-"],
            env,
            timeout=30,
            stdin="mem.mk2Dim s8 8 8\nmem.mk2Dim o8 8 8\nmem.mk2Dim src 128 128\n"
            "mem.mk2Dim out 128 128\nmem.mk2Dim big 1024 1024\nmem.mk2Dim bigout 1024 1024\n",
            cwd=str(tmp_path),
        )
        assert r.returncode == 0, r.stderr
        floor = us_per_tick("tk0", "o8", ["-e", "o8 = s8*2+1"], env)
        kept = us_per_tick("tk1", "out", ["-e", "out = src*2+1"], env)
        env_policy = dict(env)
        env_policy.pop("OMP_NUM_THREADS", None)
        big = us_per_tick("tk2", "bigout", ["-e", "bigout = big*2+1"], env_policy)
        assert floor <= 2.0, f"loop floor {floor:.2f} us/tick"
        assert kept <= 6.0, f"128^2 kept statement {kept:.2f} us/tick"
        ceiling = 150.0 if len(os.sched_getaffinity(0)) >= 4 else 400.0
        assert big <= ceiling, f"1024^2 kept statement {big:.1f} us/tick (policy)"
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)
