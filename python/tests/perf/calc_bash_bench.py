#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Olivier Guyon et al
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Per-line and per-statement cost of milk-cli / milk-script in bash and calc mode.

Not a pytest test (no test_ prefix): a benchmark to run before and after a change.

    MILK_INSTALLDIR=/path/to/install python3 calc_bash_bench.py [--quick] [--cpu N]

Columns are microseconds per line (or per statement); lower is better. Each value is the best
of several runs. Per-statement costs subtract a 1-statement run from an N-statement run, so
process startup cancels out.
"""

import argparse
import os
import shutil
import subprocess
import time


def run(argv, script, env, reps):
    """Best wall-clock time (s) of feeding @script to @argv on stdin."""
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        p = subprocess.run(argv, input=script, text=True, capture_output=True, env=env)
        best = min(best, time.perf_counter() - t0)
        if p.returncode != 0:
            raise RuntimeError(f"{argv[-2:]} rc={p.returncode}: {p.stderr[-300:]}")
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="fewer lines and repetitions")
    ap.add_argument(
        "--cpu", type=int, default=2, help="CPU to pin statement benchmarks to"
    )
    args = ap.parse_args()

    inst = os.environ.get("MILK_INSTALLDIR")
    if not inst:
        raise SystemExit("MILK_INSTALLDIR is not set")
    bindir = os.path.join(inst, "bin")
    env = dict(os.environ, LD_LIBRARY_PATH=os.path.join(inst, "lib"))
    nlines = 200 if args.quick else 500
    reps = 3 if args.quick else 5

    cli = [os.path.join(bindir, "milk-cli"), "-q"]
    script = [os.path.join(bindir, "milk-script"), "-q", "-"]
    pin = ["taskset", "-c", str(args.cpu)] if shutil.which("taskset") else []
    env1 = dict(env, OMP_NUM_THREADS="1")

    print(f"# MILK_INSTALLDIR={inst}")
    print("## per line (us), stdin piped")
    lines = {
        "bash  ':'": ":\n" * nlines,
        "bash  'x=$((1+2))'": "x=$((1+2))\n" * nlines,
        "bash  '@calc -v a 1+3.2'": "@calc -v a 1+3.2\n" * nlines,
        "calc  'a=1+3.2'": "@calc\n" + "a=1+3.2\n" * nlines + "exit\n",
    }
    print(f"{'line':30s} {'milk-cli':>10s} {'milk-script':>12s}")
    base_c = run(pin + cli, "exit\n", env1, reps)
    base_s = run(pin + script, "exit\n", env1, reps)
    for label, body in lines.items():
        t_c = run(pin + cli, body + "exit\n", env1, reps)
        t_s = run(pin + script, body + "exit\n", env1, reps)
        per_c = (t_c - base_c) * 1e6 / nlines
        per_s = (t_s - base_s) * 1e6 / nlines
        print(f"{label:30s} {per_c:10.1f} {per_s:12.1f}")

    print(f"## per statement (us), in-process, pinned to CPU {args.cpu}, 1 thread")
    stmts = [
        ("1+3.2", 0),
        ("a=1+3.2", 0),
        ("imC = imA*2.5 + imB*0.5", 8),
        ("imC = imA*2.5 + imB*0.5", 128),
        ("imC = imA*2.5 + imB*0.5", 512),
        ("imC = sin(imA) + exp(imB)", 128),
        ("imC = sin(imA) + exp(imB)", 512),
        ("imC = imA^1.7", 512),
        ("imC = imA^3", 512),
        ("imD = imA*2.5 + imB*0.5", 512),
        ("v = irms(imA - imB)", 512),
        ("imC = where((ir(imA)<200)&&(imA>0), imA, 0)", 512),
        ("zC = expi(imA)", 512),
        ("zC = zA * conj(zB)", 512),
    ]
    print(f"{'statement':44s} {'size':>6s} {'us/stmt':>10s} {'ns/px':>8s}")
    for stmt, n in stmts:
        pre = "a=1\n"
        if n:
            pre += "".join(f"mem.mk2Dim {im} {n} {n}\n" for im in ("imA", "imB", "imC"))
            pre += '@calc "imA = imA + 0.3; imB = imB + 0.2"\n'
            if stmt.startswith("zC"):
                pre += '@calc "zA = cplx(imA, imB); zB = cplx(imB, imA)"\n'
        count = 60 if n >= 512 else (300 if args.quick else 1000)
        one = run(pin + script, pre + f'@calc --repl "{stmt}"\n', env1, reps)
        body = '@calc --repl "' + "; ".join([stmt] * count) + '"\n'
        many = run(pin + script, pre + body, env1, reps)
        us = (many - one) * 1e6 / (count - 1)
        npx = f"{us * 1e3 / (n * n):8.2f}" if n else f"{'':8s}"
        print(f"{stmt:44s} {(str(n) + '^2') if n else '':>6s} {us:10.1f} {npx}")


if __name__ == "__main__":
    main()
