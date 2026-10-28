#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Olivier Guyon et al
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Per-tick cost of milk.calc compute units, by statement kind.

Not a pytest test (no test_ prefix): a benchmark to run before and after a change.

    MILK_INSTALLDIR=/path/to/install python3 unit_tick_bench.py [--seconds S]

Each unit is triggered by the stream it writes ("self-triggered"), so it runs back to back with
no sleep and no producer: the rate of its output's cnt0 is the tick rate, and 1/rate the cost of
one tick (trigger wait, FPS sync, the statements, one output post). The first row, an 8x8
statement, is the loop's own floor. Statements run with OMP_NUM_THREADS=1 unless noted.
"""

import argparse
import os
import shutil
import subprocess
import tempfile
import time

ROWS = [
    ("8^2 kept (loop floor)", "o8", ["-e", "o8 = s8*2+1"], {}),
    ("128^2 kept: out = src*2+1", "out", ["-e", "out = src*2+1"], {}),
    ("128^2 two statements on out", "out", ["-e", "out = src*2+1; out = out - 1"], {}),
    (
        "128^2 kept + param + reduction",
        "out",
        ["-p", "g:float=2 v:float=0", "-e", "v = irms(src); out = src*g + v"],
        {},
    ),
    (
        "128^2 kept, sliced: out[0:64,:] = src[0:64,:]*2",
        "out",
        ["-e", "out[0:64,:] = src[0:64,:]*2"],
        {},
    ),
    (
        "128^2 kept, axis reduction: out = src - imean(src,1)",
        "out",
        ["-e", "out = src - imean(src, 1)"],
        {},
    ),
    ("1024^2 kept, 1 thread", "bigout", ["-e", "bigout = big*2+1"], {}),
    (
        "1024^2 kept, team policy",
        "bigout",
        ["-e", "bigout = big*2+1"],
        {"OMP_NUM_THREADS": None},
    ),
]


def run_script(bindir, env, text):
    """Run @text through milk-script; its stdout."""
    return subprocess.run(
        [os.path.join(bindir, "milk-script"), "-q", "-"],
        input=text,
        text=True,
        capture_output=True,
        env=env,
    ).stdout


def cnt0(bindir, env, stream):
    out = run_script(bindir, env, f"@get -v c s.{stream}.cnt0\necho $c\n")
    return int(out.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--seconds", type=float, default=3.0, help="measurement window per row"
    )
    args = ap.parse_args()

    inst = os.environ.get("MILK_INSTALLDIR")
    if not inst:
        raise SystemExit("MILK_INSTALLDIR is not set")
    bindir = os.path.join(inst, "bin")
    work = tempfile.mkdtemp(prefix="milk_unit_bench_")
    shm = os.path.join(work, "shm")
    os.makedirs(shm)
    base = dict(
        os.environ,
        LD_LIBRARY_PATH=os.path.join(inst, "lib"),
        MILK_SHM_DIR=shm,
        MILK_PROC_DIR=shm,
        HOME=work,
        OMP_NUM_THREADS="1",
    )
    run_script(
        bindir,
        base,
        "mem.mk2Dim s8 8 8\nmem.mk2Dim o8 8 8\nmem.mk2Dim src 128 128\nmem.mk2Dim out 128 128\n"
        "mem.mk2Dim big 1024 1024\nmem.mk2Dim bigout 1024 1024\n",
    )
    print(f"# MILK_INSTALLDIR={inst}")
    print(f"{'statements':44s} {'ticks/s':>10s} {'us/tick':>9s}")
    try:
        for i, (label, stream, unit_args, env_over) in enumerate(ROWS):
            env = dict(base)
            for k, v in env_over.items():
                if v is None:
                    env.pop(k, None)
                else:
                    env[k] = v
            name = f"b{i}"
            proc = subprocess.Popen(
                [
                    os.path.join(bindir, "milk-fpsexec-milk-calc"),
                    "-n",
                    name,
                    "-t",
                    stream,
                    *unit_args,
                    "exec",
                ],
                env=env,
                cwd=work,
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                time.sleep(0.5)
                run_script(bindir, env, f"mem.imsetsempost {stream} -1\n")
                time.sleep(1.0)
                a = cnt0(bindir, env, stream)
                time.sleep(args.seconds)
                b = cnt0(bindir, env, stream)
                rate = (b - a) / args.seconds
                us = 1e6 / rate if rate > 0 else float("nan")
                print(f"{label:44s} {rate:10.0f} {us:9.1f}")
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
