"""
Regression tests for the 2026-11-06 review of the JIT, calc and CLI.

Fix-first items:

- A reduction hoisted out of a fused pass reads its input after the statement that writes it.
- `@calc --explain` compiles and prints the plan; it does not run the statements.
- A string value is printed and measured as itself (never looked up as a variable name), and a
  stored string stays text (`s = "1+2"` is not evaluated).
- NaN comparisons follow IEEE rules in the shell and on the server.
- `hypot` of large image values does not overflow.
- An identifier longer than 255 characters is an error, not a crash of the session server.
- A module-command argument too long for its token buffer fails the command instead of being cut.
- Tab completion of `f.<fps>.` reuses one FPS connection instead of mapping the FPS on every call.

Request-level robustness:
- Ctrl-C stops a running `@calc` at its next loop iteration (status 130), and the session answers
  other requests while it runs.
- A crash inside a calculator request aborts that request only; the session and its images remain.
- A live unit edit that does not compile leaves the previous statements running.

One calculator front end:
- The calculator parses the whole `@calc` text: a syntax error anywhere runs nothing, blocks may span
  lines, and properties are read when a statement runs and assigned like variables (`+=`).
- Entity paths may hold `-` in a stream name (`s.my-cam.cnt0`).
- A `def` call is inlined: reductions and divisions by a reduction of a parameter work; a random
  argument used twice, a built-in name, and a `def` inside `if` are errors.
- `--explain` leaves the session's defs alone and creates no `hist` ring.
- `pi`, `nan`, `inf` are reserved; printed digits are per shell, and never round a stored value.

Consolidation:
- `-l file.c` compiles like a JIT kernel: same compiler (`MILK_CALC_CC`), flags (shared with
  milk-cc; NaN tests work), checked cache directory (`MILK_CALC_CACHE`), and the compiler's error
  line in the message.
- `imedian`/`perc` select (no full sort) and keep their index rule; `iargmax`/`iargmin` keep their
  NaN, tie and infinity rules.

Shared tables:
- Argument values read alike in module commands, unit command lines, `@args` and `@set` (calculator
  literals), and words bind by one rule; a bool given to a unit reaches its variable.
- Stream properties, stream modifiers and @ keywords each come from one table: `s.X.list`, completion,
  `@help` and the docs agree with it.

Language rules:
- A condition is a number: a string is an error, NaN is false, in `@test`, `@assert` and `if` alike.
- A name is an image first, then a bash variable, then a calculator variable; inside brackets `end`
  is the axis end whatever variable has that name.
- `wrap_pi` maps into (-pi, pi] for scalars and images; an integer literal beyond 64 bits is a
  double; a name longer than 79 characters is an error; `trunc(x)` truncates and `irms` is the RMS.
- A failure exits 1, a syntax, name or argument error 2 (`@test`: 2 for any error); errors in text of
  several lines name the line; a `def` body is checked when defined at `@calc`.
- Removed spellings fail naming the replacement: ln, fabs, urand, grand, nrm2, trunc(x, lo, hi), the
  `_PI`-style constants, `if`/`for` ... `end`, `--tokens`.

Kernel speed, same results:
- `%`, `fmod` and `wrap_pi` vectorize and still equal C's fmod/remainder (NaN beyond |x/y| = 2^52).
- 8- and 16-bit casts and stores, and `x & c`, give the bits of the 64-bit path on float and double
  images; a one-pixel-wide operand, a tuple swap and merged self-reads keep their values.
- `iargmax`/`iargmin`/`iamax` of a float image equal those of its double copy (NaN, ties, -inf).

One copy of each piece:
- Extension call arguments bind by the module-command rules, at run time and for the unit's FPS
  alike (an operator's positional parameter, a positional after a named one); too many is an error;
  `-l` without statements exposes the library's parameters, not the built-in kernels'.
- milk.bash stops, saying why, when its builtins cannot load (no fallback to milk-call shims).
- Shell text is quoted by one lexer (msl_scan), calculator text by the calculator's: in calc mode a
  comment is cut but not a `#` in a string, and one quoted word is unwrapped, not `"4" + "5"`.
- A variable is in one store, a number or a string, whichever was assigned last; a unit's string
  parameter takes a string assignment, and a number parameter refuses one.
- One completer for bash and the prompt: keyword options and words come from the lexicon;
  calculator text completes function names (with their `(`), also inside quotes and `@{...}`.

Request path:
- A shell process keeps one connection to its session between requests; a subshell opens its own;
  a script that takes the connection's fd number, or a server that restarts, costs a reconnect only.
- One thread for the server's life captures a module command's output for the log (stdout not a
  terminal): all of it reaches the shell, through a pipe or into a file.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import struct
import subprocess

import pytest

from milk.cliwrap import HAVE_CLI
from tests.cli.conftest import (
    _install_dir,
    milk_bash,
    milk_cli_stdin,
    pty_session,
    value,
)

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.timeout(120),
]

CAM = 'mem.mk2Dim "@S:cam" 8 8 >/dev/null; @calc "cam = 2"\n'


def test_fused_reduction_reads_this_block_value(tmp_path):
    """In one block, `imean(y)` sees the `y` written by the statement before it, not the old one."""
    res = milk_bash(
        CAM + '@calc "y = cam * 0"\n'
        '@calc "for k = 1:1 { y = cam * 3; z = y - imean(y) }"; echo "rc=$?"\n'
        '@calc -v zm "imean(z)"; echo "zm=$zm"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "rc") == "0", out
    assert value(res.stdout, "zm") == "0", out


def test_explain_does_not_run(tmp_path):
    res = milk_bash(
        CAM + '@calc --explain "cam = cam + 100" >/dev/null; echo "rc=$?"\n'
        '@calc -v m "imean(cam)"; echo "m=$m"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "rc") == "0", out
    assert value(res.stdout, "m") == "2", out


def test_strings_are_values_not_names_or_math(tmp_path):
    res = milk_bash(
        'myvar=SECRETVALUE; echo "lit=$(@calc \'"myvar"\')"\n'
        '@calc "abc = 42"; echo "len=$(@calc \'strlen("abc")\')"\n'
        '@calc \'s = "1+2"\'; echo "s=$(@calc s)"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "lit") == "myvar", out
    assert value(res.stdout, "len") == "3", out
    assert value(res.stdout, "s") == "1+2", out


def test_nan_comparisons_follow_ieee(tmp_path):
    """Shell-side scalars and server-side expressions (they read an image) agree."""
    res = milk_bash(
        CAM + 'echo "eq=$(@calc "nan == nan")"; echo "ne=$(@calc "nan != nan")"\n'
        'echo "ge=$(@calc "nan >= 1")"\n'
        'echo "srv_eq=$(@calc "(imean(cam) + nan) == (imean(cam) + nan)")"\n'
        'echo "srv_ge=$(@calc "imean(cam) + nan >= 1")"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    for key, want in (
        ("eq", "0"),
        ("ne", "1"),
        ("ge", "0"),
        ("srv_eq", "0"),
        ("srv_ge", "0"),
    ):
        assert value(res.stdout, key) == want, f"{key}: {out}"


def test_image_hypot_does_not_overflow(tmp_path):
    res = milk_bash(
        'mem.mk2Dim a 4 4 >/dev/null; @calc "a = 1e20"; @calc "h = hypot(a, a)"\n'
        '@calc -v hm "imax(h)"; echo "hm=$hm"',
        tmp_path,
    )
    hm = value(res.stdout, "hm")
    assert hm is not None, res.stdout + res.stderr
    assert float(hm) == pytest.approx(math.sqrt(2.0) * 1e20, rel=1e-6)


def test_long_identifier_is_an_error_not_a_crash(tmp_path):
    res = milk_bash(
        CAM + "L=$(printf 'x%.0s' {1..600})\n"
        '@calc "cam + $L" 2>err.txt; echo "rc=$?"\n'
        '@calc -v cm "imean(cam * 3)"; echo "cm=$cm"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "rc") not in (None, "0"), out
    assert "identifier too long" in (tmp_path / "err.txt").read_text(), out
    assert value(res.stdout, "cm") == "6", out


def test_long_module_argument_fails_instead_of_being_cut(tmp_path):
    res = milk_bash(
        "mem.mk2Dim im 4 4 >/dev/null\n"
        "P=\"$PWD/$(printf 'd%.0s' {1..220})/out.fits\"\n"
        'iofits.saveFITS im "$P" --no-run 2>err.txt; echo "rc=$?"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "rc") == "1", out
    assert "longer than 199 characters" in (tmp_path / "err.txt").read_text(), out


def test_fps_completion_keeps_one_mapping(tmp_path):
    """Six TABs on `@get f.cmpl9.` leave at most one mapping of the FPS in any session process."""
    (tmp_path / "prog.calc").write_text("out9 = cam * 2\n")
    steps = [
        ("line", 'mem.mk2Dim "@S:cam" 8 8 >/dev/null', 0.5),
        (
            "line",
            "milk-fpsexec-milk-calc -n cmpl9 -t cam -f prog.calc fpsinit >/dev/null",
            1.5,
        ),
        ("raw", "@get f.cmpl9.", 0.3),
        *[("raw", "\t", 0.3)] * 6,
        ("raw", "\x15", 0.2),
        (
            "line",
            "m=0; for p in $(pgrep -s 0); do c=$(grep -c cmpl9.fps.shm /proc/$p/maps);"
            ' ((c > m)) && m=$c; done; echo "fpsmaps=$m"',
            1.0,
        ),
    ]
    out = pty_session(steps, tmp_path)
    assert value(out, "fpsmaps") == "1", out[-2000:]


def test_ctrl_c_stops_a_running_calc_while_the_session_answers(tmp_path):
    res = milk_bash(
        'mem.mk2Dim im 8 8 >/dev/null; @calc "im = 3"; t0=$EPOCHREALTIME\n'
        '( @calc "for i = 1:100000 { for j = 1:100000 { x = i + j } }" 2>loop.err\n'
        '  echo "loop_rc=$?" ) & pid=$!\n'
        "sleep 1; t1=$EPOCHREALTIME\n"
        '@test "imean(im) > 2"; echo "test_rc=$?"\n'
        'echo "test_ms=$(( (${EPOCHREALTIME/./} - ${t1/./}) / 1000 ))"\n'
        "kill -INT $pid; wait $pid\n"
        'echo "loop_ms=$(( (${EPOCHREALTIME/./} - ${t0/./}) / 1000 ))"\n'
        '@calc -v m "imean(im * 2)"; echo "after=$m"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "test_rc") == "0", out
    assert int(value(res.stdout, "test_ms")) < 500, out  # served while the loop runs
    assert value(res.stdout, "loop_rc") == "130", out
    assert int(value(res.stdout, "loop_ms")) < 5000, out
    assert "interrupted" in (tmp_path / "loop.err").read_text(), out
    assert value(res.stdout, "after") == "6", out


def test_a_crash_in_calc_keeps_the_session(tmp_path):
    (tmp_path / "crash.c").write_text(
        r"""
#include "milk_calc_ext.h"
#define CR_PARAMS(X) \
    X(".k", k, FPTYPE_FLOAT64, 1.0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, "unused")
MILK_CALC_PARAMS(crash_now, CR_PARAMS);
static int crash_img(IMAGE *out, const IMAGE *in, const struct crash_now_par *P)
{
    (void) out;
    (void) in;
    (void) P;
    volatile int *p = 0;
    *p = 1;
    return 0;
}
MILK_CALC_EXT_EXPORT(MILK_CALC_REG_IMG("crash_now", crash_img, crash_now));
"""
    )
    res = milk_bash(
        'mem.mk2Dim "@L:loc" 4 4 >/dev/null; @calc "loc = 5"\n'
        'mem.mk2Dim im 8 8 >/dev/null; @calc "im = 3"\n'
        '@calc -l crash.c "o = crash_now(im)" 2>crash.err; echo "crash_rc=$?"\n'
        '@calc -v m "imean(loc)"; echo "loc=$m"',  # a local image: this session's server lives
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "crash_rc") == "1", out
    assert "CRASH INTERCEPTED" in (tmp_path / "crash.err").read_text(), out
    assert value(res.stdout, "loc") == "5", out


def test_unit_keeps_running_when_an_edit_does_not_compile(tmp_path):
    res = milk_bash(
        CAM
        + 'milk-fpsexec-milk-calc -n le1 -t cam -e "out_le = cam * 2" fpsinit >/dev/null\n'
        "milk-fpsexec-milk-calc -n le1 run >run.log 2>&1 & pid=$!\n"
        "sleep 0.6\n"
        '@set f.le1.unit.expr="out_le = cam *"; sleep 0.2\n'
        '@calc "cam = 4"; sleep 0.3; @calc -v b "imean(out_le)"; echo "bad_edit=$b"\n'
        'echo "error=[$(@get f.le1.unit.error)]"\n'
        '@set f.le1.unit.expr="out_le = cam * 3"; sleep 0.2\n'
        '@calc "cam = 5"; sleep 0.3; @calc -v c "imean(out_le)"; echo "good_edit=$c"\n'
        "kill -INT $pid; wait $pid",
        tmp_path,
    )
    log = (tmp_path / "run.log").read_text()
    out = res.stdout + res.stderr + log
    assert value(res.stdout, "bad_edit") == "8", out
    assert value(res.stdout, "error") == "[]", out
    assert value(res.stdout, "good_edit") == "15", out
    assert "not applied, still running the previous statements" in log, out


def test_calc_text_is_checked_then_run_in_order(tmp_path):
    res = milk_bash(
        CAM + '@calc "a1 = 1; a2 = = 2" 2>/dev/null; echo "syn=$?"\n'
        'echo "a1=[$(@calc a1 2>/dev/null)]"\n'
        '@calc "s.cam.kw.K = 0; for k = 1:3 { s.cam.kw.K += 1 }"; echo "K=$(@get s.cam.kw.K)"\n'
        "@calc 'k = 0 # first\nif imin(cam) > 5 {\n  k = 1\n}\nelif imin(cam) > 1 { k = 2 }\n"
        'else { k = 3 }\'; echo "k=$(@calc k)"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "syn") == "2", out
    assert value(res.stdout, "a1") == "[]", out
    assert value(res.stdout, "K") == "3", out  # read live at each iteration
    assert value(res.stdout, "k") == "2", out


def test_dashed_stream_names(tmp_path):
    res = milk_bash(
        CAM + 'mem.mk2Dim "@S:my-cam" 6 4 >/dev/null\n'
        "echo \"local=$(@calc 's.my-cam.xsize - 1') sub=$(@calc 's.cam.xsize-1')\"\n"
        '@calc "o = cam * 0 + s.my-cam.xsize"; echo "srv=$(@calc \'imean(o)\')"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "local") == "5 sub=7", out
    assert value(res.stdout, "srv") == "6", out


def test_def_calls_are_inlined(tmp_path):
    res = milk_bash(
        CAM + '@calc "cam = ix(cam)"\n'
        '@calc "def avg(x) = imean(x); def nrm(x) = x / imax(x); def two(x) = x + x"\n'
        "echo \"avg=$(@calc 'avg(cam + 1)')\"\n"
        '@calc "o = nrm(cam)"; echo "nrm=$(@calc \'imax(o)\')"\n'
        '@calc "r = two(rand(cam))" 2>e1.txt; echo "rand=$?"\n'
        '@calc "def sqrt(x) = x" 2>e2.txt; echo "builtin=$?"\n'
        "@calc 'if 1 { def zz(x) = x }' 2>e3.txt; echo \"nested=$?\"",
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "avg") == "0.5", out  # ix(cam) + 1 on 8 columns: -3 .. 4
    assert value(res.stdout, "nrm") == "1", out
    assert value(res.stdout, "rand") == "2", out
    assert "draws random numbers" in (tmp_path / "e1.txt").read_text(), out
    assert value(res.stdout, "builtin") == "2", out
    assert "built-in function" in (tmp_path / "e2.txt").read_text(), out
    assert value(res.stdout, "nested") == "2", out


def test_explain_leaves_defs_and_rings_alone(tmp_path):
    res = milk_bash(
        CAM
        + '@calc --explain "def hid(x) = x * 2; y = hid(cam)" >/dev/null; echo "x1=$?"\n'
        "@calc 'hid(2)' 2>/dev/null; echo \"hid=$?\"\n"
        '@calc --explain "hist(cam, 4); y = cam - prev(cam)" 2>&1 |'
        ' grep -c "only in a milk.calc unit" | sed "s/^/unit=/"\n'
        '@exists s.cam_hist && echo "ring=yes" || echo "ring=no"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "x1") == "0", out
    assert value(res.stdout, "hid") == "2", out  # the def stayed out of the session
    assert value(res.stdout, "unit") == "1", out
    assert value(res.stdout, "ring") == "no", out


def test_reserved_constants_and_per_shell_digits(tmp_path):
    res = milk_bash(
        CAM + '@calc "pi = 3" 2>e.txt; echo "assign=$?"\n'
        "pi=3; echo \"bash=$(@calc 'pi * 2; y = 1')\"\n"
        "mem.mk2Dim pi 4 4 >/dev/null; echo \"image=$(@calc 'pi * 2 + imean(cam) * 0')\"\n"
        "@config digits 6 >/dev/null; echo \"d6=$(@calc 'imean(cam) / 3')\"\n"
        "( unset MILK_DPDIGITS; echo \"d15=$(@calc 'imean(cam) / 3')\" )",
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "assign") == "2", out
    assert "is a constant" in (tmp_path / "e.txt").read_text(), out
    assert value(res.stdout, "bash") == "6.28318530717959", out
    assert value(res.stdout, "image") == "6.28318530717959", out
    assert value(res.stdout, "d6") == "0.666667", out
    assert value(res.stdout, "d15") == "0.666666666666667", out


def test_stored_variables_keep_every_digit(tmp_path):
    """A variable keeps the exact value assigned, whatever the printed digits."""
    res = milk_bash(
        '@calc "x = 1/3"; @test "x == 1/3"; echo "exact=$?"\n'
        '@config digits 4 >/dev/null; @calc "y = 1/3"; @calc -v y3 "y * 3"; echo "y3=$y3"\n'
        '@test "y == 1/3"; echo "exact4=$?"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "exact") == "0", out
    assert value(res.stdout, "y3") == "1", out
    assert value(res.stdout, "exact4") == "0", out


NANFIX_C = r"""
#include <math.h>
#include "milk_calc_ext.h"
#define NF_PARAMS(X) \
    X(".v", v, FPTYPE_FLOAT64, 0.0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, "Value for NaN")
MILK_CALC_PARAMS(nanfix, NF_PARAMS);
static void nanfix_f(float *restrict out, const float *restrict in, const struct nanfix_par *P,
                     int len)
{
    for (int i = 0; i < len; i++)
    {
        out[i] = isnan(in[i]) ? (float) P->v : in[i];
    }
}
MILK_CALC_EXT_EXPORT(MILK_CALC_REG_BLK("nanfix", nanfix_f, nanfix, 1));
"""


def test_extensions_compile_like_jit_kernels(tmp_path):
    (tmp_path / "nanfix.c").write_text(NANFIX_C)
    (tmp_path / "bad.c").write_text('#include "milk_calc_ext.h"\nflot x;\n')
    cache = tmp_path / "cache"
    res = milk_bash(
        'mem.mk2Dim im 4 4 >/dev/null; @calc "im = 2; im[1, 1] = nan"\n'
        '@calc -l nanfix.c "o = nanfix(im)"; echo "rc=$?"\n'
        '@calc -v t "itot(o)"; echo "tot=$t"\n'
        '@calc -l bad.c "o2 = 1" 2>bad.err; echo "bad_rc=$?"\n'
        'milk-cc -v --shared nanfix.c -o nf.so >cc.out 2>&1; echo "cc_rc=$?"',
        tmp_path,
        extra_env={"MILK_CALC_CACHE": str(cache)},
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "rc") == "0", out
    assert value(res.stdout, "tot") == "30", out  # isnan() was not folded away
    assert any(p.name.startswith("ext_") for p in cache.iterdir()), out
    assert value(res.stdout, "bad_rc") != "0", out
    assert "unknown type name" in (tmp_path / "bad.err").read_text(), out
    assert value(res.stdout, "cc_rc") == "0", out
    assert "-fno-finite-math-only" in (tmp_path / "cc.out").read_text(), out

    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real")
    for env, msg in (
        ({"MILK_CALC_CC": "no-such-cc", "MILK_CALC_CACHE": str(cache)}, "no-such-cc"),
        ({"MILK_CALC_CACHE": str(tmp_path / "link")}, "unsafe cache directory"),
    ):
        res = milk_bash(
            '@calc -l nanfix.c "1" 2>e.txt; echo "rc=$?"', tmp_path, extra_env=env
        )
        assert value(res.stdout, "rc") != "0", res.stdout + res.stderr
        assert msg in (tmp_path / "e.txt").read_text(), res.stdout + res.stderr


def test_order_statistics_and_argmax_rules(tmp_path):
    res = milk_bash(
        'mem.mk2Dim v 8 1 >/dev/null; @calc "v = [5, 1, 8, 3, 7, 2, 6, 4]"\n'
        'mem.mk2Dim m 8 1 >/dev/null; @calc "m = [1, 0, 1, 0, 1, 0, 0, 0]"\n'
        '@calc -v a,b,c,d "imedian(v); perc(v, 0.25); imedian(v, m); perc(i16(v), 1)"\n'
        'printf "med=%s\\np25=%s\\nmmed=%s\\npmax=%s\\n" "$a" "$b" "$c" "$d"\n'
        'mem.mk2Dim w 6 4 >/dev/null; @calc "w = -inf"; @calc "x0, y0 = iargmax(w)"\n'
        '@calc "w = 3; w[4, 2] = 9; w[1, 3] = 9; w[0, 0] = nan; x1, y1 = iargmax(w)"\n'
        '@calc "w = -w; x2, y2 = iargmin(w)"\n'
        '@calc -v a,b,c,d,e,f "x0; y0; x1; y1; x2; y2"; echo "arg=$a,$b,$c,$d,$e,$f"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "med") == "5", out  # upper middle of 8 values
    assert value(res.stdout, "p25") == "3", out
    assert value(res.stdout, "mmed") == "7", out  # 5, 8, 7 under the mask
    assert value(res.stdout, "pmax") == "8", out
    assert (
        value(res.stdout, "arg") == "0,0,4,2,4,2"
    ), out  # all -inf; first of two extremes; NaN skipped


def test_argument_values_and_words_read_alike(tmp_path):
    res = milk_bash(
        CAM
        + 'mem.mk2Dim "@S:a1" 010 0x8 >/dev/null; echo "m1=$(@get s.a1.xsize),$(@get s.a1.ysize)"\n'
        'mem.mk2Dim "@S:a2" 1e1 0b11 >/dev/null; echo "m2=$(@get s.a2.xsize),$(@get s.a2.ysize)"\n'
        'mem.mk2Dim "@S:a3" 1.5 4 2>/dev/null; echo "m3=$?"\n'
        'C="milk-fpsexec-milk-calc -n av1"\n'
        '$C -t cam -p "n:int=0 b:bool=off" -e "out_av = cam * 0 + n + b" fpsinit .n=010 .b=true '
        '>/dev/null; echo "init=$?"\n'
        "$C run >run.log 2>&1 & pid=$!\n"
        'sleep 0.6; @calc "cam = 3"; sleep 0.3; @calc -v o "imean(out_av)"; echo "unit=$o"\n'
        "kill -INT $pid; wait $pid\n"
        '$C set .n=1 .n=2 >/dev/null 2>e1; echo "twice=$?"\n'
        '$C set -- -5 >/dev/null; echo "dashes=$?,$(@get f.av1.n)"\n'
        '@set f.av1.n=0b101; @set f.av1.b=false; echo "set=$(@get f.av1.n),$(@get f.av1.b)"\n'
        'h() { @args "a:int" "b:int" -- "$@"; echo "h=$a,$b"; }; h 1 .a=2\n'
        'g() { @args "n:int=5" -- "$@"; echo "g=$n"; }; g .n=.\n'
        'k() { @args "n:int" -- "$@"; echo "k=$n"; }; k 0o17',
        tmp_path,
    )
    out = res.stdout + res.stderr + (tmp_path / "run.log").read_text()
    assert value(res.stdout, "m1") == "10,8" and value(res.stdout, "m2") == "10,3", out
    assert value(res.stdout, "m3") == "2", out
    assert value(res.stdout, "init") == "0", out
    assert (
        value(res.stdout, "unit") == "11"
    ), out  # 010 is 10, and the bool reached the unit
    assert value(res.stdout, "twice") == "2", out
    assert "given twice" in (tmp_path / "e1").read_text(), out
    assert value(res.stdout, "dashes") == "0,-5", out
    assert value(res.stdout, "set") == "5,OFF", out
    assert value(res.stdout, "h") == "2,1", out  # tags first, then the positional words
    assert value(res.stdout, "g") == "5" and value(res.stdout, "k") == "15", out


def test_property_modifier_and_keyword_tables_agree(tmp_path):
    from pathlib import Path

    res = milk_bash(
        'mem.mk2Dim "@S:pl" 4 4 >/dev/null; names=$(@get s.pl.list); echo "list=$names"\n'
        'for p in $names; do @get "s.pl.$p" >/dev/null 2>&1 || echo "unreadable=$p"; done\n'
        'echo "mods=$(milkcomplete @TC arg:mem.mk2Dim:1 | tr "\\n" " ")"\n'
        'echo "kw=$(milkls --keywords | cut -d" " -f1 | tr "\\n" " ")"\n'
        "@help > help.txt; milk-call > usage.txt 2>&1",
        tmp_path,
    )
    out = res.stdout + res.stderr
    names = value(res.stdout, "list").split()
    assert names[:5] == ["xsize", "ysize", "zsize", "naxis", "nelement"], out
    assert "owner_pid" in names and "ownerPID" not in names, out
    assert "unreadable=" not in res.stdout, out
    docs = Path(__file__).resolve().parents[3] / "docs" / "guide" / "scripting.md"
    text = docs.read_text()
    bullet = text[text.index("- streams (`s.`, `l.`") : text.index("- FPS (`f.`)")]
    assert [n for n in names if f"`{n}`" not in bullet] == [], bullet
    assert value(res.stdout, "mods").split() == ["@TC64:", "@TC128:"], out
    kws = value(res.stdout, "kw").split()
    assert len(kws) == 28, out
    section = text[text.index("## Keywords") : text.index("## Module Commands")]
    table = "".join(ln for ln in section.splitlines(True) if ln.startswith("|"))
    assert sorted(kws) == sorted(re.findall(r"`@(\w+)`", table)), table
    helptext = (tmp_path / "help.txt").read_text()
    assert [k for k in kws if f"@{k}" not in helptext] == [], helptext
    assert "batch" not in (tmp_path / "usage.txt").read_text(), out


def test_conditions_are_numbers(tmp_path):
    res = milk_bash(
        CAM + '@test \'"abc"\' 2>e1.txt; echo "str=$?"; @test nan; echo "nan=$?"\n'
        '@calc "t = 0"; @calc "if nan: t = 1"; @calc \'if "abc": t = 2\' 2>/dev/null; echo "ifrc=$?"\n'
        '@calc -v t "t"; echo "t=$t"; @assert \'"abc"\' 2>/dev/null; echo "assert=$?"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "str") == "2", out
    assert (
        "a condition is a number, not a string" in (tmp_path / "e1.txt").read_text()
    ), out
    assert value(res.stdout, "nan") == "1", out
    assert value(res.stdout, "ifrc") != "0", out
    assert value(res.stdout, "t") == "0", out
    assert value(res.stdout, "assert") == "2", out


def test_name_lookup_order_and_slice_end(tmp_path):
    res = milk_bash(
        'mem.mk2Dim qi 4 4 >/dev/null; @calc "qi = 1"; qi=5; @calc -v a "imean(qi)"; echo "img_bash=$a"\n'
        'unset qi; @calc "cv = 3"; mem.mk2Dim cv 4 4 >/dev/null; @calc -v b "imean(cv)"; echo "img_calc=$b"\n'
        '@calc "bv = 2"; bv=9; @calc -v c "bv"; echo "bash_calc=$c"; unset bv\n'
        'mem.mk2Dim sl 8 8 >/dev/null; @calc "end = 3; sd = sl[0:end, :]"; @get -v n l.sd.xsize\n'
        '@calc -v e "end * 2"; echo "end=$n,$e"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "img_bash") == "1", out
    assert value(res.stdout, "img_calc") == "0", out
    assert value(res.stdout, "bash_calc") == "9", out
    assert value(res.stdout, "end") == "8,6", out


def test_numbers_and_names(tmp_path):
    long80 = "a" * 80
    res = milk_bash(
        CAM + '@calc "w = f64(cam * 0) + pi; wn = wrap_pi(-w)"\n'
        '@calc -v w1,w2,w3 "wrap_pi(pi); wrap_pi(-pi); wn[0,0]"; echo "wrap=$w1,$w2,$w3"\n'
        '@calc -v big,hex "99999999999999999999; 0x10000000000000000"; echo "big=$big,$hex"\n'
        f'@calc "{long80} = 5" 2>e1.txt; echo "long=$?"; @calc "{long80[:79]} = 5"; echo "ok79=$?"\n'
        '@calc -v tr,rms,sd "trunc(-2.7); irms(cam); istd(cam)"; echo "fn=$tr,$rms,$sd"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "wrap") == ",".join(["3.14159265358979"] * 3), out
    assert value(res.stdout, "big") == "1e+20,1.84467440737096e+19", out
    assert value(res.stdout, "long") == "2", out
    assert (
        "identifier too long (max 79 characters)" in (tmp_path / "e1.txt").read_text()
    ), out
    assert value(res.stdout, "ok79") == "0", out
    assert value(res.stdout, "fn") == "-2,2,0", out


def test_error_status_position_and_def_checks(tmp_path):
    res = milk_bash(
        "mem.mk2Dim ea 4 4 >/dev/null; mem.mk2Dim eb 8 8 >/dev/null\n"
        '@calc "c = ea + eb" 2>/dev/null; s1=$?; @calc "c = nosuch + 1" 2>/dev/null; s2=$?\n'
        '@calc "c = sqrt(1, 2)" 2>/dev/null; s3=$?; @test "imean(ea + eb) > 0" 2>/dev/null; s4=$?\n'
        'echo "status=$s1,$s2,$s3,$s4"\n'
        "@calc $'q1 = 1\\nq2 = (2 +' 2>e1.txt; @calc $'q1 = 1\\nq2 = ea + eb' 2>e2.txt\n"
        '@calc "def bad(x) = x + nosuch" 2>e3.txt; echo "def=$?"\n'
        '@calc "def good(x) = x * 2 + imean(ea)"; @calc -v g "good(1)"; echo "good=$g"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "status") == "1,2,2,2", out
    assert "(line 2, column" in (tmp_path / "e1.txt").read_text(), out
    assert "(line 2) in 'q2 = ea + eb'" in (tmp_path / "e2.txt").read_text(), out
    assert value(res.stdout, "def") == "2", out
    assert (
        "def 'bad': undefined identifier 'nosuch'" in (tmp_path / "e3.txt").read_text()
    ), out
    assert value(res.stdout, "good") == "2", out


def test_removed_spellings_name_their_replacement(tmp_path):
    res = milk_bash(
        CAM + 'for t in "ln(1)" "fabs(-1)" "rand() + urand()" "grand()" "nrm2(cam)" '
        '"trunc(5, 0, 3)" "_PI"; do @calc -v z "$t" 2>>e.txt; echo "rc=$?"; done\n'
        "@calc $'if 1\\nz = 2\\nend' 2>>e.txt; @calc 'for k = 1:2; z = k; end' 2>>e.txt\n"
        "for o in --tokens; do @calc $o 2>>e.txt; milk-fpsexec-milk-calc $o 2>>e.txt; done; true",
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert set(re.findall(r"^rc=(\d+)$", res.stdout, re.M)) == {"2"}, out
    err = (tmp_path / "e.txt").read_text()
    for hint in (
        "use log",
        "use abs",
        "use rand",
        "use randn",
        "use norm",
        "use clamp(x, lo, hi)",
        "use pi, exp(1)",
        "'if ... end' was removed",
        "'for ... end' was removed",
        "'--tokens' was removed: use --list",
    ):
        assert hint in err, err
    assert err.count("'--tokens' was removed") == 2, err


def test_mod_and_wrap_pi_stay_exact(tmp_path):
    res = milk_bash(
        "mem.mk2Dim a 16 16 >/dev/null; mem.mk2Dim y 16 16 >/dev/null; @config digits 17 >/dev/null\n"
        '@calc "a = (rand(a, 1) - 0.5) * 2000; y = (rand(y, 2) + 0.01) * 3"\n'
        '@calc "m = a % y; f = fmod(a, y); w = wrap_pi(a)"\n'
        "for k in 0 17 101 202 255; do x=$((k % 16)); j=$((k / 16)); "
        'echo "px=$(@calc -v r "a[$x,$j]"; echo $r) $(@calc -v r "y[$x,$j]"; echo $r) '
        '$(@calc -v r "m[$x,$j]"; echo $r) $(@calc -v r "f[$x,$j]"; echo $r) $(@calc -v r "w[$x,$j]"; echo $r)"; done\n'
        '@calc -v big "1e20 % 3"; echo "big=$big"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    f32 = lambda v: struct.unpack("f", struct.pack("f", v))[0]
    rows = re.findall(r"^px=(.*)$", res.stdout, re.M)
    assert len(rows) == 5, out
    for row in rows:
        a, y, m, f, w = map(float, row.split())
        ref = f32(math.fmod(a, y))
        assert m == ref and f == ref, (row, ref, out)
        assert w == f32(math.remainder(a, f32(2 * math.pi))), (row, out)
    assert value(res.stdout, "big") == "nan", out


def test_casts_masks_and_broadcasts_keep_values(tmp_path):
    vals = "300 -2 2.7 -2.7 70000.5 -70000.5 2147483904 nan inf"
    res = milk_bash(
        'mem.mk2Dim a 8 8 >/dev/null; mem.mk2Dim "@TF64:d" 8 8 >/dev/null\n'
        f'for v in {vals}; do @calc "a = $v; d = $v"\n'
        '  @calc "b = u8(a); c = i16(a); m = a & 0xF0; e = u16(d); g = d & 0xFFF"\n'
        '  echo "img=$v $(@calc "b[0,0]") $(@calc "c[0,0]") $(@calc "m[0,0]") $(@calc "e[0,0]") $(@calc "g[0,0]")"\n'
        '  echo "num=$v $(@calc "u8($v)") $(@calc "i16($v)") $(@calc "$v & 0xF0" 2>/dev/null || echo 0) '
        '$(@calc "u16($v)") $(@calc "$v & 0xFFF" 2>/dev/null || echo 0)"; done\n'
        "mem.mk2Dim A 64 8 >/dev/null; mem.mk2Dim col 1 8 >/dev/null; mem.mk2Dim U 64 8 >/dev/null; mem.mk2Dim V 64 8 >/dev/null\n"
        '@calc "A = ix(A, 0, 0); col = iy(col, 0, 0) * 100; C = A + col; U = 1; V = 2"; echo "bc=$(@calc "C[5, 3]")"\n'
        '@calc "U, V = V, U"; @calc "U = U + A; V = V - A"; echo "uv=$(@calc "U[5, 0]") $(@calc "V[5, 0]")"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    img = {
        r.split()[0]: r.split()[1:] for r in re.findall(r"^img=(.*)$", res.stdout, re.M)
    }
    num = {
        r.split()[0]: r.split()[1:] for r in re.findall(r"^num=(.*)$", res.stdout, re.M)
    }
    assert len(img) == 9 and img == num, out
    assert value(res.stdout, "bc") == "305", out
    assert value(res.stdout, "uv") == "7 -4", out


def test_float_argext_matches_double(tmp_path):
    res = milk_bash(
        'mem.mk2Dim w 100 50 >/dev/null; mem.mk2Dim "@TF64:wd" 100 50 >/dev/null\n'
        'for s in 1 2 3; do @calc "w = floor(rand(w, $s) * 20) - 10"\n'
        '  [ $s = 2 ] && @calc "w = -inf; w[1, 0] = nan; w[60, 40] = -3.4028234663852886e38"\n'
        '  [ $s = 3 ] && @calc "w = nan"\n'
        '  @calc "wd = f64(w)"; echo "f$s=$(@calc "iargmax(w); iargmin(w); iamax(w)" | tr "\n" " ")"\n'
        '  echo "d$s=$(@calc "iargmax(wd); iargmin(wd); iamax(wd)" | tr "\n" " ")"; done',
        tmp_path,
    )
    out = res.stdout + res.stderr
    for s in "123":
        assert value(res.stdout, f"f{s}") == value(res.stdout, f"d{s}"), out
    assert value(res.stdout, "f2").split()[0] == "4060", out
    assert value(res.stdout, "f3").split() == ["-1", "-1", "-1"], out


EXT_BIND_C = r"""
#include "milk_calc_ext.h"

#define G2_PARAMS(X) \
    X(".a", a, FPTYPE_FLOAT64, 1.0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, "Gain") \
    X(".b", b, FPTYPE_FLOAT64, 0.0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, "Offset")
MILK_CALC_PARAMS(gain2, G2_PARAMS);

static void gain2_f(float *restrict out, const float *restrict in, const struct gain2_par *P, int len)
{
    for (int i = 0; i < len; i++)
    {
        out[i] = in[i] * (float) P->a + (float) P->b;
    }
}

#define MOP_PARAMS(X) X(".g", g, FPTYPE_FLOAT64, 1.0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, "Gain")
#define MOP_PORTS(X) X(IN, frame, MILK_PORT_IMAGE, 0, "Input") X(OUT, o, MILK_PORT_IMAGE, 0, "Output")
MILK_CALC_OP_DECLARE(mop, MOP_PARAMS, MOP_PORTS);

static int mop_init(milk_op_ctx *ctx, const struct mop_par *P, void **state)
{
    (void) ctx;
    (void) P;
    *state = NULL;
    return 0;
}

static int mop_exec(milk_op_ctx *ctx, const struct mop_par *P, void *state)
{
    const IMAGE *in = MILK_IN(ctx, mop_frame);
    IMAGE       *o  = MILK_OUT(ctx, mop_o);
    (void) state;
    for (uint64_t i = 0; i < in->md[0].nelement; i++)
    {
        o->array.F[i] = in->array.F[i] * (float) P->g;
    }
    return 0;
}

static void mop_free(void *state)
{
    (void) state;
}

MILK_CALC_EXT_EXPORT(MILK_CALC_REG_BLK("gain2", gain2_f, gain2, 2),
                     MILK_CALC_REG_OP("mop", mop, mop_init, mop_exec, mop_free));
"""


def test_extension_arguments_bind_like_module_commands(tmp_path):
    (tmp_path / "ext.c").write_text(EXT_BIND_C)
    res = milk_bash(
        'mem.mk2Dim "@S:raw" 4 4 >/dev/null; @calc "raw = 10"\n'
        'milk-fpsexec-milk-calc -n ua -l ext.c -e "o1 = gain2(raw, a = 2, 3); o2 = mop(raw, 4)" exec '
        '>/dev/null 2>&1; echo "rc=$?"\n'
        'echo "vals=$(@get s.o1.mean) $(@get s.o2.mean)"\n'
        'echo "fps=$(for p in gain2.a gain2.b mop.g; do @get f.ua.$p >/dev/null 2>&1 && echo -n 1 || echo -n 0; done)"\n'
        'milk-fpsexec-milk-calc -n ub -l ext.c -e "o3 = gain2(raw, 1, 2, 3)" exec 2>&1 >/dev/null | '
        'grep -c "gain2: too many arguments" | sed "s/^/many=/"\n'
        'milk-fpsexec-milk-calc -n uc -l ext.c fpsinit >/dev/null 2>&1; echo "lib=$(@get f.uc.gain2.b) $(@get f.uc.mop.g)"\n'
        '@get f.uc.bblur1pix.amp >/dev/null 2>&1 && echo "builtin=1" || echo "builtin=0"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "rc") == "0", out
    assert value(res.stdout, "vals") == "23 40", out
    assert value(res.stdout, "fps") == "000", out
    assert value(res.stdout, "many") == "1", out
    assert value(res.stdout, "lib") == "0 1", out
    assert value(res.stdout, "builtin") == "0", out


def test_milk_bash_stops_without_its_builtins(tmp_path):
    (tmp_path / "bin").mkdir()
    shutil.copy(_install_dir() / "bin" / "milk.bash", tmp_path / "bin")
    (tmp_path / "bin" / "milk-cli").write_text("#!/bin/sh\n")
    (tmp_path / "bin" / "milk-cli").chmod(0o755)
    res = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{tmp_path}/bin/milk.bash"; echo "rc=$? kw=$(type -t @get)"',
        ],
        capture_output=True,
        text=True,
        timeout=20,
        env={**os.environ, "MILK_INSTALLDIR": str(tmp_path)},
    )
    assert value(res.stdout, "rc") == "1 kw=", res.stdout + res.stderr
    assert (
        "libmilkbash.so is missing" in res.stderr and "milk is not loaded" in res.stderr
    ), res.stderr


def test_calc_mode_quoting_follows_the_shell_and_calculator_lexers(tmp_path):
    res = milk_cli_stdin(
        '@calc\ns = "a # b"   # a comment\n"1 + 2"\n\'2 * 3\' # note\n"4" + "5"\n@calc\n'
        'echo "s=$(@calc s)"',
        tmp_path,
    )
    assert [ln for ln in res.stdout.splitlines() if ln] == [
        "s a # b",
        "3",
        "6",
        "45",
        "s=a # b",
    ], (
        res.stdout + res.stderr
    )


def test_a_variable_holds_what_was_assigned_last(tmp_path):
    res = milk_bash(
        '@calc \'t = 5; t = "xyz"; s = "abc"; s = 5; q = "5"\'\n'
        "echo \"vals=$(@calc t) $(@calc s) $(@calc 'q + 1')\"\n"
        'milk-fpsexec-milk-calc -n us -p "nm:str=abc g:float=1" -e \'nm = "xyz"; g = 3\' exec >/dev/null 2>&1\n'
        'echo "unit=$(@get f.us.nm) $(@get f.us.g)"\n'
        'milk-fpsexec-milk-calc -n ut -p "g:float=1" -e \'g = "xyz"\' exec 2>&1 | '
        'grep -c "g is a number parameter: it cannot take a string" | sed "s/^/refused=/"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "vals") == "xyz 5 51", out
    assert value(res.stdout, "unit") == "xyz 3", out
    assert value(res.stdout, "refused") == "1", out


def test_one_completer_for_bash_and_the_prompt(tmp_path):
    res = milk_bash(
        'echo "opt=$(milkcomplete --li "arg:@calc:1:@calc" | tr "\\n" " ")"\n'
        'echo "set=$(milkcomplete sy "arg:@config:1:@config" | tr "\\n" " ")"\n'
        'echo "pre=$(milkcomplete "" "arg:@config:3:preset" | tr "\\n" " ")"\n'
        'echo "fn=$(milkcomplete \'"a = sqr\' "arg:@calc:1:@calc")"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "opt").split() == ["--lib", "--list"], out
    assert value(res.stdout, "set").split() == ["synhl", "syndiag"], out
    assert value(res.stdout, "pre").split() == ["minimal", "std", "ao"], out
    assert value(res.stdout, "fn") == '"a = sqrt(', out

    (tmp_path / "zzfile_abc.txt").write_text("hello-zz\n")
    steps = [
        ("raw", '@calc -v r "a1 = 2 + sqr', 0.3),
        ("raw", "\t", 0.5),
        ("line", '9)"; echo "r=$r"', 0.8),
        ("raw", 'cat "zzfile_a', 0.3),
        ("raw", "\t", 0.5),
        ("line", "", 0.5),
        ("line", 'mem.mk2Dim "@S:qcam" 4 4 >/dev/null', 0.6),
        ("raw", 'echo "xs=$(@get s.qcam.xsi', 0.3),
        ("raw", "\t", 0.5),
        ("line", ')"', 0.6),
        ("raw", 'echo "v=@{sqr', 0.3),
        ("raw", "\t", 0.5),
        ("line", '25)}"', 0.6),
    ]
    out = pty_session(steps, tmp_path)
    assert [value(out, k) for k in ("r", "xs", "v")] == ["5", "4", "5"], out[-2000:]
    assert "hello-zz" in out, out[-2000:]


def test_a_shell_keeps_one_connection_between_requests(tmp_path):
    """The shell's connection stays the same socket; a subshell has its own; a script's own fds,
    numbered or `{var}`, work beside it (bash protects a close-on-exec fd from `exec N>`, so the
    connection sits high); after the server ends, the next request starts it again."""
    res = milk_bash(
        'mem.mk2Dim "@S:k6" 4 4 >/dev/null; @calc "k6 = 1"\n'
        "sock() { for f in /proc/$1/fd/*; do [[ $(readlink $f) == socket:* ]] && ((${f##*/} >= 64)) && echo ${f##*/}:$(readlink $f); done | head -1; }\n"
        'a=$(sock $$); @calc "k6 = k6 + 1"; mem.listim >/dev/null; @test "imean(k6) > 0"; b=$(sock $$)\n'
        'echo "same=$([ -n "$a" ] && [ "$a" = "$b" ] && echo 1)"\n'
        '( @calc "k6 = k6 + 1"; c=$(sock $BASHPID); echo "sub=$([ -n "$c" ] && [ "$c" != "$a" ] && echo 1)" )\n'
        'exec 3>three.txt; @calc "k6 = k6 + 1"; echo x >&3; exec 3>&-\n'
        'exec {v}>var.txt; @calc "k6 = k6 + 1"; echo y >&$v; exec {v}>&-\n'
        "echo \"fds=$(cat three.txt var.txt | tr -d '\\n') val=$(@calc 'imean(k6)')\"\n"
        "milkshutdown >/dev/null 2>&1\n"
        '@calc "k6 = k6 + 1"; echo "restart=$(@calc \'imean(k6)\')"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "same") == "1", out
    assert value(res.stdout, "sub") == "1", out
    assert value(res.stdout, "fds") == "xy val=5", out
    assert value(res.stdout, "restart") == "6", out


def test_captured_command_output_is_complete(tmp_path):
    res = milk_bash(
        'for i in $(seq 1 300); do mem.mk2Dim "cap6_$i" 2 2 >/dev/null; done\n'
        'echo "pipe=$(mem.listim | grep -c cap6_)"\n'
        'mem.listim > out.txt; echo "file=$(grep -c cap6_ out.txt)"\n'
        'echo "err=$(mem.mk2Dim 2>&1 >/dev/null | grep -ci usage)"\n'
        '@log -n 3 | grep -q cap6_ && echo "logged=1"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert [value(res.stdout, k) for k in ("pipe", "file", "logged")] == [
        "300",
        "300",
        "1",
    ], out
    assert value(res.stdout, "err") not in (None, "0"), out
