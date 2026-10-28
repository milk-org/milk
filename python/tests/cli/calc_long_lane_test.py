"""
Tests for the exact 64-bit scalar lane, the fused index reductions and `cumsum` (closing round of the
2026-10-27 engine review).

A kept scalar statement used to compute in double, so `k * 3 % 1000` with `k = 2^53 + 1` gave 976 in a
unit and 979 on the text path. The scalar tape now carries an `int64_t` lane next to the double one
and evaluates integer operands with the text engine's `cli_calc_long_binop`, including its overflow
fallback to double and its shift semantics. `iargmax`, `iargmin` and `iamax` are fused reductions
(the guide said so; they fell back to text), and `cumsum(expr[, axis])` is a pre-stage that the guide
listed but the engine did not have.
"""

from __future__ import annotations

import pytest

from milk.cliwrap import HAVE_CLI
from tests.cli.conftest import milk_bash, value

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.timeout(120),
]

HELPERS = r"""
stop() { kill -INT "$1" 2>/dev/null; for _ in $(seq 60); do kill -0 "$1" 2>/dev/null || break; sleep 0.05; done
         kill -0 "$1" 2>/dev/null && echo "still running"; wait "$1" 2>/dev/null; }
"""

# 2^53 + 1: (k * 3) is not representable in double, so a double lane gives 976, an exact one 979.
K = 9007199254740993


def test_kept_scalar_statements_are_exact_in_a_unit(tmp_path):
    """`y = k * 3 % 1000` is kept as a scalar tape; it and the stored `z = k * 3` must be exact,
    `>>` by 64 or more is 0 as on the text path, and an overflowing product becomes a double.
    """
    (tmp_path / "prog.calc").write_text(
        "y = k * 3 % 1000\nz = k * 3\nw = k << 1\nv = (-5) >> 64\nbig = k * k\n"
        "out[0,0] = y\nout[1,0] = z % 1000\nout[2,0] = v\nout[3,0] = w - 2 * k\nout[0,1] = big > 1e31\n"
    )
    res = milk_bash(
        HELPERS
        + 'mem.mk2Dim "@S:src" 4 4 >/dev/null; mem.mk2Dim "@S:out" 4 4 >/dev/null; @calc "src = 1; out = 0"\n'
        'milk-fpsexec-milk-calc -n lane -t src -p "k:int=0" -f prog.calc fpsinit >/dev/null\n'
        "milk-fpsexec-milk-calc -n lane run >run.log 2>&1 & pid=$!\n"
        f"sleep 0.6; @set f.lane.k={K}; sleep 0.1; "
        '@calc "src = 2"; sleep 0.3; @calc "src = 3"; sleep 0.3\n'
        '@calc -v a,b,c,d,e "out[0,0]; out[1,0]; out[2,0]; out[3,0]; out[0,1]"; echo "lane=$a,$b,$c,$d,$e"\n'
        'echo "error=[$(@get f.lane.unit.error)]"; stop $pid\n'
        f'@calc -v t "k = {K}; k * 3 % 1000"; echo "text=$t"\n'
        '@calc --explain "y = k * 3 % 1000; z = k * 3"',
        tmp_path,
    )
    out = res.stdout + res.stderr + (tmp_path / "run.log").read_text()
    assert value(res.stdout, "error") == "[]", out
    assert value(res.stdout, "text") == "979", out
    assert value(res.stdout, "lane") == "979,979,0,0,1", out
    assert res.stdout.count("kept: scalar tape") == 2, out


@pytest.mark.parametrize("nthreads", [1, 4])
def test_index_reductions_match_the_eager_path(tmp_path, nthreads: int):
    """NaNs skipped, ties to the first index, -1 when every element is NaN; fused equals eager."""
    res = milk_bash(
        "mem.mk2Dim a 2048 1024 >/dev/null; mem.mk2Dim b 64 64 >/dev/null\n"
        '@calc "a = urand(a, 3) - 0.5; a[3,2] = sqrt(-1); a[100,7] = sqrt(-1)"\n'
        '@calc "a[5,1] = imax(a); a[1500,600] = imax(a); a[7,900] = imin(a); a[2000,1000] = imin(a)"\n'
        '@calc "b = sqrt(-1) * urand(b, 4)"\n'
        '@calc -v f1,f2,f3 "iargmax(a); iargmin(a); iamax(a)"; echo "fused=$f1,$f2,$f3"\n'
        '@calc -v e1,e2,e3 "iargmax(a[::1b]); iargmin(a[::1b]); iamax(a[::1b])"; echo "eager=$e1,$e2,$e3"\n'
        '@calc -v x,y "2 * iargmin(a) - iamax(a) + 1; 2 * iargmin(a[::1b]) - iamax(a[::1b]) + 1"; echo "expr=$x,$y"\n'
        '@calc -v n1,n2 "iargmax(b); iargmax(b[::1b])"; echo "allnan=$n1,$n2"\n'
        '@calc --explain "m = iargmax(a)"',
        tmp_path,
        extra_env={"MILK_CALC_THREADS": str(nthreads)},
    )
    out = res.stdout + res.stderr
    fused = value(res.stdout, "fused")
    assert fused == value(res.stdout, "eager"), out
    assert (
        fused == "2053,1566754,1566754"
    ), out  # (5,1); (1500,600) is the later tie; |min| > max
    x, y = value(res.stdout, "expr").split(",")
    assert x == y, out
    assert value(res.stdout, "allnan") == "-1,-1", out
    assert "kept: scalar tape: 1 op, 1 reduction" in res.stdout, out


@pytest.mark.parametrize("nthreads", [1, 4])
def test_cumsum_along_each_axis(tmp_path, nthreads: int):
    """The cumulative sum equals the prefix totals along axes 0, 1 and 2, in a fused pass and on the
    text path, and a bad axis is an error."""
    res = milk_bash(
        "mem.mk2Dim a 600 7 >/dev/null; mem.mk3Dim c 300 5 4 >/dev/null\n"
        '@calc "a = urand(a, 3) - 0.5; c = urand(c, 5) - 0.3"\n'
        '@calc "a0 = cumsum(a); a1 = cumsum(a, 1); c2 = cumsum(c, 2); c0 = cumsum(c, 0) * 2 + 1"\n'
        '@calc "e0 = cumsum(a[::1b], 0); e1 = cumsum(a[::1b], 1); e2 = cumsum(c[::1b], 2)"\n'
        '@calc -v p0,p1,p2,p3 "a0[299,3]; itot(a[0:299,3]); a1[7,5]; itot(a[7,0:5])"; echo "p=$p0,$p1,$p2,$p3"\n'
        '@calc -v q0,q1,q2,q3 "c2[41,2,3]; itot(c[41,2,0:3]); c0[256,4,1]; 2 * itot(c[0:256,4,1]) + 1"; echo "q=$q0,$q1,$q2,$q3"\n'
        '@calc -v d0,d1,d2 "imax(abs(a0 - e0)); imax(abs(a1 - e1)); imax(abs(c2 - e2))"; echo "d=$d0,$d1,$d2"\n'
        '@calc -v s0,s1 "a0[599,6]; itot(a[:,6])"; echo "last=$s0,$s1"\n'
        'bad=none; @calc -v bad "cumsum(a, 2)" 2>bad.err; echo "bad=$bad"\n'
        '@calc --explain "o = cumsum(a, 1) * 2"',
        tmp_path,
        extra_env={"MILK_CALC_THREADS": str(nthreads)},
    )
    out = res.stdout + res.stderr + (tmp_path / "bad.err").read_text()
    for key in ("p", "q"):
        v = [float(x) for x in value(res.stdout, key).split(",")]
        assert abs(v[0] - v[1]) < 1e-4 and abs(v[2] - v[3]) < 1e-4, (key, v, out)
    assert all(float(x) == 0.0 for x in value(res.stdout, "d").split(",")), out
    s0, s1 = (float(x) for x in value(res.stdout, "last").split(","))
    assert abs(s0 - s1) < 1e-4, out
    assert (
        value(res.stdout, "bad") == "none"
        and "axis" in (tmp_path / "bad.err").read_text()
    ), out
    assert "1 pre-stage" in res.stdout, out
