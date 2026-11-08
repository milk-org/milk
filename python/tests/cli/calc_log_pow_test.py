"""
Tests for Phase 1 calculator language changes:
- `log(x)` is the natural logarithm (`ln` was removed: its error names `log`); `log10(x)` is base-10
  logarithm.
- `pow(a, b)` replaces the `^` operator, preserving exact `int64` for non-negative integer powers.
- A lone `^` fails with `'^' is not an operator: write pow(a, b) (^^ is xor)`; `^^` remains XOR.
"""

from __future__ import annotations

import pytest

from milk.cliwrap import HAVE_CLI
from tests.cli.conftest import milk_bash, value

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.timeout(60),
]


def test_log_ln_log10_and_pow_semantics(tmp_path):
    """Verify `log`, the removed `ln`, `log10`, `pow` (scalar exactness, negative exponent, images),
    lone `^` rejection message, and `^^` bitwise XOR.
    """
    res = milk_bash(
        "mem.mk2Dim x 8 8 >/dev/null\n"
        '@calc "x = (ix(x, 0, 0) + iy(x, 0, 0) + 1) * 0.25"\n'
        '@calc -v log_e,log10_100 "log(exp(1)); log10(100)"\n'
        'ln_err=$(@calc "ln(exp(1))" 2>&1); rc_ln=$?\n'
        '@calc -v p_exact,p_neg,bxor "pow(2, 62) + 1; pow(2, -1); 5 ^^ 3"\n'
        '@calc "b = 2; e = 62"; @calc -v p_srv "pow(b, e) + 1"\n'
        '@calc "y = pow(x, 2); yref = x * x; p2x = pow(2, x * 4); lim = log(exp(x)); l10 = log10(pow(10, x))"\n'
        '@calc -v im_err,im_l_err,im_l10_err "imax(abs(y - yref)); imax(abs(lim - x)); imax(abs(l10 - x))"\n'
        '@calc -v p2_0 "p2x[0,0]"\n'
        'err_caret=$(@calc "2 ^ 3" 2>&1); rc_caret=$?\n'
        'echo "logs=$log_e,$log10_100"\n'
        'echo "ln=$rc_ln $ln_err"\n'
        'echo "pows=$p_exact,$p_srv,$p_neg,$bxor"\n'
        'echo "im=$im_err,$im_l_err,$im_l10_err,$p2_0"\n'
        'echo "caret_rc=$rc_caret"\n'
        'echo "caret_err=$err_caret"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert value(res.stdout, "logs") == "1,2", out
    assert value(res.stdout, "ln").startswith("2 ") and "use log" in value(
        res.stdout, "ln"
    ), out
    assert (
        value(res.stdout, "pows") == "4611686018427387905,4611686018427387905,0.5,6"
    ), out
    im_err, im_l_err, im_l10_err, p2_0 = [
        float(v) for v in value(res.stdout, "im").split(",")
    ]
    assert im_err == 0.0, out
    assert im_l_err < 1e-5, out
    assert im_l10_err < 1e-4, out
    assert p2_0 == 2.0, out
    assert int(value(res.stdout, "caret_rc")) != 0, out
    assert "'^' is not an operator: write pow(a, b) (^^ is xor)" in value(
        res.stdout, "caret_err"
    ), out
