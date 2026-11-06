"""
Fused views and axis reductions (plan 'the-new-milk-cli-calc' step 3): slices, strided and reversed
slices, broadcasting and axis reductions run in the one-pass fused engine, as operands and as sliced
destinations, in one-shot statements and in kept unit programs.

The per-operator path is forced with a bin-1 slice (`im[::1b]`): an exact copy that the fused engine does
not read as a view. Coordinate images are centred on n/2, so expected pixel values account for that.
"""

from __future__ import annotations

import pytest

from milk.cliwrap import HAVE_CLI
from tests.cli.calcdiff.datasets import DATASETS, all_dataset_ids, offset_dataset_ids
from tests.cli.calcdiff.gen_expr import generate_cases, load_functions
from tests.cli.conftest import milk_bash, value

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.timeout(120),
]

SETUP = (
    "mem.mk2Dim a 8 6 >/dev/null; mem.mk2Dim b 8 6 >/dev/null; mem.mk3Dim cube 8 6 4 >/dev/null\n"
    "mem.mk2Dim row 8 1 >/dev/null; mem.mk2Dim col 1 6 >/dev/null\n"
    '@calc "a = ix(a) + 10*iy(a) + 0.5*urand(a, 3); b = 100 + ix(b); row = ix(row) * 2; col = iy(col) * 1000"\n'
    '@calc "cube = ix(cube) + 10*iy(cube) + 0.25*urand(cube, 5)"\n'
)


def _errs(script: str, tmp_path, keys):
    res = milk_bash(SETUP + script, tmp_path)
    assert res.returncode == 0, res.stdout + res.stderr
    out = {k: value(res.stdout, k) for k in keys}
    for k in keys:
        assert out[k] is not None, (k, res.stdout, res.stderr)
    return out, res


def test_slice_operands_strides_and_reversal(tmp_path):
    """Slices read through views give the pixels a copy would hold, including steps and reversed axes."""
    out, _ = _errs(
        '@calc "f1 = a[0:3,:] * 2 + b[4:7,:]; s1 = a[0:3::1b,:] * 2 + b[4:7::1b,:]"\n'
        '@calc -v e1 "imax(abs(f1 - s1))"; echo "e1=$e1"; echo "sh1=$(@get l.f1.xsize)x$(@get l.f1.ysize)"\n'
        '@calc "f2 = a[::2,-*] + 1"\n'
        '@calc -v p0,p1,q0,q1 "f2[0,0]; a[0,5]; f2[1,0]; a[2,5]"; echo "rev=$p0,$p1,$q0,$q1"\n'
        '@calc "g = a; g[1:4,:] = a[1:4,-*] * 1"\n'
        '@calc -v g1,g2 "g[1,0]; a[1,5]"; echo "revdst=$g1,$g2"',
        tmp_path,
        ["e1", "sh1", "rev", "revdst"],
    )
    assert float(out["e1"]) == 0.0
    assert out["sh1"] == "4x6"
    p0, p1, q0, q1 = out["rev"].split(",")
    assert float(p0) == float(p1) + 1 and float(q0) == float(q1) + 1
    g1, g2 = out["revdst"].split(",")
    assert g1 == g2


def test_broadcast_matches_per_operator_path(tmp_path):
    """A row, a column and a frame broadcast over an image or a cube, bit for bit as the fallback."""
    out, _ = _errs(
        '@calc "f3 = a - row - col; s3 = a[::1b] - row - col"\n'
        '@calc -v e3 "imax(abs(f3 - s3))"; echo "e3=$e3"\n'
        '@calc "f4 = cube * 2 - a; s4 = cube[::1b] * 2 - a"\n'
        '@calc -v e4 "imax(abs(f4 - s4))"; echo "e4=$e4"; echo "sh4=$(@get l.f4.xsize)x$(@get l.f4.ysize)x$(@get l.f4.zsize)"\n'
        '@calc "bad = a[0:3,:] + a[0:4,:]" 2>err.log; echo "badrc=$?"',
        tmp_path,
        ["e3", "e4", "sh4", "badrc"],
    )
    assert float(out["e3"]) == 0.0 and float(out["e4"]) == 0.0
    assert out["sh4"] == "8x6x4"
    assert out["badrc"] != "0"
    assert "do not match" in (tmp_path / "err.log").read_text()


def test_sliced_destination_scalar_compound_and_shm_post(tmp_path):
    """Writes through a destination view: expression, constant and compound forms; one post per statement."""
    out, _ = _errs(
        "mem.mk2Dim o 8 6 >/dev/null\n"
        '@calc "o = 0; o[0:3,:] = a[0:3,:] * 2 + 1; o[4:7,:] = 5; o[0:3,0:2] += 100"\n'
        # the same picture in closed form (ix/iy are centred: x = ix + 4, y = iy + 3)
        '@calc "r = where(ix(a) + 4 <= 3, a * 2 + 1 + where(iy(a) + 3 <= 2, 100, 0), 5)"\n'
        '@calc -v eo "imax(abs(o - r))"; echo "eo=$eo"\n'
        'mem.mk2Dim "@S:sh" 8 6 >/dev/null; c0=$(@get s.sh.cnt0)\n'
        '@calc "sh[0:3,:] = a[0:3,:] + 1"; echo "dcnt=$(( $(@get s.sh.cnt0) - c0 ))"\n'
        '@calc -v v1,v2 "sh[2,3]; a[2,3] + 1"; echo "shv=$v1,$v2"',
        tmp_path,
        ["eo", "dcnt", "shv"],
    )
    assert float(out["eo"]) == 0.0
    assert out["dcnt"] == "1"
    v1, v2 = out["shv"].split(",")
    assert v1 == v2


def test_variable_bounds_resolved_each_statement(tmp_path):
    """Bounds with variables are resolved when the statement runs, also from the statement cache."""
    out, _ = _errs(
        'k=2; @calc "f5 = a[k:k+1,:] + 0"; k=4; @calc "f5 = a[k:k+1,:] + 0"\n'
        '@calc -v t5,t5s "itot(f5); itot(a[4:5::1b,:])"; echo "dyn=$t5,$t5s"',
        tmp_path,
        ["dyn"],
    )
    t5, t5s = out["dyn"].split(",")
    assert abs(float(t5) - float(t5s)) < 1e-3


def test_axis_reductions_fused_match_fallback(tmp_path):
    """Every axis form along every axis, inside expressions and as plain assignments."""
    script = (
        '@calc "f1 = a - imean(a, 1); s1 = a[::1b] - imean(a[::1b], 1)"\n'
        '@calc -v e1 "imax(abs(f1 - s1))"; echo "e1=$e1"\n'
        '@calc "f2 = (a - imean(a, 1)) / (istd(a, 1) + 1); s2 = (a[::1b] - imean(a[::1b], 1)) / (istd(a[::1b], 1) + 1)"\n'
        '@calc -v e2 "imax(abs(f2 - s2))"; echo "e2=$e2"\n'
        "mem.mk2Dim m 8 6 >/dev/null; worst=0\n"
        "for op in itot imean imin imax istd ivar norm asum; do\n"
        '  @calc "m = ${op}(cube, 2); sm = ${op}(cube[::1b], 2)"; @calc -v e "imax(abs(m - sm)) / (imax(abs(sm)) + 1)"\n'
        '  worst=$(@calc "max($worst, $e)")\n'
        'done; echo "worst2=$worst"\n'
        'mem.mk2Dim m1 8 4 >/dev/null; @calc "m1 = itot(cube, 1); sm1 = itot(cube[::1b], 1)"; @calc -v e3 "imax(abs(m1 - sm1))"; echo "e3=$e3"\n'
        'mem.mk2Dim m0 6 4 >/dev/null; @calc "m0 = istd(cube, 0); sm0 = istd(cube[::1b], 0)"; @calc -v e4 "imax(abs(m0 - sm0)) / (imax(abs(sm0)) + 1)"; echo "e4=$e4"\n'
        '@calc -v r5,r5s "irms(cube - imean(cube, 2)); irms(cube[::1b] - imean(cube[::1b], 2))"; echo "r5=$r5,$r5s"\n'
        '@calc "bad = a - imean(a, 2)" 2>err.log; echo "badrc=$?"'
    )
    out, _ = _errs(script, tmp_path, ["e1", "e2", "worst2", "e3", "e4", "r5", "badrc"])
    assert float(out["e1"]) < 1e-4 and float(out["e2"]) < 1e-4
    assert float(out["worst2"]) < 1e-5
    assert float(out["e3"]) < 1e-3 and float(out["e4"]) < 1e-5
    r5, r5s = out["r5"].split(",")
    assert abs(float(r5) - float(r5s)) < 1e-4 * (abs(float(r5s)) + 1)
    assert out["badrc"] != "0"


def test_unit_keeps_sliced_destination_ring_buffer_and_axis_reduction(tmp_path):
    """milk.calc keeps a sliced destination, a ring buffer indexed by _n, and an axis reduction."""
    (tmp_path / "v1.calc").write_text(
        "out[0:7,:] = src[0:7,:] * 2\n"
        "hist[:,:,_n % 4] = src + _n\n"
        "out[8:15,:] = src[8:15,:] - imean(src[8:15,:], 1)\n"
    )
    res = milk_bash(
        'mem.mk2Dim "@S:src" 16 16 >/dev/null; mem.mk2Dim "@S:out" 16 16 >/dev/null; mem.mk3Dim "@S:hist" 16 16 4 >/dev/null\n'
        '@calc "src = ix(src) + 0.1 * iy(src)"\n'
        "milk-fpsexec-milk-calc -n v1 -t src -f v1.calc fpsinit >/dev/null\n"
        "milk-fpsexec-milk-calc -n v1 run >run.log 2>&1 & pid=$!\n"
        "sleep 0.5\n"
        'for i in 1 2 3 4 5 6; do @calc "src = src + 1"; sleep 0.1; done; sleep 0.3\n'
        '@calc -v e1 "imax(abs(out[0:7,:] - src[0:7,:] * 2))"; echo "e1=$e1"\n'
        '@calc -v e2 "imax(abs(out[8:15,:] - (src[8:15::1b,:] - imean(src[8:15::1b,:], 1))))"; echo "e2=$e2"\n'
        '@calc -v h1,h2 "hist[3,3,1]; hist[3,3,2]"; echo "ring=$h1,$h2"\n'
        'echo "error=[$(@get f.v1.unit.error)]"; kill -INT $pid; wait $pid 2>/dev/null',
        tmp_path,
    )
    log = (tmp_path / "run.log").read_text()
    assert value(res.stdout, "error") == "[]", res.stdout + res.stderr + log
    assert float(value(res.stdout, "e1")) == 0.0, res.stdout + log
    assert float(value(res.stdout, "e2")) < 1e-4, res.stdout + log
    h1, h2 = value(res.stdout, "ring").split(",")
    # slot k holds src + n of the last frame n = k (mod 4); src itself grows by 1 per frame, so
    # consecutive slots differ by 2 (or by -6 across the wrap)
    assert float(h2) - float(h1) in (2.0, -6.0), (h1, h2)


def test_phase5_masked_lhs_and_end_slices(tmp_path):
    """Masked LHS assignments (=, +=, -=, *=, /=) and `end` in slice bounds (§3.2 rules 1, 7)."""
    res = milk_bash(
        SETUP + '@calc "m = a; m[a > 0] = -1; ref = where(a > 0, -1, a)"\n'
        '@calc -v em1 "imax(abs(m - ref))"; echo "em1=$em1"\n'
        '@calc "m = a; m[a > 0] += 10; m[a <= 0] -= 5; m[b > 100] *= 2; m[b <= 100] /= 2"\n'
        '@calc "ref2 = where(a > 0, a + 10, a - 5); ref2 = where(b > 100, ref2 * 2, ref2 / 2)"\n'
        '@calc -v em2 "imax(abs(m - ref2))"; echo "em2=$em2"\n'
        '@calc "s_end = a[1:end-1, 0:end-2]; s_num = a[1:6, 0:3]"\n'
        '@calc -v eend "imax(abs(s_end - s_num))"; echo "eend=$eend"\n'
        '@calc "d_end = a; d_end[end-1:end, 1:end-1] = 77; d_num = a; d_num[6:7, 1:4] = 77"\n'
        '@calc -v edend "imax(abs(d_end - d_num))"; echo "edend=$edend"\n',
        tmp_path,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert float(value(res.stdout, "em1")) == 0.0
    assert float(value(res.stdout, "em2")) < 1e-5
    assert float(value(res.stdout, "eend")) == 0.0
    assert float(value(res.stdout, "edend")) == 0.0


def test_phase5_comprehensions_literals_and_linspace(tmp_path):
    """Array comprehensions, vector literals, and linspace (§3.2 rules 4, 5)."""
    res = milk_bash(
        '@calc "c1 = [k * k for k = 0:4]; c1rev = [k for k = 6:0:-2]"\n'
        '@calc "c2 = [x + 10 * y for x = 0:3, y = 0:2]"\n'
        '@calc "c3 = [x + 10 * y + 100 * z for x = 0:1, y = 0:2, z = 0:1]"\n'
        '@calc "vlit = [1.5, -2.0, 3.25, 4.0]; ls = linspace(0.0, 1.0, 5)"\n'
        '@calc -v v1,v2,v3,v4,v5 "c1[3]; c1rev[2]; c2[2,1]; c3[1,2,1]; vlit[2] + ls[1]"\n'
        'echo "vals=$v1,$v2,$v3,$v4,$v5"\n'
        'echo "sh=$(@get l.c2.xsize)x$(@get l.c2.ysize)_$(@get l.ls.xsize)"\n',
        tmp_path,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    v1, v2, v3, v4, v5 = [float(x) for x in value(res.stdout, "vals").split(",")]
    assert (v1, v2, v3, v4, v5) == (9.0, 2.0, 12.0, 121.0, 3.5)
    assert value(res.stdout, "sh") == "4x3_5"


def test_phase5_roll_chained_cmp_constants_and_math(tmp_path):
    """roll, chained comparisons, pi/nan/inf, wrap_pi, and sinc (§3.2 rules 6, 8, 12, 13, 14)."""
    res = milk_bash(
        SETUP + '@calc "r1 = roll(a, 2, -1)"\n'
        '@calc -v rp0,ra0,rp1,ra1 "r1[0,0]; a[6,1]; r1[3,5]; a[1,0]"\n'
        'echo "roll=$rp0,$ra0,$rp1,$ra1"\n'
        '@calc "ch = (-1 <= ix(a) <= 1) && (0 <= iy(a) < 2); ch_ref = (ix(a) >= -1 && ix(a) <= 1) && (iy(a) >= 0 && iy(a) < 2)"\n'
        '@calc -v ech "imax(abs(ch - ch_ref))"; echo "ech=$ech"\n'
        '@calc -v sc_ch "(1 < 2 <= 2 < 5) * 10 + (1 < 3 < 2)"; echo "sc_ch=$sc_ch"\n'
        '@calc "im_c = [pi, nan, inf, -inf]"\n'
        '@calc -v c_pi,c_nan,c_inf "abs(im_c[0] - 3.14159265); isnan(im_c[1]) && isnan(nan); !isfinite(im_c[2]) && !isfinite(inf)"\n'
        'echo "consts=$c_pi,$c_nan,$c_inf"\n'
        '@calc "w = wrap_pi([0.0, 1.5 * pi, -1.5 * pi, 2.25 * pi]); s = sinc([0.0, 0.5, 1.0, -0.5])"\n'
        '@calc -v w1,w2,w3,s0,s1,s2 "w[1]; w[2]; w[3]; s[0]; s[1]; abs(s[2])"\n'
        'echo "ws=$w1,$w2,$w3,$s0,$s1,$s2"\n',
        tmp_path,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    rp0, ra0, rp1, ra1 = [float(x) for x in value(res.stdout, "roll").split(",")]
    assert rp0 == ra0 and rp1 == ra1
    assert float(value(res.stdout, "ech")) == 0.0
    assert float(value(res.stdout, "sc_ch")) == 10.0
    c_pi, c_nan, c_inf = [float(x) for x in value(res.stdout, "consts").split(",")]
    assert c_pi < 1e-6 and c_nan == 1.0 and c_inf == 1.0
    w1, w2, w3, s0, s1, s2 = [float(x) for x in value(res.stdout, "ws").split(",")]
    import math

    assert abs(w1 - (-0.5 * math.pi)) < 1e-5
    assert abs(w2 - (0.5 * math.pi)) < 1e-5
    assert abs(w3 - (0.25 * math.pi)) < 1e-5
    assert s0 == 1.0 and abs(s1 - 2.0 / math.pi) < 1e-6 and s2 < 1e-6


def test_offset_datasets_reduction_accuracy(tmp_path):
    """Verify `istd`, `istats` and z-score within 1e-6 of a double reference at offsets 0..3e4."""
    script = (
        "mem.mk2Dim A 256 256 >/dev/null\n"
        "for off in 0 1000 10000 30000; do\n"
        '  @calc "A = $off + sin(0.37 * ix(A)) + cos(0.21 * iy(A)); Ad = f64(A)"\n'
        '  @calc "s32 = istd(A); s64 = istd(Ad); m32, st32, lo32, hi32 = istats(A); '
        'm64, st64, lo64, hi64 = istats(Ad)"\n'
        '  @calc "Z32 = (A - imean(A)) / istd(A); Z64 = f32((Ad - imean(Ad)) / istd(Ad)); '
        'zerr = imax(abs(Z32 - Z64)) / imax(abs(Z64))"\n'
        '  @calc "serr = abs(s32 - s64) / s64; sterr = abs(st32 - st64) / st64; '
        'merr = abs(m32 - m64) / max(abs(m64), 1.0)"\n'
        '  @calc -v serr,sterr,merr,zerr "serr; sterr; merr; zerr"\n'
        '  echo "off_$off=$serr,$sterr,$merr,$zerr"\n'
        "done\n"
    )
    res = milk_bash(script, tmp_path)
    assert res.returncode == 0, res.stdout + res.stderr
    for off in (0, 1000, 10000, 30000):
        vals = [float(x) for x in value(res.stdout, f"off_{off}").split(",")]
        assert len(vals) == 4
        for err_val in vals:
            assert (
                err_val <= 1e-6
            ), f"Offset {off} exceeded 1e-6 relative error vs double ref: {vals}"


def test_iargmax_iargmin_coord_unpack(tmp_path):
    """Verify 2-D and 3-D coordinate unpacking on `iargmax` and `iargmin` (§3.2 rule 11)."""
    script = (
        "mem.mk2Dim im2 16 12 >/dev/null\n"
        "mem.mk3Dim im3 8 6 4 >/dev/null\n"
        '@calc "im2 = ix(im2) + 10 * iy(im2); im2[5, 9] = 999.0; im2[11, 3] = -500.0"\n'
        '@calc "mx2, my2 = iargmax(im2); nx2, ny2 = iargmin(im2); flat2 = iargmax(im2)"\n'
        '@calc "im3 = ix(im3) + 10 * iy(im3); '
        'im3[3, 4, 2] = 9999.0; im3[6, 1, 3] = -9999.0"\n'
        '@calc "mx3, my3, mz3 = iargmax(im3); nx3, ny3, nz3 = iargmin(im3)"\n'
        '@calc "nan2 = im2 * nan; nanx, nany = iargmax(nan2)"\n'
        '@calc -v mx2,my2,nx2,ny2,flat2 "mx2; my2; nx2; ny2; flat2"\n'
        '@calc -v mx3,my3,mz3,nx3,ny3,nz3,nanx,nany "mx3; my3; mz3; nx3; ny3; nz3; nanx; nany"\n'
        'echo "c2=$mx2,$my2,$nx2,$ny2,$flat2"\n'
        'echo "c3=$mx3,$my3,$mz3,$nx3,$ny3,$nz3,$nanx,$nany"\n'
    )
    res = milk_bash(script, tmp_path)
    assert res.returncode == 0, res.stdout + res.stderr
    assert value(res.stdout, "c2") == "5,9,11,3,149"
    assert value(res.stdout, "c3") == "3,4,2,6,1,3,-1,-1"

    bad = milk_bash(
        'mem.mk2Dim im2 8 8 >/dev/null; @calc "px, py, pz = iargmax(im2)"',
        tmp_path,
    )
    assert bad.returncode != 0


def test_phase7_library_and_spatial_ops(tmp_path):
    """Verify Phase 7 library calls, spatial ops, and new §3.2 features (§5 Phase 7)."""
    script = (
        "mem.mk2Dim A 8 8 >/dev/null\n"
        "mem.mk2Dim B 8 8 >/dev/null\n"
        '@calc "A = 2.0 + 0.1 * ix(A) + 0.2 * iy(A); B = 1.5 - 0.05 * ix(B) + 0.15 * iy(B)"\n'
        '@calc "M1 = A @ B; M2 = matmul(A, B); err_mm = imax(abs(M1 - M2))"\n'
        '@calc "T1 = A.T @ B; T2 = matmul(transpose(A), B); err_tr = imax(abs(T1 - T2))"\n'
        "@calc 'E1 = einsum(\"ij,jk->ik\", A, B); err_ein = imax(abs(E1 - M2))'\n"
        '@calc "U, S, Vt = svd(A); Sonly = svd(A); err_svd = imax(abs(S - Sonly))"\n'
        '@calc "H = A + A.T; W, V = eigh(H); Wonly = eigh(H); err_eigh = imax(abs(W - Wonly))"\n'
        '@calc "F = fft(A); R = real(ifft(F)); Fs = fftshift(F); err_fft = imax(abs(R - A))"\n'
        '@calc "Hs = hstack(A, B); Vs = vstack(A, B); Ds = dstack(A, B); Cs = cat(0, A, B)"\n'
        '@calc "err_cat = imax(abs(Hs - Cs)) + abs(icount(Hs) - 128) + abs(icount(Ds) - 128)"\n'
        '@calc "R4 = rot90(rot90(A, 1), 3); R2 = rot90(rot90(A, 2), 2); '
        'err_rot = imax(abs(R4 - A)) + imax(abs(R2 - A))"\n'
        "mem.mk2Dim K 3 3 >/dev/null\n"
        '@calc "K = 1.0 / 9.0; Cv = conv2(A, K); Bm = boxmean(A, 1); '
        'err_conv = imax(abs(Cv[1:7, 1:7] - Bm[1:7, 1:7]))"\n'
        '@calc "im = 10.0 + 0 * A; im[3, 4] = 50.0; im[6, 2] = 2.0"\n'
        '@calc "mn = imin3(im); mx = imax3(im); bm = boxmean(im, 1)"\n'
        '@calc "v_mx = mx[4, 4]; v_mn = mn[5, 2]; v_bm = bm[3, 4]"\n'
        "@calc -v err_mm,err_tr,err_ein,err_svd,err_eigh,err_fft,err_cat,err_rot,err_conv "
        '"err_mm; err_tr; err_ein; err_svd; err_eigh; err_fft; err_cat; err_rot; err_conv"\n'
        '@calc -v v_mx,v_mn,v_bm "v_mx; v_mn; v_bm"\n'
        'echo "errs=$err_mm,$err_tr,$err_ein,$err_svd,$err_eigh,$err_fft,$err_cat,$err_rot,$err_conv"\n'
        'echo "res=$v_mx,$v_mn,$v_bm"\n'
    )
    res = milk_bash(script, tmp_path)
    assert res.returncode == 0, res.stdout + res.stderr
    errs = [float(x) for x in value(res.stdout, "errs").split(",")]
    for e in errs:
        assert e <= 1e-5, f"Phase 7 library op error exceeded tolerance: {errs}"
    vals = [float(x) for x in value(res.stdout, "res").split(",")]
    assert vals[0] == pytest.approx(50.0, rel=1e-6)
    assert vals[1] == pytest.approx(2.0, rel=1e-6)
    assert vals[2] == pytest.approx((8 * 10.0 + 50.0) / 9.0, rel=1e-5)


def test_datasets_and_generated_expressions(tmp_path):
    """Verify §6.2 datasets and §6.3 expression generator on the calculator engine."""
    ids = all_dataset_ids(include_large=True)
    for expected in ("default", "random", "offset_0", "offset_3e4", "cubes"):
        assert expected in DATASETS and expected in ids
    assert offset_dataset_ids() == [
        "offset_0",
        "offset_1e3",
        "offset_1e4",
        "offset_3e4",
    ]
    fns = {f["name"] for f in load_functions()}
    for required in ("sin", "cos", "log", "pow", "where", "imean", "istd", "istats"):
        assert required in fns

    cases = list(
        generate_cases(
            count=25,
            seed=42,
            kind="elementwise",
            datasets=["default"],
            include_errors=False,
        )
    )
    script_lines = [
        '@calc "A = 0.2 + 0.5 * urand(zeros(16, 16), 1); B = 0.2 + 0.5 * urand(zeros(16, 16), 2); '
        "C = zeros(16, 16); D = 0.3 + 0.4 * urand(zeros(16, 16), 3); "
        "G = 0.5 + 0.5 * urand(zeros(16, 16), 4); O = 0.1 * urand(zeros(16, 16), 5); "
        'M = where(A > 0.3, 1.0, 0.0); U = zeros(16, 16); V = zeros(16, 16); H = 7"\n'
        '@calc "x = 1.5; y = -0.5; n = 4; k = 2; t = 0.25; alpha = 0.5"\n'
        '@calc "def soft_thresh(x, t) = where(abs(x) > t, x, 0.0)"\n'
    ]
    for c in cases:
        script_lines.append(f'@calc "{c["stmt"]}" >/dev/null\n')
    res = milk_bash("".join(script_lines), tmp_path)
    assert res.returncode == 0, res.stdout + res.stderr
