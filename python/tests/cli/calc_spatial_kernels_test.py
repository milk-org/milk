"""
Regression tests for spatial, remapping, and domain kernels in the milk.calc / @calc engine:
- medfilt3(im), medfilt5(im), medfilt7(im): exact 2D SIMD median filters (3x3, 5x5, 7x7)
- pixremap(im, map[, fill]): fused indexed gather
- pixunmap(im, map[, n]): fused segmented-mean reduction with CSR gather cache
- rprof(im[, cx, cy]): azimuthal radial profile reduction and lut(ir(im), rprof(im)) composition
- gauss(im, sigma[, sigmay]): separable 2-pass 2D Gaussian blur
- interp2(im, xmap, ymap): 2D sub-pixel bilinear interpolation gather
- cx, cy[, flux] = icentroid(im[, th]): single-pass thresholded flux-weighted centroid reduction
"""

from __future__ import annotations

import math
import numpy as np
import pytest
from astropy.io import fits

from milk.cliwrap import HAVE_CLI
from tests.cli.conftest import milk_bash, value

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.timeout(120),
]


def _np_medfilt2d_clamped(arr: np.ndarray, k: int) -> np.ndarray:
    """Exact reference 2D median filter with edge-clamped (nearest) border padding."""
    r = k // 2
    padded = np.pad(arr, ((r, r), (r, r)), mode="edge")
    ny, nx = arr.shape
    windows = np.lib.stride_tricks.sliding_window_view(padded, (k, k))
    return np.median(windows.reshape(ny, nx, k * k), axis=-1)


def _np_gauss2d_clamped(arr: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Exact reference separable 2D Gaussian filter matching @calc edge-clamped kernel."""
    out = arr.astype(np.float64, copy=True)
    ny, nx = out.shape
    if sx > 0.0:
        rx = min(int(math.ceil(3.0 * sx)), 32)
        kx = np.arange(-rx, rx + 1, dtype=np.float64)
        wx = np.exp(-0.5 * (kx / sx) ** 2)
        wx /= np.sum(wx)
        padded_x = np.pad(out, ((0, 0), (rx, rx)), mode="edge")
        tmp = np.zeros_like(out)
        for d in range(-rx, rx + 1):
            tmp += wx[d + rx] * padded_x[:, rx + d : rx + d + nx]
        out = tmp
    if sy > 0.0 and ny > 1:
        ry = min(int(math.ceil(3.0 * sy)), 32)
        ky = np.arange(-ry, ry + 1, dtype=np.float64)
        wy = np.exp(-0.5 * (ky / sy) ** 2)
        wy /= np.sum(wy)
        padded_y = np.pad(out, ((ry, ry), (0, 0)), mode="edge")
        tmp = np.zeros_like(out)
        for d in range(-ry, ry + 1):
            tmp += wy[d + ry] * padded_y[ry + d : ry + d + ny, :]
        out = tmp
    return out


def test_medfilt_3_5_7_exact_and_aliasing(tmp_path):
    """medfilt3, medfilt5, medfilt7 match exact edge-clamped 2D median across float32 and float64."""
    rng = np.random.default_rng(42)
    # Use non-multiple-of-SIMD dimensions (19x13) to exercise both SIMD lanes and scalar tail
    ny, nx = 13, 19
    arr = rng.normal(loc=10.0, scale=5.0, size=(ny, nx)).astype(np.float32)
    # Inject hot pixels and a 2x2 cosmic-ray cluster
    arr[4, 7] = 500.0
    arr[8:10, 12:14] = -300.0
    fits.writeto(tmp_path / "in_f32.fits", arr, overwrite=True)
    fits.writeto(tmp_path / "in_f64.fits", arr.astype(np.float64), overwrite=True)

    res = milk_bash(
        'iofits.loadfits "in_f32.fits" im32 0 >/dev/null\n'
        'iofits.loadfits "in_f64.fits" im64 0 >/dev/null\n'
        '@calc "m3_32 = medfilt3(im32); m5_32 = medfilt5(im32); m7_32 = medfilt7(im32)"\n'
        '@calc "m3_64 = medfilt3(im64); m5_64 = medfilt5(im64); m7_64 = medfilt7(im64)"\n'
        # Fused expression with compound argument and post-scaling, plus in-place aliasing
        '@calc "f_32 = (medfilt3(im32 - 1.0) + 1.0) * 2.0"\n'
        '@calc "alias = im32; alias = medfilt5(alias)"\n'
        '@calc -v e_alias "imax(abs(alias - m5_32))"\n'
        'iofits.saveFITS m3_32 "m3_32.fits" >/dev/null\n'
        'iofits.saveFITS m5_32 "m5_32.fits" >/dev/null\n'
        'iofits.saveFITS m7_32 "m7_32.fits" >/dev/null\n'
        'iofits.saveFITS m3_64 "m3_64.fits" >/dev/null\n'
        'iofits.saveFITS m5_64 "m5_64.fits" >/dev/null\n'
        'iofits.saveFITS m7_64 "m7_64.fits" >/dev/null\n'
        'iofits.saveFITS f_32 "f_32.fits" >/dev/null\n'
        'echo "alias=$e_alias"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert res.returncode == 0, out
    assert float(value(res.stdout, "alias")) == 0.0, out

    for k, name in [(3, "m3"), (5, "m5"), (7, "m7")]:
        ref = _np_medfilt2d_clamped(arr, k)
        got32 = fits.getdata(tmp_path / f"{name}_32.fits")
        got64 = fits.getdata(tmp_path / f"{name}_64.fits")
        np.testing.assert_allclose(got32, ref, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(got64, ref, rtol=1e-12, atol=1e-12)

    got_fused = fits.getdata(tmp_path / "f_32.fits")
    np.testing.assert_allclose(
        got_fused, 2.0 * _np_medfilt2d_clamped(arr, 3), rtol=1e-5, atol=1e-5
    )


def test_pixremap_and_pixunmap_and_csr_cache(tmp_path):
    """pixremap and pixunmap match reference, including CSR cache."""
    (tmp_path / "unmap.calc").write_text(
        "u_unit = pixunmap(cam_u, map_u, 4)\n" "rem_unit = pixremap(u_unit, map_u)\n"
    )
    # Create an 8x6 image and an INT32 segment map with 5 bins (0..4) plus -1 masked pixels
    res = milk_bash(
        "mem.mk2Dim im 8 6 >/dev/null\n"
        '@calc "im = (ix(im, 0, 0) + 1) * 10 + iy(im, 0, 0)"\n'
        '@calc "map = i32(where(ix(im, 0, 0) == 0, -1, (ix(im, 0, 0) + iy(im, 0, 0)) % 5))"\n'
        '@calc "u = pixunmap(im, map)"\n'
        '@calc "u8 = pixunmap(im, map, 8)"\n'
        '@calc -v ux,uy,u8x,u8y "s.u.xsize; s.u.ysize; s.u8.xsize; s.u8.ysize"\n'
        # Compare bin 0 against explicit masked mean over map == 0
        '@calc -v u_err "abs(u[0,0] - imean(im, map == 0))"\n'
        # Test pixremap on a lookup vector with out-of-bounds (-1) fill value
        '@calc "rem0 = pixremap(u, map)"\n'
        '@calc "remf = pixremap(u, map, -99.5)"\n'
        '@calc "rem_ref = where(map >= 0, lut(map, u), 0)"\n'
        '@calc -v r_err,rf_neg "imax(abs(rem0 - rem_ref)); remf[0, 3]"\n'
        # Round-trip on piecewise constant image: pixremap(pixunmap(z, map), map) == z where map >= 0
        '@calc "z = where(map >= 0, (map + 1) * 7.5, 0); z_rt = pixremap(pixunmap(z, map), map)"\n'
        '@calc -v rt_err "imax(abs(z_rt - z))"\n'
        # Multi-frame milk.calc unit exercising CSR gather cache reuse and dynamic map invalidation
        'mem.mk2Dim "@S:cam_u" 8 8 >/dev/null; mem.mk2Dim "@S:map_u" 8 8 >/dev/null\n'
        '@calc "map_u = i32(ix(map_u, 0, 0) / 2); cam_u = (map_u + 1) * 10.0"\n'
        "milk-fpsexec-milk-calc -n uunit -t cam_u -f unmap.calc fpsinit >/dev/null\n"
        "milk-fpsexec-milk-calc -n uunit run >run.log 2>&1 & pid=$!\n"
        'sleep 0.5; @calc "cam_u = (map_u + 1) * 20.0"; sleep 0.2\n'
        '@calc -v c1_0,c1_3 "u_unit[0,0]; u_unit[3,0]"\n'
        # Update map_u (cnt0 increments -> CSR cache must rebuild)
        '@calc "map_u = i32(iy(map_u, 0, 0) / 2); cam_u = (map_u + 1) * 5.0"; sleep 0.2\n'
        "kill -INT $pid 2>/dev/null; wait $pid 2>/dev/null\n"
        '@calc -v c2_0,c2_3,rem_err "u_unit[0,0]; u_unit[3,0]; imax(abs(rem_unit - cam_u))"\n'
        'echo "shape=$ux,$uy,$u8x,$u8y"\n'
        'echo "u_err=$u_err"\n'
        'echo "r_err=$r_err,$rf_neg"\n'
        'echo "rt_err=$rt_err"\n'
        'echo "csr=$c1_0,$c1_3,$c2_0,$c2_3,$rem_err"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert res.returncode == 0, out
    assert value(res.stdout, "shape") == "5,1,8,1", out
    assert float(value(res.stdout, "u_err")) < 1e-5, out
    r_err, rf_neg = [float(x) for x in value(res.stdout, "r_err").split(",")]
    assert r_err < 1e-5 and abs(rf_neg - (-99.5)) < 1e-5, out
    assert float(value(res.stdout, "rt_err")) < 1e-5, out
    assert value(res.stdout, "csr") == "20,80,5,20,0", out


def test_rprof_and_radial_subtraction(tmp_path):
    """rprof(im[, cx, cy]) azimuthal average and 2D radial subtraction via lut(ir(im), rprof(im))."""
    res = milk_bash(
        "mem.mk2Dim im 32 32 >/dev/null\n"
        # Pure radial function f(r) = 100 - 3 * round(r)
        '@calc "r_int = floor(ir(im) + 0.5); im = 100.0 - 3.0 * r_int"\n'
        '@calc "rp = rprof(im)"\n'
        '@calc -v rpx,rpy,rp0,rp5,rp10 "s.rp.xsize; s.rp.ysize; rp[0,0]; rp[5,0]; rp[10,0]"\n'
        # Subtract azimuthal profile in 2D using lut(r_int, rprof(im))
        '@calc "res_iso = im - lut(r_int, rprof(im))"\n'
        '@calc -v iso_err "imax(abs(res_iso))"\n'
        # Off-center radial profile around (10, 12)
        '@calc "r_off = floor(ir(im, 10, 12) + 0.5); im_off = 50.0 + 2.0 * r_off"\n'
        '@calc "rp_off = rprof(im_off, 10, 12)"\n'
        '@calc -v off0,off4 "rp_off[0,0]; rp_off[4,0]"\n'
        'echo "rp=$rpx,$rpy,$rp0,$rp5,$rp10"\n'
        'echo "iso_err=$iso_err"\n'
        'echo "off=$off0,$off4"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert res.returncode == 0, out
    rpx, rpy, rp0, rp5, rp10 = [float(x) for x in value(res.stdout, "rp").split(",")]
    # On 32x32 centered at (16,16), max corner is (-16,-16) -> r = sqrt(512) = 22.627 -> round = 23 -> len 24
    assert int(rpx) == 24 and int(rpy) == 1, out
    assert (
        abs(rp0 - 100.0) < 1e-5 and abs(rp5 - 85.0) < 1e-5 and abs(rp10 - 70.0) < 1e-5
    ), out
    assert float(value(res.stdout, "iso_err")) < 1e-5, out
    off0, off4 = [float(x) for x in value(res.stdout, "off").split(",")]
    assert abs(off0 - 50.0) < 1e-5 and abs(off4 - 58.0) < 1e-5, out


def test_gauss_separable_2d_blur(tmp_path):
    """gauss(im, sigma[, sigmay]) matches exact separable 2D edge-clamped Gaussian convolution."""
    rng = np.random.default_rng(7)
    arr = rng.normal(size=(16, 20)).astype(np.float32)
    fits.writeto(tmp_path / "gin.fits", arr, overwrite=True)

    res = milk_bash(
        'iofits.loadfits "gin.fits" gin 0 >/dev/null\n'
        '@calc "g_iso = gauss(gin, 1.8); g_aniso = gauss(gin, 2.2, 0.9)"\n'
        '@calc "hp = gin - gauss(gin, 1.8)"\n'
        '@calc "ones_g = gauss(gin * 0 + 5.0, 2.5)"\n'
        '@calc -v const_err,hp_err "imax(abs(ones_g - 5.0)); imax(abs(hp - (gin - g_iso)))"\n'
        'iofits.saveFITS g_iso "g_iso.fits" >/dev/null\n'
        'iofits.saveFITS g_aniso "g_aniso.fits" >/dev/null\n'
        'echo "err=$const_err,$hp_err"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert res.returncode == 0, out
    const_err, hp_err = [float(x) for x in value(res.stdout, "err").split(",")]
    assert const_err < 1e-5 and hp_err < 1e-6, out

    got_iso = fits.getdata(tmp_path / "g_iso.fits")
    got_aniso = fits.getdata(tmp_path / "g_aniso.fits")
    ref_iso = _np_gauss2d_clamped(arr, 1.8, 1.8)
    ref_aniso = _np_gauss2d_clamped(arr, 2.2, 0.9)
    np.testing.assert_allclose(got_iso, ref_iso, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(got_aniso, ref_aniso, rtol=1e-5, atol=1e-5)


def test_interp2_bilinear_gather(tmp_path):
    """interp2(im, xmap, ymap) performs exact 2D sub-pixel bilinear interpolation with edge clamping."""
    res = milk_bash(
        "mem.mk2Dim im 12 10 >/dev/null\n"
        # Linear plane f(x, y) = 2.5 * x - 1.25 * y + 7.0
        '@calc "x0 = ix(im, 0, 0); y0 = iy(im, 0, 0); im = 2.5 * x0 - 1.25 * y0 + 7.0"\n'
        # Shift by (+0.35, +0.65) inside interior [0..10, 0..8]
        '@calc "xm = min(x0 + 0.35, 10.5); ym = min(y0 + 0.65, 8.5)"\n'
        '@calc "w = interp2(im, xm, ym); w_ref = 2.5 * xm - 1.25 * ym + 7.0"\n'
        '@calc -v plane_err "imax(abs(w - w_ref))"\n'
        # Test out-of-bounds clamping at negative and > (N-1) coordinates
        '@calc "xc = zeros(2, 1); yc = zeros(2, 1); xc[0,0] = -5.0; yc[0,0] = -3.0; xc[1,0] = 99.0; yc[1,0] = 99.0"\n'
        '@calc "wc = interp2(im, xc, yc)"\n'
        '@calc -v c0,c1,r0,r1 "wc[0,0]; wc[1,0]; im[0,0]; im[11,9]"\n'
        # Test in-place self-aliasing safety
        '@calc "al = im; al = interp2(al, xm, ym)"\n'
        '@calc -v al_err "imax(abs(al - w))"\n'
        'echo "plane_err=$plane_err"\n'
        'echo "clamp=$c0,$c1,$r0,$r1"\n'
        'echo "al_err=$al_err"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert res.returncode == 0, out
    assert float(value(res.stdout, "plane_err")) < 1e-5, out
    c0, c1, r0, r1 = [float(x) for x in value(res.stdout, "clamp").split(",")]
    assert abs(c0 - r0) < 1e-5 and abs(c1 - r1) < 1e-5, out
    assert float(value(res.stdout, "al_err")) == 0.0, out


def test_icentroid_subpixel_and_threshold_and_unit(tmp_path):
    """cx, cy[, flux] = icentroid(im[, th]) computes exact sub-pixel CoG in @calc and milk.calc units."""
    (tmp_path / "cent.calc").write_text("cx, cy, flux = icentroid(spot, 5.0)\n")
    res = milk_bash(
        'mem.mk2Dim "@S:spot" 32 32 >/dev/null\n'
        # Synthetic Gaussian spot centered at (14.35, 17.80) + uniform background pedestal of 5.0
        '@calc "dx = ix(spot, 14.35, 17.80); dy = iy(spot, 14.35, 17.80)"\n'
        '@calc "spot = 5.0 + 100.0 * exp(-0.5 * (dx*dx + dy*dy) / (2.2*2.2))"\n'
        # Threshold at 5.0 removes the uniform pedestal exactly
        '@calc "cx, cy, fl = icentroid(spot, 5.0)"\n'
        '@calc "cx2, cy2 = icentroid(spot - 5.0)"\n'
        # Below-threshold / all-zero image yields NaN, NaN, 0
        '@calc "z_cx, z_cy, z_fl = icentroid(spot, 500.0)"\n'
        '@calc -v vcx,vcy,vfl,vcx2,vcy2,zfl "cx; cy; fl; cx2; cy2; z_fl"\n'
        '@calc -v z_isnan "isnan(z_cx) && isnan(z_cy)"\n'
        # Test inside a compiled milk.calc unit
        'milk-fpsexec-milk-calc -n cunit -t spot -p "cx:float=0 cy:float=0 flux:float=0" -f cent.calc fpsinit >/dev/null\n'
        "milk-fpsexec-milk-calc -n cunit run >run.log 2>&1 & pid=$!\n"
        'sleep 0.5; @calc "spot = spot + 0"; sleep 0.2\n'
        "kill -INT $pid 2>/dev/null; wait $pid 2>/dev/null\n"
        '@calc -v ucx,ucy,ufl "f.cunit.cx; f.cunit.cy; f.cunit.flux"\n'
        'echo "c1=$vcx,$vcy,$vfl"\n'
        'echo "c2=$vcx2,$vcy2"\n'
        'echo "zero=$z_isnan,$zfl"\n'
        'echo "unit=$ucx,$ucy,$ufl"',
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert res.returncode == 0, out
    vcx, vcy, vfl = [float(x) for x in value(res.stdout, "c1").split(",")]
    vcx2, vcy2 = [float(x) for x in value(res.stdout, "c2").split(",")]
    assert abs(vcx - 14.35) < 1e-4 and abs(vcy - 17.80) < 1e-4 and vfl > 1000.0, out
    assert abs(vcx2 - vcx) < 1e-5 and abs(vcy2 - vcy) < 1e-5, out
    assert value(res.stdout, "zero") == "1,0", out
    ucx, ucy, ufl = [float(x) for x in value(res.stdout, "unit").split(",")]
    assert abs(ucx - 14.35) < 1e-3 and abs(ucy - 17.80) < 1e-3 and ufl > 500.0, out
