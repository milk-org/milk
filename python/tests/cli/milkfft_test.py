"""The milkfft plugin's transforms give numpy's results.

Purpose:     checked against numpy, a wrong axis, scale, input type or image lookup in the FFT
             module commands shows at once.
Does:        fft.do1Drfft: real -> half-spectrum along x of each row (2D) or along z of each pixel
             (3D), for float, double and uint16 input, whatever the input's place in the image
             table; fft.dofft: 2D complex FFT, forward (-1) and backward (1, unnormalized);
             imgfilt.fconv: FFT convolution (2D real transform of image and kernel), centered
             like fftshift.
Covers:      plugins/milk-extra-src/fft (libmilkfft: dofft.c, dofft_1d.c, dofft_2d.c) and
             plugins/milk-extra-src/image_filter/fconvolve.c (libmilkimagefilter).
Uses:        conftest.py: milk_bash, _install_dir; numpy and astropy.io.fits.
See also:    plugins/milk-extra-src/fft/README.md.
Constraints: skipped when libmilkfft.so is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from milk.cliwrap import HAVE_CLI
from tests.cli.conftest import _install_dir, milk_bash

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.skipif(
        not (_install_dir() / "lib" / "libmilkfft.so").exists(),
        reason="milkfft plugin not built",
    ),
    pytest.mark.timeout(120),
]


def _save_reim(names: list[str]) -> str:
    return "".join(
        f'@calc "R_{n} = real({n}); I_{n} = imag({n})"\n'
        f'iofits.saveFITS R_{n} "o_R_{n}.fits" >/dev/null\n'
        f'iofits.saveFITS I_{n} "o_I_{n}.fits" >/dev/null\n'
        for n in names
    )


def _got(tmp_path, n: str) -> np.ndarray:
    re_ = fits.getdata(tmp_path / f"o_R_{n}.fits").astype(np.float64)
    return re_ + 1j * fits.getdata(tmp_path / f"o_I_{n}.fits")


def test_do1drfft_matches_numpy(tmp_path):
    rng = np.random.default_rng(5)
    data = {  # name: (array, numpy axis of the transform)
        "r2": (rng.normal(size=(3, 8)).astype(np.float32), -1),
        "u2": (rng.integers(0, 60000, size=(3, 7)).astype(np.uint16), -1),
        "c3": (rng.normal(size=(9, 2, 3)).astype(np.float32), 0),
        "d3": (rng.normal(size=(8, 2, 3)), 0),
        "u3": (rng.integers(0, 60000, size=(8, 2, 3)).astype(np.uint16), 0),
    }
    for name, (arr, _) in data.items():
        fits.writeto(tmp_path / f"{name}.fits", arr, overwrite=True)
    res = milk_bash(
        "".join(f'iofits.loadfits "{n}.fits" {n} 0 >/dev/null\n' for n in data)
        + "".join(f"fft.do1Drfft {n} F_{n} >/dev/null\n" for n in data)
        + _save_reim([f"F_{n}" for n in data]),
        tmp_path,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    for name, (arr, axis) in data.items():
        ref = np.fft.rfft(arr.astype(np.float64), axis=axis)
        tol = 1e-12 if arr.dtype == np.float64 else 1e-6
        np.testing.assert_allclose(
            _got(tmp_path, f"F_{name}"),
            ref,
            atol=tol * max(1.0, np.abs(ref).max()),
            err_msg=name,
        )


def test_dofft_directions_and_fconv(tmp_path):
    rng = np.random.default_rng(7)
    data = {n: rng.normal(size=(6, 8)).astype(np.float32) for n in ("za", "zb", "ka")}
    for name, arr in data.items():
        fits.writeto(tmp_path / f"{name}.fits", arr, overwrite=True)
    res = milk_bash(
        "".join(f'iofits.loadfits "{n}.fits" {n} 0 >/dev/null\n' for n in data)
        + '@calc "Z = cplx(za, zb)"\n'
        "fft.dofft Z FF -1 >/dev/null; fft.dofft Z FB 1 >/dev/null\n"
        "imgfilt.fconv za ka CV >/dev/null\n"
        'iofits.saveFITS CV "o_CV.fits" >/dev/null\n' + _save_reim(["FF", "FB"]),
        tmp_path,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    z = data["za"].astype(np.float64) + 1j * data["zb"]
    np.testing.assert_allclose(_got(tmp_path, "FF"), np.fft.fft2(z), atol=1e-4)
    np.testing.assert_allclose(
        _got(tmp_path, "FB"), z.size * np.fft.ifft2(z), atol=1e-4
    )
    a, k = (data[n].astype(np.float64) for n in ("za", "ka"))
    conv = np.real(np.fft.ifft2(np.fft.fft2(a) * np.fft.fft2(k)))
    np.testing.assert_allclose(
        fits.getdata(tmp_path / "o_CV.fits"), np.fft.fftshift(conv), atol=1e-4
    )
