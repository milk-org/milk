"""013-3 e2e (milk): mlat, mkpokes zonal, measlinresp on the mlatsim simulator.

The simulated WFS is pywfssim on the delayed DM, seen by WFScamsim at F ADU in total, so each
mode of RMmodesWFS must match the numpy copy of the model below (015 §5.2). It fails if the latency handed over from mlat
to measlinresp is too short by 0.4 frame or more; too long by up to 2 frames still passes (LOOSE_ENDS, "define hardware latency"). See specs/013-*/brief.md §5.
Needs milk and cacao binaries; about 10 s of measurement (256 modes, 10 cycles at ~1 kHz).
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import pytest
from astropy.io import fits
from pyMilk.interfacing.shm import SHM

from moka.operate.plans import Plan, Step, run
from tests.conftestaux.polling import poll_until_or_fail

pytestmark = pytest.mark.e2e

F = 10240000.0  # total ADU: wfscam's fluxtotal / camgain in resources/mlatsim-conf

MEASLINRESP = """
[dmhint]
xsize = 16
ysize = 16

[compute_units.measlinresp]
exec = "cacao-fpsexec-cacaoac-measlinresp"

[compute_units.measlinresp.fps]
streamin = "dm07disp02"
streamout = "aol7_imWFS2"
ampl = 0.1
timing.NBave = 1
timing.NBexcl = 2
procinfo.enabled = true
procinfo.loopcntMax = 10
procinfo.triggermode = 0
procinfo.RTprio = 40          # max 49 (fps_processinfo_entries.c); applied by milk-makecsetandrt
"""
START = Plan("start", (Step("RunStart"),))  # dmcomb, dmdelay, pywfs, wfscam, acquWFS
CALIBRATE = Plan(
    "calibrate",
    (
        Step("RunStart", ("mlat",)),
        Step("WaitRunEnd", ("mlat",)),
        Step("MakePokes"),
        Step("SetLinResp", ("measlinresp",)),
        Step("RunStart", ("measlinresp",)),
        Step("WaitLinResp", ("measlinresp",)),
        Step("RunStop"),
    ),
)


def _pywfs(dm: np.ndarray) -> np.ndarray:
    """pywfssim (PyWFSsim.c) with every actuator lit, wlum 1, modamp 1: DM (y, x) -> (2Ny, 2Nx)."""
    ny, nx = dm.shape
    gx, gy = np.zeros_like(dm), np.zeros_like(dm)
    gx[:, :-1] = (dm[:, 1:] - dm[:, :-1]) * nx  # tilt in lambda/D
    gy[:-1, :] = (dm[1:, :] - dm[:-1, :]) * ny
    sx, sy = gx / np.sqrt(1 + gx**2), gy / np.sqrt(1 + gy**2)
    q = 1 / (4 * nx * ny)
    return np.block(
        [
            [q * (1 - sx) * (1 - sy), q * (1 + sx) * (1 - sy)],  # high x: right
            [q * (1 - sx) * (1 + sy), q * (1 + sx) * (1 + sy)],  # high y: bottom
        ]
    )


def _rt_available() -> bool:
    """milk-makecsetandrt is installed setuid root, so procinfo.RTprio takes effect."""
    path = shutil.which("milk-makecsetandrt")
    st = os.stat(path) if path else None
    return st is not None and st.st_uid == 0 and bool(st.st_mode & 0o4000)


def test_measlinresp_of_the_pywfs_matches_the_model(make_mlatsim):
    linrespsim = make_mlatsim(MEASLINRESP)
    run(linrespsim, START)
    wfs = SHM("mlatsim_wfs")
    assert wfs.nptype == np.uint16 and wfs.shape == (32, 32)
    flat = F / 1024  # exact in float32
    poll_until_or_fail(
        lambda: (wfs.get_data() == flat).all(), 5.0
    )  # DM channels at zero
    run(linrespsim, CALIBRATE, {"type": "zonal"})
    calib = linrespsim.root_folder / "calib"
    pokes = fits.getdata(calib / "RMmodesDM.fits").astype(float)
    response = fits.getdata(calib / "RMmodesWFS.fits")
    assert pokes.shape == (256, 16, 16) and response.shape == (256, 32, 32)
    a = 0.1  # measlinresp's ampl
    expected = np.array([F * (_pywfs(a * p) - _pywfs(-a * p)) / (2 * a) for p in pokes])
    # uint16 truncation: each frame is off by < 1 ADU, so a mode by < 1 / (2 * a) = 5 ADU.
    error = np.abs(response - expected).max(axis=(1, 2))
    print(f"largest |RMmodesWFS - expected|: {error.max():.4g} ADU")
    wrong = error > 6
    # Without RT priority, scheduler stalls in measlinresp's loop corrupt a few reads.
    assert wrong.sum() <= (0 if _rt_available() else len(wrong) // 20), np.flatnonzero(
        wrong
    )
    assert not (calib / "RMmodesWFS.inprogress.fits").exists()
