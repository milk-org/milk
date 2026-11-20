"""013-3 e2e (milk): mlat, mkpokes zonal, measlinresp on the mlatsim simulator.

The simulated WFS is a delayed copy of the DM seen by WFScamsim at 1000 ADU per unit, so
RMmodesWFS / 1000 must equal RMmodesDM (014 §5.2). It fails if the latency handed over from mlat
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

K = 1000.0  # ADU per DM unit: wfscam's fluxtotal / camgain in resources/mlatsim-conf

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
START = Plan("start", (Step("RunStart"),))  # runstart: dmcomb, dmdelay, wfscam, acquWFS
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


def _rt_available() -> bool:
    """milk-makecsetandrt is installed setuid root, so procinfo.RTprio takes effect."""
    path = shutil.which("milk-makecsetandrt")
    st = os.stat(path) if path else None
    return st is not None and st.st_uid == 0 and bool(st.st_mode & 0o4000)


def test_measlinresp_of_a_delayed_copy_is_the_identity(make_mlatsim):
    linrespsim = make_mlatsim(MEASLINRESP)
    run(linrespsim, START)
    wfs = SHM("mlatsim_wfs")
    assert wfs.nptype == np.uint16
    poll_until_or_fail(lambda: (wfs.get_data() == K).all(), 5.0)  # DM channels at zero
    run(linrespsim, CALIBRATE, {"type": "zonal"})
    calib = linrespsim.root_folder / "calib"
    pokes = fits.getdata(calib / "RMmodesDM.fits")
    response = fits.getdata(calib / "RMmodesWFS.fits")
    assert pokes.shape == response.shape == (256, 16, 16)
    # uint16 truncation: each frame is off by < 1 ADU, so a mode by < 1 / (2 * ampl) = 5 ADU.
    error = np.abs(response / K - pokes).max(axis=(1, 2))
    print(f"largest |RMmodesWFS / K - RMmodesDM|: {error.max():.4g}")
    wrong = error > 0.01
    # Without RT priority, scheduler stalls in measlinresp's loop corrupt a few reads.
    assert wrong.sum() <= (0 if _rt_available() else len(wrong) // 20), np.flatnonzero(
        wrong
    )
    assert not (calib / "RMmodesWFS.inprogress.fits").exists()
