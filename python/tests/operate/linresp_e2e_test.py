"""013-3 e2e (milk): mlat, mkpokes zonal, measlinresp on the mlatsim simulator.

The simulated WFS is a delayed copy of the DM, so RMmodesWFS must equal RMmodesDM. It fails if
the latency handed over from mlat to measlinresp is too short by 0.4 frame or more; too long by
up to 2 frames still passes (LOOSE_ENDS, "define hardware latency"). See specs/013-*/brief.md §5.
Needs milk and cacao binaries; about 10 s of measurement (256 modes, 10 cycles at ~1 kHz).
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import pytest
from astropy.io import fits

from moka.operate.plans import Plan, Step, run

pytestmark = pytest.mark.e2e

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
CALIBRATE = Plan(
    "calibrate",
    (
        Step("RunStart"),  # runstart: dmcomb, dmdelay, wfscam, acquWFS
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
    run(linrespsim, CALIBRATE, {"type": "zonal"})
    calib = linrespsim.root_folder / "calib"
    pokes = fits.getdata(calib / "RMmodesDM.fits")
    response = fits.getdata(calib / "RMmodesWFS.fits")
    assert pokes.shape == response.shape == (256, 16, 16)
    wrong = np.abs(response - pokes).max(axis=(1, 2)) > 1e-5
    # Without RT priority, scheduler stalls in measlinresp's loop corrupt a few reads.
    assert wrong.sum() <= (0 if _rt_available() else len(wrong) // 20), np.flatnonzero(
        wrong
    )
    assert not (calib / "RMmodesWFS.inprogress.fits").exists()
