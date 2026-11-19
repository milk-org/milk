"""013-3 e2e (milk): mlat, mkpokes zonal, measlinresp on the mlatsim simulator.

The simulated WFS is a delayed copy of the DM, so RMmodesWFS must equal RMmodesDM. It fails if
the latency handed over from mlat to measlinresp is wrong. See specs/013-*/brief.md §5.
Needs milk and cacao binaries; about 10 s of measurement (256 modes, 10 cycles at ~1 kHz).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from moka.operate.pipeline import Pipeline
from moka.operate.plans import Plan, Step, run, run_plan
from moka.provision.rootdir import init

pytestmark = pytest.mark.e2e

RESOURCES = Path(__file__).parents[1] / "resources"
MEASLINRESP = """
[dmhint]
xsize = 16
ysize = 16

[compute_units.measlinresp]
exec = "cacao-fpsexec-cacaoac-measlinresp"

[compute_units.measlinresp.fps]
streamin = "mlatsim_dm"
streamout = "mlatsim_wfs"
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
        Step("RunStart", ("dmdelay",)),
        Step("RunStart", ("wfscam",)),
        Step("RunStart", ("mlat",)),
        Step("WaitRunEnd", ("mlat",)),
        Step("MakePokes"),
        Step("SetLinResp", ("measlinresp",)),
        Step("RunStart", ("measlinresp",)),
        Step("WaitLinResp", ("measlinresp",)),
        Step("RunStop", ("wfscam", "dmdelay")),
    ),
)


def _rt_available() -> bool:
    """milk-makecsetandrt is installed setuid root, so procinfo.RTprio takes effect."""
    path = shutil.which("milk-makecsetandrt")
    st = os.stat(path) if path else None
    return st is not None and st.st_uid == 0 and bool(st.st_mode & 0o4000)


@pytest.fixture
def linrespsim(tmp_path):
    """mlatsim plus a measlinresp unit, deployed in tmp_path; torn down (tmux included) at the end."""
    conf = tmp_path / "mlatsim-conf"
    shutil.copytree(RESOURCES / "mlatsim-conf", conf)
    with (conf / "conf.toml").open("a") as stream:
        stream.write(MEASLINRESP)
    pp = Pipeline(init(conf))
    run_plan(pp, "deploy")
    yield pp
    try:
        run_plan(Pipeline(pp.root_folder), "teardown")
    finally:
        for cu in pp.config.compute_units:
            argv = ["tmux", "kill-session", "-t", pp.config.fps_name(cu)]
            subprocess.run(argv, capture_output=True)


def test_measlinresp_of_a_delayed_copy_is_the_identity(linrespsim):
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
