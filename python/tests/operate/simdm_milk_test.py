"""014-2 (milk): the simulated DM of mlatsim-conf, dmcomb and dmturb.

Needs milk and cacao binaries. See specs/014-sim-dmcomb-turb-camsim/brief.md §4.4.
"""

from __future__ import annotations

import numpy as np
from pyMilk.interfacing.fps import FPS
from pyMilk.interfacing.shm import SHM

from moka.operate.plans import Plan, Step, run
from tests.conftestaux.polling import poll_until_or_fail


def _start(pp, *compus: str) -> None:
    run(pp, Plan("start", (Step("RunStart", compus),)))


def test_dmcomb_sums_its_channels(make_mlatsim):
    pp = make_mlatsim()
    _start(pp, "dmcomb")
    a = np.arange(256, dtype=np.float32).reshape(16, 16) / 100
    b = np.full((16, 16), 0.25, dtype=np.float32)
    SHM("dm07disp02").set_data(a)
    SHM("dm07disp03").set_data(b)
    dm = SHM("dm07disp")
    poll_until_or_fail(lambda: np.array_equal(dm.get_data(), a + b), 5.0)


def test_dmturb_writes_its_channel_into_the_dm(make_mlatsim):
    pp = make_mlatsim()
    _start(pp, "dmcomb", "dmturb")
    fps = FPS(pp.config.fps_name("dmturb"))
    turb, dm = SHM("dm07disp10"), SHM("dm07disp")
    fps["turbON"] = True
    poll_until_or_fail(lambda: turb.get_data().any(), 5.0)
    poll_until_or_fail(lambda: dm.get_data().any(), 5.0)
    fps["turbON"] = False  # else the next step writes turbulence again
    fps["turbZERO"] = True
    poll_until_or_fail(lambda: not turb.get_data().any(), 5.0)
