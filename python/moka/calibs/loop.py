"""The loop's tasks SetLoopOff, LoadCM and the spool plan (017)."""

from __future__ import annotations

import numpy as np

from ..errors import ConfigError
from ..operate.plans import PLANS, Plan, Step
from ..operate.tasks import TASKS, TaskOnCompuSet, TaskOnOneCompu, stream_shape
from . import cm as _cm, linresp as _lr  # modules: may be mid-import (cycle via plans)


class SetLoopOff(TaskOnOneCompu):
    """Writes loopON false on the unit (default: the one cacao-fpsexec-mfilt unit)."""

    def locate_defaults(self) -> None:
        exe = "cacao-fpsexec-mfilt"
        units = self.pipeline.config.compute_units
        found = [cu for cu, decl in units.items() if decl.exec == exe]
        if len(found) != 1:
            raise ConfigError(
                (
                    f"SetLoopOff: no {exe} unit."
                    if not found
                    else f"SetLoopOff: {exe} units {found}; name one in the step."
                ),
                pipeline=self.pipeline.short_name,
            )
        self.compus_names = tuple(found)

    _refusal = TaskOnOneCompu._undeployed

    def forward(self) -> None:
        self.fps["loopON"] = False  # type: ignore[index]

    def success(self) -> bool:
        return self.fps is not None and not self.fps["loopON"]


class LoadCM(TaskOnCompuSet):
    """calib/CMmodes{WFS,DM}.fits into the immodes streams of the subset (wfs2modes, modes2dm)."""

    COUNT = 2  # in the step's order: WFS side first

    def _files(self):
        return [_lr._calib(self, f) for f in (_cm.CM_WFS, _cm.CM_DM)]

    def _targets(self) -> list[tuple[str, np.ndarray]]:
        """(stream, cube); FITS (NAXIS1, NAXIS2, NAXIS3) = numpy (n, y, x) -> stream (x, y, n)."""
        from astropy.io import fits

        targets = []
        for c, f in zip(self.compute_units.values(), self._files(), strict=True):
            assert c.fps is not None
            cube = np.asarray(fits.getdata(f), np.float32)
            targets.append((str(c.fps["immodes"]), cube.transpose(2, 1, 0)))
        return targets

    def _refusal(self) -> str | None:
        from astropy.io import fits

        for path in self._files():
            if not path.is_file():
                return f"{path} does not exist; run `moka mkcm`."
        wfs, dm = map(fits.getheader, self._files())
        if wfs["NAXIS3"] != dm["NAXIS3"]:
            return f"{_cm.CM_WFS} has {wfs['NAXIS3']} modes, {_cm.CM_DM} has {dm['NAXIS3']}."
        hint = self.pipeline.config.dmhint
        if hint is not None and (dm["NAXIS1"], dm["NAXIS2"]) != (
            hint.xsize,
            hint.ysize,
        ):
            return f"{_cm.CM_DM} is {dm['NAXIS1']} x {dm['NAXIS2']}, [dmhint] is {hint.xsize} x {hint.ysize}."
        for compu in self.compute_units.values():
            if compu.fps is None:
                return f"{compu.fpsname} is missing; run deploy."
            if compu.fps.run_isrunning():  # it copied its matrix at run start
                return f"{compu.fpsname} is running; stop it first."
        return None

    def forward(self) -> None:
        from pyMilk.interfacing.shm import SHM

        for name, cube in self._targets():
            SHM(name, cube, symcode=0).close()

    def success(self) -> bool:
        """Both streams hold the files' sizes and values: a new CM of the same size is not loaded yet."""
        from pyMilk.interfacing.shm import SHM

        if any(c.fps is None for c in self.compute_units.values()) or not all(
            p.is_file() for p in self._files()
        ):
            return False
        for name, cube in self._targets():
            if stream_shape(name) != cube.shape:
                return False
            shm = SHM(name, symcode=0)
            data = shm.get_data()
            shm.close()
            if not np.array_equal(data.reshape(cube.shape), cube):
                return False
        return True


for _task in (SetLoopOff, LoadCM):
    TASKS.register(_task.__name__, _task)

PLANS.register(
    "spool",
    Plan(
        "spool",
        (
            Step("SetLoopOff", ("mfilt",)),  # before anything runs
            # They copy matrix and mode count at run start.
            Step("RunStop", ("modes2dm", "mfilt", "wfs2modes")),
            Step("LoadCM", ("wfs2modes", "modes2dm")),
            Step("RunStart", ("acquWFS",)),  # skipped if it runs, as in mlat
            # Ordered, one at a time, since the step names its units.
            Step("RunStart", ("wfs2modes", "mfilt", "modes2dm")),
            Step("SetLoopOff", ("mfilt",)),  # checked again once mfilt runs
        ),
    ),
)
