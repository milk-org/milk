"""Calibration plans and tasks: mkpokes (the poke cube) and measlinresp (the response to it)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..operate.plans import PLANS, Param, Plan, Step
from ..operate.tasks import TASKS, Task, TaskOnOneCompu, WaitRunEnd, stream_shape

# module, not name: mlat.py imports operate.plans, which imports us
from . import mlat as _mlat

if TYPE_CHECKING:
    from pyMilk.interfacing.fps import FPS

# In rootdir/calib. The FPS gets them relative to the run's cwd, rootdir/work (63 chars max).
POKES, DONE = "RMmodesDM.fits", "RMmodesWFS.fits"
INPROGRESS, PREV = "RMmodesWFS.inprogress.fits", "RMmodesWFS.prev.fits"


POKE_TYPES = ("hadamard", "zonal")


def _calib(task: Task, name: str) -> Path:
    return task.pipeline.root_folder / "calib" / name


class MakePokes(Task):
    """Writes calib/RMmodesDM.fits over the [dmhint] mask; params["type"]."""

    @property
    def _type(self) -> str:
        return self.params.get("type", "hadamard")

    def _refusal(self) -> str | None:
        if self._type not in POKE_TYPES:
            return f"unknown poke type {self._type!r}; known: {', '.join(POKE_TYPES)}."
        if self.pipeline.config.dmhint is None:
            return "no [dmhint] in conf.toml."
        return None

    def forward(self) -> None:
        """Cube numpy (n, y, x) = FITS (NAXIS1, NAXIS2, NAXIS3) = DM (xsize, ysize, n)."""
        from astropy.io import fits
        from pycacao.calib import mkmodes

        mask = self.pipeline.dm_mask()
        if self._type == "zonal":
            cube = mkmodes.make_zonal(mask)
        else:
            cube = mkmodes.make_hadamard(mask)[0]
        hdu = fits.PrimaryHDU(cube.astype(np.float32))
        hdu.header["POKETYPE"] = self._type
        _calib(self, POKES).parent.mkdir(exist_ok=True)
        hdu.writeto(_calib(self, POKES), overwrite=True)

    def success(self) -> bool:
        from astropy.io import fits

        hint = self.pipeline.config.dmhint
        if hint is None or not _calib(self, POKES).is_file():
            return False
        h = fits.getheader(_calib(self, POKES))
        return (h.get("POKETYPE"), h["NAXIS1"], h["NAXIS2"]) == (
            self._type,
            hint.xsize,
            hint.ysize,
        )


class SetLinResp(TaskOnOneCompu):
    """Writes the measlinresp FPS (files, mlat's timing, params["ampl"]) and clears the outputs."""

    def locate_defaults(self) -> None:
        self._locate_exec("cacao-fpsexec-cacaoac-measlinresp")

    def _values(self) -> dict | None:
        """The FPS writes; None without mlat's timing."""
        mlat = self.pipeline.config.fps_name("mlat")
        timing = _mlat.mlat_results(self.pipeline.work_folder / f"fps.{mlat}.datadir")
        if timing is None:
            return None
        values = {
            "inmodes": f"../calib/{POKES}",
            "outmodes": f"../calib/{INPROGRESS}",
            "timing.WFSfrequ": timing[0],
            "timing.hardwlatfr": timing[1],
        }
        if "ampl" in self.params:
            values["ampl"] = self.params["ampl"]
        return values

    def _refusal(self) -> str | None:
        from astropy.io import fits

        if reason := self._undeployed():
            return reason
        if not _calib(self, POKES).is_file():
            return f"{_calib(self, POKES)} does not exist; run `moka mkpokes`."
        if self._values() is None:
            return "no latency from mlat; run `moka mlat`."

        fps: FPS = self.fps  # type: ignore[assignment]
        stream = str(fps["streamin"])
        shape = stream_shape(stream)
        if shape is None:
            return f"stream {stream} does not exist."
        cube = fits.getheader(_calib(self, POKES))
        if (cube["NAXIS1"], cube["NAXIS2"]) != shape:
            return f"{POKES} is {cube['NAXIS1']} x {cube['NAXIS2']}, {stream} is {shape[0]} x {shape[1]}."
        return None

    def forward(self) -> None:
        _calib(self, INPROGRESS).unlink(missing_ok=True)
        if _calib(self, DONE).exists():
            os.replace(_calib(self, DONE), _calib(self, PREV))
        fps: FPS = self.fps  # type: ignore[assignment]
        for key, value in (self._values() or {}).items():
            fps[key] = value

    def success(self) -> bool:
        fps, values = self.fps, self._values()
        if fps is None or values is None or _calib(self, DONE).exists():
            return False
        return not _calib(self, INPROGRESS).exists() and all(
            fps[k] == v if isinstance(v, str) else np.float32(fps[k]) == np.float32(v)
            for k, v in values.items()
        )


class WaitLinResp(TaskOnOneCompu, WaitRunEnd):
    """WaitRunEnd; then a complete run's in-progress file becomes RMmodesWFS.fits."""

    def locate_defaults(self) -> None:
        self._locate_exec("cacao-fpsexec-cacaoac-measlinresp")

    @property
    def TIMEOUT(self) -> float:  # type: ignore[override]
        from astropy.io import fits

        fps: FPS = self.fps  # type: ignore[assignment]
        modes = fits.getheader(_calib(self, POKES))["NAXIS3"]
        frames = 2 * modes * (float(fps["timing.NBave"]) + float(fps["timing.NBexcl"]))
        return (
            2
            * frames
            * float(fps["procinfo.loopcntMax"])
            / float(fps["timing.WFSfrequ"])
            + 10
        )

    def forward(self) -> None:
        fps: FPS = self.fps  # type: ignore[assignment]
        try:  # the procinfo file goes when the run exits: hold it from now
            pinfo = fps.pinfo_struct
        except FileNotFoundError:
            pinfo = None
        super().forward()
        if not self._ended():
            return
        done, total = (pinfo.loopcnt if pinfo else 0), int(fps["procinfo.loopcntMax"])
        if done < total:
            self.failure_reason = f"measlinresp ended after {done} of {total} cycles; partial result in {INPROGRESS}, last good one in {PREV}."
        elif not _calib(self, INPROGRESS).is_file():
            self.failure_reason = f"measlinresp completed but wrote no {INPROGRESS}."
        else:
            os.replace(_calib(self, INPROGRESS), _calib(self, DONE))

    def success(self) -> bool:
        return (
            super().success()
            and _calib(self, DONE).is_file()
            and not _calib(self, INPROGRESS).exists()
        )


for _task in (MakePokes, SetLinResp, WaitLinResp):
    TASKS.register(_task.__name__, _task)
for _plan in (
    Plan(
        "mkpokes",
        (Step("MakePokes"),),
        params=(Param("type", default="hadamard", positional=True),),
    ),
    Plan(
        "measlinresp",
        (
            Step("RunStart", ("acquWFS",)),
            Step("SetLinResp"),
            Step("RunStart", ("measlinresp",)),
            Step("WaitLinResp"),
        ),
        params=(Param("ampl", float, short="a"),),
    ),
):
    PLANS.register(_plan.name, _plan)
