"""Calibration plans and tasks: mkpokes (the poke cube) and measlinresp (the response to it)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .. import env
from ..operate.plans import PLANS, Param, Plan, Step
from ..operate.tasks import TASKS, Task, WaitRunEnd
from . import (
    mlat as _mlat,
)  # module, not name: mlat.py imports operate.plans, which imports us

# In rootdir/calib. The FPS gets them relative to the run's cwd, rootdir/work (63 chars max).
POKES, DONE = "RMmodesDM.fits", "RMmodesWFS.fits"
INPROGRESS, PREV = "RMmodesWFS.inprogress.fits", "RMmodesWFS.prev.fits"


POKE_TYPES = ("hadamard", "zonal")


class MakePokes(Task):
    """Writes calib/RMmodesDM.fits over the [dmhint] mask; params["type"]."""

    @property
    def _type(self) -> str:
        return self.params.get("type", "hadamard")

    @property
    def _path(self):
        return self.pipeline.root_folder / "calib" / "RMmodesDM.fits"

    def can(self) -> bool:
        if self._type not in POKE_TYPES:
            self.failure_reason = (
                f"unknown poke type {self._type!r}; known: {', '.join(POKE_TYPES)}."
            )
            return False
        if self.pipeline.config.dmhint is None:
            self.failure_reason = "no [dmhint] in conf.toml."
            return False
        return True

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
        self._path.parent.mkdir(exist_ok=True)
        hdu.writeto(self._path, overwrite=True)

    def success(self) -> bool:
        from astropy.io import fits

        hint = self.pipeline.config.dmhint
        if hint is None or not self._path.is_file():
            return False
        h = fits.getheader(self._path)
        return (h.get("POKETYPE"), h["NAXIS1"], h["NAXIS2"]) == (
            self._type,
            hint.xsize,
            hint.ysize,
        )


class SetLinResp(Task):
    """Writes the measlinresp FPS (files, mlat's timing, params["ampl"]) and clears the outputs."""

    def _calib(self, name: str) -> Path:
        return self.pipeline.root_folder / "calib" / name

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

    def can(self) -> bool:
        from astropy.io import fits
        from pyMilk.interfacing.shm import SHM

        compu = next(iter(self.compute_units.values()))
        if compu.fps is None or not compu.fps.conf_isrunning():
            self.failure_reason = f"{compu.fpsname} is missing or its conf process is not running; run deploy."
            return False
        if not self._calib(POKES).is_file():
            self.failure_reason = (
                f"{self._calib(POKES)} does not exist; run `moka mkpokes`."
            )
            return False
        if self._values() is None:
            self.failure_reason = "no latency from mlat; run `moka mlat`."
            return False
        stream = compu.fps["streamin"]
        if not (env.shm_dir() / f"{stream}.im.shm").is_file():
            self.failure_reason = f"stream {stream} does not exist."
            return False
        shm = SHM(stream, symcode=0)  # numpy (y, x), as LoadData
        shape = tuple(shm.shape)
        shm.close()
        cube = fits.getheader(self._calib(POKES))
        if (cube["NAXIS2"], cube["NAXIS1"]) != shape:
            self.failure_reason = f"{POKES} is {cube['NAXIS1']} x {cube['NAXIS2']}, {stream} is {shape[1]} x {shape[0]}."
            return False
        return True

    def forward(self) -> None:
        self._calib(INPROGRESS).unlink(missing_ok=True)
        if self._calib(DONE).exists():
            os.replace(self._calib(DONE), self._calib(PREV))
        fps = next(iter(self.compute_units.values())).fps
        for key, value in (self._values() or {}).items():
            fps[key] = value

    def success(self) -> bool:
        fps, values = next(iter(self.compute_units.values())).fps, self._values()
        if fps is None or values is None or self._calib(DONE).exists():
            return False
        return not self._calib(INPROGRESS).exists() and all(
            fps[k] == v if isinstance(v, str) else np.float32(fps[k]) == np.float32(v)
            for k, v in values.items()
        )


class WaitLinResp(WaitRunEnd):
    """WaitRunEnd; then a complete run's in-progress file becomes RMmodesWFS.fits."""

    @property
    def _calib(self) -> Path:
        return self.pipeline.root_folder / "calib"

    @property
    def TIMEOUT(self) -> float:  # type: ignore[override]
        from astropy.io import fits

        fps = next(iter(self.compute_units.values())).fps
        modes = fits.getheader(self._calib / POKES)["NAXIS3"]
        frames = 2 * modes * (fps["timing.NBave"] + fps["timing.NBexcl"])
        return 2 * frames * fps["procinfo.loopcntMax"] / fps["timing.WFSfrequ"] + 10

    def forward(self) -> None:
        fps = next(iter(self.compute_units.values())).fps
        try:  # the procinfo file goes when the run exits: hold it from now
            pinfo = fps.pinfo_struct
        except FileNotFoundError:
            pinfo = None
        super().forward()
        if not self._ended():
            return
        done, total = (pinfo.loopcnt if pinfo else 0), fps["procinfo.loopcntMax"]
        if done < total:
            self.failure_reason = f"measlinresp ended after {done} of {total} cycles; partial result in {INPROGRESS}, last good one in {PREV}."
        elif not (self._calib / INPROGRESS).is_file():
            self.failure_reason = f"measlinresp completed but wrote no {INPROGRESS}."
        else:
            os.replace(self._calib / INPROGRESS, self._calib / DONE)

    def success(self) -> bool:
        return (
            super().success()
            and (self._calib / DONE).is_file()
            and not (self._calib / INPROGRESS).exists()
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
            Step("SetLinResp", ("measlinresp",)),
            Step("RunStart", ("measlinresp",)),
            Step("WaitLinResp", ("measlinresp",)),
        ),
        params=(Param("ampl", float, short="a"),),
    ),
):
    PLANS.register(_plan.name, _plan)
