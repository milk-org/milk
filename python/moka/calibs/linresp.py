"""Calibration plans and tasks: mkpokes (the poke cube measlinresp reads)."""

from __future__ import annotations

import numpy as np

from ..operate.plans import PLANS, Param, Plan, Step
from ..operate.tasks import TASKS, Task

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


TASKS.register("MakePokes", MakePokes)
PLANS.register(
    "mkpokes",
    Plan(
        "mkpokes",
        (Step("MakePokes"),),
        params=(Param("type", default="hadamard", positional=True),),
    ),
)
