"""The mkcm plan: the control matrix pair, from the response matrix pair in calib/."""

from __future__ import annotations

import numpy as np

from ..operate.plans import PLANS, Param, Plan, Step
from ..operate.tasks import TASKS, Task

# module, not names: linresp may be mid-import (cycle via plans)
from . import linresp as _lr

CM_FLAVORS = ("tsvd", "tikhonov", "pinv")
DEFAULT_RCOND = 1e-6
CM_WFS, CM_DM = "CMmodesWFS.fits", "CMmodesDM.fits"


class MakeCM(Task):
    """Writes calib/CMmodes{WFS,DM}.fits from calib/RMmodes{DM,WFS}.fits; params flavor, nmodes, reg."""

    @property
    def _args(self) -> tuple[str, int | None, float]:
        p = self.params
        return p.get("flavor", "tsvd"), p.get("nmodes"), p.get("reg", DEFAULT_RCOND)

    def _refusal(self) -> str | None:
        from astropy.io import fits

        flavor, nmodes, reg = self._args
        if flavor not in CM_FLAVORS:
            return f"unknown CM flavor {flavor!r}; known: {', '.join(CM_FLAVORS)}."
        for name, command in ((_lr.POKES, "mkpokes"), (_lr.DONE, "measlinresp")):
            if not _lr._calib(self, name).is_file():
                return f"{_lr._calib(self, name)} does not exist; run `moka {command}`."
        n, n_wfs = (
            fits.getheader(_lr._calib(self, f))["NAXIS3"] for f in (_lr.POKES, _lr.DONE)
        )
        if n != n_wfs:
            return f"{_lr.POKES} has {n} modes, {_lr.DONE} has {n_wfs}."
        if flavor == "tikhonov" and "reg" not in self.params:
            return "tikhonov needs -r REG."
        if reg < 0:
            return f"-r {reg} is negative."
        if nmodes is not None and not 1 <= nmodes <= n:
            return f"-n {nmodes} is not in 1..{n}."
        return None

    def forward(self) -> None:
        from astropy.io import fits
        from pycacao.calib import mkcm

        flavor, nmodes, reg = self._args
        rm = [fits.getdata(_lr._calib(self, f)) for f in (_lr.POKES, _lr.DONE)]
        try:
            if flavor == "tikhonov":
                cm_wfs, cm_dm, sv = mkcm.cm_tikhonov(*rm, reg, nmodes)
            else:
                cm_wfs, cm_dm, sv = getattr(mkcm, f"cm_{flavor}")(*rm, nmodes, reg)
        except ValueError as error:
            self.failure_reason = f"{error} with -n {nmodes} -r {reg}."
            return
        kept = (
            len(cm_wfs) if flavor != "pinv" else int((sv[:nmodes] > reg * sv[0]).sum())
        )
        header = fits.Header({"CMFLAVOR": flavor, "CMREG": reg, "CMNKEPT": kept})
        if nmodes is not None:
            header["CMNCUT"] = nmodes
        primary = fits.PrimaryHDU(cm_wfs.astype(np.float32), header.copy())
        fits.HDUList([primary, fits.ImageHDU(sv, name="SINGVALS")]).writeto(
            _lr._calib(self, CM_WFS), overwrite=True
        )
        fits.PrimaryHDU(cm_dm.astype(np.float32), header).writeto(
            _lr._calib(self, CM_DM), overwrite=True
        )

    def success(self) -> bool:
        """Both files, of these parameters, none older than an RM file."""
        from astropy.io import fits

        flavor, nmodes, reg = self._args
        cms, rms = (
            [_lr._calib(self, f) for f in pair]
            for pair in ((CM_WFS, CM_DM), (_lr.POKES, _lr.DONE))
        )
        if not all(p.is_file() for p in cms + rms):
            return False
        mtime = lambda p: p.stat().st_mtime_ns  # noqa: E731
        if min(map(mtime, cms)) < max(map(mtime, rms)):
            return False
        return all(
            (h.get("CMFLAVOR"), h.get("CMNCUT")) == (flavor, nmodes)
            and np.isclose(h.get("CMREG", np.nan), reg, rtol=1e-6, atol=0)
            for h in map(fits.getheader, cms)
        )


TASKS.register("MakeCM", MakeCM)
PLANS.register(
    "mkcm",
    Plan(
        "mkcm",
        (Step("MakeCM"),),
        accepts="cold_or_consistent",  # brief §1: mkcm works on a pipeline not deployed
        params=(
            Param("flavor", default="tsvd", positional=True, choices=CM_FLAVORS),
            Param("nmodes", int, short="n", help="Keep at most NMODES modes."),
            Param(
                "reg",
                float,
                short="r",
                help="Truncation or regularisation threshold (1e-6 if not given; required for tikhonov).",
            ),
        ),
        help="Compute the control matrix pair from the response matrix pair.",
    ),
)
