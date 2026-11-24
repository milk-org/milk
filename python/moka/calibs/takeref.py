"""The takeref and takedark plans: the WFS reference averaged from imWFS1, the WFS dark from
wfsim (as cacao-aorun-026-takeref and cacao-aorun-005-takedark).
"""

from __future__ import annotations

import numpy as np

from ..operate.plans import PLANS, Param, Plan, Step
from ..operate.tasks import TASKS, Task, stream_shape

# module, not name: mlat.py imports operate.plans, which imports us
from . import linresp as _lr, mlat as _mlat

DEFAULT_SECONDS = 5.0
FRAME_TIMEOUT = 5.0  # [s] without a new imWFS1 frame, the task fails


class TakeAverage(Task):
    """Writes the float32 average of the SOURCE stream into the TARGETS streams and calib/FITS;
    params nframes or seconds. Stream names are suffixes of aol<N>_.
    """

    SOURCE: str
    TARGETS: tuple[str, ...]
    FITS: str  # in rootdir/calib
    CREATE = False  # a missing target is created, not refused

    _done = False

    def _streams(self) -> list[str]:
        """[source, *targets]."""
        n = self.pipeline.config.loop_number
        return [f"aol{n}_{name}" for name in (self.SOURCE, *self.TARGETS)]

    def _nframes(self) -> int | None:
        """-n, else -t (default 5 s) at mlat's framerate; None without mlat's."""
        if "nframes" in self.params:
            return self.params["nframes"]
        timing = _mlat.timing(self.pipeline)
        if timing is None:
            return None
        return max(1, round(self.params.get("seconds", DEFAULT_SECONDS) * timing[0]))

    def _refusal(self) -> str | None:
        if "nframes" in self.params and "seconds" in self.params:
            return "-n and -t are mutually exclusive."
        if self.params.get("nframes", 1) < 1:
            return f"-n {self.params['nframes']} is below 1."
        if self.params.get("seconds", 1.0) <= 0:
            return f"-t {self.params['seconds']} is not positive."
        if self._nframes() is None:
            return "no framerate from mlat; run `moka mlat` or give -n."
        src, *targets = self._streams()
        shape = stream_shape(src)
        if shape is None:
            return f"stream {src} does not exist."
        for name in targets:
            target = stream_shape(name)
            if target is None and not self.CREATE:
                return f"stream {name} does not exist."
            if target not in (None, shape):
                return f"{src} is {shape}, {name} is {target}."
        return None

    def forward(self) -> None:
        from astropy.io import fits
        from pyMilk.interfacing.shm import SHM

        src, *targets = self._streams()
        n = self._nframes()
        assert n is not None
        with SHM(src, symcode=0) as shm:
            total = np.zeros(shm.shape, np.float64)
            for k in range(n):
                # The first flushes the semaphore: only frames from now on.
                frame = shm.get_data(
                    check=True,
                    checkSemAndFlush=k == 0,
                    timeout=FRAME_TIMEOUT,
                    return_none_on_timeout=True,
                )
                if frame is None:
                    self.failure_reason = (
                        f"no new {src} frame for {FRAME_TIMEOUT} s, after {k} of {n}."
                    )
                    return
                total += frame

        ref = (total / n).astype(np.float32)

        for name in targets:
            if stream_shape(name) is None:
                SHM(name, ref, symcode=0).close()
                continue
            with SHM(name, symcode=0) as out:
                out.set_data(ref)
        # stream (x, y) -> numpy (y, x) = FITS (NAXIS1, NAXIS2) = (x, y)
        hdu = fits.PrimaryHDU(ref.T)
        hdu.header["NFRAMES"] = n
        _lr._calib(self, self.FITS).parent.mkdir(exist_ok=True)
        hdu.writeto(_lr._calib(self, self.FITS), overwrite=True)
        self._done = True

    def success(self) -> bool:
        """Only once forward() averaged: every run takes a new average."""
        return self._done


class TakeRef(TakeAverage):
    """The WFS reference, from aol<N>_imWFS1. acquWFS subtracts wfsrefc into imWFS2."""

    SOURCE = "imWFS1"
    TARGETS = ("wfsref", "wfsrefc")
    FITS = "wfsref.fits"


class TakeDark(TakeAverage):
    """The WFS dark, from aol<N>_wfsim. acquWFS subtracts wfsdark into imWFS0."""

    SOURCE = "wfsim"
    TARGETS = ("wfsdark",)
    FITS = "wfsdark.fits"
    CREATE = True  # acquWFS does not create wfsdark


PARAMS = (
    Param("nframes", int, short="n", help="Average NFRAMES frames."),
    Param(
        "seconds",
        float,
        short="t",
        help="Average SECONDS of frames at mlat's framerate (5 s if neither -n nor -t).",
    ),
)

TASKS.register("TakeRef", TakeRef)
TASKS.register("TakeDark", TakeDark)
PLANS.register(
    "takeref",
    Plan(
        "takeref",
        (Step("RunStart", ("acquWFS",)), Step("TakeRef")),
        params=PARAMS,
        help="Average imWFS1 into the WFS reference wfsref and wfsrefc.",
    ),
)
PLANS.register(
    "takedark",
    Plan(
        "takedark",
        (Step("TakeDark"),),
        params=PARAMS,
        help="Average wfsim into the WFS dark wfsdark.",
    ),
)
