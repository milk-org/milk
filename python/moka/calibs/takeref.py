"""The takeref plan: the WFS reference, averaged from imWFS1 (as cacao-aorun-026-takeref)."""

from __future__ import annotations

import numpy as np

from ..operate.plans import PLANS, Param, Plan, Step
from ..operate.tasks import TASKS, Task, stream_shape

# module, not name: mlat.py imports operate.plans, which imports us
from . import linresp as _lr, mlat as _mlat

REF = "wfsref.fits"  # in rootdir/calib
DEFAULT_SECONDS = 5.0
FRAME_TIMEOUT = 5.0  # [s] without a new imWFS1 frame, the task fails


class TakeRef(Task):
    """Writes the float32 average of aol<N>_imWFS1 into aol<N>_wfsref, aol<N>_wfsrefc and
    calib/wfsref.fits; params nframes or seconds. acquWFS subtracts wfsrefc into imWFS2.
    """

    _done = False

    def _streams(self) -> tuple[str, str, str]:
        """(source, *targets)."""
        n = self.pipeline.config.loop_number
        return f"aol{n}_imWFS1", f"aol{n}_wfsref", f"aol{n}_wfsrefc"

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
        for name in targets:
            shapes = stream_shape(src), stream_shape(name)
            for stream, shape in zip((src, name), shapes, strict=True):
                if shape is None:
                    return f"stream {stream} does not exist."
            if shapes[0] != shapes[1]:
                return f"{src} is {shapes[0]}, {name} is {shapes[1]}."
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
            with SHM(name, symcode=0) as out:
                out.set_data(ref)
        # stream (x, y) -> numpy (y, x) = FITS (NAXIS1, NAXIS2) = (x, y)
        hdu = fits.PrimaryHDU(ref.T)
        hdu.header["NFRAMES"] = n
        _lr._calib(self, REF).parent.mkdir(exist_ok=True)
        hdu.writeto(_lr._calib(self, REF), overwrite=True)
        self._done = True

    def success(self) -> bool:
        """Only once forward() averaged: every takeref takes a new reference."""
        return self._done


TASKS.register("TakeRef", TakeRef)
PLANS.register(
    "takeref",
    Plan(
        "takeref",
        (Step("RunStart", ("acquWFS",)), Step("TakeRef")),
        params=(
            Param("nframes", int, short="n", help="Average NFRAMES frames."),
            Param(
                "seconds",
                float,
                short="t",
                help="Average SECONDS of frames at mlat's framerate (5 s if neither -n nor -t).",
            ),
        ),
        help="Average imWFS1 into the WFS reference wfsref and wfsrefc.",
    ),
)
