#!/usr/bin/env python
"""Posts the PSF of the DM phase, over the hinted DM mask, to the stream aol<loop_number>_psf.

Not a task nor a plan: run it by hand from the rootdir, `python conf/scripts/dm_psf.py`.
"""

import time
from pathlib import Path

import numpy as np

from moka.operate.pipeline import Pipeline
from pyMilk.interfacing.shm import SHM

DM_NAME = "dm07disp"  # the simulated DM, of the mlatsim pipeline
FFT_SIZE = (32, 32)
PERIOD = 0.1  # [s]

ROOTDIR = Path(__file__).resolve().parents[2]  # this file: rootdir/conf/scripts/
PIPELINE = Pipeline(ROOTDIR)
MASK = PIPELINE.dm_mask()


def psf(dm_map: np.ndarray, wavelength_um: float = 1.0) -> np.ndarray:
    """dm_map is taken in um"""
    field = np.exp(1j * 2 * np.pi * dm_map / wavelength_um) * MASK
    return (np.abs(np.fft.fftshift(np.fft.fft2(field, FFT_SIZE))) ** 2).astype(
        np.float32
    )


def main() -> None:
    loop_number = PIPELINE.config.loop_number
    dm = SHM(DM_NAME)
    out = SHM(f"aol{loop_number}_psf", psf(dm.get_data(True)))
    while True:
        time.sleep(PERIOD)
        out.set_data(psf(dm.get_data(True)))


if __name__ == "__main__":
    main()
