#!/usr/bin/env python
"""
shmImshow.py
Visual image for 2D data in shm.

Code extrapolated from LESIA's HRAA group "suricate" project

Usage:
    shmImshow.py <name> [options]

Options:
    <name>        SHM file to link: $MILK_SHM_DIR/<name>.im.shm
    --fr=<val>    Fps requested [default: 10]
    -s=<val>      Data orientation symcode (0-7) [default: 4]
    -z=<val>        Zoom level / bin rate for a half character [default: 4]
"""

import time
import numpy as np

import matplotlib.cm

from pyMilk.interfacing.isio_shmlib import SHM


def print_rgb_stdout(rgb_matrix):

    print("\x1b[37m\x1b[40m")

    for i in range(rgb_matrix.shape[0] // 2):
        for j in range(rgb_matrix.shape[1]):
            # Use background for 2*n + 1 row, foreground for 2*n row
            print(
                f"\x1b[48;2;{rgb_matrix[2*i+1,j,0]};{rgb_matrix[2*i+1,j,1]};{rgb_matrix[2*i+1,j,2]}m"
                f"\x1b[38;2;{rgb_matrix[2*i,j,0]};{rgb_matrix[2*i,j,1]};{rgb_matrix[2*i,j,2]}m\u2580",
                end="",
            )
        # Set background to black, foreground to white, and line feed
        print("\x1b[37m\x1b[40m")

    # Set background to black
    print("\x1b[40m", end="")
    # Stray odd number of lines ?
    if rgb_matrix.shape[0] % 2:
        for j in range(rgb_matrix.shape[1]):
            print(
                f"\x1b[38;2;{rgb_matrix[-1,j,0]};{rgb_matrix[-1,j,1]};{rgb_matrix[-1,j,2]}m\u2580",
                end="",
            )

        # Set background to black, foreground to white, and line feed
        print("\x1b[37m\x1b[40m")


class ShmTermshowClass:
    """
    ShmImshowClass

    Encapsulates widgets
    """

    def __init__(
        self, shmname: str, symcode: int = 0, targetFps: float = 5.0, bin_fac=8
    ) -> None:

        # Init grabber and get a buffer for warm-up
        self.shm = SHM(shmname, symcode=symcode)

        # Will be allocated in self._toggleDarkSub if used
        self.shmDark = None  # type: SHM
        self.symcode = symcode
        self.grabData()  # We need self.data set to initialize the plot

        # Timing
        self.lastTime = time.time()
        self.count = 0
        self.fps = 0.0
        self.targetFps = targetFps

        # pixels to a half block:
        self.bin_fac = bin_fac

        # Attempt to find a dark
        self._toggleDarkSub()

        # MATPLOTLIB
        self.mpl_mappable = matplotlib.cm.ScalarMappable(
            cmap="viridis"
        )  # Add normalizer here?

    def _toggleDarkSub(self) -> None:
        if self.shmDark is not None:
            self._doDarkSub = not self._doDarkSub
            return

        try:
            self.shmDark = SHM(self.shm.FNAME + "_dark")
            self._doDarkSub = True
        except:
            print("Can't find a dark for this shm")
            self.shmDark = None
            self._doDarkSub = False

    def grabData(self) -> None:
        self.data = self.shm.get_data()

    def grabDark(self) -> None:
        self.dark = self.shmDark.get_data()

    def update(self) -> None:
        self.grabData()
        if self._doDarkSub:
            self.grabDark()
            image = self.data - self.dark
        else:
            image = self.data

        m, M = np.percentile(image, [0.1, 99.9])

        # Sizing - could do better than padding zeros...
        if image.shape[0] % self.bin_fac != 0:
            r_add = self.bin_fac - (image.shape[0] % self.bin_fac)
            image = np.r_[image, np.zeros((r_add, image.shape[1]), dtype=image.dtype)]
        if image.shape[1] % self.bin_fac != 0:
            c_add = self.bin_fac - (image.shape[1] % self.bin_fac)
            image = np.c_[image, np.zeros((c_add, image.shape[0]), dtype=image.dtype)]

        # Binning
        image = np.mean(
            image.reshape(
                image.shape[0] // self.bin_fac,
                self.bin_fac,
                image.shape[1] // self.bin_fac,
                self.bin_fac,
            ),
            axis=(1, 3),
        ).astype(np.float32)

        # Color limits / rgb conversion
        self.rgb = self.mpl_mappable.to_rgba(
            (image - m) / (M - m), bytes=True, norm=False
        )

        print_rgb_stdout(self.rgb)

    def run(self) -> None:
        # Clear terminal
        print("\x1bc")
        while True:
            # Move cursor
            print("\x1b[H")
            self.update()
            time.sleep(1.0 / self.targetFps)


def main():
    # Parse
    from docopt import docopt

    doc = docopt(__doc__)
    doc["-s"] = int(doc["-s"])
    doc["--fr"] = float(doc["--fr"])
    doc["-z"] = int(doc["-z"])

    plotter = ShmTermshowClass(
        doc["<name>"], symcode=doc["-s"], targetFps=doc["--fr"], bin_fac=doc["-z"]
    )

    plotter.run()


if __name__ == "__main__":
    main()
