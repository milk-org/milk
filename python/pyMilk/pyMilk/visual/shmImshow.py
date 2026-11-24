#!/usr/bin/env python
"""
shmImshow.py
Visual image for 2D data in shm.

Code extrapolated from LESIA's HRAA group "suricate" project

Usage:
    shmImshow.py <name> [options]

Options:
    <name>        SHM file to link: $MILK_SHM_DIR/<name>.im.shm
    --fr=<val>    Fps requested [default: 8]
    -s=<val>      Data orientation symcode (0-7) [default: 0]
"""

from __future__ import annotations

import typing as typ

import time
import numpy as np

import pyqtgraph as pg
from pyqtgraph import QtGui as QtG, QtCore as QtC, QtWidgets as QtW

import warnings

# pyqtgraph compares its default +-1e307 view limits to float32 ranges (numpy 2 promotion)
warnings.filterwarnings(
    "ignore", "overflow encountered in cast", RuntimeWarning, "pyqtgraph"
)

from pyMilk.interfacing.shm import SHM

try:  # moved depending on Qt version
    QShortcut = QtW.QShortcut
except:
    QShortcut = QtG.QShortcut

SYMCODE_TRANS = [
    QtG.QTransform(0, 1, 1, 0, 0, 0),
    QtG.QTransform(0, 1, -1, 0, 0, 0),
    QtG.QTransform(0, -1, 1, 0, 0, 0),
    QtG.QTransform(0, -1, -1, 0, 0, 0),
    QtG.QTransform(1, 0, 0, 1, 0, 0),
    QtG.QTransform(-1, 0, 0, 1, 0, 0),
    QtG.QTransform(1, 0, 0, -1, 0, 0),
    QtG.QTransform(-1, 0, 0, -1, 0, 0),
]


class ShmImshowClass:
    """
    ShmImshowClass

    Encapsulates widgets
    """

    def __init__(self, shmname: str, symcode: int = 0, targetFps: float = 5.0) -> None:

        # Init grabber and get a buffer for warm-up
        self.shm = SHM(shmname)

        # Will be allocated in self._toggleDarkSub if used
        self.shmDark: SHM | None = None
        self.symcode = symcode
        self.grabData()  # We need self.data set to initialize the plot

        # Timing
        self.lastTime = time.time()
        self.count = 0
        self.fps = 0.0

        # Set the plot area
        self.win = pg.GraphicsLayoutWidget()
        self.win.show()
        self.view = self.win.addViewBox(lockAspect=True, row=0, col=0)
        # self.view.show()
        self.imgItem = pg.ImageItem(border="w")
        self.imgItem.setLookupTable(1)
        self.view.addItem(self.imgItem)
        self.imgItem.setTransform(SYMCODE_TRANS[self.symcode])

        # Add the histogram
        self.hist = pg.HistogramLUTItem(self.imgItem)  # creates histo
        self.win.addItem(self.hist)  # add hist in main window
        self.hist.autoHistogramRange()  # inits levels
        self.hist.gradient.loadPreset("thermal")
        self.hist.setMaximumWidth(200)

        self.hist_autoscale = True

        # TODO
        self.titleStr = f"Display of SHM data: {shmname}"
        self.title = self.win.addLabel(self.titleStr, col=0, row=1, color="w")

        # Bind a couple shortcuts
        self.initShortcuts()

        # Prep the imshow
        # ====================

        # Timing - monitor fps and trigger refresh
        self.timer = QtC.QTimer()
        self.targetFps = targetFps
        self.timer.setInterval(int(1000.0 / self.targetFps))
        self.timer.timeout.connect(self.update)
        self.timer.start()

    def initShortcuts(self) -> None:

        # List of shortcuts in the form (Key, callback)
        shortcut_descr = [
            ("H", self._printHelp),
            ("SPACE", self._togglePause),
            ("P", self._togglePause),
            ("S", self._toggleSymcode),
            ("L", self._swapLogZ),
            ("X", self._quit),
            ("[", self._slower),
            ("]", self._faster),
            ("D", self._toggleDarkSub),
            ("A", self._toggle_hist_autoscale),
        ]

        # Extra internals for tracking states related to the shortcuts
        self._pause = False
        self._logZ = False
        self._doDarkSub = False

        # Create and bind the shorcuts
        for key, func in shortcut_descr:
            tmp = QShortcut(QtG.QKeySequence(key), self.win)
            tmp.activated.connect(func)

    def _printHelp(self) -> None:
        print("------------------------")
        print("This should be the help.")
        print(" Will do later. Maybe ! ")
        print("------------------------")
        print(
            "\n".join(
                [
                    "H:    Display this help",
                    "X:                 Quit",
                    "           DATA        ",
                    "D: Toggle dark subtract",
                    "",
                    "         DISPLAY       ",
                    "L:     Toggle log scale",
                    "S:   Toggle orientation",
                    "P or SPACE:       Pause",
                    "[:             -20% FPS",
                    "]:             +20% FPS",
                ]
            )
        )

    def _quit(self) -> None:
        self.timer.stop()
        self.win.close()

    def _toggleSymcode(self) -> None:
        self.symcode = (self.symcode + 1) % 8
        self.imgItem.setTransform(SYMCODE_TRANS[self.symcode])

    def _swapLogZ(self) -> None:
        self._logZ = not self._logZ

    def _togglePause(self) -> None:
        if self._pause:
            self.timer.start()
            self._pause = False
        else:
            self.timer.stop()
            self._pause = True

    def _toggleDarkSub(self) -> None:
        if self.shmDark is not None:
            self._doDarkSub = not self._doDarkSub
            return

        try:
            self.shmDark = SHM(self.shm.FNAME + "_dark")
            self._doDarkSub = True
        except:
            print("Can't find a dark for this shm - creating one")
            self.shmDark = SHM(
                self.shm.FNAME + "_dark", np.zeros_like(self.data, dtype=np.float32)
            )
            self._doDarkSub = False

    def _slower(self) -> None:
        self.targetFps = max(0.8 * self.targetFps, 0.1)
        self.timer.setInterval(round(1000.0 / self.targetFps))

    def _faster(self) -> None:
        self.targetFps = min(1.2 * self.targetFps, 50.0)
        self.timer.setInterval(round(1000.0 / self.targetFps))

    def grabData(self) -> None:
        self.data = self.shm.get_data()

    def grabDark(self) -> None:
        assert self.shmDark is not None
        self.dark = self.shmDark.get_data()

    def _toggle_hist_autoscale(self):
        self.hist_autoscale = not self.hist_autoscale
        if not self.hist_autoscale:
            self.hist.setHistogramRange(*self.hist.getLevels())  # type: ignore

    def update(self) -> None:
        self.grabData()
        if self._doDarkSub:
            self.grabDark()
            image = self.data - self.dark  # Wait this is super wrong in log scale !
        else:
            image = self.data
        # image[0, :] = np.median(image[1:,0])

        if hasattr(self, "_logZ") and self._logZ:
            image = np.log10(np.clip(image, 1.0, None))

        if self.hist_autoscale:
            self.imgItem.setImage(image)
            self.hist.setLevels(image[1:].min(), image[1:].max())
        else:
            self.imgItem.setImage(image, levels=self.hist.getLevels())

        self.updateTimerAndCounter()

    def updateTimerAndCounter(self) -> None:
        now = time.time()
        dt = now - self.lastTime
        self.lastTime = now
        if self.count <= 2:
            self.fps = 1.0 / dt
        else:
            smooth = np.clip(dt, 0.0, 1.0)
            self.fps = self.fps * (1 - smooth) + 1 / dt * smooth

        self.count += 1
        self.title.setText(self.titleStr + f" - {self.fps:.2f} FPS")


def main():
    # Parse
    from docopt import docopt

    doc = docopt(__doc__)
    doc["-s"] = int(doc["-s"])
    doc["--fr"] = float(doc["--fr"])

    # Launch app and let the class spawn the widget
    # I'm doing it this way hoping to be able to reuse this for multi-display windows
    app = QtW.QApplication([doc["<name>"] + ".im.shm"])
    app.setQuitOnLastWindowClosed(True)
    plotter = ShmImshowClass(doc["<name>"], symcode=doc["-s"], targetFps=doc["--fr"])
    app.exec_()


if __name__ == "__main__":
    main()
