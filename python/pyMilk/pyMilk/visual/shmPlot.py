#!/usr/bin/env python
"""
shmPlot.py
Visual plotter for 1D or 2D data in shm.
Plots the rows of the data vs. the column index

Usage:
    shmPlot.py <name> [options]

Options:
    <name>        SHM file to link: $MILK_SHM_DIR/<name>.im.shm
    --fr=<val>    Fps requested [default: 10]
    -t            Transpose data
"""

import time
import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets

from pyMilk.interfacing.isio_shmlib import SHM

PENS = [
    pg.mkPen(c, style=st)
    for st in (QtCore.Qt.SolidLine, QtCore.Qt.DashLine, QtCore.Qt.DashDotLine)
    for c in "rmygcb"
]


class ShmPlotClass:
    """
    ShmPlotClass
    """

    def __init__(
        self, shmname: str, transpose: bool = False, targetFps: float = 5.0
    ) -> None:

        # Init grabber and get a buffer for warm-up
        self.shm = SHM(shmname)

        # Will be allocated in self._toggleDarkSub if used
        self.shmDark = None  # type: SHM
        self.transpose = transpose
        self.grabData()  # We need self.data set to initialize the plot

        # Timing
        self.lastTime = time.time()
        self.count = 0
        self.fps = 0.0

        # Set the plot area
        self.plot = pg.PlotWidget()
        self.plot.show()

        self.title = f"{shmname} - press [h] for help"
        self.plot.setTitle(self.title)
        self.plot.setLabel("bottom", "Column index", units="")
        self.plot.setLabel("left", "Amplitude", units=f"{self.data.dtype}")
        self.plotItem = self.plot.getPlotItem()  # For further use

        # Bind a couple shortcuts
        self.initShortcuts()

        # Prep the curves
        self.curves = [
            self.plot.plot(pen=PENS[i % len(PENS)]) for i in range(len(self.data))
        ]

        # Timing - monitor fps and trigger refresh
        self.timer = QtCore.QTimer()
        self.targetFps = targetFps
        self.timer.setInterval(int(1000.0 / self.targetFps))
        self.timer.timeout.connect(self.update)
        self.timer.start()

    def initShortcuts(self) -> None:
        QW, QG = QtWidgets, QtGui

        # List of shortcuts in the form (Key, callback)
        shortcut_descr = [
            ("H", self._printHelp),
            ("G", self._swapGrid),
            ("SPACE", self._togglePause),
            ("P", self._togglePause),
            ("K", self._swapLogX),
            ("L", self._swapLogY),
            ("X", self._quit),
            ("[", self._slower),
            ("]", self._faster),
            ("D", self._toggleDarkSub),
        ]

        # Extra internals for tracking states related to the shortcuts
        self._gridState = 0
        self._pause = False
        self._logx = False
        self._logy = False
        self._doDarkSub = False

        # Create and bind the shorcuts
        for key, func in shortcut_descr:
            tmp = QW.QShortcut(QG.QKeySequence(key), self.plot)
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
                    "G:          Toggle grid",
                    "K:         Toggle log x",
                    "L:         Toggle log y",
                    "P or SPACE:       Pause",
                    "[:             -20% FPS",
                    "]:             +20% FPS",
                ]
            )
        )

    def _quit(self) -> None:
        self.timer.stop()
        self.plot.close()

    def _swapGrid(self) -> None:
        self._gridState = (self._gridState + 1) % 4

        self.plotItem.showGrid(
            *[(False, False), (False, True), (True, True), (True, False)][
                self._gridState
            ]
        )

    def _swapLogX(self) -> None:
        self._logx = not self._logx
        self.plotItem.setLogMode(x=self._logx, y=self._logy)

    def _swapLogY(self) -> None:
        self._logy = not self._logy
        self.plotItem.setLogMode(x=self._logx, y=self._logy)

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
            print("Can't find a dark for this shm")
            self.shmDark = None
            self._doDarkSub = False

    def _slower(self) -> None:
        self.targetFps = max(0.8 * self.targetFps, 0.1)
        self.timer.setInterval(1000.0 / self.targetFps)

    def _faster(self) -> None:
        self.targetFps = min(1.2 * self.targetFps, 50.0)
        self.timer.setInterval(1000.0 / self.targetFps)

    def grabData(self) -> None:
        if not self.transpose:
            self.data = self.shm.get_data()
        else:
            self.data = self.shm.get_data().T
        if len(self.data.shape) == 1:  # Pad a dimension
            self.data = self.data[None, :]
        if self.data.shape[1] == 1:  # It went padded the wrong way already
            self.data = self.data.T

    def grabDark(self) -> None:
        # More tricky - darks don't change often - maybe we can save bandpass ?
        if not self.transpose:
            self.dark = self.shmDark.get_data()
        else:
            self.dark = self.shmDark.get_data().T
        if len(self.dark.shape) == 1:  # Pad a dimension
            self.dark = self.dark[None, :]
        if self.dark.shape[1] == 1:  # It went padded the wrong way already
            self.dark = self.dark.T

    def update(self) -> None:
        self.grabData()
        if self._doDarkSub:
            self.grabDark()
            [c.setData(self.data[i] - self.dark[i]) for i, c in enumerate(self.curves)]
        else:
            [c.setData(self.data[i]) for i, c in enumerate(self.curves)]

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
        self.plot.setTitle(self.title + f" - {self.fps:.2f} FPS")


def main():
    # Parse
    from docopt import docopt

    doc = docopt(__doc__)
    doc["--fr"] = float(doc["--fr"])

    # Launch app and let the class spawn the widget
    # I'm doing it this way hoping to be able to reuse this for multi-display windows
    app = QtWidgets.QApplication([doc["<name>"] + ".im.shm"])
    app.setQuitOnLastWindowClosed(True)
    plotter = ShmPlotClass(doc["<name>"], transpose=doc["-t"], targetFps=doc["--fr"])
    app.exec_()


if __name__ == "__main__":
    main()
