from __future__ import annotations

import typing as typ

"""
Trying a new approach here that respects the CPP code more.
I'll define a type stub, and then refer to the bare nanobind import

Technically, we should carry all the docstrings from the nanobind cpp here, since this
is what static analyzers will parse.

TODO: would be much nice if we had a binding of the enums for CTRLval, etc.
"""

import os, time
from enum import IntEnum

from .shm import MILK_SHM_DIR

from . import glib_loader_fix
from .. import ProcessInfoWrap
from ..ProcessInfoWrap import processinfo as ProcessInfoC

if typ.TYPE_CHECKING:
    import numpy as np


class LoopStatEnum(IntEnum):
    INIT = 0
    ACTIVE = 1
    PAUSED = 2
    STOPPED = 3
    ERROR = 4
    SPINNING = 5
    CRASHED = 6


class CtrlValEnum(IntEnum):
    RUN = 0
    PAUSE = 1
    STEP = 2
    EXIT = 3


class ProcessInfo:

    def __init__(self, fps_name: str | None, pid: int):
        fps_slug = "." + fps_name if fps_name else ""
        file_name = MILK_SHM_DIR() + f"/proc{fps_slug}.{pid:06d}.shm"

        self.name = f"{fps_slug}.{pid:06d}"

        if not os.path.isfile(file_name):
            raise FileNotFoundError(f"ProcessInfo file {file_name} is missing.")

        self._pinfo = ProcessInfoC()
        self._pinfo.link(file_name)

    @property
    def loopstat(self) -> LoopStatEnum:
        return LoopStatEnum(self._pinfo.loopstat)

    @property
    def ctrlval(self) -> CtrlValEnum:
        return CtrlValEnum(self._pinfo.CTRLval)

    @ctrlval.setter
    def ctrlval(self, value: CtrlValEnum):
        # TODO must always check the underlying file still exists !
        # TODO Or sort of an autorelink
        # TODO But then fps should allow a PID agnostic-autorelink
        self._pinfo.CTRLval = value

    def busywait_timings(self) -> tuple[np.ndarray, np.ndarray]:
        while self._pinfo.timerindex != ProcessInfoWrap.PROCESSINFO_NBtimer - 1:
            ...
        return self._pinfo.texecstart, self._pinfo.texecend

    @property
    def loopcnt(self):
        return self._pinfo.loopcnt

    def busywait_until_loopcnt(self, value: int, timeout: float | None = 1.0) -> None:
        t = time.monotonic()
        while self._pinfo.loopcnt <= value:
            if timeout and t + timeout < time.monotonic():
                raise TimeoutError(
                    f"Timeout waiting for procinfo {self.name} count to reach {value}"
                )
