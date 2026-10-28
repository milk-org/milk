"""
A task manager can run tasks, display pretty stuff, check if tasks have failed or not, etc.

. What's a task ?
- Something that admits a do, a success, a clean undo, a best-effort undo.
- A task could depend on another task. But what of skips, what of forces?

Something like -f: skip dependencies
               -F: don't try-unclean on error

A task should mostly be zero-status, so static function collections ain't stupid here.
That + a protocol.

A pipeline has tasks, so a pipeline should be able to
- collect its own tasks
- execute them in sequence. Basically that's the task manager.

There are two pipelines:
- PutativePipeline; it has PTasks
- ExistingPipelne; it has ETasks

NO.

A pipeline is entirely defined by its configuration folder.
That's it.
"""

from __future__ import annotations
import typing as typ

import os, contextlib
from pathlib import Path

from .task_models import SimpleTask
from ..errors import TaskDidNotSucceedError

from ..config.substitute import denest_toml_dicts
from .. import env

import subprocess


class TestConfig(SimpleTask):
    """
    maps to former cacao-setup -t
    """

    ...


class LoadDataFiles(SimpleTask):
    """
    Loads every [dataloads] entry source file (from rootdir/initdata)
    into target SHM streams.
    """

    def can(self) -> bool:
        p = self.pipeline
        if not os.path.isdir(p.root_folder):
            self.failure_reason = f"{p.root_folder} does not exist."
            return False
        for entry in p.dataloads:
            source_path = p.root_folder / "initdata" / entry.source
            if not os.path.isfile(source_path):
                self.failure_reason = f"{source_path} does not exist."
                return False
        return True

    def success(self) -> bool:
        p = self.pipeline
        if not p.dataloads:
            return True

        # TODO potential insert of dataloads abstraction here.
        from astropy.io import fits
        from pyMilk.interfacing.shm import SHM

        for entry in p.dataloads:
            if not os.path.isfile(f"{env.shm_dir()}/{entry.target}.im.shm"):
                return False
            expected = fits.getdata(str(p.root_folder / "initdata" / entry.source))
            shm = SHM(entry.target, symcode=0)  # same axis order as forward()
            try:
                # Ignore byte order: FITS is big-endian.
                if (
                    shm.shape != expected.shape
                    or shm.nptype != expected.dtype.newbyteorder("=")
                ):
                    return False
            finally:
                shm.close()
        return True

    def forward(self) -> None:
        p = self.pipeline
        if len(p.dataloads) > 0:
            from astropy.io import fits
            from pyMilk.interfacing.shm import SHM

        for entry in p.dataloads:
            source_path = p.root_folder / "initdata" / entry.source
            # TODO milk-FITS2shm should absolutely not be a fpsexec lol
            # TODO TODO write docs on the capabilities of milk-pipes and the default pipelines (aosim, dmcomb, aoloop)
            import numpy as np

            s = SHM(entry.target, fits.getdata(str(source_path)), symcode=0)
            s.close()


class DeployFPS(SimpleTask):

    def can(self) -> bool:
        p = self.pipeline
        if not os.path.isdir(p.root_folder):
            self.failure_reason = f"{p.root_folder} does not exist."
            return False
        if not os.path.isdir(p.work_folder):
            self.failure_reason = f"{p.work_folder} does not exist."
            return False
        return True

    def success(self) -> bool:
        p = self.pipeline
        return all(
            c.fps is not None and c.fps.is_valid() for c in p.compute_units.values()
        )

    def forward(self) -> None:

        p = self.pipeline

        # Probably should ensure that cwd is the config's ROOTDIR

        for cu_name, compu in p.compute_units.items():
            spec = p.config.compute_units[cu_name]

            with contextlib.chdir(p.work_folder):
                compu.fpsinit()

            assert compu.fps

            # TODO probably a method on the CompU should provide the denested iterator.
            denested_fps_config = denest_toml_dicts(spec.fps)

            for fp_name, fp_value in denested_fps_config.items():
                compu.fps[fp_name] = fp_value  # type: ignore[arg-type]


class StartConfProcesses(SimpleTask):
    TIMEOUT_ALL = 5.0

    # TODO if the conf processes exist and hog the tmux, we won't have a restart.

    def can(self) -> bool:
        p = self.pipeline

        # Folder consistency: rootdir/rundir must have been laid out already
        if not os.path.isdir(p.root_folder):
            self.failure_reason = f"{p.root_folder} does not exist."
            return False
        if not os.path.isdir(p.work_folder):
            self.failure_reason = f"{p.work_folder} does not exist."
            return False

        # Every compute unit's FPS must exist and be a well-formed deployment
        for compu in p.compute_units.values():
            if compu.fps is None:
                self.failure_reason = f"FPS for {compu.fpsname} not found."
                return False
            if not compu.fps.is_valid():
                self.failure_reason = f"FPS for {compu.fpsname} is invalid."
                return False

        # TODO get & check the rootdir for all FPSs

        return True

    def success(self) -> bool:
        compus = self.pipeline.compute_units.values()
        return all(c.fps is not None and c.fps.conf_isrunning() for c in compus)

    def forward(self) -> None:
        p = self.pipeline

        # Probably should ensure that cwd is the config's ROOTDIR
        # The below could also probably be handled by a FPSManager of pyMilk
        compus = p.compute_units.values()
        for compu in compus:
            # TODO breaks if tmux is busy already (in particular poor cleaning of repeated tests.)
            compu.confstart(tmux=True)  # TODO dispatch env to tmuxes !!

        import time

        start = time.monotonic()
        while time.monotonic() - start < self.TIMEOUT_ALL:
            if all(c.fps.conf_isrunning() for c in compus):  # type: ignore
                return
            time.sleep(0.01)
        stuck = [c.fpsname for c in compus if not c.fps.conf_isrunning()]  # type: ignore
        raise TaskDidNotSucceedError(
            f"Conf processes not running for {', '.join(stuck)} (tmux session busy?)",
            task=type(self).__name__,
            pipeline=p.short_name,
        )


class CompoundTask(SimpleTask):
    ...

    # TODO what about a compound task ?
    # TODO which tasks are reversible ?
    # TODO what's next? Naming of DM based tasks, and then getting into the aorun suite.
