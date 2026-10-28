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
import glob
import shutil
from pathlib import Path

from .task_models import SimpleTask
from ..errors import TaskDidNotSucceedError

from .toml_manipulation import denest_toml_dicts
from .. import env

import subprocess


# TODO note for future: a good task is idempotent -- would alleviate calls to can/success.
# TODO note for future: do we keep
class InitialFolderSetup(SimpleTask):

    def can(self) -> bool:
        # task_do has already found success False, so an existing rootdir is partial.
        if self.pipeline.root_folder.exists():
            self.failure_reason = (
                f"partial rootdir {self.pipeline.root_folder}; remove it"
            )
            return False
        return True

    def success(self) -> bool:
        root_dir = self.pipeline.root_folder
        if not os.path.isdir(root_dir) or not os.path.isfile(root_dir / "conf.toml"):
            return False

        # More ?

        return True

    def forward(self):
        pp = self.pipeline

        loop_name = pp.short_name
        conf_dir = pp.conf_folder
        root_dir = pp.root_folder

        root_dir.mkdir(parents=True, exist_ok=False)

        # ---- logging ----
        (root_dir / "fpsCTRL.log").unlink(missing_ok=True)
        (root_dir / "fpsCTRL.log").symlink_to(
            env.shm_dir() / f"fpsCTRL-{loop_name}.log"
        )

        pp.log_folder.mkdir(parents=True, exist_ok=True)
        (root_dir / "logdir").unlink(missing_ok=True)
        (root_dir / "logdir").symlink_to(pp.log_folder)

        pp.work_folder.mkdir(parents=True, exist_ok=True)

        fpstmuxenv = conf_dir / "fpstmuxenv"
        if fpstmuxenv.exists():
            shutil.copy(fpstmuxenv, root_dir)

        data_dir = conf_dir / "data"
        if data_dir.exists():
            shutil.copytree(data_dir, root_dir / "data", dirs_exist_ok=True)

        shutil.copy(conf_dir / "conf.toml", root_dir / f"conf.toml")

        for fname in glob.glob(str(conf_dir / "aorun-*")):
            shutil.copy(fname, root_dir)

        scripts_dir = root_dir / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        for fname in glob.glob(str(conf_dir / "scripts" / "*")):
            shutil.copy(fname, scripts_dir)


class TestConfig(SimpleTask):
    """
    maps to former cacao-setup -t
    """

    ...


class LoadDataFiles(SimpleTask):
    """
    Loads every [dataloads] entry source file (from the deployed rootdir/data)
    into target SHM streams.
    """

    def can(self) -> bool:
        p = self.pipeline
        if not os.path.isdir(p.root_folder):
            self.failure_reason = f"{p.root_folder} does not exist."
            return False
        for entry in p.dataloads:
            source_path = p.root_folder / "data" / entry.source
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
            expected = fits.getdata(str(p.root_folder / "data" / entry.source))
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
            source_path = p.root_folder / "data" / entry.source
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
