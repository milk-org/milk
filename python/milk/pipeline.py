"""
Goals for today

Deploy 3 pipelines:
1) DMComb
2) DMSim -> linsim -> WFSSim
3) AOloop

# How deploy ?
# fpsexec -> cwd + exec name + name

# trylink compute units
# if exist by name, find pwd
# if not, use a pwd to do the deploy

# Do factory functions as alternate constructors
# factory:: from_scan
# factory:: from_conf_file



# Levels of deployment akin to cacao-loop-deploy
"""

from __future__ import annotations

from .compute_unit import ComputeUnit
from .config import confresolve
from .config.layout import is_rootdir
from .operate.plans import run_plan
from .operate.state import State, require
from .operate.tasks import Task
from . import env
from .errors import (
    ConfigError,
    LinkageError,
    TaskCannotRunError,
    TaskDidNotSucceedError,
)

import os
import shutil
from pathlib import Path


class Pipeline:
    # Or should it be a _deployed pipeline_ ?
    # Or should deployed / deployable / undeployable different things ?

    def __init__(self, target: str | Path) -> None:
        """target: a rootdir path, or a short name whose rootdir is <cwd>/<target>-rootdir."""
        root = Path(target)
        if not is_rootdir(root):
            root = Path.cwd() / f"{target}-rootdir"
            if not is_rootdir(root):
                raise ConfigError(
                    f"Pipeline {target}: neither {Path(target)} nor {root} is a rootdir.",
                    path=root,
                )
        self.root_folder = root.absolute()
        self.config = confresolve(self.root_folder)

        self.long_name = self.config.long_name
        self.dataloads = self.config.dataloads
        self.systemlinkage = self.config.systemlinkage

        self.short_name = self.config.name
        self.loop_number = self.config.loop_number

        self.work_folder = self.root_folder / "work"

    @classmethod
    def from_folder(cls, rootdir: str | Path) -> Pipeline:
        if not is_rootdir(rootdir):
            raise ConfigError(f"{rootdir} is not a rootdir.", path=Path(rootdir))
        return cls(Path(rootdir).absolute())

    @classmethod
    def from_parent(cls, parent: str | Path, short_name: str) -> Pipeline:
        return cls.from_folder(Path(parent) / f"{short_name}-rootdir")

    def task_do(self, task: Task | type[Task]) -> Pipeline:
        if isinstance(task, Task):
            pass
        elif issubclass(task, Task):
            task = task(self)
        else:
            raise ValueError(f"Bad argument: subclass or instance of Task required.")
        if task.success():
            # log unneeded
            return self

        if task.can():
            task.forward()
        else:
            raise TaskCannotRunError(
                f"Task {type(task).__name__} cannot be executed:\n"
                f"    {task.failure_reason}",
                task=type(task).__name__,
                pipeline=self.short_name,
            )

        if not task.success():
            # TODO if partial success is tolerated ?
            raise TaskDidNotSucceedError(
                f"Task {type(task).__name__} did not succeed:\n"
                f"    {task.failure_reason}",
                task=type(task).__name__,
                pipeline=self.short_name,
            )

        return self  # So as to be able to chain

    @property
    def compute_units(self) -> dict[str, ComputeUnit]:
        """Fresh views keyed like config.compute_units. The only ComputeUnit construction in milk/."""
        return {
            cu: ComputeUnit(spec.exec, self.config.fps_name(cu))
            for cu, spec in self.config.compute_units.items()
        }

    def any_run_active(self) -> bool:
        # TODO: check this question has sense: is the pipeline in a correct state?
        return any(
            c.fps is not None and c.fps.run_isrunning()
            for c in self.compute_units.values()
        )

    def _linkage_paths(self, linkage_name: str) -> list[tuple[Path, Path]]:
        if linkage_name not in self.systemlinkage:
            raise LinkageError(
                f"Unknown system linkage {linkage_name!r}", pipeline=self.short_name
            )
        shm = env.shm_dir()
        return [
            (shm / f"{local}.im.shm", shm / f"{remote}.im.shm")
            for local, remote in self.systemlinkage[linkage_name].items()
        ]

    def _linked_paths(self, linkage_name: str) -> list[Path]:
        """The local stream files of the linkage that still point at its remote."""
        return [
            local
            for local, remote in self._linkage_paths(linkage_name)
            if local.is_symlink() and Path(os.readlink(local)) == remote
        ]

    def link(self, name: str) -> None:
        """Point each local stream file of linkage ``name`` at its remote one."""
        st = require(self, "consistent")
        if st.state in (State.RUNNING, State.MIXED):
            raise LinkageError(
                "Pipeline is running; stop it first", pipeline=self.short_name
            )
        for local, remote in self._linkage_paths(name):
            local.unlink(missing_ok=True)
            local.symlink_to(remote)

    def unlink(self, name: str) -> None:
        """Remove the linkage's symlinks, leaving any that point elsewhere."""
        for local in self._linked_paths(name):
            local.unlink()

    def linked(self) -> list[str]:
        """The configured linkages that have a symlink in place."""
        return [name for name in self.systemlinkage if self._linked_paths(name)]

    def nuke_unsafe(self) -> None:
        """run_plan("teardown"), then rm -rf the rootdir even if teardown failed. Never prompts."""
        try:
            run_plan(self, "teardown")
        finally:
            shutil.rmtree(self.root_folder)


class DMCombPipeline: ...


class AOSimPipeline: ...


class AOPipeline:
    # What's an AO pipeline ?
    # Anything that contains an mfilt.
    ...
