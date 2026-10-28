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
from .infra.task_models import SimpleTask
from . import env
from .errors import (
    BestEffortError,
    ConfigError,
    LinkageError,
    TaskCannotRunError,
    TaskDidNotSucceedError,
)

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

_LINKAGE_TAG = "systemlinkage.tag"


class Pipeline:
    # Or should it be a _deployed pipeline_ ?
    # Or should deployed / deployable / undeployable different things ?

    def __init__(self, parent_folder: str | Path, long_name: str = "") -> None:

        self.parent_folder = Path(parent_folder).absolute()

        self.long_name = long_name

        self.conf_folder = self.parent_folder / f"{self.long_name}-conf"
        if not os.path.isdir(self.conf_folder):
            raise ConfigError(
                f"Pipeline from {self.conf_folder}: folder does not exist.",
                path=self.conf_folder,
            )
        self.config = confresolve(self.conf_folder)  # Raises if not existing

        self.dataloads = self.config.dataloads
        self.systemlinkage = self.config.systemlinkage

        self.short_name = self.config.name
        self.loop_number = self.config.loop_number

        self.root_folder = self.parent_folder / f"{self.short_name}-rootdir"
        self.log_folder = self.parent_folder / f"{self.short_name}-logdir"

        self.work_folder = self.root_folder / "work"

    def task_do(self, task: SimpleTask | type[SimpleTask]) -> Pipeline:
        if isinstance(task, SimpleTask):
            pass
        elif issubclass(task, SimpleTask):
            task = task(self)
        else:
            raise ValueError(
                f"Bad argument: subclass or instance of SimpleTask required."
            )
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
                f"Task {task} did not complete to success",
                task=type(task).__name__,
                pipeline=self.short_name,
            )

        return self  # So as to be able to chain

    def task_undo(self): ...

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

    def apply_system_linkage(self, linkage_name: str) -> None:
        """Point each local stream file at its remote one, replacing existing symlinks."""
        pairs = self._linkage_paths(linkage_name)
        if self.any_run_active():
            raise LinkageError(
                "Pipeline is running; stop it first", pipeline=self.short_name
            )
        # TODO assert no other incompatible linkage is configured

        # Check everything first so a failure leaves nothing half-applied.
        for local, remote in pairs:
            # TODO: make these warnings, or have a strict mode.
            # if not remote.exists():
            #     raise LinkageError(f"Remote stream missing: {remote}")
            # if local.exists() and not local.is_symlink():
            #     raise LinkageError(
            #         f"Refusing to replace real file: {local}"
            #     )
            pass

        for local, remote in pairs:
            local.unlink(missing_ok=True)
            local.symlink_to(remote)

        stamp = datetime.now(timezone.utc).isoformat()
        (self.root_folder / _LINKAGE_TAG).write_text(f"{linkage_name}\n{stamp}\n")

    def deapply_system_linkage(self, linkage_name: str) -> None:
        """Remove the linkage's symlinks, leaving any that point elsewhere."""
        pairs = self._linkage_paths(linkage_name)
        if self.any_run_active():
            raise LinkageError(
                "Pipeline is running; stop it first", pipeline=self.short_name
            )
        for local, remote in pairs:
            if local.is_symlink() and Path(os.readlink(local)) == remote:
                local.unlink()

    def nuke(self) -> None:
        """Unconditionally tear down: runstop, deapply linkages, confstop, rm -rf root.

        Meant for test machines. Every step is best-effort: a failure does not stop the
        next step, and all failures are raised together as a BestEffortError at the end.
        """
        failures: list[tuple[str, Exception]] = []

        def attempt(label: str, fn) -> None:
            try:
                fn()
            except Exception as error:
                failures.append((label, error))

        compus = self.compute_units
        for cu, compu in compus.items():
            if compu.fps is not None:
                attempt(f"runstop {cu}", compu.runstop)
        for linkage in self.systemlinkage:
            attempt(
                f"deapply system linkage {linkage}",
                lambda linkage=linkage: self.deapply_system_linkage(linkage),
            )
        for cu, compu in compus.items():
            if compu.fps is not None:
                attempt(f"confstop {cu}", compu.confstop)
        for cu, compu in compus.items():
            if compu.fps is not None:
                attempt(f"destroy {cu}", compu.fps.destroy)
        if self.root_folder.exists() or self.root_folder.is_symlink():
            attempt(
                f"rm -rf {self.root_folder}", lambda: shutil.rmtree(self.root_folder)
            )
        if failures:
            raise BestEffortError(failures, pipeline=self.short_name)


class DMCombPipeline: ...


class AOSimPipeline: ...


class AOPipeline:
    # What's an AO pipeline ?
    # Anything that contains an mfilt.
    ...
