"""Tasks (design-v7 §4.2): the steps of plans, registered in TASKS under their class name."""

from __future__ import annotations

import abc
import contextlib
import os
import subprocess
from collections.abc import Collection
from typing import TYPE_CHECKING

from .. import env
from .compute_unit import ComputeUnit
from ..config.substitute import denest_toml_dicts
from ..errors import ConfigError
from ..registry import Registry
from ..wait import wait_until
from . import rundir

if TYPE_CHECKING:
    from pyMilk.interfacing.fps import FPS

    from .pipeline import Pipeline


class Task(abc.ABC):
    TIMEOUT: float = 10.0  # bound for forward()'s wait on success()
    failure_reason: str = "no reason given."

    def __init__(
        self, pipeline: Pipeline, compus: Collection[str] | None = None
    ) -> None:
        unknown = sorted(set(compus or ()) - pipeline.config.compute_units.keys())
        if unknown:
            raise ConfigError(
                f"Unknown compute units {unknown} for {type(self).__name__}.",
                pipeline=pipeline.short_name,
            )
        self.pipeline = pipeline
        self.compus_names = compus

    @property
    def compute_units(self) -> dict[str, ComputeUnit]:
        return {
            cu: c
            for cu, c in self.pipeline.compute_units.items()
            if self.compus_names is None or cu in self.compus_names
        }

    def _existing(self) -> list[tuple[ComputeUnit, FPS]]:
        """The subset's CompUs whose FPS exists, with that FPS."""
        return [(c, c.fps) for c in self.compute_units.values() if c.fps is not None]

    @abc.abstractmethod
    def can(self) -> bool: ...

    @abc.abstractmethod
    def forward(self) -> None: ...

    @abc.abstractmethod
    def success(self) -> bool: ...


class ClaimRundir(Task):
    def can(self) -> bool:
        p = self.pipeline
        for claim in rundir.all_claims():
            if claim.name == p.short_name and claim.rootdir != str(p.root_folder):
                self.failure_reason = f"{p.short_name} is claimed by {claim.rootdir}."
                return False
            if claim.name != p.short_name and claim.loop_number == p.loop_number:
                self.failure_reason = (
                    f"loop_number {p.loop_number} is claimed by {claim.name}."
                )
                return False
        return True

    def forward(self) -> None:
        p = self.pipeline
        rundir.write_claim(
            rundir.Claim.new(p.short_name, p.loop_number, str(p.root_folder))
        )

    def success(self) -> bool:
        name = self.pipeline.short_name
        claim = rundir.read_claim(name)
        return (
            claim is not None
            and claim.rootdir == str(self.pipeline.root_folder)
            and rundir.read_boot_id(name) == rundir.current_boot_id()
        )


class InitFPS(Task):
    def can(self) -> bool:
        if not os.path.isdir(self.pipeline.work_folder):
            self.failure_reason = f"{self.pipeline.work_folder} does not exist."
            return False
        return True

    def forward(self) -> None:
        for cu, compu in self.compute_units.items():
            with contextlib.chdir(self.pipeline.work_folder):
                compu.fpsinit()
            for key, value in denest_toml_dicts(
                self.pipeline.config.compute_units[cu].fps
            ).items():
                compu.fps[key] = value  # type: ignore[index]

    def success(self) -> bool:
        return all(
            c.fps is not None and c.fps.is_valid() for c in self.compute_units.values()
        )


class LoadData(Task):
    """Loads every [dataloads] source file (from rootdir/initdata) into its target SHM stream."""

    def can(self) -> bool:
        for entry in self.pipeline.dataloads:
            source_path = self.pipeline.root_folder / "initdata" / entry.source
            if not os.path.isfile(source_path):
                self.failure_reason = f"{source_path} does not exist."
                return False
        return True

    def forward(self) -> None:
        from astropy.io import fits
        from pyMilk.interfacing.shm import SHM

        for entry in self.pipeline.dataloads:
            source_path = self.pipeline.root_folder / "initdata" / entry.source
            SHM(entry.target, fits.getdata(str(source_path)), symcode=0).close()

    def success(self) -> bool:
        from astropy.io import fits
        from pyMilk.interfacing.shm import SHM

        for entry in self.pipeline.dataloads:
            if not (env.shm_dir() / f"{entry.target}.im.shm").is_file():
                return False
            expected = fits.getdata(
                str(self.pipeline.root_folder / "initdata" / entry.source)
            )
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


class ConfStart(Task):
    def can(self) -> bool:
        for compu in self.compute_units.values():
            if compu.fps is None or not compu.fps.is_valid():
                self.failure_reason = f"FPS {compu.fpsname} is missing or invalid."
                return False
        return True

    def forward(self) -> None:
        for compu, fps in self._existing():
            if not fps.conf_isrunning():
                compu.confstart(tmux=True)
        wait_until(self.success, self.TIMEOUT)

    def success(self) -> bool:
        stuck = [
            c.fpsname
            for c in self.compute_units.values()
            if c.fps is None or not c.fps.conf_isrunning()
        ]
        if stuck:
            self.failure_reason = f"Conf processes not running for {', '.join(stuck)} (tmux session busy?)"
        return not stuck


class RunStart(Task):
    def can(self) -> bool:
        for compu in self.compute_units.values():
            if compu.fps is None or not compu.fps.conf_isrunning():
                self.failure_reason = f"{compu.fpsname} is missing or its conf process is not running; run deploy."
                return False
        return True

    def forward(self) -> None:
        for compu, fps in self._existing():
            if not fps.run_isrunning():
                compu.runstart(tmux=True)
        if not wait_until(self.success, self.TIMEOUT):
            stuck = [c.fpsname for c, f in self._existing() if not f.run_isrunning()]
            self.failure_reason = f"Run not started for {', '.join(stuck)}."

    def success(self) -> bool:
        return all(
            c.fps is not None and c.fps.run_isrunning()
            for c in self.compute_units.values()
        )


class RunStop(Task):
    def can(self) -> bool:
        return True

    def forward(self) -> None:
        for compu, fps in self._existing():
            if fps.run_isrunning():
                compu.runstop()
        wait_until(self.success, self.TIMEOUT)

    def success(self) -> bool:
        return not any(fps.run_isrunning() for _, fps in self._existing())


class WaitRunEnd(Task):
    """Waits for the subset's runs to end by themselves; never stops anything."""

    TIMEOUT = 120.0

    def can(self) -> bool:
        return True

    def forward(self) -> None:
        if not wait_until(self.success, self.TIMEOUT, interval=0.05):
            running = [c.fpsname for c, f in self._existing() if f.run_isrunning()]
            self.failure_reason = f"Still running: {', '.join(running)}."

    def success(self) -> bool:
        return not any(fps.run_isrunning() for _, fps in self._existing())


class DeapplyAllLinkages(Task):
    def can(self) -> bool:
        return True

    def forward(self) -> None:
        self.pipeline.unlink("all")

    def success(self) -> bool:
        return not self.pipeline.linked()


class ConfStop(Task):
    def can(self) -> bool:
        return True

    def forward(self) -> None:
        for compu, fps in self._existing():
            if fps.conf_isrunning():
                compu.confstop()
        wait_until(self.success, self.TIMEOUT)

    def success(self) -> bool:
        return not any(fps.conf_isrunning() for _, fps in self._existing())


def _tmux_has(session: str) -> bool:
    argv = ["tmux", "has-session", "-t", session]
    return not subprocess.run(argv, capture_output=True).returncode


class KillTmux(Task):
    def can(self) -> bool:
        return True

    def _live(self) -> list[str]:
        if not self.pipeline.config.teardown_kill_tmux:
            return []
        return [c.fpsname for c in self.compute_units.values() if _tmux_has(c.fpsname)]

    def forward(self) -> None:
        for session in self._live():
            subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)

    def success(self) -> bool:
        return not self._live()


class DestroyFPS(Task):
    def can(self) -> bool:
        return True

    def forward(self) -> None:
        for _, fps in self._existing():
            fps.destroy()

    def success(self) -> bool:
        return not self._existing()


class ReleaseRundir(Task):
    def can(self) -> bool:
        return True

    def forward(self) -> None:
        rundir.release(self.pipeline.short_name)

    def success(self) -> bool:
        return not rundir.rundir_path(self.pipeline.short_name).exists()


TASKS: Registry[type[Task]] = Registry("task")
for _task in Task.__subclasses__():
    TASKS.register(_task.__name__, _task)
