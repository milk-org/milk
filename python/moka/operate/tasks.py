"""Tasks: the steps of plans, registered in TASKS under their class name."""

from __future__ import annotations

import abc
import contextlib
import os
import subprocess
from collections.abc import Callable, Sequence
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


def stream_shape(name: str) -> tuple[int, ...] | None:
    """md size of the stream, (x, y[, n]) in milk's order; None if it does not exist."""
    from pyMilk.interfacing.shm import SHM

    if not (env.shm_dir() / f"{name}.im.shm").is_file():
        return None
    shm = SHM(name, symcode=0)
    md = shm.IMAGE.md
    shape = tuple(md.size[: md.naxis])
    shm.close()
    return shape


class Task(abc.ABC):
    """A step on a Pipeline: its folders, files, streams, claims. Takes no compute units."""

    TIMEOUT: float = 10.0  # bound for forward()'s wait on success()
    failure_reason: str = "no reason given."

    def __init__(
        self,
        pipeline: Pipeline,
        compus: Sequence[str] | None = None,
        params: dict | None = None,
    ) -> None:
        if compus is not None:
            raise ConfigError(
                f"{type(self).__name__} takes no compute units.",
                pipeline=pipeline.short_name,
            )
        self.pipeline = pipeline
        # the plan's Param values; a task reads the keys it knows
        self.params = params or {}

    def _refusal(self) -> str | None:
        """Why forward() must not run; None when it may."""
        return None

    def can(self) -> bool:
        """Whether _refusal() is None; else keeps the reason in failure_reason."""
        reason = self._refusal()
        if reason is not None:
            self.failure_reason = reason
        return reason is None

    @abc.abstractmethod
    def forward(self) -> None: ...

    @abc.abstractmethod
    def success(self) -> bool: ...


class TaskOnCompuSet(Task):
    """A task on a subset of the pipeline's compute units, in the subset's order."""

    # the subset has exactly this many units, named by the step
    COUNT: int | None = None

    def __init__(
        self,
        pipeline: Pipeline,
        compus: Sequence[str] | None = None,
        params: dict | None = None,
    ) -> None:
        super().__init__(pipeline, None, params)
        name = type(self).__name__
        if compus is None:
            self.locate_defaults()
        else:
            self.compus_names = tuple(compus)
        unknown = sorted(set(self.compus_names) - pipeline.config.compute_units.keys())
        if unknown:
            raise ConfigError(
                f"Unknown compute units {unknown} for {name}.",
                pipeline=pipeline.short_name,
            )
        if self.COUNT not in (None, len(self.compus_names)):
            raise ConfigError(
                f"{name} takes {self.COUNT} compute unit(s), got {list(self.compus_names)}.",
                pipeline=pipeline.short_name,
            )

    def locate_defaults(self) -> None:
        """Sets the subset of a step that names none: every unit; ConfigError if COUNT is set."""
        if self.COUNT is not None:
            raise ConfigError(
                f"{type(self).__name__} needs its {self.COUNT} compute unit(s) named in the step.",
                pipeline=self.pipeline.short_name,
            )
        self.compus_names = tuple(self.pipeline.config.compute_units)

    @property
    def compute_units(self) -> dict[str, ComputeUnit]:
        """The subset's CompUs, in the subset's order."""
        every = self.pipeline.compute_units
        return {cu: every[cu] for cu in self.compus_names}

    def _existing(self) -> list[tuple[ComputeUnit, FPS]]:
        """The subset's CompUs whose FPS exists, with that FPS."""
        return [(c, c.fps) for c in self.compute_units.values() if c.fps is not None]

    def _lacking(self, ok: Callable[[FPS], object]) -> list[str]:
        """fpsnames of the subset's CompUs whose FPS is missing or fails ok."""
        units = self.compute_units.values()
        return [c.fpsname for c in units if c.fps is None or not ok(c.fps)]

    def _undeployed(self) -> str | None:
        """A _refusal(): an FPS of the subset is missing or its conf process is not running."""
        lacking = self._lacking(lambda fps: fps.conf_isrunning())
        if lacking:
            return f"{lacking[0]} is missing or its conf process is not running; run deploy."
        return None


class TaskOnOneCompu(TaskOnCompuSet):
    """A task on exactly one compute unit."""

    COUNT = 1

    @property
    def compu(self) -> ComputeUnit:
        return self.pipeline.compute_units[self.compus_names[0]]

    @property
    def fps(self) -> FPS | None:
        return self.compu.fps


class ClaimRundir(Task):
    def _refusal(self) -> str | None:
        p = self.pipeline
        for claim in rundir.all_claims():
            if claim.name == p.short_name and claim.rootdir != str(p.root_folder):
                return f"{p.short_name} is claimed by {claim.rootdir}."
            if claim.name != p.short_name and claim.loop_number == p.config.loop_number:
                return f"loop_number {p.config.loop_number} is claimed by {claim.name}."
        return None

    def forward(self) -> None:
        p = self.pipeline
        rundir.write_claim(
            rundir.Claim.new(p.short_name, p.config.loop_number, str(p.root_folder))
        )

    def success(self) -> bool:
        name = self.pipeline.short_name
        claim = rundir.read_claim(name)
        return (
            claim is not None
            and claim.rootdir == str(self.pipeline.root_folder)
            and rundir.read_boot_id(name) == rundir.current_boot_id()
        )


class InitFPS(TaskOnCompuSet):
    def _refusal(self) -> str | None:
        if not os.path.isdir(self.pipeline.work_folder):
            return f"{self.pipeline.work_folder} does not exist."
        return None

    def forward(self) -> None:
        for cu, compu in self.compute_units.items():
            with contextlib.chdir(self.pipeline.work_folder):
                compu.fpsinit()
            for key, value in denest_toml_dicts(
                self.pipeline.config.compute_units[cu].fps
            ).items():
                compu.fps[key] = value  # type: ignore[index]
            compu.autopopulate_trigger_stream()

    def success(self) -> bool:
        return not self._lacking(lambda fps: fps.is_valid())


class LoadData(Task):
    """Loads every [dataloads] source file (from rootdir/initdata) into its target SHM stream."""

    def _refusal(self) -> str | None:
        for entry in self.pipeline.config.dataloads:
            source_path = self.pipeline.root_folder / "initdata" / entry.source
            if not os.path.isfile(source_path):
                return f"{source_path} does not exist."
        return None

    def forward(self) -> None:
        from astropy.io import fits
        from pyMilk.interfacing.shm import SHM

        for entry in self.pipeline.config.dataloads:
            source_path = self.pipeline.root_folder / "initdata" / entry.source
            SHM(entry.target, fits.getdata(str(source_path)), symcode=0).close()

    def success(self) -> bool:
        from astropy.io import fits
        from pyMilk.interfacing.shm import SHM

        for entry in self.pipeline.config.dataloads:
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


class ConfStart(TaskOnCompuSet):
    def _refusal(self) -> str | None:
        invalid = self._lacking(lambda fps: fps.is_valid())
        if invalid:
            return f"FPS {invalid[0]} is missing or invalid."
        return None

    def forward(self) -> None:
        for compu, fps in self._existing():
            if not fps.conf_isrunning():
                compu.confstart(tmux=True)
        wait_until(self.success, self.TIMEOUT)

    def success(self) -> bool:
        stuck = self._lacking(lambda fps: fps.conf_isrunning())
        if stuck:
            self.failure_reason = f"Conf processes not running for {', '.join(stuck)} (tmux session busy?)"
        return not stuck


class RunStart(TaskOnCompuSet):
    """Starts the subset (default: config.runstart) one unit at a time, in order;
    all at once when nobody wrote an order (no step units, no runstart key)."""

    ordered = True

    def locate_defaults(self) -> None:
        config = self.pipeline.config
        self.compus_names = tuple(config.runstart)
        self.ordered = "runstart" in config.model_fields_set

    _refusal = TaskOnCompuSet._undeployed

    def forward(self) -> None:
        for compu, fps in self._existing():
            if not fps.run_isrunning():
                compu.runstart(tmux=True)
                if self.ordered and not wait_until(fps.run_isrunning, self.TIMEOUT):
                    self.failure_reason = f"Run not started for {compu.fpsname}."
                    return
        if not self.ordered and not wait_until(self.success, self.TIMEOUT):
            down = self._lacking(lambda fps: fps.run_isrunning())
            self.failure_reason = f"Run not started for {', '.join(down)}."

    def success(self) -> bool:
        return not self._lacking(lambda fps: fps.run_isrunning())


class RunStop(TaskOnCompuSet):
    def forward(self) -> None:
        for compu, fps in self._existing():
            if fps.run_isrunning():
                compu.runstop()
        wait_until(self.success, self.TIMEOUT)

    def success(self) -> bool:
        return not any(fps.run_isrunning() for _, fps in self._existing())


class WaitRunEnd(TaskOnCompuSet):
    """Waits for the subset's runs to end by themselves; never stops anything."""

    TIMEOUT = 120.0

    def _ended(self) -> bool:
        return not any(fps.run_isrunning() for _, fps in self._existing())

    def forward(self) -> None:
        if not wait_until(self._ended, self.TIMEOUT, interval=0.05):
            running = [c.fpsname for c, f in self._existing() if f.run_isrunning()]
            self.failure_reason = f"Still running: {', '.join(running)}."

    def success(self) -> bool:
        return self._ended()


class DeapplyAllLinkages(Task):
    def forward(self) -> None:
        self.pipeline.unlink("all")

    def success(self) -> bool:
        return not self.pipeline.linked()


class ConfStop(TaskOnCompuSet):
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


class KillTmux(TaskOnCompuSet):
    def _live(self) -> list[str]:
        if not self.pipeline.config.teardown_kill_tmux:
            return []
        return [c.fpsname for c in self.compute_units.values() if _tmux_has(c.fpsname)]

    def forward(self) -> None:
        for session in self._live():
            subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)

    def success(self) -> bool:
        return not self._live()


class DestroyFPS(TaskOnCompuSet):
    def forward(self) -> None:
        for _, fps in self._existing():
            fps.destroy()

    def success(self) -> bool:
        return not self._existing()


class ReleaseRundir(Task):
    def forward(self) -> None:
        rundir.release(self.pipeline.short_name)

    def success(self) -> bool:
        return not rundir.rundir_path(self.pipeline.short_name).exists()


TASKS: Registry[type[Task]] = Registry("task")
for _task in Task.__subclasses__() + TaskOnCompuSet.__subclasses__():
    if not _task.__abstractmethods__:
        TASKS.register(_task.__name__, _task)
