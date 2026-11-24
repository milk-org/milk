from __future__ import annotations

from .compute_unit import ComputeUnit
from ..config.resolve import confresolve
from ..config.layout import is_rootdir
from .plans import PLANS, Plan, run_plan, steps_from_list
from .state import State, gate, state
from .tasks import TASKS, Task
from ..registry import Registry
from .. import env
from ..errors import (
    ConfigError,
    LinkageError,
    TaskCannotRunError,
    TaskDidNotSucceedError,
)

import functools
import os
import runpy
import shutil
from pathlib import Path


class Pipeline:
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
        self.short_name = self.config.name
        self.work_folder = self.root_folder / "work"

    @classmethod
    def from_folder(cls, rootdir: str | Path) -> Pipeline:
        rootdir = Path(rootdir)
        if not is_rootdir(rootdir):
            raise ConfigError(f"{rootdir} is not a rootdir.", path=rootdir)
        return cls(rootdir.absolute())

    @functools.cached_property
    def _local(self) -> tuple[Registry[Plan], Registry[type[Task]]]:
        """Built on first access: the [plans.<name>] tables, then conf/scripts/moka_plans.py."""
        plans: Registry[Plan] = Registry("plan", parent=PLANS)
        tasks: Registry[type[Task]] = Registry("task", parent=TASKS)
        for name, decl in self.config.plans.items():
            steps = steps_from_list(decl.steps)
            options = decl.model_dump(exclude={"steps"}, exclude_none=True)
            plans.register(name, Plan(name, steps, **options))
        path = self.root_folder / "conf" / "scripts" / "moka_plans.py"
        if path.is_file():  # not sandboxed: trusted like [[hooks]] commands
            ns = runpy.run_path(str(path))  # no bytecode, nothing left in sys.modules
            for task in ns.get("TASKS", []):
                tasks.register(task.__name__, task)
            plans = Registry("plan", parent=plans)  # so that Python overrides TOML
            for plan in ns.get("PLANS", []):
                plans.register(plan.name, plan)
        return plans, tasks

    @property
    def plans(self) -> Registry[Plan]:
        """PLANS, plus and over this pipeline's own plans."""
        return self._local[0]

    @property
    def tasks(self) -> Registry[type[Task]]:
        """TASKS, plus and over this pipeline's own tasks."""
        return self._local[1]

    def task_do(self, task: Task | type[Task]) -> Pipeline:
        self._do(task)
        return self  # So as to be able to chain

    def _do(self, task: Task | type[Task]) -> bool:
        """task_do; True when forward() was called."""
        if isinstance(task, type):
            task = task(self)
        if task.success():
            return False
        name = type(task).__name__
        if not task.can():
            raise TaskCannotRunError(
                f"Task {name} cannot be executed:\n    {task.failure_reason}",
                task=name,
                pipeline=self.short_name,
            )
        task.forward()
        if not task.success():
            raise TaskDidNotSucceedError(
                f"Task {name} did not succeed:\n    {task.failure_reason}",
                task=name,
                pipeline=self.short_name,
            )
        return True

    @property
    def compute_units(self) -> dict[str, ComputeUnit]:
        """Fresh views keyed like config.compute_units. The only ComputeUnit construction in moka/."""
        return {
            cu: ComputeUnit(spec.exec, self.config.fps_name(cu))
            for cu, spec in self.config.compute_units.items()
        }

    def dm_mask(self):
        """Bool numpy (y, x) of [dmhint]: actuators strictly within the beam; all without one."""
        import numpy as np

        hint = self.config.dmhint
        assert hint is not None
        if hint.beam_rad is None:
            return np.ones((hint.ysize, hint.xsize), bool)
        from pycacao.calib.mkmodes import make_disk

        beam = hint.beam_xcent, hint.beam_ycent, hint.beam_rad
        return make_disk(beam, (hint.xsize, hint.ysize))

    def any_run_active(self) -> bool:
        return any(
            c.fps is not None and c.fps.run_isrunning()
            for c in self.compute_units.values()
        )

    def _linkage_paths(self, linkage_name: str) -> list[tuple[Path, Path]]:
        if linkage_name not in self.config.systemlinkage:
            raise LinkageError(
                f"Unknown system linkage {linkage_name!r}", pipeline=self.short_name
            )
        shm = env.shm_dir()
        return [
            (shm / f"{local}.im.shm", shm / f"{remote}.im.shm")
            for local, remote in self.config.systemlinkage[linkage_name].items()
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
        st = state(self)
        gate(self.short_name, st, "consistent")
        if st.state in (State.RUNNING, State.MIXED):
            raise LinkageError(
                "Pipeline is running; stop it first", pipeline=self.short_name
            )
        for local, remote in self._linkage_paths(name):
            local.unlink(missing_ok=True)
            local.symlink_to(remote)

    def unlink(self, name: str) -> None:
        """Remove the linkage's symlinks, leaving any that point elsewhere; ``all``: every linkage."""
        for linkage in self.config.systemlinkage if name == "all" else [name]:
            for local in self._linked_paths(linkage):
                local.unlink()

    def linked(self) -> list[str]:
        """The configured linkages that have a symlink in place."""
        return [name for name in self.config.systemlinkage if self._linked_paths(name)]

    def nuke_unsafe(self) -> None:
        """run_plan("teardown"), then rm -rf the rootdir even if teardown failed. Never prompts."""
        try:
            run_plan(self, "teardown")
        finally:
            shutil.rmtree(self.root_folder)
