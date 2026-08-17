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

import os
import glob
import shutil
from pathlib import Path

from .task_models import SimpleTask

if typ.TYPE_CHECKING:
    from pyMilk.interfacing.fps import FPVal

    FPValNest: typ.TypeAlias = FPVal | dict[str, "FPValNest"]


class InitialFolderSetup(SimpleTask):

    def can(self) -> bool:
        # check logdir does NOT exist
        # check rootdir does NOT exist
        ...

    def success(self) -> bool: ...

    def forward(self):
        loop_name = self.pipeline.short_name
        conf_dir = self.pipeline.conf_folder
        root_dir = self.pipeline.root_folder

        print(f"CACAO_LOOPNAME    : {loop_name}")
        print(f"CONFDIR           : {conf_dir}")
        print(f"CACAO_LOOPROOTDIR : {root_dir}")

        root_dir.mkdir(parents=True, exist_ok=False)

        # ---- logging ----
        (root_dir / "fpsCTRL.log").unlink(missing_ok=True)
        (root_dir / "fpsCTRL.log").symlink_to(
            os.environ["MILK_SHM_DIR"] + f"/fpsCTRL-{loop_name}.log"
        )

        logdir = Path.cwd() / f"logdir-{loop_name}"
        logdir.mkdir(parents=True, exist_ok=True)
        (root_dir / "logdir").unlink(missing_ok=True)
        (root_dir / "logdir").symlink_to(logdir)

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

        # used by the matching cleanup task, to wait on tmux teardown
        self.fpsctrl_tmux_name = f"{loop_name}_fpsCTRL"


class TestConfig(SimpleTask):
    """
    maps to former cacao-setup -t
    """

    ...


class DeployFPS(SimpleTask):

    def can(self) -> bool: ...

    def success(self) -> bool: ...

    def forward(self) -> None:
        from ..session import ComputeSession

        p = self.pipeline

        # Probably should ensure that cwd is the config's ROOTDIR

        for session_name, exec in p.sessions.items():
            session_fps_config = p.session_configs[session_name]
            session = ComputeSession(exec, session_name + f"_{p.loop_number:03d}")

            session.fpsinit()
            assert session.fps

            # TODO probably should be a utility somewhere to flatten fps dicts
            def denest_fps(fp_param_dict: dict[str, FPValNest]) -> dict[str, FPVal]:
                copied: dict[str, FPVal] = {}
                for key, value in fp_param_dict.items():
                    if isinstance(value, dict):
                        new_value = denest_fps(value)
                        for kk, vv in new_value.items():
                            copied[f"{key}.{kk}"] = vv
                    else:
                        copied[key] = value

                return copied

            # TODO probably a method on the Session should provide the denested iterator.
            denested_fps_config = denest_fps(session_fps_config)

            for fp_name, fp_value in denested_fps_config.items():
                session.fps[fp_name] = fp_value


class StartConfProcesses(SimpleTask):
    TIMEOUT_ALL = 5.0

    def can(self) -> bool:
        # Find all sessions expected by this config
        # Check FPSs exist with the correct name
        # TODO remember at some point task.can must be called and raise.
        ...

    def success(self) -> bool: ...

    def forward(self) -> None:
        from ..session import ComputeSession

        p = self.pipeline

        # Probably should ensure that cwd is the config's ROOTDIR
        # The below could also probably be handled by a FPSManager of pyMilk
        sessions: list[ComputeSession] = []  # cache to apply global timeout
        for session_name, exec in p.sessions.items():
            session_fps_config = p.session_configs[session_name]
            session = ComputeSession(exec, session_name + f"_{p.loop_number:03d}")

            session.confstart(tmux=True)  # TODO dispatch env to tmuxes !!
            sessions += [session]

        # TODO should this be the success function ?
        import time

        start = time.monotonic()
        while time.monotonic() - start < self.TIMEOUT_ALL:
            if all([s.fps.conf_isrunning() for s in sessions]):  # type: ignore
                break
            time.sleep(0.01)
        else:
            raise TimeoutError(
                f"Timeout on StartConfProcesses for {self.pipeline.short_name} [loop {self.pipeline.loop_number}]"
            )


class CompoundTask(SimpleTask): ...


# Todo a compound task ?
