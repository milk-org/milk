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
from ..session import ComputeSession

from .toml_manipulation import denest_toml_dicts


class InitialFolderSetup(SimpleTask):

    def can(self) -> bool:
        # check logdir does NOT exist
        # check rootdir does NOT exist
        return True  # TODO

    def success(self) -> bool:
        root_dir = self.pipeline.root_folder
        if not os.path.isdir(root_dir):
            return False
        if not os.path.isfile(root_dir / "conf.toml"):
            return False

        # More ?

        return True

    def forward(self):
        pp = self.pipeline

        loop_name = pp.short_name
        conf_dir = pp.conf_folder
        root_dir = pp.root_folder

        print(f"CACAO_LOOPNAME    : {loop_name}")
        print(f"CONFDIR           : {conf_dir}")
        print(f"CACAO_LOOPROOTDIR : {root_dir}")

        root_dir.mkdir(parents=True, exist_ok=False)

        # ---- logging ----
        (root_dir / "fpsCTRL.log").unlink(missing_ok=True)
        (root_dir / "fpsCTRL.log").symlink_to(
            os.environ["MILK_SHM_DIR"] + f"/fpsCTRL-{loop_name}.log"
        )

        pp.log_folder.mkdir(parents=True, exist_ok=True)
        (root_dir / "logdir").unlink(missing_ok=True)
        (root_dir / "logdir").symlink_to(pp.log_folder)

        pp.run_folder.mkdir(parents=True, exist_ok=True)

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


class DeployFPS(SimpleTask):

    def can(self) -> bool:
        p = self.pipeline
        if not os.path.isdir(p.root_folder):
            return False
        if not os.path.isdir(p.run_folder):
            return False
        return True

    def success(self) -> bool:
        p = self.pipeline
        for session_name, exec in p.sessions.items():
            session = ComputeSession(exec, session_name + f"_{p.loop_number:03d}")
            if not session.fps or not session.fps.is_valid():
                return False

        return True

    def forward(self) -> None:

        p = self.pipeline

        # Probably should ensure that cwd is the config's ROOTDIR

        for session_name, exec in p.sessions.items():
            session_fps_config = p.session_configs[session_name]
            session = ComputeSession(exec, session_name + f"_{p.loop_number:03d}")

            with contextlib.chdir(p.run_folder):
                session.fpsinit()

            assert session.fps

            # TODO probably a method on the Session should provide the denested iterator.
            denested_fps_config = denest_toml_dicts(session_fps_config)

            for fp_name, fp_value in denested_fps_config.items():
                session.fps[fp_name] = fp_value


class StartConfProcesses(SimpleTask):
    TIMEOUT_ALL = 5.0

    def can(self) -> bool:
        p = self.pipeline

        # Folder consistency: rootdir/rundir must have been laid out already
        if not os.path.isdir(p.root_folder):
            return False
        if not os.path.isdir(p.run_folder):
            return False

        # Every session's FPS must exist and be a well-formed deployment
        for session_name, exec in p.sessions.items():
            session = ComputeSession(exec, session_name + f"_{p.loop_number:03d}")
            if session.fps is None or not session.fps.is_valid():
                return False

        # TODO get & check the rootdir for all FPSs

        return True

    def success(self) -> bool:
        # TODO I deffo want a Pipeline.sessions property now
        p = self.pipeline
        sessions = [
            ComputeSession(exec, s_name + f"_{p.loop_number:03d}")
            for (s_name, exec) in p.sessions.items()
        ]
        if any([s.fps is None for s in sessions]):
            return False
        return all([s.fps is not None and s.fps.conf_isrunning() for s in sessions])

    def forward(self) -> None:
        from ..session import ComputeSession

        p = self.pipeline

        # Probably should ensure that cwd is the config's ROOTDIR
        # The below could also probably be handled by a FPSManager of pyMilk
        sessions: list[ComputeSession] = []  # cache to apply global timeout
        for session_name, exec in p.sessions.items():
            session = ComputeSession(exec, session_name + f"_{p.loop_number:03d}")
            # TODO breaks if tmux is busy already (in particular poor cleaning of repeated tests.)
            session.confstart(tmux=True)  # TODO dispatch env to tmuxes !!
            sessions += [session]

        import time

        start = time.monotonic()
        while time.monotonic() - start < self.TIMEOUT_ALL:
            if all([s.fps.conf_isrunning() for s in sessions]):  # type: ignore
                break
            time.sleep(0.01)
        else:  # Should it raise ? We're gonna call self.success for that
            ...
            # raise TimeoutError(
            #    f"Timeout on StartConfProcesses for {self.pipeline.short_name} [loop {self.pipeline.loop_number}]"
            # )


class CompoundTask(SimpleTask):
    ...

    # TODO what about a compound task ?
    # TODO which tasks are reversible ?
    # TODO what's next? Naming of DM based tasks, and then getting into the aorun suite.
