"""
Goals for today

Deploy 3 pipelines:
1) DMComb
2) DMSim -> linsim -> WFSSim
3) AOloop

# How deploy ?
# fpsexec -> cwd + exec name + name

# trylink sessions
# if exist by name, find pwd
# if not, use a pwd to do the deploy

# Do factory functions as alternate constructors
# factory:: from_scan
# factory:: from_conf_file



# Levels of deployment akin to cacao-loop-deploy
"""

from __future__ import annotations

from .session import ComputeSession
from .infra.toml_manipulation import load_pipeline_config
from .infra.task_models import SimpleTask, NoCanTaskError, NoSuccessTaskError

import subprocess
from pathlib import Path


class Pipeline:
    # Or should it be a _deployed pipeline_ ?
    # Or should deployed / deployable / undeployable different things ?

    def __init__(self, parent_folder: str | Path, long_name: str = "") -> None:

        self.parent_folder = Path(parent_folder).absolute()

        self.long_name = long_name

        self.conf_folder = self.parent_folder / f"{self.long_name}-conf"
        self.config = load_pipeline_config(self.conf_folder)

        self.sessions = self.config.sessions
        self.session_configs = self.config.session_configs

        self.short_name = self.config.name
        self.loop_number = self.config.loop_number

        self.root_folder = self.parent_folder / f"{self.short_name}-rootdir"
        self.log_folder = self.parent_folder / f"{self.short_name}-logdir"

        self.run_folder = self.root_folder / "rundir"

    def clone_to(self, new_parent_folder: str | Path) -> Pipeline:
        new_parent_folder = Path(new_parent_folder)
        new_conf_folder = new_parent_folder / f"{self.long_name}-conf"
        new_conf_folder.mkdir(parents=True, exist_ok=True)

        # trailing slashes: copy conf_folder's contents, not the folder itself
        subprocess.run(
            ["rsync", "-a", f"{self.conf_folder}/", f"{new_conf_folder}/"],
            check=True,
        )

        return Pipeline(new_parent_folder, self.long_name)

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
            raise NoCanTaskError(f"Task {task} cannot be executed")

        if not task.success():
            # TODO if partial success is tolerated ?
            raise NoSuccessTaskError(f"Task {task} did not complete to success")

        return self  # So as to be able to chain

    def task_undo(self): ...

    def get_session_abs_name(self, session_relative_name: str) -> str:
        # TODO UNLESS it's a DM session !!
        return session_relative_name + f"_{self.loop_number:03d}"

    def get_session(self, session_relative_name: str) -> ComputeSession:
        absolute_name = self.get_session_abs_name(session_relative_name)

        return ComputeSession(self.sessions[session_relative_name], absolute_name)


class DMCombPipeline: ...


class AOSimPipeline: ...


class AOPipeline:
    # What's an AO pipeline ?
    # Anything that contains an mfilt.
    ...
