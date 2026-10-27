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

import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from pyMilk.interfacing.shm import MILK_SHM_DIR

from .infra import exceptions as exc

_CLONE_LOG = "clone-rsync.log"
_LINKAGE_TAG = "systemlinkage.tag"


def _git(folder: Path, *args: str) -> subprocess.CompletedProcess[str]:
    # Inherited Git routing must not redirect operations outside the clone.
    env = os.environ.copy()
    for key in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_COMMON_DIR",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    ):
        env.pop(key, None)
    return subprocess.run(
        ["git", "-C", str(folder), *args],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def _has_local_git(folder: Path) -> bool:
    metadata = folder / ".git"
    if metadata.is_symlink():
        raise ValueError(f"Refusing symlinked Git metadata: {metadata}")
    if not metadata.exists():
        return False
    root = Path(_git(folder, "rev-parse", "--show-toplevel").stdout.strip())
    if root.resolve() != folder.resolve():
        raise ValueError(f"Git repository is not rooted at {folder}")
    return True


class Pipeline:
    # Or should it be a _deployed pipeline_ ?
    # Or should deployed / deployable / undeployable different things ?

    def __init__(self, parent_folder: str | Path, long_name: str = "") -> None:

        self.parent_folder = Path(parent_folder).absolute()

        self.long_name = long_name

        self.conf_folder = self.parent_folder / f"{self.long_name}-conf"
        if not os.path.isdir(self.conf_folder):
            raise exc.PipelineConfFolderNotFoundException(
                f"Pipeline from {self.conf_folder}: folder does not exist."
            )
        self.config = load_pipeline_config(self.conf_folder)  # Raises if not existing

        self.sessions = self.config.sessions
        self.session_configs = self.config.session_configs
        self.dataloads = self.config.dataloads
        self.systemlinkage = self.config.systemlinkage

        self.short_name = self.config.name
        self.loop_number = self.config.loop_number

        self.root_folder = self.parent_folder / f"{self.short_name}-rootdir"
        self.log_folder = self.parent_folder / f"{self.short_name}-logdir"

        self.run_folder = self.root_folder / "rundir"
        self._clone_source: Path | None = None
        self._clone_rsync_output = ""

    @staticmethod
    def check_clone_target(conf_folder: str | Path) -> None:
        """Reject a dirty destination repository before copying any files."""
        folder = Path(conf_folder)
        if (folder / _CLONE_LOG).is_symlink():
            raise ValueError(f"Refusing symlinked clone log: {folder / _CLONE_LOG}")
        if _has_local_git(folder):
            status = _git(
                folder,
                "status",
                "--porcelain",
                "--untracked-files=all",
            ).stdout
            if status:
                raise ValueError(
                    f"Clone target has uncommitted changes: {folder}\n{status}"
                )

    def clone_to(self, new_parent_folder: str | Path) -> Pipeline:
        """Copy the configuration, retaining rsync output but performing no Git operations."""
        new_parent_folder = Path(new_parent_folder)
        new_conf_folder = new_parent_folder / f"{self.long_name}-conf"
        new_conf_folder.mkdir(parents=True, exist_ok=True)

        # trailing slashes: copy conf_folder's contents, not the folder itself
        result = subprocess.run(
            [
                "rsync",
                "-a",
                "--itemize-changes",
                "--exclude=.git",
                f"--exclude=/{_CLONE_LOG}",
                "--",
                f"{self.conf_folder}/",
                f"{new_conf_folder}/",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        cloned = Pipeline(new_parent_folder, self.long_name)
        cloned._clone_source = self.conf_folder.resolve()
        cloned._clone_rsync_output = result.stdout
        return cloned

    def update_local_git(self) -> None:
        """Commit clone changes; skip the commit when no staged changes exist.

        Call check_clone_target before clone_to to reject uncommitted destination changes.
        Update logs are appended only when rsync reports changes. Git identity and signing
        settings are inherited. Failures leave copied files in place.
        """
        if self._clone_source is None:
            raise ValueError(
                "Git clone recording requires a pipeline returned by clone_to"
            )

        folder = self.conf_folder.resolve()
        existing = _has_local_git(folder)
        source = self._clone_source
        if existing:
            if self._clone_rsync_output:
                log = folder / _CLONE_LOG
                if log.is_symlink():
                    raise ValueError(f"Refusing symlinked clone log: {log}")
                with log.open("a", encoding="utf-8") as stream:
                    stream.write(
                        f"\nClone at {datetime.now(timezone.utc).isoformat()}"
                        f" from {source} to {folder}\n{self._clone_rsync_output}"
                    )
        else:
            _git(folder, "init")

        _git(folder, "add", "--all", "--", ".")
        if existing and self._clone_rsync_output:
            _git(folder, "add", "--force", "--", _CLONE_LOG)

        try:
            _git(folder, "diff", "--cached", "--quiet")
        except subprocess.CalledProcessError as error:
            if error.returncode != 1:
                raise
        else:
            return

        prefix = "Update clone" if existing else "Initial commit -- clone"
        _git(folder, "commit", "-m", f"{prefix} from {source} to {folder}")

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
            raise NoCanTaskError(task)

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

    def is_running(self) -> bool:
        for name in self.sessions:
            # TODO: check this question has sense: is the pipeline in a correct state?
            fps = self.get_session(name).fps
            if fps is not None and fps.run_isrunning():
                return True
        return False

    def _linkage_paths(self, linkage_name: str) -> list[tuple[Path, Path]]:
        if linkage_name not in self.systemlinkage:
            raise exc.UnknownSystemLinkageException(linkage_name)
        shm = Path(MILK_SHM_DIR())
        return [
            (shm / f"{local}.im.shm", shm / f"{remote}.im.shm")
            for local, remote in self.systemlinkage[linkage_name].items()
        ]

    def apply_system_linkage(self, linkage_name: str) -> None:
        """Point each local stream file at its remote one, replacing existing symlinks."""
        pairs = self._linkage_paths(linkage_name)
        if self.is_running():
            raise exc.SystemLinkageException("Pipeline is running; stop it first")
        # TODO assert no other incompatible linkage is configured

        # Check everything first so a failure leaves nothing half-applied.
        for local, remote in pairs:
            if not remote.exists():
                raise exc.SystemLinkageException(f"Remote stream missing: {remote}")
            if local.exists() and not local.is_symlink():
                raise exc.SystemLinkageException(
                    f"Refusing to replace real file: {local}"
                )

        for local, remote in pairs:
            local.unlink(missing_ok=True)
            local.symlink_to(remote)

        stamp = datetime.now(timezone.utc).isoformat()
        (self.root_folder / _LINKAGE_TAG).write_text(f"{linkage_name}\n{stamp}\n")

    def deapply_system_linkage(self, linkage_name: str) -> None:
        """Remove the linkage's symlinks, leaving any that point elsewhere."""
        pairs = self._linkage_paths(linkage_name)
        if self.is_running():
            raise exc.SystemLinkageException("Pipeline is running; stop it first")
        for local, remote in pairs:
            if local.is_symlink():  # and Path(os.readlink(local)) == remote:
                local.unlink()

    def nuke(self) -> None:
        """Unconditionally tear down: runstop, deapply linkages, confstop, rm -rf root.

        Meant for test machines. Every step is best-effort: failures are reported
        and the next step still runs.
        """

        def attempt(label: str, fn) -> None:
            print(label)
            try:
                fn()
            except Exception as error:
                print(f"  warning: {error}")

        for name in self.sessions:
            session = self.get_session(name)
            if session.fps is not None:
                attempt(f"runstop {name}", session.runstop)
        for linkage in self.systemlinkage:
            attempt(
                f"deapply system linkage {linkage}",
                lambda linkage=linkage: self.deapply_system_linkage(linkage),
            )
        for name in self.sessions:
            session = self.get_session(name)
            if session.fps is not None:
                attempt(f"confstop {name}", session.confstop)
        for name in self.sessions:
            session = self.get_session(name)
            if session.fps is not None:
                session.fps.destroy()
        if self.root_folder.exists() or self.root_folder.is_symlink():
            attempt(
                f"rm -rf {self.root_folder}", lambda: shutil.rmtree(self.root_folder)
            )


class DMCombPipeline: ...


class AOSimPipeline: ...


class AOPipeline:
    # What's an AO pipeline ?
    # Anything that contains an mfilt.
    ...
