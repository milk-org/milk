from __future__ import annotations
import typing as typ

from ..pipeline import Pipeline
from .exceptions import SystemLinkageException, UnknownSystemLinkageException

from pathlib import Path

import os
import subprocess
import click

if typ.TYPE_CHECKING:
    from .task_models import SimpleTask

MILK_SAMPLES_DIR = os.environ["MILK_ROOT"] + "/milk-examples/pipelines/"
CACAO_SAMPLES_DIR = os.environ["MILK_ROOT"] + "/plugins/cacao-src/examples/"


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def milk_pipes():
    """
    Clone, initialize and spool up a milk/cacao pipeline.

    \b
    Subcommands:
      clone   Clone a pipeline configuration to the current folder
      init    Set up the folders of a cloned pipeline
      spool   Deploy and start the processes of an initialized pipeline
      sys     Apply a [systemlinkage] mode of a pipeline
      nuke    Stop, unlink and delete the root directory of a pipeline (destructive)
    """


@milk_pipes.command("clone", context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-m", "--milk", "milk_exs", is_flag=True, help="<name> is a milk example pipeline"
)
@click.option(
    "-c",
    "--cacao",
    "cacao_exs",
    is_flag=True,
    help="<name> is a cacao example pipeline",
)
@click.option(
    "-f",
    "--full-path",
    "full_path",
    is_flag=True,
    help="<name> is the path to a pipeline configuration folder",
)
@click.argument("name", required=False)
def milk_pipes_clone(
    milk_exs: bool,
    cacao_exs: bool,
    full_path: bool,
    name: str | None,
):
    """
    Clone a pipeline configuration into the current folder.

    Exactly one of -m, -c, -f must be given. With -m/-c, <name> is looked up
    in the milk/cacao examples. With -f, <name> is the path to the
    configuration (`<path>/<name>-conf` or `<path>/<name>`).
    Without <name>, the available configurations are listed.

    Commit the clone in its own Git repository, respecting .gitignore.
    Existing repositories must be clean; updates append clone-rsync.log.
    No commit is created when there are no staged changes.
    Git identity and signing must already be configured. Failed commits
    leave the copied files in place for inspection.
    """
    # TODO milk-pipes clone <full_path> should be allowed.
    # TODO a --to <destination folder> should be allowed.
    # TODO add a name resolution for <conf_name> <loop_name> <folder_name>
    # TODO -l to get a list of clonables
    # TODO add a confname, aoname, aonum override during the cloning
    if sum([milk_exs, cacao_exs, full_path]) != 1:
        raise click.UsageError(
            "Exactly one of -m/--milk, -c/--cacao, -f/--full-path " "is required"
        )

    if name is None:
        if milk_exs:
            browse_and_print_conf_folders(MILK_SAMPLES_DIR)
        elif cacao_exs:
            browse_and_print_conf_folders(CACAO_SAMPLES_DIR)
        else:
            raise click.UsageError("-f/--full-path requires <name>")
        return

    milk_pipes_clone_function(name, milk_exs, cacao_exs, full_path)


def milk_pipes_clone_function(
    name: str,
    milk_exs: bool = False,
    cacao_exs: bool = False,
    full_path: bool = False,
    dest_folder: str | Path | None = None,
) -> Pipeline:
    """Clone to dest_folder (default cwd), then record the returned pipeline in Git."""
    if milk_exs:
        parent, long_name = Path(MILK_SAMPLES_DIR), name
    elif cacao_exs:
        parent, long_name = Path(CACAO_SAMPLES_DIR), name
    elif full_path:
        full = Path(name.rstrip("/"))
        parent, long_name = full.parent, full.name.removesuffix("-conf")
    else:
        raise click.UsageError("One of milk_exs, cacao_exs, full_path needed")

    dest = Path(dest_folder) if dest_folder is not None else Path.cwd()
    pp = Pipeline(parent, long_name)  # Raise on missing folder
    target = dest / f"{long_name}-conf"
    if pp.conf_folder.resolve() == target.resolve():
        raise click.UsageError(
            f"{long_name}-conf is already in {dest}; nothing to clone."
        )

    try:
        Pipeline.check_clone_target(target)
        cloned = pp.clone_to(dest)
        cloned.update_local_git()
    except subprocess.CalledProcessError as error:
        detail = error.stderr or error.stdout or str(error)
        raise click.ClickException(
            f"Clone/Git recording failed for {target}: {detail.strip()}"
        ) from error
    except (OSError, ValueError) as error:
        raise click.ClickException(
            f"Clone/Git recording failed for {target}: {error}"
        ) from error
    print(f"Cloned {pp.conf_folder} to {cloned.conf_folder}")
    return cloned


_folder_option = click.option(
    "--folder",
    "folder",
    type=click.Path(),
    default=None,
    help="Parent folder to the pipeline configuration (defaults to cwd)",
)


@milk_pipes.command("init", context_settings={"help_option_names": ["-h", "--help"]})
@_folder_option
@click.argument("name", required=False)
def milk_pipes_init(folder: str | None, name: str | None):
    """
    Perform the initial folder setup of a cloned pipeline <name>.

    Without <name>, the pipelines found in the folder are listed.
    """
    if name is None:
        browse_and_print_conf_folders(folder or os.getcwd())
        return
    milk_pipes_init_function(name, folder)


@milk_pipes.command("spool", context_settings={"help_option_names": ["-h", "--help"]})
@_folder_option
@click.argument("name", required=False)
def milk_pipes_spool(folder: str | None, name: str | None):
    """
    Deploy FPSs, start conf processes and load data files of pipeline <name>.

    Without <name>, the pipelines found in the folder are listed.
    """
    if name is None:
        browse_and_print_conf_folders(folder or os.getcwd())
        return
    milk_pipes_spool_function(name, folder)


@milk_pipes.command("sys", context_settings={"help_option_names": ["-h", "--help"]})
@_folder_option
@click.argument("name")
@click.argument("linkage")
def milk_pipes_sys(folder: str | None, name: str, linkage: str):
    """
    Apply the [systemlinkage.<LINKAGE>] symlinks of pipeline <name>.
    """
    milk_pipes_sys_function(name, linkage, folder)


def milk_pipes_sys_function(name: str, linkage: str, folder: str | None = None) -> None:
    pp = Pipeline(folder if folder is not None else os.getcwd(), name)
    try:
        pp.apply_system_linkage(linkage)
    except (UnknownSystemLinkageException, SystemLinkageException) as error:
        raise click.ClickException(str(error)) from error


@milk_pipes.command("nuke", context_settings={"help_option_names": ["-h", "--help"]})
@_folder_option
@click.argument("name")
def milk_pipes_nuke(folder: str | None, name: str):
    """
    Unconditionally tear down pipeline <name>, for test machines.

    \b
    Runs, without confirmation:
      1. runstop on all sessions
      2. deapply all system linkages
      3. confstop on all sessions
      4. rm -rf the pipeline root directory
    """
    milk_pipes_nuke_function(name, folder)


def milk_pipes_nuke_function(name: str, folder: str | None = None) -> None:
    Pipeline(folder if folder is not None else os.getcwd(), name).nuke()


def milk_pipes_init_function(name: str, folder: str | None = None) -> None:
    _run_tasks(name, folder, _get_tasks_configure())


def milk_pipes_spool_function(name: str, folder: str | None = None) -> None:
    _run_tasks(name, folder, _get_tasks_run())


def _run_tasks(
    name: str,
    folder: str | None,
    tasks: list[type[SimpleTask]],
) -> None:
    pp = Pipeline(
        folder if folder is not None else os.getcwd(), name
    )  # Raise on missing folder

    for Task in tasks:
        print(f"Executing task {Task.__name__}...")
        pp.task_do(Task)


def _get_tasks_configure() -> list[type[SimpleTask]]:
    return _parse_tasks(["InitialFolderSetup"])


def _get_tasks_run() -> list[type[SimpleTask]]:
    # TODO: "some sort of PRELOAD"
    return _parse_tasks(["DeployFPS", "StartConfProcesses", "LoadDataFiles"])


def _parse_tasks(task_names: list[str]) -> list[type[SimpleTask]]:
    from . import deploy_tasks as _d

    TASK_DB = {
        "InitialFolderSetup": _d.InitialFolderSetup,
        "LoadDataFiles": _d.LoadDataFiles,
        "DeployFPS": _d.DeployFPS,
        "StartConfProcesses": _d.StartConfProcesses,
    }
    return [TASK_DB[name] for name in task_names]


def browse_and_print_conf_folders(folder: str | Path):
    folder = Path(folder)
    if not os.path.isdir(folder):
        print(f"{folder} is not a directory.")
        return

    print(f"Valid configurations found in {folder}:")
    for sf in os.listdir(folder):
        subfolder = Path(sf)
        if (
            sf.endswith("-conf")
            and os.path.isdir(folder / subfolder)
            and os.path.isfile(folder / subfolder / "conf.toml")
        ):
            print(f" -    {subfolder}")
    print("---")
