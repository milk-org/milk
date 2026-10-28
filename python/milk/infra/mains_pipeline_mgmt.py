from __future__ import annotations
import typing as typ

from ..config.resolve import list_pipeline_dirs
from ..pipeline import Pipeline
from ..errors import BestEffortError, LinkageError

import os
import click

if typ.TYPE_CHECKING:
    from .task_models import SimpleTask


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def milk_pipes():
    """
    Initialize and spool up a milk/cacao pipeline.

    \b
    Subcommands:
      init    Set up the folders of a cloned pipeline
      spool   Deploy and start the processes of an initialized pipeline
      sys     Apply a [systemlinkage] mode of a pipeline
      nuke    Stop, unlink and delete the root directory of a pipeline (destructive)
    """


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
        click.echo("\n".join(list_pipeline_dirs(folder or os.getcwd())))
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
        click.echo("\n".join(list_pipeline_dirs(folder or os.getcwd())))
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
    except LinkageError as error:
        raise click.ClickException(str(error)) from error


@milk_pipes.command("nuke", context_settings={"help_option_names": ["-h", "--help"]})
@_folder_option
@click.argument("name")
def milk_pipes_nuke(folder: str | None, name: str):
    """
    Unconditionally tear down pipeline <name>, for test machines.

    \b
    Runs, without confirmation:
      1. runstop on all compute units (CompUs)
      2. deapply all system linkages
      3. confstop on all compute units (CompUs)
      4. destroy the FPSs of all compute units (CompUs)
      5. rm -rf the pipeline root directory # TODO TBC ? Maybe we want a slightly less aggressive teardown.
    """
    milk_pipes_nuke_function(name, folder)


def milk_pipes_nuke_function(name: str, folder: str | None = None) -> None:
    pp = Pipeline(folder if folder is not None else os.getcwd(), name)
    try:
        pp.nuke()
    except BestEffortError as error:
        for label, failure in error.failures:
            click.echo(f"{label}\n  warning: {failure}")


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
