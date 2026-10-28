from __future__ import annotations

from ..pipeline import Pipeline
from ..errors import BestEffortError, LinkageError

import click


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def milk_pipes():
    """
    Spool up and tear down an initialized milk/cacao pipeline (see `milk-pipeinit init`).

    \b
    TARGET is a rootdir path, or a short name: <cwd>/<TARGET>-rootdir,
    or <DIR>/<TARGET>-rootdir with --parent DIR.

    \b
    Subcommands:
      spool     Deploy and start the processes of a pipeline
      sys       Apply a [systemlinkage] mode of a pipeline
      teardown  Stop, unlink and destroy the FPSs of a pipeline; keep its rootdir
      nuke      teardown, then delete the rootdir (destructive)
    """


_parent_option = click.option(
    "--parent",
    type=click.Path(file_okay=False),
    default=None,
    help="Folder holding <TARGET>-rootdir.",
)


def _pipeline(target: str, parent: str | None) -> Pipeline:
    return Pipeline(target) if parent is None else Pipeline.from_parent(parent, target)


def _warn_failures(fn) -> None:
    try:
        fn()
    except BestEffortError as error:
        for label, failure in error.failures:
            click.echo(f"{label}\n  warning: {failure}")


@milk_pipes.command("spool", context_settings={"help_option_names": ["-h", "--help"]})
@_parent_option
@click.argument("target")
def milk_pipes_spool(parent: str | None, target: str):
    """
    Deploy FPSs, start conf processes and load data files of pipeline TARGET.
    """
    from . import deploy_tasks as _d

    pp = _pipeline(target, parent)
    for Task in (_d.DeployFPS, _d.StartConfProcesses, _d.LoadDataFiles):
        print(f"Executing task {Task.__name__}...")
        pp.task_do(Task)


@milk_pipes.command("sys", context_settings={"help_option_names": ["-h", "--help"]})
@_parent_option
@click.argument("target")
@click.argument("linkage")
def milk_pipes_sys(parent: str | None, target: str, linkage: str):
    """
    Apply the [systemlinkage.<LINKAGE>] symlinks of pipeline TARGET.
    """
    try:
        _pipeline(target, parent).apply_system_linkage(linkage)
    except LinkageError as error:
        raise click.ClickException(str(error)) from error


@milk_pipes.command(
    "teardown", context_settings={"help_option_names": ["-h", "--help"]}
)
@_parent_option
@click.argument("target")
def milk_pipes_teardown(parent: str | None, target: str):
    """
    Best-effort runstop, deapply linkages, confstop and destroy FPSs of pipeline TARGET.

    The rootdir and its work/ folder are kept.
    """
    _warn_failures(_pipeline(target, parent).teardown)


@milk_pipes.command("nuke", context_settings={"help_option_names": ["-h", "--help"]})
@_parent_option
@click.option("--yes", is_flag=True, help="Do not ask for the short name.")
@click.argument("target")
def milk_pipes_nuke(parent: str | None, yes: bool, target: str):
    """
    teardown pipeline TARGET, then rm -rf its rootdir.

    Asks to type the pipeline's short name first, unless --yes.
    """
    pp = _pipeline(target, parent)
    if (
        not yes
        and click.prompt(f"Type {pp.short_name!r} to delete {pp.root_folder}")
        != pp.short_name
    ):
        raise click.ClickException("Short name mismatch; nothing deleted.")
    _warn_failures(pp.nuke_unsafe)
