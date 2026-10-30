"""The ``milk-pipes`` console script.

NB: Prefer lazy imports, this file is a net cost on the CLI startup time.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from .pipeline import Pipeline


class _PlansGroup(click.Group):
    """The fixed commands, plus one command per PLANS name, read at call time."""

    def _plans(self) -> list[str]:
        from .plans import PLANS

        clash = [name for name in PLANS.names() if name in self.commands]
        if clash:
            raise click.ClickException(f"Plans {clash} clash with fixed commands.")
        return PLANS.names()

    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted([*self.commands, *self._plans()])

    def get_command(self, ctx: click.Context, name: str) -> click.Command | None:
        if name in self._plans():
            return _plan_command(name)
        return self.commands.get(name)


@click.group(cls=_PlansGroup, context_settings={"help_option_names": ["-h", "--help"]})
def milk_pipes():
    """
    Operate an initialized milk/cacao pipeline (see `milk-pipeinit init`).

    \b
    -p/--pipe PIPE is a rootdir path, or a short name: <cwd>/<PIPE>-rootdir.
    Without -p, cwd must be a rootdir.
    """


_pipe_option = click.option(
    "-p", "--pipe", default=None, help="Rootdir path or short name; default: cwd."
)


def _pipeline(pipe: str | None) -> Pipeline:
    """Resolve -p/--pipe to a Pipeline and say which rootdir it is."""
    from .pipeline import Pipeline

    pp = Pipeline.from_folder(Path.cwd()) if pipe is None else Pipeline(pipe)
    click.echo(f"rootdir: {pp.root_folder}")
    return pp


@contextlib.contextmanager
def _user_errors():
    """BestEffortError: one warning per failure, exit 1. Other MilkErrors: a ClickException."""
    from ..errors import BestEffortError, MilkError

    try:
        yield
    except BestEffortError as error:
        for label, failure in error.failures:
            click.echo(f"warning: {label}: {failure}")
        raise click.exceptions.Exit(1) from error
    except MilkError as error:
        raise click.ClickException(str(error)) from error


def _plan_command(name: str) -> click.Command:
    @click.command(name, help=f"Run the {name} plan.")
    @_pipe_option
    def command(pipe: str | None):
        from .plans import run_plan

        with _user_errors():
            run_plan(_pipeline(pipe), name)

    return command


@milk_pipes.command("state")
@_pipe_option
def milk_pipes_state(pipe: str | None):
    """
    Show the state of the pipeline, why, and a summary of its drift from its confdir.
    """
    from ..errors import MilkError
    from ..provision.rootdir import drift
    from .state import state

    with _user_errors():
        pp = _pipeline(pipe)
        st = state(pp)
    click.echo(f"{pp.short_name}: {st.state.value}")
    for reason in st.reasons:
        click.echo(f"  reason: {reason}")
    try:
        report = drift(pp.root_folder)
    except MilkError as error:
        click.echo(f"drift: unavailable ({error.message})")
        return
    n = sum(
        row.source != "unchanged" or row.local != "unchanged"
        for row in report.artifacts
    )
    click.echo(
        f"drift: {n} artifact(s); run `milk-pipeinit drift {pp.root_folder}`"
        if n
        else "drift: none"
    )
    if report.milk_version:
        recorded, installed = report.milk_version
        click.echo(f"milk: rootdir made with {recorded}, installed {installed}")


@milk_pipes.command("link")
@_pipe_option
@click.argument("linkage")
def milk_pipes_link(pipe: str | None, linkage: str):
    """
    Apply the [systemlinkage.<LINKAGE>] symlinks of the pipeline.
    """
    with _user_errors():
        _pipeline(pipe).link(linkage)


@milk_pipes.command("unlink")
@_pipe_option
@click.argument("linkage")
def milk_pipes_unlink(pipe: str | None, linkage: str):
    """
    Remove the [systemlinkage.<LINKAGE>] symlinks of the pipeline.
    """
    with _user_errors():
        _pipeline(pipe).unlink(linkage)


@milk_pipes.command("nuke")
@_pipe_option
@click.option("--yes", is_flag=True, help="Do not ask for the short name.")
def milk_pipes_nuke(pipe: str | None, yes: bool):
    """
    teardown the pipeline, then rm -rf its rootdir, even if teardown fails.

    Asks to type the pipeline's short name first, unless --yes.
    """
    with _user_errors():
        pp = _pipeline(pipe)
        if (
            not yes
            and click.prompt(f"Type {pp.short_name!r} to delete {pp.root_folder}")
            != pp.short_name
        ):
            raise click.ClickException("Short name mismatch; nothing deleted.")
        pp.nuke_unsafe()
