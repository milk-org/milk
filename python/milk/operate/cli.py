"""The ``milk-pipes`` console script.

NB: Prefer lazy imports, this file is a net cost on the CLI startup time.
"""

from __future__ import annotations

import contextlib
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
    TARGET is a rootdir path, or a short name: <cwd>/<TARGET>-rootdir,
    or <DIR>/<TARGET>-rootdir with --parent DIR.
    """


_parent_option = click.option(
    "--parent",
    type=click.Path(file_okay=False),
    default=None,
    help="Folder holding <TARGET>-rootdir.",
)


def _pipeline(target: str, parent: str | None) -> Pipeline:
    from .pipeline import Pipeline

    return Pipeline(target) if parent is None else Pipeline.from_parent(parent, target)


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
    @_parent_option
    @click.argument("target")
    def command(parent: str | None, target: str):
        from .plans import run_plan

        with _user_errors():
            run_plan(_pipeline(target, parent), name)

    return command


@milk_pipes.command("state")
@_parent_option
@click.argument("target")
def milk_pipes_state(parent: str | None, target: str):
    """
    Show the state of pipeline TARGET, why, and a summary of its drift from its confdir.
    """
    from ..errors import MilkError
    from ..provision.rootdir import drift
    from .state import state

    with _user_errors():
        pp = _pipeline(target, parent)
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
@_parent_option
@click.argument("target")
@click.argument("linkage")
def milk_pipes_link(parent: str | None, target: str, linkage: str):
    """
    Apply the [systemlinkage.<LINKAGE>] symlinks of pipeline TARGET.
    """
    with _user_errors():
        _pipeline(target, parent).link(linkage)


@milk_pipes.command("unlink")
@_parent_option
@click.argument("target")
@click.argument("linkage")
def milk_pipes_unlink(parent: str | None, target: str, linkage: str):
    """
    Remove the [systemlinkage.<LINKAGE>] symlinks of pipeline TARGET.
    """
    with _user_errors():
        _pipeline(target, parent).unlink(linkage)


@milk_pipes.command("nuke")
@_parent_option
@click.option("--yes", is_flag=True, help="Do not ask for the short name.")
@click.argument("target")
def milk_pipes_nuke(parent: str | None, yes: bool, target: str):
    """
    teardown pipeline TARGET, then rm -rf its rootdir, even if teardown fails.

    Asks to type the pipeline's short name first, unless --yes.
    """
    with _user_errors():
        pp = _pipeline(target, parent)
        if (
            not yes
            and click.prompt(f"Type {pp.short_name!r} to delete {pp.root_folder}")
            != pp.short_name
        ):
            raise click.ClickException("Short name mismatch; nothing deleted.")
        pp.nuke_unsafe()
