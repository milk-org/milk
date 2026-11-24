"""The ``moka`` console script.

NB: Prefer lazy imports, this file is a net cost on the CLI startup time.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

from ..cli import ErrstackGroup

if TYPE_CHECKING:
    from ..errors import ConfigError
    from ..registry import Registry
    from .pipeline import Pipeline
    from .plans import Plan
    from .state import PipelineState


class _PlansGroup(ErrstackGroup):
    """The fixed commands, plus one command per plan of the pipeline of -p/--pipe (else cwd)."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        """Peek at -p/--pipe wherever it is, before help or command resolution need the plans."""
        peek = click.Command(
            None,
            params=[click.Option(["-p", "--pipe"])],
            add_help_option=False,
            context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
        )
        ctx.meta["moka.pipe"] = peek.make_context(
            "moka", list(args), resilient_parsing=True
        ).params["pipe"]
        return super().parse_args(ctx, args)

    def _plans(self, ctx: click.Context) -> Registry[Plan]:
        """The pipeline's plans; the built-in ones if it does not load, so help never fails."""
        if "moka.plans" not in ctx.meta:  # resolved once per invocation
            from .plans import PLANS

            try:
                ctx.meta["moka.plans"] = (_load(ctx).plans, None)
            except Exception as error:
                ctx.meta["moka.plans"] = (PLANS, error)
        plans = ctx.meta["moka.plans"][0]
        clash = [name for name in plans.names() if name in self.commands]
        if clash:
            raise click.ClickException(f"Plans {clash} clash with fixed commands.")
        return plans

    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted([*self.commands, *self._plans(ctx).names()])

    def get_command(self, ctx: click.Context, name: str) -> click.Command | None:
        plans = self._plans(ctx)
        if name in plans:
            return _plan_command(plans[name])
        error = ctx.meta["moka.plans"][1]
        if name not in self.commands and error is not None:
            raise error  # rather than "No such command"
        return self.commands.get(name)

    def format_commands(
        self, ctx: click.Context, formatter: click.HelpFormatter
    ) -> None:
        """Each command's short help, then its signature on a second line."""
        names = self.list_commands(ctx)
        limit = formatter.width - 6 - max(map(len, names))
        rows = []
        for name in names:
            cmd = self.get_command(ctx, name)
            rows += [
                (name, cmd.get_short_help_str(limit)),
                ("", f"     {_signature(ctx, cmd)}"),
            ]
        with formatter.section("Commands"):
            formatter.write_dl(rows)


def _signature(ctx: click.Context, cmd: click.Command) -> str:
    """``name ARG [-o METAVAR] [--flag]``, from the command's own parameters."""
    pieces = [cmd.name]
    for param in cmd.params:
        if isinstance(param, click.Argument):
            pieces.append(param.make_metavar(ctx))
        elif param.is_flag:
            pieces.append(f"[{param.opts[0]}]")
        else:
            pieces.append(f"[{param.opts[0]} {param.make_metavar(ctx)}]")
    return " ".join(pieces)


def _pipe(**kwargs):
    return click.option(
        "-p",
        "--pipe",
        default=None,
        metavar="PIPE",
        help="Rootdir path or short name; default: cwd.",
        **kwargs,
    )


_pipe_option = _pipe()


@click.group(cls=_PlansGroup, context_settings={"help_option_names": ["-h", "--help"]})
@_pipe(expose_value=False)  # read by _PlansGroup.parse_args
def moka():
    """
    Operate an initialized milk/cacao pipeline (see `mokainit init`).

    \b
    -p/--pipe PIPE is a rootdir path, or a short name: <cwd>/<PIPE>-rootdir.
    Without -p, cwd must be a rootdir. The commands include the plans of that pipeline.
    """


def _load(ctx: click.Context) -> Pipeline:
    """The pipeline of -p/--pipe (else cwd), loaded once per invocation."""
    meta = ctx.meta
    if "moka.pipeline" not in meta:
        from .pipeline import Pipeline

        pipe = meta.get("moka.pipe")
        meta["moka.pipeline"] = (
            Pipeline.from_folder(Path.cwd()) if pipe is None else Pipeline(pipe)
        )
    return meta["moka.pipeline"]


def _pipeline(pipe: str | None) -> Pipeline:
    """The invocation's pipeline, saying which rootdir it is."""
    ctx = click.get_current_context()
    ctx.meta.setdefault("moka.pipe", pipe)  # already set under _PlansGroup
    pp = _load(ctx)
    click.echo(f"rootdir: {pp.root_folder}")
    return pp


def _state(pipe: str | None) -> tuple[Pipeline, PipelineState]:
    """The invocation's pipeline and its state, printing `<short_name>: <state>`."""
    from .state import state

    pp = _pipeline(pipe)
    st = state(pp)
    click.echo(f"{pp.short_name}: {st.state.value}")
    return pp, st


def _na(error: ConfigError) -> str:
    return f"n/a ({error.message.split(';')[0].removesuffix('.')})"


def _plan_command(plan: Plan) -> click.Command:
    @click.command(plan.name, help=f"Run the {plan.name} plan.")
    @_pipe_option
    def command(pipe: str | None, compus: tuple[str, ...] = (), **values):
        from .plans import run_plan

        params = {k: v for k, v in values.items() if v is not None}
        run_plan(_pipeline(pipe), plan.name, compus or None, params)

    if plan.compus_arg:
        command = click.argument("compus", nargs=-1, metavar="[CU]...")(command)
    for p in plan.params:
        if p.positional:
            decl = click.argument(
                p.name, type=p.type, default=p.default, required=False
            )
        else:
            flags = (
                [f"--{p.name}"] if p.short is None else [f"-{p.short}", f"--{p.name}"]
            )
            decl = click.option(
                *flags, type=p.type, default=p.default, metavar=p.name.upper()
            )
        command = decl(command)
    return command


@moka.command("state")
@_pipe_option
def moka_state(pipe: str | None):
    """
    Show the state of the pipeline, why, and a summary of its drift from its confdir.
    """
    from ..errors import MilkError
    from ..provision.rootdir import drift

    pp, st = _state(pipe)
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
        f"drift: {n} artifact(s); run `mokainit drift {pp.root_folder}`"
        if n
        else "drift: none"
    )
    if report.moka_version:
        recorded, installed = report.moka_version
        click.echo(f"moka: rootdir made with {recorded}, installed {installed}")


@moka.command("plans")
@_pipe_option
def moka_plans(pipe: str | None):
    """
    List the plans of the pipeline, and whether the current state accepts each.
    """
    from ..errors import ConfigError, StateError
    from .plans import PLANS, build
    from .state import gate

    pp, st = _state(pipe)
    for name in pp.plans.names():
        plan = pp.plans[name]
        origin = ""
        if name not in PLANS or PLANS[name] is not plan:
            origin = " (local, overrides built-in)" if name in PLANS else " (local)"
        try:
            build(pp, plan)
            gate(pp.short_name, st, plan.accepts)
            status = "ok"
        except ConfigError as error:
            status = _na(error)
        except StateError as error:
            reason = error.message.split(" is ", 1)[-1].removesuffix(".")
            status = f"no ({reason})"
        labels = " ".join(step.label for step in plan.steps)
        click.echo(f"{name}{origin}  {labels}  {status}")


@moka.command("tasks")
@_pipe_option
@click.argument("plan", required=False)
def moka_tasks(pipe: str | None, plan: str | None):
    """
    List the tasks (of PLAN, if given) with their success and can; never runs them.
    """
    from ..errors import ConfigError
    from .plans import build_step, steps_from_list

    pp, _ = _state(pipe)
    steps = (
        pp.plans[plan].steps if plan is not None else steps_from_list(pp.tasks.names())
    )
    for step in steps:
        try:
            task = build_step(pp, step)
        except ConfigError as error:
            click.echo(f"{step.label}  {_na(error)}")
            continue
        done, can = task.success(), task.can()
        row = f"{step.label}  {'yes' if done else 'no'}  {'yes' if can else 'no'}"
        click.echo(row if can else f"{row}  {task.failure_reason}")


@moka.command("link")
@_pipe_option
@click.argument("linkage", required=False)
def moka_link(pipe: str | None, linkage: str | None):
    """
    Apply the [systemlinkage.<LINKAGE>] symlinks of the pipeline.

    Without LINKAGE, list the configured linkages.
    """
    pp = _pipeline(pipe)
    if linkage is None:
        for name in pp.config.systemlinkage:
            click.echo(name)
        return
    pp.link(linkage)


@moka.command("unlink")
@_pipe_option
@click.argument("linkage")
def moka_unlink(pipe: str | None, linkage: str):
    """
    Remove the [systemlinkage.<LINKAGE>] symlinks of the pipeline; `all`: of every linkage.
    """
    _pipeline(pipe).unlink(linkage)


@moka.command("nuke")
@_pipe_option
@click.option("--yes", is_flag=True, help="Do not ask for the short name.")
def moka_nuke(pipe: str | None, yes: bool):
    """
    teardown the pipeline, then rm -rf its rootdir, even if teardown fails.

    Asks to type the pipeline's short name first, unless --yes.
    """
    pp = _pipeline(pipe)
    if (
        not yes
        and click.prompt(f"Type {pp.short_name!r} to delete {pp.root_folder}")
        != pp.short_name
    ):
        raise click.ClickException("Short name mismatch; nothing deleted.")
    pp.nuke_unsafe()
