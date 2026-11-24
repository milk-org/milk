"""The ``moka`` console script.

NB: Prefer lazy imports, this file is a net cost on the CLI startup time.
"""

from __future__ import annotations

import sys
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
    from .tasks import Task


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
        error = ctx.meta["moka.plans"][1]
        if name in plans:
            if error is None:
                tasks = _load(ctx).tasks
            else:
                from .tasks import TASKS as tasks
            return _plan_command(plans[name], tasks)
        if name not in self.commands and error is not None:
            raise error  # rather than "No such command"
        return self.commands.get(name)

    def format_commands(
        self, ctx: click.Context, formatter: click.HelpFormatter
    ) -> None:
        """Each command's short help, then its signature on a second line."""
        from .. import ui

        color = ui.use_color(sys.stdout)
        names = self.list_commands(ctx)
        limit = formatter.width - 6 - max(map(len, names))
        rows = []
        for name in names:
            cmd = self.get_command(ctx, name)
            assert cmd is not None
            signature = ui.style("dim", _signature(ctx, cmd), color)
            rows += [
                (name, cmd.get_short_help_str(limit)),
                ("", f"     {signature}"),
            ]
        with formatter.section("Commands"):
            formatter.write_dl(rows)


def _signature(ctx: click.Context, cmd: click.Command) -> str:
    """``name ARG [-o METAVAR] [--flag]``, from the command's own parameters."""
    assert cmd.name is not None
    pieces = [cmd.name]
    for param in cmd.params:
        if isinstance(param, click.Argument):
            pieces.append(param.make_metavar(ctx))
        elif isinstance(param, click.Option) and param.is_flag:
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


def _echo_table(rows: list) -> None:
    """Print aligned rows, cut to the terminal width only when stdout is one."""
    import shutil

    from .. import ui

    width = shutil.get_terminal_size().columns if sys.stdout.isatty() else None
    for line in ui.table(rows, max_width=width):
        click.echo(line)


def _na(error: ConfigError) -> str:
    return f"n/a ({error.message.split(';')[0].removesuffix('.')})"


_ACCEPTS = {
    "any": "Runs in any state.",
    "consistent": "Needs a deployed pipeline.",
    "cold_or_consistent": "Runs on a cold or a deployed pipeline.",
}
_ON_FAILURE = {
    "abort": "Stops at the first failure.",
    "best_effort": "Runs every step, then reports the failures.",
}


def _epilog(plan: Plan, tasks: Registry[type[Task]]) -> str:
    """What the usage line cannot say: the positional params, the steps, the gate; no rewrap."""
    blocks = []
    for p in (p for p in plan.params if p.positional):
        name = p.name.upper()
        line = f"{name} is one of {', '.join(p.choices)}" if p.choices else f"{name}: {p.help}"
        blocks.append(line + (f" (default: {p.default})." if p.default is not None else "."))
    steps = ["Steps:"]
    for i, step in enumerate(plan.steps, 1):
        cls = tasks[step.task] if step.task in tasks else None
        doc = (cls.__dict__.get("__doc__") or "").strip().split("\n")[0] if cls else ""
        steps.append(f"  {i}. {step.label}  {doc}".rstrip())
    sentence = f"{_ACCEPTS[plan.accepts]} {_ON_FAILURE[plan.on_failure]}"
    return "\n\n".join(f"\b\n{b}" for b in ["\n".join(blocks), "\n".join(steps), sentence] if b)


def _plan_command(plan: Plan, tasks: Registry[type[Task]]) -> click.Command:
    @click.command(
        plan.name,
        help=plan.help or f"Run the {plan.name} plan.",
        epilog=_epilog(plan, tasks),
    )
    @_pipe_option
    def command(pipe: str | None, compus: tuple[str, ...] = (), **values):
        from .plans import run_plan

        params = {k: v for k, v in values.items() if v is not None}
        run_plan(_pipeline(pipe), plan.name, compus or None, params)

    if plan.compus_arg:
        command = click.argument("compus", nargs=-1, metavar="[CU]...")(command)
    for p in plan.params:
        kind = click.Choice(p.choices) if p.choices else p.type
        if p.positional:
            decl = click.argument(
                p.name,
                type=kind,
                default=p.default,
                required=False,
                metavar=f"[{p.name.upper()}]",  # not Choice's [a|b]
            )
        else:
            flags = (
                [f"--{p.name}"] if p.short is None else [f"-{p.short}", f"--{p.name}"]
            )
            decl = click.option(
                *flags,
                type=kind,
                default=p.default,
                metavar=p.name.upper(),
                help=p.help,
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
    from .. import ui
    from ..errors import ConfigError, StateError
    from .plans import PLANS, build
    from .state import gate

    pp, st = _state(pipe)
    ctx = click.get_current_context()
    color = ui.use_color(sys.stdout)
    rows = [[ui.style("dim", "USAGE", color), ui.style("dim", "NOW", color)]]
    for name in pp.plans.names():
        plan = pp.plans[name]
        origin = ""
        if name not in PLANS or PLANS[name] is not plan:
            origin = "  (local, overrides built-in)" if name in PLANS else "  (local)"
        try:
            build(pp, plan)
            gate(pp.short_name, st, plan.accepts)
            status, role = "ok", "ok"
        except ConfigError as error:
            status, role = _na(error), "dim"
        except StateError as error:
            reason = error.message.split(" is ", 1)[-1].removesuffix(".")
            status, role = f"no ({reason})", "warn"
        usage = _signature(ctx, _plan_command(plan, pp.tasks))
        labels = " -> ".join(step.label for step in plan.steps)
        rows += [
            [
                ui.style("name", name, color) + usage[len(name) :] + origin,
                ui.style(role, status, color),
            ],
            [ui.style("dim", f"  {labels}", color)],
        ]
    _echo_table(rows)


@moka.command("tasks")
@_pipe_option
@click.argument("plan", required=False)
def moka_tasks(pipe: str | None, plan: str | None):
    """
    List the tasks (of PLAN, if given) with their success and can; never runs them.
    """
    from .. import ui
    from ..errors import ConfigError
    from .plans import build_step, steps_from_list

    pp, _ = _state(pipe)
    steps = (
        pp.plans[plan].steps if plan is not None else steps_from_list(pp.tasks.names())
    )
    color = ui.use_color(sys.stdout)
    rows = [[ui.style("dim", h, color) for h in ("TASK", "DONE", "CAN", "REASON")]]
    for step in steps:
        label = ui.style("name", step.label, color)
        try:
            task = build_step(pp, step)
        except ConfigError as error:
            rows.append([label, ui.style("dim", _na(error), color)])
            continue
        done, can = task.success(), task.can()
        row = [label]
        for flag in (done, can):
            row.append(ui.style("ok" if flag else "warn", "yes" if flag else "no", color))
        rows.append(row if can else [*row, task.failure_reason])
    _echo_table(rows)


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
