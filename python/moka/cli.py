"""What the ``moka`` and ``mokainit`` console scripts share.

NB: Prefer lazy imports, this file is a net cost on the CLI startup time.
"""

from __future__ import annotations

import click


def _set_meta(ctx: click.Context, param: click.Parameter, value: bool) -> None:
    assert param.name is not None
    ctx.meta[param.name] = value


def _set_color(ctx: click.Context, param: click.Parameter, value: str) -> None:
    if value != "auto":  # auto: leave what the caller of main() set, usually None
        ctx.color = value == "always"


# shell -> click.shell_completion class. To add zsh or fish, add "zsh": "ZshComplete" or
# "fish": "FishComplete" (both ship with click); nothing else changes.
SHELLS = {"bash": "BashComplete"}


@click.command("completion")
@click.argument("shell", type=click.Choice(list(SHELLS)), metavar="SHELL")
def _completion(shell: str) -> None:
    """Print this command's tab completion script for SHELL, to eval in ~/.bashrc."""
    from click import shell_completion

    root = click.get_current_context().find_root()
    prog = root.info_name  # how main() names it, so the same as its env var below
    var = f"_{prog}_COMPLETE".replace("-", "_").upper()  # click's own rule, in main()
    complete = getattr(shell_completion, SHELLS[shell])(root.command, {}, prog, var)
    click.echo(complete.source())


class ErrstackGroup(click.Group):
    """A group whose errors print as one message, without stack trace, unless --errstack.

    --pdb opens an ipdb post-mortem session on the error instead.
    A MilkError prints as ``[FAIL] <message>``, unless a plan run already told it. Every error exits 1.
    ``moka`` log records print to stderr (see ui.PlanHandler).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.add_command(_completion)
        for flag, help in (
            ("--errstack", "Show the stack trace of errors."),
            ("--pdb", "Open an ipdb post-mortem session on errors."),
        ):
            self.params.append(
                click.Option(
                    [flag],
                    is_flag=True,
                    expose_value=False,
                    is_eager=True,
                    callback=_set_meta,
                    help=help,
                )
            )
        self.params.append(
            click.Option(
                ["--color"],
                type=click.Choice(["auto", "always", "never"]),
                default="auto",
                expose_value=False,
                is_eager=True,
                callback=_set_color,
                help="Colors: auto (a terminal and no NO_COLOR), always or never.",
            )
        )

    def invoke(self, ctx: click.Context):
        import logging

        from . import ui
        from .errors import BestEffortError, MilkError, TaskError

        log = logging.getLogger("moka")
        handler = ui.PlanHandler()
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        try:
            return super().invoke(ctx)
        except (click.ClickException, click.exceptions.Exit, click.Abort):
            raise
        except Exception as error:
            if ctx.meta.get("pdb"):
                import ipdb

                ipdb.post_mortem(error.__traceback__)
                raise click.exceptions.Exit(1) from error
            if ctx.meta.get("errstack"):
                raise
            if not isinstance(error, MilkError):
                raise click.ClickException(
                    f"{type(error).__name__}: {error}\n(--errstack for the stack trace)"
                ) from error
            # a plan's failure was already told step by step, then by its verdict line
            if not (handler.failed_plan and isinstance(error, (TaskError, BestEffortError))):
                color = ui.use_color(handler.stream)
                first, *context = str(error).split("\n")
                click.echo(f"{ui.tag(ui.FAILED, color)} {ui.highlight_ticks(first, color)}", err=True)
                for line in context:
                    click.echo(ui.style("dim", line, color), err=True)
            raise click.exceptions.Exit(1) from error
        finally:
            log.removeHandler(handler)
            handler.close()
