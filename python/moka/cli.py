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


class ErrstackGroup(click.Group):
    """A group whose errors print as one message, without stack trace, unless --errstack.

    --pdb opens an ipdb post-mortem session on the error instead.
    BestEffortError prints one warning per failure. Every error exits 1.
    ``moka`` log records print to stderr.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
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
        from .errors import BestEffortError, MilkError

        log = logging.getLogger("moka")
        handler = logging.StreamHandler()  # the current sys.stderr
        handler.setFormatter(ui.LogFormatter())
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
            if isinstance(error, BestEffortError):
                for label, failure in error.failures:
                    log.warning("%s: %s", label, failure)
                raise click.exceptions.Exit(1) from error
            if isinstance(error, MilkError):
                raise click.ClickException(str(error)) from error
            raise click.ClickException(
                f"{type(error).__name__}: {error}\n(--errstack for the stack trace)"
            ) from error
        finally:
            log.removeHandler(handler)
