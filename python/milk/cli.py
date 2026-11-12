"""What the ``milk-pipes`` and ``milk-pipeinit`` console scripts share.

NB: Prefer lazy imports, this file is a net cost on the CLI startup time.
"""

from __future__ import annotations

import click


def _set_errstack(ctx: click.Context, param: click.Parameter, value: bool) -> None:
    ctx.meta["errstack"] = value


class ErrstackGroup(click.Group):
    """A group whose errors print as one message, without stack trace, unless --errstack.

    BestEffortError prints one warning per failure. Every error exits 1.
    ``milk`` log records print to stderr.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.params.append(
            click.Option(
                ["--errstack"],
                is_flag=True,
                expose_value=False,
                is_eager=True,
                callback=_set_errstack,
                help="Show the stack trace of errors.",
            )
        )

    def invoke(self, ctx: click.Context):
        import logging

        from .errors import BestEffortError, MilkError

        log = logging.getLogger("milk")
        handler = logging.StreamHandler()  # the current sys.stderr
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        try:
            return super().invoke(ctx)
        except (click.ClickException, click.exceptions.Exit, click.Abort):
            raise
        except Exception as error:
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
