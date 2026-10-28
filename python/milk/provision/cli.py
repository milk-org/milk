"""The ``milk-pipeinit`` console script (design-v7 A.5). Errors propagate."""

from __future__ import annotations

from pathlib import Path

import click
import tomli_w

from ..config import confresolve
from ..config.resolve import list_pipeline_dirs
from .clone import clone
from .hooks import consolidate

_dir = click.Path(exists=True, file_okay=False, path_type=Path)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def milk_pipeinit():
    """Clone, consolidate and resolve pipeline configuration folders (confdirs)."""


@milk_pipeinit.command("list")
@click.argument("folder", type=_dir, default=".")
def pipeinit_list(folder: Path):
    """List the confdirs and rootdirs in FOLDER (default: cwd)."""
    click.echo("\n".join(list_pipeline_dirs(folder)))


@milk_pipeinit.command("clone")
@click.argument("src", type=_dir)
@click.argument("dest_parent", type=click.Path(path_type=Path), default=".")
def pipeinit_clone(src: Path, dest_parent: Path):
    """Copy the conf files of confdir SRC into DEST_PARENT (default: cwd)."""
    click.echo(f"Cloned {src} to {clone(src, dest_parent)}")


@milk_pipeinit.command("consolidate")
@click.argument("confdir", type=_dir)
def pipeinit_consolidate(confdir: Path):
    """Run the pending conf hooks of CONFDIR; print the ids of those that ran."""
    for hook_id in consolidate(confdir):
        click.echo(hook_id)


@milk_pipeinit.command("resolve")
@click.argument("confdir", type=_dir)
def pipeinit_resolve(confdir: Path):
    """Print the resolved configuration of CONFDIR as TOML."""
    config = confresolve(confdir)
    click.echo(
        tomli_w.dumps(config.model_dump(mode="json", exclude_none=True)), nl=False
    )
