"""The ``milk-pipeinit`` console script (design-v7 A.5). Errors propagate."""

from __future__ import annotations

from pathlib import Path

import click
import tomli
import tomli_w

from ..config import confresolve
from ..config.layout import list_pipeline_dirs
from ..pipeline import Pipeline
from .clone import clone
from .hooks import consolidate
from .rootdir import drift, init, resync

_dir = click.Path(exists=True, file_okay=False, path_type=Path)


def _parse_set(ctx, param, values: tuple[str, ...]) -> dict[str, object]:
    """K=V pairs; V is a TOML value when it parses, else the raw string."""
    out = {}
    for item in values:
        key, _, raw = item.partition("=")
        try:
            out[key] = tomli.loads(f"v = {raw}")["v"]
        except tomli.TOMLDecodeError:
            out[key] = raw
    return out


_set_option = click.option(
    "--set", "overrides", multiple=True, callback=_parse_set, help="Override K=V."
)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def milk_pipeinit():
    """Clone, consolidate, resolve and init confdirs; drift and resync rootdirs."""


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
@click.option("--refetch", multiple=True, help="Rerun hook ID even if resolved.")
@click.option("--refetch-all", is_flag=True, help="Rerun every hook.")
def pipeinit_consolidate(confdir: Path, refetch: tuple[str, ...], refetch_all: bool):
    """Run the pending conf hooks of CONFDIR; print the ids of those that ran."""
    for hook_id in consolidate(confdir, "all" if refetch_all else refetch):
        click.echo(hook_id)


@milk_pipeinit.command("resolve")
@click.argument("confdir", type=_dir)
def pipeinit_resolve(confdir: Path):
    """Print the resolved configuration of CONFDIR (or a rootdir) as TOML."""
    config = confresolve(confdir)
    click.echo(
        tomli_w.dumps(config.model_dump(mode="json", exclude_none=True)), nl=False
    )


@milk_pipeinit.command("init")
@click.argument("confdir", type=_dir)
@click.option(
    "--parent", type=click.Path(path_type=Path), help="Default: CONFDIR's parent."
)
@_set_option
def pipeinit_init(confdir: Path, parent: Path | None, overrides: dict):
    """Consolidate CONFDIR and copy it into a new <name>-rootdir."""
    click.echo(f"Initialized {init(confdir, parent, overrides)}")


def _echo_diff(resolved_diff: dict, structural: list[str]) -> None:
    for key, (old, new) in resolved_diff.items():
        click.echo(f"  {key}: {old!r} -> {new!r}")
    for line in structural:
        click.echo(f"STRUCTURAL: {line}")


@milk_pipeinit.command("drift")
@click.argument("rootdir", type=_dir)
def pipeinit_drift(rootdir: Path):
    """Print what differs between ROOTDIR, its records and its confdir."""
    report = drift(rootdir)
    for row in report.artifacts:
        if (row.source, row.local) != ("unchanged", "unchanged"):
            click.echo(f"{row.key}: source {row.source}, local {row.local}")
    if report.milk_version:
        click.echo("milk version: {} -> {}".format(*report.milk_version))
    _echo_diff(report.resolved_diff, report.structural)


@milk_pipeinit.command("resync")
@click.argument("rootdir", type=_dir)
@click.option("--artifact", "artifacts", multiple=True, help="Only this key.")
@_set_option
@click.option(
    "--overwrite-local", is_flag=True, help="Replace locally edited artifacts."
)
def pipeinit_resync(
    rootdir: Path, artifacts: tuple[str, ...], overrides: dict, overwrite_local: bool
):
    """Bring the confdir's changes into ROOTDIR. Never deploys."""
    report = resync(
        rootdir,
        run_active=lambda: Pipeline(rootdir).any_run_active(),
        artifacts=artifacts or None,
        overrides=overrides,
        overwrite_local=overwrite_local,
    )
    for key in report.applied:
        click.echo(f"applied {key}")
    for key, reason in report.refused.items():
        click.echo(f"refused {key}: {reason}")
    click.echo("Resynced: teardown → deploy may be needed.")
    _echo_diff(report.resolved_diff, report.structural)
