"""The ``milk-show`` console script. matplotlib is imported lazily."""

from __future__ import annotations

from pathlib import Path

import click

from ..operate.cli import _pipe_option, _pipeline


def mlat_figure(datadir: Path | str):
    """Figure of ``hardwlatency.dat`` (value against dt [us]); 'r' re-reads the file."""
    import matplotlib.pyplot as plt
    import numpy as np

    path = Path(datadir) / "hardwlatency.dat"
    fig, ax = plt.subplots()
    (line,) = ax.plot([], [], ".")
    ax.set(xlabel="dt [µs]", ylabel="value", title=str(path))

    def redraw() -> None:
        _, dt, value = np.loadtxt(path, comments="#", unpack=True)
        line.set_data(dt, value)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()

    def on_key(event) -> None:
        if event.key == "r":
            redraw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    return fig


@click.group()
def milk_show() -> None:
    """Display milk results."""


@milk_show.command()
@_pipe_option
@click.option(
    "-w",
    "--workdir",
    type=click.Path(exists=True, file_okay=False),
    help="Work folder holding the FPS datadirs; default: <rootdir>/work.",
)
def mlat(pipe: str | None, workdir: str | None) -> None:
    """Plot the latency graph of the pipeline's mlat (press r to redraw)."""
    import matplotlib.pyplot as plt

    pp = _pipeline(pipe)
    work = Path(workdir) if workdir else pp.work_folder
    datadir = work / f"fps.{pp.config.fps_name('mlat')}.datadir"
    if not (datadir / "hardwlatency.dat").is_file():
        raise click.ClickException(f"No hardwlatency.dat in {datadir}.")
    mlat_figure(datadir)
    plt.show()
