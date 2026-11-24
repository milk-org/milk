"""The ``mokashow`` console script. matplotlib is imported lazily."""

from __future__ import annotations

from pathlib import Path

import click

from ..calibs.mlat import mlat_results
from ..operate.cli import _pipe_option, _pipeline


def _mlat_kernel(t, knee, width, frame):
    """1-frame rect window convolved with the derivative of a unit-height sigmoid.

    That is the sigmoid rising at ``knee`` minus the same one falling a frame later.
    The logistic's slope at its knee matches a linear ramp ``width`` long, so
    width -> 0 gives a 1-frame rect starting at ``knee``, and width = 1 frame
    is close to the triangle of base 2 frames peaking at knee + frame/2.
    """
    import numpy as np

    def sigmoid(edge):
        return 0.5 * (1.0 + np.tanh(2.0 * (t - edge) / width))

    return sigmoid(knee) - sigmoid(knee + frame)


def _grid_fit(model, t, y, knees, widths):
    """(knee, width) minimizing sum((y - model(t, knee, width))**2) over the grid."""
    best_sse, best = float("inf"), None
    for w in widths:
        sse = ((y - model(t[None, :], knees[:, None], w)) ** 2).sum(axis=1)
        i = int(sse.argmin())
        if sse[i] < best_sse:
            best_sse, best = sse[i], (float(knees[i]), float(w))
    return best


def mlat_figure(datadir: Path | str):
    """Figure of ``hardwlatency.dat`` (value against dt [us]); 'r' re-reads the files.

    When mlat's ``cacaovars.bash`` is present, overlays the baseline (median of the
    dt < 0 points), a triangle one frame wide on each side peaking at the measured
    latency, a vertical bar halfway up its rising side, and the best-fit kernel
    (see ``_mlat_kernel``) with the triangle's height.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    datadir = Path(datadir)
    path = datadir / "hardwlatency.dat"
    fig, ax = plt.subplots()
    (line,) = ax.plot([], [], ".")
    (baseline,) = ax.plot([], [], "-", color="gray", label="baseline")
    (triangle,) = ax.plot([], [], "-", color="C3", label="latency")
    (kernel,) = ax.plot([], [], "-", color="C1", label="best kernel")
    halfway = ax.axvline(np.nan, color="C2", ls="--", label="mid rising edge")
    text = ax.text(0.98, 0.98, "", transform=ax.transAxes, ha="right", va="top")
    ax.set(xlabel="dt [µs]", ylabel="value", title=str(path))

    def redraw() -> None:
        _, dt, value = np.loadtxt(path, comments="#", unpack=True)
        line.set_data(dt, value)
        res = mlat_results(datadir)
        if res is None:
            for artist in (baseline, triangle, kernel):
                artist.set_data([], [])
            halfway.set_xdata([np.nan])
            text.set_text("")
        else:
            fps, lat_fr = res
            frame = 1e6 / fps  # [us]
            lat = lat_fr * frame  # [us]
            base = float(np.median(value[dt < 0]))
            # Peak height as mlat estimates it: mean of the middle third of the
            # sorted values within half a frame of the peak.
            near = np.sort(value[np.abs(dt - lat) <= 0.5 * frame])
            n = len(near)
            peak = (
                float(near[n // 3 : max(2 * n // 3, n // 3 + 1)].mean()) if n else base
            )
            baseline.set_data([dt.min(), dt.max()], [base, base])
            triangle.set_data([lat - frame, lat, lat + frame], [base, peak, base])
            halfway.set_xdata([lat - 0.5 * frame])
            height = peak - base

            def model(t, knee, width):
                return base + height * _mlat_kernel(t, knee, width, frame)

            # scipy.optimize.curve_fit would do better; coarse-then-fine grid
            # search instead, to avoid the scipy dependency for now.
            knee, width = _grid_fit(
                model,
                dt,
                value,
                np.linspace(lat - frame, lat, 201),
                np.geomspace(1e-3, 2.0, 60) * frame,
            )
            knee, width = _grid_fit(
                model,
                dt,
                value,
                np.linspace(knee - 0.01 * frame, knee + 0.01 * frame, 101),
                np.geomspace(width / 1.2, width * 1.2, 41),
            )
            tk = np.linspace(knee - 1.5 * frame, knee + 2.5 * frame, 400)
            kernel.set_data(tk, model(tk, knee, width))
            text.set_text(
                f"latency = {lat_fr:.3f} frame = {lat:.1f} µs\nframerate = {fps:.1f} Hz"
                f"\nkernel knee = {knee:.1f} µs, width = {width:.1f} µs"
            )
            ax.legend(loc="upper left")
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
def mokashow() -> None:
    """Display milk results."""


@mokashow.command()
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


def _modes(kind: str, pipe: str | None) -> None:
    import matplotlib.pyplot as plt

    from .modes import CUBES, modes_figure

    calib = _pipeline(pipe).root_folder / "calib"
    missing = [n for n in CUBES[kind] if not (calib / n).is_file()]
    if missing:
        raise click.ClickException(f"No {', '.join(missing)} in {calib}.")
    modes_figure(calib, kind)
    plt.show()


@mokashow.command()
@_pipe_option
def rm(pipe: str | None) -> None:
    """Browse calib/RMmodes{WFS,DM}.fits (r: reload, w/d: WFS/DM mask)."""
    _modes("rm", pipe)


@mokashow.command()
@_pipe_option
def cm(pipe: str | None) -> None:
    """Browse calib/CMmodes{WFS,DM}.fits and their singular values (r: reload, w/d: masks)."""
    _modes("cm", pipe)
