"""``mokashow rm`` / ``mokashow cm``: browse the mode cubes of rootdir/calib.

matplotlib is imported lazily.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from ..calibs.cm import CM_DM, CM_WFS
from ..calibs.linresp import DONE, POKES

# kind -> (WFS cube, DM cube), in rootdir/calib
CUBES = {"rm": (DONE, POKES), "cm": (CM_WFS, CM_DM)}
MASKS = ("wfsmask.fits", "dmmask.fits")  # optional, in rootdir/calib
MASK_KEYS = ("w", "d")  # toggle the WFS / DM mask


def stats(frame, mask=None) -> str:
    """RMS, P2V, mean and std of ``frame`` over ``mask != 0``, or all pixels."""
    import numpy as np

    x = frame[mask != 0] if mask is not None else frame.ravel()
    if x.size == 0:
        return "empty mask"
    return (
        f"rms {np.sqrt(np.mean(x**2)):.3g}  p2v {np.ptp(x):.3g}\n"
        f"mean {x.mean():.3g}  std {x.std():.3g}"
    )


def _cm_title(header) -> str:
    ncut = header.get("CMNCUT")
    return (
        f"{header.get('CMFLAVOR', '?')}  -r {header.get('CMREG', '?')}"
        f"  -n {'-' if ncut is None else ncut}  kept {header.get('CMNKEPT', '?')}"
    )


def modes_figure(calib: Path | str, kind: str):
    """WFS and DM panels of one mode of the ``kind`` ("rm" or "cm") pair in ``calib``.

    A slider picks the mode; Play steps through them at the fps of the box. Each panel's
    stats run over its mask (``MASKS``, toggled by its check box or ``MASK_KEYS``) when
    that file exists, else over all pixels. For "cm", a third panel plots the singular
    values with the flavor, -r, -n and kept count. 'r' re-reads the files.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.io import fits
    from matplotlib.widgets import Button, CheckButtons, Slider, TextBox

    calib = Path(calib)
    names = CUBES[kind]
    fig = plt.figure(figsize=(13 if kind == "cm" else 9, 5.5))
    grid = fig.add_gridspec(1, 3 if kind == "cm" else 2, bottom=0.3, wspace=0.5)
    axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
    images = [ax.imshow(np.zeros((1, 1)), cmap="RdBu_r", origin="lower") for ax in axes]
    if kind == "cm":
        ax_sv = fig.add_subplot(grid[0, 2])
        (sv_line,) = ax_sv.semilogy([], [], ".-", ms=3)
        sv_reg = ax_sv.axhline(np.nan, color="C3", ls="--", label="-r")
        sv_ncut = ax_sv.axvline(np.nan, color="C2", ls="--", label="-n")
        ax_sv.set(xlabel="singular mode", ylabel="σ / σ0")
        ax_sv.legend(loc="lower left")
    for im in images:
        fig.colorbar(im, ax=im.axes, fraction=0.046)

    slider = Slider(
        fig.add_axes((0.08, 0.15, 0.55, 0.04)), "mode", 0, 1, valinit=0, valstep=1
    )
    play = Button(fig.add_axes((0.70, 0.14, 0.08, 0.06)), "Play")
    fps = TextBox(fig.add_axes((0.86, 0.14, 0.06, 0.06)), "fps ", initial="10")
    check = CheckButtons(fig.add_axes((0.08, 0.02, 0.3, 0.09)), list(MASKS))
    timer = fig.canvas.new_timer(interval=100)
    data = SimpleNamespace(cubes=[], masks=[None, None], n=0)

    def show(k: int) -> None:
        on = check.get_status()
        for i, (ax, im, cube) in enumerate(zip(axes, images, data.cubes)):
            k = min(k, data.n - 1)
            frame = cube[k]
            a = float(np.abs(frame).max()) or 1.0
            im.set_data(frame)
            im.set_extent((-0.5, frame.shape[1] - 0.5, -0.5, frame.shape[0] - 0.5))
            im.set_clim(-a, a)
            mask = data.masks[i] if on[i] else None
            ax.set_title(f"{names[i]}[{k}]")
            ax.set_xlabel(stats(frame, mask) + ("  (mask)" if mask is not None else ""))
        fig.canvas.draw_idle()

    def load() -> None:
        """Read every file first, so a failed read changes nothing."""
        cubes = [fits.getdata(calib / name) for name in names]
        masks = [None, None]
        for i, name in enumerate(MASKS):
            path = calib / name
            if path.is_file():
                m = fits.getdata(path)
                if m.shape == cubes[i].shape[1:]:
                    masks[i] = m
                else:
                    print(
                        f"{path}: shape {m.shape}, not {cubes[i].shape[1:]}; ignored."
                    )
        if kind == "cm":
            with fits.open(calib / CM_WFS) as hdus:
                header = hdus[0].header
                sv = hdus["SINGVALS"].data if "SINGVALS" in hdus else np.array([])
            sv_line.set_data(np.arange(len(sv)), sv / sv[0] if len(sv) else sv)
            reg = header.get("CMREG")
            sv_reg.set_ydata([reg if reg and len(sv) else np.nan] * 2)
            # kept singular modes are k < CMNCUT: the line sits between kept and cut
            sv_ncut.set_xdata([header.get("CMNCUT", np.nan) - 0.5] * 2)
            ax_sv.set_title(_cm_title(header) if len(sv) else "no SINGVALS HDU")
            ax_sv.relim()
            ax_sv.autoscale_view()
        old = data.masks
        data.cubes, data.masks, data.n = cubes, masks, min(len(c) for c in cubes)
        for i, name in enumerate(MASKS):
            check.labels[i].set_text(name if masks[i] is not None else f"no {name}")
            # on when a mask appears, off when it goes; else the operator's choice
            if (masks[i] is None) != (old[i] is None) and check.get_status()[i] == (
                masks[i] is None
            ):
                check.set_active(i)
        slider.valmax = max(data.n - 1, 1)  # Slider refuses valmin == valmax
        slider.ax.set_xlim(0, slider.valmax)
        if slider.val > data.n - 1:
            slider.set_val(data.n - 1)  # calls show
        else:
            show(int(slider.val))

    def tick() -> None:
        slider.set_val((int(slider.val) + 1) % data.n)

    def on_check(_label) -> None:
        for i, status in enumerate(check.get_status()):
            if status and data.masks[i] is None:  # no such mask: stays off
                check.set_active(i)
                return  # set_active calls us again
        show(int(slider.val))

    def toggle_play(_event=None) -> None:
        if play.label.get_text() == "Play":
            timer.start()
            play.label.set_text("Pause")
        else:
            timer.stop()
            play.label.set_text("Play")
        fig.canvas.draw_idle()

    def set_fps(text: str) -> None:
        fps.stop_typing()  # Enter releases the keys for r/w/d
        try:
            value = float(text)
            if value <= 0:
                raise ValueError
        except ValueError:
            fps.set_val(f"{1000 / timer.interval:g}")  # back to the current rate
            return
        timer.interval = max(1, round(1000 / value))

    def on_key(event) -> None:
        if fps.capturekeystrokes:
            return
        if event.key == "r":
            try:
                load()
            except OSError as error:
                print(f"not reloaded: {error}")
        elif event.key in MASK_KEYS:
            check.set_active(MASK_KEYS.index(event.key))

    timer.add_callback(tick)
    slider.on_changed(lambda v: show(int(v)))
    play.on_clicked(toggle_play)
    fps.on_submit(set_fps)
    check.on_clicked(on_check)
    fig.canvas.mpl_connect("key_press_event", on_key)
    load()
    # The widgets must outlive this call; tests drive them through it too.
    fig.moka = SimpleNamespace(
        slider=slider, play=play, fps=fps, check=check, timer=timer, tick=tick
    )
    return fig
