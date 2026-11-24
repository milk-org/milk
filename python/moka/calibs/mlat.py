"""The mlat plan, and the results mlat leaves in its datadir."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..operate.plans import PLANS, Plan, Step

if TYPE_CHECKING:
    from ..operate.pipeline import Pipeline


def mlat_results(datadir: Path) -> tuple[float, float] | None:
    """(framerate [Hz], latency [frame]) from the ``cacaovars.bash`` mlat writes in its own datadir, or None."""
    path = datadir / "cacaovars.bash"
    if not path.is_file():
        return None
    env = dict(
        line.removeprefix("export ").strip().split("=", 1)
        for line in path.read_text().splitlines()
        if "=" in line
    )
    try:
        return float(env["CACAO_WFSFRATE"]), float(env["CACAO_LATENCYHARDWFR"])
    except (KeyError, ValueError):
        return None


def timing(pipeline: Pipeline) -> tuple[float, float] | None:
    """mlat_results of the pipeline's mlat FPS."""
    mlat = pipeline.config.fps_name("mlat")
    return mlat_results(pipeline.work_folder / f"fps.{mlat}.datadir")


PLANS.register(
    "mlat",
    Plan(
        "mlat",
        (
            Step("RunStart", ("acquWFS",)),
            Step("RunStart", ("mlat",)),
            Step("WaitRunEnd", ("mlat",)),
        ),
        help="Measure the hardware latency.",
    ),
)
