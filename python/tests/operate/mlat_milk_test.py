"""O1 goal line (milk): run mlat over a moka pipeline and plot its latency graph.

Needs milk and cacao binaries, matplotlib, and the session spoofing of tests/conftestaux/milk.py.
See specs/011-o1-steps-runstart-mlat/brief.md and tests/resources/mlatsim-conf/.
"""

from __future__ import annotations

import pytest

from moka.operate import tasks as _tasks

if not hasattr(
    _tasks, "RunStart"
):  # parked until O1 lands: P1 deletes these two statements
    pytest.skip("O1 not implemented yet", allow_module_level=True)

import matplotlib
from pyMilk.interfacing.fps import FPS

from moka.operate.pipeline import Pipeline
from moka.operate.plans import Plan, Step, run
from moka.operate.state import State, state
from moka.show.cli import mlat_figure

MEASURE = Plan(
    "mlat",
    (
        Step("RunStart"),  # runstart: dmcomb, dmdelay, wfscam, acquWFS
        Step("RunStart", ("mlat",)),
        Step("WaitRunEnd", ("mlat",)),
        Step("RunStop"),
    ),
)
DELAYSEC = 0.003  # compute_units.dmdelay.fps.delaysec in mlatsim-conf


@pytest.fixture
def mlatsim(make_mlatsim):
    return make_mlatsim()


def _measure(pp: Pipeline) -> None:
    run(pp, MEASURE)


def test_mlat_measures_the_simulated_latency(mlatsim):
    _measure(mlatsim)
    assert (
        state(mlatsim).state is State.CONFIGURED
    )  # dmdelay and wfscam stopped, mlat ended

    name = mlatsim.config.fps_name("mlat")
    fps = FPS(name)
    rate, latency_fr = fps["out.framerateHz"], fps["out.latencyfr"]
    assert 100 < rate < 5000
    latency_s = latency_fr / rate
    assert 0.5 * DELAYSEC < latency_s < 2 * DELAYSEC + 2 / rate

    # The run writes into rootdir/work, whatever the operator's cwd (here: the session tmp dir).
    datadir = mlatsim.work_folder / f"fps.{name}.datadir"
    lines = (datadir / "hardwlatency.dat").read_text().splitlines()
    assert sum(line.startswith("#") for line in lines) == 50  # NBiter


def test_a_second_measurement_runs_again(mlatsim):
    datadir = mlatsim.work_folder / f"fps.{mlatsim.config.fps_name('mlat')}.datadir"
    _measure(mlatsim)
    first = (datadir / "hardwlatency.dat").stat().st_mtime_ns
    _measure(mlatsim)  # RunStart[mlat] is not skipped: the first run has ended
    assert (datadir / "hardwlatency.dat").stat().st_mtime_ns > first
