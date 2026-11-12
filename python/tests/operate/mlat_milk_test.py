"""O1 goal line (milk): run mlat over a milk-pipes pipeline and plot its latency graph.

Needs milk and cacao binaries, matplotlib, and the session spoofing of tests/conftestaux/milk.py.
See specs/011-o1-steps-runstart-mlat/brief.md and tests/resources/mlatsim-conf/.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from milk.operate import tasks as _tasks

if not hasattr(
    _tasks, "RunStart"
):  # parked until O1 lands: P1 deletes these two statements
    pytest.skip("O1 not implemented yet", allow_module_level=True)

import matplotlib
from pyMilk.interfacing.fps import FPS

from milk.operate.pipeline import Pipeline
from milk.operate.plans import Plan, Step, run, run_plan
from milk.operate.state import State, state
from milk.provision.rootdir import init
from milk.show.cli import mlat_figure

RESOURCES = Path(__file__).parents[1] / "resources"
MEASURE = Plan(
    "mlat",
    (
        Step("RunStart", ("dmdelay",)),
        Step("RunStart", ("wfscam",)),
        Step("RunStart", ("mlat",)),
        Step("WaitRunEnd", ("mlat",)),
        Step("RunStop", ("wfscam", "dmdelay")),
    ),
)
DELAYSEC = 0.003  # compute_units.dmdelay.fps.delaysec in mlatsim-conf


@pytest.fixture
def mlatsim(tmp_path):
    """mlatsim initialized and deployed in tmp_path; torn down (tmux included) at the end."""
    conf = tmp_path / "mlatsim-conf"
    shutil.copytree(RESOURCES / "mlatsim-conf", conf)
    pp = Pipeline(init(conf))
    run_plan(pp, "deploy")
    yield pp
    try:
        run_plan(Pipeline(pp.root_folder), "teardown")
    finally:
        for cu in pp.config.compute_units:
            argv = ["tmux", "kill-session", "-t", pp.config.fps_name(cu)]
            subprocess.run(argv, capture_output=True)


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
