"""013-3: the measlinresp plan, the SetLinResp and WaitLinResp tasks, on paper
(see specs/013-*/brief.md §5). The FPS is a dict; the streamin stream and the files are real.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from pyMilk.interfacing.shm import SHM

from moka.errors import TaskCannotRunError, TaskDidNotSucceedError
from moka.operate.pipeline import Pipeline
from moka.operate.plans import PLANS, Step
from moka.operate.tasks import TASKS

UNITS = """
[compute_units.mlat]
exec = "cacao-fpsexec-cacaopt-mlat"

[compute_units.measlinresp]
exec = "cacao-fpsexec-cacaoac-measlinresp"
"""
CACAOVARS = "export CACAO_WFSFRATE=1000.0\nexport CACAO_LATENCYHARDWFR=2.5\n"
FILES = "../calib/RMmodesDM.fits", "../calib/RMmodesWFS.inprogress.fits"


class _FPS(dict):
    """The measlinresp FPS: its values, conf/run flags, and the procinfo of its last run."""

    conf, run, pinfo = True, False, None

    def is_valid(self) -> bool:
        return True

    def conf_isrunning(self) -> bool:
        return self.conf

    def run_isrunning(self) -> bool:
        return self.run

    @property
    def pinfo_struct(self):
        if self.pinfo is None:
            raise FileNotFoundError("no procinfo file")
        return self.pinfo


@pytest.fixture
def lr(make_tmp_pipeline, pp_env, monkeypatch):
    """pp with: 5 pokes of a 4 x 3 DM (numpy (5, 3, 4)), a 4 x 3 streamin, mlat's cacaovars.bash, a paper FPS."""
    pp = make_tmp_pipeline(UNITS)
    pp.fps = _FPS(
        {
            "streamin": "lr_dm",
            "inmodes": "",
            "outmodes": "",
            "ampl": 0.1,  # from conf
            "timing.WFSfrequ": 0.0,
            "timing.hardwlatfr": 0.0,
            "timing.NBave": 1,
            "timing.NBexcl": 1,
            "procinfo.loopcntMax": 3,
        }
    )
    compu = SimpleNamespace(fpsname="basic_measlinresp", fps=pp.fps)
    monkeypatch.setattr(
        Pipeline, "compute_units", property(lambda self: {"measlinresp": compu})
    )
    pp.calib = pp.root_folder / "calib"
    pp.calib.mkdir()
    fits.writeto(pp.calib / "RMmodesDM.fits", np.zeros((5, 3, 4), np.float32))
    SHM(
        "lr_dm", np.zeros((4, 3), np.float32), symcode=0
    ).close()  # symcode 0: numpy (x, y)
    pp.datadir = pp.work_folder / "fps.basic_mlat.datadir"
    pp.datadir.mkdir(parents=True)
    (pp.datadir / "cacaovars.bash").write_text(CACAOVARS)
    pp.set = lambda params=None: pp.task_do(TASKS["SetLinResp"](pp, None, params))
    pp.wait = lambda: pp.task_do(TASKS["WaitLinResp"](pp))
    return pp


def test_plan_steps_and_its_ampl_option():
    plan = PLANS["measlinresp"]
    assert plan.steps == (
        Step("RunStart", ("acquWFS",)),
        Step("SetLinResp"),
        Step("RunStart", ("measlinresp",)),
        Step("WaitLinResp"),
    )
    (param,) = plan.params
    assert (param.name, param.type, param.short) == ("ampl", float, "a")
    assert (plan.on_failure, plan.accepts) == ("abort", "consistent")


def _no_fps(pp):
    pp.fps.conf = False


def _no_cube(pp):
    (pp.calib / "RMmodesDM.fits").unlink()


def _no_latency(pp):
    (pp.datadir / "cacaovars.bash").unlink()


def _bad_latency(pp):
    (pp.datadir / "cacaovars.bash").write_text("export CACAO_WFSFRATE=fast\n")


def _no_stream(pp):
    pp.fps["streamin"] = "lr_nope"


def _bad_shape(pp):
    fits.writeto(
        pp.calib / "RMmodesDM.fits", np.zeros((5, 4, 3), np.float32), overwrite=True
    )


@pytest.mark.parametrize(
    "spoil, reason",
    [
        (_no_fps, "basic_measlinresp is missing or its conf process is not running"),
        (_no_cube, r"RMmodesDM.fits does not exist; run `moka mkpokes`"),
        (_no_latency, r"no latency from mlat; run `moka mlat`"),
        (_bad_latency, r"no latency from mlat; run `moka mlat`"),
        (_no_stream, "stream lr_nope does not exist"),
        (_bad_shape, r"RMmodesDM.fits is 3 x 4, lr_dm is 4 x 3"),
    ],
)
def test_set_refuses_with_its_reason(lr, spoil, reason):
    spoil(lr)
    with pytest.raises(TaskCannotRunError, match=reason):
        lr.set()
    assert lr.fps["inmodes"] == ""


def test_set_writes_files_and_mlat_timing_and_ampl_only_when_given(lr):
    lr.set()
    assert (lr.fps["inmodes"], lr.fps["outmodes"]) == FILES
    assert (lr.fps["timing.WFSfrequ"], lr.fps["timing.hardwlatfr"]) == (1000.0, 2.5)
    assert lr.fps["ampl"] == 0.1
    lr.set({"ampl": 0.2})
    assert lr.fps["ampl"] == 0.2


def test_set_archives_the_last_rm_and_deletes_a_leftover(lr):
    (lr.calib / "RMmodesWFS.fits").write_text("new")
    (lr.calib / "RMmodesWFS.prev.fits").write_text("old")
    (lr.calib / "RMmodesWFS.inprogress.fits").write_text("partial")
    lr.set()
    assert (lr.calib / "RMmodesWFS.prev.fits").read_text() == "new"
    assert not (lr.calib / "RMmodesWFS.fits").exists()
    assert not (lr.calib / "RMmodesWFS.inprogress.fits").exists()


def test_set_keeps_prev_when_there_is_nothing_to_archive(lr):
    (lr.calib / "RMmodesWFS.prev.fits").write_text("old")
    lr.set()
    assert (lr.calib / "RMmodesWFS.prev.fits").read_text() == "old"


def test_a_second_set_archives_again_though_the_fps_is_already_set(lr):
    lr.set()
    (lr.calib / "RMmodesWFS.fits").write_text("new")
    lr.set()
    assert (lr.calib / "RMmodesWFS.prev.fits").read_text() == "new"
    assert not (lr.calib / "RMmodesWFS.fits").exists()


def _ended(pp, loopcnt: int | None, file: bool = True) -> None:
    """The run has ended after loopcnt cycles (None: no procinfo), with or without its file."""
    pp.fps["timing.WFSfrequ"] = 1000.0
    pp.fps.pinfo = None if loopcnt is None else SimpleNamespace(loopcnt=loopcnt)
    if file:
        (pp.calib / "RMmodesWFS.inprogress.fits").write_text("result")


@pytest.mark.timeout(5)
def test_wait_renames_after_a_complete_run_without_waiting_for_the_timeout(
    lr, monkeypatch
):
    monkeypatch.setattr(TASKS["WaitLinResp"], "TIMEOUT", 3600.0)
    _ended(lr, 3)
    lr.wait()
    assert (lr.calib / "RMmodesWFS.fits").read_text() == "result"
    assert not (lr.calib / "RMmodesWFS.inprogress.fits").exists()


@pytest.mark.parametrize(
    "loopcnt, file, reason",
    [
        (1, True, "ended after 1 of 3 cycles; partial result in RMmodesWFS.inprogress"),
        (None, True, "ended after 0 of 3 cycles"),
        (3, False, "completed but wrote no RMmodesWFS.inprogress.fits"),
    ],
)
def test_wait_fails_without_renaming(lr, loopcnt, file, reason):
    _ended(lr, loopcnt, file)
    with pytest.raises(TaskDidNotSucceedError, match=reason):
        lr.wait()
    assert not (lr.calib / "RMmodesWFS.fits").exists()
    assert (lr.calib / "RMmodesWFS.inprogress.fits").exists() == file


def test_wait_timeout_is_twice_the_expected_duration_plus_10s(lr):
    lr.fps["timing.WFSfrequ"] = 1000.0
    task = TASKS["WaitLinResp"](lr)
    # 2 (pos, neg) x 5 modes x (1 + 1) frames x 3 cycles / 1000 Hz = 0.06 s
    assert task.TIMEOUT == pytest.approx(2 * 0.06 + 10)
