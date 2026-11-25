"""017-1, 017-2: the SetLoopOff and LoadCM tasks, on paper, and the spool plan (see
specs/017-*/brief.md §3, §4).
The FPSs are dicts; the CM files are real; streams are an in-memory dict.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pyMilk.interfacing.shm
import pytest
from astropy.io import fits

from moka.calibs import loop
from moka.errors import ConfigError, TaskCannotRunError
from moka.operate.pipeline import Pipeline
from moka.operate.plans import PLANS, Step
from moka.operate.tasks import TASKS

UNITS = """
[dmhint]
xsize = 3
ysize = 2

[compute_units.wfs2modes]
exec = "milk-fpsexec-linalg-MVMextract"

[compute_units.mfilt]
exec = "cacao-fpsexec-mfilt"

[compute_units.modes2dm]
exec = "milk-fpsexec-linalg-MVMextract"
"""


class _FPS(dict):
    """An FPS: its values, conf/run flags, and how many writes it got."""

    conf, run, writes = True, False, 0

    def conf_isrunning(self) -> bool:
        return self.conf

    def run_isrunning(self) -> bool:
        return self.run

    def __setitem__(self, key, value) -> None:
        self.writes += 1
        super().__setitem__(key, value)


@pytest.fixture
def lp(make_tmp_pipeline, pp_env, monkeypatch):
    """pp with paper FPSs for wfs2modes, mfilt, modes2dm, a dict of streams, calib/."""
    pp = make_tmp_pipeline(UNITS)
    pp.fps = {
        "wfs2modes": _FPS(immodes="t_cmwfs"),
        "mfilt": _FPS(loopON=True),
        "modes2dm": _FPS(immodes="t_cmdm"),
    }
    units = {
        cu: SimpleNamespace(fpsname=f"basic_{cu}", fps=fps)
        for cu, fps in pp.fps.items()
    }
    monkeypatch.setattr(Pipeline, "compute_units", property(lambda self: units))
    pp.units = units

    pp.streams, pp.stream_writes = {}, 0

    class _SHM:
        def __init__(self, name, data=None, symcode=0):
            self.name = name
            if data is not None:
                pp.streams[name] = np.array(data)
                pp.stream_writes += 1

        def get_data(self):
            return pp.streams[self.name]

        def close(self) -> None:
            pass

    monkeypatch.setattr(pyMilk.interfacing.shm, "SHM", _SHM)
    monkeypatch.setattr(
        loop,
        "stream_shape",
        lambda name: pp.streams[name].shape if name in pp.streams else None,
    )
    pp.calib = pp.root_folder / "calib"
    pp.calib.mkdir()
    pp.loopoff = lambda: pp.task_do(TASKS["SetLoopOff"](pp, ("mfilt",)))
    pp.load = lambda: pp.task_do(TASKS["LoadCM"](pp, ("wfs2modes", "modes2dm")))
    return pp


def _pair(pp, n=5, wfs=(4, 3), dm=(3, 2), seed=0):
    """Writes calib/CMmodes{WFS,DM}.fits: n modes, sizes (x, y), so numpy (n, y, x)."""
    rng = np.random.default_rng(seed)
    cubes = [rng.random((n, y, x)).astype(np.float32) for x, y in (wfs, dm)]
    for name, cube in zip(("CMmodesWFS.fits", "CMmodesDM.fits"), cubes):
        fits.writeto(pp.calib / name, cube, overwrite=True)
    return cubes


# Without a subset: LoadCM is a config error, SetLoopOff finds the mfilt unit


def test_without_a_subset_loadcm_is_a_config_error(lp):
    with pytest.raises(ConfigError, match="named in the step"):
        TASKS["LoadCM"](lp)


def test_without_a_subset_loopoff_is_the_one_mfilt_unit(lp):
    assert TASKS["SetLoopOff"](lp).compus_names == ("mfilt",)


def test_without_a_subset_loopoff_needs_an_mfilt_unit(make_tmp_pipeline, pp_env):
    with pytest.raises(ConfigError, match="no cacao-fpsexec-mfilt unit"):
        TASKS["SetLoopOff"](make_tmp_pipeline())


def test_without_a_subset_loopoff_needs_a_single_mfilt_unit(make_tmp_pipeline, pp_env):
    pp = make_tmp_pipeline(
        UNITS + '[compute_units.mfilt2]\nexec = "cacao-fpsexec-mfilt"\n'
    )
    with pytest.raises(
        ConfigError, match=r"\['mfilt', 'mfilt2'\]; name one in the step"
    ):
        TASKS["SetLoopOff"](pp)


# SetLoopOff


def test_loopoff_writes_false_when_it_was_true(lp):
    lp.loopoff()
    assert lp.fps["mfilt"]["loopON"] is False


def test_loopoff_is_a_noop_when_already_false(lp):
    dict.__setitem__(lp.fps["mfilt"], "loopON", False)
    lp.loopoff()
    assert lp.fps["mfilt"].writes == 0


def test_loopoff_refuses_without_a_conf_process(lp):
    lp.fps["mfilt"].conf = False
    with pytest.raises(
        TaskCannotRunError, match="basic_mfilt is missing .* not running; run deploy"
    ):
        lp.loopoff()
    assert lp.fps["mfilt"]["loopON"] is True


# LoadCM: refusals


def _no_wfs(pp):
    (pp.calib / "CMmodesWFS.fits").unlink()


def _no_dm(pp):
    (pp.calib / "CMmodesDM.fits").unlink()


def _other_mode_counts(pp):
    fits.writeto(
        pp.calib / "CMmodesDM.fits", np.zeros((4, 2, 3), np.float32), overwrite=True
    )


def _dm_not_dmhint(pp):
    fits.writeto(
        pp.calib / "CMmodesDM.fits", np.zeros((5, 2, 4), np.float32), overwrite=True
    )


def _no_fps(pp):
    pp.units["modes2dm"].fps = None


def _wfs2modes_runs(pp):
    pp.fps["wfs2modes"].run = True


def _modes2dm_runs(pp):
    pp.fps["modes2dm"].run = True


@pytest.mark.parametrize(
    "spoil, reason",
    [
        (_no_wfs, r"CMmodesWFS.fits does not exist; run `moka mkcm`"),
        (_no_dm, r"CMmodesDM.fits does not exist; run `moka mkcm`"),
        (_other_mode_counts, "CMmodesWFS.fits has 5 modes, CMmodesDM.fits has 4"),
        (_dm_not_dmhint, r"CMmodesDM.fits is 4 x 2, \[dmhint\] is 3 x 2"),
        (_no_fps, "basic_modes2dm is missing; run deploy"),
        (_wfs2modes_runs, "basic_wfs2modes is running; stop it first"),
        (_modes2dm_runs, "basic_modes2dm is running; stop it first"),
    ],
)
def test_load_refuses_with_its_reason(lp, spoil, reason):
    _pair(lp)
    spoil(lp)
    with pytest.raises(TaskCannotRunError, match=reason):
        lp.load()
    assert lp.streams == {}


# LoadCM: what it writes


def test_load_writes_a_non_square_pair_in_stream_order_to_the_fps_targets(lp):
    wfs, dm = _pair(lp)  # WFS 4 x 3, DM 3 x 2, 5 modes
    lp.load()
    assert set(lp.streams) == {"t_cmwfs", "t_cmdm"}
    for name, cube, shape in (("t_cmwfs", wfs, (4, 3, 5)), ("t_cmdm", dm, (3, 2, 5))):
        written = lp.streams[name]
        assert (written.shape, written.dtype) == (shape, np.float32)
        x, y, k = shape[0] - 1, shape[1] - 1, 2
        assert written[x, y, k] == cube[k, y, x]
        assert np.array_equal(written, cube.transpose(2, 1, 0))


def test_load_repeat_is_a_noop(lp):
    _pair(lp)
    lp.load()
    lp.load()
    assert lp.stream_writes == 2


def test_load_loads_a_pair_of_the_same_size_and_other_values(lp):
    _pair(lp)
    lp.load()
    wfs, dm = _pair(lp, seed=1)
    lp.load()
    assert np.array_equal(lp.streams["t_cmwfs"], wfs.transpose(2, 1, 0))
    assert np.array_equal(lp.streams["t_cmdm"], dm.transpose(2, 1, 0))


def test_load_replaces_the_streams_with_another_mode_count(lp):
    _pair(lp)
    lp.load()
    _pair(lp, n=7)
    lp.load()
    assert lp.streams["t_cmwfs"].shape == (4, 3, 7)
    assert lp.streams["t_cmdm"].shape == (3, 2, 7)


# The spool plan


def test_spool_plan_has_its_six_steps():
    plan = PLANS["spool"]
    assert plan.steps == (
        Step("SetLoopOff", ("mfilt",)),
        Step("RunStop", ("modes2dm", "mfilt", "wfs2modes")),
        Step("LoadCM", ("wfs2modes", "modes2dm")),
        Step("RunStart", ("acquWFS",)),
        Step("RunStart", ("wfs2modes", "mfilt", "modes2dm")),
        Step("SetLoopOff", ("mfilt",)),
    )
    assert (plan.on_failure, plan.accepts, plan.params) == ("abort", "consistent", ())
