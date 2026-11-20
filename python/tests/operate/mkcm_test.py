"""016-2: the mkcm plan and the MakeCM task, on hand-written RM files (6 pokes, 3 x 2 DM,
4 x 3 WFS; see specs/016-*/brief.md §5)."""

from __future__ import annotations

import os

import numpy as np
import pytest
from astropy.io import fits
from click.testing import CliRunner

from moka.errors import TaskCannotRunError, TaskDidNotSucceedError
from moka.operate.cli import moka
from moka.operate.plans import PLANS, Step, run_plan
from moka.operate.tasks import TASKS

N, DM, WFS = 6, (2, 3), (3, 4)  # numpy (y, x)


@pytest.fixture
def cm(make_tmp_pipeline, pp_env):
    """A never-deployed pipeline with RM files in calib/, a do(**params) helper, and paths."""
    pp = make_tmp_pipeline()
    calib = pp.root_folder / "calib"
    calib.mkdir()
    pp.rm_dm, pp.rm_wfs = calib / "RMmodesDM.fits", calib / "RMmodesWFS.fits"
    pp.cm_wfs, pp.cm_dm = calib / "CMmodesWFS.fits", calib / "CMmodesDM.fits"
    rng = np.random.default_rng(16)
    fits.writeto(pp.rm_dm, rng.normal(size=(N, *DM)).astype(np.float32))
    fits.writeto(pp.rm_wfs, rng.normal(size=(N, *WFS)).astype(np.float32))

    def do(**params):
        pp.task_do(TASKS["MakeCM"](pp, None, params))
        return fits.getheader(pp.cm_wfs), fits.getheader(pp.cm_dm)

    pp.do = do
    pp.mtimes = lambda: (pp.cm_wfs.stat().st_mtime_ns, pp.cm_dm.stat().st_mtime_ns)
    return pp


def test_plan_has_one_step_a_positional_flavor_and_n_r():
    plan = PLANS["mkcm"]
    assert plan.steps == (Step("MakeCM"),)
    flavor, nmodes, reg = plan.params
    assert (flavor.name, flavor.positional, flavor.default) == ("flavor", True, "tsvd")
    assert (nmodes.name, nmodes.type, nmodes.short) == ("nmodes", int, "n")
    assert (reg.name, reg.type, reg.short) == ("reg", float, "r")
    help = CliRunner().invoke(moka, ["-h"]).output
    assert "mkcm [-p PIPE] [FLAVOR] [-n NMODES] [-r REG]" in help


@pytest.mark.parametrize(
    "params, reason",
    [
        ({"flavor": "lsq"}, "unknown CM flavor 'lsq'; known: tsvd, tikhonov, pinv"),
        ({"flavor": "tikhonov"}, "tikhonov needs -r REG"),
        ({"reg": -0.1}, "-r -0.1 is negative"),
        ({"nmodes": 0}, "-n 0 is not in 1..6"),
        ({"nmodes": 7}, "-n 7 is not in 1..6"),
    ],
)
def test_bad_parameters_are_refused(cm, params, reason):
    with pytest.raises(TaskCannotRunError, match=reason):
        cm.do(**params)
    assert not cm.cm_wfs.exists() and not cm.cm_dm.exists()


@pytest.mark.parametrize(
    "missing, reason",
    [("rm_dm", "run `moka mkpokes`"), ("rm_wfs", "run `moka measlinresp`")],
)
def test_a_missing_rm_file_is_refused(cm, missing, reason):
    getattr(cm, missing).unlink()
    with pytest.raises(TaskCannotRunError, match=reason):
        cm.do()
    assert not cm.cm_wfs.exists() and not cm.cm_dm.exists()


def test_rm_files_of_different_mode_counts_are_refused(cm):
    fits.writeto(cm.rm_wfs, np.ones((5, *WFS), np.float32), overwrite=True)
    with pytest.raises(
        TaskCannotRunError, match="RMmodesDM.fits has 6 modes, RMmodesWFS.fits has 5"
    ):
        cm.do()
    assert not cm.cm_wfs.exists()


@pytest.mark.parametrize(
    "params, slices, kept",
    [
        ({}, N, N),
        ({"flavor": "tsvd", "nmodes": 4}, 4, 4),
        ({"flavor": "tikhonov", "reg": 0.01, "nmodes": 3}, 3, 3),
        ({"flavor": "pinv", "nmodes": 2}, N, 2),
    ],
)
def test_each_flavor_writes_both_files(cm, params, slices, kept):
    h_wfs, h_dm = cm.do(**params)
    with fits.open(cm.cm_wfs) as hdus:
        assert hdus[0].data.shape == (slices, *WFS) and hdus[0].data.dtype == np.dtype(
            ">f4"
        )
        sv = hdus["SINGVALS"].data
        assert sv.shape == (N,) and (np.diff(sv) <= 0).all()
    dm = fits.getdata(cm.cm_dm)
    assert dm.shape == (slices, *DM) and dm.dtype == np.dtype(">f4")
    for h in (h_wfs, h_dm):
        assert h["CMFLAVOR"] == params.get("flavor", "tsvd")
        assert h["CMREG"] == params.get("reg", 1e-6)
        assert h.get("CMNCUT") == params.get("nmodes")
        assert h["CMNKEPT"] == kept
    if "nmodes" not in params:
        assert "CMNCUT" not in h_wfs and "CMNCUT" not in h_dm


def test_a_repeat_is_a_no_op_and_a_parameter_change_rewrites(cm):
    cm.do()
    before = cm.mtimes()
    cm.do(flavor="tsvd", reg=1e-6)  # the default, given
    assert cm.mtimes() == before
    for params in (
        {"flavor": "pinv"},
        {"flavor": "pinv", "nmodes": 3},
        {"flavor": "pinv", "nmodes": 3, "reg": 1e-3},
    ):
        h_wfs, _ = cm.do(**params)
        assert (h_wfs["CMFLAVOR"], h_wfs.get("CMNCUT"), h_wfs["CMREG"]) == (
            params["flavor"],
            params.get("nmodes"),
            params.get("reg", 1e-6),
        )


@pytest.mark.parametrize("rm", ["rm_dm", "rm_wfs"])
def test_a_newer_rm_file_rewrites(cm, rm):
    cm.do()
    earlier = getattr(cm, rm).stat().st_mtime_ns - 10**9
    for path in (cm.cm_wfs, cm.cm_dm):  # the RM file is now the newer one
        os.utime(path, ns=(earlier, earlier))
    cm.do()
    assert min(cm.mtimes()) > earlier


def test_a_hand_made_pair_is_replaced(cm):
    fits.writeto(cm.cm_wfs, np.zeros((2, *WFS), np.float32))
    fits.writeto(cm.cm_dm, np.zeros((2, *DM), np.float32))
    h_wfs, h_dm = cm.do()
    assert h_wfs["NAXIS3"] == h_dm["NAXIS3"] == N


def test_an_empty_kept_set_fails_and_leaves_the_pair_untouched(cm):
    cm.do()
    before = cm.mtimes()
    fits.writeto(cm.rm_wfs, np.zeros((N, *WFS), np.float32), overwrite=True)
    with pytest.raises(
        TaskDidNotSucceedError, match="no singular mode kept with -n None -r 1e-06"
    ):
        cm.do()
    assert cm.mtimes() == before


def test_mkcm_runs_on_a_pipeline_never_deployed(cm):
    run_plan(cm, "mkcm", params={"flavor": "pinv"})
    assert fits.getheader(cm.cm_dm)["CMFLAVOR"] == "pinv"
