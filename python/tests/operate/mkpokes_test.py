"""013-2: the mkpokes plan, the MakePokes task and Pipeline.dm_mask, on a 4 x 3 [dmhint]
(see specs/013-*/brief.md §4)."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from moka.errors import TaskCannotRunError
from moka.operate.plans import PLANS, Step
from moka.operate.tasks import TASKS

DMHINT = "\n[dmhint]\nxsize = 4\nysize = 3\n"
BEAM = "beam_xcent = 1.5\nbeam_ycent = 1.0\nbeam_rad = 1.6\n"


@pytest.fixture
def pokes(make_tmp_pipeline, pp_env):
    """make(extra) -> pp with a do(type) helper and its cube path."""

    def make(extra: str = DMHINT):
        pp = make_tmp_pipeline(extra)
        pp.path = pp.root_folder / "calib" / "RMmodesDM.fits"

        def do(kind: str | None = None) -> np.ndarray:
            params = {} if kind is None else {"type": kind}
            pp.task_do(TASKS["MakePokes"](pp, None, params))
            return fits.getdata(pp.path)

        pp.do = do
        return pp

    return make


def test_plan_has_one_step_and_a_positional_type_defaulting_to_hadamard():
    plan = PLANS["mkpokes"]
    assert plan.steps == (Step("MakePokes"),)
    (param,) = plan.params
    assert (param.name, param.positional, param.default) == ("type", True, "hadamard")


def test_dm_mask_is_every_actuator_without_a_beam(pokes):
    mask = pokes().dm_mask()
    assert mask.shape == (3, 4) and mask.all()


def test_dm_mask_is_the_disk_strictly_within_the_beam(pokes):
    mask = pokes(DMHINT + BEAM).dm_mask()
    assert mask.astype(int).tolist() == [[0, 1, 1, 0], [1, 1, 1, 1], [0, 1, 1, 0]]


def test_a_partial_beam_is_refused_at_config_load(pokes):
    with pytest.raises(Exception, match="beam_xcent, beam_ycent, beam_rad"):
        pokes(DMHINT + "beam_rad = 1.0\n")


def test_zonal_is_one_1_per_slice(pokes):
    pp = pokes()
    cube = pp.do("zonal")
    assert cube.shape == (12, 3, 4) and cube.dtype == np.dtype(">f4")
    assert (cube.sum(axis=(1, 2)) == 1).all() and cube.max() == 1
    assert (cube.reshape(12, -1).sum(axis=0) == 1).all()  # one slice per actuator
    header = fits.getheader(pp.path)
    assert (header["POKETYPE"], header["NAXIS1"], header["NAXIS2"]) == ("zonal", 4, 3)


def test_zonal_pokes_only_the_beam(pokes):
    pp = pokes(DMHINT + BEAM)
    cube = pp.do("zonal")
    assert cube.shape == (8, 3, 4)
    assert (cube.sum(axis=0).astype(bool) == pp.dm_mask()).all()


def test_hadamard_is_the_default_and_actuator_series_are_orthogonal(pokes):
    pp = pokes()
    cube = pp.do()
    assert cube.shape == (16, 3, 4)  # next power of two of 12
    assert fits.getheader(pp.path)["POKETYPE"] == "hadamard"
    series = cube.reshape(16, 12)  # one column per actuator
    assert np.allclose(series.T @ series, 16 * np.eye(12))


def test_unknown_type_is_refused_listing_the_types(pokes):
    pp = pokes()
    with pytest.raises(TaskCannotRunError, match="unknown poke type.*hadamard, zonal"):
        pp.do("fourier")
    assert not pp.path.exists()


def test_missing_dmhint_is_refused(pokes):
    with pytest.raises(TaskCannotRunError, match=r"no \[dmhint\]"):
        pokes("").do()


def test_a_repeat_is_a_no_op_and_a_type_change_overwrites(pokes):
    pp = pokes()
    pp.do("zonal")
    before = pp.path.stat().st_mtime_ns
    pp.do("zonal")
    assert pp.path.stat().st_mtime_ns == before
    assert pp.do("hadamard").shape[0] == 16
    assert fits.getheader(pp.path)["POKETYPE"] == "hadamard"


def test_a_cube_of_another_dm_size_is_replaced(pokes):
    pp = pokes()
    pp.path.parent.mkdir()
    fits.writeto(
        pp.path, np.zeros((5, 3, 3), np.float32), fits.Header({"POKETYPE": "zonal"})
    )
    assert pp.do("zonal").shape == (12, 3, 4)


def test_a_hand_made_cube_is_replaced_by_mkpokes(pokes):
    pp = pokes()
    pp.path.parent.mkdir()
    fits.writeto(pp.path, np.zeros((5, 3, 4), np.float32))  # no POKETYPE
    assert pp.do("zonal").shape[0] == 12
