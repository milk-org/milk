"""R1c acceptance tests: CompU views (X3) and fixes B1, B4–B7. See specs/003-r1c-compu-views-fixes/brief.md."""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

from milk.compute_unit import ComputeUnit
from milk.errors import BestEffortError, TaskCannotRunError, TaskDidNotSucceedError
from milk.infra.deploy_tasks import (
    DeployFPS,
    InitialFolderSetup,
    LoadDataFiles,
    StartConfProcesses,
)

# ---- X3: compute units come from the pipeline ----


def test_compute_units_are_read_only_views(make_tmp_pipeline):
    pp = make_tmp_pipeline()
    compus = pp.compute_units
    assert compus.keys() == pp.config.compute_units.keys()
    for cu, compu in compus.items():
        assert isinstance(compu, ComputeUnit)
        assert compu.fpsname == pp.config.fps_name(cu)
        assert compu.exec_name == pp.config.compute_units[cu].exec
    with pytest.raises(AttributeError):
        pp.compute_units = {}  # type: ignore[misc]


def test_compute_units_refresh(cloned_pipeline):
    """milk: a destroyed FPS drops out of the next view; a redeployed one is relinked."""
    pp = cloned_pipeline
    pp.task_do(InitialFolderSetup).task_do(DeployFPS)
    assert all(
        c.fps is not None and c.fps.is_valid() for c in pp.compute_units.values()
    )

    fps0 = pp.compute_units["delay0"].fps
    assert fps0 is not None
    fps0.destroy()
    assert pp.compute_units["delay0"].fps is None
    assert pp.compute_units["delay1"].fps is not None

    with contextlib.chdir(pp.work_folder):
        pp.compute_units["delay0"].fpsinit()
    relinked = pp.compute_units["delay0"].fps
    assert relinked is not None and relinked.is_valid()


def test_any_run_active(make_tmp_pipeline, fake_compus):
    pp = make_tmp_pipeline()
    assert not pp.any_run_active()
    fake_compus["delay1"].fps.run_isrunning = lambda: True
    assert pp.any_run_active()


# ---- Bug fixes ----


def test_B1_invalid_fps_cannot_confstart(make_tmp_pipeline, fake_compus):
    pp = make_tmp_pipeline()
    pp.work_folder.mkdir(parents=True)
    fake_compus["delay1"].fps.is_valid = lambda: False
    task = StartConfProcesses(pp)
    assert not task.can()
    assert "basic_delay1" in task.failure_reason


def test_B4_partial_rootdir_is_refused(make_tmp_pipeline):
    pp = make_tmp_pipeline()
    pp.root_folder.mkdir()  # partial: no conf.toml
    with pytest.raises(TaskCannotRunError) as err:
        pp.task_do(InitialFolderSetup)
    assert str(pp.root_folder) in str(err.value)


def test_B5_loaded_data_must_match_shape_and_dtype(make_tmp_pipeline):
    from astropy.io import fits
    from pyMilk.interfacing.shm import SHM

    pp = make_tmp_pipeline(
        '\n[[dataloads]]\nsource = "flat.fits"\ntarget = "r1c_flat"\n'
    )
    (pp.root_folder / "data").mkdir(parents=True)
    data = np.arange(96, dtype=np.float32).reshape(8, 12)
    fits.writeto(pp.root_folder / "data" / "flat.fits", data)
    task = LoadDataFiles(pp)
    try:
        SHM("r1c_flat", np.zeros((4, 4), np.float32)).close()
        assert not task.success()  # wrong shape

        pp.task_do(LoadDataFiles)
        assert task.success()

        SHM("r1c_flat", data.astype(np.float64)).close()
        assert not task.success()  # wrong dtype
    finally:
        SHM("r1c_flat").destroy()


def test_B6_confstart_timeout_names_compus(make_tmp_pipeline, fake_compus, monkeypatch):
    monkeypatch.setattr(StartConfProcesses, "TIMEOUT_ALL", 0.05)
    pp = make_tmp_pipeline()
    pp.work_folder.mkdir(parents=True)
    with pytest.raises(TaskDidNotSucceedError) as err:
        pp.task_do(StartConfProcesses)
    assert "basic_delay0" in str(err.value) and "basic_delay1" in str(err.value)
    assert all("confstart" in f.calls for f in fake_compus.values())


def test_B7_nuke_attempts_every_step(
    make_tmp_pipeline, fake_compus, tmp_path, monkeypatch
):
    monkeypatch.setenv("MILK_SHM_DIR", str(tmp_path))
    pp = make_tmp_pipeline()
    pp.root_folder.mkdir()
    for fake in fake_compus.values():
        fake.destroy_error = RuntimeError("boom")

    with pytest.raises(BestEffortError) as err:
        pp.nuke()

    labels = [label for label, _ in err.value.failures]
    assert len(labels) == 2 and all("destroy" in label for label in labels)
    assert all("destroy" in f.calls for f in fake_compus.values())
    assert not pp.root_folder.exists()
