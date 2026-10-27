from __future__ import annotations

import os
import shutil

import pytest

from milk.infra import exceptions as exc
from milk.pipeline import Pipeline


@pytest.fixture
def pipeline(pytestconfig, tmp_path, monkeypatch):
    monkeypatch.setenv("MILK_SHM_DIR", str(tmp_path))
    pipe = Pipeline(pytestconfig.rootpath / "tests" / "resources", "pipelinebasic")
    monkeypatch.setattr(pipe, "is_running", lambda: False)
    pipe.root_folder = tmp_path / "rootdir"
    os.makedirs(pipe.root_folder, exist_ok=True)
    return pipe


def _make_remotes(shm_dir):
    for name in ("otherloop_output", "input_remote"):
        (shm_dir / f"{name}.im.shm").write_bytes(b"")


def test_apply_and_deapply(pipeline, tmp_path):
    _make_remotes(tmp_path)
    pipeline.apply_system_linkage("shared")
    link = tmp_path / "input.im.shm"
    assert link.is_symlink()
    assert link.resolve() == (tmp_path / "otherloop_output.im.shm").resolve()
    tag = (pipeline.root_folder / "systemlinkage.tag").read_text().splitlines()
    assert tag[0] == "shared" and tag[1]

    pipeline.apply_system_linkage("shared")  # idempotent

    pipeline.deapply_system_linkage("shared")
    assert not link.exists() and not link.is_symlink()


def test_unknown_linkage(pipeline):
    with pytest.raises(exc.UnknownSystemLinkageException):
        pipeline.apply_system_linkage("nope")


@pytest.mark.xfail(reason="Probably symlinking on a missing image should be allowed.")
def test_missing_remote(pipeline, tmp_path):
    with pytest.raises(exc.SystemLinkageException):
        pipeline.apply_system_linkage("shared")
    assert not (tmp_path / "input.im.shm").is_symlink()


@pytest.mark.xfail(
    reason="Probably overriding a real file with a symlink should be allowed"
)
def test_refuses_real_file(pipeline, tmp_path):
    _make_remotes(tmp_path)
    (tmp_path / "input.im.shm").write_bytes(b"real")
    with pytest.raises(exc.SystemLinkageException):
        pipeline.apply_system_linkage("shared")


def test_running_refused(pipeline, monkeypatch, tmp_path):
    _make_remotes(tmp_path)
    monkeypatch.setattr(pipeline, "is_running", lambda: True)
    with pytest.raises(exc.SystemLinkageException):
        pipeline.apply_system_linkage("shared")


def test_deapply_keeps_foreign_symlink(pipeline, tmp_path):
    _make_remotes(tmp_path)
    foreign = tmp_path / "input.im.shm"
    foreign.symlink_to(tmp_path / "input_remote.im.shm")
    pipeline.deapply_system_linkage("shared")
    assert foreign.is_symlink()


def test_cli_sys(pipeline, tmp_path, pytestconfig, monkeypatch):
    from click.testing import CliRunner

    from milk.infra import mains_pipeline_mgmt as mgmt

    _make_remotes(tmp_path)
    monkeypatch.setattr(Pipeline, "is_running", lambda self: False)
    # Copy the conf so the rootdir tag lands in tmp_path, not the repo.
    shutil.copytree(
        pytestconfig.rootpath / "tests" / "resources" / "pipelinebasic-conf",
        tmp_path / "pipelinebasic-conf",
    )
    folder = str(tmp_path)
    (tmp_path / "basic-rootdir").mkdir()  # normally created by `milk-pipes init`
    runner = CliRunner()

    result = runner.invoke(
        mgmt.milk_pipes, ["sys", "--folder", folder, "pipelinebasic", "shared"]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "input.im.shm").is_symlink()

    result = runner.invoke(
        mgmt.milk_pipes, ["sys", "--folder", folder, "pipelinebasic", "nope"]
    )
    assert result.exit_code != 0
