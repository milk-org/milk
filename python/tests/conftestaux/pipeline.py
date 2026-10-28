"""Pipeline fixtures: tmp copies of pipelinebasic, a live-deployable clone, fake CompUs."""

from __future__ import annotations

import contextlib
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import libtmux
import pytest

from milk.pipeline import Pipeline
from milk.provision import clone

RESOURCES = Path(__file__).parents[1] / "resources"


@pytest.fixture
def make_tmp_pipeline(tmp_path):
    """A pipelinebasic copy in tmp_path, with extra TOML appended to its conf.toml."""

    def make(extra_toml: str = "") -> Pipeline:
        conf = tmp_path / "pipelinebasic-conf"
        shutil.copytree(RESOURCES / "pipelinebasic-conf", conf)
        if extra_toml:
            with (conf / "conf.toml").open("a") as stream:
                stream.write(extra_toml)
        return Pipeline(tmp_path, "pipelinebasic")

    return make


@pytest.fixture
def cloned_pipeline():
    """pipelinebasic cloned under the session's tmp cwd; best-effort nuked before and after."""
    os.makedirs("AOloop/", exist_ok=True)
    clone(RESOURCES / "pipelinebasic-conf", Path("AOloop"))
    cloned = Pipeline("AOloop", "pipelinebasic")
    with contextlib.suppress(Exception):
        cloned.nuke()
    yield cloned

    # Guard: never rmtree outside the session tmp tree.
    assert cloned.parent_folder.is_relative_to(os.getcwd())
    if ".nox" in cloned.parent_folder.parts:
        assert "tmp" in cloned.parent_folder.parts
    else:
        assert cloned.parent_folder.is_relative_to("/tmp")
    with contextlib.suppress(Exception):
        cloned.nuke()
    tsrv = libtmux.Server()
    for cu in cloned.config.compute_units:
        session = tsrv.sessions.get(
            session_name=cloned.config.fps_name(cu), default=None
        )
        if session is not None:
            session.kill()
    shutil.rmtree(cloned.conf_folder)


def _fake_compu(fpsname: str) -> SimpleNamespace:
    """Stand-in for a deployed, valid, idle ComputeUnit; tests flip its attributes."""
    compu = SimpleNamespace(fpsname=fpsname, calls=[], destroy_error=None)

    def destroy() -> None:
        compu.calls.append("destroy")
        if compu.destroy_error is not None:
            raise compu.destroy_error

    compu.fps = SimpleNamespace(
        is_valid=lambda: True,
        conf_isrunning=lambda: False,
        run_isrunning=lambda: False,
        destroy=destroy,
    )
    compu.confstart = lambda tmux=False: compu.calls.append("confstart")
    compu.confstop = lambda: compu.calls.append("confstop")
    compu.runstop = lambda: compu.calls.append("runstop")
    return compu


@pytest.fixture
def fake_compus(monkeypatch):
    """Replace Pipeline.compute_units with fakes; returns {cu: fake} for pipelinebasic."""
    fakes = {cu: _fake_compu(f"basic_{cu}") for cu in ("delay0", "delay1")}
    monkeypatch.setattr(Pipeline, "compute_units", property(lambda self: fakes))
    return fakes
