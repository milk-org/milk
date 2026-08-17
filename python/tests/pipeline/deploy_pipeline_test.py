from __future__ import annotations

import os, shutil
import libtmux
import pytest
from pathlib import Path

from milk.pipeline import Pipeline
from milk.session import ComputeSession


@pytest.fixture
def pipeline(pytestconfig):
    return Pipeline(pytestconfig.rootpath / "tests" / "resources", "pipelinebasic")


@pytest.fixture
def cloned_pipeline(pipeline: Pipeline):
    # We're probably in some temp pwd.
    os.makedirs("AOloop/", exist_ok=True)

    cloned = pipeline.clone_to("./AOloop")  # new cloned pipeline obj
    yield cloned

    assert cloned.parent_folder.is_relative_to(os.getcwd())
    assert cloned.parent_folder.is_relative_to("/tmp")  # before deleting

    # Kill all the tmuxes that are associated with the pipeline
    tsrv = libtmux.Server()
    for sname in cloned.sessions:
        # TODO better tmux API integrated directly with session/pipeline
        abs_name = cloned.get_session_abs_name(sname)
        session = tsrv.sessions.get(session_name=abs_name, default=None)
        if session is not None:
            session.kill()

    shutil.rmtree(cloned.conf_folder)


def test_make_pipeline(pipeline):
    """
    Test that we can instantiate the pipeline directly in the source test folder
    """
    pp: Pipeline = pipeline
    assert pp.loop_number == 0
    assert pp.long_name == "pipelinebasic"

    assert pp.parent_folder.is_absolute()
    assert pp.parent_folder.is_relative_to(Path.home())


def test_make_cloned_pipeline(cloned_pipeline):
    """
    Clone the pipeline to a working folder
    """
    pp: Pipeline = cloned_pipeline
    assert pp.loop_number == 0
    assert pp.long_name == "pipelinebasic"

    assert pp.parent_folder.is_absolute()
    assert pp.parent_folder.is_relative_to(os.getcwd())
    assert pp.parent_folder.is_relative_to("/tmp")


def test_deploy_fps_call_by_class(cloned_pipeline: Pipeline):
    pp = cloned_pipeline

    from milk.infra.deploy_tasks import DeployFPS

    pp(DeployFPS)

    for sname in pp.sessions:
        sesh = pp.get_session(sname)
        assert sesh.fps is not None
        assert sesh.fps.is_valid()

        assert sesh.fps.name == pp.get_session_abs_name(sname)

        sesh.fps.destroy()


def test_deploy_fps_call_by_instance(cloned_pipeline: Pipeline):
    pp = cloned_pipeline

    from milk.infra.deploy_tasks import (
        DeployFPS,
        InitialFolderSetup,
        StartConfProcesses,
    )

    pp(InitialFolderSetup)(DeployFPS)(StartConfProcesses)

    for sname in pp.sessions:  # TODO this is bad naming urgh
        sesh = pp.get_session(sname)
        assert sesh.fps
        assert sesh.fps.conf_isrunning()
