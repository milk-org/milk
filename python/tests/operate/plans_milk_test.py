"""R4a acceptance tests (milk): the deploy and teardown plans on real FPSs, processes and SHM.

Needs milk binaries and the session spoofing of tests/conftestaux/milk.py.
See specs/009-r4a-rundir-state-plans/brief.md.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from pyMilk.interfacing.fps import FPS

from milk.errors import InconsistentStateError
from milk.operate import rundir
from milk.operate.plans import run_plan
from milk.operate.state import State, state
from milk.operate.tasks import ClaimRundir, InitFPS
from milk.operate.pipeline import Pipeline
from milk.provision.rootdir import init

RESOURCES = Path(__file__).parents[1] / "resources"
NAMES = ("basic_delay0", "basic_delay1")


def _fps_exists(name: str) -> bool:
    try:
        FPS(name)
    except Exception:
        return False
    return True


@pytest.fixture
def live(tmp_path, request):
    """pipelinebasic initialized in tmp_path. Parametrize indirectly with init overrides.
    Teardown always runs at the end, whatever the test did."""
    conf = tmp_path / "pipelinebasic-conf"
    shutil.copytree(RESOURCES / "pipelinebasic-conf", conf)
    overrides = getattr(request, "param", None) or {}
    pp = Pipeline(init(conf, overrides=overrides))
    yield pp
    final = Pipeline(pp.root_folder)
    try:
        run_plan(final, "teardown")
    except Exception:
        pass


def test_deploy_from_cold(live):
    assert state(live).state is State.COLD
    run_plan(live, "deploy")
    st = state(live)
    assert st.state is State.CONFIGURED
    assert rundir.read_claim("basic").rootdir == str(live.root_folder)
    for name in NAMES:
        assert (
            Path(FPS(name).md.workdir).resolve()
            == (live.root_folder / "work").resolve()
        )


def test_deploy_is_idempotent(live):
    run_plan(live, "deploy")
    run_plan(live, "deploy")
    assert state(live).state is State.CONFIGURED


def test_deploy_resumes_a_partial_deploy(live):
    live.task_do(ClaimRundir)
    live.task_do(InitFPS)
    assert state(live).state is State.DEPLOYED
    run_plan(live, "deploy")
    assert state(live).state is State.CONFIGURED


def test_inconsistent_then_teardown_then_deploy(live):
    run_plan(live, "deploy")
    FPS("basic_delay1").destroy()
    assert state(live).state is State.INCONSISTENT
    with pytest.raises(InconsistentStateError, match="teardown"):
        run_plan(live, "deploy")
    run_plan(live, "teardown")
    assert state(live).state is State.COLD
    run_plan(live, "deploy")
    assert state(live).state is State.CONFIGURED


def test_teardown_after_reboot(live, monkeypatch):
    run_plan(live, "deploy")
    monkeypatch.setattr("milk.operate.rundir.current_boot_id", lambda: "another-boot")
    assert state(live).state is State.INCONSISTENT
    run_plan(live, "teardown")
    assert state(live).state is State.COLD


def test_teardown_cleans_fps_without_rundir(live):
    run_plan(live, "deploy")
    rundir.release("basic")  # rundir lost, FPSs still there
    assert state(live).state is State.INCONSISTENT
    run_plan(live, "teardown")
    assert not any(_fps_exists(n) for n in NAMES)
    assert state(live).state is State.COLD


def test_teardown_then_redeploy(live):
    run_plan(live, "deploy")
    run_plan(live, "teardown")
    run_plan(live, "deploy")
    assert state(live).state is State.CONFIGURED
