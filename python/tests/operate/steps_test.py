"""O1 acceptance tests (py): run() of any plan, and the RunStart/RunStop/WaitRunEnd tasks.

See specs/011-o1-steps-runstart-mlat/brief.md. FPSs are simulated (`paper`, `fake_compus`),
rundirs live in a self-cleaning MILK_SHM_DIR (`shm_dir_cleanafter`), and the boot id is pinned
(`boot`); see tests/conftestaux/operate.py and tests/conftestaux/pipeline.py.
"""

from __future__ import annotations

import pytest

from milk.operate import tasks as _tasks

if not hasattr(
    _tasks, "RunStart"
):  # parked until O1 lands: P1 deletes these two statements
    pytest.skip("O1 not implemented yet", allow_module_level=True)

from milk.errors import (
    ColdSystemError,
    ConfigError,
    TaskCannotRunError,
    TaskDidNotSucceedError,
)
from milk.operate.plans import PLANS, Plan, Step, run, run_plan
from milk.operate.tasks import TASKS, ClaimRundir, RunStart, RunStop, Task, WaitRunEnd

FPS = ("basic_delay0", "basic_delay1")


@pytest.fixture
def pp(make_tmp_pipeline, shm_dir_cleanafter, boot, paper):
    """pipelinebasic, initialized, cold, with a private SHM dir and simulated FPSs."""
    return make_tmp_pipeline()


@pytest.fixture
def deployed(pp, paper):
    """pp, claimed for real, with every config FPS present and its conf process running."""
    pp.task_do(ClaimRundir)
    for name in FPS:
        paper[name] = (True, False)
    return pp


@pytest.fixture
def log(monkeypatch):
    """Probe tasks A and B that record their forward(); registries restored after."""
    seen: list[str] = []
    monkeypatch.setattr(TASKS, "_items", dict(TASKS._items))
    monkeypatch.setattr(PLANS, "_items", dict(PLANS._items))
    for name in ("A", "B"):
        TASKS.register(
            name,
            type(
                name,
                (Task,),
                {
                    "ran": False,
                    "can": lambda self: True,
                    "forward": lambda self: (
                        seen.append(f"{type(self).__name__}:{self.compus_names}"),
                        setattr(self, "ran", True),
                    ),
                    "success": lambda self: self.ran,
                },
            ),
        )
    return seen


# ---- run(): any list of steps as a plan ----


def test_run_executes_steps_in_order_with_subsets(deployed, log):
    run(deployed, Plan("adhoc", (Step("A", ("delay1",)), Step("B"))))
    assert log == ["A:('delay1',)", "B:None"]


@pytest.mark.parametrize("bad", [Step("Nope"), Step("B", ("nope",))])
def test_run_checks_every_step_before_running_any(deployed, log, bad):
    with pytest.raises(ConfigError, match="(?i)nope"):
        run(deployed, Plan("adhoc", (Step("A"), bad)))
    assert log == []  # A did not run


def test_run_gates_on_the_plan_state(pp, log):
    with pytest.raises(ColdSystemError, match="deploy"):
        run(pp, Plan("adhoc", (Step("A"),)))
    assert log == []


def test_named_plans_still_go_through_run_plan(deployed, log, monkeypatch):
    PLANS.register("p", Plan("p", (Step("A"),)))
    run_plan(deployed, "p")
    assert log == ["A:None"]


def test_conf_plans_key_does_not_restrict_run(deployed, log):
    toml = deployed.root_folder / "conf" / "conf.toml"
    toml.write_text('plans = ["deploy", "teardown"]\n' + toml.read_text())
    from milk.operate.pipeline import Pipeline

    run(Pipeline(deployed.root_folder), Plan("adhoc", (Step("A"),)))
    assert log == ["A:None"]


# ---- RunStart / RunStop / WaitRunEnd ----


def test_run_tasks_are_registered():
    assert {"RunStart", "RunStop", "WaitRunEnd"} <= set(TASKS.names())


def test_runstart_starts_only_the_named_compu(pp, fake_compus):
    for fake in fake_compus.values():
        fake.conf = True
    pp.task_do(RunStart(pp, ["delay1"]))
    assert fake_compus["delay1"].calls == ["runstart"]
    assert fake_compus["delay0"].calls == []


def test_runstart_skips_a_running_compu(pp, fake_compus):
    for fake in fake_compus.values():
        fake.conf = fake.run = True
    pp.task_do(RunStart(pp, ["delay1"]))
    assert all(f.calls == [] for f in fake_compus.values())


def test_runstart_needs_the_conf_process(pp, fake_compus):
    with pytest.raises(TaskCannotRunError, match="deploy"):
        pp.task_do(RunStart(pp, ["delay0"]))
    assert fake_compus["delay0"].calls == []


def test_runstart_fails_when_the_run_never_shows(pp, fake_compus, monkeypatch):
    monkeypatch.setattr(RunStart, "TIMEOUT", 0.05)
    fake = fake_compus["delay0"]
    fake.conf = True
    fake.runstart = lambda: fake.calls.append("runstart")  # run never appears
    with pytest.raises(TaskDidNotSucceedError, match="basic_delay0"):
        pp.task_do(RunStart(pp, ["delay0"]))


def test_runstop_stops_only_the_named_compu(pp, fake_compus):
    for fake in fake_compus.values():
        fake.conf = fake.run = True
    pp.task_do(RunStop(pp, ["delay0"]))
    assert fake_compus["delay0"].calls == ["runstop"]
    assert fake_compus["delay1"].calls == []


def test_waitrunend_waits_without_stopping(pp, fake_compus):
    fake = fake_compus["delay1"]
    polls = iter([True, True, True])  # running for three polls, then done
    fake.fps.run_isrunning = lambda: next(polls, False)
    pp.task_do(WaitRunEnd(pp, ["delay1"]))
    assert fake.calls == []  # never stopped


def test_waitrunend_times_out(pp, fake_compus, monkeypatch):
    monkeypatch.setattr(WaitRunEnd, "TIMEOUT", 0.05)
    fake_compus["delay1"].run = True
    with pytest.raises(TaskDidNotSucceedError, match="basic_delay1"):
        pp.task_do(WaitRunEnd(pp, ["delay1"]))
    assert fake_compus["delay1"].calls == []
