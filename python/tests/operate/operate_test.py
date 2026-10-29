"""R4a acceptance tests (py): registry, rundir and claim, state(), runner, plans, linkages.

See specs/009-r4a-rundir-state-plans/brief.md. No milk binaries: FPSs are simulated by the `paper`
fixture (tests/conftestaux/operate.py); rundirs live in a private MILK_SHM_DIR (`shm`).
"""

from __future__ import annotations

import pytest

import tomli

from milk.errors import (
    BestEffortError,
    ColdSystemError,
    ConfigError,
    InconsistentStateError,
    LinkageError,
    TaskCannotRunError,
)
from milk.operate import rundir
from milk.operate.plans import PLANS, Plan, Step, run_plan
from milk.operate.state import State, state
from milk.operate.tasks import TASKS, ClaimRundir, DeapplyLinkages, ReleaseRundir, Task
from milk.registry import Registry

FPS = {"basic_delay0", "basic_delay1"}


@pytest.fixture
def pp(make_tmp_pipeline, shm, boot, paper):
    """pipelinebasic, initialized, cold, with a private SHM dir and simulated FPSs."""
    return make_tmp_pipeline()


def _deploy_on_paper(pp, paper, conf=False, run=False):
    """Claim the rundir for real and pretend every config FPS exists (and conf/run as given)."""
    pp.task_do(ClaimRundir)
    for name in FPS:
        paper[name] = (conf, run)


def _with_conf(pp, replace: tuple[str, str] | None = None, prepend: str = ""):
    """The same rootdir with its conf.toml hand-edited (live), resolved afresh."""
    from milk.pipeline import Pipeline

    toml = pp.root_folder / "conf" / "conf.toml"
    text = toml.read_text()
    if replace:
        text = text.replace(*replace)
    toml.write_text(prepend + text)
    return Pipeline(pp.root_folder)


# ---- registry ----


def test_registry():
    reg = Registry[int]("widget")
    assert reg.register("a", 1) == 1
    assert reg["a"] == 1 and "a" in reg and reg.names() == ["a"]
    with pytest.raises(ValueError):
        reg.register("a", 2)
    with pytest.raises(ConfigError, match="a"):  # unknown names list the known ones
        reg["b"]


def test_registries_are_populated():
    from milk.provision.hooks import HOOK_RUNNERS

    assert {"fetch", "command"} <= set(HOOK_RUNNERS.names())
    assert {"deploy", "teardown"} <= set(PLANS.names())
    assert {
        "ClaimRundir",
        "InitFPS",
        "LoadData",
        "ConfStart",
        "RunStop",
        "DeapplyLinkages",
        "ConfStop",
        "KillTmux",
        "DestroyFPS",
        "ReleaseRundir",
    } <= set(TASKS.names())


def test_generic_plans_shape():
    deploy, teardown = PLANS["deploy"], PLANS["teardown"]
    assert [s.task for s in deploy.steps] == [
        "ClaimRundir",
        "InitFPS",
        "LoadData",
        "ConfStart",
    ]
    assert (deploy.on_failure, deploy.accepts) == ("abort", "cold_or_consistent")
    assert [s.task for s in teardown.steps] == [
        "RunStop",
        "DeapplyLinkages",
        "ConfStop",
        "KillTmux",
        "DestroyFPS",
        "ReleaseRundir",
    ]
    assert (teardown.on_failure, teardown.accepts) == ("best_effort", "any")


# ---- rundir and claim ----


def test_claim_writes_the_rundir(pp, shm):
    pp.task_do(ClaimRundir)
    path = shm / "pipe_rundirs" / "basic"
    assert (path / "boot_id").read_text().strip() == "boot-A"
    claim = tomli.loads((path / "claim.toml").read_text())  # A.4 is TOML
    assert claim["name"] == "basic" and claim["loop_number"] == 0
    assert claim["rootdir"] == str(pp.root_folder)
    assert rundir.read_claim("basic").rootdir == str(pp.root_folder)


def test_claim_refuses_same_loop_number(pp, shm):
    rundir.write_claim(
        rundir.Claim.new(name="other", loop_number=0, rootdir="/elsewhere")
    )
    with pytest.raises(TaskCannotRunError, match="loop_number"):
        pp.task_do(ClaimRundir)


def test_claim_refuses_a_rundir_owned_by_another_rootdir(pp, shm):
    rundir.write_claim(
        rundir.Claim.new(name="basic", loop_number=0, rootdir="/elsewhere")
    )
    with pytest.raises(TaskCannotRunError, match="/elsewhere"):
        pp.task_do(ClaimRundir)


def test_release_removes_the_rundir(pp, shm):
    pp.task_do(ClaimRundir)
    pp.task_do(ReleaseRundir)
    assert not (shm / "pipe_rundirs" / "basic").exists()


# ---- state ----


def test_state_cold(pp):
    st = state(pp)
    assert st.state is State.COLD


def test_state_fps_without_rundir(pp, paper):
    paper["basic_delay0"] = (False, False)
    st = state(pp)
    assert st.state is State.INCONSISTENT and any("rundir" in r for r in st.reasons)


def test_state_stale_boot(pp, paper, boot):
    _deploy_on_paper(pp, paper)
    boot("boot-B")
    st = state(pp)
    assert st.state is State.INCONSISTENT and any("boot" in r for r in st.reasons)


def test_state_config_fps_missing(pp, paper):
    _deploy_on_paper(pp, paper)
    del paper["basic_delay1"]
    st = state(pp)
    assert st.state is State.INCONSISTENT and any(
        "basic_delay1" in r for r in st.reasons
    )


@pytest.mark.parametrize(
    "flags, expected",
    [
        (
            {"basic_delay0": (False, False), "basic_delay1": (False, False)},
            State.DEPLOYED,
        ),
        (
            {"basic_delay0": (True, False), "basic_delay1": (True, False)},
            State.CONFIGURED,
        ),
        ({"basic_delay0": (True, True), "basic_delay1": (True, True)}, State.RUNNING),
        ({"basic_delay0": (True, True), "basic_delay1": (True, False)}, State.MIXED),
    ],
)
def test_state_consistent(pp, paper, flags, expected):
    _deploy_on_paper(pp, paper)
    paper.update(flags)
    assert state(pp).state is expected


# ---- runner ----


class _Probe(Task):
    """Records what the runner does with it; behaviour set per test through class attributes."""

    log: list[str] = []
    ok_before = False
    allowed = True
    ran = False

    def can(self) -> bool:
        self.failure_reason = "probe refuses"
        return self.allowed

    def forward(self) -> None:
        self.ran = True
        _Probe.log.append(f"{type(self).__name__}:{self.compus}")

    def success(self) -> bool:
        return self.ran or self.ok_before


@pytest.fixture
def probes(monkeypatch):
    """Two probe tasks and helpers to register throwaway plans; registries restored after."""
    _Probe.log = []
    monkeypatch.setattr(TASKS, "_items", dict(TASKS._items))
    monkeypatch.setattr(PLANS, "_items", dict(PLANS._items))
    first = type("ProbeA", (_Probe,), {})
    second = type("ProbeB", (_Probe,), {})
    TASKS.register("ProbeA", first)
    TASKS.register("ProbeB", second)

    def plan(name, *steps, on_failure="abort", accepts="any"):
        PLANS.register(name, Plan(name, tuple(steps), on_failure, accepts))

    return first, second, plan


def test_runner_gates_on_state(pp, paper, probes, boot):
    _, _, plan = probes
    plan("p_consistent", Step("ProbeA"), accepts="consistent")
    plan("p_deploylike", Step("ProbeA"), accepts="cold_or_consistent")
    with pytest.raises(ColdSystemError, match="deploy"):
        run_plan(pp, "p_consistent")
    run_plan(pp, "p_deploylike")  # cold is fine

    _deploy_on_paper(pp, paper)
    boot("boot-B")  # now inconsistent
    with pytest.raises(InconsistentStateError, match="teardown"):
        run_plan(pp, "p_deploylike")
    with pytest.raises(InconsistentStateError, match="teardown"):
        run_plan(pp, "p_consistent")


def test_runner_checks_state_once(pp, probes, monkeypatch):
    import milk.operate.plans as plans_module

    _, _, plan = probes
    plan("p", Step("ProbeA"), Step("ProbeB"), Step("ProbeA"))
    calls = []
    real = plans_module.state
    monkeypatch.setattr(plans_module, "state", lambda p: calls.append(1) or real(p))
    run_plan(pp, "p")
    assert len(calls) == 1


def test_runner_skips_successful_steps(pp, probes):
    first, _, plan = probes
    first.ok_before = True
    plan("p", Step("ProbeA"))
    run_plan(pp, "p")
    assert _Probe.log == []


def test_runner_abort_stops_at_first_failure(pp, probes):
    first, second, plan = probes
    first.allowed = False
    plan("p", Step("ProbeA"), Step("ProbeB"))
    with pytest.raises(TaskCannotRunError, match="probe refuses"):
        run_plan(pp, "p")
    assert _Probe.log == []  # ProbeB never ran


def test_runner_best_effort_runs_everything(pp, probes):
    first, second, plan = probes
    first.allowed = False
    plan("p", Step("ProbeA"), Step("ProbeB"), on_failure="best_effort")
    with pytest.raises(BestEffortError) as err:
        run_plan(pp, "p")
    assert [label for label, _ in err.value.failures] == ["ProbeA"]
    assert _Probe.log == ["ProbeB:None"]


def test_runner_passes_compu_subsets(pp, probes):
    _, _, plan = probes
    plan("p", Step("ProbeA", ("delay1",)), Step("ProbeB"))
    run_plan(pp, "p")
    assert _Probe.log == ["ProbeA:('delay1',)", "ProbeB:None"]


def test_unknown_compu_in_subset_refused(pp, probes):
    _, _, plan = probes
    plan("p", Step("ProbeA", ("nope",)))
    with pytest.raises(ConfigError, match="nope"):
        run_plan(pp, "p")


def test_conf_plans_key_restricts(pp, probes):
    _, _, plan = probes
    plan("p", Step("ProbeA"))
    restricted = _with_conf(pp, prepend='plans = ["deploy", "teardown"]\n')
    with pytest.raises(ConfigError, match="'p'"):
        run_plan(restricted, "p")
    bogus = _with_conf(
        pp, replace=('plans = ["deploy", "teardown"]', 'plans = ["nope"]')
    )
    with pytest.raises(ConfigError, match="nope"):
        run_plan(bogus, "deploy")


def test_unknown_plan_refused(pp):
    with pytest.raises(ConfigError, match="nosuchplan"):
        run_plan(pp, "nosuchplan")


def test_teardown_kill_tmux_key():
    from milk.config.model import PipelineConfig

    base = {"long_name": "x", "name": "x", "loop_number": 1}
    assert PipelineConfig.model_validate(base).teardown_kill_tmux is False
    assert PipelineConfig.model_validate(
        {**base, "teardown_kill_tmux": True}
    ).teardown_kill_tmux


# ---- linkages ----


def _remotes(shm):
    for name in ("otherloop_output", "input_remote"):
        (shm / f"{name}.im.shm").write_bytes(b"")


def test_link_requires_a_deployed_pipeline(pp, shm):
    _remotes(shm)
    with pytest.raises(ColdSystemError, match="deploy"):
        pp.link("shared")


def test_link_refused_while_running(pp, paper, shm):
    _remotes(shm)
    _deploy_on_paper(pp, paper, conf=True, run=True)
    with pytest.raises(LinkageError, match="stop"):
        pp.link("shared")


def test_link_and_unlink(pp, paper, shm):
    _remotes(shm)
    _deploy_on_paper(pp, paper, conf=True)
    pp.link("shared")
    assert (shm / "input.im.shm").resolve() == (
        shm / "otherloop_output.im.shm"
    ).resolve()
    pp.link("shared")  # idempotent
    pp.unlink("shared")
    assert not (shm / "input.im.shm").is_symlink()


def test_unlink_keeps_foreign_symlinks(pp, paper, shm):
    _remotes(shm)
    _deploy_on_paper(pp, paper, conf=True)
    pp.link("shared")
    (shm / "input.im.shm").unlink()
    (shm / "input.im.shm").symlink_to(
        shm / "input_remote.im.shm"
    )  # someone else's link now
    pp.unlink("shared")
    assert (shm / "input.im.shm").is_symlink()


def test_deapply_removes_configured_links(pp, paper, shm):
    _remotes(shm)
    _deploy_on_paper(pp, paper, conf=True)
    pp.link("shared")
    pp.task_do(DeapplyLinkages)
    assert not (shm / "input.im.shm").is_symlink()
