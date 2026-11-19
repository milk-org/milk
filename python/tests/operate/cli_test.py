"""R4b acceptance tests (py): the moka CLI over the plan registry.

See specs/010-r4b-operate-cli/brief.md. FPSs are simulated (`paper`), rundirs live in a self-cleaning
MILK_SHM_DIR (`shm_dir_cleanafter`), and the boot id is pinned (`boot`); see tests/conftestaux/operate.py.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner


from moka.errors import BestEffortError
from moka.operate.cli import moka
from moka.operate.plans import PLANS, Plan, Step
from moka.operate.state import State, state
from moka.operate.tasks import ClaimRundir

FPS = ("basic_delay0", "basic_delay1")


@pytest.fixture
def pp(make_tmp_pipeline, shm_dir_cleanafter, boot, paper):
    """pipelinebasic, initialized from tmp_path/pipelinebasic-conf, cold, simulated FPSs."""
    return make_tmp_pipeline()


@pytest.fixture
def plans(monkeypatch):
    """Register throwaway plans; the registry is restored after the test."""
    monkeypatch.setattr(PLANS, "_items", dict(PLANS._items))
    return lambda name: PLANS.register(
        name, Plan(name, (Step("ReleaseRundir"),), accepts="any")
    )


@pytest.fixture
def calls(monkeypatch):
    """Record run_plan(pipeline, name) calls made by the CLI instead of running them."""
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "moka.operate.plans.run_plan",
        lambda pipeline, name, compus=None: seen.append(
            (str(pipeline.root_folder), name)
        ),
    )
    return seen


def _invoke(*args: str, input: str | None = None):
    return CliRunner().invoke(moka, list(args), input=input)


def _text(result) -> str:
    """Everything the user would see, including an uncaught exception."""
    return result.output + ("" if result.exception is None else str(result.exception))


def _deploy_on_paper(pp, paper, conf=True, run=False):
    pp.task_do(ClaimRundir)
    for name in FPS:
        paper[name] = (conf, run)


# ---- command set ----


def test_help_lists_plans_and_fixed_commands():
    result = _invoke("--help")
    assert result.exit_code == 0, result.output
    for name in ("deploy", "teardown", "state", "link", "unlink", "nuke"):
        assert name in result.output
    assert "deploy [-p PIPE]" in result.output
    assert "unlink [-p PIPE] LINKAGE" in result.output
    for gone in ("spool", "sys"):
        assert _invoke(gone, "-h").exit_code != 0


def test_plans_become_commands_when_registered(pp, plans, calls):
    plans("probe")  # registered after the CLI module was imported
    assert "probe" in _invoke("--help").output
    result = _invoke("probe", "-p", str(pp.root_folder))
    assert result.exit_code == 0, _text(result)
    assert calls == [(str(pp.root_folder), "probe")]


def test_no_pipe_means_cwd(pp, plans, calls, monkeypatch, tmp_path):
    plans("probe")
    monkeypatch.chdir(pp.root_folder)
    result = _invoke("probe")
    assert result.exit_code == 0, _text(result)
    assert calls == [(str(pp.root_folder), "probe")]
    assert f"rootdir: {pp.root_folder}" in result.output
    monkeypatch.chdir(tmp_path)
    result = _invoke("probe")
    assert result.exit_code == 1 and "is not a rootdir" in result.output
    assert _invoke("probe", "-p", "nope").exit_code == 1


def test_plan_command_targets(pp, calls, monkeypatch):
    assert (
        _invoke("deploy", "-p", str(pp.root_folder.parent / "basic-rootdir")).exit_code
        == 0
    )
    monkeypatch.chdir(pp.root_folder.parent)
    assert _invoke("teardown", "--pipe", "basic").exit_code == 0
    assert calls == [(str(pp.root_folder), "deploy"), (str(pp.root_folder), "teardown")]


def test_plan_named_like_a_fixed_command_is_refused(plans):
    plans("state")
    result = _invoke("--help")
    assert result.exit_code != 0 and "state" in _text(result)


def test_plan_errors_reach_the_user(pp):
    result = _invoke("teardown", "-p", str(pp.root_folder))  # real run_plan, cold: fine
    assert result.exit_code == 0, _text(result)
    refused = _invoke("link", "-p", str(pp.root_folder), "shared")  # cold: refused
    assert refused.exit_code != 0 and "deploy" in _text(refused)


def test_best_effort_failures_are_warnings_and_exit_1(pp, monkeypatch):
    def failing(pipeline, name, compus=None):
        raise BestEffortError([("RunStop", RuntimeError("boom"))], pipeline="basic")

    monkeypatch.setattr("moka.operate.plans.run_plan", failing)
    result = _invoke("teardown", "-p", str(pp.root_folder))
    assert result.exit_code == 1
    assert "RunStop" in result.stderr and "boom" in result.stderr


# ---- runstart / runstop take compute units (specs/012-moka-plans-cli/brief.md §3) ----


def test_help_shows_the_cu_arguments_of_the_two_plans_only():
    lines = [line.strip() for line in _invoke("--help").output.splitlines()]
    assert "runstart [-p PIPE] [CU]..." in lines
    assert "runstop [-p PIPE] [CU]..." in lines
    assert "deploy [-p PIPE]" in lines and "teardown [-p PIPE]" in lines


def test_runstart_without_cu_runs_the_plan_as_registered(pp, paper, started):
    _deploy_on_paper(pp, paper)
    result = _invoke("runstart", "-p", str(pp.root_folder))
    assert result.exit_code == 0, _text(result)
    assert started == ["delay0", "delay1"]


def test_runstart_and_runstop_take_compute_units(pp, paper, started, fake_compus):
    _deploy_on_paper(pp, paper)
    result = _invoke("runstart", "-p", str(pp.root_folder), "delay1")
    assert result.exit_code == 0, _text(result)
    assert started == ["delay1"]
    fake_compus["delay0"].run = True
    result = _invoke("runstop", "-p", str(pp.root_folder), "delay0")
    assert result.exit_code == 0, _text(result)
    assert fake_compus["delay0"].calls == ["runstop"]
    assert fake_compus["delay1"].run  # not named: left running


def test_runstart_refuses_an_unknown_unit(pp, paper, started):
    _deploy_on_paper(pp, paper)
    result = _invoke("runstart", "-p", str(pp.root_folder), "nope")
    assert result.exit_code != 0 and "nope" in _text(result)
    assert started == []


@pytest.mark.parametrize("plan", ["deploy", "teardown"])
def test_other_plans_refuse_compute_units(pp, plan):
    result = _invoke(plan, "-p", str(pp.root_folder), "delay0")
    assert result.exit_code == 2, _text(result)  # click usage error
    assert state(pp).state is State.COLD  # nothing ran


# ---- state ----


def test_state_cold_and_clean(pp):
    result = _invoke("state", "-p", str(pp.root_folder))
    assert result.exit_code == 0, _text(result)
    assert "cold" in result.output
    assert "drift: none" in result.output


def test_state_inconsistent_shows_reasons(pp, paper):
    paper["basic_delay0"] = (False, False)  # an FPS without a rundir
    result = _invoke("state", "-p", str(pp.root_folder))
    assert result.exit_code == 0, _text(result)  # state never fails on a pipeline state
    assert "inconsistent" in result.output and "rundir" in result.output


def test_state_configured(pp, paper):
    _deploy_on_paper(pp, paper)
    assert "configured" in _invoke("state", "-p", str(pp.root_folder)).output


def test_state_summarises_drift(pp, tmp_path):
    toml = tmp_path / "pipelinebasic-conf" / "conf.toml"
    toml.write_text("# edited in the confdir\n" + toml.read_text())
    result = _invoke("state", "-p", str(pp.root_folder))
    assert result.exit_code == 0, _text(result)
    assert "drift: 1" in result.output and "mokainit drift" in result.output


def test_state_without_its_confdir(pp, tmp_path):
    import shutil

    shutil.rmtree(tmp_path / "pipelinebasic-conf")
    result = _invoke("state", "-p", str(pp.root_folder))
    assert result.exit_code == 0, _text(result)
    assert "drift: unavailable" in result.output


def test_state_reports_moka_version_change(pp):
    prov = pp.root_folder / "provenance.toml"
    text = prov.read_text()
    start = text.index('moka_version = "') + len('moka_version = "')
    end = text.index('"', start)
    prov.write_text(text[:start] + "0.0.0-old" + text[end:])
    output = _invoke("state", "-p", str(pp.root_folder)).output
    assert "0.0.0-old" in output


# ---- link / unlink ----


def test_link_and_unlink(pp, paper, shm_dir_cleanafter):
    for name in ("otherloop_output", "input_remote"):
        (shm_dir_cleanafter / f"{name}.im.shm").write_bytes(b"")
    _deploy_on_paper(pp, paper)
    result = _invoke("link", "-p", str(pp.root_folder), "shared")
    assert result.exit_code == 0, _text(result)
    assert (shm_dir_cleanafter / "input.im.shm").is_symlink()
    result = _invoke("unlink", "-p", str(pp.root_folder), "shared")
    assert result.exit_code == 0, _text(result)
    assert not (shm_dir_cleanafter / "input.im.shm").is_symlink()


def test_unlink_all(pp, paper, shm_dir_cleanafter):
    _deploy_on_paper(pp, paper)
    assert _invoke("link", "-p", str(pp.root_folder), "shared").exit_code == 0
    result = _invoke("unlink", "-p", str(pp.root_folder), "all")
    assert result.exit_code == 0, result.output
    assert pp.linked() == []


def test_unknown_linkage(pp, paper):
    _deploy_on_paper(pp, paper)
    result = _invoke("link", "-p", str(pp.root_folder), "nope")
    assert result.exit_code != 0 and "nope" in _text(result)


def test_link_without_linkage_lists_them(pp):
    result = _invoke("link", "-p", str(pp.root_folder))
    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[1:] == list(pp.systemlinkage)


# ---- nuke ----


def test_nuke_deletes_even_when_teardown_fails(pp, monkeypatch):
    def failing(pipeline, name):
        raise BestEffortError([("ConfStop", RuntimeError("stuck"))], pipeline="basic")

    monkeypatch.setattr("moka.operate.pipeline.run_plan", failing)
    result = _invoke("nuke", "-p", str(pp.root_folder), "--yes")
    assert result.exit_code == 1 and "stuck" in result.output
    assert not pp.root_folder.exists()
