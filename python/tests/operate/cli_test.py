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


# ---- pipeline-local plans (specs/012-moka-plans-cli/brief.md §4) ----

PROBE = '\n[plans.probe]\nsteps = ["ReleaseRundir"]\naccepts = "any"\n'
BUILTIN = ("deploy", "teardown", "runstart", "runstop", "state", "link", "nuke")


def _commands(result) -> list[str]:
    """The command names of a help output: the first word of its two-space-indented rows."""
    rows = result.output.split("Commands:")[1].splitlines()
    return [row.split()[0] for row in rows if row[:2] == "  " and row[2:3].strip()]


@pytest.fixture
def loads(monkeypatch) -> list:
    """The confresolve calls made by Pipeline(): one per pipeline load."""
    import moka.operate.pipeline as module

    seen: list = []
    real = module.confresolve
    monkeypatch.setattr(
        module, "confresolve", lambda *args: seen.append(args) or real(*args)
    )
    return seen


def test_local_plan_is_a_command_with_p(make_tmp_pipeline, pp_env, loads):
    pp = make_tmp_pipeline(PROBE)
    del loads[:]
    for args in (("probe", "-p", str(pp.root_folder)), ("-p", pp.root_folder, "probe")):
        result = _invoke(*map(str, args))
        assert result.exit_code == 0, _text(result)
    assert len(loads) == 2  # the pipeline is loaded once per invocation
    help = _invoke("-h", "-p", str(pp.root_folder))
    assert help.exit_code == 0, _text(help)
    assert "probe" in _commands(help) and set(BUILTIN) <= set(_commands(help))


def test_a_pipeline_that_does_not_load_is_tried_once_and_says_why(pp, loads, tmp_path):
    marker = tmp_path / "imports"
    scripts = pp.root_folder / "conf" / "scripts"
    scripts.mkdir()
    (scripts / "moka_plans.py").write_text(
        f"open({str(marker)!r}, 'a').write('x')\nraise RuntimeError('boom')\n"
    )
    assert _invoke("-h", "-p", str(pp.root_folder)).exit_code == 0
    assert marker.read_text() == "x"  # one import for the whole help
    result = _invoke("touch", "-p", str(pp.root_folder))  # not a built-in: say why
    assert result.exit_code != 0 and "boom" in _text(result)
    del loads[:]
    result = _invoke("touch", "-p", "nope")
    assert result.exit_code != 0 and "rootdir" in _text(result)
    assert len(loads) == 0  # "nope" is no rootdir: nothing to resolve
    result = _invoke("nosuchplan", "-p", str(tmp_path))
    assert result.exit_code != 0 and "rootdir" in _text(result)


def test_unknown_command_on_a_good_pipeline(make_tmp_pipeline, pp_env):
    pp = make_tmp_pipeline(PROBE)
    result = _invoke("nosuchplan", "-p", str(pp.root_folder))
    assert result.exit_code == 2 and "No such command" in _text(result)


def test_local_plan_is_a_command_from_cwd(make_tmp_pipeline, pp_env, monkeypatch):
    pp = make_tmp_pipeline(PROBE)
    monkeypatch.chdir(pp.root_folder)
    assert "probe" in _commands(_invoke("-h"))
    result = _invoke("probe")
    assert result.exit_code == 0, _text(result)


def test_local_plans_do_not_leak_into_other_invocations(
    make_tmp_pipeline, pp_env, tmp_path, monkeypatch
):
    pp = make_tmp_pipeline(PROBE)
    assert "probe" in _commands(_invoke("-h", "-p", str(pp.root_folder)))
    monkeypatch.chdir(tmp_path)  # not a rootdir
    assert "probe" not in _commands(_invoke("-h"))


def test_help_without_a_resolvable_pipeline_lists_the_builtins(
    pp, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)  # not a rootdir
    for args in (("-h",), ("-h", "-p", "nope"), ("--help", "--pipe", str(tmp_path))):
        result = _invoke(*args)
        assert result.exit_code == 0, _text(result)
        assert set(BUILTIN) <= set(_commands(result))


@pytest.mark.parametrize("breakage", ["conf", "moka_plans"])
def test_help_with_a_pipeline_that_does_not_load_lists_the_builtins(pp, breakage):
    if breakage == "conf":
        (pp.root_folder / "conf" / "conf.toml").write_text("this is not toml")
    else:
        scripts = pp.root_folder / "conf" / "scripts"
        scripts.mkdir()
        (scripts / "moka_plans.py").write_text("raise RuntimeError('boom')\n")
    result = _invoke("-h", "-p", str(pp.root_folder))
    assert result.exit_code == 0, _text(result)
    assert set(BUILTIN) <= set(_commands(result))


def test_local_plan_named_like_a_fixed_command_is_refused(make_tmp_pipeline, pp_env):
    pp = make_tmp_pipeline(PROBE.replace("plans.probe", "plans.state"))
    for args in (("-h",), ("state",)):
        result = _invoke(*args, "-p", str(pp.root_folder))
        assert result.exit_code != 0 and "state" in _text(result)
        assert "clash" in _text(result)


# ---- moka plans (specs/012-moka-plans-cli/brief.md §5) ----

LOCAL = (
    PROBE
    + '\n[plans.ghost]\nsteps = ["RunStop[ghost]"]\n'
    + '\n[plans.notask]\nsteps = ["Nope"]\n'
    + '\n[plans.teardown]\nsteps = ["RunStop", "ReleaseRundir"]\naccepts = "any"\n'
)


def _row(result, name: str) -> str:
    """The one output line whose first word is ``name``."""
    (row,) = [line for line in result.output.splitlines() if line.split()[:1] == [name]]
    return row


def test_plans_is_a_fixed_command():
    lines = [line.strip() for line in _invoke("--help").output.splitlines()]
    assert "plans [-p PIPE]" in lines


def test_plans_on_a_cold_pipeline(pp):
    result = _invoke("plans", "-p", str(pp.root_folder))
    assert result.exit_code == 0, _text(result)
    assert "basic: cold" in result.output.splitlines()
    deploy = _row(result, "deploy")
    assert "ClaimRundir InitFPS LoadData ConfStart" in deploy and deploy.endswith("ok")
    assert _row(result, "teardown").endswith("ok")  # accepts any state
    assert _row(result, "runstop").endswith("ok")
    runstart = _row(result, "runstart")
    assert "RunStart" in runstart and runstart.endswith("no (cold; run deploy)")
    for name in ("deploy", "teardown", "runstart", "runstop"):
        assert "local" not in _row(result, name)


def test_plans_on_a_configured_pipeline(pp, paper):
    _deploy_on_paper(pp, paper)
    result = _invoke("plans", "-p", str(pp.root_folder))
    assert result.exit_code == 0, _text(result)
    assert "basic: configured" in result.output.splitlines()
    for name in ("deploy", "teardown", "runstart", "runstop"):
        assert _row(result, name).endswith("ok")


def test_plans_on_an_inconsistent_pipeline(pp, paper):
    paper["basic_delay0"] = (False, False)  # an FPS without a rundir
    result = _invoke("plans", "-p", str(pp.root_folder))
    assert result.exit_code == 0, _text(result)
    assert _row(result, "teardown").endswith("ok")
    deploy = _row(result, "deploy")
    assert "no (inconsistent" in deploy and deploy.endswith("run teardown)")


def test_plans_shows_local_plans_and_what_cannot_be_built(make_tmp_pipeline, pp_env):
    pp = make_tmp_pipeline(LOCAL)
    result = _invoke("plans", "-p", str(pp.root_folder))
    assert result.exit_code == 0, _text(result)
    probe = _row(result, "probe")
    assert "(local)" in probe and "ReleaseRundir" in probe and probe.endswith("ok")
    teardown = _row(result, "teardown")
    assert "(local, overrides built-in)" in teardown
    assert "RunStop ReleaseRundir" in teardown and "DestroyFPS" not in teardown
    assert _row(result, "ghost") == (
        "ghost (local)  RunStop[ghost]  n/a (Unknown compute units ['ghost'] for RunStop)"
    )
    assert _row(result, "notask") == "notask (local)  Nope  n/a (Unknown task 'Nope')"
    assert "local" not in _row(result, "deploy")


def test_plans_needs_a_pipeline(pp, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # not a rootdir
    for args in (("plans",), ("plans", "-p", "nope")):
        result = _invoke(*args)
        assert result.exit_code == 1 and "rootdir" in _text(result)
    scripts = pp.root_folder / "conf" / "scripts"
    scripts.mkdir()
    (scripts / "moka_plans.py").write_text("raise RuntimeError('boom')\n")
    result = _invoke("plans", "-p", str(pp.root_folder))
    assert result.exit_code == 1 and "boom" in _text(result)


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
