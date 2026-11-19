"""012 commit 2 acceptance tests (py): pipeline-local plans and tasks.

See specs/012-moka-plans-cli/brief.md §4. Plans come from `[plans.<name>]` tables of conf.toml and
from `scripts/moka_plans.py` (lists TASKS and PLANS); each Pipeline exposes them, over the built-in
ones, as `pipeline.plans` and `pipeline.tasks`. The CLI side is in cli_test.py.
"""

from __future__ import annotations

import logging
import shutil

import pytest

from moka.config.resolve import confresolve
from moka.errors import ConfigError
from moka.operate.pipeline import Pipeline
from moka.operate.plans import PLANS, Plan, Step, run, run_plan
from moka.operate.state import State, state
from moka.operate.tasks import TASKS
from moka.provision.rootdir import drift, init
from moka.registry import Registry


def _table(name: str, steps: str, more: str = "") -> str:
    return f"\n[plans.{name}]\nsteps = {steps}\n{more}\n"


# ---- Registry with a parent ----


def test_registry_child_falls_back_to_its_parent():
    parent = Registry[int]("widget")
    parent.register("a", 1)
    child = Registry[int]("widget", parent=parent)
    child.register("c", 3)
    assert child["a"] == 1 and child["c"] == 3
    assert "a" in child and "c" in child and "c" not in parent
    assert child.names() == ["a", "c"] and parent.names() == ["a"]
    parent.register("b", 2)  # registered later: still seen through the child
    assert child["b"] == 2 and child.names() == ["a", "b", "c"]
    with pytest.raises(ConfigError, match="nope.*known: a, b, c"):
        child["nope"]


def test_registry_child_overrides_its_parent_with_a_warning(caplog):
    parent = Registry[int]("widget")
    parent.register("a", 1)
    child = Registry[int]("widget", parent=parent)
    with caplog.at_level(logging.WARNING):
        child.register("a", 10)
    assert "widget" in caplog.text and "'a'" in caplog.text
    assert child["a"] == 10 and parent["a"] == 1 and child.names() == ["a"]


# ---- [plans.<name>] tables ----


def test_toml_plan_is_parsed(make_tmp_pipeline):
    steps = '["RunStart[delay0]", "WaitRunEnd[delay0,delay1]", "RunStop"]'
    more = 'on_failure = "best_effort"\naccepts = "any"'
    pp = make_tmp_pipeline(_table("warmup", steps) + _table("sweep", steps, more))
    expected = (
        Step("RunStart", ("delay0",)),
        Step("WaitRunEnd", ("delay0", "delay1")),
        Step("RunStop"),
    )
    assert pp.plans["warmup"] == Plan("warmup", expected)  # defaults as in Plan
    assert pp.plans["sweep"] == Plan("sweep", expected, "best_effort", "any")


def test_local_registries_fall_back_to_the_builtins(make_tmp_pipeline):
    pp = make_tmp_pipeline(_table("warmup", '["RunStop"]'))
    assert pp.plans["deploy"] is PLANS["deploy"]
    assert pp.tasks["RunStop"] is TASKS["RunStop"]
    assert pp.plans.names() == sorted([*PLANS.names(), "warmup"])
    assert "warmup" not in PLANS  # the module-level registry is not touched
    assert pp.plans is pp.plans and pp.tasks is pp.tasks  # built once


@pytest.mark.parametrize(
    "toml, match",
    [
        ('plans = ["deploy", "teardown"]\n', "plans"),  # the old restriction list
        (_table("x", '["RunStart["]'), "steps"),
        (_table("x", '["[delay0]"]'), "steps"),
        (_table("x", '["Run Start"]'), "steps"),
        (_table("x", '"RunStop"'), "steps"),
        ("\n[plans.x]\naccepts = 'any'\n", "steps"),
        (_table("x", '["RunStop"]', "bogus = 1"), "bogus"),
        (_table("x", '["RunStop"]', 'on_failure = "explode"'), "on_failure"),
        (_table("x", '["RunStop"]', 'accepts = "sometimes"'), "accepts"),
    ],
)
def test_toml_plan_bad_syntax(make_tmp_pipeline, tmp_path, toml, match):
    make_tmp_pipeline()
    path = tmp_path / "pipelinebasic-conf" / "conf.toml"
    text = path.read_text()
    # A top-level key goes first; a table goes last.
    path.write_text(text + toml if toml.startswith("\n") else toml + text)
    with pytest.raises(ConfigError, match=match):
        confresolve(tmp_path / "pipelinebasic-conf")


def test_toml_plan_names_are_checked_when_the_plan_is_built(make_tmp_pipeline, pp_env):
    pp = make_tmp_pipeline(
        _table("badtask", '["Nope"]', 'accepts = "any"')
        + _table("badunit", '["RunStop[ghost]"]', 'accepts = "any"')
    )
    assert pp.plans["badtask"].steps == (Step("Nope"),)
    with pytest.raises(ConfigError, match="Nope"):
        run_plan(pp, "badtask")
    with pytest.raises(ConfigError, match="ghost"):
        run_plan(pp, "badunit")


# ---- scripts/moka_plans.py ----


def test_python_file_declares_tasks_and_plans(make_tmp_pipeline, touch_source, pp_env):
    pp = make_tmp_pipeline(moka_plans=touch_source)
    assert (pp.root_folder / "conf" / "scripts" / "moka_plans.py").is_file()
    assert pp.tasks["Touch"].__name__ == "Touch"
    assert pp.plans["touch"] == Plan("touch", (Step("Touch"),), accepts="any")
    assert "Touch" not in TASKS and "touch" not in PLANS
    rows = drift(
        pp.root_folder
    ).artifacts  # importing the file leaves no trace in conf/
    assert all(row.source == row.local == "unchanged" for row in rows)

    run_plan(pp, "touch")  # the plan and its task resolve through the pipeline
    touched = pp.work_folder / "touched"
    assert touched.exists()
    touched.unlink()
    run(pp, Plan("adhoc", (Step("Touch"),), accepts="any"))
    assert touched.exists()


def test_toml_plan_may_use_a_python_task(make_tmp_pipeline, touch_source, pp_env):
    pp = make_tmp_pipeline(_table("t2", '["Touch"]', 'accepts = "any"'), touch_source)
    run_plan(pp, "t2")
    assert (pp.work_folder / "touched").exists()


def test_python_file_is_imported_lazily(make_tmp_pipeline, pp_env):
    pp = make_tmp_pipeline(moka_plans="raise RuntimeError('boom')\n")  # init: no import
    pp = Pipeline(pp.root_folder)  # constructs
    assert state(pp).state is State.COLD
    with pytest.raises(Exception, match="boom"):
        pp.plans
    with pytest.raises(Exception, match="boom"):
        Pipeline(pp.root_folder).tasks


# ---- overrides ----


def test_toml_plan_overrides_a_builtin_with_a_warning(
    make_tmp_pipeline, pp_env, caplog
):
    pp = make_tmp_pipeline(_table("deploy", '["ReleaseRundir"]', 'accepts = "any"'))
    with caplog.at_level(logging.WARNING):
        assert pp.plans["deploy"].steps == (Step("ReleaseRundir"),)
    assert "'deploy'" in caplog.text
    assert len(PLANS["deploy"].steps) == 4  # the built-in is intact
    run_plan(pp, "deploy")  # runs the local one: nothing is deployed
    assert state(pp).state is State.COLD


def test_python_plan_overrides_a_toml_plan_with_a_warning(
    make_tmp_pipeline, touch_source, caplog
):
    pp = make_tmp_pipeline(_table("touch", '["RunStop"]'), touch_source)
    with caplog.at_level(logging.WARNING):
        assert pp.plans["touch"].steps == (Step("Touch"),)
    assert "'touch'" in caplog.text


def test_python_task_overrides_a_builtin_with_a_warning(
    make_tmp_pipeline, touch_source, caplog
):
    source = touch_source.replace("Touch", "RunStop")
    pp = make_tmp_pipeline(moka_plans=source)
    with caplog.at_level(logging.WARNING):
        assert pp.tasks["RunStop"] is not TASKS["RunStop"]
    assert "'RunStop'" in caplog.text


# ---- two pipelines in one process ----


def test_local_definitions_do_not_leak_between_pipelines(
    make_tmp_pipeline, touch_source, tmp_path
):
    one = make_tmp_pipeline(moka_plans=touch_source)
    conf = tmp_path / "other" / "pipelinebasic-conf"
    shutil.copytree(tmp_path / "pipelinebasic-conf", conf)
    renamed = touch_source.replace("Touch", "Poke").replace("touch", "poke")
    (conf / "scripts" / "moka_plans.py").write_text(renamed)
    two = Pipeline(init(conf))

    assert "touch" in one.plans and "Touch" in one.tasks
    assert "poke" in two.plans and "Poke" in two.tasks
    assert "poke" not in one.plans and "Poke" not in one.tasks
    assert "touch" not in two.plans and "Touch" not in two.tasks
    assert not {"touch", "poke"} & set(PLANS.names())
    assert not {"Touch", "Poke"} & set(TASKS.names())
