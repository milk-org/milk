"""Plans: named sequences of task steps, and the one runner."""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Sequence
from typing import TYPE_CHECKING

from ..errors import BestEffortError, MilkError
from ..registry import Registry
from ..config.model import Accepts, OnFailure
from .state import gate, state

if TYPE_CHECKING:
    from .pipeline import Pipeline
    from .tasks import Task


@dataclass(frozen=True)
class Step:
    task: str
    compus: tuple[str, ...] | None = None

    @classmethod
    def parse(cls, label: str) -> Step:
        """The inverse of label."""
        task, _, compus = label.partition("[")
        return cls(task, tuple(compus.rstrip("]").split(",")) if compus else None)

    @property
    def label(self) -> str:
        return (
            self.task
            if self.compus is None
            else f"{self.task}[{','.join(self.compus)}]"
        )


@dataclass(frozen=True)
class Param:
    """A value its plan's command takes; every task of the plan gets it in ``params[name]``."""

    name: str
    type: type = str  # str, int or float
    default: object = None
    positional: bool = False  # CLI: [NAME]; else -<short>/--<name>
    short: str | None = None


@dataclass(frozen=True)
class Plan:
    name: str
    steps: tuple[Step, ...]
    on_failure: OnFailure = "abort"
    accepts: Accepts = "consistent"
    compus_arg: bool = False  # the CLI takes [CU]...
    params: tuple[Param, ...] = ()


PLANS: Registry[Plan] = Registry("plan")


def build_step(pipeline: Pipeline, step: Step, params: dict | None = None) -> Task:
    """ConfigError if the task or a unit the step names is missing."""
    return pipeline.tasks[step.task](pipeline, step.compus, params)


def build(
    pipeline: Pipeline, plan: Plan, params: dict | None = None
) -> list[tuple[str, Task]]:
    """(label, task) of each step, all given the same params; ConfigError if a task or unit it names is missing."""
    return [(s.label, build_step(pipeline, s, params)) for s in plan.steps]


def run(pipeline: Pipeline, plan: Plan, params: dict | None = None) -> None:
    """Run any plan: build every task first (so ConfigError precedes side effects), gate, then do."""
    tasks = build(pipeline, plan, params)
    gate(pipeline.short_name, state(pipeline), plan.accepts)
    failures: list[tuple[str, Exception]] = []
    for label, task in tasks:
        try:
            pipeline.task_do(task)
        except MilkError as error:
            if plan.on_failure == "abort":
                raise
            failures.append((label, error))
    if failures:
        raise BestEffortError(failures, pipeline=pipeline.short_name)


def run_plan(
    pipeline: Pipeline,
    name: str,
    compus: Sequence[str] | None = None,
    params: dict | None = None,
) -> None:
    """compus: run every step of the plan on these compute units instead; params: see Param."""
    plan = pipeline.plans[name]
    if compus is not None:
        steps = tuple(replace(s, compus=tuple(compus)) for s in plan.steps)
        plan = replace(plan, steps=steps)
    run(pipeline, plan, params)


for _plan in (
    Plan(
        "deploy",
        tuple(map(Step, "ClaimRundir InitFPS LoadData ConfStart".split())),
        "abort",
        "cold_or_consistent",
    ),
    Plan(
        "teardown",
        tuple(
            map(
                Step,
                "RunStop DeapplyAllLinkages ConfStop KillTmux DestroyFPS ReleaseRundir".split(),
            )
        ),
        "best_effort",
        "any",
    ),
    Plan("runstart", (Step("RunStart"),), "abort", "consistent", compus_arg=True),
    Plan("runstop", (Step("RunStop"),), "best_effort", "any", compus_arg=True),
    Plan(
        "mlat",
        (
            Step("RunStart", ("acquWFS",)),
            Step("RunStart", ("mlat",)),
            Step("WaitRunEnd", ("mlat",)),
        ),
    ),
):
    PLANS.register(_plan.name, _plan)

from ..calibs import (
    linresp as _linresp,
)  # noqa: E402,F401  registers its plans and tasks
