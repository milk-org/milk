"""Plans (design-v7 §4.2): named sequences of task steps, and the one runner."""

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
class Plan:
    name: str
    steps: tuple[Step, ...]
    on_failure: OnFailure = "abort"
    accepts: Accepts = "consistent"
    compus_arg: bool = False  # the CLI takes [CU]...


PLANS: Registry[Plan] = Registry("plan")


def run(pipeline: Pipeline, plan: Plan) -> None:
    """Run any plan: build every task first (so ConfigError precedes side effects), gate, then do."""
    tasks = [(s.label, pipeline.tasks[s.task](pipeline, s.compus)) for s in plan.steps]
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
    pipeline: Pipeline, name: str, compus: Sequence[str] | None = None
) -> None:
    """compus: run every step of the plan on these compute units instead."""
    plan = pipeline.plans[name]
    if compus is not None:
        steps = tuple(replace(s, compus=tuple(compus)) for s in plan.steps)
        plan = replace(plan, steps=steps)
    run(pipeline, plan)


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
):
    PLANS.register(_plan.name, _plan)
