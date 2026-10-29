"""Plans (design-v7 §4.2): named sequences of task steps, and the one runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from ..errors import BestEffortError, ConfigError, MilkError
from ..registry import Registry
from .state import Accepts, gate, state
from .tasks import TASKS

if TYPE_CHECKING:
    from ..pipeline import Pipeline


@dataclass(frozen=True)
class Step:
    task: str
    compus: tuple[str, ...] | None = None

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
    on_failure: Literal["abort", "best_effort"] = "abort"
    accepts: Accepts = "consistent"


PLANS: Registry[Plan] = Registry("plan")


def run_plan(pipeline: Pipeline, name: str) -> None:
    plan = PLANS[name]
    allowed = pipeline.config.plans
    if allowed is not None:
        for listed in allowed:
            PLANS[listed]
        if name not in allowed:
            raise ConfigError(
                f"Plan {name!r} is not in this pipeline's plans {allowed}.",
                pipeline=pipeline.short_name,
            )
    gate(pipeline.short_name, state(pipeline), plan.accepts)
    failures: list[tuple[str, Exception]] = []
    for step in plan.steps:
        try:
            pipeline.task_do(TASKS[step.task](pipeline, step.compus))
        except MilkError as error:
            if plan.on_failure == "abort":
                raise
            failures.append((step.label, error))
    if failures:
        raise BestEffortError(failures, pipeline=pipeline.short_name)


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
                "RunStop DeapplyLinkages ConfStop KillTmux DestroyFPS ReleaseRundir".split(),
            )
        ),
        "best_effort",
        "any",
    ),
):
    PLANS.register(_plan.name, _plan)
