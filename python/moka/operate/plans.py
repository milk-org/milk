"""Plans: named sequences of task steps, and the one runner."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from collections.abc import Sequence
from typing import TYPE_CHECKING

from ..errors import (
    BestEffortError,
    MilkError,
    TaskCannotRunError,
    TaskDidNotSucceedError,
)
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


def steps_from_list(labels: Sequence[str]) -> tuple[Step, ...]:
    """The steps of these labels (see Step.parse)."""
    return tuple(map(Step.parse, labels))


@dataclass(frozen=True)
class Param:
    """A value its plan's command takes; every task of the plan gets it in ``params[name]``."""

    name: str
    type: type = str  # str, int or float
    default: object = None
    positional: bool = False  # CLI: [NAME]; else -<short>/--<name>
    short: str | None = None
    help: str = ""  # CLI: the option's help; a positional is described in the epilog
    choices: tuple[str, ...] | None = None  # CLI: the only values it accepts


@dataclass(frozen=True)
class Plan:
    name: str
    steps: tuple[Step, ...]
    on_failure: OnFailure = "abort"
    accepts: Accepts = "consistent"
    compus_arg: bool = False  # the CLI takes [CU]...
    params: tuple[Param, ...] = ()
    # CLI: one line; `moka -h` shows the first
    help: str = field(default="", compare=False)


PLANS: Registry[Plan] = Registry("plan")
_log = logging.getLogger("moka.plan")


def build_step(pipeline: Pipeline, step: Step, params: dict | None = None) -> Task:
    """ConfigError if the task or a unit the step names is missing."""
    return pipeline.tasks[step.task](pipeline, step.compus, params)


def build(
    pipeline: Pipeline, plan: Plan, params: dict | None = None
) -> list[tuple[str, Task]]:
    """(label, task) of each step, all given the same params; ConfigError if a task or unit it names is missing."""
    return [(s.label, build_step(pipeline, s, params)) for s in plan.steps]


def run(pipeline: Pipeline, plan: Plan, params: dict | None = None) -> None:
    """Run any plan: build every task first (so ConfigError precedes side effects), gate, then do.

    Tells its progress to the ``moka.plan`` logger at INFO, with ``extra={"moka": {...}}``.
    """
    start = time.monotonic()
    tasks = build(pipeline, plan, params)
    st = state(pipeline)
    labels = [label for label, _ in tasks]
    info = {"plan": plan.name, "pipeline": pipeline.short_name, "n": len(tasks)}

    def log(event: str, text: str, head: str = "", **more) -> None:
        payload = {**info, **more, "event": event, "text": text}
        _log.info(head + text, extra={"moka": payload})

    def verdict(text: str, failed: list[str]) -> None:
        seconds = time.monotonic() - start
        log("final", text, ok=not failed, failed=failed, seconds=seconds)

    head = f"{plan.name} ({st.state.value}): "
    log("plan", " -> ".join(labels), head, state=st.state.value, labels=labels)
    gate(pipeline.short_name, st, plan.accepts)
    failures: list[tuple[str, Exception]] = []
    for i, (label, task) in enumerate(tasks, 1):
        head = f"{plan.name} [{i}/{len(tasks)}] {label}: "
        log("start", "started", head, i=i, label=label)
        t0, error, reason = time.monotonic(), None, ""
        try:
            outcome = "done" if pipeline._do(task) else "already"
        except Exception as caught:
            error = caught
            refused = isinstance(error, TaskCannotRunError)
            outcome = "refused" if refused else "failed"
            known = refused or isinstance(error, TaskDidNotSucceedError)
            reason = task.failure_reason if known else str(error).split("\n")[0]
        s = time.monotonic() - t0
        text = {
            "already": "already done",
            "done": f"done ({s:.1f} s)",
            "refused": f"refused: {reason}",
            "failed": f"failed ({s:.1f} s): {reason}",
        }[outcome]
        more = {"outcome": outcome, "reason": reason, "seconds": s}
        log("end", text, head, i=i, label=label, **more)
        if isinstance(error, MilkError) and plan.on_failure == "best_effort":
            failures.append((label, error))
        elif error is not None:
            verdict(f"{plan.name}: stopped at step {i}/{len(tasks)} ({label})", [label])
            raise error
    failed = [label for label, _ in failures]
    if failed:
        n = f"{len(failed)} of {len(tasks)} steps"
        verdict(f"{plan.name}: {n} failed ({', '.join(failed)})", failed)
        raise BestEffortError(failures, pipeline=pipeline.short_name)
    verdict(f"{plan.name} done in {time.monotonic() - start:.1f} s", [])


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
        steps_from_list(["ClaimRundir", "InitFPS", "LoadData", "ConfStart"]),
        "abort",
        "cold_or_consistent",
        help="Claim the rundir, create the FPSs, load the data, start the conf processes.",
    ),
    Plan(
        "teardown",
        steps_from_list(
            [
                "RunStop",
                "DeapplyAllLinkages",
                "ConfStop",
                "KillTmux",
                "DestroyFPS",
                "ReleaseRundir",
            ]
        ),
        "best_effort",
        "any",
        help="Stop everything and release the rundir; the rootdir is kept.",
    ),
    Plan(
        "runstart",
        (Step("RunStart"),),
        "abort",
        "consistent",
        compus_arg=True,
        help="Start the run processes (of [CU]..., default: the runstart list).",
    ),
    Plan(
        "runstop",
        (Step("RunStop"),),
        "best_effort",
        "any",
        compus_arg=True,
        help="Stop the run processes (of [CU]..., default: all).",
    ),
):
    PLANS.register(_plan.name, _plan)

from ..calibs import (  # noqa: E402,F401  register their plans and tasks
    mlat as _mlat,
    linresp as _linresp,
    cm as _cm,
    loop as _loop,
    takeref as _takeref,
)
