"""The derived pipeline state (design-v7 §4.1): rundir and live FPSs, classified."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

from pyMilk.interfacing.fps import FPS

from .. import env
from ..errors import ColdSystemError, InconsistentStateError
from . import rundir

if TYPE_CHECKING:
    from .pipeline import Pipeline


Accepts = Literal["cold_or_consistent", "consistent", "any"]


class State(str, Enum):
    COLD = "cold"
    INCONSISTENT = "inconsistent"
    DEPLOYED = "deployed"
    CONFIGURED = "configured"
    RUNNING = "running"
    MIXED = "mixed"


@dataclass(frozen=True)
class PipelineState:
    state: State
    reasons: tuple[str, ...] = ()


def fps_status(name: str) -> tuple[bool, bool] | None:
    """(conf_running, run_running) of FPS ``name``, or None if it does not exist."""
    if not (env.shm_dir() / f"{name}.fps.shm").is_file():
        return None
    fps = FPS(name)
    return fps.conf_isrunning(), fps.run_isrunning()


def state(pipeline: Pipeline) -> PipelineState:
    config = pipeline.config
    name = config.name
    status = {
        config.fps_name(cu): fps_status(config.fps_name(cu))
        for cu in config.compute_units
    }
    claim = rundir.read_claim(name)
    if claim is None:
        present = [n for n, s in status.items() if s is not None]
        if present:
            return PipelineState(
                State.INCONSISTENT,
                (f"FPS {', '.join(present)} exist without a rundir",),
            )
        return PipelineState(State.COLD)

    reasons: tuple[str, ...] = ()
    if claim.rootdir != str(pipeline.root_folder):
        reasons = (f"claimed by {claim.rootdir}",)
    if rundir.read_boot_id(name) != rundir.current_boot_id():
        reasons += ("stale rundir from a previous boot",)
        return PipelineState(State.INCONSISTENT, reasons)
    missing = [n for n, s in status.items() if s is None]
    if missing:
        reasons += (f"FPS missing: {', '.join(missing)}",)
        return PipelineState(State.INCONSISTENT, reasons)
    conf = [s[0] for s in status.values()]
    run = [s[1] for s in status.values()]
    if all(run):
        st = State.RUNNING
    elif not any(run) and all(conf):
        st = State.CONFIGURED
    elif not any(run) and not any(conf):
        st = State.DEPLOYED
    else:
        st = State.MIXED
    return PipelineState(st, reasons)


def gate(name: str, st: PipelineState, accepts: Accepts) -> None:
    """Raise unless ``st`` is acceptable to ``accepts``."""
    if accepts == "any":
        return
    if st.state is State.INCONSISTENT:
        raise InconsistentStateError(
            f"Pipeline {name} is inconsistent ({'; '.join(st.reasons)}); run teardown.",
            pipeline=name,
        )
    if st.state is State.COLD and accepts == "consistent":
        raise ColdSystemError(f"Pipeline {name} is cold; run deploy.", pipeline=name)


def require(pipeline: Pipeline, accepts: Accepts) -> PipelineState:
    st = state(pipeline)
    gate(pipeline.short_name, st, accepts)
    return st
