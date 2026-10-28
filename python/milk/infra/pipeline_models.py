from __future__ import annotations

"""
Define what a pipeline config ought to be?

Use toml

Really it needs what comes from cacaovars.bash
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator
import pydantic

# TODO we should be able to import pyMilk validator without hauling ImageStreamIOWrap here.
from pyMilk.interfacing.shm import check_SHM_name


class ProcessInfoData(BaseModel):

    model_config = {"extra": "forbid"}

    # fmt: off
    RTprio            : int | None   = None  # >= 0, < 90
    cset              : str | None   = None
    taskset           : str | None   = None
    NBthread          : int | None   = None
    enabled           : bool | None  = None
    loopcntMax        : int | None   = None  # >= -1
    triggermode       : int | None   = None
    triggersname      : str | None   = None
    MeasureTiming     : bool | None  = None
    semindexrequested : int | None   = None
    triggerdelay      : float | None = None
    triggertimeout    : float | None = None
    # fmt: on


class DataLoadEntryModel(BaseModel):
    """A single dataloads entry: file to load and target stream name."""

    model_config = {"extra": "forbid"}

    source: str
    target: str


class ComputeUnitSpec(BaseModel):
    """One ``[compute_units.<cu>]`` table: its executable and its FPS parameters."""

    model_config = {"extra": "forbid", "frozen": True}

    exec: str
    fps: dict[str, Any] = Field(default_factory=dict)

    @field_validator("fps")
    @classmethod
    def _check_procinfo(cls, v: dict[str, Any]) -> dict[str, Any]:
        if "procinfo" in v:
            try:
                ProcessInfoData.model_validate(v["procinfo"])
            except pydantic.ValidationError as err:
                raise ValueError(f"procinfo: {err}") from err
        return v

    @property
    def procinfo(self) -> ProcessInfoData | None:
        if "procinfo" not in self.fps:
            return None
        return ProcessInfoData.model_validate(self.fps["procinfo"])


class PipelineConfig(BaseModel):
    """Validated conf.toml. Top-level keys outside these fields are refused."""

    model_config = {"extra": "forbid", "frozen": True}

    long_name: str
    name: str
    loop_number: int
    kind: str = "generic"
    compute_units: dict[str, ComputeUnitSpec] = Field(default_factory=dict)
    dataloads: list[DataLoadEntryModel] = Field(default_factory=list)
    # mode name -> {local stream owned by this pipeline: remote stream it points to}
    systemlinkage: dict[str, dict[str, str]] = Field(default_factory=dict)

    @field_validator("systemlinkage")
    @classmethod
    def _check_systemlinkage(cls, v: dict[str, dict[str, str]]):
        out: dict[str, dict[str, str]] = {}
        for mode, links in v.items():
            out[mode] = {}
            for local, remote in links.items():
                # TODO this where to focus if we eventually use various MILK_SHM_DIR
                local, remote = check_SHM_name(local), check_SHM_name(remote)
                if not local or not remote or local == remote:
                    raise ValueError(
                        f"systemlinkage.{mode}: invalid link {local!r} -> {remote!r}"
                    )
                out[mode][local] = remote
        return out

    def fps_name(self, cu: str) -> str:
        """FPS name of compute unit ``cu``, ``<name>_<cu>`` (design-v7 D-19).

        The only place this expression is written.
        """
        return f"{self.name}_{cu}"
