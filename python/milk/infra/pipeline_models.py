from __future__ import annotations

"""
Define what a pipeline config ought to be?

Use toml

Really it needs what comes from cacaovars.bash
"""

from pathlib import Path

from dataclasses import dataclass, fields

from pydantic import BaseModel, Field, field_validator, model_validator


@dataclass(frozen=True)
class LOOP_INFO:
    full_name: str
    n: int

    # Defining __iter__ to allow unpacking a LOOP_INFO object as if a tuple
    def __iter__(self):
        return (getattr(self, field.name) for field in fields(self))


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

    @classmethod
    def from_str(cls, raw: str) -> DataLoadEntryModel:
        """Parse a "source -- target" formatted dataloads entry."""
        parts = raw.split("--")
        if len(parts) != 2:
            raise ValueError(
                f"dataloads entry {raw!r} must be formatted as 'source -- target'"
            )
        source, target = (p.strip() for p in parts)
        return cls(source=source, target=target)


class PipelineConfigModel(BaseModel):
    """Schema for conf.toml: validates required top-level keys."""

    model_config = {"extra": "allow"}

    long_name: str
    name: str
    loop_number: int
    sessions: dict[str, str]
    dataloads: list[DataLoadEntryModel] = Field(default_factory=list)
    session_configs_: dict[str, dict] = Field(default_factory=dict)

    @field_validator("dataloads", mode="before")
    @classmethod
    def _parse_dataloads(cls, v):
        if not isinstance(v, list):
            return v
        return [
            DataLoadEntryModel.from_str(item) if isinstance(item, str) else item
            for item in v
        ]

    @model_validator(mode="after")
    def _check_session_configs_exist(self) -> PipelineConfigModel:
        extra = self.model_extra or {}
        missing = [name for name in self.sessions if name not in extra]
        if missing:
            raise ValueError(
                f"session(s) {missing} declared in [sessions] have no matching config table"
            )
        return self

    @model_validator(mode="after")
    def _parse_session_dicts_and_check_procinfo_specs(self) -> PipelineConfigModel:
        for name in self.sessions:
            if self.model_extra and name in self.model_extra:
                self.session_configs_[name] = self.model_extra[name]
                if "procinfo" in self.session_configs_[name]:
                    # Trigger schema errors but don't override the dict
                    ProcessInfoData(**self.session_configs_[name]["procinfo"])
        return self


class PipelineConfig:

    def __init__(self):
        # no long name, long name is the folder loading.
        # However, should probably still store it here...
        self.name: str = "shortloopname"

        self.loop_number: int = 0
        self.dm_number: int | None = None

        # [sessions] table: session name -> executable name
        self.sessions: dict[str, str] = {}
        # per-session config table, e.g. [delay0]
        self.session_configs: dict[str, dict] = {}

        # [dataloads] array: list of source/target entries
        self.dataloads: list[DataLoadEntryModel] = []

        # Really I want schemas and dynamic parsing...
