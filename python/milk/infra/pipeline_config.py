from __future__ import annotations

"""
Define what a pipeline config ought to be?

Use toml

Really it needs what comes from cacaovars.bash
"""

from pathlib import Path

from dataclasses import dataclass, fields

import tomli, tomli_w
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

    RTprio: int | None = None  # >= 0, < 90
    cset: str | None = None
    taskset: str | None = None
    NBthread: int | None = None
    enabled: bool | None = None
    loopcntMax: int | None = None  # >= -1
    triggermode: int | None = None
    triggersname: str | None = None
    MeasureTiming: bool | None = None
    semindexrequested: int | None = None
    triggerdelay: float | None = None
    triggertimeout: float | None = None


class PipelineConfigModel(BaseModel):
    """Schema for conf.toml: validates required top-level keys."""

    model_config = {"extra": "allow"}

    long_name: str
    name: str
    loop_number: int
    sessions: dict[str, str]
    session_configs_: dict[str, dict] = Field(default_factory=dict)

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

        # Really I want schemas and dynamic parsing...


def load_pipeline_config(conf_folder: str | Path) -> PipelineConfig:
    with open(Path(conf_folder) / "conf.toml", "rb") as f:
        data = tomli.load(f)

    parsed = PipelineConfigModel(**data)  # Will raise ValidationError

    config = PipelineConfig()
    config.name = parsed.name
    config.loop_number = parsed.loop_number
    config.sessions = parsed.sessions
    # Each session name is also a top-level table holding its own config
    config.session_configs = {
        session_name: data[session_name] for session_name in config.sessions
    }

    return config
