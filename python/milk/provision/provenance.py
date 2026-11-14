"""provenance.toml (A.3): the only reader and writer, apart from confresolve's [overrides]."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import tomli
import tomli_w
from pydantic import BaseModel

from .artifacts import Fingerprint


class ArtifactRecord(BaseModel):
    """A recorded artifact; hook products also carry their tag's hook_id and decl_digest."""

    content_hash: str
    size: int
    hook_id: str | None = None
    decl_digest: str | None = None

    @property
    def fingerprint(self) -> Fingerprint:
        return Fingerprint(self.content_hash, self.size)


class Provenance(BaseModel):
    milk_version: str
    source_confdir: str
    initialized: datetime
    last_resync: datetime | None = None
    overrides: dict[str, object] = {}
    artifacts: dict[str, ArtifactRecord] = {}


def read_provenance(rootdir: Path) -> Provenance:
    with open(Path(rootdir) / "provenance.toml", "rb") as f:
        return Provenance.model_validate(tomli.load(f))


def write_provenance(rootdir: Path, prov: Provenance) -> None:
    text = tomli_w.dumps(prov.model_dump(exclude_none=True))
    (Path(rootdir) / "provenance.toml").write_text(text)
