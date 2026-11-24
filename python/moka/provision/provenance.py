"""provenance.toml: the only reader and writer, apart from confresolve's [overrides]."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import tomli
import tomli_w
from pydantic import AliasChoices, BaseModel, Field

from ..errors import ProvisionError
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
    # Rootdirs made before the milk -> moka rename record milk_version.
    moka_version: str = Field(
        validation_alias=AliasChoices("moka_version", "milk_version")
    )
    source_confdir: str
    initialized: datetime
    last_resync: datetime | None = None
    overrides: dict[str, object] = {}
    artifacts: dict[str, ArtifactRecord] = {}


def read_provenance(rootdir: Path) -> Provenance:
    path = Path(rootdir) / "provenance.toml"
    if not path.is_file():
        raise ProvisionError(
            "Not a rootdir (no provenance.toml); pass the *-rootdir.", path=rootdir
        )
    with open(path, "rb") as f:
        return Provenance.model_validate(tomli.load(f))


def write_provenance(rootdir: Path, prov: Provenance) -> None:
    text = tomli_w.dumps(prov.model_dump(exclude_none=True))
    (Path(rootdir) / "provenance.toml").write_text(text)
