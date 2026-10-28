"""Conf hooks: tagfiles (A.2), the hook runners, and consolidate (design-v7 §0, D-6)."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import tomli
import tomli_w
from pydantic import BaseModel, ValidationError

from ..config import confresolve
from ..config.model import CommandHookDecl, FetchHookDecl
from ..errors import ProvisionError
from .artifacts import Fingerprint, decl_digest, fingerprint, status


def _tagfile(product: Path) -> Path:
    return product.with_name(product.name + ".tag")


class Tag(BaseModel):
    """A.2 tagfile, ``data/<product>.tag``. ``source`` and ``time`` are informational."""

    model_config = {"extra": "forbid", "frozen": True}

    hook_id: str
    decl_digest: str
    content_hash: str
    size: int
    source: str
    time: datetime

    @property
    def fingerprint(self) -> Fingerprint:
        return Fingerprint(self.content_hash, self.size)


def _load_tag(product: Path) -> Tag | None:
    path = _tagfile(product)
    if not path.is_file():
        return None
    with open(path, "rb") as f:
        try:
            return Tag.model_validate(tomli.load(f))
        except (tomli.TOMLDecodeError, ValidationError) as err:
            raise ProvisionError(f"Invalid tagfile: {err}", path=path) from err


def read_tag(product: Path) -> dict | None:
    """The A.2 tag of ``product``, or None when it has none."""
    tag = _load_tag(Path(product))
    return tag and tag.model_dump()


def _run(hook_id: str, argv: list[str], cwd: Path) -> None:
    result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ProvisionError(
            f"Hook {hook_id!r} failed.", cmd=shlex.join(argv), stderr=result.stderr
        )


def _fetch(decl: FetchHookDecl, confdir: Path) -> None:
    source = decl.source if ":" in decl.source else str(confdir / decl.source)
    if os.path.isdir(source):
        source += "/"
    _run(decl.id, ["rsync", "-a", "--delete", source, decl.product], confdir / "data")


def _command(decl: CommandHookDecl, confdir: Path) -> None:
    _run(decl.id, decl.command, confdir / "data")


RUNNERS = {"fetch": _fetch, "command": _command}


def consolidate(confdir: Path) -> list[str]:
    """Run the pending hooks of ``confdir``; returns their ids, in declaration order."""
    confdir = Path(confdir)
    data = confdir / "data"
    ran = []
    for decl in confresolve(confdir).hooks:
        product = data / decl.product
        digest = decl_digest(decl.model_dump(mode="json"))
        tag = _load_tag(product)
        recorded = tag and tag.fingerprint
        current = fingerprint(product) if os.path.lexists(product) else None
        if status(recorded, current, tag and tag.decl_digest, digest) == "unchanged":
            continue

        if product.is_dir() and not product.is_symlink():
            shutil.rmtree(product)
        else:
            product.unlink(missing_ok=True)
        data.mkdir(exist_ok=True)
        RUNNERS[decl.type](decl, confdir)
        if not os.path.lexists(product):
            raise ProvisionError(
                f"Hook {decl.id!r} did not create {product}.", path=product
            )
        fp = fingerprint(product)
        tag = Tag(
            hook_id=decl.id,
            decl_digest=digest,
            content_hash=fp.content_hash,
            size=fp.size,
            source=decl.source if decl.type == "fetch" else shlex.join(decl.command),
            time=datetime.now(timezone.utc).replace(microsecond=0),
        )
        _tagfile(product).write_text(tomli_w.dumps(tag.model_dump()))
        ran.append(decl.id)
    return ran
