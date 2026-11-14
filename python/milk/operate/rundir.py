"""The rundir (design-v7 A.4, simplified): a deploy's claim for this boot, in
``$MILK_SHM_DIR/pipe_rundirs/<name>/``."""

from __future__ import annotations

import getpass
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import tomli
import tomli_w

from .. import env


def current_boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def rundir_path(name: str) -> Path:
    return env.shm_dir() / "pipe_rundirs" / name


@dataclass(frozen=True)
class Claim:
    name: str
    loop_number: int
    rootdir: str
    deployed: datetime
    user: str
    pid: int

    @classmethod
    def new(cls, name: str, loop_number: int, rootdir: str) -> Claim:
        now = datetime.now(timezone.utc).replace(microsecond=0)
        return cls(name, loop_number, rootdir, now, getpass.getuser(), os.getpid())


def read_claim(name: str) -> Claim | None:
    path = rundir_path(name) / "claim.toml"
    return Claim(**tomli.loads(path.read_text())) if path.is_file() else None


def read_boot_id(name: str) -> str | None:
    path = rundir_path(name) / "boot_id"
    return path.read_text().strip() if path.is_file() else None


def write_claim(claim: Claim) -> None:
    path = rundir_path(claim.name)
    path.mkdir(parents=True, exist_ok=True)
    (path / "boot_id").write_text(current_boot_id() + "\n")
    (path / "claim.toml").write_text(tomli_w.dumps(asdict(claim)))


def all_claims() -> list[Claim]:
    root = env.shm_dir() / "pipe_rundirs"
    return [c for p in sorted(root.glob("*")) if (c := read_claim(p.name))]


def release(name: str) -> None:
    shutil.rmtree(rundir_path(name), ignore_errors=True)
