"""Confdir builders shared by the provisioning tests (R3a, R3b)."""

from __future__ import annotations

from pathlib import Path

import pytest


def make_confdir(parent: Path, hooks: str = "", name: str = "mini") -> Path:
    """A one-CompU confdir <name>-conf under parent, with optional [[hooks]] TOML appended."""
    conf = parent / f"{name}-conf"
    conf.mkdir(parents=True)
    (conf / "conf.toml").write_text(
        f'long_name = "{name}"\nname = "{name}"\nloop_number = 3\n'
        '[compute_units.cu]\nexec = "milk-fpsexec-mem-streamdelay"\n'
        '[compute_units.cu.fps]\nin_name = "in"\nout_name = "{compute_units.cu.fps.in_name}_out"\n'
        f"{hooks}\n"
    )
    return conf


def fetch_hook(source: Path, product: str = "flat.bin", hook_id: str = "flat") -> str:
    return f'[[hooks]]\nid = "{hook_id}"\ntype = "fetch"\nsource = "{source}"\nproduct = "{product}"\n'


@pytest.fixture
def src_file(tmp_path) -> Path:
    """A 200-byte file outside any confdir, for fetch hooks."""
    path = tmp_path / "src" / "flat.bin"
    path.parent.mkdir()
    path.write_bytes(b"\x00\x01" * 100)
    return path
