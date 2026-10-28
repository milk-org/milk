"""R3a acceptance tests: fingerprints, conf files, conf hooks, consolidate, clone, milk-pipeinit.

See specs/007-r3a-artifacts-consolidate/brief.md. Pure Python (rsync and sh must be on PATH).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import tomli
from click.testing import CliRunner

from milk.errors import ConfigError, ProvisionError
from milk.provision import (
    Fingerprint,
    clone,
    conf_files,
    consolidate,
    decl_digest,
    fingerprint,
    read_tag,
    status,
)
from milk.provision.cli import milk_pipeinit

TAG_KEYS = {"hook_id", "decl_digest", "content_hash", "size", "source", "time"}


def _confdir(parent: Path, hooks: str = "", name: str = "mini") -> Path:
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


def _fetch_hook(source: Path, product: str = "flat.bin", hook_id: str = "flat") -> str:
    return f'[[hooks]]\nid = "{hook_id}"\ntype = "fetch"\nsource = "{source}"\nproduct = "{product}"\n'


@pytest.fixture
def src_file(tmp_path) -> Path:
    path = tmp_path / "src" / "flat.bin"
    path.parent.mkdir()
    path.write_bytes(b"\x00\x01" * 100)
    return path


# ---- Fingerprints ----


def test_file_fingerprint_is_bytes_only(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"same")
    b.write_bytes(b"same")
    os.chmod(b, 0o600)
    os.utime(b, (0, 0))
    assert fingerprint(a) == fingerprint(b)
    assert fingerprint(a).size == 4
    assert fingerprint(a).content_hash.startswith("sha256:")
    b.write_bytes(b"diff")
    assert fingerprint(a) != fingerprint(b)


def test_directory_fingerprint(tmp_path):
    def tree(root: Path) -> Path:
        (root / "sub").mkdir(parents=True)
        (root / "sub" / "x.txt").write_text("x")
        (root / "y.txt").write_text("y")
        return root

    a, b = tree(tmp_path / "a"), tree(tmp_path / "b")
    assert fingerprint(a) == fingerprint(b)

    (b / "y.txt").rename(b / "z.txt")  # same bytes, different relpath
    assert fingerprint(a) != fingerprint(b)

    (b / "z.txt").rename(b / "y.txt")
    (b / "empty").mkdir()  # empty directories count
    assert fingerprint(a) != fingerprint(b)


def test_symlinks_refused(tmp_path):
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "real").write_text("r")
    (tmp_path / "d" / "link").symlink_to(tmp_path / "d" / "real")
    with pytest.raises(ProvisionError):
        fingerprint(tmp_path / "d")
    with pytest.raises(ProvisionError):
        fingerprint(tmp_path / "d" / "link")


def test_conf_files(tmp_path):
    conf = _confdir(tmp_path)
    for rel in [
        "inc/streams.toml",
        "scripts/a.sh",
        "scripts/sub/b.sh",
        "data/x.toml",
        "README.md",
    ]:
        (conf / rel).parent.mkdir(parents=True, exist_ok=True)
        (conf / rel).write_text("")
    assert conf_files(conf) == [
        Path(p)
        for p in ["conf.toml", "inc/streams.toml", "scripts/a.sh", "scripts/sub/b.sh"]
    ]


# ---- Status and declaration digest ----

F1, F2 = Fingerprint("sha256:1", 1), Fingerprint("sha256:2", 1)


@pytest.mark.parametrize(
    "recorded, current, rdecl, cdecl, expected",
    [
        (F1, F1, None, None, "unchanged"),
        (F1, F2, None, None, "modified"),
        (F1, F1, "sha256:a", "sha256:a", "unchanged"),
        (F1, F1, "sha256:a", "sha256:b", "stale"),
        (
            F1,
            F2,
            "sha256:a",
            "sha256:b",
            "modified",
        ),  # changed bytes win over a changed decl
        (None, F1, None, None, "added"),
        (F1, None, None, None, "removed"),
    ],
)
def test_status(recorded, current, rdecl, cdecl, expected):
    assert status(recorded, current, rdecl, cdecl) == expected


def test_decl_digest():
    assert decl_digest({"id": "a", "type": "fetch"}) == decl_digest(
        {"type": "fetch", "id": "a"}
    )
    assert decl_digest({"id": "a"}) != decl_digest({"id": "b"})


# ---- Conf hooks and consolidate ----


def test_fetch_file_and_tag(tmp_path, src_file):
    conf = _confdir(tmp_path, _fetch_hook(src_file))
    assert consolidate(conf) == ["flat"]
    product = conf / "data" / "flat.bin"
    assert product.read_bytes() == src_file.read_bytes()

    tag = read_tag(product)
    assert tag is not None and set(tag) == TAG_KEYS
    assert tag["hook_id"] == "flat"
    assert tag["content_hash"] == fingerprint(product).content_hash
    assert (
        tomli.loads((conf / "data" / "flat.bin.tag").read_text()) == tag
    )  # A.2 is TOML

    assert consolidate(conf) == []  # resolved hooks are skipped


def test_fetch_directory(tmp_path):
    src = tmp_path / "srcdir"
    (src / "sub").mkdir(parents=True)
    (src / "sub" / "m.fits").write_bytes(b"m")
    conf = _confdir(tmp_path, _fetch_hook(src, product="mats"))
    assert consolidate(conf) == ["flat"]
    assert (conf / "data" / "mats" / "sub" / "m.fits").read_bytes() == b"m"
    assert fingerprint(conf / "data" / "mats") == fingerprint(src)


def test_command_hook(tmp_path):
    hook = '[[hooks]]\nid = "gen"\ntype = "command"\ncommand = ["sh", "-c", "echo hi > gen.txt"]\nproduct = "gen.txt"\n'
    conf = _confdir(tmp_path, hook)
    assert consolidate(conf) == ["gen"]
    assert (conf / "data" / "gen.txt").read_text() == "hi\n"
    assert consolidate(conf) == []


def test_command_without_product_refused(tmp_path):
    hook = '[[hooks]]\nid = "gen"\ntype = "command"\ncommand = ["true"]\nproduct = "gen.txt"\n'
    with pytest.raises(ProvisionError, match="gen"):
        consolidate(_confdir(tmp_path, hook))


def test_stale_declaration_reruns(tmp_path, src_file):
    conf = _confdir(tmp_path, _fetch_hook(src_file))
    consolidate(conf)
    twin = tmp_path / "twin.bin"
    twin.write_bytes(src_file.read_bytes())  # same bytes, new declaration
    toml = conf / "conf.toml"
    toml.write_text(toml.read_text().replace(str(src_file), str(twin)))
    assert consolidate(conf) == ["flat"]
    assert consolidate(conf) == []


def test_modified_product_is_refetched(tmp_path, src_file):
    conf = _confdir(tmp_path, _fetch_hook(src_file))
    consolidate(conf)
    (conf / "data" / "flat.bin").write_bytes(b"local edit")
    assert consolidate(conf) == ["flat"]
    assert (conf / "data" / "flat.bin").read_bytes() == src_file.read_bytes()


def test_fetched_symlink_refused(tmp_path):
    src = tmp_path / "srcdir"
    src.mkdir()
    (src / "real").write_text("r")
    (src / "link").symlink_to(src / "real")
    with pytest.raises(ProvisionError):
        consolidate(_confdir(tmp_path, _fetch_hook(src, product="d")))


@pytest.mark.parametrize(
    "hooks",
    [
        '[[hooks]]\nid = "x"\ntype = "teleport"\nproduct = "p"\n',
        '[[hooks]]\nid = "x"\ntype = "fetch"\nsource = "/s"\nproduct = "../p"\n',
        '[[hooks]]\nid = "x"\ntype = "fetch"\nsource = "/s"\nproduct = "a/p"\n',
        '[[hooks]]\nid = "x"\ntype = "fetch"\nsource = "/s"\nproduct = "p.tag"\n',
        '[[hooks]]\nid = "x"\ntype = "fetch"\nproduct = "p"\n',  # fetch without source
        '[[hooks]]\nid = "x"\ntype = "fetch"\nsource = "/s"\nproduct = "p"\n'
        '[[hooks]]\nid = "x"\ntype = "fetch"\nsource = "/s"\nproduct = "q"\n',  # duplicate id
        '[[hooks]]\nid = "x"\ntype = "fetch"\nsource = "/s"\nproduct = "p"\n'
        '[[hooks]]\nid = "y"\ntype = "fetch"\nsource = "/s"\nproduct = "p"\n',  # duplicate product
    ],
    ids=[
        "unknown-type",
        "dotdot",
        "subpath",
        "tag-suffix",
        "no-source",
        "dup-id",
        "dup-product",
    ],
)
def test_bad_hook_declarations_refused_at_load(tmp_path, hooks):
    with pytest.raises(ConfigError):
        consolidate(_confdir(tmp_path, hooks))


# ---- Clone ----


def test_clone(tmp_path, src_file):
    src = _confdir(tmp_path / "a", _fetch_hook(src_file))
    (src / "scripts").mkdir()
    (src / "scripts" / "align.sh").write_text("#!/bin/sh\n")
    (src / "README.md").write_text("not conf")
    consolidate(src)

    dest = clone(src, tmp_path / "b")
    assert dest == tmp_path / "b" / "mini-conf"
    assert conf_files(dest) == conf_files(src)
    assert not (dest / "data").exists()
    assert not (dest / "README.md").exists()
    assert not (dest / ".git").exists()
    assert (dest / ".gitignore").read_text().splitlines() == ["data/"]


def test_clone_refuses_existing_target(tmp_path):
    src = _confdir(tmp_path / "a")
    (tmp_path / "b" / "mini-conf").mkdir(parents=True)
    with pytest.raises(ProvisionError):
        clone(src, tmp_path / "b")
    assert list((tmp_path / "b" / "mini-conf").iterdir()) == []


# ---- milk-pipeinit ----


def test_cli_list(tmp_path):
    _confdir(tmp_path, name="one")
    (tmp_path / "two-rootdir").mkdir()
    (tmp_path / "junk-conf").mkdir()  # no conf.toml: not a confdir
    result = CliRunner().invoke(milk_pipeinit, ["list", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "one-conf" in result.output and "two-rootdir" in result.output
    assert "junk-conf" not in result.output


def test_cli_clone_and_consolidate(tmp_path, src_file):
    src = _confdir(tmp_path / "a", _fetch_hook(src_file))
    runner = CliRunner()
    result = runner.invoke(milk_pipeinit, ["clone", str(src), str(tmp_path / "b")])
    assert result.exit_code == 0, result.output
    dest = tmp_path / "b" / "mini-conf"
    result = runner.invoke(milk_pipeinit, ["consolidate", str(dest)])
    assert result.exit_code == 0, result.output
    assert "flat" in result.output
    assert (dest / "data" / "flat.bin").exists()


def test_cli_resolve(tmp_path):
    conf = _confdir(tmp_path)
    result = CliRunner().invoke(milk_pipeinit, ["resolve", str(conf)])
    assert result.exit_code == 0, result.output
    assert (
        tomli.loads(result.output)["compute_units"]["cu"]["fps"]["out_name"] == "in_out"
    )

    (conf / "conf.toml").write_text("bogus = 1\n")
    assert CliRunner().invoke(milk_pipeinit, ["resolve", str(conf)]).exit_code != 0


def test_milk_pipes_clone_is_gone():
    from milk.infra.mains_pipeline_mgmt import milk_pipes

    assert CliRunner().invoke(milk_pipes, ["clone", "-h"]).exit_code != 0
