"""R3b acceptance tests: init, rootdir resolve, drift, resync, milk-pipeinit, operate on the new layout.

See specs/008-r3b-init-drift-resync/brief.md. Pure Python (rsync and sh must be on PATH).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import tomli
from click.testing import CliRunner


from milk.config.resolve import confresolve
from milk.errors import ConfigError, ProvisionError
from milk.provision.artifacts import fingerprint, status
from milk.provision.hooks import consolidate
from milk.provision.cli import milk_pipeinit
from milk.provision.provenance import read_provenance
from milk.provision.rootdir import drift, init, resync
from tests.conftestaux.provision import fetch_hook, make_confdir

NEVER_RUNNING = lambda: False  # noqa: E731


@pytest.fixture
def confdir(tmp_path, src_file) -> Path:
    """mini-conf with an include, a script and one fetch hook (data/flat.bin)."""
    conf = make_confdir(tmp_path, fetch_hook(src_file))
    toml = conf / "conf.toml"
    toml.write_text('includes = ["inc.toml"]\n' + toml.read_text())
    (conf / "inc.toml").write_text("[compute_units.cu.fps]\ndelaysec = 0.01\n")
    (conf / "scripts").mkdir()
    (conf / "scripts" / "a.sh").write_text("#!/bin/sh\n")
    return conf


@pytest.fixture
def rootdir(confdir) -> Path:
    return init(confdir)


def _edit(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    assert old in text, f"{old!r} not in {path}"
    path.write_text(text.replace(old, new, 1))


def _row(report, key: str):
    return next(a for a in report.artifacts if a.key == key)


def _fps(conf) -> dict:
    return conf.compute_units["cu"].fps


# ---- init ----


def test_init_layout(confdir, rootdir):
    assert rootdir == confdir.parent / "mini-rootdir"
    for rel in [
        "conf/conf.toml",
        "conf/inc.toml",
        "conf/scripts/a.sh",
        "initdata/flat.bin",
        "initdata/flat.bin.tag",
        "provenance.toml",
    ]:
        assert (rootdir / rel).is_file(), rel
    assert (rootdir / "work").is_dir() and (rootdir / "logs").is_dir()
    assert not (rootdir / "conf" / "data").exists()


def test_init_provenance(confdir, rootdir):
    prov = read_provenance(rootdir)
    assert Path(prov.source_confdir) == confdir.resolve()
    assert prov.milk_version and prov.last_resync is None and prov.overrides == {}
    assert set(prov.artifacts) == {
        "conf.toml",
        "inc.toml",
        "scripts/a.sh",
        "data/flat.bin",
    }
    assert (
        prov.artifacts["conf.toml"].content_hash
        == fingerprint(rootdir / "conf" / "conf.toml").content_hash
    )
    hook = prov.artifacts["data/flat.bin"]
    assert hook.hook_id == "flat" and hook.decl_digest.startswith("sha256:")
    assert tomli.loads((rootdir / "provenance.toml").read_text())[
        "source_confdir"
    ]  # A.3 is TOML


def test_init_consolidates_pending_hooks(confdir):
    assert not (confdir / "data").exists()
    init(confdir)
    assert (confdir / "data" / "flat.bin").is_file()


def test_init_refuses_existing_rootdir(confdir, rootdir):
    before = (rootdir / "provenance.toml").read_text()
    with pytest.raises(ProvisionError):
        init(confdir)
    assert (rootdir / "provenance.toml").read_text() == before


def test_init_parent(confdir, tmp_path):
    assert (
        init(confdir, parent=tmp_path / "elsewhere")
        == tmp_path / "elsewhere" / "mini-rootdir"
    )


def test_init_refuses_identity_override(confdir):
    with pytest.raises(ConfigError):
        init(confdir, overrides={"name": "other"})


# ---- rootdir resolve ----


def test_rootdir_resolves_without_its_confdir(confdir):
    rootdir = init(confdir, overrides={"compute_units.cu.fps.in_name": "wfs"})
    shutil.rmtree(confdir)
    fps = _fps(confresolve(rootdir))
    assert (
        fps["in_name"] == "wfs" and fps["out_name"] == "wfs_out"
    )  # replayed before substitution
    assert fps["delaysec"] == 0.01  # include copied into conf/


def test_rootdir_hand_edit_is_live(rootdir):
    _edit(rootdir / "conf" / "inc.toml", "0.01", "0.02")
    assert _fps(confresolve(rootdir))["delaysec"] == 0.02


# ---- drift ----


def test_drift_clean_after_init(rootdir):
    report = drift(rootdir)
    assert {(a.source, a.local) for a in report.artifacts} == {
        ("unchanged", "unchanged")
    }
    assert (
        report.resolved_diff == {}
        and report.structural == []
        and report.milk_version is None
    )


def test_drift_source_moved_on(confdir, rootdir):
    _edit(confdir / "inc.toml", "0.01", "0.05")
    report = drift(rootdir)
    row = _row(report, "inc.toml")
    assert (row.source, row.local) == ("modified", "unchanged")
    assert report.resolved_diff == {"compute_units.cu.fps.delaysec": (0.01, 0.05)}
    assert report.structural == []


def test_drift_cosmetic_edit_has_no_resolved_diff(confdir, rootdir):
    (confdir / "inc.toml").write_text(
        "# comment\n" + (confdir / "inc.toml").read_text()
    )
    report = drift(rootdir)
    assert _row(report, "inc.toml").source == "modified"
    assert report.resolved_diff == {}


def test_drift_added_and_removed_conf_files(confdir, rootdir):
    (confdir / "scripts" / "a.sh").unlink()
    (confdir / "scripts" / "b.sh").write_text("#!/bin/sh\n")
    report = drift(rootdir)
    assert _row(report, "scripts/a.sh").source == "removed"
    assert _row(report, "scripts/b.sh").source == "added"


def test_drift_local_edit(rootdir):
    _edit(rootdir / "conf" / "inc.toml", "0.01", "0.02")
    (rootdir / "initdata" / "flat.bin").write_bytes(b"local")
    report = drift(rootdir)
    assert (_row(report, "inc.toml").source, _row(report, "inc.toml").local) == (
        "unchanged",
        "modified",
    )
    assert _row(report, "data/flat.bin").local == "modified"
    # resolved_diff is (rootdir, confdir): the local edit is live on the rootdir side.
    assert report.resolved_diff == {"compute_units.cu.fps.delaysec": (0.02, 0.01)}


def test_drift_stale_fetch(confdir, rootdir, src_file, tmp_path):
    twin = tmp_path / "twin.bin"
    twin.write_bytes(src_file.read_bytes())
    _edit(
        confdir / "conf.toml", str(src_file), str(twin)
    )  # new declaration, same bytes
    assert _row(drift(rootdir), "data/flat.bin").source == "stale"


def test_drift_refetched_bytes(confdir, rootdir, src_file):
    src_file.write_bytes(b"new bytes")
    assert consolidate(confdir) == []  # upstream changes are not detected...
    assert consolidate(confdir, refetch=["flat"]) == [
        "flat"
    ]  # ...the user asks for a refetch
    assert _row(drift(rootdir), "data/flat.bin").source == "modified"


@pytest.mark.parametrize(
    "edit, expected",
    [
        (
            (
                "[compute_units.cu]",
                '[compute_units.cu2]\nexec = "milk-fpsexec-mem-streamdelay"\n[compute_units.cu]',
            ),
            "cu2",
        ),
        (("milk-fpsexec-mem-streamdelay", "milk-fpsexec-other"), "cu"),
    ],
    ids=["compu-added", "exec-changed"],
)
def test_drift_structural(confdir, rootdir, edit, expected):
    _edit(confdir / "conf.toml", *edit)
    structural = drift(rootdir).structural
    assert structural and any(expected in line for line in structural)


def test_drift_reports_package_version_change(rootdir):
    _edit(
        rootdir / "provenance.toml", read_provenance(rootdir).milk_version, "0.0.0-old"
    )
    report = drift(rootdir)
    assert report.milk_version is not None and report.milk_version[0] == "0.0.0-old"
    assert {a.local for a in report.artifacts} == {"unchanged"}  # not artifact drift


def test_status_absent_on_both_sides_is_unchanged():
    # A file added in the confdir is absent from both the record and the rootdir: no local drift.
    assert status(None, None) == "unchanged"


# ---- refetch ----


def test_refetch_all_and_unknown_id(confdir, src_file):
    consolidate(confdir)
    src_file.write_bytes(b"v2")
    assert consolidate(confdir, refetch="all") == ["flat"]
    assert (confdir / "data" / "flat.bin").read_bytes() == b"v2"
    with pytest.raises(ProvisionError, match="nope"):
        consolidate(confdir, refetch=["nope"])


def test_cli_consolidate_refetch(confdir, src_file):
    runner = CliRunner()
    assert runner.invoke(milk_pipeinit, ["consolidate", str(confdir)]).exit_code == 0
    src_file.write_bytes(b"v2")
    result = runner.invoke(
        milk_pipeinit, ["consolidate", str(confdir), "--refetch", "flat"]
    )
    assert result.exit_code == 0 and "flat" in result.output
    result = runner.invoke(
        milk_pipeinit, ["consolidate", str(confdir), "--refetch-all"]
    )
    assert result.exit_code == 0 and "flat" in result.output


# ---- layout sniffing ----


def test_sniff_and_list(confdir, rootdir, tmp_path):
    from milk.config.layout import is_confdir, is_rootdir, list_pipeline_dirs

    assert is_confdir(confdir) and not is_rootdir(confdir)
    assert is_rootdir(rootdir) and not is_confdir(rootdir)
    (tmp_path / "empty-rootdir").mkdir()  # the name alone is not enough
    assert list_pipeline_dirs(tmp_path) == ["mini-conf", "mini-rootdir"]
    assert not is_rootdir(tmp_path / "empty-rootdir")
    renamed = rootdir.rename(tmp_path / "renamed")  # ...but it is required
    assert not is_rootdir(renamed)
    assert not is_confdir(confdir.rename(tmp_path / "renamed-conf-copy"))
    assert list_pipeline_dirs(tmp_path) == []


# ---- resync ----


def test_resync_refused_while_running(confdir, rootdir):
    _edit(confdir / "inc.toml", "0.01", "0.05")
    before = (rootdir / "provenance.toml").read_text()
    with pytest.raises(ProvisionError, match="run"):
        resync(rootdir, run_active=lambda: True)
    assert (rootdir / "provenance.toml").read_text() == before
    assert "0.01" in (rootdir / "conf" / "inc.toml").read_text()


def test_resync_refused_on_identity_change(confdir, rootdir):
    _edit(confdir / "conf.toml", '\nname = "mini"', '\nname = "other"')
    before = (rootdir / "conf" / "conf.toml").read_text()
    with pytest.raises(ProvisionError, match="init"):
        resync(rootdir, run_active=NEVER_RUNNING)
    assert (rootdir / "conf" / "conf.toml").read_text() == before


def test_resync_applies_and_records(confdir, rootdir):
    _edit(confdir / "inc.toml", "0.01", "0.05")
    report = resync(rootdir, run_active=NEVER_RUNNING)
    assert report.applied == ["inc.toml"] and report.refused == {}
    assert report.resolved_diff == {"compute_units.cu.fps.delaysec": (0.01, 0.05)}
    assert _fps(confresolve(rootdir))["delaysec"] == 0.05
    assert read_provenance(rootdir).last_resync is not None
    assert {(a.source, a.local) for a in drift(rootdir).artifacts} == {
        ("unchanged", "unchanged")
    }


def test_resync_refuses_local_edit_but_applies_the_rest(confdir, rootdir):
    _edit(confdir / "inc.toml", "0.01", "0.05")
    (confdir / "scripts" / "a.sh").write_text("#!/bin/sh\necho v2\n")
    _edit(rootdir / "conf" / "inc.toml", "0.01", "0.02")  # local edit

    report = resync(rootdir, run_active=NEVER_RUNNING)
    assert report.applied == ["scripts/a.sh"]
    assert set(report.refused) == {"inc.toml"}
    assert _fps(confresolve(rootdir))["delaysec"] == 0.02  # local edit kept

    report = resync(rootdir, run_active=NEVER_RUNNING, overwrite_local=True)
    assert report.applied == ["inc.toml"]
    assert _fps(confresolve(rootdir))["delaysec"] == 0.05


def test_resync_selected_artifacts_only(confdir, rootdir):
    _edit(confdir / "inc.toml", "0.01", "0.05")
    (confdir / "scripts" / "a.sh").write_text("#!/bin/sh\necho v2\n")
    report = resync(rootdir, run_active=NEVER_RUNNING, artifacts=["scripts/a.sh"])
    assert report.applied == ["scripts/a.sh"]
    assert _row(drift(rootdir), "inc.toml").source == "modified"


def test_resync_added_and_removed(confdir, rootdir):
    (confdir / "scripts" / "a.sh").unlink()
    (confdir / "scripts" / "b.sh").write_text("#!/bin/sh\n")
    resync(rootdir, run_active=NEVER_RUNNING)
    assert not (rootdir / "conf" / "scripts" / "a.sh").exists()
    assert (rootdir / "conf" / "scripts" / "b.sh").is_file()
    assert set(read_provenance(rootdir).artifacts) == {
        "conf.toml",
        "inc.toml",
        "scripts/b.sh",
        "data/flat.bin",
    }


def test_resync_changed_hook(confdir, rootdir, src_file, tmp_path):
    twin = tmp_path / "twin.bin"
    twin.write_bytes(b"other bytes")
    _edit(confdir / "conf.toml", str(src_file), str(twin))
    report = resync(rootdir, run_active=NEVER_RUNNING)  # consolidates the confdir first
    assert report.applied == ["conf.toml", "data/flat.bin"]  # sorted keys
    assert (rootdir / "initdata" / "flat.bin").read_bytes() == b"other bytes"
    assert (rootdir / "initdata" / "flat.bin.tag").read_text() == (
        confdir / "data" / "flat.bin.tag"
    ).read_text()
    assert {(a.source, a.local) for a in drift(rootdir).artifacts} == {
        ("unchanged", "unchanged")
    }


def test_resync_structural_report(confdir, rootdir):
    _edit(
        confdir / "conf.toml",
        "[compute_units.cu]",
        '[compute_units.cu2]\nexec = "milk-fpsexec-mem-streamdelay"\n[compute_units.cu]',
    )
    report = resync(rootdir, run_active=NEVER_RUNNING)
    assert report.applied == ["conf.toml"]
    assert any("cu2" in line for line in report.structural)


def test_resync_overrides_recorded_and_replayed(rootdir):
    report = resync(
        rootdir,
        run_active=NEVER_RUNNING,
        overrides={"compute_units.cu.fps.delaysec": 0.3},
    )
    assert report.applied == []
    assert report.resolved_diff == {"compute_units.cu.fps.delaysec": (0.01, 0.3)}
    assert read_provenance(rootdir).overrides == {"compute_units.cu.fps.delaysec": 0.3}
    assert _fps(confresolve(rootdir))["delaysec"] == 0.3
    with pytest.raises(ConfigError):
        resync(rootdir, run_active=NEVER_RUNNING, overrides={"loop_number": 9})


# ---- milk-pipeinit ----


def test_cli_init_set_and_resolve(confdir):
    runner = CliRunner()
    result = runner.invoke(
        milk_pipeinit,
        [
            "init",
            str(confdir),
            "--set",
            "compute_units.cu.fps.delaysec=0.5",
            "--set",
            "compute_units.cu.fps.in_name=wfs",
        ],
    )
    assert result.exit_code == 0, result.output
    rootdir = confdir.parent / "mini-rootdir"
    result = runner.invoke(milk_pipeinit, ["resolve", str(rootdir)])
    assert result.exit_code == 0, result.output
    fps = tomli.loads(result.output)["compute_units"]["cu"]["fps"]
    assert (
        fps["delaysec"] == 0.5 and fps["in_name"] == "wfs"
    )  # TOML-typed when parseable, else string


def test_cli_drift_and_resync_banner(confdir, rootdir):
    runner = CliRunner()
    _edit(
        confdir / "conf.toml",
        "[compute_units.cu]",
        '[compute_units.cu2]\nexec = "milk-fpsexec-mem-streamdelay"\n[compute_units.cu]',
    )
    result = runner.invoke(milk_pipeinit, ["drift", str(rootdir)])
    assert result.exit_code == 0, result.output
    assert "conf.toml" in result.output and "modified" in result.output

    result = runner.invoke(milk_pipeinit, ["resync", str(rootdir)])
    assert result.exit_code == 0, result.output
    assert (
        "teardown" in result.output
        and "deploy" in result.output
        and "cu2" in result.output
    )

    result = runner.invoke(
        milk_pipeinit, ["resync", str(rootdir)]
    )  # nothing left to apply
    assert (
        result.exit_code == 0 and "teardown" in result.output
    )  # the banner is always printed


# ---- the operate layer reads the rootdir ----


def test_pipeline_reads_the_rootdir_only(confdir, rootdir):
    from milk.operate.pipeline import Pipeline

    shutil.rmtree(confdir)
    pp = Pipeline(rootdir)
    assert pp.root_folder == rootdir and pp.config == confresolve(rootdir)
    assert pp.work_folder == rootdir / "work"


def test_pipeline_targets(rootdir, monkeypatch, tmp_path):
    from milk.operate.pipeline import Pipeline

    assert Pipeline.from_folder(rootdir).root_folder == rootdir
    monkeypatch.chdir(rootdir.parent)
    assert Pipeline("mini").root_folder == rootdir  # short name: <cwd>/<name>-rootdir
    assert (
        Pipeline(str(rootdir)).root_folder == rootdir
    )  # an existing rootdir path wins
    monkeypatch.chdir(tmp_path / "src")
    with pytest.raises(ConfigError, match="mini"):
        Pipeline("mini")


def test_teardown_keeps_the_rootdir(rootdir, shm_dir_cleanafter):
    from milk.operate.plans import run_plan
    from milk.operate.pipeline import Pipeline

    (rootdir / "work" / "fps.cu.datadir").mkdir()
    run_plan(Pipeline(rootdir), "teardown")  # on a cold pipeline
    assert (rootdir / "work").is_dir()
    assert (rootdir / "work" / "fps.cu.datadir").is_dir()  # reusable data survives
    assert (rootdir / "provenance.toml").is_file()


def test_cli_nuke_asks_for_the_short_name(rootdir, shm_dir_cleanafter):
    from milk.operate.cli import milk_pipes

    runner = CliRunner()
    result = runner.invoke(milk_pipes, ["nuke", "-p", str(rootdir)], input="wrong\n")
    assert result.exit_code != 0 and rootdir.is_dir()
    result = runner.invoke(milk_pipes, ["nuke", "-p", str(rootdir)], input="mini\n")
    assert result.exit_code == 0, result.output
    assert not rootdir.exists()


def test_cli_nuke_yes_and_short_name(confdir, shm_dir_cleanafter, monkeypatch):
    from milk.operate.cli import milk_pipes

    rootdir = init(confdir)
    monkeypatch.chdir(rootdir.parent)
    result = CliRunner().invoke(milk_pipes, ["nuke", "-p", "mini", "--yes"])
    assert result.exit_code == 0, result.output
    assert not rootdir.exists()


def test_dataloads_read_from_initdata(tmp_path, src_file):
    from milk.operate.tasks import LoadData
    from milk.operate.pipeline import Pipeline

    conf = make_confdir(
        tmp_path,
        fetch_hook(src_file) + '[[dataloads]]\nsource = "flat.bin"\ntarget = "flat"\n',
    )
    task = LoadData(Pipeline(init(conf)))
    assert task.can(), task.failure_reason


def test_milk_pipes_init_is_gone():
    from milk.operate.cli import milk_pipes

    assert CliRunner().invoke(milk_pipes, ["init", "-h"]).exit_code != 0
