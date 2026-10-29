"""R2 acceptance tests: confresolve layers, includes, overrides, substitution. See specs/004-r2-confresolve/brief.md."""

from __future__ import annotations

from pathlib import Path

import pytest

import milk.config.resolve
from milk.config import confresolve
from milk.errors import ConfigError
from milk.operate.pipeline import Pipeline
from milk.provision.rootdir import init

RESOURCES = Path(__file__).parents[1] / "resources"


def _confset(
    tmp_path: Path, head: str = "", fps: str = "", tail: str = "", **files: str
) -> Path:
    """Minimal one-CompU confdir. head: top-level keys; fps: keys of [compute_units.cu.fps];
    tail: extra tables; files: relpath -> content (use '/' in relpath via '__')."""
    conf = tmp_path / "mini-conf"
    conf.mkdir()
    (conf / "conf.toml").write_text(
        'long_name = "mini"\nname = "mini"\nloop_number = 3\n'
        f"{head}\n"
        '[compute_units.cu]\nexec = "milk-fpsexec-mem-streamdelay"\n'
        f'[compute_units.cu.fps]\nin_name = "in"\ndelaysec = 0.01\n{fps}\n'
        f"{tail}\n"
    )
    for relpath, content in files.items():
        path = conf / relpath.replace("__", "/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return conf


def _fps(conf) -> dict:
    return conf.compute_units["cu"].fps


# ---- Baseline ----


def test_resolves_pipelinebasic():
    conf = confresolve(RESOURCES / "pipelinebasic-conf")
    assert conf.name == "basic" and conf.kind == "generic"
    assert conf.compute_units["delay1"].fps["in_name"] == "middle"


def test_pipeline_config_comes_from_confresolve(tmp_path):
    rootdir = init(RESOURCES / "pipelinebasic-conf", parent=tmp_path)
    assert Pipeline(rootdir).config == confresolve(RESOURCES / "pipelinebasic-conf")


def test_every_call_reresolves(tmp_path):
    conf = _confset(tmp_path)
    assert _fps(confresolve(conf))["delaysec"] == 0.01
    toml = conf / "conf.toml"
    toml.write_text(toml.read_text().replace("delaysec = 0.01", "delaysec = 0.5"))
    assert _fps(confresolve(conf))["delaysec"] == 0.5


# ---- Layers ----


def test_includes_merge_in_order_below_conf_toml(tmp_path):
    a = "[compute_units.cu.fps]\ntimebuff = 10\ndelaysec = 5.0\n"
    b = "[compute_units.cu.fps]\ntimebuff = 20\n"
    conf = _confset(
        tmp_path,
        head='includes = ["a.toml", "common/b.toml"]',
        **{"a.toml": a, "common__b.toml": b},
    )
    fps = _fps(confresolve(conf))
    assert fps["timebuff"] == 20  # later include wins
    assert fps["delaysec"] == 0.01  # conf.toml wins over includes


def test_table_arrays_are_replaced_whole(tmp_path):
    inc = '[[dataloads]]\nsource = "a.fits"\ntarget = "a"\n[[dataloads]]\nsource = "b.fits"\ntarget = "b"\n'
    conf = _confset(
        tmp_path,
        head='includes = ["inc.toml"]',
        tail='[[dataloads]]\nsource = "c.fits"\ntarget = "c"\n',
        **{"inc.toml": inc},
    )
    assert [d.target for d in confresolve(conf).dataloads] == ["c"]


def test_kind_defaults_are_the_lowest_layer(tmp_path, monkeypatch):
    defaults = {"compute_units": {"cu": {"fps": {"delaysec": 9.0, "timebuff": 7}}}}
    monkeypatch.setattr(milk.config.resolve, "kind_defaults", lambda kind: defaults)
    fps = _fps(confresolve(_confset(tmp_path)))
    assert fps["timebuff"] == 7
    assert fps["delaysec"] == 0.01


def test_unknown_kind_refused(tmp_path):
    with pytest.raises(ConfigError, match="nope"):
        confresolve(_confset(tmp_path, head='kind = "nope"'))


def test_overrides_are_the_top_layer_and_feed_substitution(tmp_path):
    conf = _confset(tmp_path, fps='out_name = "{compute_units.cu.fps.in_name}_out"')
    fps = _fps(
        confresolve(
            conf,
            {
                "compute_units.cu.fps.delaysec": 0.5,
                "compute_units.cu.fps.in_name": "wfs",
            },
        )
    )
    assert fps["delaysec"] == 0.5
    assert fps["out_name"] == "wfs_out"


# ---- Refusals ----


@pytest.mark.parametrize("key", ["long_name", "name", "loop_number"])
def test_identity_override_refused(tmp_path, key):
    with pytest.raises(ConfigError, match=key):
        confresolve(_confset(tmp_path), {key: "x"})


def test_override_through_non_table_refused(tmp_path):
    with pytest.raises(ConfigError):
        confresolve(_confset(tmp_path), {"compute_units.cu.exec.sub": 1})


@pytest.mark.parametrize(
    "include",
    ["/etc/passwd", "../outside.toml", "missing.toml", "sub/../../outside.toml"],
)
def test_bad_include_refused(tmp_path, include):
    (tmp_path / "outside.toml").write_text("")
    with pytest.raises(ConfigError):
        confresolve(_confset(tmp_path, head=f'includes = ["{include}"]'))


def test_nested_includes_refused(tmp_path):
    conf = _confset(
        tmp_path,
        head='includes = ["a.toml"]',
        **{"a.toml": 'includes = ["b.toml"]\n', "b.toml": ""},
    )
    with pytest.raises(ConfigError):
        confresolve(conf)


def test_validation_error_is_config_error_with_path(tmp_path):
    conf = _confset(tmp_path, head="bogus = 1")
    with pytest.raises(ConfigError, match="bogus") as err:
        confresolve(conf)
    assert err.value.path == conf / "conf.toml"


def test_substitution_error_is_config_error(tmp_path):
    with pytest.raises(ConfigError):
        confresolve(_confset(tmp_path, fps='out_name = "{nowhere.key}"'))


# ---- Substitution ----


def test_substitution_across_layers(tmp_path):
    inc = '[compute_units.cu.fps]\nlabel = "{name}_{compute_units.cu.fps.in_name}"\n'
    conf = _confset(
        tmp_path,
        head='includes = ["inc.toml"]',
        fps='out_name = "{compute_units.cu.fps.label}"',
        **{"inc.toml": inc},
    )
    fps = _fps(confresolve(conf))
    assert fps["label"] == "mini_in"  # include references conf.toml
    assert fps["out_name"] == "mini_in"  # conf.toml references the include


def test_table_arrays_pass_through_untouched(tmp_path):
    conf = _confset(
        tmp_path, tail='[[dataloads]]\nsource = "{name}.fits"\ntarget = "t"\n'
    )
    assert confresolve(conf).dataloads[0].source == "{name}.fits"


def test_table_arrays_cannot_be_referenced(tmp_path):
    conf = _confset(
        tmp_path,
        fps='out_name = "{dataloads.0.target}"',
        tail='[[dataloads]]\nsource = "a.fits"\ntarget = "t"\n',
    )
    with pytest.raises(ConfigError):
        confresolve(conf)
