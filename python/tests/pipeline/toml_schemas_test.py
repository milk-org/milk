from __future__ import annotations

import shutil

import pytest
import tomli
import pydantic
from pydantic import BaseModel

from milk.errors import ConfigError
from milk.infra import pipeline_models as pconf
from milk.confresolve import confresolve


@pytest.fixture
def conf_dir(pytestconfig):
    return pytestconfig.rootpath / "tests" / "resources" / "pipelinebasic-conf"


@pytest.fixture
def load_raw_toml(conf_dir):
    with open(conf_dir / "conf.toml", "rb") as f:
        return tomli.load(f)


@pytest.fixture
def load_modeled_toml(load_raw_toml):
    return pconf.PipelineConfig.model_validate(load_raw_toml)


def test_valid_config(load_raw_toml):
    conf = pconf.PipelineConfig.model_validate(load_raw_toml)
    assert conf.long_name == "pipelinebasic"
    assert conf.name == "basic"
    assert conf.loop_number == 0
    assert set(conf.compute_units) == {"delay0", "delay1"}
    assert isinstance(conf.compute_units["delay0"], pconf.ComputeUnitSpec)
    assert conf.compute_units["delay0"].exec == "milk-fpsexec-mem-streamdelay"


def test_systemlinkage(load_modeled_toml):
    assert load_modeled_toml.systemlinkage["standalone"] == {}
    assert load_modeled_toml.systemlinkage["shared"]["input"] == "otherloop_output"


@pytest.mark.parametrize("local, remote", [("a", "a"), ("a", "a.im.shm"), ("a", "")])
def test_systemlinkage_bad(load_raw_toml, local, remote):
    load_raw_toml["systemlinkage"]["shared"] = {local: remote}
    with pytest.raises(pydantic.ValidationError):
        pconf.PipelineConfig.model_validate(load_raw_toml)


def test_missing_identity_key(load_raw_toml):
    del load_raw_toml["loop_number"]
    with pytest.raises(pydantic.ValidationError):
        pconf.PipelineConfig.model_validate(load_raw_toml)


def test_missing_exec_names_cu(load_raw_toml):
    del load_raw_toml["compute_units"]["delay0"]["exec"]
    with pytest.raises(pydantic.ValidationError) as err:
        pconf.PipelineConfig.model_validate(load_raw_toml)
    msg = str(err.value)
    assert "delay0" in msg
    assert "exec" in msg


def test_malformed_procinfo_names_field(load_raw_toml):
    fps = load_raw_toml["compute_units"]["delay0"]["fps"]
    del fps["procinfo"]["enabled"]
    fps["procinfo"]["enabld"] = False  # typo

    with pytest.raises(pydantic.ValidationError) as err:
        pconf.PipelineConfig.model_validate(load_raw_toml)
    assert "enabld" in str(err.value)


def test_string_form_dataload_refused(load_raw_toml):
    load_raw_toml["dataloads"] = ["sim_matrix_flat1.fits:aol1_simu_modes"]
    with pytest.raises(pydantic.ValidationError):
        pconf.PipelineConfig.model_validate(load_raw_toml)


def test_cu_without_fps_is_valid(load_raw_toml):
    load_raw_toml["compute_units"]["bare"] = {"exec": "milk-fpsexec-mem-streamdelay"}
    conf = pconf.PipelineConfig.model_validate(load_raw_toml)
    assert conf.compute_units["bare"].fps == {}
    assert conf.compute_units["bare"].procinfo is None


def test_cu_named_tasks_is_valid(load_raw_toml):
    load_raw_toml["compute_units"]["tasks"] = {"exec": "milk-fpsexec-mem-streamdelay"}
    conf = pconf.PipelineConfig.model_validate(load_raw_toml)
    assert "tasks" in conf.compute_units


def test_pipeline_config_is_frozen(load_modeled_toml):
    with pytest.raises(pydantic.ValidationError):
        load_modeled_toml.name = "other"


@pytest.mark.parametrize("key", ["foo", "tasks", "sessions", "delay0"])
def test_loader_refuses_unknown_top_level_key(tmp_path, conf_dir, key):
    toml_path = tmp_path / "conf.toml"
    toml_path.write_text(f"{key} = 1\n" + (conf_dir / "conf.toml").read_text())

    with pytest.raises(ConfigError) as err:
        confresolve(tmp_path)
    assert key in str(err.value)
    assert err.value.path == toml_path


def test_loader_refuses_old_key_under_cu(tmp_path, conf_dir):
    shutil.copy(conf_dir / "conf.toml", tmp_path / "conf.toml")
    toml_path = tmp_path / "conf.toml"
    text = toml_path.read_text()
    text = text.replace(
        "[compute_units.delay0]\n", '[compute_units.delay0]\nkind = "old"\n', 1
    )
    toml_path.write_text(text)

    with pytest.raises(ConfigError) as err:
        confresolve(tmp_path)
    assert "kind" in str(err.value)
    assert err.value.path == toml_path


@pytest.mark.parametrize("value", ["'scalar'", "[1, 2]"], ids=["scalar", "array"])
def test_loader_refuses_bad_procinfo_shape(tmp_path, value):
    toml_path = tmp_path / "conf.toml"
    toml_path.write_text(
        'long_name = "x"\n'
        'name = "basic"\n'
        "loop_number = 0\n"
        "[compute_units.delay0]\n"
        "exec = 'milk-fpsexec-mem-streamdelay'\n"
        "[compute_units.delay0.fps]\n"
        f"procinfo = {value}\n"
    )

    with pytest.raises(ConfigError) as err:
        confresolve(tmp_path)
    assert "procinfo" in str(err.value)
    assert err.value.path == toml_path
    assert isinstance(err.value.__cause__, pydantic.ValidationError)


def test_model_no_longer_has_legacy_names():
    assert not hasattr(pconf, "PipelineConfigModel")
    assert not hasattr(pconf, "LOOP_INFO")
    assert issubclass(pconf.PipelineConfig, BaseModel)
