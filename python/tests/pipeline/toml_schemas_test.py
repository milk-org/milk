from __future__ import annotations

import pytest
import tomli
import pydantic

from milk.infra import pipeline_config as pconf


@pytest.fixture
def load_raw_toml(pytestconfig):
    conf_path = (
        pytestconfig.rootpath
        / "tests"
        / "resources"
        / "pipelinebasic-conf"
        / "conf.toml"
    )
    with open(conf_path, "rb") as f:
        return tomli.load(f)


@pytest.fixture
def load_modeled_toml(load_raw_toml):
    return pconf.PipelineConfigModel(**load_raw_toml)


def test_valid_config(load_raw_toml):
    # Nothing, simply testing the fixture.
    conf = pconf.PipelineConfigModel(**load_raw_toml)
    assert conf.long_name == "pipelinebasic"
    assert conf.name == "basic"
    assert conf.loop_number == 0


def test_missing_toml_basic(load_raw_toml):
    del load_raw_toml["loop_number"]
    with pytest.raises(pydantic.ValidationError):
        conf = pconf.PipelineConfigModel(**load_raw_toml)


def test_missing_fps_def(load_raw_toml):
    assert "delay0" in load_raw_toml["sessions"]
    assert "delay0" in load_raw_toml  # FPS descriptor for the ComputeSession

    del load_raw_toml["delay0"]
    with pytest.raises(pydantic.ValidationError):
        conf = pconf.PipelineConfigModel(**load_raw_toml)


def test_malformed_pinfo(load_raw_toml):
    assert "delay0" in load_raw_toml["sessions"]
    assert "delay0" in load_raw_toml  # FPS descriptor for the ComputeSession
    assert "procinfo" in load_raw_toml["delay0"]

    del load_raw_toml["delay0"]["procinfo"]["enabled"]
    load_raw_toml["delay0"]["procinfo"]["enabld"] = False  # typo

    with pytest.raises(pydantic.ValidationError):
        conf = pconf.PipelineConfigModel(**load_raw_toml)
