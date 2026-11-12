from __future__ import annotations

import importlib
import os
import pkgutil
import sys

import pytest

import milk
from milk import cliwrap, env
from milk.errors import EnvError


@pytest.fixture
def no_milk_env(monkeypatch):
    for var in [v for v in os.environ if v.startswith("MILK_")]:
        monkeypatch.delenv(var)
    cliwrap.clear_caches()
    yield
    cliwrap.clear_caches()


def test_imports_without_milk_env(no_milk_env, monkeypatch):
    names = [m.name for m in pkgutil.walk_packages(milk.__path__, "milk.")]
    assert "milk.env" in names
    for name in ["milk", *names]:
        monkeypatch.delitem(sys.modules, name, raising=False)
        importlib.import_module(name)
    cliwrap.clear_caches()
    assert cliwrap.HAVE_CLI is False


def test_functions_raise_env_error_without_env(no_milk_env):
    with pytest.raises(EnvError, match="MILK_INSTALLDIR"):
        env.installdir()
    with pytest.raises(EnvError, match="MILK_INSTALLDIR"):
        cliwrap._find_build_tags()


@pytest.mark.parametrize("func", [env.root, env.installdir])
def test_env_error_names_missing_var(func, monkeypatch, tmp_path):
    var = "MILK_ROOT" if func is env.root else "MILK_INSTALLDIR"
    monkeypatch.delenv(var, raising=False)
    with pytest.raises(EnvError, match=var):
        func()
    monkeypatch.setenv(var, "")
    with pytest.raises(EnvError, match=var):
        func()
    monkeypatch.setenv(var, str(tmp_path / "missing"))
    with pytest.raises(EnvError, match=var):
        func()
    monkeypatch.setenv(var, str(tmp_path))
    assert func() == tmp_path


def test_lib_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("MILK_INSTALLDIR", str(tmp_path))
    with pytest.raises(EnvError, match="libnope.so"):
        env.lib("libnope.so")


def test_shm_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("MILK_SHM_DIR", raising=False)
    with pytest.raises(EnvError, match="MILK_SHM_DIR"):
        env.shm_dir()
    monkeypatch.setenv("MILK_SHM_DIR", "")
    with pytest.raises(EnvError, match="MILK_SHM_DIR"):
        env.shm_dir()
    monkeypatch.setenv("MILK_SHM_DIR", str(tmp_path / "missing"))
    with pytest.raises(EnvError, match="MILK_SHM_DIR"):
        env.shm_dir()
    monkeypatch.setenv("MILK_SHM_DIR", str(tmp_path))
    assert env.shm_dir() == tmp_path
