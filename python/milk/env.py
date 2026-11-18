"""The only reader of ``MILK_*`` environment variables.

Every function reads the environment when called and caches nothing, so importing ``milk`` needs
no milk environment and tests can monkeypatch the variables.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

from pyMilk.interfacing.shm import MILK_SHM_DIR
from pyMilk.ProcessInfoWrap import PYMILK_ENGINE

from .errors import EnvError

if PYMILK_ENGINE != "SYSTEM":
    # A BUNDLED pyMilk carries its own engine copy, which may diverge from the installed milk.
    warnings.warn(
        f"pyMilk is built with the {PYMILK_ENGINE} engine; milk prefers SYSTEM "
        "(reinstall pyMilk with PYMILK_ENGINE=SYSTEM)",
        stacklevel=2,
    )


def _require_dir(var: str) -> Path:
    value = os.environ.get(var, "")
    if not value:
        raise EnvError(f"Environment variable {var} is unset or empty")
    path = Path(value)
    if not path.is_dir():
        raise EnvError(f"Environment variable {var} is not a directory", path=path)
    return path


def root() -> Path:
    """The milk source tree, ``$MILK_ROOT``."""
    return _require_dir("MILK_ROOT")


def examples(source: str) -> Path:
    """The pipeline examples of ``source`` ("milk" or "cacao") under ``$MILK_ROOT``."""
    sub = {
        "milk": "milk-examples/pipelines",
        "cacao": "plugins/cacao-src/AOloopControl/examples",
    }[source]
    path = root() / sub
    if not path.is_dir():
        raise EnvError(f"No {source} examples directory", path=path)
    return path


def installdir() -> Path:
    """The milk install tree, ``$MILK_INSTALLDIR``."""
    return _require_dir("MILK_INSTALLDIR")


def shm_dir() -> Path:
    """The shared-memory directory, as pyMilk reports it."""
    _require_dir("MILK_SHM_DIR")
    return Path(MILK_SHM_DIR())


def lib(name: str) -> Path:
    """Path of the milk shared library ``name`` (e.g. ``libmilkcommon.so``)."""
    path = installdir() / "lib" / name
    if not path.is_file():
        raise EnvError(f"Milk library {name} not found", path=path)
    return path
