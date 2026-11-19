"""
shm_names.py

Pure-python validators for files in $MILK_SHM_DIR.

Deliberately free of compiled-binding imports (ImageStreamIOWrap, FpsWrap,
ProcessInfoWrap) so that it can be imported shallowly, e.g. by config validators.
"""

from __future__ import annotations

import os


def MILK_SHM_DIR() -> str:
    """
    When testing, we may have to spoof the environment AFTER importing this module.
    It's important to regenerate MILK_SHM_DIR dynamically for this reason.
    """
    return os.environ["MILK_SHM_DIR"]


def _check_name(fname: str, ext: str) -> str:
    """
    Check if fname is
    - a bare name, with or without `ext`
    - a full path + filename (+ `ext`) located at the root of $MILK_SHM_DIR

    Returns:
    The bare name (no directory, no extension)

    Raises:
    ValueError if it's a path-like name that is not at the root of $MILK_SHM_DIR
    """
    if "/" in fname:
        pre, fname = os.path.split(fname)

        # We use it a lot in here, no need to call the dynamic env func every time.
        SHM_DIR = MILK_SHM_DIR()

        # Third condition is for future use if we allow subdirs
        if not pre.startswith(SHM_DIR) or (
            len(pre) > len(SHM_DIR) and pre[len(SHM_DIR)] != "/"
        ):
            raise ValueError(f"Only files in MILK_SHM_DIR ({SHM_DIR}) are supported")

        # Do we need to make folders ? We can't... maybe some day
        if "/" in pre[len(SHM_DIR) :]:
            raise ValueError(
                f"Only files at the root of MILK_SHM_DIR ({SHM_DIR}) are supported"
            )

    if fname.endswith(ext):
        return fname[: -len(ext)]
    return fname


def check_SHM_name(fname: str) -> str:
    """
    Validate an image stream name, ISIO style (`name`) or xaosim style
    (`$MILK_SHM_DIR/name.im.shm`). Returns `name`.
    """
    return _check_name(fname, ".im.shm")


def check_FPS_name(fname: str) -> str:
    """
    Validate an FPS name, `name` or `$MILK_SHM_DIR/name.fps.shm`. Returns `name`.
    """
    return _check_name(fname, ".fps.shm")


def check_PROCINFO_name(fname: str) -> str:
    """
    Validate a processinfo name, `proc[.<fps>].<pid>` or
    `$MILK_SHM_DIR/proc[.<fps>].<pid>.shm`. Returns `proc[.<fps>].<pid>`.
    """
    name = _check_name(fname, ".shm")
    if not name.startswith("proc."):
        raise ValueError(f"ProcessInfo name {name} does not start with 'proc.'")
    return name


def build_SHM_path(name: str) -> str:
    return f"{MILK_SHM_DIR()}/{check_SHM_name(name)}.im.shm"


def build_FPS_path(name: str) -> str:
    return f"{MILK_SHM_DIR()}/{check_FPS_name(name)}.fps.shm"


def build_PROCINFO_path(name: str) -> str:
    return f"{MILK_SHM_DIR()}/{check_PROCINFO_name(name)}.shm"
