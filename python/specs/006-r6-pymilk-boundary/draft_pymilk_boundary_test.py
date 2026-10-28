"""R6 acceptance tests: config validation without native bindings; ISIO contract. See specs/006-r6-pymilk-boundary/brief.md.

The FPS side of the contract is tests/infra/compute_unit_fps_runtime_sync_test.py (17 FPS types,
milk-written and pyMilk-read). It is left untouched here.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pydantic
import pytest
import tomli

from milk.infra.pipeline_models import PipelineConfig

BASIC_TOML = (
    Path(__file__).parents[1] / "resources" / "pipelinebasic-conf" / "conf.toml"
)
NATIVE = ("pyMilk.ImageStreamIOWrap", "pyMilk.FpsWrap", "pyMilk.ProcessInfoWrap")


def _raw_basic() -> dict:
    with open(BASIC_TOML, "rb") as f:
        return tomli.load(f)


# ---- Config validation is pure Python (py) ----


def test_config_validation_without_native_bindings():
    """Blocking every pyMilk native module must not stop config validation."""
    script = textwrap.dedent(f"""
        import sys, tomli
        for name in {NATIVE!r}:
            sys.modules[name] = None  # any import of it now raises ImportError
        from milk.infra.pipeline_models import PipelineConfig
        with open({str(BASIC_TOML)!r}, "rb") as f:
            conf = PipelineConfig.model_validate(tomli.load(f))
        assert conf.systemlinkage["shared"]["input"] == "otherloop_output"
        """)
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr


def test_shm_name_suffix_is_stripped():
    raw = _raw_basic()
    raw["systemlinkage"]["shared"] = {"input": "elsewhere.im.shm"}
    assert PipelineConfig.model_validate(raw).systemlinkage["shared"] == {
        "input": "elsewhere"
    }


@pytest.mark.parametrize("remote", ["/milk/shm/elsewhere.im.shm", "sub/elsewhere"])
def test_shm_name_with_path_refused(remote):
    raw = _raw_basic()
    raw["systemlinkage"]["shared"] = {"input": remote}
    with pytest.raises(pydantic.ValidationError):
        PipelineConfig.model_validate(raw)


# ---- ISIO contract: a pyMilk-written stream read by milk's own ImageStreamIO (milk) ----


def test_isio_contract_pymilk_writes_milk_reads():
    from pyMilk.interfacing.shm import SHM

    exe = shutil.which("isio-stream-info")
    assert (
        exe
    ), "isio-stream-info not on PATH: build milk with src/isio-tools (needs pkg-config ImageStreamIO)"

    shm = SHM("r6_contract", np.zeros((3, 5), np.float32))
    try:
        shm.set_data(np.ones((3, 5), np.float32))  # cnt0 moves past 0
        md = shm.md
        proc = subprocess.run([exe, "r6_contract"], capture_output=True, text=True)
        context = (
            f"pyMilk ISIO version {md.version!r}; milk isio-stream-info said:\n"
            f"{proc.stdout}{proc.stderr}"
        )
        assert proc.returncode == 0 and "READ ERROR" not in proc.stdout, context

        # "<name>  <cnt0>  <datatype>   <naxis> [ <s0> <s1> <s2> ]"
        fields = proc.stdout.replace("[", " ").replace("]", " ").split()
        name, cnt0, datatype, naxis, *size = fields
        assert name == "r6_contract", context
        assert int(cnt0) == md.cnt0, context
        assert int(datatype) == int(md.datatype), context
        assert int(naxis) == md.naxis == 2, context
        assert [int(s) for s in size[: md.naxis]] == list(md.size), context
    finally:
        shm.destroy()
