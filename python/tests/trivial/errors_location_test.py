from __future__ import annotations

import pytest

from milk.operate import compute_unit
from milk.errors import (
    BestEffortError,
    ComputeUnitError,
    MilkError,
    TaskCannotRunError,
    TaskError,
)


def test_milk_error_context_in_str():
    err = MilkError("boom", pipeline="loop", cmd=["a", "b"], stderr="bad")
    assert err.pipeline == "loop" and err.compu is None and err.path is None
    text = str(err)
    assert text.startswith("boom")
    assert "pipeline: loop" in text and "stderr: bad" in text
    assert "compu" not in text and "task" not in text


def test_hierarchy():
    assert issubclass(TaskCannotRunError, TaskError)
    assert issubclass(TaskError, MilkError)


def test_best_effort_error_message():
    first, second = ValueError("one"), OSError("two")
    err = BestEffortError([("step a", first), ("step b", second)], pipeline="loop")
    assert err.failures == [("step a", first), ("step b", second)]
    text = str(err)
    assert "step a: one" in text and "step b: two" in text
    assert "pipeline: loop" in text


def test_compute_unit_error_context(monkeypatch):
    class _NoFPS(Exception):
        pass

    def _no_fps(name):
        raise _NoFPS

    monkeypatch.setattr(compute_unit, "FPS", _no_fps)
    monkeypatch.setattr(compute_unit, "FPSDoesntExistError", _NoFPS)
    # `sh <fpsname>:<cmd>` fails: no such script, with a message on stderr
    compu = compute_unit.ComputeUnit("sh", "myfps")
    with pytest.raises(ComputeUnitError) as exc:
        compu._subprocess_complete("fps")
    err = exc.value
    assert err.compu == "myfps"
    assert err.cmd == ["sh", "myfps:fps"]
    assert err.stderr
    text = str(err)
    assert "compu: myfps" in text and "cmd:" in text and "stderr:" in text
