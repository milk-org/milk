from __future__ import annotations

import ast
from pathlib import Path

import pytest

import milk
from milk import compute_unit
from milk.errors import (
    BestEffortError,
    ComputeUnitError,
    MilkError,
    TaskCannotRunError,
    TaskError,
)


def _base_name(base: ast.expr) -> str:
    if isinstance(base, ast.Attribute):
        return base.attr
    if isinstance(base, ast.Name):
        return base.id
    return ""


def test_only_errors_module_defines_exceptions():
    root = Path(milk.__file__).parent
    offenders = []
    for path in root.rglob("*.py"):
        if path == root / "errors.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ClassDef):
                continue
            for base in map(_base_name, node.bases):
                if base in ("Exception", "BaseException") or base.endswith(
                    ("Error", "Exception")
                ):
                    offenders.append(f"{path.relative_to(root)}:{node.name}")
    assert not offenders


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
