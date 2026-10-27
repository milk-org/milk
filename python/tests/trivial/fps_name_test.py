from __future__ import annotations

import ast
from pathlib import Path

import milk
from milk.infra.pipeline_models import PipelineConfig

PKG = Path(milk.__file__).parent


def _package_sources() -> dict[Path, str]:
    return {p: p.read_text() for p in PKG.rglob("*.py")}


def test_single_fps_name_definition():
    defs = []
    for path, source in _package_sources().items():
        for node in ast.walk(ast.parse(source, filename=str(path))):
            if isinstance(node, ast.FunctionDef) and node.name == "fps_name":
                defs.append(path)
    assert len(defs) == 1
    assert defs[0].name == "pipeline_models.py"


def test_no_legacy_loop_suffix_in_sources():
    for path, source in _package_sources().items():
        assert "loop_number:03d" not in source, path


def test_fps_name_value():
    conf = PipelineConfig(long_name="pipelinebasic", name="basic", loop_number=0)
    assert conf.fps_name("delay0") == "basic_delay0"


def test_fps_name_independent_of_loop_number():
    conf = PipelineConfig(long_name="pipelinebasic", name="basic", loop_number=7)
    assert conf.fps_name("delay0") == "basic_delay0"


def _is_fps_name_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "fps_name"
    )


def _fps_name_locals(func: ast.AST) -> set[str]:
    return {
        target.id
        for node in ast.walk(func)
        if isinstance(node, ast.Assign) and _is_fps_name_call(node.value)
        for target in node.targets
        if isinstance(target, ast.Name)
    }


def _is_name_cu_fstring(node: ast.JoinedStr) -> bool:
    # f"{a}_{b}": two interpolations joined by an underscore literal.
    parts = node.values
    return any(
        isinstance(left, ast.FormattedValue)
        and isinstance(mid, ast.Constant)
        and isinstance(mid.value, str)
        and mid.value.startswith("_")
        and isinstance(right, ast.FormattedValue)
        for left, mid, right in zip(parts, parts[1:], parts[2:])
    )


def test_deploy_and_pipeline_sites_use_fps_name():
    """Audit of every package site that names an FPS (US2/AC2, SC-003).

    Result: every ``ComputeUnit(...)`` construction in ``milk/pipeline.py`` and
    ``milk/infra/deploy_tasks.py`` takes their FPS name from ``PipelineConfig.fps_name(cu)``,
    directly or through a local assigned from it. No f-string outside ``fps_name`` builds
    ``<name>_<cu>``.
    """
    constructions = 0
    offenders = []
    for path, source in _package_sources().items():
        rel = path.relative_to(PKG)
        tree = ast.parse(source, filename=str(path))
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            is_definition = func.name == "fps_name"
            locals_from_fps_name = _fps_name_locals(func)
            for node in ast.walk(func):
                if isinstance(node, ast.JoinedStr) and not is_definition:
                    if _is_name_cu_fstring(node):
                        offenders.append(f"{rel}:{node.lineno}: f-string builds a name")
                if not (
                    isinstance(node, ast.Call)
                    and getattr(node.func, "id", "") == "ComputeUnit"
                ):
                    continue
                constructions += 1
                fps_arg = node.args[1] if len(node.args) > 1 else None
                if _is_fps_name_call(fps_arg):
                    continue
                if isinstance(fps_arg, ast.Name) and fps_arg.id in locals_from_fps_name:
                    continue
                offenders.append(
                    f"{rel}:{node.lineno}: ComputeUnit FPS name not from fps_name"
                )
    assert constructions, "audit found no ComputeUnit constructions"
    assert not offenders, offenders


def test_no_session_vocabulary_in_package():
    # The word "session" is retired from milk's own names (SC-004). tmux keeps its own term.
    tmux_allowlist: set[str] = set()
    offenders = []
    for path in PKG.rglob("*.py"):
        rel = path.relative_to(PKG)
        if "session" in str(rel).lower() and str(rel) not in tmux_allowlist:
            offenders.append(str(rel))
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            names = []
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                names.append(node.name)
            elif isinstance(node, ast.arg):
                names.append(node.arg)
            elif isinstance(node, ast.Attribute):
                names.append(node.attr)
            elif isinstance(node, ast.Name):
                names.append(node.id)
            for name in names:
                if "session" in name.lower() and name not in tmux_allowlist:
                    offenders.append(f"{rel}:{name}")
    assert not offenders, offenders
