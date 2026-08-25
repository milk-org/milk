from __future__ import annotations

import re
import typing as typ

import tomli
from pathlib import Path

from .pipeline_config import PipelineConfig, PipelineConfigModel

if typ.TYPE_CHECKING:
    from pyMilk.interfacing.fps import FPVal

    FPValNest: typ.TypeAlias = FPVal | dict[str, "FPValNest"]


def load_pipeline_config(conf_folder: str | Path) -> PipelineConfig:
    with open(Path(conf_folder) / "conf.toml", "rb") as f:
        pre_sub_data = tomli.load(f)

    data = substitute_toml_variables_in_nested(pre_sub_data)

    # This will raise a ValidationError if the toml isn't good
    parsed = PipelineConfigModel(**data)  # type: ignore

    config = PipelineConfig()
    config.name = parsed.name
    config.loop_number = parsed.loop_number
    config.sessions = parsed.sessions
    # Each session name is also a top-level table holding its own config
    config.session_configs = {  # type: ignore
        session_name: data[session_name] for session_name in config.sessions
    }

    return config


def denest_toml_dicts(fp_param_dict: dict[str, FPValNest]) -> dict[str, FPVal]:
    copied: dict[str, FPVal] = {}
    for key, value in fp_param_dict.items():
        if isinstance(value, dict):
            new_value = denest_toml_dicts(value)
            for kk, vv in new_value.items():
                copied[f"{key}.{kk}"] = vv
        else:
            copied[key] = value

    return copied


def renest_toml_dicts(fp_param_dict: dict[str, FPVal]) -> dict[str, FPValNest]:
    copied: dict[str, FPValNest] = {}
    for key, value in fp_param_dict.items():
        if "." in key:
            head, tail = key.split(".", 1)
            if not head in copied:
                copied[head] = {}
            copied[head][tail] = value  # type: ignore
        else:
            copied[key] = value

    for key in copied:
        if isinstance(copied[key], dict):
            copied[key] = renest_toml_dicts(copied[key])  # type: ignore

    return copied


def substitute_toml_variables_in_nested(
    data: dict[str, FPValNest],
) -> dict[str, FPValNest]:
    denested = denest_toml_dicts(data)
    substitute_toml_variables_in_denested(denested)
    return renest_toml_dicts(denested)


from dataclasses import dataclass

# Matches {literal} or {literal:format} blocks, e.g. "{a.b}" or "{a.b:.2f}"
_SUBST_BLOCK_RE = re.compile(r"\{(?P<literal>[^{}:]+)(?::(?P<format>[^{}]*))?\}")


def substitute_toml_variables_in_denested(data: dict[str, FPVal]):
    class Substitution:
        def __init__(self, s: str):
            self.raw = s
            self.refs: list[tuple[str, str | None]] = [
                (m.group("literal").strip(), m.group("format"))
                for m in _SUBST_BLOCK_RE.finditer(s)
            ]
            # Same blocks with the literal stripped, e.g. "{a.b:.2f}" -> "{:.2f}"
            self.formattable = _SUBST_BLOCK_RE.sub(
                lambda m: (
                    "{:" + m.group("format") + "}"
                    if m.group("format") is not None
                    else "{}"
                ),
                s,
            )
            self.depends_upon: set[str] = {a for (a, _) in self.refs}
            self.depended_by: set[str] = set()

        def substitute(self, data: dict[str, FPVal]) -> FPVal:
            if self.formattable == "{}":  # Single token substitution, maintain type.
                return data[self.refs[0][0]]
            return self.formattable.format(*[data[k] for (k, _) in self.refs])

    subs: dict[str, Substitution] = {}
    unresolved: set[str] = set()
    resolved: set[str] = set()
    resolvable: set[str] = set()

    for key, value in data.items():
        if not isinstance(value, str) or not "{" in value:
            resolved.add(key)
        else:
            subs[key] = Substitution(value)
            unresolved.add(key)

    # Invert the dependency graph and validate all keys do exist !!
    for key, sub in subs.items():
        for dep in sub.depends_upon:
            if dep not in data:
                raise ValueError(
                    f"Variable {dep} needed by {key} (={sub.raw}) not found in toml data."
                )
            if dep in subs:
                subs[dep].depended_by.add(key)

        if all({v in resolved for v in sub.depends_upon}):
            unresolved.remove(key)
            resolvable.add(key)

    while len(resolvable) > 0:
        resolving = resolvable.pop()
        data[resolving] = subs[resolving].substitute(data)
        resolved.add(resolving)
        for dep in subs[resolving].depended_by:
            if dep in unresolved:
                if all({v in resolved for v in subs[dep].depends_upon}):
                    unresolved.remove(dep)
                    resolvable.add(dep)

    if len(unresolved) > 0:
        raise ValueError(
            f"Cannot resolve variable substitutions: remaining {unresolved}"
        )
