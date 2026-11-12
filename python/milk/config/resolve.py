"""confresolve: the only way to get a :class:`PipelineConfig` (design-v7 §4.4, D-5).

Layers, lowest to highest: kind defaults, ``includes``, conf.toml, overrides; then ``{var}``
substitution and validation. No side effects, no cache.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

import pydantic
import tomli

from ..errors import ConfigError
from .layout import is_rootdir
from .model import PipelineConfig
from .substitute import substitute_toml_variables_in_nested

_log = logging.getLogger(__name__)
_MODELS_CACHE: set[int] = set()  # config fingerprints already logged in this process

IDENTITY_KEYS = ("long_name", "name", "loop_number")


def kind_defaults(kind: str) -> dict:
    """The kind-defaults layer: {} for "generic"; ConfigError for anything else. R5 replaces the body."""
    if kind != "generic":
        raise ConfigError(f"Unknown kind {kind!r}.")
    return {}


def _merge(low: dict, high: dict) -> dict:
    out = dict(low)
    for key, value in high.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            value = _merge(out[key], value)
        out[key] = value
    return out


def _load(path: Path) -> dict:
    if not path.is_file():
        raise ConfigError(f"Missing config file {path}.", path=path)
    with open(path, "rb") as f:
        return tomli.load(f)


def _set_override(data: dict, dotted: str, value: object, toml: Path) -> None:
    if dotted in IDENTITY_KEYS:
        raise ConfigError(f"Cannot override identity key {dotted!r}.", path=toml)
    *heads, leaf = dotted.split(".")
    node = data
    for head in heads:
        child = node.get(head, {})
        if not isinstance(child, dict):
            raise ConfigError(
                f"Override {dotted!r}: {head!r} is not a table.", path=toml
            )
        node[head] = node = dict(child)
    node[leaf] = value


def confresolve(
    confset: Path, overrides: Mapping[str, object] | None = None
) -> PipelineConfig:
    """confset = the folder holding conf.toml, or a rootdir: then its conf/ with the recorded overrides."""
    confset = Path(confset)
    if is_rootdir(confset):
        recorded = _load(confset / "provenance.toml").get("overrides", {})
        return confresolve(confset / "conf", {**recorded, **(overrides or {})})
    toml = confset / "conf.toml"
    top = _load(toml)

    data: dict = {}
    for rel in top.pop("includes", []):
        path = (confset / rel).resolve()
        if Path(rel).is_absolute() or not path.is_relative_to(confset.resolve()):
            raise ConfigError(f"Include {rel!r} is outside {confset}.", path=toml)
        included = _load(path)
        if "includes" in included:
            raise ConfigError(f"Include {rel!r} has its own includes.", path=path)
        data = _merge(data, included)
    data = _merge(data, top)
    data = _merge(kind_defaults(data.get("kind", "generic")), data)

    for dotted, value in (overrides or {}).items():
        _set_override(data, dotted, value, toml)

    try:
        config = PipelineConfig.model_validate(
            substitute_toml_variables_in_nested(data)
        )
        fingerprint = hash((config.model_dump_json(), toml.absolute()))
        if config.model_extra and fingerprint not in _MODELS_CACHE:
            _MODELS_CACHE.add(fingerprint)
            _log.info("%s: extra keys %s", toml, sorted(config.model_extra))
        return config
    except pydantic.ValidationError as err:
        keys = (
            f"  {'.'.join(map(str, e['loc'])) or '<root>'}: {e['msg']}"
            for e in err.errors()
        )
        raise ConfigError("Invalid config:\n" + "\n".join(keys), path=toml) from err
    except ValueError as err:
        raise ConfigError(str(err), path=toml) from err
