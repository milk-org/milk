"""Named registries of tasks, plans and hook runners."""

from __future__ import annotations

import logging
from typing import Generic, TypeVar

from .errors import ConfigError

T = TypeVar("T")

_log = logging.getLogger(__name__)


class Registry(Generic[T]):
    """A lookup falls back to ``parent``; a name registered here overrides the parent's."""

    def __init__(self, what: str, parent: Registry[T] | None = None) -> None:
        self.what = what
        self.parent = parent
        self._items: dict[str, T] = {}

    def register(self, name: str, item: T) -> T:
        if name in self._items:
            raise ValueError(f"Duplicate {self.what} {name!r}.")
        if self.parent is not None and name in self.parent:
            _log.warning("Local %s %r overrides another one.", self.what, name)
        self._items[name] = item
        return item

    def __getitem__(self, name: str) -> T:
        if name in self._items:
            return self._items[name]
        if self.parent is not None and name in self.parent:
            return self.parent[name]
        raise ConfigError(
            f"Unknown {self.what} {name!r}; known: {', '.join(self.names())}."
        )

    def __contains__(self, name: str) -> bool:
        return name in self._items or (self.parent is not None and name in self.parent)

    def names(self) -> list[str]:
        inherited = self.parent.names() if self.parent is not None else []
        return sorted({*inherited, *self._items})
