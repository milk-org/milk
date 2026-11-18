"""Named registries of tasks, plans and hook runners."""

from __future__ import annotations

from typing import Generic, TypeVar

from .errors import ConfigError

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, what: str) -> None:
        self.what = what
        self._items: dict[str, T] = {}

    def register(self, name: str, item: T) -> T:
        if name in self._items:
            raise ValueError(f"Duplicate {self.what} {name!r}.")
        self._items[name] = item
        return item

    def __getitem__(self, name: str) -> T:
        if name not in self._items:
            raise ConfigError(
                f"Unknown {self.what} {name!r}; known: {', '.join(self.names())}."
            )
        return self._items[name]

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def names(self) -> list[str]:
        return sorted(self._items)
