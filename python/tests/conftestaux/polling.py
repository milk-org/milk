from __future__ import annotations

import time
from typing import Callable

import pytest

from milk.wait import wait_until


def poll_until_or_fail(
    predicate: Callable[[], object],
    t: float,
    interval: float | None = 1e-5,
    msg: str | None = None,
) -> float:
    """Poll `predicate` until truthy; return elapsed seconds, or pytest.fail after `t`."""
    start = time.monotonic()
    if not wait_until(predicate, t, interval or 0):
        name = getattr(predicate, "__qualname__", repr(predicate))
        pytest.fail(msg or f"{name} still falsy after {t}s")
    return time.monotonic() - start
