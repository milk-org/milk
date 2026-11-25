"""The milk wait primitive."""

from __future__ import annotations

import time
from typing import Callable


def wait_until(
    predicate: Callable[[], object], timeout: float, interval: float = 1e-3
) -> bool:
    """Poll ``predicate`` until truthy (True) or ``timeout`` seconds have passed (False)."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval)
    return True
