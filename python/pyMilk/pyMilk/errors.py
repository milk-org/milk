from __future__ import annotations


class AutoRelinkError(Exception):
    pass


class AutoRelinkTypeError(AutoRelinkError):
    pass


class AutoRelinkSizeError(AutoRelinkError):
    pass
