"""milk-pipeinit: file operations on confdirs, never the live system (design-v7 §4.9)."""

from .artifacts import Fingerprint, conf_files, decl_digest, fingerprint, status
from .clone import clone
from .hooks import consolidate, read_tag

__all__ = [
    "Fingerprint",
    "fingerprint",
    "conf_files",
    "decl_digest",
    "status",
    "consolidate",
    "read_tag",
    "clone",
]
