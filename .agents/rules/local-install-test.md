---
description: Install and test milk without root, under _build/.
---

# Local Install and Test

**Never** install to the system prefix (`/usr/local/`, `/usr/`, etc.): the agent has no root
privileges and `sudo` is not available. Do not use `sudo`, and do not copy binaries into system
directories.

Install and test as [Dev installs without sudo](../../docs/developer/tooling.md#6-dev-installs-without-sudo)
says (prefix under `_build/`, environment variables, cleanup).
