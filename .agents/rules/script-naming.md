---
description: Naming and execution convention for shell scripts and milk scripts.
---

# Script Naming and Execution Rules

When creating or refactoring scripts in the Milk framework, adhere strictly to the following conventions:

- **Executable OS Scripts** (`milk-*`): Scripts designed to be run as standalone shell executables MUST be prefixed with `milk-` (e.g., `milk-check`, `milk-logshim`). They must be directly executable and MUST include comprehensive support for a `-h` or `--help` command-line argument.
- **Native CLI Scripts** (`*.milk`): Scripts authored in the native scripting language designed to be evaluated by the CLI orchestrator MUST use the `.milk` extension (e.g., `makecircleofdisks.milk`). They are not intended to be installed as standalone OS-level commands; instead, they should normally be run by passing them to the CLI orchestrator: `milk-cli -s scriptname.milk` (or via a shebang that invokes `milk-cli -s`).
