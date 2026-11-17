---
description: Naming and execution convention for shell scripts and milk scripts.
---

# Script Naming and Execution Rules

When creating or refactoring scripts in the `milk` framework, consult the reference documentation
first:

- [`docs/guide/scripting.md`](../../docs/guide/scripting.md) — top-level `.milk` / `milk.bash`
  language summary, module commands, entities, events, and examples
- [`docs/guide/calculator.md`](../../docs/guide/calculator.md) — `@calc` expressions, functions, and
  `milk-fpsexec-milk-calc` / `milk-fpsexec-milk-script` units
- [`docs/guide/sessions.md`](../../docs/guide/sessions.md) — `@session` management and
  `source milk.bash`
- [`docs/guide/scripts.md`](../../docs/guide/scripts.md) — installed standalone `milk-*` scripts

## Naming & Execution Conventions

- **Executable OS Scripts (`milk-*`)**: Standalone shell executables installed to `bin/` MUST be
  prefixed with `milk-` (e.g., `milk-check`, `milk-cc`) and support `-h` / `--help`.
- **Milk Scripts (`*.milk`)**: Scripts using `module.command`, `@keyword`, and `@{…}` expansions
  MUST use the `.milk` extension and run via `#!/usr/bin/env milk-script`, `milk-cli -s file.milk`,
  or `@source file.milk`.
