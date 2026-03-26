# Copilot Code Review Instructions for milk

> **Source of truth**: The full rules live in
> [`.agents/rules/`](../../.agents/rules/). This file
> is a focused checklist for Copilot's automated PR
> review. See each rule file for details and rationale.

## Style — `.agents/rules/code-style-guide.md`

- Lines must not exceed **80 characters**
- **Linux kernel C style**, one argument per line
- **Kernel-Doc** (`/** ... */`) on new public functions
- Every `.c` must **explicitly include** its headers
- Minimize variable scope with **code blocks** `{ }`

## Architecture — `.agents/rules/architecture-principles.md`

- Flag **new cross-module `#include`** that may violate
  the dependency graph (`docs/dependency_graph.md`)
- Standalone executables must **never link `CLIcore`** —
  use `_compute` library variants
- Dual-mode files need `#ifdef MILK_NO_CLI` guards

## Performance — `.agents/rules/performance-practices.md`

- Flag **`printf`/`fprintf`** in compute functions
  without `if (VERBOSE > 0)` guard
- Flag **`malloc`/`free` inside per-frame loops**
- Flag **`sqrt()`/`pow()`/`fabs()`** on float data —
  use `sqrtf()`/`powf()`/`fabsf()`
- Flag bare **`0.5`** in float arithmetic — use `0.5f`
- Flag standalone **`if`** for datatype dispatch —
  use `else if` chains
- Flag **hand-written matrix multiply** — use BLAS
- Flag **`pow(2, n)`** for integer n — use `1 << n`
- Suggest **`restrict`** on array pointer parameters

## FPS — `.agents/rules/fpsexec-conventions.md`

- V2 template: 8-section layout from
  `src/milk_module_example/examplefunc_fps_cli_poc.c`
- Must have `.description` in `FPS_APP_INFO` for `-h1`
- CMake: use `add_milk_standalone()`, not manual pattern

## Git — `.agents/rules/git-workflow.md`

- PRs target **`framework-dev`** — never `dev`/`main`
- Conventional commit prefixes: `feat:`, `fix:`, etc.
- AI PRs need "Prompt Summary" and "AI Authorship"
