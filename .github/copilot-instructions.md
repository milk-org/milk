# Copilot Instructions — milk

This file provides GitHub Copilot with project-specific
context. For the full agent onboarding guide, see
[AGENTS.md](../../AGENTS.md) at the repo root.

## Project Overview

`milk` is a real-time image-processing framework for
Adaptive Optics. It uses shared-memory IPC: streams
(`ImageStreamIO`), parameter sync (`FPS`), and heartbeat
telemetry (`processinfo`).

## Coding Conventions

- Line length ≤ 80 characters
- Linux kernel C coding style
- Kernel-Doc (`/** ... */`) for function documentation
- Every `.c` file explicitly includes the headers it uses
- Use code blocks `{ }` to minimize variable scope
- Function prototypes: multi-line, one argument per line
- Prioritize runtime performance

## Templates

- **New FPS compute unit**: follow
  `src/milk_module_example/examplefunc_fps_cli_poc.c`
- **Stream processing**: follow
  `src/milk_module_example/examplefunc4_streamprocess.c`
- **FPS-enabled function**: follow
  `src/milk_module_example/examplefunc2_FPS.c`
- **CMakeLists.txt**: follow
  `src/milk_module_example/CMakeLists.txt`

## Key Rules

- Check `docs/dependency_graph.md` before adding
  cross-module dependencies.
- Standalone executables must link `_compute` variants,
  never `CLIcore`. Use `add_milk_standalone()` or
  `add_cacao_standalone()` CMake helpers.
- Always compile-test after C or CMake changes.
- Update module README when source files change.
- Update `docs/programmers_guide.md` after architectural
  changes.

## Dual-Mode Files

Files compiled as both CLI library and standalone use:

```c
#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"
```

## Agent Rules and Workflows

See `.agents/rules/` for always-on guardrails and
`.agents/workflows/` for on-demand task templates.
Full index: `docs/code_assist.md`.
