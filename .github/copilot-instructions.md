# Copilot Instructions — milk

> **Source of truth**: [`AGENTS.md`](../../AGENTS.md)
> and [`.agents/rules/`](../../.agents/rules/) contain
> the full, maintained coding rules. This file provides
> Copilot with a compact summary. When in doubt, defer
> to the agent rules.

milk is a **real-time image-processing framework** for
Adaptive Optics. Architecture: shared-memory streams
(`ImageStreamIO`), parameters (`FPS`), heartbeat
(`processinfo`). See `AGENTS.md` §1 for details.

## Quick Reference

| Topic | Source of Truth |
|-------|----------------|
| Coding style | `.agents/rules/code-style-guide.md` |
| Performance | `.agents/rules/performance-practices.md` |
| Architecture | `.agents/rules/architecture-principles.md` |
| FPS executables | `.agents/rules/fpsexec-conventions.md` |
| CMake | `.agents/rules/cmake-conventions.md` |
| Git/PR workflow | `.agents/rules/git-workflow.md` |
| V2 template | `src/milk_module_example/examplefunc_fps_cli_poc.c` |
| Dependency graph | `docs/dependency_graph.md` |
| Full onboarding | `AGENTS.md` |

## Critical Rules (Summary)

- Lines ≤ 80 characters, Linux kernel C style
- Kernel-Doc on functions, explicit `#include`s
- `framework-dev` branch only — never `dev`/`main`
- Use `sqrtf()` not `sqrt()` for float data
- No `printf` in compute hot paths
- No `malloc`/`free` inside per-frame loops
- Standalone executables link `_compute` variants
