# Agent Onboarding — milk

This document is the **fast-start guide for AI coding
agents** working on the `milk` codebase. Read it first,
then follow the curated pointers below to build deep
context efficiently.

> [!TIP]
> Human contributors: this document doubles as an
> architecture cheat-sheet. Everything here is accurate
> for humans too.

---

## Quick Start (5 minutes)

1. Read sections 1–3 below (What is milk, Reading
   order, Source tree map).
2. Run `/compile-test` to verify your build works.
3. For your first task, try `/add-function` or
   `/create-fpsexec`.
4. Read the relevant rules in `.agents/rules/` as
   you encounter them.

**Good first contributions** (suited for AI-assisted
work):

- Add Kernel-Doc to undocumented functions
  (use `batch-kernel-doc` skill)
- Add CLI robustness tests (use `cli-test-writer`)
- Fix type consistency gaps (`/check-type-consistency`)
- Update module READMEs
- Migrate V1 fpsexec code to V2 (`/migrate-to-v2`)

---

## 1. What Is milk?

`milk` is a **real-time image-processing framework**
designed for Adaptive Optics (AO) and high-performance
scientific computing. Its defining characteristic is a
**shared-memory micro-service architecture**: many small
standalone executables (compute units) communicate via
zero-copy tensors in `/dev/shm/` and are configured
through a shared-memory parameter system.

**Three pillars:**

| Pillar                               | Purpose                                                     | Shared Memory Path  |
| ------------------------------------ | ----------------------------------------------------------- | ------------------- |
| **ImageStreamIO** (Streams)          | Zero-copy n-dimensional data passing between processes      | `/dev/shm/*.im.shm` |
| **FPS** (Function Processing System) | Real-time parameter sync, state control (run/stop/conf)     | `/dev/shm/fps.*`    |
| **processinfo**                      | Heartbeat telemetry, loop rate profiling, health monitoring | `/dev/shm/proc.*`   |

Compute units run inside **tmux sessions** for fault
isolation — a crash in one unit never takes down others.

---

## 2. Read These Next (in order)

| Priority   | Document                                                 | What you learn                                                                                      |
| ---------- | -------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| 🔴 **1st** | [`docs/programmers_guide.md`](docs/programmers_guide.md) | Architecture overview, V2 compute unit template, directory map, CMake conventions, header hierarchy |
| 🔴 **2nd** | [`docs/dependency_graph.md`](docs/dependency_graph.md)   | Full build-tier diagrams, library link tables, `_compute` variant patterns                          |
| 🟠 **3rd** | [`docs/streams.md`](docs/streams.md)                     | `IMGID` C API, stream creation/connection, semaphore model, stream modifiers (`@S:`, `@L:`, `@F:`)  |
| 🟠 **4th** | [`docs/fps.md`](docs/fps.md)                             | FPS parameter types, tmux dispatch, `milk-fpsCTRL`, `fpslist.txt` workflow                          |
| 🟡 **5th** | [`docs/code_assist.md`](docs/code_assist.md)             | Index of all agent rules and workflows                                                              |
| 🟡 **6th** | [`docs/procinfo.md`](docs/procinfo.md)                   | `PROCESSINFO` C API, loop profiling                                                                 |

---

## 3. Source Tree Map

```
milk/
├── src/
│   ├── engine/                    ← Core libraries (POSIX only, no CLI)
│   │   ├── ImageStreamIO/         ← Zero-copy shared memory streams
│   │   ├── libfps/                ← FPS core (parameter management)
│   │   ├── libprocessinfo/        ← Heartbeat + process tracking
│   │   └── libmilkdata/           ← IMGID struct, image utilities
│   ├── cli/                       ← User-facing tools
│   │   ├── libmilkscript/         ← Standalone scripting engine
│   │   ├── CLIcore/               ← Interactive shell framework
│   │   ├── libmilkTUI/            ← TUI widget library (ncurses)
│   │   └── streamCTRL/            ← Stream monitor tool
│   ├── coremods/                  ← Core computation modules
│   │   └── COREMOD_{arith,memory,tools,iofits}/
│   └── milk_module_example/       ← ★ START HERE for templates
├── plugins/
│   ├── milk-extra-src/            ← General plugins (fft, linalg, image*)
│   └── cacao-src/ → ~/src/cacao   ← AO loop control (symlink)
├── .agents/
│   ├── rules/                     ← Always-on agent guardrails
│   ├── skills/                    ← Specialized instruction sets
│   └── workflows/                 ← On-demand task templates
└── docs/                          ← Documentation
```

**Key template files** (always start from these):

- **New FPS compute unit**: `src/milk_module_example/examplefunc_fps_cli_poc.c`
- **Stream processing**: `src/milk_module_example/examplefunc4_streamprocess.c`
- **FPS-enabled function**: `src/milk_module_example/examplefunc2_FPS.c`
- **CMakeLists.txt**: `src/milk_module_example/CMakeLists.txt`

---

## 4. Key Code Patterns

### 4.1 The V2 8-Section Layout

Every FPS compute unit follows this standardized layout
(see `examplefunc_fps_cli_poc.c`):

| Section | Content                                                                   |
| ------- | ------------------------------------------------------------------------- |
| 1       | `FPS_APP_INFO` — name, CLI keyword, description                           |
| 2       | Local C variables for parameters                                          |
| 3       | `FPS_PARAMS` X-macro — binds C vars to FPS shared memory                  |
| 4       | `fpsexec()` — pure computation function                                   |
| 5       | `CLIcmddata` — CLI registry scoping                                       |
| 6       | Compute wrapper — `INSERT_STD_PROCINFO_COMPUTEFUNC_*`                     |
| 7       | Module registration — `CLIADDCMD_*` (guarded by `#ifndef FPS_STANDALONE`) |
| 8       | `FPS_MAIN_STANDALONE_V2` — standalone `main()`                            |

### 4.2 Dual-Mode Files

Source files compiled both as shared library (CLI) and
standalone executable use conditional includes:

```c
#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"   /* stubs for standalone */
#else
#include "CLIcore.h"              /* full CLI types */
#endif
#include "fps.h"
```

### 4.3 `_compute` Library Variants

Standalone executables must **never** link `CLIcore`.
Instead they link `_compute` variants of libraries
(compiled with `MILK_NO_CLI`):

```
milkfft        → milkfft_compute        (standalone safe)
milkstatistic  → milkstatistic_compute   (standalone safe)
```

Use `add_milk_standalone()` or `add_cacao_standalone()`
CMake helpers — they set up the correct link set
automatically.

### 4.4 IMGID API

`IMGID` is the standard way to reference images/streams:

```c
IMGID img = imgid_make_from_name("mystream");
img.mdt->naxis = 2;
img.mdt->size[0] = 128;
img.mdt->size[1] = 128;
img.mdt->shared = 1;
imgid_mkimage(&img);

// Access pixel data:
float *data = img.im->array.F;
```

---

## 5. Common Pitfalls

> [!CAUTION]
> These are the mistakes agents most commonly make.

1. **Adding cross-module dependencies without checking
   `docs/dependency_graph.md`.** The build is layered —
   engine → core → full. Adding a dependency in the wrong
   direction breaks lower tiers.

2. **Linking standalone executables to CLIcore.** Use
   `_compute` variants only. Run
   `milk-check-standalone-deps` to verify.

3. **Implicit header includes.** Every `.c` file must
   include exactly the headers it uses. Don't rely on
   `CLIcore.h` pulling in `math.h` or `stdlib.h`.

4. **Forgetting to update `docs/programmers_guide.md`**
   after architectural changes (the
   `maintain-programmers-guide` rule enforces this).

5. **Lines > 100 characters.** The project enforces
   short lines for readability.

6. **Using placeholders in module READMEs.** Always write
   accurate descriptions derived from source code.

7. **Not compiling after edits.** Always run the
   `/compile-test` workflow after modifying C or CMake
   files.

---

## 6. Agent Rules (Always Active)

These rules in `.agents/rules/` are automatically loaded
and enforced. Know what they require:

| Rule                                 | Key Requirement                                                                   |
| ------------------------------------ | --------------------------------------------------------------------------------- |
| `agent-docs-update.md`               | Update AGENTS.md and code_assist.md when agent files change                       |
| `architecture-principles.md`         | Check dependency graph before adding deps                                         |
| `cmake-conventions.md`               | Use `PUBLIC`/`INTERFACE` properties; modules own their headers                    |
| `code-style-guide.md`                | 100-char lines, Kernel-Doc, Linux kernel style, explicit includes                 |
| `common-agent-mistakes.md`           | Consolidated checklist of frequent AI code-generation pitfalls                    |
| `compile-after-edit.md`              | Always compile-test after C/CMake changes                                         |
| `concurrency-practices.md`           | Semaphore protocol, FPS sync, process coordination                                |
| `defensive-programming-practices.md` | Buffer safety, pointer discipline, bounded input validation                       |
| `documentation-site.md`              | MkDocs structure, page creation, tag categories                                   |
| `documentation-standards.md`         | Consistent markdown, shell prompts, link checking                                 |
| `error-handling-practices.md`        | Use milkDebugTools.h macros for errors                                            |
| `files-directories.md`               | cacao lives at `plugins/cacao-src` → `~/src/cacao`                                |
| `fpsexec-conventions.md`             | V2 template, 8-section layout, `-h1` support                                      |
| `git-workflow.md`                    | Small changes direct to `framework-dev`; ask user for branch/PR on larger changes |
| `help-consistency.md`                | Cross-check all sibling help sources                                              |
| `help-message-standard.md`           | Unified help format, flags (`-h`/`-h1`/`-hm`), color palette via `milk_help.h`    |
| `local-install-test.md`              | Install to `_build/_install`; never `sudo` or system paths                        |
| `maintain-programmers-guide.md`      | Update programmer's guide on arch changes                                         |
| `module-deps-declaration.md`         | MODULE_DEPS and INIT_MODULE_LIB_DEPS macros                                       |
| `naming-conventions.md`              | File, function, variable, macro naming; loop index types                          |
| `performance-practices.md`           | SIMD, BLAS, pointer alignment, type dispatch, CPU pinning                         |
| `readme-update.md`                   | Update module README when files change                                            |
| `run-milk-commands.md`               | Environment setup, SHM cleanup, tmux guidance                                     |
| `script-docs.md`                     | Update `docs/scripts.md` when scripts change                                      |
| `script-naming.md`                   | `milk-*` for OS executables, `.milk` for CLI scripts                              |
| `shared-memory-safety.md`            | SHM cleanup, stale detection, stream creation                                     |
| `testing-practices.md`               | Run tests after changes; add regression tests                                     |
| `tui-browser-testing.md`             | milk TUIs cannot be tested using browser testing tools                            |
| `whatsnew-update.md`                 | Add entry to `docs/whatsnew.md` for significant features                          |

---

## 7. Skills

Skills are specialized instruction sets located in
`.agents/skills/`. Each skill provides the agent with
deep context, helper scripts, and rules for a specific
technical domain. Agents consult these automatically
when domain-specific tasks require extended capabilities.

| Skill                        | When to use                                                                                    |
| ---------------------------- | ---------------------------------------------------------------------------------------------- |
| `advanced-math-patterns`     | High-performance mathematical and DSP operations, BLAS, FFT, and vectorization                 |
| `api-quick-reference`        | API cheat sheet for IMGID, processinfo macros, stream variables, datatypes, and parameter sync |
| `batch-kernel-doc`           | Systematic Kernel-Doc documentation passes                                                     |
| `cli-test-writer`            | Writing CLI robustness test cases                                                              |
| `cmake-patterns`             | Module CMake setup, standalone builds, `_compute` variants                                     |
| `debug-cli-behavior`         | Investigating CLI crashes, display bugs, missing errors                                        |
| `diagnose-build-failure`     | Triaging CMake/GCC build errors                                                                |
| `feature-planner`            | Structured planning and decomposition for new features                                         |
| `fps-parameter-guide`        | FPS parameter types, flags, X-macro patterns                                                   |
| `imagestream-internals`      | SHM stream layout, semaphore protocol, circular buffers                                        |
| `milk-script-writer`         | Generate correct milk-cli scripts from natural language prompts                                |
| `module-loading-internals`   | Debugging module registration, empty commands                                                  |
| `optimize-compute-function`  | Systematic performance optimization methodology                                                |
| `plugin-creator`             | Scaffolding a new plugin module with CMake and module registration                             |
| `pr-preparation`             | Packaging work into a pull request                                                             |
| `pseudocode-to-compute-unit` | Translating algorithms to V2 compute units                                                     |
| `refactor-c-source`          | Splitting large C files into smaller modules                                                   |
| `stream-modifier-guide`      | IMGID parsing, `@S:`/`@L:`/`@F:` modifiers, slice syntax                                       |

---

## 8. Workflows (On-Demand)

Invoke these as slash commands when working on the
listed task types:

| Command                     | When to use                                              |
| --------------------------- | -------------------------------------------------------- |
| `/add-cli-command`          | Adding a CLI command to a module                         |
| `/add-function`             | Adding a function to an existing module                  |
| `/add-new-module`           | Creating a new module (core or plugin)                   |
| `/add-stream-processor`     | Creating a stream processing loop                        |
| `/compile-test`             | After any C or CMake edit                                |
| `/create-fpsexec`           | Scaffolding a new standalone executable                  |
| `/create-plugin`            | Scaffolding a new plugin module with full boilerplate    |
| `/add-stream-processor`     | Creating a stream processing loop                        |
| `/add-cli-command`          | Adding a CLI command to a module                         |
| `/fix-bug`                  | Investigating, fixing, and verifying a bug               |
| `/migrate-to-v2`            | Converting V1 fpsexec code to V2 template                |
| `/review-pr`                | Reviewing a PR for standards compliance                  |
| `/setup-dev-environment`    | First-time development setup                             |
| `/update-programmers-guide` | After architectural changes                              |
| `/audit-code-quality`       | Audit code for readability, simplicity, and organization |
| `/audit-help-consistency`   | After editing help text anywhere                         |
| `/check-type-consistency`   | Auditing data type handling in switch blocks             |
| `/cli-robustness-test`      | Running CLI robustness test suite                        |
| `/inspect-machine-code`     | Assembly inspection for performance                      |
| `/sync-worktree`            | Syncing worktree to latest framework-dev                 |
| `/update-docs-site`         | Adding or updating MkDocs pages                          |
| `/update-scripts-docs`      | After adding or modifying shell scripts                  |

---

## 9. Build System Quick Reference

### Build tiers

| Tier      | CMake Flags                        | What's built                                      |
| --------- | ---------------------------------- | ------------------------------------------------- |
| Engine    | `-DUSE_COREMODS=OFF -DUSE_CLI=OFF` | ImageStreamIO, milkfps, milkdata, milkprocessinfo |
| Core      | `-DUSE_CLI=OFF -DUSE_CFITSIO=OFF`  | Engine + COREMODs (arith, memory, tools)          |
| Core+FITS | `-DUSE_CLI=OFF`                    | Core + COREMOD_iofits                             |
| Full      | _(defaults)_                       | Everything: CLI + all plugins                     |

### Compile-test cycle

```bash
$ cd _build
$ make -j$(nproc)
$ make install
$ ctest --output-on-failure
```

### Standalone CMake helpers

```cmake
add_milk_standalone(myname  myname.c)           # milk-fpsexec-myname
add_cacao_standalone(myname myname.c)           # cacao-fpsexec-myname
add_cacao_standalone_plugins(myname myname.c fft imagegen)
```

---

## 10. cacao (Adaptive Optics Plugin)

`cacao` source lives at `plugins/cacao-src/`, which is a
symlink to `~/src/cacao`. When editing cacao modules,
treat this symlink directory as part of the workspace.

cacao modules follow the same V2 template conventions as
milk modules, but use `add_cacao_standalone()` or
`add_cacao_standalone_plugins()` for their executables.

---

## 11. Compile-Time Guards

| Macro            | Set By                   | Effect                                                    |
| ---------------- | ------------------------ | --------------------------------------------------------- |
| `MILK_NO_CLI`    | CMake `-DMILK_NO_CLI`    | Excludes CLI code, uses `CLIcore_standalone.h`            |
| `FPS_STANDALONE` | CMake `-DFPS_STANDALONE` | Includes standalone `main()` via `FPS_MAIN_STANDALONE_V2` |
| `USE_CLI`        | CMake option             | Controls whether CLI targets are built                    |
| `USE_CFITSIO`    | CMake option             | Controls cfitsio-dependent modules                        |
| `USE_COREMODS`   | CMake option             | Controls COREMOD compilation                              |

---

## 12. Coding Style Summary

- **Line length**: ≤ 100 characters
- **Style**: Linux kernel C coding style
- **Documentation**: Document each function’s purpose in
  the corresponding `.h`; keep `.c` comments focused on
  implementation details
- **Includes**: Every `.c` file includes exactly what it
  needs — no implicit transitive includes
- **Variable scope**: Use code blocks `{ }` to minimize
  variable scope
- **Function prototypes**: Multi-line, one argument per
  line
- **Compiler flags**: `-Wall -Wextra` always; `-Werror`
  in CI

---

_This document points to — but does not duplicate —
detailed docs. For the full picture, follow the reading
order in section 2._
