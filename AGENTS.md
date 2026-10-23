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
3. For your first task, try `/add-runnable`.
4. Read the relevant rules in `.agents/rules/` as
   you encounter them.

**Good first contributions** (suited for AI-assisted
work):

- Add Kernel-Doc to undocumented functions
  (use `batch-kernel-doc` skill)
- Add CLI robustness tests (use `cli-test-writer`)
- Fix type consistency gaps (`/check-type-consistency`)
- Update module READMEs

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
| 🔴 **1st** | [`docs/arch/programmers_guide.md`](docs/arch/programmers_guide.md) | Architecture overview, `MILK_RUNNABLE` compute units, directory map, CMake conventions, header hierarchy |
| 🔴 **2nd** | [`docs/arch/dependency_graph.md`](docs/arch/dependency_graph.md)   | Full build-tier diagrams, library link tables, standalone executable cmake patterns                 |
| 🟠 **3rd** | [`docs/guide/streams.md`](docs/guide/streams.md)                     | `IMGID` C API, stream creation/connection, semaphore model, stream modifiers (`@S:`, `@L:`)         |
| 🟠 **4th** | [`docs/guide/fps.md`](docs/guide/fps.md)                             | FPS parameter types, tmux dispatch, `milk-fpsCTRL`, `fpslist.txt` workflow                          |
| 🟡 **5th** | [`docs/developer/code_assist.md`](docs/developer/code_assist.md)             | Index of all agent rules and workflows                                                              |
| 🟡 **6th** | [`docs/guide/procinfo.md`](docs/guide/procinfo.md)                   | `PROCESSINFO` C API, loop profiling                                                                 |

---

## 3. Source Tree Map

```
milk/
├── src/
│   ├── engine/                    ← Core libraries (POSIX only, no CLI)
│   │   ├── ImageStreamIO/         ← Zero-copy shared memory streams
│   │   ├── libfps/                ← FPS core (parameter management)
│   │   ├── libcomputeunit/        ← FPS runtime (standalone main, tmux, conf/run)
│   │   ├── libfpsseq/             ← FPS sequencer library
│   │   ├── libmilkcommon/         ← Common utilities, debug macros
│   │   ├── libprocessinfo/        ← Heartbeat + process tracking
│   │   └── libmilkdata/           ← IMGID struct, image utilities
│   ├── tui/                       ← Engine-tier TUIs (fpsCTRL, procCTRL)
│   ├── cli/                       ← User-facing tools
│   │   ├── libmilkscript/         ← Standalone scripting engine
│   │   ├── CLIcore/               ← Interactive shell framework
│   │   └── overview/              ← milk-CTRL dashboard TUI (milk-streamCTRL: a link that opens it on F3)
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

### 4.1 The `MILK_RUNNABLE` Descriptor

Every compute unit (CLI command and/or fpsexec) is an exported `const MILK_RUNNABLE`
(`src/engine/libcomputeunit/milk_runnable.h`): identity (`.app`), an X-macro parameter table
(`MILK_PARAMS_DEFINE`), a compute function using `MILK_PROCINFO_*(self)`, and optionally
`EXPORT_FPSEXEC(...)` to build an executable. The module lists its runnables in
`MILK_MODULE_RUNNABLES(...)`. Details and template (`examplefunc_fps_cli_poc.c`):
[`docs/arch/programmers_guide.md`](docs/arch/programmers_guide.md) §3.

### 4.2 Compile-Once Sources

Runnable sources include `milk_runnable.h` and are compiled once (`-DMILK_NO_CLI`) by
`milk_add_runnable_module()`; they never include `CLIcore.h` directly.

### 4.3 Standalone Linking

Generated fpsexecs (`milk_add_fpsexec_stubs()`) must **never** link `CLIcore`; they link the module
library and the engine/coremod libraries.

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
   `docs/arch/dependency_graph.md`.** The build is layered —
   engine → core → full. Adding a dependency in the wrong
   direction breaks lower tiers.

2. **Linking fpsexecs to CLIcore.** Use `milk_add_fpsexec_stubs()`.
   Verify with `pytest -q --color=no --tb=short tests/trivial/build_sanity_test.py`
   (from `python/`, `MILK_INSTALLDIR` set).

3. **Implicit header includes.** Every `.c` file must
   include exactly the headers it uses. Don't rely on
   `CLIcore.h` pulling in `math.h` or `stdlib.h`.

4. **Forgetting to update `docs/arch/programmers_guide.md`**
   after architectural changes.

5. **Lines > 100 characters.** The project enforces
   short lines for readability.

6. **Using placeholders in module READMEs.** Always write
   accurate descriptions derived from source code.

7. **Not compiling after edits.** Always run the
   `/compile-test` workflow after modifying C or CMake
   files.

---

## 6. Agent Rules, Skills and Workflows

Everything lives in `.agents/`; list the directories for the current index and open a file for its
trigger conditions and required actions (human-readable overview:
[`docs/developer/code_assist.md`](docs/developer/code_assist.md)).

- **`.agents/rules/`** — always-on guardrails (code style, naming, error handling, performance, shared
  memory safety, CMake, documentation, `runnable-conventions`...). Rules are loaded automatically.
- **`.agents/skills/`** — specialized instruction sets with helper content, consulted when a task needs
  domain depth (API quick reference, ImageStreamIO internals, stream modifiers, CMake patterns,
  build-failure triage, CLI debugging, optimization, TUI creation, PR preparation...).
- **`.agents/workflows/`** — on-demand task templates invoked as slash commands, one per file
  (`/add-runnable`, `/add-module`, `/compile-test`, `/fix-bug`, `/review-pr`, `/setup-dev-environment`,
  `/update-docs-site`...).

## 9. Build System Quick Reference

### Build tiers

| Tier           | CMake Flags                        | What's built                                                  |
| -------------- | ---------------------------------- | ------------------------------------------------------------- |
| Engine         | `-DUSE_COREMODS=OFF -DUSE_CLI=OFF` | ImageStreamIO, milkcommon, milkfps, milkdata, milkprocessinfo |
| Core           | `-DUSE_CLI=OFF`                    | Engine + COREMODs (arith, memory, tools, iofits)              |
| Core (no FITS) | `-DUSE_CLI=OFF -DUSE_CFITSIO=OFF`  | Core without COREMOD_iofits                                   |
| Full           | _(defaults)_                       | Everything: CLI + all plugins                                 |

### Compile-test cycle

```bash
$ cd _build
$ make -j$(nproc)
$ make install
$ ctest --output-on-failure
```

### Module CMakeLists.txt

Follow `src/milk_module_example/CMakeLists.txt`; skeleton and
helpers in `docs/arch/programmers_guide.md` §6.

---

## 10. cacao (Adaptive Optics Plugin)

`cacao` source lives at `plugins/cacao-src/`, which is a
symlink to `~/src/cacao`. When editing cacao modules,
treat this symlink directory as part of the workspace.

cacao modules are `MILK_RUNNABLE` modules like milk modules; their fpsexecs are
declared with `EXPORT_FPSEXEC(cacao, "id", unit)`.

---

## 11. Compile-Time Guards

| Macro            | Set By                   | Effect                                                    |
| ---------------- | ------------------------ | --------------------------------------------------------- |
| `MILK_NO_CLI`    | CMake `-DMILK_NO_CLI`    | Excludes CLI code, uses `CLIcore_standalone.h`            |
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
