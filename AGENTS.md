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

## 1. What Is milk?

`milk` is a **real-time image-processing framework**
designed for Adaptive Optics (AO) and high-performance
scientific computing. Its defining characteristic is a
**shared-memory micro-service architecture**: many small
standalone executables (compute units) communicate via
zero-copy tensors in `/dev/shm/` and are configured
through a shared-memory parameter system.

**Three pillars:**

| Pillar | Purpose | Shared Memory Path |
|--------|---------|--------------------|
| **ImageStreamIO** (Streams) | Zero-copy n-dimensional data passing between processes | `/dev/shm/*.im.shm` |
| **FPS** (Function Processing System) | Real-time parameter sync, state control (run/stop/conf) | `/dev/shm/fps.*` |
| **processinfo** | Heartbeat telemetry, loop rate profiling, health monitoring | `/dev/shm/proc.*` |

Compute units run inside **tmux sessions** for fault
isolation — a crash in one unit never takes down others.

---

## 2. Read These Next (in order)

| Priority | Document | What you learn |
|----------|----------|----------------|
| 🔴 **1st** | [`docs/programmers_guide.md`](docs/programmers_guide.md) | Architecture overview, V2 compute unit template, directory map, CMake conventions, header hierarchy |
| 🔴 **2nd** | [`docs/dependency_graph.md`](docs/dependency_graph.md) | Full build-tier diagrams, library link tables, `_compute` variant patterns |
| 🟠 **3rd** | [`docs/streams.md`](docs/streams.md) | `IMGID` C API, stream creation/connection, semaphore model, stream modifiers (`@S:`, `@L:`, `@F:`) |
| 🟠 **4th** | [`docs/fps.md`](docs/fps.md) | FPS parameter types, tmux dispatch, `milk-fpsCTRL`, `fpslist.txt` workflow |
| 🟡 **5th** | [`docs/code_assist.md`](docs/code_assist.md) | Index of all agent rules and workflows |
| 🟡 **6th** | [`docs/procinfo.md`](docs/procinfo.md) | `PROCESSINFO` C API, loop profiling |

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

| Section | Content |
|---------|---------|
| 1 | `FPS_APP_INFO` — name, CLI keyword, description |
| 2 | Local C variables for parameters |
| 3 | `FPS_PARAMS` X-macro — binds C vars to FPS shared memory |
| 4 | `fpsexec()` — pure computation function |
| 5 | `CLIcmddata` — CLI registry scoping |
| 6 | Compute wrapper — `INSERT_STD_PROCINFO_COMPUTEFUNC_*` |
| 7 | Module registration — `CLIADDCMD_*` (guarded by `#ifndef FPS_STANDALONE`) |
| 8 | `FPS_MAIN_STANDALONE_V2` — standalone `main()` |

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
img.naxis = 2;
img.size[0] = 128;  img.size[1] = 128;
img.shared = 1;
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

5. **Lines > 80 characters.** The project enforces short
   lines for readability.

6. **Using placeholders in module READMEs.** Always write
   accurate descriptions derived from source code.

7. **Not compiling after edits.** Always run the
   `/compile-test` workflow after modifying C or CMake
   files.

---

## 6. Agent Rules (Always Active)

These rules in `.agents/rules/` are automatically loaded
and enforced. Know what they require:

| Rule | Key Requirement |
|------|-----------------|
| `architecture-principles.md` | Check dependency graph before adding deps |
| `cmake-conventions.md` | Use `PUBLIC`/`INTERFACE` properties; modules own their headers |
| `code-style-guide.md` | 80-char lines, Kernel-Doc, Linux kernel style, explicit includes |
| `compile-after-edit.md` | Always compile-test after C/CMake changes |
| `documentation-standards.md` | Consistent markdown, shell prompts, link checking |
| `files-directories.md` | cacao lives at `plugins/cacao-src` → `~/src/cacao` |
| `fpsexec-conventions.md` | V2 template, 8-section layout, `-h1` support |
| `git-workflow.md` | All changes via PRs from feature branches into `framework-dev`. NO pushes to `dev`. |
| `help-consistency.md` | Cross-check all sibling help sources |
| `maintain-programmers-guide.md` | Update programmer's guide on arch changes |
| `performance-practices.md` | SIMD, BLAS, pointer alignment, type dispatch, CPU pinning |
| `readme-update.md` | Update module README when files change |
| `run-milk-commands.md` | Environment setup, SHM cleanup, tmux guidance |
| `script-docs.md` | Update `docs/scripts.md` when scripts change |

---

## 7. Workflows (On-Demand)

Invoke these as slash commands when working on the
listed task types:

| Command | When to use |
|---------|-------------|
| `/compile-test` | After any C or CMake edit |
| `/create-fpsexec` | Scaffolding a new standalone executable |
| `/add-new-module` | Creating a new plugin module from scratch |
| `/add-function` | Adding a function to an existing module |
| `/add-stream-processor` | Creating a stream processing loop |
| `/add-cli-command` | Adding a CLI command to a module |
| `/update-programmers-guide` | After architectural changes |
| `/audit-help-consistency` | After editing help text anywhere |
| `/check-type-consistency` | Auditing data type handling in switch blocks |
| `/update-scripts-docs` | After adding or modifying shell scripts |

---

## 8. Build System Quick Reference

### Build tiers

| Tier | CMake Flags | What's built |
|------|-------------|--------------|
| Engine | `-DUSE_COREMODS=OFF -DUSE_CLI=OFF` | ImageStreamIO, milkfps, milkdata, milkprocessinfo |
| Core | `-DUSE_CLI=OFF -DUSE_CFITSIO=OFF` | Engine + COREMODs (arith, memory, tools) |
| Core+FITS | `-DUSE_CLI=OFF` | Core + COREMOD_iofits |
| Full | *(defaults)* | Everything: CLI + all plugins |

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

## 9. cacao (Adaptive Optics Plugin)

`cacao` source lives at `plugins/cacao-src/`, which is a
symlink to `~/src/cacao`. When editing cacao modules,
treat this symlink directory as part of the workspace.

cacao modules follow the same V2 template conventions as
milk modules, but use `add_cacao_standalone()` or
`add_cacao_standalone_plugins()` for their executables.

---

## 10. Compile-Time Guards

| Macro | Set By | Effect |
|-------|--------|--------|
| `MILK_NO_CLI` | CMake `-DMILK_NO_CLI` | Excludes CLI code, uses `CLIcore_standalone.h` |
| `FPS_STANDALONE` | CMake `-DFPS_STANDALONE` | Includes standalone `main()` via `FPS_MAIN_STANDALONE_V2` |
| `USE_CLI` | CMake option | Controls whether CLI targets are built |
| `USE_CFITSIO` | CMake option | Controls cfitsio-dependent modules |
| `USE_COREMODS` | CMake option | Controls COREMOD compilation |

---

## 11. Coding Style Summary

- **Line length**: ≤ 80 characters
- **Style**: Linux kernel C coding style
- **Documentation**: Kernel-Doc (`/** ... */`) above
  functions in `.c` files; brief descriptions in `.h`
- **Includes**: Every `.c` file includes exactly what it
  needs — no implicit transitive includes
- **Variable scope**: Use code blocks `{ }` to minimize
  variable scope
- **Function prototypes**: Multi-line, one argument per
  line
- **Compiler flags**: `-Wall -Wextra` always; `-Werror`
  in CI

---

*This document points to — but does not duplicate —
detailed docs. For the full picture, follow the reading
order in section 2.*
