# Build Tiers

!!! note
The milk build system supports multiple build tiers,
controlled by CMake options. Lower tiers have fewer external
dependencies and produce a smaller footprint.

---

## Overview

```text
┌─────────────────────────────────────────────────────┐
│  Full Build (default)                               │
│  CLI (milk-cli) + plugins + example module          │
│  Optional: readline, GSL, CUDA                      │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │  Core + FITS                                  │  │
│  │  COREMOD_iofits                               │  │
│  │  Requires: cfitsio                            │  │
│  │                                               │  │
│  │  ┌─────────────────────────────────────────┐  │  │
│  │  │  Core                                   │  │  │
│  │  │  COREMOD_arith, COREMOD_memory,         │  │  │
│  │  │  COREMOD_tools                          │  │  │
│  │  │                                         │  │  │
│  │  │  ┌───────────────────────────────────┐  │  │  │
│  │  │  │  Engine                           │  │  │  │
│  │  │  │  ImageStreamIO, libprocessinfo,   │  │  │  │
│  │  │  │  libfps, libmilkdata              │  │  │  │
│  │  │  │  POSIX only: pthread, rt, m, dl   │  │  │  │
│  │  │  └───────────────────────────────────┘  │  │  │
│  │  └─────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

---

## CMake Options

| Option         | Default | Description                        |
| -------------- | ------- | ---------------------------------- |
| `USE_COREMODS` | ON      | Build core modules (COREMOD\_\*)   |
| `USE_CFITSIO`  | ON      | Build FITS I/O (requires cfitsio)  |
| `USE_CLI`      | ON      | Build interactive CLI (`milk-cli`) |
| `USE_READLINE` | ON      | Enable readline for CLI input      |
| `USE_GSL`      | ON      | Enable GSL for plugins             |

!!! important
**Dependency chain:**

    - `USE_CLI=ON` automatically enables `USE_COREMODS`
    - Plugins are only built when `USE_COREMODS=ON`
    - `USE_CFITSIO=OFF` excludes `COREMOD_iofits` and compiles
      remaining modules without cfitsio linkage

---

## Quick Reference

| Tier            | cmake command                               | What you get                                      |
| --------------- | ------------------------------------------- | ------------------------------------------------- |
| **Engine**      | `cmake .. -DUSE_COREMODS=OFF -DUSE_CLI=OFF` | ImageStreamIO, processinfo, FPS, standalone tools |
| **Core**        | `cmake .. -DUSE_CLI=OFF -DUSE_CFITSIO=OFF`  | + COREMOD_arith, memory, tools (no cfitsio)       |
| **Core + FITS** | `cmake .. -DUSE_CLI=OFF`                    | + COREMOD_iofits (requires cfitsio)               |
| **Full**        | `cmake ..`                                  | + CLI, all plugins, example module                |
| **Full + CUDA** | `cmake .. -DUSE_CUDA=ON`                    | + GPU acceleration                                |

All tiers are built with:

```bash
$ mkdir _build && cd _build
$ cmake .. [OPTIONS]        # see table above
$ make -j$(nproc)
$ sudo make install
```

---

## Behavior When cfitsio Is Disabled

When built with `USE_CFITSIO=OFF`, FITS-dependent code paths
are compiled out via `#ifdef USE_CFITSIO` guards:

| Module           | Effect                                                                               |
| ---------------- | ------------------------------------------------------------------------------------ |
| `COREMOD_iofits` | Not built — `iofits.loadfits` / `iofits.saveFITS` unavailable                        |
| `COREMOD_memory` | `logshmim`, `saveall`, `stream_pixmapdecode` print a warning instead of writing FITS |
| `COREMOD_arith`  | No impact (never used cfitsio)                                                       |
| `COREMOD_tools`  | No impact                                                                            |

!!! tip
The engine and core tiers are ideal for embedded or
real-time deployments where only shared-memory stream
processing is needed and disk I/O to FITS files is
unnecessary.

---

← [Documentation Index](../index.md)
