# Build Tiers

<!-- prettier-ignore -->
!!! note
    The milk build system supports multiple build tiers, controlled by CMake options. Lower tiers
    have fewer external dependencies and produce a smaller footprint.

---

## Overview

The default build is the **Full build**, but without extra dependencies.

| Build layer     | Code pulled in         | Dependencies                                                                                       |
| --------------- | ---------------------- | -------------------------------------------------------------------------------------------------- |
| **Full build**  | CLI (`milk-cli`)       | ncurses, readline                                                                                  |
|                 | plugins                | Per-need (see [CMake system](../developer/dependency_system.md)) - CUDA, MAGMA, OpenBLAS, MKL, ... |
| **Core + FITS** | COREMOD_iofits         | cfitsio                                                                                            |
| **Core**        | COREMOD_memory         |                                                                                                    |
|                 | COREMOD_arith          |                                                                                                    |
|                 | COREMOD_tools          |                                                                                                    |
| **Engine**      | milk engine libs       | All POSIX-only                                                                                     |
|                 | ImageStreamIO          |                                                                                                    |
|                 | libprocessinfo, libfps |                                                                                                    |

## Optional dependencies

Some optional dependencies are expected by MILK and enable additional features

| Option         | Default | Description                           |
| -------------- | ------- | ------------------------------------- |
| `USE_READLINE` | ON      | Enable readline for CLI input         |
| `USE_GSL`      | ON      | Enable GSL for plugins                |
| `USE_CUDA`     | OFF     | CUDA GPU computations                 |
| `USE_MAGMA`    | OFF     | MAGMA linear algebra on GPU routines  |
| `USE_OPENBLAS` | OFF     | BLAS: multicore linear algebra on CPU |
| `USE_MKL`      | OFF     | Intel MKL compute library             |
| `USE_LAPACKE`  | OFF     | Lapacke linear algebra library        |

If you have the dependency `<X>` installed, you may request MILK to fetch it with `-DUSE_<X>=ON`.

CMake will issue warnings when some features are disabled due to a missing dependency -- either because `-DUSE_<X>=OFF` was set, or because the library wasn't found by CMake.

<!-- prettier-ignore -->
!!! note
    **There is crosstalk between those dependencies:**
    - MKL always provides BLAS and Lapacke
    - OpenBLAS always provides BLAS, sometimes LAPACKE
    - Lapacke can be installed standalone, and MKL and OpenBLAS deactivated (in which case BLAS
      features are missing, but Lapacke features are enabled).

See [Dependency System](../developer/dependency_system.md) on how to integrate these optional
dependencies as a developer.

## How to build each tier of MILK

| Tier               | cmake command                               | What you get                                      |
| ------------------ | ------------------------------------------- | ------------------------------------------------- |
| **Engine**         | `cmake .. -DUSE_COREMODS=OFF -DUSE_CLI=OFF` | ImageStreamIO, processinfo, FPS, standalone tools |
| **Core**           | `cmake .. -DUSE_CLI=OFF -DUSE_CFITSIO=OFF`  | + COREMOD_arith, memory, tools (no cfitsio)       |
| **Core + FITS**    | `cmake .. -DUSE_CLI=OFF`                    | + COREMOD_iofits (requires cfitsio)               |
| **Full**           | `cmake ..`                                  | + CLI, all plugins, example module                |
| **Full + options** | `cmake .. -DUSE_<X_OPTIONAL_DEP>=ON`        | + GPU acceleration                                |

All tiers are built with:

```bash
$ mkdir _build && cd _build
$ cmake .. [OPTIONS]        # see table above
$ make -j$(nproc)
$ sudo make install
```

<!-- prettier-ignore-start -->
!!! info
    __Quirks on the build variables:__
    - `USE_CLI=ON` automatically enables `USE_COREMODS=ON`
    - Plugins are only built when `USE_COREMODS=ON`
    - `USE_CFITSIO=OFF` excludes `COREMOD_iofits` and compiles remaining modules without cfitsio linkage.

    ## Behavior When `cfitsio` Is Disabled

!!! info
    __when FITS capability is removed from -DUSE_CFITSIO=OFF__
    FITS-dependent code paths are compiled out via `#ifdef USE_CFITSIO` guards:

    | Module           | Effect                                                                               |
    | ---------------- | ------------------------------------------------------------------------------------ |
    | `COREMOD_iofits` | Not built — `iofits.loadfits` / `iofits.saveFITS` unavailable                        |
    | `COREMOD_memory` | `logshmim`, `saveall`, `stream_pixmapdecode` print a warning instead of writing FITS |
    | `COREMOD_arith`  | No impact (never used cfitsio)                                                       |
    | `COREMOD_tools`  | No impact                                                                            |

!!! tip
    The engine and core tiers are ideal for embedded or real-time deployments where only
    shared-memory stream processing is needed and disk I/O to FITS files is unnecessary.
<!-- prettier-ignore-end -->

---

← [Documentation Index](../index.md)
,
